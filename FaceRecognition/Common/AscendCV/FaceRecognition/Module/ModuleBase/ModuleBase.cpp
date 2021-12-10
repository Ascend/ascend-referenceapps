/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1 Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2 Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3 Neither the names of the copyright holders nor the names of the
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * Description: base class of application pipeline modules.
 * Date: 2020/04/16
 * History:
 * [2020-04-16]:
 */
#include "ModuleBase.h"

#include <unistd.h>

#include "Log/Log.h"

namespace ascendFaceRecognition {
const double TIME_COUNTS = 1000.0;
void ModuleBase::AssignInitArgs(ModuleInitArgs &initArgs)
{
#ifdef ASCEND_FACE_USE_ACL
    aclContext_ = initArgs.context;
    runMode_ = initArgs.runMode;
#endif
    pipelineName_ = initArgs.pipelineName;
    moduleName_ = initArgs.moduleName;
    instanceId_ = initArgs.instanceId;
    inputQueue_ = initArgs.inputQueue;
    pipeMode_ = initArgs.pipelineMode;
}

// run module instance in a new thread created
APP_ERROR ModuleBase::Run()
{
    LogDebug << moduleName_ << "[" << instanceId_ << "]"
             << ":: Run.";

    processThr_ = std::thread(&ModuleBase::ProcessThread, this);

    return APP_ERR_OK;
}

// get the data from input queue then call Process function in the new thread
void ModuleBase::ProcessThread()
{
    APP_ERROR ret;
#ifdef ASCEND_FACE_USE_ACL
    ret = aclrtSetCurrentContext(aclContext_);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to set context for " << moduleName_ << "[" << instanceId_ << "]"
                 << ", ret=" << ret << "(" << GetAppErrCodeInfo(ret) << ").";
        return;
    }
#endif
    // if the module has no input queue, call Process function directly.
    if (withoutInputQueue_ == true) {
        ret = Process(nullptr);
        if (ret != APP_ERR_OK) {
            LogError << "Fail to process data for " << moduleName_ << "[" << instanceId_ << "]"
                     << ", ret=" << ret << "(" << GetAppErrCodeInfo(ret) << ").";
        }
        return;
    }
    if (inputQueue_ == nullptr) {
        LogFatal << "Invalid input queue of " << moduleName_ << "[" << instanceId_ << "].";
        return;
    }
    LogDebug << "Input queue for " << moduleName_ << "[" << instanceId_ << "], inputQueue=" << inputQueue_;
    // repeatly pop data from input queue and call the Process funtion. Results will be pushed to output queues.
    while (!isStop_) {
        LogDebug << "Begin to pop data from input queue for " << moduleName_ << "[" << instanceId_ << "].";
        std::shared_ptr<void> frameAiInfo = nullptr;
        ret = inputQueue_->Pop(frameAiInfo);
        if (ret == APP_ERR_QUEUE_STOPED) {
            LogInfo << "Stop input queue for " << moduleName_ << "[" << instanceId_ << "].";
            break;
        } else if (ret != APP_ERR_OK || frameAiInfo == nullptr) {
            LogError << "Fail to get data from input queue for " << moduleName_ << "[" << instanceId_ << "]"
                     << ", ret=" << ret << "(" << GetAppErrCodeInfo(ret) << ").";
            continue;
        }

        CallProcess(frameAiInfo);
    }
}

void ModuleBase::CallProcess(std::shared_ptr<void> &frameAiInfo)
{
    struct timeval startTime = { 0, 0 };
    struct timeval endTime = { 0, 0 };
    gettimeofday(&startTime, nullptr);
    APP_ERROR ret = Process(frameAiInfo);
    gettimeofday(&endTime, nullptr);
    double costMs =
        (endTime.tv_sec - startTime.tv_sec) * TIME_COUNTS + (endTime.tv_usec - startTime.tv_usec) / TIME_COUNTS;
    int queueSize = inputQueue_->GetSize();
#ifdef ASCEND_ACL_OPEN_VESION
    const int queueLogSize = 500;
#else
    const int queueLogSize = 50;
#endif
    if (queueSize > queueLogSize) {
        LogWarn << "[Statistic] [Module] [" << moduleName_ << "] [" << instanceId_ << "] [QueueSize] [" << queueSize <<
            "] [Process] [" << costMs << " ms]";
    }

    if (ret != APP_ERR_OK) {
        LogError << "Fail to process data for " << moduleName_ << "[" << instanceId_ << "]"
                 << ", ret=" << ret << "(" << GetAppErrCodeInfo(ret) << ").";
    }
}

void ModuleBase::SetOutputInfo(ModuleType moduleType, ModuleConnectType connectType,
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec)
{
    if (outputQueVec.size() == 0) {
        LogFatal << "outputQueVec is Empty! " << moduleType;
        return;
    }

    ModuleOutputInfo outputInfo;
    outputInfo.moduleType = moduleType;
    outputInfo.connectType = connectType;
    outputInfo.outputQueVec = outputQueVec;
    outputInfo.outputQueVecSize = outputQueVec.size();
    outputQueMap_[moduleType] = outputInfo;
}

void ModuleBase::SetInputVec(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue)
{
    inputQueue_ = inputQueue;
}

void ModuleBase::SendToNextModule(ModuleType moduleType, std::shared_ptr<void> outputData, int channelId)
{
    if (outputQueMap_.find(moduleType) == outputQueMap_.end()) {
        LogFatal << "No Next Module Type " << moduleType;
        return;
    }

    ModuleOutputInfo outputInfo = outputQueMap_[moduleType];

    if (outputInfo.connectType == MODULE_CONNECT_ONE) {
        outputInfo.outputQueVec[0]->Push(outputData, true);
    }
    if (outputInfo.connectType == MODULE_CONNECT_CHANNEL) {
        uint32_t ch = channelId % outputInfo.outputQueVecSize;
        if (ch >= outputInfo.outputQueVecSize) {
            LogFatal << "No Next Module!";
            return;
        }
        outputInfo.outputQueVec[ch]->Push(outputData, true);
    } else if (outputInfo.connectType == MODULE_CONNECT_PAIR) {
        outputInfo.outputQueVec[instanceId_]->Push(outputData, true);
    } else if (outputInfo.connectType == MODULE_CONNECT_RANDOM) {
        outputInfo.outputQueVec[sendCount_ % outputInfo.outputQueVecSize]->Push(outputData, true);
    }
    sendCount_++;
}

// clear input queue and stop the thread of the instance, called before destroy the instance
APP_ERROR ModuleBase::Stop()
{
#ifdef ASCEND_MODULE_USE_ACL
    APP_ERROR ret = aclrtSetCurrentContext(aclContext_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ModuleManager: fail to set context, ret[%d]" << ret << ".";
        return ret;
    }
#endif

    // stop input queue
    isStop_ = true;
    if (inputQueue_ != nullptr) {
        inputQueue_->Stop();
        inputQueue_->Clear();
    }
    LogInfo << moduleName_ << "[" << instanceId_ << "] queue stop success";
    if (processThr_.joinable()) {
        processThr_.join();
    }
    LogInfo << moduleName_ << "[" << instanceId_ << "] thread join success";
    APP_ERROR ret = DeInit();
    LogInfo << moduleName_ << "[" << instanceId_ << "] process end";
    return ret;
}
} // namespace ascendFaceRecognition
