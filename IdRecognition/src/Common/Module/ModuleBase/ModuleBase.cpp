/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ModuleBase.h"
#include "Log/Log.h"

namespace ascendFaceRecognition {
ModuleBase::ModuleBase()
{
}
ModuleBase::~ModuleBase()
{
    if (!isStop_) {
        StopAndDestroyQueue();
    }
}
void ModuleBase::AssignInitArgs(const ModuleInitArgs &initArgs)
{
#ifdef ASCEND_FACE_USE_ACL
    aclContext_ = initArgs.context;
    runMode_ = initArgs.runMode;
#endif
    moduleName_ = initArgs.moduleName;
    instanceId_ = initArgs.instanceId;
    inputQueue_ = initArgs.inputQueue;
    outputQueVec_ = initArgs.outputQueVec;
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
    LogDebug << "queue information: outputQueVec_[" << instanceId_ << "]=" << outputQueVec_[instanceId_];
    // repeatly pop data from input queue and call the Process funtion. Results will be pushed to output queues.
    while (!isStop_) {
        LogDebug << "Begin to pop data from input queue for " << moduleName_ << "[" << instanceId_ << "].";
        std::shared_ptr<void> frameAiInfo = nullptr;
        ret = inputQueue_->Pop(frameAiInfo);
        if (ret == APP_ERR_QUEUE_STOPED) {
            LogDebug << "Stop input queue for " << moduleName_ << "[" << instanceId_ << "]"
                     << ".";
            break;
        } else if (ret != APP_ERR_OK || frameAiInfo == nullptr) {
            LogError << "Fail to get data from input queue for " << moduleName_ << "[" << instanceId_ << "]"
                     << ", ret=" << ret << "(" << GetAppErrCodeInfo(ret) << ").";
            continue;
        }

        ret = Process(frameAiInfo);
        if (ret != APP_ERR_OK) {
            LogError << "Fail to process data for " << moduleName_ << "[" << instanceId_ << "]"
                     << ", ret=" << ret << "(" << GetAppErrCodeInfo(ret) << ").";
        }
    }
}

// clear input queue and stop the thread of the instance, called before destroy the instance
void ModuleBase::StopAndDestroyQueue()
{
    isStop_ = true;
    inputQueue_->Stop();

    if (processThr_.joinable()) {
        processThr_.join();
    }
}
} // namespace ascendFaceRecognition