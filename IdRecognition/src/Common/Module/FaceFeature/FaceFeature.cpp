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
#include "FaceFeature.h"

#include <iostream>
#include <algorithm>

#include "ConfigParser/ConfigParser.h"
#include "Log/Log.h"
#include "FileManager/FileManager.h"

namespace ascendFaceRecognition {
const int DYNAMIC_BATCH_MODEL_INPUTNUM = 2; // first is data, secondary is dynamic bacth info

FaceFeature::~FaceFeature()
{
    if (!isDeInited_) {
        DeInit();
    }
}

APP_ERROR FaceFeature::ParseConfig(ConfigParser &configParser)
{
    LogInfo << "FaceFeature[" << instanceId_ << "]: start to parse config.";
    std::string itemCfgStr(moduleName_ + std::to_string(instanceId_) + std::string(".width"));
    APP_ERROR ret = configParser.GetIntValue(itemCfgStr, width_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    alignedWidth_ = ALIGN_UP(width_, FACE_FEATURE_WIDTH_ALIGN);

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".height");
    ret = configParser.GetIntValue(itemCfgStr, height_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    alignedHeight_ = ALIGN_UP(height_, FACE_FEATURE_HEIGHT_ALIGN);

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".input_channel");
    ret = configParser.GetIntValue(itemCfgStr, inputChn_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".model_path");
    ret = configParser.GetStringValue(itemCfgStr, modelPath_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + ".dynamicBatchSize";
    ret = configParser.GetVectorUint32Value(itemCfgStr, dynamicBatchSize_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get dynamic batch size named " << itemCfgStr << ".";
        return ret;
    }
    dataSzie_ = alignedHeight_ * alignedWidth_ * inputChn_;
    // Sort by descending order
    sort(dynamicBatchSize_.begin(), dynamicBatchSize_.end(), std::greater<uint32_t>());
    for (size_t i = 0; i < dynamicBatchSize_.size(); ++i) {
        LogDebug << "dynamicBatchSize_[" << i << "]=" << dynamicBatchSize_[i];
    }
    return ret;
}

APP_ERROR FaceFeature::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "FaceFeature[" << initArgs.instanceId << "]: start to init.";
    AssignInitArgs(initArgs);
    modelProcess_ = std::make_shared<ModelProcess>();
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to parse config params, ret=" << ret << "("
                 << GetAppErrCodeInfo(ret) << ").";
        return ret;
    }
    isStop_ = false;
    ret = modelProcess_->Init(modelPath_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to init model.";
        return ret;
    }

    ret = Resnet18DynamicBatchSizeInputBufferMalloc();
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: malloc input buffer fail.";
        return ret;
    }
    ret = Resnet18DynamicBatchSizeOutputBufferMalloc();
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: malloc ouput buffer fail.";
        return ret;
    }
    return ret;
}

APP_ERROR FaceFeature::Resnet18DynamicBatchSizeInputBufferMalloc()
{
    aclmdlDesc *modelDesc = modelProcess_->GetModelDesc();
    size_t inputNum = aclmdlGetNumInputs(modelDesc);
    if (inputNum != DYNAMIC_BATCH_MODEL_INPUTNUM) {
        LogError << "Dynamic inputNum of restnet18 is 2 but not " << inputNum;
        return APP_ERR_COMM_CONNECTION_FAILURE;
    } else {
        LogDebug << "FaceFeature[" << instanceId_ << "]: inputNum =" << inputNum << ".";
    }
    for (size_t i = 0; i < inputNum; ++i) {
        void *buffer = nullptr;
        // modify size
        size_t size = aclmdlGetInputSizeByIndex(modelDesc, i);
        APP_ERROR ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceFeature[" << instanceId_ << "]: resnet18 aclrtMalloc fail(ret=" << ret
                     << "), buffer=" << buffer << ", size=" << size << ".";
            if (buffer != nullptr) {
                aclrtFree(buffer);
                buffer = nullptr;
            }
            return ret;
        }
        inputBuffers_.push_back(buffer);
        inputSizes_.push_back(size);
        LogDebug << "FaceFeature[" << instanceId_ << "]: i=" << i << ", size=" << size << ".";
    }
    if (inputSizes_[0] != dataSzie_ * dynamicBatchSize_[0]) {
        LogError << "inputSizes_ error, inputSizes_[0]=" << inputSizes_[0] << ", dataSzie_=" << dataSzie_
                 << ", dynamicBatchSize_[0]=" << dynamicBatchSize_[0];
        return APP_ERR_COMM_CONNECTION_FAILURE;
    }
    return APP_ERR_OK;
}


APP_ERROR FaceFeature::Resnet18DynamicBatchSizeOutputBufferMalloc()
{
    aclmdlDesc *modelDesc = modelProcess_->GetModelDesc();
    size_t outputNum = aclmdlGetNumOutputs(modelDesc);
    LogDebug << "FaceFeature[" << instanceId_ << "]: outputNum = " << outputNum << ".";
    for (size_t i = 0; i < outputNum; ++i) {
        void *buffer = nullptr;
        size_t size = aclmdlGetOutputSizeByIndex(modelDesc, i);
        APP_ERROR ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceFeature[" << instanceId_ << "]: output aclrtMalloc fail(ret=" << ret
                     << "), buffer=" << buffer << ", size=" << (int)size << ".";
            if (buffer != nullptr) {
                aclrtFree(buffer);
                buffer = nullptr;
            }
            return ret;
        }
        outputBuffers_.push_back(buffer);
        outputSizes_.push_back(size);
        LogDebug << "FaceFeature[" << instanceId_ << "]: i = " << i << ", size=" << size << ".";
    }
    if (outputSizes_[0] % dynamicBatchSize_[0] != 0) {
        LogError << "outputSizes_ error, outputSizes_[0]=" << outputSizes_[0]
                 << ", dynamicBatchSize_[0]=" << dynamicBatchSize_[0];
        return APP_ERR_COMM_CONNECTION_FAILURE;
    }
    featrueSize_ = outputSizes_[0] / dynamicBatchSize_[0];
    return APP_ERR_OK;
}

APP_ERROR FaceFeature::DeInit()
{
    LogInfo << "FaceFeature[" << instanceId_ << "]: deinit start.";

    isDeInited_ = true;
    StopAndDestroyQueue();

    modelProcess_->DeInit();

    for (size_t i = 0; i < inputBuffers_.size(); i++) {
        if (inputBuffers_[i] != nullptr) {
            aclrtFree(inputBuffers_[i]);
            inputBuffers_[i] = nullptr;
        }
    }

    for (size_t i = 0; i < outputBuffers_.size(); i++) {
        if (outputBuffers_[i] != nullptr) {
            aclrtFree(outputBuffers_[i]);
            outputBuffers_[i] = nullptr;
        }
    }

    return APP_ERR_OK;
}

APP_ERROR FaceFeature::PreParaData(std::vector<FaceObject *> &faceObjectQueue) const
{
    LogDebug << "FaceFeature[" << instanceId_ << "]: prepare data.";
    APP_ERROR ret;
    uint8_t *dataBufferPtr = static_cast<uint8_t *>(inputBuffers_[0]);
    for (size_t i = 0; i < faceObjectQueue.size(); ++i) {
        FaceObject *faceObjectPtr = faceObjectQueue[i];
        if (dataSzie_ != faceObjectPtr->imgAffine.buf.dataSize) {
            LogError << "check input data fail, dataSize=" << dataSzie_ << ", alignedWidth_=" << alignedWidth_
                     << ", alignedHeight_=" << alignedHeight_ << ", inputChn_=" << inputChn_;
            LogError << "input dataSize=" << faceObjectPtr->imgAffine.buf.dataSize
                     << ", widthAligned=" << faceObjectPtr->imgAffine.widthAligned
                     << ", heightAligned=" << faceObjectPtr->imgAffine.heightAligned
                     << ", format=" << faceObjectPtr->imgAffine.format;
            return APP_ERR_ACL_FAILURE;
        }
        if (runMode_ == ACL_HOST) { // under ACL_HOST mode, copy memory from host to device
            ret = aclrtMemcpy(dataBufferPtr, dataSzie_, faceObjectPtr->imgAffine.buf.hostData.get(),
                faceObjectPtr->imgAffine.buf.dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
        } else if (runMode_ == ACL_DEVICE) { // under ACL_DEVICE mode, copy memory from device to device
            ret = aclrtMemcpy(dataBufferPtr, dataSzie_, faceObjectPtr->imgAffine.buf.hostData.get(),
                faceObjectPtr->imgAffine.buf.dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
        }
        if (ret != APP_ERR_OK) {
            LogError << "FaceFeature[" << instanceId_ << "]::PreParaData: memcopy failed, dataSzie_=" << dataSzie_
                     << ", src_size=" << faceObjectPtr->imgAffine.buf.dataSize;
            return ret;
        }
        dataBufferPtr += dataSzie_;
    }
    return APP_ERR_OK;
}

APP_ERROR FaceFeature::PostData(std::vector<FaceObject *> &faceObjectQueue) const
{
    LogDebug << "FaceFeature[" << instanceId_ << "]: post process.";
    APP_ERROR ret;
    uint8_t *resPtr = static_cast<uint8_t *>(outputBuffers_[0]);
    for (size_t i = 0; i < faceObjectQueue.size(); i++) {
        auto outBuffer = std::make_shared<uint8_t>();
        outBuffer.reset(new uint8_t[featrueSize_], std::default_delete<uint8_t[]>());
        if (runMode_ == ACL_HOST) { // under ACL_HOST mode, copy memory from device to host
            ret = aclrtMemcpy(outBuffer.get(), featrueSize_, resPtr, featrueSize_, ACL_MEMCPY_DEVICE_TO_HOST);
        } else if (runMode_ == ACL_DEVICE) { // under ACL_DEVICE mode, copy memory from device to device
            ret = aclrtMemcpy(outBuffer.get(), featrueSize_, resPtr, featrueSize_, ACL_MEMCPY_DEVICE_TO_DEVICE);
        }
        if (ret != APP_ERR_OK) {
            LogError << "FaceFeature[" << instanceId_ << "]::PreParaData: memcopy failed.";
            return ret;
        }

        /* featureVector.hostData host address as output */
        faceObjectQueue[i]->featureVector.hostData = outBuffer;
        faceObjectQueue[i]->featureVector.dataSize = featrueSize_;
        resPtr += featrueSize_;
    }
    return APP_ERR_OK;
}

APP_ERROR FaceFeature::Process(std::shared_ptr<void> inputData)
{
    faceFeatureStatic_.RunTimeStatisticStart("FaceFeature_excute_time", instanceId_);
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::static_pointer_cast<FrameAiInfo>(inputData);
    LogDebug << "FaceFeature[" << instanceId_ << "]: Begin to process data, face.size=" << frameAiInfo->face.size();
    if (frameAiInfo.get() == nullptr) {
        LogError << "frameAiInfo.get() is null";
        return APP_ERR_COMM_FAILURE;
    }
    std::vector<FaceObject *> faceObjectQueue;
    size_t i = 0;
    size_t bestBatchSize = 0;
    while (i < frameAiInfo->face.size()) {
        ffPreStatic_.RunTimeStatisticStart("Resnet18Pre_excute_time", instanceId_);
        APP_ERROR ret = FindTheBestBatchSize(bestBatchSize, frameAiInfo->face.size() - i);
        if (ret != APP_ERR_OK) {
            LogError << "get the best batch size fail, left face num is " << (frameAiInfo->face.size() - i);
            return APP_ERR_COMM_FAILURE;
        }
        for (size_t j = 0; j < bestBatchSize; ++j) {
            faceObjectQueue.push_back(&(frameAiInfo->face[i + j]));
        }
        i += bestBatchSize;

        ret = PreParaData(faceObjectQueue);
        if (ret != APP_ERR_OK) {
            LogError << "FaceFeature[" << instanceId_ << "]: [FaceFeatureEngine] prepare data failed!";
            faceObjectQueue.clear();
            return ret;
        }
        ffPreStatic_.RunTimeStatisticStop(bestBatchSize);
        ffModelStatic_.RunTimeStatisticStart("Resnet18_model_excute_time", instanceId_);
        ret = modelProcess_->ModelInference(inputBuffers_, inputSizes_, outputBuffers_, outputSizes_, bestBatchSize);
        if (ret != APP_ERR_OK) {
            LogError << "FaceFeature[" << instanceId_ << "]: [FaceFeatureEngine] infer error!";
            faceObjectQueue.clear();
            return ret;
        }
        ffModelStatic_.RunTimeStatisticStop(bestBatchSize);

        ffPostStatic_.RunTimeStatisticStart("Resnet18Post_excute_time", instanceId_);
        ret = PostData(faceObjectQueue);

        ffPostStatic_.RunTimeStatisticStop(bestBatchSize);
        faceObjectQueue.clear();
    }
    faceFeatureStatic_.RunTimeStatisticStop();
    outputQueVec_[frameAiInfo->info.channelId]->Push(frameAiInfo, true);
    return APP_ERR_OK;
}

APP_ERROR FaceFeature::FindTheBestBatchSize(size_t &bestBatchSize, const size_t leftFaceNum) const
{
    if (leftFaceNum == 0) {
        LogError << "left face number is 0";
        return APP_ERR_COMM_FAILURE;
    }
    for (size_t i = 0; i < dynamicBatchSize_.size(); ++i) {
        if (leftFaceNum >= dynamicBatchSize_[i]) {
            bestBatchSize = dynamicBatchSize_[i];
            return APP_ERR_OK;
        }
    }
    return APP_ERR_COMM_FAILURE;
}

double FaceFeature::GetRunTimeAvg()
{
    return faceFeatureStatic_.GetRunTimeAvg();
}
} // namespace ascendFaceRecognition