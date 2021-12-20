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

#include "MOTEmbedding.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <cstring>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "FrameCache/FrameCache.h"

namespace ascendFaceRecognition {
namespace {
const uint32_t CPU_NORMALIZATION_MODE = 0;
} // namespace

MOTEmbedding::MOTEmbedding()
{
    isStop_ = true;
    instanceId_ = -1;
}

MOTEmbedding::~MOTEmbedding() {}

APP_ERROR MOTEmbedding::InitResource(void)
{
    LogDebug << "MOTEmbedding[" << instanceId_ << "]: Begin to init device resource.";
    size_t inputNum = modelInfer_->GetModelNumInputs();
    if (inputNum != 1) {
        LogFatal << "input tensor size is invaild" << inputNum;
        return APP_ERR_COMM_FAILURE;
    }
    LogDebug << "inputNum = " << inputNum;
    for (size_t i = 0; i < inputNum; i++) {
        void *buffer = nullptr;
        size_t size = modelInfer_->GetModelInputSizeByIndex(i);
        APP_ERROR ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != APP_ERR_OK) {
            LogFatal << "MOTEmbedding[" << instanceId_ << "]: create input buffer failed!";
            return ret;
        }
        inputBuffers_.push_back(buffer);
        inputSizes_.push_back(size);
        LogDebug << "MOTEmbedding[" << instanceId_ << "]: model input tensor i = " << i << ", size= " << size;
    }

    size_t outputNum = modelInfer_->GetModelNumOutputs();
    if (outputNum == 0) {
        LogFatal << "MOTEmbedding[" << instanceId_ << "]: output tensor size is invaild " << outputNum;
        return APP_ERR_COMM_FAILURE;
    }
    LogDebug << "MOTEmbedding[" << instanceId_ << "]: outputNum = " << outputNum;
    for (size_t i = 0; i < outputNum; i++) {
        void *buffer = nullptr;
        size_t size = modelInfer_->GetModelOutputSizeByIndex(i);
        APP_ERROR ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != APP_ERR_OK) {
            LogFatal << "MOTEmbedding[" << instanceId_ << "]: create output failed!";
            return ret;
        }
        outputBuffers_.push_back(buffer);
        outputSizes_.push_back(size);
        LogDebug << "MOTEmbedding[" << instanceId_ << "]: model output tensor i = " << i << ", size = " << size;
    }

    return APP_ERR_OK;
}

APP_ERROR MOTEmbedding::ParseConfig(ConfigParser &configParser)
{
    LogDebug << "MOTEmbedding[" << instanceId_ << "]: begin to parse config values.";
    std::string itemCfgStr = moduleName_ + std::string(".batch_size");
    APP_ERROR ret = configParser.GetUnsignedIntValue(itemCfgStr, batchSize_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".enable");
    ret = configParser.GetIntValue(itemCfgStr, enable_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".model_path");
    ret = configParser.GetStringValue(itemCfgStr, modelPath_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".normal_mode");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, normalMode_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    LogDebug << "MOTEmbedding[" << instanceId_ << "]" << " batchSize_:" << batchSize_ <<
        " modelPath_:" << modelPath_.c_str();

    return ret;
}

APP_ERROR MOTEmbedding::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "MOTEmbedding[" << instanceId_ << "]: Begin to init MOT feature extraction instance" <<
        initArgs.instanceId;

    AssignInitArgs(initArgs);

    isStop_ = false;

    // initialize config params
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "MOTEmbedding[" << instanceId_ << "]: Fail to parse config params, ret=" << ret << "(" <<
            GetAppErrCodeInfo(ret) << ").";
        return ret;
    }

    // init model
    modelInfer_ = ModelResource::GetInstance().GetModelProcess(modelPath_, instanceId_);
    if (modelInfer_ == nullptr) {
        LogFatal << "FaceDetection[" << instanceId_ << "]::init model failed";
        return APP_ERR_COMM_FAILURE;
    }

    // init resource
    ret = InitResource();
    if (ret != APP_ERR_OK) {
        LogFatal << "MOTEmbedding[" << instanceId_ << "]: init resource failed";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR MOTEmbedding::DeInit(void)
{
    LogDebug << "MOTEmbedding[" << instanceId_ << "]::begin to deinit.";
    for (uint32_t i = 0; i < outputBuffers_.size(); i++) {
        aclrtFree(outputBuffers_[i]);
        outputBuffers_[i] = nullptr;
    }
    for (uint32_t i = 0; i < inputBuffers_.size(); i++) {
        aclrtFree(inputBuffers_[i]);
        inputBuffers_[i] = nullptr;
    }

    return APP_ERR_OK;
}

APP_ERROR MOTEmbedding::PreParaData(std::vector<std::shared_ptr<FaceObject>> &faceObjectQueue)
{
    LogDebug << "MOTEmbedding[" << instanceId_ << "]: prepare model inputs.";
    int8_t *dataBufferPtr = (int8_t *)inputBuffers_[0];

    for (uint32_t i = 0; i < faceObjectQueue.size(); i++) {
        std::shared_ptr<FaceObject> faceObjectPtr = faceObjectQueue[i];
#ifndef ASCEND_ACL_OPEN_VESION
        APP_ERROR ret = aclrtMemcpy(dataBufferPtr, inputSizes_[0] / batchSize_,
            faceObjectPtr->imgEmbedding.buf.deviceData.get(), inputSizes_[0] / batchSize_, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != APP_ERR_OK) {
            LogError << "MOTEmbedding[" << instanceId_ << "]: Fail to copy memory.";
            return APP_ERR_COMM_FAILURE;
        }
#else
        std::copy(faceObjectPtr->imgEmbedding.buf.deviceData.get(),
            faceObjectPtr->imgEmbedding.buf.deviceData.get() + inputSizes_[0] / batchSize_, dataBufferPtr);
#endif
        dataBufferPtr += inputSizes_[0] / batchSize_;
    }

    return APP_ERR_OK;
}

APP_ERROR MOTEmbedding::PreParaData(std::shared_ptr<FaceObject> faceObject)
{
    LogDebug << "MOTEmbedding[" << instanceId_ << "]: prepare model inputs.";
    if (faceObjectQueue_.size() < batchSize_) {
        uint8_t *dataBufferPtr = (uint8_t *)inputBuffers_[0];
        dataBufferPtr += inputSizes_[0] / batchSize_ * faceObjectQueue_.size();
#ifndef ASCEND_ACL_OPEN_VESION
        APP_ERROR ret = aclrtMemcpy(dataBufferPtr, inputSizes_[0] / batchSize_,
            faceObject->imgEmbedding.buf.deviceData.get(), inputSizes_[0] / batchSize_, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != APP_ERR_OK) {
            LogError << "MOTEmbedding[" << instanceId_ << "]: Fail to copy memory.";
            return APP_ERR_COMM_FAILURE;
        }
#else
        std::copy(faceObject->imgEmbedding.buf.deviceData.get(),
            faceObject->imgEmbedding.buf.deviceData.get() + inputSizes_[0] / batchSize_, dataBufferPtr);
#endif
        faceObjectQueue_.push_back(faceObject);
        return APP_ERR_OK;
    }
    return APP_ERR_COMM_FAILURE;
}

float MOTEmbedding::FeatureNorm(const float *feature, const uint32_t &featureSize)
{
    // calculate norm
    float norm = 0.f;
    for (uint32_t i = 0; i < featureSize; i++) {
        norm += feature[i] * feature[i];
    }
    norm = sqrt(norm);
    return norm;
}

APP_ERROR MOTEmbedding::PostData(std::vector<std::shared_ptr<FaceObject>> &faceObjectQueue)
{
    LogDebug << "MOTEmbedding[" << instanceId_ << "]:post process model outputs.";
    int featureVecterSize = outputSizes_[0] / batchSize_; // acltodo for special model. outputSizes_[0] / batchSize_;
    uint8_t *resPtr = (uint8_t *)outputBuffers_[0];       // acltodo for special model.  outputBuffers_[0];
    APP_ERROR ret = APP_ERR_OK;
    for (size_t i = 0; i < faceObjectQueue.size(); i++) {
        auto outBuffer = std::shared_ptr<uint8_t>();
        outBuffer.reset(new uint8_t[featureVecterSize], std::default_delete<uint8_t[]>());
        if (outBuffer == nullptr) {
            LogFatal << "embedding aclrtMalloc failed";
            return APP_ERR_COMM_FAILURE;
        }

        if (runMode_ == ACL_HOST) { // under ACL_HOST mode, memory from device to host
            ret = aclrtMemcpy(outBuffer.get(), featureVecterSize, resPtr, featureVecterSize, ACL_MEMCPY_DEVICE_TO_HOST);
        } else if (runMode_ == ACL_DEVICE) { // under ACL_DEVICE mode, memory from device to device
            std::copy(resPtr, resPtr + featureVecterSize, outBuffer.get());
            ret = APP_ERR_OK;
        }
        if (ret != APP_ERR_OK) {
            LogError << "MOTEmbedding[" << instanceId_ << "]: aclrtMemcpy error!";
            return ret;
        }

        float norm = 1.f;
        if (normalMode_ == CPU_NORMALIZATION_MODE) {
            norm = FeatureNorm((float *)outBuffer.get(), featureVecterSize / sizeof(float));
        }

        if (runMode_ == ACL_HOST) { // reset host resources
            faceObjectQueue[i]->embedding.hostData = outBuffer;
        } else if (runMode_ == ACL_DEVICE) { // reset device resources
            faceObjectQueue[i]->embedding.deviceData = outBuffer;
        }
        faceObjectQueue[i]->embedding.dataSize = featureVecterSize;
        faceObjectQueue[i]->embeddingNorm = norm;
        resPtr += featureVecterSize;
    }

    return APP_ERR_OK;
}

APP_ERROR MOTEmbedding::ProcessFaceObjectQueue()
{
    // inference
    APP_ERROR ret = modelInfer_->ModelInference(inputBuffers_, inputSizes_, outputBuffers_, outputSizes_);
    if (ret != APP_ERR_OK) {
        LogError << "MOTEmbedding[" << instanceId_ << "]: infer error!";
        faceObjectQueue_.clear();
        return ret;
    }
    // post process outputs
    PostData(faceObjectQueue_);

    for (size_t i = 0; i < faceObjectQueue_.size(); i++) {
        std::shared_ptr<FaceObject> theFace = faceObjectQueue_[i];
        std::shared_ptr<FrameAiInfo> theFrameAiInfo =
            FrameCache::GetInstance(theFace->frameInfo.channelId)->GetFrame(theFace->frameInfo.frameId);
        theFrameAiInfo->face[theFrameAiInfo->embeddingCount] = *theFace;
        theFrameAiInfo->embeddingCount++;
        if ((uint32_t)theFrameAiInfo->embeddingCount == theFrameAiInfo->face.size()) {
            LogDebug << "[Cache] " << theFrameAiInfo->embeddingCount << " Faces Recv Send Ch:" <<
                theFrameAiInfo->info.channelId << " frame:" << theFrameAiInfo->info.frameId;
            SendToNextModule(MT_MOT_CONNECTION, theFrameAiInfo, theFrameAiInfo->info.channelId);
            FrameCache::GetInstance(theFrameAiInfo->info.channelId)->ClearFrame(theFrameAiInfo->info.frameId);
        }
    }
    return ret;
}

APP_ERROR MOTEmbedding::ProcessEmptyFaceFrame(std::shared_ptr<FrameAiInfo> frameAiInfo)
{
    SendToNextModule(MT_MOT_CONNECTION, frameAiInfo, frameAiInfo->info.channelId);
    FrameCache::GetInstance(frameAiInfo->info.channelId)->ClearFrame(frameAiInfo->info.frameId);
    return APP_ERR_OK;
}

APP_ERROR MOTEmbedding::ProcessMulitFaceFrame(std::shared_ptr<FrameAiInfo> frameAiInfo,
    std::shared_ptr<FaceObject> faceObject)
{
    if (enable_ && faceObject->imgEmbedding.buf.deviceData.get() != nullptr) {
        if (faceObjectQueue_.size() < batchSize_) {
            APP_ERROR ret = PreParaData(faceObject);
            if (ret != APP_ERR_OK) {
                LogError << "MOTEmbedding[" << instanceId_ << "]: preprocess error! ret=" << ret;
                return ret;
            }
            faceObject->imgEmbedding.buf.deviceData.reset();
        }
        if (faceObjectQueue_.size() < batchSize_) {
            return APP_ERR_OK;
        }
        APP_ERROR ret = ProcessFaceObjectQueue();
        if (ret != APP_ERR_OK) {
            faceObjectQueue_.clear();
            return ret;
        }
        faceObjectQueue_.clear();
    } else {
        frameAiInfo->face[frameAiInfo->embeddingCount] = *faceObject;
        frameAiInfo->embeddingCount++;
        LogDebug << "FrameCache MOTEmbedding Recv " << faceObject->frameInfo.channelId << "_" <<
            faceObject->frameInfo.frameId << "Count " << frameAiInfo->embeddingCount;
        // Read Cache
        if ((uint32_t)frameAiInfo->embeddingCount == frameAiInfo->face.size()) {
            LogDebug << "[Cache] " << frameAiInfo->embeddingCount << " Faces Recv Send Ch:" <<
                frameAiInfo->info.channelId << " frame:" << frameAiInfo->info.frameId;
            SendToNextModule(MT_MOT_CONNECTION, frameAiInfo, frameAiInfo->info.channelId);
            FrameCache::GetInstance(frameAiInfo->info.channelId)->ClearFrame(frameAiInfo->info.frameId);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR MOTEmbedding::Process(std::shared_ptr<void> inputData)
{
    std::shared_ptr<FaceObject> faceObject = std::static_pointer_cast<FaceObject>(inputData);
    if (faceObject.get() == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }

    LogDebug << "MOTEmbedding Ch #" << faceObject->frameInfo.channelId << " Frame #" << faceObject->frameInfo.frameId;

    std::shared_ptr<FrameAiInfo> frameAiInfo =
        FrameCache::GetInstance(faceObject->frameInfo.channelId)->GetFrame(faceObject->frameInfo.frameId);
    if (frameAiInfo == nullptr) {
        LogError << "FrameCache MOTEmbedding[" << instanceId_ << "]: Cann't Found FrameAiInfo!";
        return APP_ERR_COMM_FAILURE;
    }

    APP_ERROR ret;
    if (frameAiInfo->face.size() == 0) {
        ret = ProcessEmptyFaceFrame(frameAiInfo);
    } else {
        ret = ProcessMulitFaceFrame(frameAiInfo, faceObject);
    }
    return ret;
}
} // namespace ascendFaceRecognition
