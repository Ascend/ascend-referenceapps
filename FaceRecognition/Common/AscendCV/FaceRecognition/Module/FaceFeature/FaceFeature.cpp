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
#include <cstring>
#include <cmath>

#include "ConfigParser/ConfigParser.h"
#include "Log/Log.h"
#include "FileEx/FileEx.h"

namespace ascendFaceRecognition {
namespace {
const int FACE_FEATURE_WIDTH_ALIGN = 16;
const int FACE_FEATURE_HEIGHT_ALIGN = 2;
const uint32_t CPU_NORMALIZATION_MODE = 0;
}

FaceFeature::FaceFeature()
{
    isStop_ = true;
    instanceId_ = -1;
}

FaceFeature::~FaceFeature() {}

APP_ERROR FaceFeature::ParseConfig(ConfigParser &configParser)
{
    // load and parse config file
    LogDebug << "FaceFeature[" << instanceId_ << "]: start to parse config.";

    std::string itemCfgStr = moduleName_ + "." + pipelineName_ + std::string(".batch_size");
    APP_ERROR ret = configParser.GetUnsignedIntValue(itemCfgStr, batchSize_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".width");
    ret = configParser.GetIntValue(itemCfgStr, width_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    alignedWidth = ALIGN_UP(width_, FACE_FEATURE_WIDTH_ALIGN);

    itemCfgStr = moduleName_ + std::string(".height");
    ret = configParser.GetIntValue(itemCfgStr, height_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    alignedHeight = ALIGN_UP(height_, FACE_FEATURE_HEIGHT_ALIGN);

    itemCfgStr = moduleName_ + std::string(".input_channel");
    ret = configParser.GetIntValue(itemCfgStr, inputChn_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + "." + pipelineName_ + std::string(".model_path");
    ret = configParser.GetStringValue(itemCfgStr, modelPath_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".normal_mode");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, normalMode_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    inputSize_ = batchSize_ * inputChn_ * alignedHeight * alignedWidth;

    return ret;
}

APP_ERROR FaceFeature::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "FaceFeature[" << initArgs.instanceId << "]: start to init.";
    // initialize member variables
    AssignInitArgs(initArgs);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to parse config params, ret=" << ret;
        return ret;
    }
    isStop_ = false;
    modelInfer_ = ModelResource::GetInstance().GetModelProcess(modelPath_); // init and get model
    if (modelInfer_ == nullptr) {
        LogFatal << "FaceFeature[" << instanceId_ << "]::init model failed";
        return APP_ERR_COMM_FAILURE;
    }
    size_t inputNum = modelInfer_->GetModelNumInputs();
    LogDebug << "FaceFeature[" << instanceId_ << "]: inputNum =" << inputNum;
    for (size_t i = 0; i < inputNum; i++) {
        void *buffer = nullptr;
        // modify size
        size_t size = modelInfer_->GetModelInputSizeByIndex(i);
        ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceFeature[" << instanceId_ << "]: aclrtMalloc fail.ret=" << ret;
            return ret;
        }
        inputBuffers_.push_back(buffer);
        inputSizes_.push_back(size);
        LogDebug << "FaceFeature[" << instanceId_ << "]: i=" << i << ", size=" << size;
    }
    size_t outputNum = modelInfer_->GetModelNumOutputs();
    LogDebug << "FaceFeature[" << instanceId_ << "]: outputNum = " << outputNum;
    for (size_t i = 0; i < outputNum; i++) {
        void *buffer = nullptr;
        size_t size = modelInfer_->GetModelOutputSizeByIndex(i);
        ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceFeature[" << instanceId_ << "]: output aclrtMalloc fail.ret=" << ret;
            return ret;
        }
        outputBuffers_.push_back(buffer);
        outputSizes_.push_back(size);
        LogDebug << "FaceFeature[" << instanceId_ << "]: i = " << i << ", size=" << size;
#ifndef ASCEND_ACL_OPEN_VESION
        auto bufferHost = std::shared_ptr<uint8_t>();
        bufferHost.reset(new uint8_t[size], std::default_delete<uint8_t[]>());
        outputBuffersHost_.push_back(bufferHost);
#endif
    }
    return ret;
}

APP_ERROR FaceFeature::DeInit(void)
{
    LogDebug << "FaceFeature[" << instanceId_ << "]: deinit start";
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

APP_ERROR FaceFeature::PreParaData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue)
{
    LogDebug << "FaceFeature[" << instanceId_ << "]: prepare data.";
    APP_ERROR ret = 0;
    int8_t *dataBufferPtr = (int8_t *)inputBuffers_[0];
    for (uint32_t i = 0; i < faceObjectQueue.size(); i++) {
        std::shared_ptr<FaceObject> faceObjectPtr = faceObjectQueue[i];
        if (runMode_ == ACL_HOST) { // under ACL_HOST mode, memory from host to device
            ret = aclrtMemcpy(dataBufferPtr, inputSizes_[0] / batchSize_, faceObjectPtr->imgAffine.buf.deviceData.get(),
                faceObjectPtr->imgAffine.buf.dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
        } else if (runMode_ == ACL_DEVICE) { // under ACL_DEVICE mode, memory from device to device
            std::copy(faceObjectPtr->imgAffine.buf.deviceData.get(),
                faceObjectPtr->imgAffine.buf.deviceData.get() + (inputSizes_[0] / batchSize_), dataBufferPtr);
            ret = APP_ERR_OK;
        }
        if (ret != APP_ERR_OK) {
            LogError << "FaceFeature[" << instanceId_ << "]::PreParaData: memcpy failed, dest_size=" <<
                (inputSizes_[0] / batchSize_) << ", src_size=" << faceObjectPtr->imgAffine.buf.dataSize;
            return ret;
        }
        dataBufferPtr += inputSizes_[0] / batchSize_;
    }
    // pad to kBatchSize
    if (faceObjectQueue.size() < (size_t)batchSize_) {
        int padNum = batchSize_ - faceObjectQueue.size();
        ret = aclrtMemset(dataBufferPtr, inputSizes_[0] / batchSize_ * padNum, static_cast<char>(0),
            inputSizes_[0] / batchSize_ * padNum);
        if (ret != APP_ERR_OK) {
            LogError << "FaceFeature[" << instanceId_ << "]::PreParaData: memcpy failed.";
            return ret;
        }
    }
    return APP_ERR_OK;
}

float FaceFeature::FeatureNorm(const float *feature, const uint32_t &featureSize)
{
    // calculate norm
    float norm = 0.f;
    for (uint32_t i = 0; i < featureSize; i++) {
        norm += feature[i] * feature[i];
    }
    norm = sqrt(norm);
    return norm;
}

APP_ERROR FaceFeature::PostData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue)
{
    LogDebug << "FaceFeature[" << instanceId_ << "]: post process.";
    APP_ERROR ret = 0;
    int featureVecterSize = outputSizes_[0] / batchSize_;
    uint8_t *resPtr = (uint8_t *)outputBuffers_[0];
    for (size_t i = 0; i < faceObjectQueue.size(); i++) {
        auto outBuffer = std::make_shared<uint8_t>();
        outBuffer.reset(new uint8_t[featureVecterSize], std::default_delete<uint8_t[]>());
        if (runMode_ == ACL_HOST) { // under ACL_HOST mode, memory from device to host
            ret = aclrtMemcpy(outBuffer.get(), featureVecterSize, resPtr, featureVecterSize, ACL_MEMCPY_DEVICE_TO_HOST);
        } else if (runMode_ == ACL_DEVICE) { // under ACL_DEVICE mode, memory from device to device
            std::copy(resPtr, resPtr + featureVecterSize, outBuffer.get());
            ret = APP_ERR_OK;
        }
        if (ret != APP_ERR_OK) {
            LogError << "FaceFeature[" << instanceId_ << "]::PreParaData: memcpy failed.";
            return ret;
        }
        float norm = 1.f;
        if (normalMode_ == CPU_NORMALIZATION_MODE) {
            norm = FeatureNorm((float *)outBuffer.get(), featureVecterSize / sizeof(float));
        }
#ifdef ASCEND_ACL_OPEN_VESION
        faceObjectQueue[i]->featureVector.deviceData = outBuffer;
#else
        faceObjectQueue[i]->featureVector.hostData = outBuffer;
#endif
        faceObjectQueue[i]->featureVector.dataSize = featureVecterSize;
        faceObjectQueue[i]->featureNorm = norm;
        resPtr += featureVecterSize;
    }
    return APP_ERR_OK;
}

APP_ERROR FaceFeature::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "FaceFeature[" << instanceId_ << "]: Begin to process data.";
    std::shared_ptr<FaceObject> faceObject = std::static_pointer_cast<FaceObject>(inputData);
    if (faceObject.get() == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }
    faceObjectVec_.push_back(faceObject);
    if (faceObjectVec_.size() < batchSize_)
        return APP_ERR_OK;
    // data prepare
    APP_ERROR ret = PreParaData(faceObjectVec_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceFeature[" << instanceId_ << "]: [FaceFeatureEngine] prepare data failed!";
        faceObjectVec_.clear();
        return ret;
    }
    // model inference
    ret = modelInfer_->ModelInference(inputBuffers_, inputSizes_, outputBuffers_, outputSizes_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceFeature[" << instanceId_ << "]: [FaceFeatureEngine] infer error!";
        faceObjectVec_.clear();
        return ret;
    }
    // data postprocess
    ret = PostData(faceObjectVec_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceFeature[" << instanceId_ << "]: [FaceFeatureEngine] postprocess data failed!";
        faceObjectVec_.clear();
        return ret;
    }
    for (uint32_t i = 0; i < faceObjectVec_.size(); i++) {
        std::shared_ptr<FaceObject> faceObject = faceObjectVec_[i];
        if (faceObject->frameInfo.mode == FRAME_MODE_SEARCH) {
            SendToNextModule(MT_FACE_SEARCH, faceObject, faceObject->frameInfo.channelId);
        } else {
            SendToNextModule(MT_FACE_STOCK, faceObject, faceObject->frameInfo.channelId);
        }
    }
    faceObjectVec_.clear();
    return APP_ERR_OK;
}
} // namespace ascendFaceRecognition
