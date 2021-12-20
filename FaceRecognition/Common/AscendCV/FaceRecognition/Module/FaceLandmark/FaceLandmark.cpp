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
#include "FaceLandmark.h"
#include <fstream>
#include <iostream>
#include <memory>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "PointerDeleter/PointerDeleter.h"
#include "TestCV/TestCV.h"

namespace ascendFaceRecognition {
namespace {
const uint8_t DVPP_ALIGN_BASE_16 = 16;
const uint8_t DVPP_ALIGN_BASE_2 = 2;
const double YUV420SP_RATIO = 1.5;
} // namespace
FaceLandmark::FaceLandmark()
{
    isStop_ = true;
    instanceId_ = -1;
}

FaceLandmark::~FaceLandmark() {}

APP_ERROR FaceLandmark::InitResource(void)
{
    LogDebug << "FaceLandmark[" << instanceId_ << "]: Begin to init device resource.";
    size_t inputNum = modelInfer_->GetModelNumInputs();
    APP_ERROR ret = APP_ERR_OK;
    if (inputNum != 1) {
        LogFatal << "input tensor size is invaild" << inputNum;
        return APP_ERR_COMM_FAILURE;
    }
    LogDebug << "inputNum = " << inputNum;
    for (size_t i = 0; i < inputNum; i++) {
        void *buffer = nullptr;
        size_t size = modelInfer_->GetModelInputSizeByIndex(i);
        ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceLandmark[" << instanceId_ << "]: create input buffer failed!";
            return ret;
        }
        inputBuffers_.push_back(buffer);
        inputSizes_.push_back(size);
        LogDebug << "FaceLandmark[" << instanceId_ << "]: model input tensor i = " << i << ", size= " << size;
    }

    size_t outputNum = modelInfer_->GetModelNumOutputs();
    if (outputNum == 0) {
        LogFatal << "FaceLandmark[" << instanceId_ << "]: output tensor size is invaild " << outputNum;
        return APP_ERR_COMM_FAILURE;
    }
    LogDebug << "FaceLandmark[" << instanceId_ << "]: outputNum = " << outputNum;
    for (size_t i = 0; i < outputNum; i++) {
        void *buffer = nullptr;
        size_t size = modelInfer_->GetModelOutputSizeByIndex(i);
        ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceLandmark[" << instanceId_ << "]: create output failed!";
            return ret;
        }
        outputBuffers_.push_back(buffer);
        outputSizes_.push_back(size);
        LogDebug << "FaceLandmark[" << instanceId_ << "]: model output tensor i = " << i << ", size = " << size;
#ifndef ASCEND_ACL_OPEN_VESION
        auto bufferHost = std::shared_ptr<uint8_t>();
        bufferHost.reset(new uint8_t[size], std::default_delete<uint8_t[]>());
        if (bufferHost == nullptr) {
            LogFatal << "FaceLandmark[" << instanceId_ << "]: create output failed!";
            return APP_ERR_COMM_ALLOC_MEM;
        }
        outputBuffersHost_.push_back(bufferHost);
#endif
    }

    return APP_ERR_OK;
}

APP_ERROR FaceLandmark::ParseConfig(ConfigParser &configParser)
{
    LogDebug << "FaceLandmark[" << instanceId_ << "]: begin to parse config values.";
    std::string itemCfgStr;
    std::vector<APP_ERROR> vecRet;
    const uint8_t itemCfgMaxNum = 64;
    vecRet.resize(itemCfgMaxNum);
    uint8_t indexVecRet = 0;

    itemCfgStr = moduleName_ + std::string(".batch_size");
    vecRet[indexVecRet++] = configParser.GetUnsignedIntValue(itemCfgStr, batchSize_);

    itemCfgStr = moduleName_ + std::string(".width");
    vecRet[indexVecRet++] = configParser.GetIntValue(itemCfgStr, width_);

    itemCfgStr = moduleName_ + std::string(".height");
    vecRet[indexVecRet++] = configParser.GetIntValue(itemCfgStr, height_);

    itemCfgStr = moduleName_ + std::string(".input_channel");
    vecRet[indexVecRet++] = configParser.GetIntValue(itemCfgStr, inputChn_); // colour channel

    itemCfgStr = moduleName_ + std::string(".model_path");
    vecRet[indexVecRet++] = configParser.GetStringValue(itemCfgStr, modelPath_);

    itemCfgStr = std::string("max_face_num_per_frame");
    vecRet[indexVecRet++] = configParser.GetUnsignedIntValue(itemCfgStr, maxFaceNumPerFrame_);

    itemCfgStr = moduleName_ + std::string(".heatmapWidth");
    vecRet[indexVecRet++] = configParser.GetIntValue(itemCfgStr, heatmapWidth_);

    itemCfgStr = moduleName_ + std::string(".heatmapHeight");
    vecRet[indexVecRet++] = configParser.GetIntValue(itemCfgStr, heatmapHeight_);

    itemCfgStr = moduleName_ + std::string(".keyPointWeight");
    vecRet[indexVecRet++] = configParser.GetFloatValue(itemCfgStr, keyPointWeight_);

    itemCfgStr = moduleName_ + std::string(".eulerWeight");
    vecRet[indexVecRet++] = configParser.GetFloatValue(itemCfgStr, eulerWeight_);

    itemCfgStr = moduleName_ + std::string(".bigFaceWeight");
    vecRet[indexVecRet++] = configParser.GetFloatValue(itemCfgStr, bigFaceWeight_);

    itemCfgStr = moduleName_ + std::string(".minOutputScore");
    vecRet[indexVecRet++] = configParser.GetFloatValue(itemCfgStr, minOutputScore_);

    itemCfgStr = moduleName_ + std::string(".registModeEnble");
    vecRet[indexVecRet++] = configParser.GetIntValue(itemCfgStr, registModeEnble_);

    for (uint8_t i = 0; i < indexVecRet; ++i) {
        if (vecRet[i] != APP_ERR_OK) {
            return vecRet[i];
        }
    }

    LogDebug;
    LogDebug << "FaceLandmark[" << instanceId_ << "]" << " batchSize_:" << batchSize_ <<
        " width_:" << width_ << " height_:" << height_ << " inputChn_:" << inputChn_ << " modelPath_:" <<
        modelPath_.c_str() << " maxFaceNumPerFrame_:" << maxFaceNumPerFrame_ << " heatmapWidth_:" << heatmapWidth_ <<
        " heatmapHeight:" << heatmapHeight_ << " registModeEnble_:" << registModeEnble_;

    return APP_ERR_OK;
}

APP_ERROR FaceLandmark::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "FaceLandmark[" << instanceId_ << "]: Begin to init face landmark instance" << initArgs.instanceId;

    AssignInitArgs(initArgs);

    isStop_ = false;

    // initialize config params
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceLandmark[" << instanceId_ << "]: Fail to parse config params, ret=" << ret << "(" <<
            GetAppErrCodeInfo(ret) << ").";
        return ret;
    }

    if (maxFaceNumPerFrame_ < 1) {
        LogFatal << "FaceLandmark[" << instanceId_ << "]: max_face_num_per_frame must be larger than 0";
        return APP_ERR_COMM_FAILURE;
    }

    // init and get model
    modelInfer_ = ModelResource::GetInstance().GetModelProcess(modelPath_);
    if (modelInfer_ == nullptr) {
        LogFatal << "FaceLandmark[" << instanceId_ << "]: init model failed";
        return ret;
    }
    // init resource
    ret = InitResource();
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceLandmark[" << instanceId_ << "]: init resource failed";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR FaceLandmark::DeInit(void)
{
    LogDebug << "FaceLandmark[" << instanceId_ << "]::begin to deinit.";
    
    outputBufferShared_.clear();
    for (uint32_t i = 0; i < outputBuffers_.size(); i++) {
        acldvppFree(outputBuffers_[i]);
        outputBuffers_[i] = nullptr;
    }
    for (uint32_t i = 0; i < inputBuffers_.size(); i++) {
        acldvppFree(inputBuffers_[i]);
        inputBuffers_[i] = nullptr;
    }
    return APP_ERR_OK;
}

APP_ERROR FaceLandmark::PreParaData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue)
{
    LogDebug << "FaceLandmark[" << instanceId_ << "]: prepare model inputs.";
    uint8_t *dataBufferPtr = (uint8_t *)inputBuffers_[0];
    APP_ERROR ret;
    for (uint32_t i = 0; i < faceObjectQueue.size(); i++) {
        auto faceObject = faceObjectQueue[i];
        if (runMode_ == ACL_HOST) { // under ACL_HOST mode, memory from host to device
            ret = aclrtMemcpy(dataBufferPtr, inputSizes_[0] / batchSize_, faceObject->imgQuality.buf.deviceData.get(),
                inputSizes_[0] / batchSize_, ACL_MEMCPY_DEVICE_TO_DEVICE);
        } else if (runMode_ == ACL_DEVICE) { // under ACL_DEVICE mode, memory from device to device
            uint8_t *srcData = (uint8_t *)faceObject->imgQuality.buf.deviceData.get();
            std::copy(srcData, srcData + (inputSizes_[0] / batchSize_), dataBufferPtr);
            ret = APP_ERR_OK;
        }
        if (ret != APP_ERR_OK) {
            LogError << "FaceLandmark[" << instanceId_ << "]: Fail to copy memory from host to device.";
            return ret;
        }
        dataBufferPtr += inputSizes_[0] / batchSize_;
    }
    return APP_ERR_OK;
}

void FaceLandmark::SendData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue)
{
    for (uint32_t i = 0; i < faceObjectQueue.size(); i++) {
        float score = faceObjectQueue[i]->faceQuality.score;
        if (score < minOutputScore_) {
            LogError << "FaceLandmark[" << instanceId_ << "]: score is too low, score=" << score;
            continue;
        }

        if (registModeEnble_ == 0) {
            SendToNextModule(MT_WARP_AFFINE, faceObjectQueue[i], faceObjectQueue[i]->frameInfo.channelId);
            continue;
        }

        if (registFaceObject_.get() == nullptr || ((registFaceObject_.get() != nullptr) 
            && (registFaceObject_->faceQuality.score < faceObjectQueue[i]->faceQuality.score))) {
            registFaceObject_ = faceObjectQueue[i];
        }
    }

    if (registModeEnble_ == 1) {
        if (registFaceObject_.get() != nullptr) {
            SendToNextModule(MT_WARP_AFFINE, registFaceObject_, registFaceObject_->frameInfo.channelId);
            registFaceObject_.reset();
        }
    }
}

APP_ERROR FaceLandmark::PostData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue)
{
    APP_ERROR ret = APP_ERR_OK;
#ifndef ASCEND_ACL_OPEN_VESION
    for (uint32_t i = 0; i < outputBuffersHost_.size(); i++) {
        ret = aclrtMemcpy(outputBuffersHost_[i].get(), outputSizes_[i], 
            outputBuffers_[i], outputSizes_[i], ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != APP_ERR_OK) {
            LogError << "FaceLandmark[" << instanceId_ << "]: Fail to copy memory.";
            return ret; 
        }
    }
#endif
    for (uint32_t i = 0; i < faceObjectQueue.size(); i++) {
        float keyPointScore = CalKeyPointScore(i); // calculate point score
        float eulerScore = CalEulerScore(i);
        float bigFaceScore = CalBigFaceScore(faceObjectQueue[i]);
        float score = keyPointWeight_ * keyPointScore + eulerWeight_ * eulerScore + bigFaceWeight_ * bigFaceScore;
        faceObjectQueue[i]->faceQuality.score = score;

        if (score < minOutputScore_) {
            LogError << "FaceLandmark[" << instanceId_ << "]: score is too low, score=" << score << ", kpScore:" <<
                keyPointScore << ", eulerScore:" << eulerScore << ", bigFaceScore:" << bigFaceScore;
            continue;
        }
        // calculate point
        ret = CalKPCoordinate(i, faceObjectQueue[i]);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceLandmark[" << instanceId_ << "]: calculate key point error.";
            continue;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR FaceLandmark::ProcessWrapper(std::vector<std::shared_ptr<FaceObject>> &faceObjectQueue)
{
    LogDebug << "FaceLandmark[" << instanceId_ << "]: resize data.";

    LogDebug << "FaceLandmark[" << instanceId_ << "]: prepare data.";
    APP_ERROR ret = PreParaData(faceObjectQueue);
    if (ret != APP_ERR_OK) {
        LogError << "FaceLandmark[" << instanceId_ << "]: prepare data error!";
        faceObjectQueue.clear();
        return ret;
    }

    LogDebug << "FaceLandmark[" << instanceId_ << "]: inference.";
    ret = modelInfer_->ModelInference(inputBuffers_, inputSizes_, outputBuffers_, outputSizes_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceLandmark[" << instanceId_ << "]: infer error!";
        faceObjectQueue.clear();
        return ret;
    }

    return ret;
}

APP_ERROR FaceLandmark::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "FaceLandmark[" << instanceId_ << "]: Begin to process data.";
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::static_pointer_cast<FrameAiInfo>(inputData);
    bool last_face = false;
    std::vector<std::shared_ptr<FaceObject>> faceObjectQueue;
    APP_ERROR ret;

    if (frameAiInfo.get() == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }
    LogDebug << "FaceLandmark[" << instanceId_ << "]: hello " << frameAiInfo->detectResult.size();

    for (int i = 0; i < (int)frameAiInfo->face.size(); i++) {
        std::shared_ptr<FaceObject> faceObject = std::make_shared<FaceObject>();
        *faceObject = frameAiInfo->face[i];
        faceObjectQueue.push_back(faceObject);
        last_face = (i == (int)frameAiInfo->detectResult.size() - 1);
        // in register mode, we need not to fill the batch
        if ((faceObjectQueue.size() < batchSize_) && (!last_face)) {
            continue;
        }

        ret = ProcessWrapper(faceObjectQueue);
        if (ret != APP_ERR_OK) {
            return ret;
        }

        LogDebug << "FaceLandmark[" << instanceId_ << "]: postprocess.";
        PostData(faceObjectQueue);
        SendData(faceObjectQueue);
        faceObjectQueue.clear();
    }
    return APP_ERR_OK;
}

APP_ERROR FaceLandmark::CalKPCoordinate(int pos, std::shared_ptr<FaceObject> faceObject)
{
    int batchLen = outputSizes_[1] / batchSize_;
#ifdef ASCEND_ACL_OPEN_VESION
    float *heatmapPtr = (float *)((uint8_t *)outputBuffers_[1] + pos * batchLen);
#else
    float *heatmapPtr = (float *)((uint8_t *)outputBuffersHost_[1].get() + pos * batchLen);
#endif
    uint32_t doubleH = 2;
    uint32_t landmarkLen = heatmapHeight_ * doubleH * sizeof(float);
    auto landmarks = std::make_shared<uint8_t>();
    landmarks.reset(new uint8_t[landmarkLen], std::default_delete<uint8_t[]>());

    uint8_t *landmarkPtr = landmarks.get();

    const float elementPositionLimit = 48.0;
    for (int i = 0; i < heatmapHeight_; i++) {
        float *tmpPtr = heatmapPtr + i * heatmapWidth_;

        int position = std::max_element(tmpPtr, tmpPtr + heatmapWidth_) - tmpPtr;

        float x = (float)((position % uint32_t(elementPositionLimit)) / elementPositionLimit);
        std::copy((uint8_t *)&x, (uint8_t *)&x + sizeof(float), landmarkPtr);
        landmarkPtr += sizeof(float);
        float y = (float)((position / elementPositionLimit) / elementPositionLimit);
        std::copy((uint8_t *)&y, (uint8_t *)&y + sizeof(float), landmarkPtr);
        landmarkPtr += sizeof(float);
    }
#ifdef ASCEND_ACL_OPEN_VESION
    faceObject->landmarks.deviceData = landmarks;
#else
    faceObject->landmarks.hostData = landmarks;
#endif
    faceObject->landmarks.dataSize = landmarkLen;

    return APP_ERR_OK;
}

float FaceLandmark::CalKeyPointScore(int pos)
{
    int batchLen = outputSizes_[1] / batchSize_;
#ifdef ASCEND_ACL_OPEN_VESION
    float *heatmapPtr = (float *)((uint8_t *)outputBuffers_[1] + pos * batchLen);
#else
    float *heatmapPtr = (float *)((uint8_t *)outputBuffersHost_[1].get() + pos * batchLen);
#endif
    float score = 0.0;
    const float elementScoreLimit = 0.2;
    for (int i = 0; i < heatmapHeight_; i++) {
        float *tmpPtr = heatmapPtr + i * heatmapWidth_;
        float tmpScore = *std::max_element(tmpPtr, tmpPtr + heatmapWidth_);
        score += ((tmpScore > elementScoreLimit) ? elementScoreLimit : tmpScore);
    }
    return score;
}

float FaceLandmark::CalEulerScore(int pos)
{
    int batchLen = outputSizes_[0] / batchSize_;
#ifdef ASCEND_ACL_OPEN_VESION
    float *euler_ptr = (float *)((uint8_t *)outputBuffers_[0] + pos * batchLen);
#else
    float *euler_ptr = (float *)((uint8_t *)outputBuffersHost_[0].get() + pos * batchLen);
#endif
    const uint16_t degree90 = 90;
    uint8_t indexEulerPtr = 0;
    float yaw = fabs(euler_ptr[indexEulerPtr++]) * degree90;
    float pitch = fabs(euler_ptr[indexEulerPtr++]) * degree90;
    float roll = fabs(euler_ptr[indexEulerPtr++]) * degree90;

    const uint8_t pitchConstant = 6;
    pitch = (pitch > pitchConstant) ? (pitch - pitchConstant) : 0;

    return (degree90 - yaw) / degree90 + (degree90 - pitch) / degree90 + (degree90 - roll) / degree90;
}

float FaceLandmark::CalBigFaceScore(std::shared_ptr<FaceObject> face)
{
    float width = face->info.width;
    float height = face->info.height;

    const uint16_t maxFaceHW = 60;
    const uint16_t normFaceHW = 50;
    const double faceStretchRatio = 1.2;
    const double faceScoreConstant = 3600.0;

    width = (width > normFaceHW) ? maxFaceHW : (width * faceStretchRatio);
    height = (height > normFaceHW) ? maxFaceHW : (height * faceStretchRatio);

    return 1 - fabs(maxFaceHW - width) * fabs(maxFaceHW - height) / faceScoreConstant;
}
} // namespace ascendFaceRecognition
