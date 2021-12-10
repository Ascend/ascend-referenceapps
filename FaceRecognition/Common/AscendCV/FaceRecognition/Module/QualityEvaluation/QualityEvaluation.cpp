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
#include "QualityEvaluation.h"

#include <cmath>
#include <memory>

#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "PointerDeleter/PointerDeleter.h"
#include "FaceBlockingMap.h"

namespace ascendFaceRecognition {
QualityEvaluation::QualityEvaluation()
{
    isStop_ = true;
    instanceId_ = -1;
}

QualityEvaluation::~QualityEvaluation()
{
}

APP_ERROR QualityEvaluation::InitResource(void)
{
    LogDebug << "QualityEvaluation[" << instanceId_ << "]: Begin to init device resource.";
    size_t inputNum = modelInfer_->GetModelNumInputs();
    if (inputNum != 1) {
        LogFatal << "input tensor size is invaild" << inputNum;
        return APP_ERR_COMM_FAILURE;
    }
    LogDebug << "inputNum = " << inputNum;
    APP_ERROR ret = APP_ERR_OK;
    for (size_t i = 0; i < inputNum; i++) {
        void *buffer = nullptr;
        size_t size = modelInfer_->GetModelInputSizeByIndex(i);
        ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != APP_ERR_OK) {
            LogFatal << "QualityEvaluation[" << instanceId_ << "]: create input buffer failed!";
            return ret;
        }
        inputBuffers_.push_back(buffer);
        inputSizes_.push_back(size);
        LogDebug << "QualityEvaluation[" << instanceId_ << "]: model input tensor i = " << i << ", size= " << size;
    }

    size_t outputNum = modelInfer_->GetModelNumOutputs();
    if (outputNum == 0) { // acltodo if(outputNum != 1)
        LogFatal << "QualityEvaluation[" << instanceId_ << "]: output tensor size is invaild " << outputNum;
        return APP_ERR_COMM_FAILURE;
    }

    LogDebug << "QualityEvaluation[" << instanceId_ << "]: outputNum = " << outputNum;
    for (size_t i = 0; i < outputNum; i++) {
        void *buffer = nullptr;
        size_t size = modelInfer_->GetModelOutputSizeByIndex(i);
        ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != APP_ERR_OK) {
            LogFatal << "QualityEvaluation[" << instanceId_ << "]: create output failed!";
            return ret;
        }
        outputBuffers_.push_back(buffer);
        outputSizes_.push_back(size);
        LogDebug << "QualityEvaluation[" << instanceId_ << "]: model output tensor i = " << i << ", size = " << size;
#ifndef ASCEND_ACL_OPEN_VESION
        auto bufferHost = std::shared_ptr<uint8_t>();
        bufferHost.reset(new uint8_t[size], std::default_delete<uint8_t[]>());
        if (bufferHost == nullptr) {
            LogFatal << "QualityEvaluation[" << instanceId_ << "]: create output failed!";
            return APP_ERR_COMM_ALLOC_MEM;
        }
        outputBuffersHost_.push_back(bufferHost);
#endif
    }

    return APP_ERR_OK;
}

APP_ERROR QualityEvaluation::ParseConfig(ConfigParser &configParser)
{
    LogDebug << "QualityEvaluation[" << instanceId_ << "]: begin to parse config values.";
    std::string itemCfgStr;

    std::vector<APP_ERROR> vecRet;
    const uint32_t maxCfgItemsNum = 32;
    uint32_t indexVec = 0;
    vecRet.resize(maxCfgItemsNum);

    itemCfgStr = moduleName_ + std::string(".batch_size");
    vecRet[indexVec++] = configParser.GetUnsignedIntValue(itemCfgStr, batchSize_);
    itemCfgStr = moduleName_ + std::string(".heatmapWidth");
    vecRet[indexVec++] = configParser.GetIntValue(itemCfgStr, heatmapWidth_);
    itemCfgStr = moduleName_ + std::string(".heatmapHeight");
    vecRet[indexVec++] = configParser.GetIntValue(itemCfgStr, heatmapHeight_);
    itemCfgStr = moduleName_ + std::string(".keyPointWeight");
    vecRet[indexVec++] = configParser.GetFloatValue(itemCfgStr, keyPointWeight_);
    itemCfgStr = moduleName_ + std::string(".eulerWeight");
    vecRet[indexVec++] = configParser.GetFloatValue(itemCfgStr, eulerWeight_);
    itemCfgStr = moduleName_ + std::string(".bigFaceWeight");
    vecRet[indexVec++] = configParser.GetFloatValue(itemCfgStr, bigFaceWeight_);
    itemCfgStr = moduleName_ + std::string(".minOutputScore");
    vecRet[indexVec++] = configParser.GetFloatValue(itemCfgStr, minOutputScore_);
    itemCfgStr = moduleName_ + std::string(".model_path");
    vecRet[indexVec++] = configParser.GetStringValue(itemCfgStr, modelPath_);
    itemCfgStr = moduleName_ + std::string(".maxSendTime");
    vecRet[indexVec++] = configParser.GetDoubleValue(itemCfgStr, maxSendTime_);

    for (uint32_t i = 0; i < indexVec; ++i) {
        if (vecRet[i] != APP_ERR_OK) {
            return vecRet[i];
        }
    }

    LogDebug;
    LogDebug << "QualityEvaluation[" << instanceId_ << "]: batchSize_:" << batchSize_ << " heatmapWidth_:" <<
        heatmapWidth_ << " heatmapHeight_:" << heatmapHeight_ << " keyPointWeight_:" << keyPointWeight_ <<
        " eulerWeight_:" << eulerWeight_ << " bigFaceWeight_:" << bigFaceWeight_ << " modelPath_:" <<
        modelPath_.c_str();

    return APP_ERR_OK;
}

APP_ERROR QualityEvaluation::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "QualityEvaluation[" << instanceId_ << "]: Begin to init quality evaluation instance" <<
        initArgs.instanceId;

    AssignInitArgs(initArgs);

    isStop_ = false;

    // initialize config params
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "QualityEvaluation[" << instanceId_ << "]: Fail to parse config params, ret=" << ret << "(" <<
            GetAppErrCodeInfo(ret) << ").";
        return ret;
    }

    // init model
    modelInfer_ = ModelResource::GetInstance().GetModelProcess(modelPath_, instanceId_);
    if (modelInfer_ == nullptr) {
        LogFatal << "QualityEvaluation[" << instanceId_ << "]::init model failed";
        return APP_ERR_COMM_FAILURE;
    }

    // init resource
    ret = InitResource();
    if (ret != APP_ERR_OK) {
        LogFatal << "QualityEvaluation[" << instanceId_ << "]: init resource failed";
        return ret;
    }

    sendThread_ = std::thread(&QualityEvaluation::SendThread, this);
    return APP_ERR_OK;
}

APP_ERROR QualityEvaluation::DeInit(void)
{
    LogDebug << "QualityEvaluation[" << instanceId_ << "]::begin to deinit.";

    for (uint32_t i = 0; i < outputBuffers_.size(); i++) {
        aclrtFree(outputBuffers_[i]);
        outputBuffers_[i] = nullptr;
    }
    for (uint32_t i = 0; i < inputBuffers_.size(); i++) {
        aclrtFree(inputBuffers_[i]);
        inputBuffers_[i] = nullptr;
    }

    sendStop_ = true;
    if (sendThread_.joinable()) {
        sendThread_.join();
    }

    return APP_ERR_OK;
}


APP_ERROR QualityEvaluation::PreParaData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue)
{
    LogDebug << "QualityEvaluation[" << instanceId_ << "]: prepare model inputs.";
    int batchLen = inputSizes_[0] / batchSize_;
    int8_t *dataBufferPtr = (int8_t *)inputBuffers_[0];
    APP_ERROR ret = 0;
    for (uint32_t i = 0; i < faceObjectQueue.size(); i++) {
        if (runMode_ == ACL_HOST) { // under ACL_HOST mode, memory from host to device
            ret = aclrtMemcpy(dataBufferPtr, batchLen, faceObjectQueue[i]->imgQuality.buf.deviceData.get(), batchLen,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
        } else if (runMode_ == ACL_DEVICE) { // under ACL_DEVICE mode, memory from device to device
            std::copy(faceObjectQueue[i]->imgQuality.buf.deviceData.get(),
                faceObjectQueue[i]->imgQuality.buf.deviceData.get() + batchLen, dataBufferPtr);
            ret = APP_ERR_OK;
        }
        if (ret != APP_ERR_OK) {
            LogError << "QualityEvaluation[" << instanceId_ << "]: Fail to copy memory.";
            return ret;
        }
        dataBufferPtr += batchLen;
    }
    return APP_ERR_OK;
}

APP_ERROR QualityEvaluation::PreParaData(std::shared_ptr<FaceObject> faceObject)
{
    if (faceObjectVec_.size() < batchSize_) {
        uint32_t batchLen = inputSizes_[0] / batchSize_;
        uint8_t *dataBufferPtr = (uint8_t *)inputBuffers_[0] + faceObjectVec_.size() * batchLen;
        APP_ERROR ret = APP_ERR_OK;
        if (runMode_ == ACL_HOST) { // under ACL_HOST mode, memory from host to device
            ret = aclrtMemcpy(dataBufferPtr, batchLen, faceObject->imgQuality.buf.deviceData.get(), batchLen,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
        } else if (runMode_ == ACL_DEVICE) { // under ACL_DEVICE mode, memory from device to device
            std::copy(faceObject->imgQuality.buf.deviceData.get(),
                faceObject->imgQuality.buf.deviceData.get() + batchLen, dataBufferPtr);
            ret = APP_ERR_OK;
        }
        if (ret != APP_ERR_OK) {
            LogError << "QualityEvaluation[" << instanceId_ << "]: Fail to copy memory.";
            return ret;
        }
        faceObjectVec_.push_back(faceObject);
        return APP_ERR_OK;
    }
    return APP_ERR_COMM_FAILURE;
}

APP_ERROR QualityEvaluation::PostData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue)
{
#ifndef ASCEND_ACL_OPEN_VESION
    for (uint32_t i = 0; i < outputBuffersHost_.size(); i++) {
        APP_ERROR ret = aclrtMemcpy(outputBuffersHost_[i].get(), outputSizes_[i], outputBuffers_[i], outputSizes_[i],
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != APP_ERR_OK) {
            LogError << "QualityEvaluation[" << instanceId_ << "]: Fail to copy memory.";
            return ret;
        }
    }
#endif
    for (int i = 0; i < (int)faceObjectQueue.size(); i++) {
        float keyPointScore = CalKeyPointScore(i);
        float eulerScore = CalEulerScore(i);
        float bigFaceScore = CalBigFaceScore(faceObjectQueue[i]);
        // wights in the configuration file, can be adjusted manually
        float score = keyPointWeight_ * keyPointScore + eulerWeight_ * eulerScore + bigFaceWeight_ * bigFaceScore;
        faceObjectQueue[i]->faceQuality.score = score;
        LogDebug << "QualityEvaluation[" << instanceId_ << "]:"
                 << "kpScore:" << keyPointScore << ", eulerScore:" << eulerScore << ", bigFaceScore:" << bigFaceScore;

        // filter
        if (score < minOutputScore_) {
            LogDebug << "QualityEvaluation[" << instanceId_ << "]: score is too low, score=" << score << ", kpScore:" <<
                keyPointScore << ", eulerScore:" << eulerScore << ", bigFaceScore:" << bigFaceScore;
            continue;
        }
        auto &faceMap = FaceBlockingMap::GetInstance(faceObjectQueue[i]->frameInfo.channelId);
        int32_t id = faceObjectQueue[i]->trackInfo.id;
        std::shared_ptr<FaceObject> faceObject = faceMap->Get(id);
        if (faceObject.get() == nullptr) {
            CalKPCoordinate(i, faceObjectQueue[i]);
            faceMap->Insert(id, faceObjectQueue[i]);
        } else {
            if (faceObject->faceQuality.score < faceObjectQueue[i]->faceQuality.score) {
                CalKPCoordinate(i, faceObjectQueue[i]);
                faceMap->Insert(id, faceObjectQueue[i]);
            }
        }
    }
    return APP_ERR_OK;
}

float QualityEvaluation::CalKeyPointScore(int pos)
{
    int batchLen = outputSizes_[1] / batchSize_;
#ifdef ASCEND_ACL_OPEN_VESION
    float *heatmapPtr = (float *)((uint8_t *)outputBuffers_[1] + pos * batchLen);
#else
    float *heatmapPtr = (float *)((uint8_t *)outputBuffersHost_[1].get() + pos * batchLen);
#endif
    float score = 0;
    const float elementScoreLimit = 0.2;
    for (int i = 0; i < heatmapHeight_; i++) {
        float *tmpPtr = heatmapPtr + i * heatmapWidth_;
        float tmpScore = *std::max_element(tmpPtr, tmpPtr + heatmapWidth_);
        score += ((tmpScore > elementScoreLimit) ? elementScoreLimit : tmpScore);
    }
    return score;
}

APP_ERROR QualityEvaluation::CalKPCoordinate(int pos, std::shared_ptr<FaceObject> faceObject)
{
    int batchLen = outputSizes_[1] / batchSize_;
#ifdef ASCEND_ACL_OPEN_VESION
    float *heatmapPtr = (float *)((uint8_t *)outputBuffers_[1] + pos * batchLen);
#else
    float *heatmapPtr = (float *)((uint8_t *)outputBuffersHost_[1].get() + pos * batchLen);
#endif
    const uint8_t doubleH = 2;
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

float QualityEvaluation::CalEulerScore(int pos)
{
    int batchLen = outputSizes_[0] / batchSize_;
#ifdef ASCEND_ACL_OPEN_VESION
    float *euler_ptr = (float *)((uint8_t *)outputBuffers_[0] + pos * batchLen);
#else
    float *euler_ptr = (float *)((uint8_t *)outputBuffersHost_[0].get() + pos * batchLen);
#endif
    const uint32_t degree90 = 90;
    uint16_t indexEulerPtr = 0;
    float yaw = fabs(euler_ptr[indexEulerPtr++]) * degree90;
    float pitch = fabs(euler_ptr[indexEulerPtr++]) * degree90;
    float roll = fabs(euler_ptr[indexEulerPtr++]) * degree90;
    const uint32_t pitchConstant = 6;
    pitch = (pitch > pitchConstant) ? (pitch - pitchConstant) : 0;
    return (degree90 - yaw) / degree90 + (degree90 - pitch) / degree90 + (degree90 - roll) / degree90;
}

float QualityEvaluation::CalBigFaceScore(std::shared_ptr<FaceObject> face)
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

APP_ERROR QualityEvaluation::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "QualityEvaluation[" << instanceId_ << "]: Begin to process data.";
    std::shared_ptr<FaceObject> faceObject = std::static_pointer_cast<FaceObject>(inputData);

    if (faceObject.get() == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }

    APP_ERROR ret;
    using Time = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<float>;
    using Second = std::chrono::duration<double, std::ratio<1, 1>>;
    auto currentTime = Time::now();

    // faceObject is ignored if its status is either lost or new
    // faceObject is ignored if exist time is greater than maxSendTime_
    Duration duration = currentTime - faceObject->trackInfo.createTime;
    double lastTime = std::chrono::duration_cast<Second>(duration).count();
    if (faceObject->trackInfo.flag == NEW_FACE || faceObject->trackInfo.flag == LOST_FACE || lastTime > maxSendTime_) {
        return APP_ERR_OK;
    }
    // get channels set
    channels_.insert(faceObject->frameInfo.channelId);
    if (faceObjectVec_.size() < batchSize_) {
        APP_ERROR ret = PreParaData(faceObject);
        faceObject->imgQuality.buf.deviceData.reset();
        if (ret != APP_ERR_OK) {
            LogError << "QualityEvaluation[" << instanceId_ << "]: PreProcess Error.ret=" << ret;
            return ret;
        }
    }

    if (faceObjectVec_.size() < batchSize_) {
        return APP_ERR_OK;
    }

    // model inference
    ret = modelInfer_->ModelInference(inputBuffers_, inputSizes_, outputBuffers_, outputSizes_);
    if (ret != APP_ERR_OK) {
        faceObjectVec_.clear();
        return ret;
    }
    PostData(faceObjectVec_);
    faceObjectVec_.clear();
    return APP_ERR_OK;
}

void QualityEvaluation::SendData()
{
    std::this_thread::sleep_for(std::chrono::seconds(sleepTime_));
    using Time = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<float>;
    using Second = std::chrono::duration<double, std::ratio<1, 1>>;
    auto currentTime = Time::now();
    for (auto iter = channels_.begin(); iter != channels_.end(); iter++) {
        auto &faceMap = FaceBlockingMap::GetInstance(*iter);
        std::vector<int32_t> keys = faceMap->Keys();
        for (int32_t key : keys) {
            std::shared_ptr<FaceObject> face = faceMap->Get(key);
            if (face.get() == nullptr) {
                continue;
            }
            Duration duration = currentTime - face->trackInfo.createTime;
            double lastTime = std::chrono::duration_cast<Second>(duration).count();
            if (lastTime > maxSendTime_ || face->trackInfo.flag == LOST_FACE) {
                SendToNextModule(MT_WARP_AFFINE, face, face->frameInfo.channelId);
                faceMap->Clear(key);
            }
        }
    }
}

void QualityEvaluation::SendThread()
{
    while (!sendStop_) {
        // send data if exist time is greater than maxSendTime_
        SendData();
        std::this_thread::sleep_for(std::chrono::seconds(sleepTime_));
    }
}
} // namespace ascendFaceRecognition
