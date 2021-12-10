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

#include "FaceAttribute.h"

#include <iostream>
#include <string>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "ConfigParser/ConfigParser.h"
#include "PointerDeleter/PointerDeleter.h"

#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "TestCV/TestCV.h"

namespace ascendFaceRecognition {
namespace {
// age stage, represent interval of age
const int AGE_STAGE_0 = 0;
const int AGE_STAGE_1 = 1;
const int AGE_STAGE_2 = 2;
const int AGE_STAGE_3 = 3;
const int AGE_STAGE_4 = 4;
const int AGE_STAGE_5 = 5;
const int AGE_STAGE_6 = 6;
const int AGE_STAGE_7 = 7;
const int AGE_STAGE_8 = 8;

// specific age value
const int AGE_0 = 0;
const int AGE_3 = 3;
const int AGE_10 = 10;
const int AGE_20 = 20;
const int AGE_30 = 30;
const int AGE_40 = 40;
const int AGE_50 = 50;
const int AGE_60 = 60;
const int AGE_70 = 70;

const int INDEX_TYPE_0 = 0;
const int INDEX_TYPE_1 = 1;
}

FaceAttribute::FaceAttribute()
{
    isStop_ = true;
    instanceId_ = -1;
}

FaceAttribute::~FaceAttribute() {}

APP_ERROR FaceAttribute::ParseConfig(ConfigParser &configParser)
{
    // load and parse config file
    LogDebug << "FaceAttribute[" << instanceId_ << "]: start to parse config.";
    std::string itemCfgStr;

    itemCfgStr = moduleName_ + std::string(".batch_size");
    APP_ERROR ret = configParser.GetIntValue(itemCfgStr, batchSize_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceAttribute[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".width");
    ret = configParser.GetIntValue(itemCfgStr, width_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceAttribute[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    alignedWidth_ = ALIGN_UP(width_, FACE_ATTRIBUTE_WIDTH_ALIGN);

    itemCfgStr = moduleName_ + std::string(".height");
    ret = configParser.GetIntValue(itemCfgStr, height_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceAttribute[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    alignedHeight_ = ALIGN_UP(height_, FACE_ATTRIBUTE_HEIGHT_ALIGN);

    itemCfgStr = moduleName_ + std::string(".input_channel");
    ret = configParser.GetIntValue(itemCfgStr, inputChn_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceAttribute[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".model_path");
    ret = configParser.GetStringValue(itemCfgStr, modelPath_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceAttribute[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    return ret;
}

APP_ERROR FaceAttribute::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "FaceAttribute[" << initArgs.instanceId << "]: start to init.";
    // initialize member variables
    AssignInitArgs(initArgs);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceAttribute[" << instanceId_ << "]: Fail to parse config params, ret=" << ret << "(" <<
            GetAppErrCodeInfo(ret) << ").";
        return ret;
    }

    isStop_ = false;
    modelInfer_ = ModelResource::GetInstance().GetModelProcess(modelPath_);
    if (modelInfer_ == nullptr) {
        LogFatal << "QualityEvaluation[" << instanceId_ << "]::init model failed";
        return APP_ERR_COMM_FAILURE;
    }

    size_t inputNum = modelInfer_->GetModelNumInputs();
    LogDebug << "FaceAttribute[" << instanceId_ << "]: inputNum =" << inputNum;
    for (size_t i = 0; i < inputNum; i++) {
        void *buffer = nullptr;
        // modify size
        size_t size = modelInfer_->GetModelInputSizeByIndex(i);
        ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceAttribute[" << instanceId_ << "]: aclrtMalloc fail(ret=" << ret << "), buffer=" <<
                buffer << ", size=" << size;
            return ret;
        }

        inputBuffers_.push_back(buffer);
        inputSizes_.push_back(size);
        LogDebug << "FaceAttribute[" << instanceId_ << "]: i=" << i << ", size=" << size;
    }

    size_t outputNum = modelInfer_->GetModelNumOutputs();
    for (size_t i = 0; i < outputNum; i++) {
        void *buffer = nullptr;
        size_t size = modelInfer_->GetModelOutputSizeByIndex(i);
        ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceAttribute[" << instanceId_ << "]: output aclrtMalloc fail.ret=" << ret;
            return ret;
        }

        outputBuffers_.push_back(buffer);
        outputSizes_.push_back(size);
        LogDebug << "FaceAttribute[" << instanceId_ << "]: i = " << i << ", size=" << size;

#ifndef ASCEND_ACL_OPEN_VESION
        auto bufferHost = std::make_shared<uint8_t>();
        bufferHost.reset(new uint8_t[size], std::default_delete<uint8_t[]>());
        outputBuffersHost_.push_back(bufferHost);
#endif
    }

    return ret;
}

APP_ERROR FaceAttribute::DeInit(void)
{
    LogDebug << "FaceAttribute[" << instanceId_ << "]: deinit start";
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

APP_ERROR FaceAttribute::PreParaData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue)
{
    LogDebug << "FaceAttribute[" << instanceId_ << "]: prepare model inputs.";
    int8_t *dataBufferPtr = (int8_t *)inputBuffers_[0];

    APP_ERROR ret = 0;
    int batchLen = inputSizes_[0] / batchSize_;
    for (uint32_t i = 0; i < faceObjectQueue.size(); i++) {
        if (runMode_ == ACL_HOST) {
            // under ACL_HOST mode, memory from host to device
            ret = aclrtMemcpy(dataBufferPtr, batchLen, faceObjectQueue[i]->imgAffine.buf.deviceData.get(), batchLen,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
        } else if (runMode_ == ACL_DEVICE) { // under ACL_DEVICE mode, memory from device to device
            uint8_t *srcData = faceObjectQueue[i]->imgAffine.buf.deviceData.get();
            std::copy(srcData, srcData + batchLen, dataBufferPtr);
            ret = APP_ERR_OK;
        }
        if (ret != APP_ERR_OK) {
            LogError << "FaceAttribute[" << instanceId_ << "]: Fail to copy memory from host to device.";
            return ret;
        }
        dataBufferPtr += batchLen;
    }

    // pad to kBatchSize
    if ((int)faceObjectQueue.size() < batchSize_) {
        int padNum = batchSize_ - faceObjectQueue.size();
        aclrtMemset(dataBufferPtr, batchLen * padNum, static_cast<char>(0), batchLen * padNum);
    }

    return APP_ERR_OK;
}

APP_ERROR FaceAttribute::PostData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue)
{
    LogDebug << "FaceAttribute[" << instanceId_ << "]: post process.";
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
    for (size_t i = 0; i < faceObjectQueue.size(); i++) {
        int age = GetAge(i);
        int gender = GetGender(i);
        int mask = IsMask(i);
        faceObjectQueue[i]->personInfo.age = age;
        faceObjectQueue[i]->personInfo.gender = (gender == 0) ? "male" : "female";
        faceObjectQueue[i]->personInfo.mask = (mask == 0) ? "mask" : "normal";
    }
    return APP_ERR_OK;
}

APP_ERROR FaceAttribute::PreFilter(std::shared_ptr<FaceObject> faceObject,
    std::vector<std::shared_ptr<FaceObject>> &faceObjectQueue)
{
    float width = faceObject->info.width;
    float height = faceObject->info.height;

    // filter by area
    if (width * height >= minOutputArea_) {
        faceObjectQueue.push_back(faceObject);
    }

    return APP_ERR_OK;
}

APP_ERROR FaceAttribute::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "FaceAttribute[" << instanceId_ << "]: Begin to process data.";
    std::shared_ptr<FaceObject> faceObject = std::static_pointer_cast<FaceObject>(inputData);

    if (faceObject.get() == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }
    // filter data
    APP_ERROR ret = PreFilter(faceObject, faceObjectVec_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceAttribute[" << instanceId_ << "]: pre-filter data error!";
        return ret;
    }
    LogDebug << "FaceAttribute[" << instanceId_ << "]: face size is " << faceObjectVec_.size();

    if (faceObjectVec_.size() < (uint32_t)batchSize_)
        return APP_ERR_OK;

    // data preprocess
    ret = PreParaData(faceObjectVec_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceAttribute[" << instanceId_ << "]: [FaceAttributeEngine] prepare data failed!";
        faceObjectVec_.clear();
        return ret;
    }
    // model inference
    ret = modelInfer_->ModelInference(inputBuffers_, inputSizes_, outputBuffers_, outputSizes_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceAttribute[" << instanceId_ << "]: [FaceAttributeEngine] infer error!";
        faceObjectVec_.clear();
        return ret;
    }
    // data postprocess
    ret = PostData(faceObjectVec_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceAttribute[" << instanceId_ << "]: [FaceAttributeEngine] postprocess data failed!";
        faceObjectVec_.clear();
        return ret;
    }
    for (auto faceObject : faceObjectVec_) {
        SendToNextModule(MT_FACE_FEATURE, faceObject, faceObject->frameInfo.channelId);
    }
    faceObjectVec_.clear();
    return APP_ERR_OK;
}

int FaceAttribute::GetAge(const int &batchIdx)
{
    const int ageOffset = 9;
    int batchLen = outputSizes_[0] / batchSize_;
#ifdef ASCEND_ACL_OPEN_VESION
    float *batchAddress = (float *)((uint8_t *)outputBuffers_[0] + batchIdx * batchLen);
#else
    float *batchAddress = (float *)((uint8_t *)outputBuffersHost_[0].get() + batchIdx * batchLen);
#endif
    int index = std::max_element(batchAddress, batchAddress + ageOffset) - batchAddress;
    index = index % ageOffset;
    int age = 0;
    switch (index) {
        case AGE_STAGE_0:
            age = AGE_0;
            break;
        case AGE_STAGE_1:
            age = AGE_3;
            break;
        case AGE_STAGE_2:
            age = AGE_10;
            break;
        case AGE_STAGE_3:
            age = AGE_20;
            break;
        case AGE_STAGE_4:
            age = AGE_30;
            break;
        case AGE_STAGE_5:
            age = AGE_40;
            break;
        case AGE_STAGE_6:
            age = AGE_50;
            break;
        case AGE_STAGE_7:
            age = AGE_60;
            break;
        case AGE_STAGE_8:
            age = AGE_70;
            break;
        default:
            age = AGE_0;
            break;
    }
    return age;
}

int FaceAttribute::GetGender(const int &batchIdx)
{
    const int addrOffset = 2;
    int batchLen = outputSizes_[1] / batchSize_;
#ifdef ASCEND_ACL_OPEN_VESION
    float *batchAddress = (float *)((uint8_t *)outputBuffers_[1] + batchIdx * batchLen);
#else
    float *batchAddress = (float *)((uint8_t *)outputBuffersHost_[1].get() + batchIdx * batchLen);
#endif
    int index = std::max_element(batchAddress, batchAddress + addrOffset) - batchAddress;
    index = index % addrOffset;
    int gender = 0;
    switch (index) {
        case INDEX_TYPE_0:
            gender = 0;
            break;
        case INDEX_TYPE_1:
            gender = 1;
            break;
        default:
            gender = 0;
            break;
    }
    return gender;
}

int FaceAttribute::IsMask(const int &batchIdx)
{
    const int offset = 2;
    const int addrOffset = 2;
    int batchLen = outputSizes_[offset] / batchSize_;
#ifdef ASCEND_ACL_OPEN_VESION
    float *batchAddress = (float *)((uint8_t *)outputBuffers_[offset] + batchIdx * batchLen);
#else
    float *batchAddress = (float *)((uint8_t *)outputBuffersHost_[offset].get() + batchIdx * batchLen);
#endif
    int index = std::max_element(batchAddress, batchAddress + addrOffset) - batchAddress;
    index = index % offset;
    int isMask = 0;
    switch (index) {
        case INDEX_TYPE_0:
            isMask = 0;
            break;
        case INDEX_TYPE_1:
            isMask = 1;
            break;
        default:
            isMask = 0;
            break;
    }
    return isMask;
}
} // namespace ascendFaceRecognition
