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

#include "TextRecognition.h"

#include <iostream>
#include <fstream>

#include "ResultProcess/ResultProcess.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
#include "FileManager/FileManager.h"
#include "PointerDeleter/PointerDeleter.h"
#include "CommonDataType/CommonDataType.h"

using namespace ascendBaseModule;

namespace {
    const int CHECK_FREQUENCY = 5;
    const size_t INPUT_BATCH_INDEX = 0;
    const size_t INPUT_HEIGHT_INDEX = 1;
    const size_t INPUT_WIDTH_INDEX = 2;
}

TextRecognition::TextRecognition() {}

TextRecognition::~TextRecognition() {}

APP_ERROR TextRecognition::ParseConfigCommon(ConfigParser &configParser)
{
    std::string itemCfgStr = std::string("SystemConfig.deviceId");
    APP_ERROR ret = configParser.GetUnsignedIntValue(itemCfgStr, deviceId_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    itemCfgStr = std::string("SystemConfig.debugMode");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, debugMode_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    itemCfgStr = moduleName_ + std::string(".timeoutInterval");
    ret = configParser.GetDoubleValue(itemCfgStr, timeoutInterval_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    return ret;
}

APP_ERROR TextRecognition::ParseConfigModel(ConfigParser &configParser)
{
    std::string itemCfgStr = moduleName_ + std::string(".modelPath");
    APP_ERROR ret = configParser.GetStringValue(itemCfgStr, modelPath_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    itemCfgStr = moduleName_ + std::string(".modelName");
    ret = configParser.GetStringValue(itemCfgStr, modelName_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    itemCfgStr = moduleName_ + std::string(".modelType");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, modelType_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    itemCfgStr = moduleName_ + std::string(".modelHeight");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, modelHeight_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    itemCfgStr = moduleName_ + std::string(".modelWidth");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, modelWidth_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    itemCfgStr = moduleName_ + std::string(".batchSize");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, batchSize_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    itemCfgStr = moduleName_ + std::string(".keysFilePath");
    ret = configParser.GetStringValue(itemCfgStr, keysFilePath_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    return ret;
}

APP_ERROR TextRecognition::ModelProcessInit(void)
{
    modelProcess_.reset(new ModelProcess(deviceId_, modelName_));
    char tempPath[PATH_MAX];
    if (realpath(modelPath_.c_str(), tempPath) == nullptr) {
        LogError << "TextRecognition[" << instanceId_ << "]: Failed to get the real path of " << modelPath_.c_str();
        return APP_ERR_COMM_NO_EXIST;
    }
    APP_ERROR ret = modelProcess_->Init(std::string(tempPath));
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to execute init model, ret = " << ret << ".";
        return ret;
    }
    ret = CheckModelInputInfo();
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to check model input info, ret = " << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR TextRecognition::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    AssignInitArgs(initArgs);
    LogDebug << "TextRecognition[" << instanceId_ << "]: TextDetection begin to init instance " \
             << initArgs.instanceId << ".";
    APP_ERROR ret = ParseConfigCommon(configParser);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    ret = ParseConfigModel(configParser);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    ret = ModelProcessInit();
    if (ret != APP_ERR_OK) {
        return ret;
    }

    // Create dvpp stream and dvpp object for cropping the text
    ret = aclrtCreateStream(&dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to execute aclrtCreateStream, ret=" << ret << ".";
        return ret;
    }

    dvppObjPtr_.reset(new DvppCommon(dvppStream_));
    ret = dvppObjPtr_->Init();
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to create dvpp channel, ret = " << ret << ".";
        return ret;
    }

    char tempPath[PATH_MAX];
    if (realpath(keysFilePath_.c_str(), tempPath) == nullptr) {
        LogError << "Failed to get the real path of " << keysFilePath_.c_str();
        return APP_ERR_COMM_NO_EXIST;
    }
    ret = LoadKeysUTF8File(std::string(tempPath), keysVec_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to load " << tempPath <<", ret = " \
                 << ret << ".";
        return ret;
    }
    CreateThread();

    LogDebug << "TextRecognition[" << instanceId_ << "]: TextRecognition init successfully.";
    return APP_ERR_OK;
}

APP_ERROR TextRecognition::DeInit(void)
{
    LogDebug << "TextRecognition[" << instanceId_ << "]: TextRecognition begin to deinit.";

    APP_ERROR ret = APP_ERR_OK;

    thread_->join();

    if (dvppObjPtr_ != nullptr) {
        ret = dvppObjPtr_->DeInit();
        if (ret != APP_ERR_OK) {
            LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to execute dvpp DeInit, ret = " << ret << ".";
            return ret;
        }
        dvppObjPtr_ = nullptr;
    }

    if (modelProcess_ != nullptr) {
        ret = modelProcess_->DeInit();
        if (ret != APP_ERR_OK) {
            LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to execute model DeInit, ret = " << ret << ".";
            return ret;
        }
        modelProcess_ = nullptr;
    }

    ret = aclrtDestroyStream(dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "TextRecognition[" << instanceId_ << "]: Failed to destroy dvpp stream, ret = " \
                 << ret << ".";
        return ret;
    }
    dvppStream_ = nullptr;

    LogDebug << "TextRecognition[" << instanceId_ << "]: TextRecognition deinit successfully.";
    return ret;
}

void TextRecognition::CreateThread(void)
{
    thread_.reset(new std::thread(&TextRecognition::WatchThread, this));
}

void TextRecognition::WatchThread(void)
{
    APP_ERROR ret = aclrtSetCurrentContext(aclContext_);
    if (ret != APP_ERR_OK) {
        LogError << "TextRecognition[" << instanceId_ << "]: Failed to set current context, ret = " << ret << ".";
    }
    startTime_ = std::chrono::high_resolution_clock::now();
    while (!isStop_) {
        usleep(timeoutInterval_ * SEC2MS / CHECK_FREQUENCY);
        auto endTime = std::chrono::high_resolution_clock::now();
        double costTime = std::chrono::duration<double, std::milli>(endTime - startTime_).count();
        if (costTime > timeoutInterval_) {
            std::unique_lock<std::mutex> lock(mtx_);
            if (sendInfoVec_.size() > 0) {
                BatchModelInference(sendInfoVec_.size());
            }
        }
    }
}

APP_ERROR TextRecognition::BatchModelInference(uint32_t itemNum)
{
    void *ptrBuffer = nullptr;
    size_t singleBatchSize = sendInfoVec_[0]->itemInfo->resizeData.lenOfByte;
    size_t totalDataSize = singleBatchSize * batchSize_;
    APP_ERROR ret = aclrtMalloc(&ptrBuffer, totalDataSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != APP_ERR_OK) {
        LogError << "TextRecognition[" << instanceId_ << "]: Failed to malloc memory, ret = " << ret << ".";
        return ret;
    }
    std::shared_ptr<void> ptrBufferManager(ptrBuffer, aclrtFree);
    ret = PrepareModelInputBuffer(ptrBufferManager, singleBatchSize, itemNum);
    if (ret != APP_ERR_OK) {
        LogError << "TextRecognition[" << instanceId_ << "]: Failed to prepare input buffer, ret = " << ret << ".";
        return ret;
    }
    std::vector<void *> outputBuffers;
    std::vector<size_t> outputSizes;
    std::vector<void *> inputBuffers = { ptrBufferManager.get() };
    std::vector<size_t> inputSizes = { totalDataSize };
    ret = PrepareModelOutputBuffer(outputBuffers, outputSizes);
    if (ret != APP_ERR_OK) {
        LogError << "TextRecognition[" << instanceId_ << "]: Failed to prepare output buffer, ret = " << ret << ".";
        return ret;
    }

    ret = modelProcess_->ModelInference(inputBuffers, inputSizes, outputBuffers, outputSizes);
    if (ret != APP_ERR_OK) {
        LogError << "TextRecognition[" << instanceId_ << "]: Failed to execute ModelInference, ret = " << ret << ".";
        modelProcess_->ReleaseModelBuffer(outputBuffers);
        return ret;
    }

    // PostProcess
    ret = RecognizePostProcess(outputBuffers, outputSizes, itemNum);
    for (size_t i = 0; i < itemNum; i++) {
        SendToNextModule(MT_ResultProcess, sendInfoVec_[i], 0);
    }
    modelProcess_->ReleaseModelBuffer(outputBuffers);
    if (ret != APP_ERR_OK) {
        LogError << "TextRecognition[" << instanceId_ << "]: Failed to release model buffer, ret = " << ret << ".";
        return ret;
    }
    sendInfoVec_.erase(sendInfoVec_.begin(), sendInfoVec_.begin() + sendInfoVec_.size());
    return APP_ERR_OK;
}

APP_ERROR TextRecognition::Process(std::shared_ptr<void> inputData)
{
    std::shared_ptr<SendInfo> sendData = std::static_pointer_cast<SendInfo>(inputData);
    if (sendData->itemNum == 0) {
        SendToNextModule(MT_ResultProcess, sendData, 0);
        return APP_ERR_OK;
    }
    textRecognitionStatic_.RunTimeStatisticStart("TextRecognitionModel_Execute_Time", instanceId_, true);
    LogDebug << "TextRecognition[" << instanceId_ << "]: [" << sendData->imageName << "-" \
             << sendData->itemInfo->itemId << "]: Process start.";
    APP_ERROR ret = Preprocess(sendData);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    if (debugMode_ != 0) {
        SaveResizedImage(sendData);
    }

    std::unique_lock<std::mutex> lock(mtx_);
    sendInfoVec_.push_back(sendData);
    if (sendInfoVec_.size() != batchSize_) {
        return APP_ERR_OK;
    }
    ret = BatchModelInference(batchSize_);
    if (ret != APP_ERR_OK) {
        LogError << "TextRecognition[" << instanceId_ << "]: Failed to execute ModelInference, ret = " << ret << ".";
        return ret;
    }
    startTime_ = std::chrono::high_resolution_clock::now();

    textRecognitionStatic_.RunTimeStatisticStop();
    LogDebug << "TextRecognition[" << instanceId_ << "]: [" << sendData->imageName << "-" \
             << sendData->itemInfo->itemId << "]: Process end.";
    return APP_ERR_OK;
}

APP_ERROR TextRecognition::Preprocess(std::shared_ptr<SendInfo> sendData)
{
    // Prepare the input info for resize
    DvppDataInfo resizeInput;
    resizeInput.width = sendData->itemInfo->perspectiveData.width;
    resizeInput.height = sendData->itemInfo->perspectiveData.height;
    resizeInput.widthStride = DVPP_ALIGN_UP(sendData->itemInfo->perspectiveData.width, VPC_STRIDE_WIDTH);
    resizeInput.heightStride = DVPP_ALIGN_UP(sendData->itemInfo->perspectiveData.height, VPC_STRIDE_HEIGHT);
    resizeInput.dataSize = sendData->itemInfo->perspectiveData.lenOfByte;
    resizeInput.data = sendData->itemInfo->perspectiveData.data.get();

    // Begin to resize the cropped image to adapte the model
    DvppDataInfo resizeOutput;
    resizeOutput.width = modelWidth_;
    resizeOutput.height = modelHeight_;
    resizeOutput.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    APP_ERROR ret = dvppObjPtr_->CombineResizeProcess(resizeInput, resizeOutput, true, VPC_PT_PADDING);
    if (ret != APP_ERR_OK) {
        LogError << "TextRecognition[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                 << sendData->itemInfo->itemId << "]: Failed to process resize, ret = " << ret << ".";
        return ret;
    }

    std::shared_ptr<DvppDataInfo> itemResizeImg = dvppObjPtr_->GetResizedImage();
    sendData->itemInfo->resizeData.data.reset(itemResizeImg->data, acldvppFree);
    sendData->itemInfo->resizeData.lenOfByte = itemResizeImg->dataSize;

    if (debugMode_ != 0) {
        LogDebug << "TextRecognition[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                 << sendData->itemInfo->itemId << "]: Input: " << resizeInput.width << ", " << resizeInput.height;
        LogDebug << "TextRecognition[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                 << sendData->itemInfo->itemId << "]: Output: " << resizeOutput.width << ", " << resizeOutput.height;
    }
    return APP_ERR_OK;
}

APP_ERROR TextRecognition::CheckModelInputInfo(void)
{
    aclmdlIODims inputDims;
    APP_ERROR ret = aclmdlGetInputDims((aclmdlDesc*)modelProcess_->GetModelDesc(), 0, &inputDims);
    if (ret != APP_ERR_OK) {
        LogError << "TextRecognition[" << instanceId_ << "]: Failed to get input dims, ret = " << ret << ".";
        return ret;
    }
    if (batchSize_ != inputDims.dims[INPUT_BATCH_INDEX]) {
        LogError << "TextRecognition[" << instanceId_ << "]: Model batch size is incorrect.";
        return APP_ERR_INFER_FIND_MODEL_DESC_FAIL;
    }
    if (modelHeight_ != inputDims.dims[INPUT_HEIGHT_INDEX]) {
        LogError << "TextRecognition[" << instanceId_ << "]: Model height is incorrect.";
        return APP_ERR_INFER_FIND_MODEL_DESC_FAIL;
    }
    if (modelWidth_ != inputDims.dims[INPUT_WIDTH_INDEX]) {
        LogError << "TextRecognition[" << instanceId_ << "]: Model width is incorrect.";
        return APP_ERR_INFER_FIND_MODEL_DESC_FAIL;
    }
    return APP_ERR_OK;
}

void TextRecognition::SaveResizedImage(std::shared_ptr<SendInfo> sendData)
{
    void *hostPtr = nullptr;
    APP_ERROR ret = aclrtMallocHost(&hostPtr, sendData->itemInfo->resizeData.lenOfByte);
    if (ret != APP_ERR_OK) {
        LogWarn << "TextRecognition[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                << sendData->itemInfo->itemId << "]: Failed to malloc on host, ret = " << ret << ".";
        return;
    }
    std::shared_ptr<void> hostSharedPtr(hostPtr, aclrtFreeHost);
    ret = aclrtMemcpy(hostPtr, sendData->itemInfo->resizeData.lenOfByte, sendData->itemInfo->resizeData.data.get(),
                      sendData->itemInfo->resizeData.lenOfByte, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != APP_ERR_OK) {
        LogWarn << "TextRecognition[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                << sendData->itemInfo->itemId << "]: Failed to memcpy from device to host, ret = " \
                << ret << ".";
        return;
    }
    // Save the resized result
    std::stringstream fileName;
    fileName << "resize_image_" << sendData->imageName << "_" << sendData->itemInfo->itemId;
    SaveFileWithTimeStamp(hostSharedPtr, sendData->itemInfo->resizeData.lenOfByte, moduleName_, fileName.str(), ".yuv");
}

APP_ERROR TextRecognition::PrepareModelInputBuffer(std::shared_ptr<void> &ptrBufferManager, size_t singleBatchSize,
                                                   uint32_t itemNum)
{
    uint32_t pos = 0;
    for (size_t i = 0; i < itemNum; i++) {
        APP_ERROR ret = aclrtMemcpy((uint8_t *)ptrBufferManager.get() + pos, singleBatchSize,
                                    sendInfoVec_[i]->itemInfo->resizeData.data.get(),
                                    singleBatchSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != APP_ERR_OK) {
            LogError << "TextRecognition[" << instanceId_ << "]: Failed to copy memory, ret = " << ret << ".";
            return ret;
        }
        pos += singleBatchSize;
    }
    if (itemNum < batchSize_) {
        size_t resSize = singleBatchSize * (batchSize_ - itemNum);
        APP_ERROR ret = aclrtMemset((uint8_t *)ptrBufferManager.get() + pos, resSize, 0, resSize);
        if (ret != APP_ERR_OK) {
            LogError << "TextRecognition[" << instanceId_ << "]: Failed to set memory, ret = " << ret << ".";
            return ret;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR TextRecognition::PrepareModelOutputBuffer(std::vector<void *> &outputBuffers,
                                                    std::vector<size_t> &outputSizes)
{
    // Get model description
    aclmdlDesc *modelDesc = modelProcess_->GetModelDesc();
    size_t outputSize = aclmdlGetNumOutputs(modelDesc);
    for (size_t i = 0; i < outputSize; i++) {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
        void *outputBuffer = nullptr;
        APP_ERROR ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != APP_ERR_OK) {
            LogError << "TextRecognition[" << instanceId_ << "]: Failed to malloc model output buffer, ret = " << ret;
            // Free the buffer malloced successfully before return error
            modelProcess_->ReleaseModelBuffer(outputBuffers);
            return ret;
        }
        outputBuffers.push_back(outputBuffer);
        outputSizes.push_back(bufferSize);
    }
    return APP_ERR_OK;
}

APP_ERROR TextRecognition::RecognizePostProcess(const std::vector<void *> &outputBuffers,
                                                const std::vector<size_t> &outputSizes, uint32_t itemNum)
{
    const size_t outputLen = outputSizes.size();
    if (outputLen <= 0) {
        LogError << "TextRecognition[" << instanceId_ << "]: Failed to get model output data.";
        return APP_ERR_INFER_GET_OUTPUT_FAIL;
    }

    APP_ERROR ret;
    std::vector<std::shared_ptr<void>> singleResult;
    for (size_t j = 0; j < outputLen; j++) {
        void *hostPtrBuffer = nullptr;
        ret = (APP_ERROR)aclrtMallocHost(&hostPtrBuffer, outputSizes[j]);
        if (ret != APP_ERR_OK) {
            LogError << "TextRecognition[" << instanceId_ << "]: Failed to malloc host memory, ret = " << ret << ".";
            return ret;
        }
        std::shared_ptr<void> hostPtrBufferManager(hostPtrBuffer, aclrtFreeHost);
        ret = (APP_ERROR)aclrtMemcpy(hostPtrBuffer, outputSizes[j], outputBuffers[j], outputSizes[j],
                                     ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != APP_ERR_OK) {
            LogError << "TextRecognition[" << instanceId_ << "]: Failed to memcpy device to host, ret = " << ret << ".";
            return ret;
        }

        singleResult.push_back(hostPtrBufferManager);
    }
    if (modelType_ == CHINESE_OCR) {
        RecognizeOutputChineseOCR(singleResult, outputSizes, itemNum);
        return APP_ERR_OK;
    } else if (modelType_ == CRNN) {
        RecognizeOutputCRNN(singleResult, outputSizes, itemNum);
        return APP_ERR_OK;
    }
    LogError << "TextRecognition[" << instanceId_ << "]: Failed to recognize output, modelType_ is invalid, ret = " \
             << ret << ".";
    return APP_ERR_COMM_INVALID_PARAM;
}

void TextRecognition::PostProcessChineseOCR(const std::vector<std::shared_ptr<void>> featLayerData,
                                            const std::vector<size_t> outputSizes, size_t imageIndex,
                                            uint32_t charactersNum, uint32_t& cnt)
{
    float* res = static_cast<float *>(featLayerData[0].get());
    std::vector<int> maxList;
    std::string recognizeStr = {};
    const uint32_t preOneIndex = 1;
    const uint32_t preTwoIndex = 2;
    for (size_t i = 0; i < charactersNum; i++) {
        int max = 0;
        for (int j = 0; j < keysNum_; j++) {
            if (res[imageIndex * charactersNum * keysNum_ + i * keysNum_ + max] < res[cnt++]) {
                max = j;
            }
        }
        maxList.push_back(max);
    }
    for (size_t k = 0; k < maxList.size(); k++) {
        if ((maxList[k] != keysNum_ - 1) && (!(k > preTwoIndex && maxList[k] == maxList[k - preOneIndex]) \
            || (k > preTwoIndex && maxList[k] == maxList[k - preTwoIndex]))) {
            recognizeStr += keysVec_[maxList[k]];
        }
    }
    sendInfoVec_[imageIndex]->itemInfo->textContent = recognizeStr;
}

void TextRecognition::RecognizeOutputChineseOCR(const std::vector<std::shared_ptr<void>> featLayerData,
                                                const std::vector<size_t> outputSizes, uint32_t itemNum)
{
    uint32_t charactersNum = outputSizes[0] / keysNum_ / batchSize_ / sizeof(float);
    uint32_t cnt = 0;
    for (size_t k = 0; k < itemNum; k++) {
        PostProcessChineseOCR(featLayerData, outputSizes, k, charactersNum, cnt);
    }
}

void TextRecognition::RecognizeOutputCRNN(const std::vector<std::shared_ptr<void>> featLayerData,
                                          const std::vector<size_t> outputSizes, uint32_t itemNum)
{
    long long int* res = static_cast<long long int *>(featLayerData[0].get());
    int charactersNum = outputSizes[0] / batchSize_ / sizeof(long long int);
    long long int placeholderIndex = keysVec_.size();
    long long int preIndex = keysVec_.size();
    long long int curIndex = 0;
    std::string recognizeStr = {};
    for (size_t k = 0; k < itemNum; k++) {
        for (int i = 0; i < charactersNum; i++) {
            curIndex = res[i + charactersNum * k];
            if (curIndex != placeholderIndex && curIndex != preIndex) {
                recognizeStr += keysVec_[curIndex];
            }
            preIndex = curIndex;
        }
        sendInfoVec_[k]->itemInfo->textContent = recognizeStr;
        curIndex = 0;
        recognizeStr = "";
    }
}

APP_ERROR TextRecognition::LoadKeysUTF8File(const std::string& fileName, std::vector<std::string>& keysVector)
{
    std::ifstream ifile(fileName, std::ios::binary);
    if (!ifile) {
        return APP_ERR_COMM_OPEN_FAIL;
    }
    unsigned char key;
    while (ifile >> key) {
        std::string strTmp(1, key);
        unsigned char mask = 0x80;
        // Calculate the bytes number of utf-8 character
        int bytesNum = 0;
        while (mask & key) {
            bytesNum++;
            mask >>= 1;
        }
        if (bytesNum > 0) {
            bytesNum--;
        }
        while (bytesNum--) {
            ifile >> key;
            strTmp += key;
        }
        keysVector.push_back(strTmp);
        keysNum_++;
    }
    keysVector.push_back("");
    keysNum_++;
    return APP_ERR_OK;
}
