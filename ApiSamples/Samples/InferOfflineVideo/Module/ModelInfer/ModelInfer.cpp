/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ModelInfer.h"
#include "PostProcess/PostProcess.h"
#include "Singleton.h"

using namespace ascendBaseModule;

size_t ModelBufferSize::outputSize_ = {};
std::vector<size_t> ModelBufferSize::bufferSize_ = {};
namespace {
    const int YOLOV3_CAFFE = 0;
    const int YOLOV3_TF = 1;
    const int BUFFER_SZIE = 5;
}

ModelInfer::ModelInfer()
{
    isStop_ = false;
}

ModelInfer::~ModelInfer() {}

APP_ERROR ModelInfer::ParseConfig(ConfigParser &configParser)
{
    std::string itemCfgStr = moduleName_ + std::string(".modelWidth");
    APP_ERROR ret = configParser.GetUnsignedIntValue(itemCfgStr, modelWidth_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInfer[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".modelHeight");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, modelHeight_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInfer[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".modelType");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, modelType_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInfer[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".modelName");
    ret = configParser.GetStringValue(itemCfgStr, modelName_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInfer[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".modelPath");
    ret = configParser.GetStringValue(itemCfgStr, modelPath_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInfer[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = std::string("SystemConfig.deviceId");
    ret = configParser.GetIntValue(itemCfgStr, deviceId_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInfer[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    return ret;
}

APP_ERROR ModelInfer::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "Begin to init instance " << initArgs.instanceId;
    AssignInitArgs(initArgs);

    int ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInfer[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret) << ".";
        return ret;
    }

    modelProcess_.reset(new ModelProcess(deviceId_, modelName_));
    LogDebug << "modelPath_ = "<< modelPath_;
    ret = modelProcess_->Init(modelPath_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInfer[" << instanceId_ << "]: Fail to init ModelProcess." << GetAppErrCodeInfo(ret) << ".";
        return ret;
    }

    aclmdlDesc *modelDesc = modelProcess_->GetModelDesc();
    size_t outputSize = aclmdlGetNumOutputs(modelDesc);
    ModelBufferSize::outputSize_ = outputSize;
    for (size_t i = 0; i < outputSize; i++) {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
        ModelBufferSize::bufferSize_.push_back(bufferSize);
    }

    for (size_t i = 0; i < BUFFER_SZIE; ++i) {
        std::vector<void *> temp;
        for (size_t j = 0; j < outputSize; ++j) {
            void *outputBuffer = nullptr;
            APP_ERROR ret = aclrtMalloc(&outputBuffer, ModelBufferSize::bufferSize_[j], ACL_MEM_MALLOC_NORMAL_ONLY);
            if (ret != APP_ERR_OK) {
                LogError << "Failed to malloc buffer, size is " << ModelBufferSize::bufferSize_[j];
                return ret;
            }
            temp.push_back(outputBuffer);
        }
        buffers_.push(temp);
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInfer::Process(std::shared_ptr<void> commonData)
{
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
    if (data->eof) {
        SendToNextModule(MT_PostProcess, data, data->channelId);
        return APP_ERR_OK;
    }
    srcImageWidth_ = data->srcWidth;
    srcImageHeight_ = data->srcHeight;
    std::vector<RawData> modelOutput;

    APP_ERROR ret = YoloProcess(data->dvppData, modelOutput);
    if (ret != APP_ERR_OK) {
        acldvppFree(data->dvppData->data);
        LogError << "Failed to YoloProcess, ret=" << ret;
        return ret;
    }
    acldvppFree(data->dvppData->data);
    data->inferOutput = std::move(modelOutput);
    data->modelWidth = modelWidth_;
    data->modelHeight = modelHeight_;
    data->modelType = modelType_;
    SendToNextModule(MT_PostProcess, data, data->channelId);
    return APP_ERR_OK;
}

APP_ERROR ModelInfer::DeInit(void)
{
    LogInfo << "ModelInfer[" << instanceId_ << "]: ModelInfer::begin to deinit.";

    // Release objects resource
    modelProcess_->DeInit();

    while (!buffers_.empty()) {
        std::vector<void *> buffer = buffers_.front();
        buffers_.pop();
        for (auto& j : buffer) {
            aclrtFree(j);
        }
    }
    LogInfo << "ModelInfer[" << instanceId_ << "]: ModelInfer deinit success.";
    return APP_ERR_OK;
}

/*
 * @description: YoloV3 inference process
 * @param vpcData The input data
 * @param modelOutput Get output data from inference
 */
APP_ERROR ModelInfer::YoloProcess(std::shared_ptr<DvppDataInfo> &vpcData, std::vector<RawData> &modelOutput)
{
    std::vector<void *> inputDataBuffers;
    std::vector<size_t> buffersSize;
    std::shared_ptr<void> yoloInfo;

    APP_ERROR ret = InputBuffMalloc(vpcData, inputDataBuffers, buffersSize, yoloInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to execute InputBuffMalloc, ret = " << ret;
        return ret;
    }

    std::vector<void *> outBuf = buffers_.front();
    buffers_.pop();
    buffers_.push(outBuf);
    std::vector<size_t> outSizes = ModelBufferSize::bufferSize_;

    ret = modelProcess_->ModelInference(inputDataBuffers, buffersSize, outBuf, outSizes);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to execute ModelInference, ret = " << ret;
        return ret;
    }
    for (size_t i = 0; i < outBuf.size(); i++) {
        RawData rawDevData = RawData();
        rawDevData.data.reset(outBuf[i], [](void*) {});
        rawDevData.lenOfByte = outSizes[i];
        modelOutput.push_back(std::move(rawDevData));
    }

    return APP_ERR_OK;
}

/*
 * @description: Apply memory for model inference input Data
 * @param: vpcData The resize result to inference
 * @param: inputDataBuffers Input buffer
 * @param: buffersSize Input buffer sizes
 * @param yoloInfo The second input of the model, the memory needs to be released
 */
APP_ERROR ModelInfer::InputBuffMalloc(std::shared_ptr<DvppDataInfo> &vpcData, std::vector<void *> &inputDataBuffers,
    std::vector<size_t> &buffersSize, std::shared_ptr<void>& yoloInfo)
{
    inputDataBuffers.push_back(vpcData->data);
    buffersSize.push_back(vpcData->dataSize);
    if (modelType_ == YOLOV3_CAFFE) {
        float imgInfo[IMAGE_INFO_ARRAY_SIZE] = {0};
        imgInfo[MODEL_HEIGHT_INDEX] = modelHeight_;
        imgInfo[MODEL_WIDTH_INDEX] = modelWidth_;
        imgInfo[IMAGE_HEIGHT_INDEX] = srcImageHeight_;
        imgInfo[IMAGE_WIDTH_INDEX] = srcImageWidth_;
        void *yoloImgInfo = nullptr;
        uint32_t imgInfoInputSize = sizeof(float) * IMAGE_INFO_ARRAY_SIZE;
        APP_ERROR ret = acldvppMalloc(&yoloImgInfo, imgInfoInputSize);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to malloc buffer, size is " << imgInfoInputSize << ", ret = " << ret;
            return ret;
        }
        yoloInfo.reset(yoloImgInfo, acldvppFree);
        ret = aclrtMemcpy(yoloImgInfo, imgInfoInputSize, imgInfo, imgInfoInputSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to execute aclrtMemcpy, ret = " << ret;
            return ret;
        }
        inputDataBuffers.push_back(yoloImgInfo);
        buffersSize.push_back(sizeof(float) * IMAGE_INFO_ARRAY_SIZE);
    }
    return APP_ERR_OK;
}
