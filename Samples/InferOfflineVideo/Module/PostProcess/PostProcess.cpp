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

#include "PostProcess.h"
#include <sstream>
#include <atomic>
#include "Singleton.h"
#include "FileManager/FileManager.h"

using namespace ascendBaseModule;

namespace {
const int YOLOV3_CAFFE = 0;
const int YOLOV3_TF = 1;
const int BUFFER_SIZE = 5;
const int FILE_SIZE = 52428800; // 50M
}

PostProcess::PostProcess()
{
    isStop_ = false;
}

PostProcess::~PostProcess() {}

APP_ERROR PostProcess::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "Begin to init instance " << initArgs.instanceId;

    AssignInitArgs(initArgs);

    for (int i = 0; i < BUFFER_SIZE; ++i) {
        std::vector<void *> temp;
        for (size_t j = 0; j < ModelBufferSize::outputSize_; j++) {
            void *hostPtrBuffer = nullptr;
            APP_ERROR ret = (APP_ERROR)aclrtMallocHost(&hostPtrBuffer, ModelBufferSize::bufferSize_[j]);
            if (ret != APP_ERR_OK) {
                LogError << "Failed to malloc output buffer of model on host, ret = " << ret;
                return ret;
            }
            temp.push_back(hostPtrBuffer);
        }
        buffers_.push(temp);
    }

    if (access(resultPathName_.c_str(), 0) != 0) {
        APP_ERROR ret = mkdir(resultPathName_.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to create result directory: " << resultPathName_ << ", ret = " << ret;
            return ret;
        }
    }

    resultName_ = resultPathName_ + "/result_" + std::to_string(initArgs.instanceId) + ".txt";
    resultBakName_ = resultPathName_ + "/result_" + std::to_string(initArgs.instanceId) + ".bak";
    return APP_ERR_OK;
}

void PostProcess::ConstructData(const std::vector<ObjDetectInfo> &objInfos,
    const std::shared_ptr<DeviceStreamData> &dataToSend)  const
{
    for (size_t k = 0; k < objInfos.size(); ++k) {
        ObjectDetectInfo detectInfo;
        detectInfo.location.leftTopX = objInfos[k].leftTopX;
        detectInfo.location.leftTopY = objInfos[k].leftTopY;
        detectInfo.location.rightBottomX = objInfos[k].rightBotX;
        detectInfo.location.rightBottomY = objInfos[k].rightBotY;
        detectInfo.confidence = objInfos[k].confidence;
        detectInfo.classId = objInfos[k].classId;
        dataToSend->detectResult.push_back(detectInfo);
    }
}

APP_ERROR PostProcess::WriteResult(const std::vector<ObjDetectInfo> &objInfos, uint32_t channelId, uint32_t frameId)
    const
{
    std::string timeString;
    GetCurTimeString(timeString);
    // Create result file under result directory
    SetFileDefaultUmask();
    std::ofstream tfile(resultName_, std::ios::app);
    // Check result file validity
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << resultName_;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    tfile.seekp(0, tfile.end);
    size_t dstFileSize = tfile.tellp();
    tfile.close();
    if (dstFileSize > FILE_SIZE) {
        if (access(resultBakName_.c_str(), 0) == APP_ERR_OK) {
            APP_ERROR ret = remove(resultBakName_.c_str());
            if (ret != APP_ERR_OK) {
                LogError << "remove " << resultBakName_ << " failed." << std::endl;
                return ret;
            }
        }
        APP_ERROR ret = rename(resultName_.c_str(), resultBakName_.c_str());
        if (ret != APP_ERR_OK) {
            LogError << "rename " << resultName_ << " failed." << std::endl;
            return ret;
        }
    }
    tfile.open(resultName_, std::ios::app);
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << resultName_;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    tfile << "[Date:" << timeString << " Channel:" << channelId << " Frame:" << frameId
          << "] Object detected number is " << objInfos.size() << std::endl;
    // Write inference result into file
    for (uint32_t i = 0; i < objInfos.size(); i++) {
        tfile << "#Obj" << i << ", " << "box(" << objInfos[i].leftTopX << ", " << objInfos[i].leftTopY << ", "
              << objInfos[i].rightBotX << ", " << objInfos[i].rightBotY << ") "
              << " confidence: " << objInfos[i].confidence << "  lable: " << objInfos[i].classId << std::endl;
    }

    tfile << std::endl;
    tfile.close();
    return APP_ERR_OK;
}

APP_ERROR PostProcess::YoloPostProcess(std::vector<RawData> &modelOutput, std::shared_ptr<DeviceStreamData> &dataToSend,
    uint32_t modelType)
{
    const size_t outputLen = modelOutput.size();
    if (outputLen <= 0) {
        LogError << "Failed to get model output data";
        return APP_ERR_INFER_GET_OUTPUT_FAIL;
    }

    std::vector<ObjDetectInfo> objInfos;
    APP_ERROR ret;
    if (modelType == YOLOV3_CAFFE) {
        ret = GetObjectInfoCaffe(modelOutput, objInfos);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to get Caffe model output, ret = " << ret;
            return ret;
        }
    } else {
        ret = GetObjectInfoTensorflow(modelOutput, objInfos);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to get Caffe model output, ret = " << ret;
            return ret;
        }
    }

    ConstructData(objInfos, dataToSend);
    // Write object info to result file
    ret = WriteResult(objInfos, dataToSend->channelId, dataToSend->framId);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to write result, ret = " << ret;
    }
    return APP_ERR_OK;
}

APP_ERROR PostProcess::GetObjectInfoCaffe(std::vector<RawData> &modelOutput, std::vector<ObjDetectInfo> &objInfos)
{
    std::vector<std::shared_ptr<void>> hostPtr;
    std::vector<void *> buffer = buffers_.front();
    buffers_.pop();
    buffers_.push(buffer);
    for (size_t j = 0; j < modelOutput.size(); j++) {
        void *hostPtrBuffer = buffer[j];
        std::shared_ptr<void> hostPtrBufferManager(hostPtrBuffer, [](void *) {});
        APP_ERROR ret = (APP_ERROR)aclrtMemcpy(hostPtrBuffer, modelOutput[j].lenOfByte, modelOutput[j].data.get(),
            modelOutput[j].lenOfByte, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to copy output buffer of model from device to host, ret = " << ret;
            return ret;
        }
        hostPtr.push_back(hostPtrBufferManager);
    }
    uint32_t objNum = ((uint32_t *)(hostPtr[1].get()))[0];
    for (uint32_t k = 0; k < objNum; k++) {
        int pos = 0;
        ObjDetectInfo objInfo = {};
        objInfo.leftTopX = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfo.leftTopY = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfo.rightBotX = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfo.rightBotY = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfo.confidence = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfo.classId = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfos.push_back(objInfo);
    }
    return APP_ERR_OK;
}

APP_ERROR PostProcess::GetObjectInfoTensorflow(std::vector<RawData> &modelOutput, std::vector<ObjDetectInfo> &objInfos)
{
    std::vector<std::shared_ptr<void>> hostPtr;
    std::vector<void *> buffer = buffers_.front();
    buffers_.pop();
    buffers_.push(buffer);
    for (size_t j = 0; j < modelOutput.size(); j++) {
        void *hostPtrBuffer = buffer[j];
        std::shared_ptr<void> hostPtrBufferManager(hostPtrBuffer, [](void *) {});
        APP_ERROR ret = (APP_ERROR)aclrtMemcpy(hostPtrBuffer, modelOutput[j].lenOfByte, modelOutput[j].data.get(),
            modelOutput[j].lenOfByte, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to copy output buffer of model from device to host, ret = " << ret;
            return ret;
        }
        hostPtr.push_back(hostPtrBufferManager);
    }
    Yolov3DetectionOutput(hostPtr, objInfos, yoloImageInfo_);
    return APP_ERR_OK;
}

APP_ERROR PostProcess::Process(std::shared_ptr<void> inputData)
{
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(inputData);
    if (data->eof) {
        Singleton::GetInstance().GetStopedStreamNum()++;
        if (Singleton::GetInstance().GetStopedStreamNum() == Singleton::GetInstance().GetStreamPullerNum()) {
            Singleton::signalRecieved = true;
        }
        return APP_ERR_OK;
    }

    std::shared_ptr<DeviceStreamData> detectInfo = std::make_shared<DeviceStreamData>();
    detectInfo->framId = data->frameId;
    detectInfo->channelId = data->channelId;

    APP_ERROR ret = YoloPostProcess(data->inferOutput, detectInfo, data->modelType);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to run YoloPostProcess, ret = " << ret;
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR PostProcess::DeInit(void)
{
    while (!buffers_.empty()) {
        std::vector<void *> buffer = buffers_.front();
        buffers_.pop();
        for (auto& j : buffer) {
            aclrtFreeHost(j);
        }
    }
    return APP_ERR_OK;
}
