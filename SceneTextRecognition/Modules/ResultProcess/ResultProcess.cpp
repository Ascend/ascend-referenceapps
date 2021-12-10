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

#include "ResultProcess.h"

#include <iostream>

#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
#include "FileManager/FileManager.h"
#include "PointerDeleter/PointerDeleter.h"
#include "CommonDataType/CommonDataType.h"
#include "Common/CommonType.h"
#include "ImageReader/ImageReader.h"

using namespace ascendBaseModule;

bool* ResultProcess::finish_ = nullptr;
FinalResult* ResultProcess::finalResult_ = nullptr;
CallBack ResultProcess::callback_ = nullptr;

ResultProcess::ResultProcess() {}

ResultProcess::~ResultProcess() {}

void ResultProcess::RegisterCallBack(CallBack callback, bool& flag, FinalResult& result)
{
    callback_ = callback;
    finish_ = &flag;
    finalResult_ = &result;
}

APP_ERROR ResultProcess::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    AssignInitArgs(initArgs);
    LogDebug << "ResultProcess[" << instanceId_ << "]: ResultProcess begin to init instance "
             << initArgs.instanceId << ".";
    std::string itemCfgStr = moduleName_ + std::string(".savePath");
    APP_ERROR ret = configParser.GetStringValue(itemCfgStr, savePath_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ResultProcess[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    itemCfgStr = moduleName_ + std::string(".enableCallback");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, enableCallback_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ResultProcess[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    ret = CreateDir(savePath_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ResultProcess[" << instanceId_ << "]: Failed to create result directory " << savePath_ \
                 << ", ret = " << ret << ".";
        return ret;
    }

    LogDebug << "ResultProcess[" << instanceId_ << "]: ResultProcess init successfully.";
    return APP_ERR_OK;
}

APP_ERROR ResultProcess::DeInit(void)
{
    LogDebug << "ResultProcess[" << instanceId_ << "]: ResultProcess begin to deinit.";
    LogDebug << "ResultProcess[" << instanceId_ << "]: ResultProcess deinit successfully.";
    return APP_ERR_OK;
}

APP_ERROR ResultProcess::Process(std::shared_ptr<void> inputData)
{
    resultProcessStatic_.RunTimeStatisticStart("ResultProcess_Execute_Time", instanceId_, true);
    LogDebug << "ResultProcess[" << instanceId_ << "]: ResultProcess process start.";
    std::shared_ptr<SendInfo> sendData = std::static_pointer_cast<SendInfo>(inputData);
    if (sendData->itemNum == 0) {
        imageNum_++;
        resultProcessStatic_.RunTimeStatisticStop();
        Statistic::GlobalTimeStatisticStop();
    } else {
        size_t imgId = sendData->imageId;
        textInfoMap_[imgId].insert(
            std::pair<uint32_t, std::string> { sendData->itemInfo->itemId, sendData->itemInfo->textContent });
        textCoordinatesMap_[imgId].insert(std::pair<uint32_t, std::array<int, TEXT_BOX_COORDINATES_NUM>> {
            sendData->itemInfo->itemId, sendData->itemInfo->orgCoordinate });

        if (textInfoMap_[imgId].size() == sendData->itemNum) {
            WriteRecognizeResult(sendData->imageName, textCoordinatesMap_[imgId], textInfoMap_[imgId]);
            textInfoMap_.erase(imgId);
            textCoordinatesMap_.erase(imgId);
            imageNum_++;
            resultProcessStatic_.RunTimeStatisticStop();
            Statistic::GlobalTimeStatisticStop();
        }
    }

    // When all image is processed, set flag StopFlag::signalRecieved to notify the main program to exit
    if ((ImageReader::GetTotalImageNum() > 0) && (imageNum_ == ImageReader::GetTotalImageNum())) {
        StopFlag::signalRecieved = true;
    }

    LogDebug << "ResultProcess[" << instanceId_ << "]: ResultProcess process end.";
    return APP_ERR_OK;
}

void ResultProcess::WriteRecognizeResult(std::string imageName,
                                         std::map<uint32_t, std::array<int, TEXT_BOX_COORDINATES_NUM>> &boxMap,
                                         std::map<uint32_t, std::string> &textMap)
{
    // Result file name use the time stamp as a suffix
    std::string timeString;
    GetCurTimeString(timeString);

    // Create result file under result directory
    std::string resultFileName = savePath_ + "/" + imageName + '_' + timeString + ".txt";
    std::ofstream tfile(resultFileName);
    // Check result file validity
    if (tfile.fail()) {
        LogWarn << "ResultProcess[" << instanceId_ << "]: [" << imageName \
                << "]: Failed to save recognition result, errno = " << errno << ".";
        return;
    }

    tfile << "Image: " << imageName << std::endl;
    for (size_t i = 0; i < textMap.size(); i++) {
        tfile << "Item" << i << "[";
        for (size_t j = 0; j < TEXT_BOX_COORDINATES_NUM; j += TEXT_BOX_COORDINATES_DIM) {
            tfile << "(" << std::to_string(boxMap[i][j]) << ", " << std::to_string(boxMap[i][j + 1]) << ")";
            if (j < TEXT_BOX_COORDINATES_NUM - TEXT_BOX_COORDINATES_DIM) {
                tfile << ", ";
            }
        }
        tfile << "]: " << textMap[i] << std::endl;
    }
    if (enableCallback_ && callback_) {
        FinalResult temp { boxMap, textMap };
        callback_(temp, *finalResult_, *finish_);
    }
}