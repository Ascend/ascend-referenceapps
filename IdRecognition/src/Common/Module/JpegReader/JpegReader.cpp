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
#include "JpegReader.h"

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#include "Log/Log.h"
#include "Statistic/Statistic.h"
#include "FileManager/FileManager.h"
#include "DvppCommon/DvppCommon.h"

namespace ascendFaceRecognition {
const int FILENAME_LENGTH_7 = 7;
const int FILENAME_LENGTH_4 = 4;
const int MIN_IMAGE_WIDTH = 16;
const int MIN_IMAGE_HEIGHT = 2;

// this value is used to avoid memory waste and potential performance dragging down. shall be updated in the
// specific application.
const int JPEG_READER_SLEEP_TIME = 20;
JpegReader::JpegReader()
{
    withoutInputQueue_ = true;
}

JpegReader::~JpegReader()
{
    if (!isDeInited_) {
        DeInit();
    }
}

APP_ERROR JpegReader::ParseConfig(ConfigParser &configParser)
{
    LogInfo << "JpegReader[" << instanceId_ << "]: begin to parse config values.";
    APP_ERROR ret = APP_ERR_OK;
    std::string itemCfgStr;

    if (pipeMode_ == PIPELINE_MODE_SEARCH) {
        itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".jpegPath");
        ret = configParser.GetStringValue(itemCfgStr, jpegFolderPath_);
        if (ret != APP_ERR_OK) {
            return ret;
        }
    } else if (pipeMode_ == PIPELINE_MODE_REG) {
        itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".regConfigPath");
        ret = configParser.GetStringValue(itemCfgStr, regConfigPath_);
        if (ret != APP_ERR_OK) {
            return ret;
        }
    } else {
        LogFatal << "JpegReader[" << instanceId_ << "]:pipeline mode not supported.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".isDisplay");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, isDisplay_);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    return ret;
}

APP_ERROR JpegReader::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "JpegReader[" << instanceId_ << "]: begin to init jpeg reader instance.";

    // 1. init member variables
    AssignInitArgs(initArgs);

    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "JpegReader[" << instanceId_ << "]: Fail to parse config params, ret=" << ret << "("
                 << GetAppErrCodeInfo(ret) << ").";
        return ret;
    }
    isInited_ = true;
    return ret;
}

APP_ERROR JpegReader::DeInit(void)
{
    LogInfo << "JpegReader[" << instanceId_ << "]: deinit start.";
    isDeInited_ = true;

    isStop_ = true;

    if (processThr_.joinable()) {
        processThr_.join();
    }
    return APP_ERR_OK;
}

APP_ERROR JpegReader::ScanFolder(std::string folderPath)
{
    LogDebug << "JpegReader[" << instanceId_ << "]: ScanFolder.";
    APP_ERROR ret = APP_ERR_OK;
    DIR *dir;
    struct dirent *ent = nullptr;

    if ((dir = opendir(folderPath.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            std::string fileName = ent->d_name;
            if ((pipeMode_ == PIPELINE_MODE_SEARCH) && (fileName.size() > FILENAME_LENGTH_4) &&
                (fileName.substr(fileName.size() - FILENAME_LENGTH_4) == ".jpg")) {
                LogDebug << "JpegReader[" << instanceId_ << "]: find jpeg: " << fileName << ".";
                fileNameSet_.push_back(fileName);
            } else if ((pipeMode_ == PIPELINE_MODE_REG) && (fileName.size() > FILENAME_LENGTH_7) &&
                (fileName.substr(fileName.size() - FILENAME_LENGTH_7) == ".regcfg")) {
                LogDebug << "JpegReader[" << instanceId_ << "]: find file: " << fileName << ".";
                fileNameSet_.push_back(fileName);
            }
        }
        closedir(dir);
    } else {
        LogError << "JpegReader[" << instanceId_ << "]: can't open path: " << folderPath << ".";
        return APP_ERR_COMM_NO_EXIST;
    }

    return ret;
}

APP_ERROR JpegReader::GetImageData(std::string &filePath, std::shared_ptr<StreamRawData> &output)
{
    LogDebug << "JpegReader[" << instanceId_ << "]: GetImageData.";

    std::string realPath;
    if (GetRealPath(filePath, realPath) != APP_ERR_OK) {
        LogError << "JpegReader[" << instanceId_ << "]: can't get canonical path: " << filePath << ".";
        return APP_ERR_COMM_NO_EXIST;
    }
    FILE *pFile = fopen(realPath.c_str(), "r");
    if (pFile == nullptr) {
        LogError << "JpegReader[" << instanceId_ << "]: can't open file: " << filePath << ".";
        return APP_ERR_COMM_NO_EXIST;
    }

    // get image data
    fseek(pFile, 0, SEEK_END);
    long fileSize = ftell(pFile);
    if (fileSize <= 0) {
        LogError << "JpegReader[" << instanceId_ << "]: fail to get file size.";
        fclose(pFile);
        return APP_ERR_COMM_NO_EXIST;
    }
    output->hostData.reset(new uint8_t[fileSize], [](uint8_t *p) {
        LogDebug << "JpegReader free imgBuff addr: " << static_cast<void *>(p);
        delete[] p;
    });
    rewind(pFile);
    int imgDataSize = fread(output->hostData.get(), sizeof(char), fileSize, pFile);
    int components = {};
    DvppCommon::GetJpegImageInfo(output->hostData.get(), fileSize, output->info.width, output->info.height, components);
    LogDebug << "JpegReader[" << instanceId_ << "]: image width= " << output->info.width << ", height= "
             << output->info.height << ", dataSize= " << imgDataSize << " bytes.";
    fclose(pFile);
    output->dataSize = fileSize;
    output->info.channelId = instanceId_;
    output->info.format = format_;
    output->info.isEos = 0;
    output->info.isDisplay = isDisplay_;
    output->info.frameId = frameNum_++;
    output->info.imagePath = filePath;
    output->frameType = 0;

    return APP_ERR_OK;
}

APP_ERROR JpegReader::GetConfigureData(ConfigParser &configParser, std::string &imageFilePath, int &personCount)
{
    APP_ERROR ret = APP_ERR_OK;
    std::string itemCfgStr;

    for (int i = 0; i < personCount; i++) {
        std::shared_ptr<StreamRawData> output = std::make_shared<StreamRawData>();

        itemCfgStr = "Person" + std::to_string(i) + ".ImagePath";
        ret = configParser.GetStringValue(itemCfgStr, imageFilePath);
        if (ret != APP_ERR_OK) {
            continue;
        }
        ret = GetImageData(imageFilePath, output);
        if (ret != APP_ERR_OK) {
            LogError << "JpegReader[" << instanceId_ << "]: fail to get image data.";
            continue;
        }

        itemCfgStr = "Person" + std::to_string(i) + ".Name";
        ret = configParser.GetStringValue(itemCfgStr, output->info.personInfo.name);
        if (ret != APP_ERR_OK) {
            LogError << "JpegReader[" << instanceId_ << "]: fail to get person name.";
            continue;
        }

        itemCfgStr = "Person" + std::to_string(i) + ".Gender";
        ret = configParser.GetStringValue(itemCfgStr, output->info.personInfo.gender);
        if (ret != APP_ERR_OK) {
            LogError << "JpegReader[" << instanceId_ << "]: fail to get person gender.";
            continue;
        }

        itemCfgStr = "Person" + std::to_string(i) + ".Age";
        ret = configParser.GetUnsignedIntValue(itemCfgStr, output->info.personInfo.age);
        if (ret != APP_ERR_OK) {
            LogError << "JpegReader[" << instanceId_ << "]: fail to get person age.";
            continue;
        }

        ret = outputQueVec_[instanceId_]->Push(output, true); // send data to next module under registration mode
        if (ret != APP_ERR_OK) {
            LogError << "JpegReader[" << instanceId_ << "]: fail to get push data.";
        }
    }

    return ret;
}

APP_ERROR JpegReader::GetPersonData(std::vector<std::string>::iterator it)
{
    LogDebug << "JpegReader[" << instanceId_ << "]: GetPersonData.";
    std::string configFilePath;
    std::string imageFilePath;
    std::string itemCfgStr;
    int personCount = {};
    ConfigParser configParser;

    LogDebug << "JpegReader[" << instanceId_ << "]: begin to parse config file: " << (*it) << ".";
    configFilePath = regConfigPath_ + "/" + (*it);
    APP_ERROR ret = configParser.ParseConfig(configFilePath);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = "PersonCount";
    ret = configParser.GetIntValue(itemCfgStr, personCount);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    ret = GetConfigureData(configParser, imageFilePath, personCount);
    if (ret != APP_ERR_OK) {
        LogDebug << "JpegReader[" << instanceId_ << "]: parse preson info failed.";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR JpegReader::Process(std::shared_ptr<void> inputData)
{
    Statistic::GlobalTimeStatisticStart("HostTotalRunTime");
    LogDebug << "JpegReader[" << instanceId_ << "]: process.";
    APP_ERROR ret = APP_ERR_OK;
    std::string jpegFilePath;
    // 1. get file list
    if (pipeMode_ == PIPELINE_MODE_SEARCH) {
        ret = ScanFolder(jpegFolderPath_);
        if (ret != APP_ERR_OK) {
            LogError << "JpegReader[" << instanceId_ << "]: fail to scan folder " << jpegFolderPath_ << ".";
            return ret;
        }
    } else if (pipeMode_ == PIPELINE_MODE_REG) {
        ret = ScanFolder(regConfigPath_);
        if (ret != APP_ERR_OK) {
            LogError << "JpegReader[" << instanceId_ << "]: fail to scan folder " << jpegFolderPath_ << ".";
            return ret;
        }
    }
    // 2. read file
    std::vector<std::string>::iterator it = fileNameSet_.begin();
    while ((it != fileNameSet_.end()) && (!stop_)) {
        jpegReaderStatic_.RunTimeStatisticStart("JpegReader_Excute_Time", instanceId_);
        if (pipeMode_ == PIPELINE_MODE_SEARCH) {
            std::shared_ptr<StreamRawData> output = std::make_shared<StreamRawData>();
            jpegFilePath = jpegFolderPath_ + "/" + (*it);
            ret = GetImageData(jpegFilePath, output);
            if (ret != APP_ERR_OK) {
                LogError << "JpegReader[" << instanceId_ << "]: fail to get image data.";
                it++;
                continue;
            }
            ret = outputQueVec_[instanceId_]->Push(output, true); // send data to next module under search mode
            if (ret != APP_ERR_OK) {
                LogError << "JpegReader[" << instanceId_ << "]: fail to get push data.";
            }
            LogDebug << "JpegReader[" << instanceId_ << "]: output queue " << outputQueVec_[instanceId_] << ".";
        } else if (pipeMode_ == PIPELINE_MODE_REG) {
            // in registration mode, push data is done in the GetPersonData function.
            ret = GetPersonData(it);
            if (ret != APP_ERR_OK) {
                LogError << "JpegReader[" << instanceId_ << "]: fail to get person data.";
            }
        }
        it++;
        jpegReaderStatic_.RunTimeStatisticStop();
        std::this_thread::sleep_for(std::chrono::milliseconds(JPEG_READER_SLEEP_TIME));
    }
    ShowEndMessage();

    return ret;
}

void JpegReader::ShowEndMessage() const
{
    if (pipeMode_ == PIPELINE_MODE_REG) {
        LogInfo << "IdRecognition demo finish registration successfully on channel " << instanceId_;
    }
    if (pipeMode_ == PIPELINE_MODE_SEARCH) {
        LogInfo << "IdRecognition demo finish search successfully on channel " << instanceId_;
    }
}

uint32_t JpegReader::GetFormat() const
{
    return format_;
}

uint32_t JpegReader::GetInstanceId() const
{
    return instanceId_;
}

uint32_t JpegReader::GetIsDisplay() const
{
    return isDisplay_;
}

uint32_t JpegReader::GetFrameNum() const
{
    return frameNum_;
}

void JpegReader::SetFrameNum()
{
    frameNum_++;
}

std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> JpegReader::GetOutputQueVec() const
{
    return outputQueVec_[instanceId_];
}
} // namespace ascendFaceRecognition
