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

#include "ImageReader.h"

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#include "Log/Log.h"
#include "RegistApi/RegistApi.h"
#include "FileManager/FileManager.h"
#include "Common.h"
#ifdef ASCEND_ACL_OPEN_VESION
#include "HdcChannel/HdcChannel.h"
#endif

static std::vector<std::string> Split(const std::string &src, const std::string &pattern)
{
    std::vector<std::string> res;
    if (src == "")
        return res;
    std::string strs = src + pattern;
    size_t pos = strs.find(pattern);

    while (pos != strs.npos) {
        std::string temp = strs.substr(0, pos);
        if (temp != "")
            res.push_back(temp);
        strs = strs.substr(pos + 1, strs.size());
        pos = strs.find(pattern);
    }
    return res;
}

namespace ascendFaceRecognition {
ImageReader::ImageReader()
{
    withoutInputQueue_ = true;
}

APP_ERROR ImageReader::ParseConfig(ConfigParser &configParser)
{
    // load and parse config file
    LogDebug << "ImageReader[" << instanceId_ << "]: begin to parse config values.";
    std::string itemCfgStr;

    itemCfgStr = moduleName_ + std::string(".jpegPath");
    APP_ERROR ret = configParser.GetStringValue(itemCfgStr, jpegFolderPath_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    return ret;
}

APP_ERROR ImageReader::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "ImageReader[" << instanceId_ << "]: begin to init jpeg reader instance.";

    // 1. init member variables
    AssignInitArgs(initArgs);

    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageReader[" << instanceId_ << "]: Fail to parse config params, ret=" << ret << "(" <<
            GetAppErrCodeInfo(ret) << ").";
        return ret;
    }
    return ret;
}

APP_ERROR ImageReader::DeInit(void)
{
    return APP_ERR_OK;
}

APP_ERROR ImageReader::ScanFolder(std::string folderPath)
{
    LogDebug << "ImageReader[" << instanceId_ << "]: ScanFolder.";
    DIR *dir;
    struct dirent *ent = nullptr;
    const int filePrefixSize = 4;
    // open dir
    if ((dir = opendir(folderPath.c_str())) == nullptr) {
        return APP_ERR_COMM_NO_EXIST;
    }
    // read dir
    while ((ent = readdir(dir)) != nullptr) {
        std::string fileName = ent->d_name;
        if (fileName.size() > filePrefixSize && fileName.substr(fileName.size() - filePrefixSize) == ".jpg") {
            LogInfo << "ImageReader[" << instanceId_ << "]: find jpeg: " << fileName.c_str() << ".";
            fileNameSet_.push_back(fileName);
        }
    }
    closedir(dir);
    return APP_ERR_OK;
}

APP_ERROR ImageReader::GetImageData(std::string &filePath, std::string uuid)
{
    LogDebug << "ImageReader[" << instanceId_ << "]: GetImageData.";
    uint64_t fileSize;
    void *buff = nullptr;
    int numRead;

    std::string resolvedPath;
    APP_ERROR ret = GetRealPath(filePath, resolvedPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    // open image file
    FILE *pFile = fopen(resolvedPath.c_str(), "r");
    if (pFile == nullptr) {
        LogError << "ImageReader[" << instanceId_ << "]: can't open file: " << filePath.c_str() << ".";
        return APP_ERR_COMM_NO_EXIST;
    }
    LogDebug << "ImageReader[" << instanceId_ << "]: image file name - " << filePath.c_str() << "";

    // get image data
    fseek(pFile, 0, SEEK_END);
    fileSize = ftell(pFile);
    if (fileSize <= 0) {
        LogFatal << "get fileSize failed.";
        fclose(pFile);
        return APP_ERR_COMM_FAILURE;
    }
    buff = malloc(fileSize);
    if (buff == nullptr) {
        LogFatal << "buff malloc " << fileSize << " failed.";
        fclose(pFile);
        return APP_ERR_COMM_ALLOC_MEM;
    }
    rewind(pFile);
    numRead = fread(buff, sizeof(char), fileSize, pFile);
    LogDebug << "ImageReader[" << instanceId_ << "]: read " << numRead << "bytes.";
    fclose(pFile);

    std::string faceStr((char *)buff, fileSize);
    std::shared_ptr<DataTrans> dataTrans = std::make_shared<DataTrans>();
    ret = RegistApi::GetInstance()->RegistTheFace(uuid, faceStr, dataTrans);
    if (ret != APP_ERR_OK) {
        LogError << "ImageReader[" << instanceId_ << "]: RegistTheFace failed ret=" << ret;
    }
#ifdef ASCEND_ACL_OPEN_VESION
    HdcChannel::GetInstance()->SendData(HDC_REGIST_CH_INDEX, dataTrans);
#else
    SendToNextModule(MT_IMAGE_DECODER, dataTrans, 0);
#endif
    free(buff);
    return APP_ERR_OK;
}

APP_ERROR ImageReader::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "ImageReader[" << instanceId_ << "]: process.";
    std::string jpegFilePath;

    // open and scan folder
    APP_ERROR ret = ScanFolder(jpegFolderPath_);
    if (ret != APP_ERR_OK) {
        LogWarn << "ImageReader[" << instanceId_ << "]: fail to scan folder " << jpegFolderPath_ << ".";
        return APP_ERR_OK;
    }

    // 2. read file
    std::vector<std::string>::iterator it = fileNameSet_.begin();
    while ((it != fileNameSet_.end()) && (!stop_)) {
        std::vector<std::string> stringList = Split((*it), ".jpg");
        jpegFilePath = jpegFolderPath_ + "/" + (*it);
        ret = GetImageData(jpegFilePath, stringList[0]);
        it++;
    }

    LogDebug << "ImageReader[" << instanceId_ << "]: process end.";
    return ret;
}
} // namespace ascendFaceRecognition
