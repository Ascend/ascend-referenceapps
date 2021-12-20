/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#include "AsynLog.h"

#include <sstream>
#include <iostream>

#include "FileManager/FileManager.h"

namespace {
std::string logFile = "./logs/log.log"; // default log file
std::string logFileBak = "./logs/log.log.bak";
const int FILE_SIZE = 52428800; // 50M
const int ASYNLOG_QUEUE_SIZE = 10000;
}

AsynLog& AsynLog::GetInstance()
{
    static AsynLog instance;
    return instance;
}

APP_ERROR AsynLog::SetLogMode(int mode)
{
    if (mode < SYNC_MODE || mode > ASYN_MODE) {
        std::cout << "set log mode failed, mode is " << mode << std::endl;
        return APP_ERR_COMM_INVALID_PARAM;
    }
    mode_ = mode;
    return APP_ERR_OK;
}

uint32_t AsynLog::GetLogMode() const
{
    return mode_;
}

bool AsynLog::IsLogQueueEmpty() const
{
    return logDataQueue_->IsEmpty();
}

APP_ERROR AsynLog::PushToLogQueue(const std::string &logData)
{
    logDataQueue_->Push(logData, true);
    return APP_ERR_OK;
}

APP_ERROR AsynLog::Run()
{
    std::cout << "Asyn log Running." << std::endl;
    mode_ = ASYN_MODE;
    logDataQueue_ = std::make_shared<BlockingQueue<std::string>>(ASYNLOG_QUEUE_SIZE);
    processThr_ = std::thread(&AsynLog::LogThreadFunc, this);
    return APP_ERR_OK;
}

void AsynLog::LogThreadFunc()
{
    while (!isStop_) {
        std::string logData;
        int ret = logDataQueue_->Pop(logData);
        if (ret == APP_ERR_QUEUE_STOPED) {
            std::cout << "log queue Stopped" << std::endl;
            return;
        }
        if (ret != APP_ERR_OK) {
            std::cout << "Fail to get data from log queue, ret=" << ret << "(" << GetAppErrCodeInfo(ret) << ").\n";
            return;
        }
        CallProcessLog(logData);
    }
}

void AsynLog::CallProcessLog(const std::string &logData)
{
    // cout to screen
    std::cout << logData << std::endl;
    // log to the file
    CreateDirRecursivelyByFile(logFile);
    std::ofstream fs(logFile, std::ios::app);
    if (!fs.is_open()) {
        std::cout << "open file " << logFile << " fail" << std::endl;
        return;
    }
    fs.seekp(0, fs.end);
    size_t dstFileSize = fs.tellp();
    fs << logData << std::endl;
    fs.close();

    if (dstFileSize < FILE_SIZE) {
        return;
    }
    // dstFileSize >= FILE_SIZE
    if (access(logFileBak.c_str(), 0) == 0) {
        int ret = remove(logFileBak.c_str());
        if (ret != 0) {
            std::cout << "remove " << logFileBak << " failed." << std::endl;
            return;
        }
    }
    int ret = rename(logFile.c_str(), logFileBak.c_str());
    if (ret != 0) {
        std::cout << "rename " << logFile << " failed." << std::endl;
        return;
    }
}

APP_ERROR AsynLog::Stop()
{
    isStop_ = true;
    if (logDataQueue_ != nullptr) {
        logDataQueue_->Stop();
        logDataQueue_->Clear();
    }
    if (processThr_.joinable()) {
        processThr_.join();
    }
    return APP_ERR_OK;
}