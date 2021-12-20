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

#ifndef ASYNLOG_H
#define ASYNLOG_H

#include <string>
#include <memory>
#include <thread>

#include "BlockingQueue/BlockingQueue.h"
#include "ErrorCode/ErrorCode.h"

class AsynLog
{
public:
    static AsynLog& GetInstance();
    APP_ERROR Run();
    APP_ERROR Stop();
    APP_ERROR PushToLogQueue(const std::string &logData);
    APP_ERROR SetLogMode(int mode);
    uint32_t GetLogMode() const;
    bool IsLogQueueEmpty() const;

public:
    enum LogMode { // Sync/Asyn type of log
        SYNC_MODE = 0,
        ASYN_MODE = 1,
    };

private:
    void LogThreadFunc();
    void CallProcessLog(const std::string &logData);

private:
    std::shared_ptr<BlockingQueue<std::string>> logDataQueue_ = nullptr;
    std::thread processThr_ = {};
    bool isStop_ = false;
    uint32_t mode_ = 0;
};
#endif