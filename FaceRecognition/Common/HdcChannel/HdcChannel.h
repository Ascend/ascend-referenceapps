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

#ifndef HDC_CHANNEL_H
#define HDC_CHANNEL_H

#include <unistd.h>
#include <memory>
#include <pthread.h>
#include <vector>

#include "BlockingQueue/BlockingQueue.h"
#include "DataType/DataType.h"
#include "Hdc/Hdc.h"

namespace ascendFaceRecognition {
enum HdcChannelMode {
    HDC_CHANNEL_HOST,
    HDC_CHANNEL_DEVICE
};

enum HdcChannelDirection {
    HDC_CHANNEL_SEND,
    HDC_CHANNEL_RECV
};

struct ThreadData {
    bool isStop = false;
    HdcChannelDirection direction = HDC_CHANNEL_SEND;
    Hdc *hdc = nullptr;
    HDC_SESSION session = nullptr;
    std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> dataQueue = nullptr;
};

class HdcChannel {
public:
    static std::shared_ptr<HdcChannel> GetInstance();
    HdcChannel() : hdcMode_(HDC_CHANNEL_HOST) {}
    ~HdcChannel();
    APP_ERROR Init(int deviceId, HdcChannelMode hdcMode, int sendChCount, int recvChCount);

    APP_ERROR SendData(int hdcChannelId, std::shared_ptr<void> data);

    // 获取接收队列
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> GetRecvChannelVec(int startIndex, int endIndex);

    APP_ERROR Run();

    APP_ERROR DeInit();

private:
    void RegisterHdcChannel(HdcChannelDirection direction, int count);
    void RegisterHdcChWithQueue(HdcChannelDirection direction,
        std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> dataQueue);
    static void *ReciveDataProcess(void *args);
    static void *SendDataProcess(void *args);

    int deviceId_ = 0;
    std::shared_ptr<Hdc> hdc_ = {};
    std::vector<HDC_SESSION> sessions_ = {};
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> sendChVec_ = {}; //
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> recvChVec_ = {}; //
    std::vector<ThreadData> threadsData_ = {};                          // device max is 64, host max is 1024
    std::vector<pthread_t> threadsId_ = {};
    int channelCount_ = 1;
    HdcChannelMode hdcMode_ = HDC_CHANNEL_HOST;
    bool isStop_ = false;
};
} // namespace ascendFaceRecognition

#endif