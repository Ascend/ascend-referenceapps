/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
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

#ifndef INC_DELIVERY_CHANNEL_H
#define INC_DELIVERY_CHANNEL_H

#include <memory>
#include <map>
#include "ErrorCode/ErrorCode.h"
#include "BlockingQueue/BlockingQueue.h"

const uint32_t DELIVERY_VIDEO_STREAM_INDEX = 0;
const uint32_t DELIVERY_PICTURE_INDEX = 1000;

class DeliveryChannel {
public:
    DeliveryChannel() {}
    ~DeliveryChannel() {}
    static std::shared_ptr<DeliveryChannel> GetInstance();
    void SendData(const uint32_t &channelId, std::shared_ptr<void> data);
    APP_ERROR GetData(const uint32_t &channelId, std::shared_ptr<void> &data);
    APP_ERROR Regist(const uint32_t &channelId, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> queue);
private:
    bool IsExsist(const uint32_t channelId);
private:
    std::mutex mtx_ = {};
    std::map<uint32_t, std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> queueMap_ = {};
};
#endif // _INC_DELIVERY_CHANNEL_H
