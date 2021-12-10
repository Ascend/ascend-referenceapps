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

#include "DeliveryChannel.h"
#include "Log/Log.h"

std::shared_ptr<DeliveryChannel> DeliveryChannel::GetInstance()
{
    static std::shared_ptr<DeliveryChannel> deliveryChannel = std::make_shared<DeliveryChannel>();
    return deliveryChannel;
}

void DeliveryChannel::SendData(const uint32_t &channelId, std::shared_ptr<void> data)
{
    if (!IsExsist(channelId)) {
        LogWarn << "DeliveryChannel:No Channel:" << channelId;
        return;
    }
    queueMap_[channelId]->Push(data, true);
}

APP_ERROR DeliveryChannel::Regist(const uint32_t &channelId,
    std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> queue)
{
    std::lock_guard<std::mutex> guard(mtx_);
    LogDebug << "Regist channelId=" << channelId;
    if (queue.get() == nullptr) {
        LogError << "DeliveryChannel: Regist fail";
        return APP_ERR_COMM_FAILURE;
    }
    queueMap_[channelId] = queue;
    return APP_ERR_OK;
}

APP_ERROR DeliveryChannel::GetData(const uint32_t &channelId, std::shared_ptr<void> &data)
{
    if (!IsExsist(channelId)) {
        return APP_ERR_COMM_NO_EXIST;
    }
    APP_ERROR ret = queueMap_[channelId]->Pop(data);
    return ret;
}

bool DeliveryChannel::IsExsist(const uint32_t channelId)
{
    std::lock_guard<std::mutex> guard(mtx_);
    if (queueMap_.find(channelId) == queueMap_.end()) {
        return false;
    }
    return true;
}