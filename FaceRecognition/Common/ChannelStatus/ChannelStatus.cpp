/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: 
 * Author: 
 * Create: 
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

#include "ChannelStatus.h"
namespace ascendFaceRecognition {
ChannelStatus::ChannelStatus() {}
ChannelStatus::~ChannelStatus() {}
std::shared_ptr<ChannelStatus> ChannelStatus::GetInstance()
{
    static std::shared_ptr<ChannelStatus> channelStatus = std::make_shared<ChannelStatus>();
    return channelStatus;
}

bool ChannelStatus::IsAlive(uint32_t channelId) 
{
    if (statusMap_.find(channelId) == statusMap_.end()) {
        return false;
    }
    return statusMap_[channelId];
}

void ChannelStatus::Update(uint32_t channelId, bool status)
{
    statusMap_[channelId] = status;
}
}