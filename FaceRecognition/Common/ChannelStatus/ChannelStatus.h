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

#ifndef INC_CHANNEL_STATUS_H
#define INC_CHANNEL_STATUS_H

#include <mutex>
#include <memory>
#include <map>
#include <cstdint>

namespace ascendFaceRecognition {
class ChannelStatus {
public:
    static std::shared_ptr<ChannelStatus> GetInstance();
    ChannelStatus();
    ~ChannelStatus();

    bool IsAlive(uint32_t channelId);
    void Update(uint32_t channelId, bool status);

private:
    std::mutex mtx_ = {};
    std::map<uint32_t, bool> statusMap_ = {};
};
}
#endif