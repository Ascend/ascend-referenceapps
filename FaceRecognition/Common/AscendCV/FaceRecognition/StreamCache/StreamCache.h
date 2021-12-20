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

#ifndef INC_STREAM_CACHE_H
#define INC_STREAM_CACHE_H

#include <map>
#include <memory>
#include <mutex>

#include "DataType/DataType.h"
#include "DataTrans/DataTrans.pb.h"

namespace ascendFaceRecognition {
class StreamCache {
public:
    static std::shared_ptr<StreamCache> GetInstance(uint32_t channelId);

    StreamCache();

    ~StreamCache();

    void CacheFrame(std::shared_ptr<DataTrans> dataTransInfo);

    std::shared_ptr<DataTrans> GetFrame(uint32_t frameId);

    void ClearFrame(uint32_t frameId);

    bool IsFrameExist(uint32_t frameId) const;

    uint32_t GetSize() const;

private:
    std::map<uint32_t, std::shared_ptr<DataTrans>> frameMap = {};
    std::mutex mtx = {};
};
}

#endif
