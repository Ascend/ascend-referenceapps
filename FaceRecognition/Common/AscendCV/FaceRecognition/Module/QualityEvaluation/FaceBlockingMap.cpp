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

#include "FaceBlockingMap.h"
#include <chrono>

namespace ascendFaceRecognition {
FaceBlockingMap::FaceBlockingMap() {}

FaceBlockingMap::~FaceBlockingMap() {}

std::shared_ptr<FaceBlockingMap> &FaceBlockingMap::GetInstance(const int32_t &channelId)
{
    static std::map<uint32_t, std::shared_ptr<FaceBlockingMap>> faceChannelMaps;
    static std::mutex channelMutex;
    std::lock_guard<std::mutex> guard(channelMutex);
    if (faceChannelMaps.find(channelId) != faceChannelMaps.end()) {
        return faceChannelMaps[channelId];
    } else {
        faceChannelMaps[channelId] = std::make_shared<FaceBlockingMap>();
        return faceChannelMaps[channelId];
    }
}

void FaceBlockingMap::Insert(const int32_t &id, std::shared_ptr<FaceObject> faceObject)
{
    std::lock_guard<std::mutex> guard(mtx_);
    faceMap_[id] = faceObject;
    keys_.insert(id);
}

std::shared_ptr<FaceObject> FaceBlockingMap::Get(const int32_t &id)
{
    std::lock_guard<std::mutex> guard(mtx_);
    if (faceMap_.find(id) != faceMap_.end()) {
        return faceMap_[id];
    } else {
        return nullptr;
    }
}

std::shared_ptr<FaceObject> FaceBlockingMap::Pop(const int32_t &id)
{
    std::lock_guard<std::mutex> guard(mtx_);
    if (faceMap_.find(id) != faceMap_.end()) {
        auto faceObject = faceMap_[id];
        faceMap_.erase(id);
        keys_.erase(id);
        return faceObject;
    } else {
        return nullptr;
    }
}

void FaceBlockingMap::Clear(const int32_t &id)
{
    std::lock_guard<std::mutex> guard(mtx_);
    faceMap_.erase(id);
    keys_.erase(id);
    return;
}

size_t FaceBlockingMap::Size() const
{
    return faceMap_.size();
}

std::vector<int32_t> FaceBlockingMap::Keys()
{
    std::vector<int32_t> keys;
    std::lock_guard<std::mutex> guard(mtx_);
    for (auto iter = keys_.begin(); iter != keys_.end(); iter++) {
        keys.push_back(*iter);
    }
    return keys;
}
} // namespace ascendFaceRecognition
