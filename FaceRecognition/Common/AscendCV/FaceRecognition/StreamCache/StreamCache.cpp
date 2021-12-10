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

#include "StreamCache.h"
#include <string>
#include "Log/Log.h"

namespace ascendFaceRecognition {
StreamCache::StreamCache() = default;

StreamCache::~StreamCache()
{
    frameMap.clear();
}

std::shared_ptr<StreamCache> StreamCache::GetInstance(uint32_t channelId)
{
    static std::map<uint32_t, std::shared_ptr<StreamCache>> streamCacheMap;
    if (streamCacheMap.find(channelId) != streamCacheMap.end()) {
        return streamCacheMap[channelId];
    }
    std::shared_ptr<StreamCache> streamCache = std::make_shared<StreamCache>();
    streamCacheMap[channelId] = streamCache;
    return streamCache;
}

void StreamCache::CacheFrame(std::shared_ptr<DataTrans> dataTransInfo)
{
    std::lock_guard<std::mutex> guard(mtx);

    uint32_t key = dataTransInfo->streamdata().info().frameid();
    frameMap[key] = dataTransInfo;
    LogDebug << "StreamCache Cache " << key;
}

std::shared_ptr<DataTrans> StreamCache::GetFrame(uint32_t frameId)
{
    std::lock_guard<std::mutex> guard(mtx);
    if (frameMap.find(frameId) != frameMap.end()) {
        return frameMap[frameId];
    }
    return nullptr;
}

void StreamCache::ClearFrame(uint32_t frameId)
{
    std::lock_guard<std::mutex> guard(mtx);
    LogDebug << "StreamCache Clear " << frameId << " [Todo]";
    if (frameMap.find(frameId) != frameMap.end()) {
        LogDebug << "StreamCache Clear " << frameId << " [Success]";
        frameMap.erase(frameId);
    }
}

uint32_t StreamCache::GetSize() const
{
    return frameMap.size();
}

bool StreamCache::IsFrameExist(uint32_t frameId) const
{
    return frameMap.find(frameId) != frameMap.end();
}
} // namespace ascendFaceRecognition