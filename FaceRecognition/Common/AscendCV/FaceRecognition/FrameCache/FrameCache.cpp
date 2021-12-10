/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1 Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2 Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3 Neither the names of the copyright holders nor the names of the
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * Description: frame cache.
 */
#include "FrameCache.h"
#include <string>
#include "Log/Log.h"

namespace ascendFaceRecognition {
FrameCache::FrameCache() {}

FrameCache::~FrameCache()
{
    frameMap_.clear();
}

std::shared_ptr<FrameCache> FrameCache::GetInstance(uint32_t channelId)
{
    static std::map<uint32_t, std::shared_ptr<FrameCache>> frameCacheMap;
    static std::mutex frameCacheMutex;
    std::lock_guard<std::mutex> guard(frameCacheMutex);
    if (frameCacheMap.find(channelId) != frameCacheMap.end()) {
        return frameCacheMap[channelId];
    }
    std::shared_ptr<FrameCache> frameCache = std::make_shared<FrameCache>();
    frameCacheMap[channelId] = frameCache;
    return frameCache;
}

void FrameCache::CacheFrame(std::shared_ptr<FrameAiInfo> frameAiInfo)
{
    std::lock_guard<std::mutex> guard(mtx);
    uint32_t key = GetStoreKey(frameAiInfo->info.frameId);
    frameMap_[key] = frameAiInfo;
    LogDebug << "FrameCache Cache " << key << " face count " << frameAiInfo->face.size();
}

std::shared_ptr<FrameAiInfo> FrameCache::GetFrame(uint32_t frameId)
{
    std::lock_guard<std::mutex> guard(mtx);
    uint32_t key = GetStoreKey(frameId);
    if (frameMap_.find(key) != frameMap_.end()) {
        return frameMap_[key];
    }
    return nullptr;
}

void FrameCache::ClearFrame(uint32_t frameId)
{
    std::lock_guard<std::mutex> guard(mtx);
    uint32_t key = GetStoreKey(frameId);
    LogDebug << "FrameCache Clear " << key << " [Todo]";
    if (frameMap_.find(key) != frameMap_.end()) {
        LogDebug << "FrameCache Clear " << key << " [Success]";
        frameMap_.erase(key);
    }
}

uint32_t FrameCache::GetStoreKey(uint32_t frameId) const
{
    return frameId;
}

uint32_t FrameCache::GetSize() const
{
    return frameMap_.size();
}
} // namespace ascendFaceRecognition
