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
#ifndef INC_FRAME_CACHE_H
#define INC_FRAME_CACHE_H

#include <map>
#include <memory>
#include <mutex>

#include "DataType/DataType.h"

namespace ascendFaceRecognition {
class FrameCache {
public:
    static std::shared_ptr<FrameCache> GetInstance(uint32_t channelId);
    FrameCache();
    ~FrameCache();

    void CacheFrame(std::shared_ptr<FrameAiInfo> frameAiInfo);
    std::shared_ptr<FrameAiInfo> GetFrame(uint32_t frameId);
    void ClearFrame(uint32_t frameId);
    uint32_t GetSize() const;

private:
    uint32_t GetStoreKey(uint32_t frameId) const;

private:
    std::map<uint32_t, std::shared_ptr<FrameAiInfo>> frameMap_ = {};
    std::mutex mtx = {};
};
}

#endif
