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


#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <queue>
#include <mutex>
#include "ErrorCode/ErrorCode.h"
#include "DataType/DataType.h"
#include "BlockingQueue/BlockingQueue.h"

namespace ascendFaceRecognition {
class MemoryPool {
public:
    MemoryPool();
    ~MemoryPool();
    static std::shared_ptr<MemoryPool> NewMemoryPoolResource();
    APP_ERROR Init(uint32_t blockSize, uint32_t blockNum);
    APP_ERROR DeInit();
    std::shared_ptr<uint8_t> GetMemoryBlock();

private:
    void RecycleMemoryBlock(const uint32_t &index);

private:
    std::shared_ptr<BlockingQueue<uint32_t>> idleQueue_ = nullptr;
    uint32_t blockNum_ = 0;
    uint32_t blockSize_ = 0;
    uint8_t *blockBufferHead_ = nullptr;
    bool isDeInit_ = false;
};
}

#endif
