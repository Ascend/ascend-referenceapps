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

#include "MemoryPool.h"
#include "acl/ops/acl_dvpp.h"
#include "Log/Log.h"
#include "acl/acl.h"

namespace {
const int DEFAULT_OFFSET_DVPP = 32; // acldvppMalloc requires more 32 Bytes for every picture
}

namespace ascendFaceRecognition {
MemoryPool::MemoryPool() {}
MemoryPool::~MemoryPool()
{
    DeInit();
}

std::shared_ptr<MemoryPool> MemoryPool::NewMemoryPoolResource()
{
    static auto memoryPoolQueue = std::make_shared<BlockingQueue<std::shared_ptr<MemoryPool>>>(MODULE_QUEUE_SIZE);
    std::shared_ptr<MemoryPool> memoryPool = std::make_shared<MemoryPool>();
    memoryPoolQueue->Push(memoryPool, true);
    return memoryPool;
}

APP_ERROR MemoryPool::Init(uint32_t blockSize, uint32_t blockNum)
{
    blockSize_ = blockSize;
    blockNum_ = blockNum;
    uint32_t totalSize = (blockSize + DEFAULT_OFFSET_DVPP) * blockNum;
    APP_ERROR ret = acldvppMalloc((void **)&blockBufferHead_, totalSize);
    if (ret != APP_ERR_OK) {
        LogError << "MemoryPool: apply memory error ret=" << ret;
        return ret;
    }

    idleQueue_ = std::make_shared<BlockingQueue<uint32_t>>(MODULE_QUEUE_SIZE);
    if (idleQueue_.get() == nullptr) {
        LogError << "MemoryPool: apply queue error";
        return APP_ERR_COMM_FAILURE;
    }

    for (uint32_t i = 0; i < blockNum_; i++) {
        idleQueue_->Push(i, true);
    }
    return APP_ERR_OK;
}

APP_ERROR MemoryPool::DeInit()
{
    if (idleQueue_.get() != nullptr) {
        idleQueue_->Stop();
        idleQueue_->Clear();
        idleQueue_ = nullptr;
    }

    if (blockBufferHead_ != nullptr) {
        APP_ERROR ret = acldvppFree(blockBufferHead_);
        if (ret != APP_ERR_OK) {
            LogError << "MemoryPool: memory free error ret=" << ret;
            return ret;
        }
        blockBufferHead_ = nullptr;
    }

    isDeInit_ = true;
    return APP_ERR_OK;
}


std::shared_ptr<uint8_t> MemoryPool::GetMemoryBlock()
{
    uint32_t index = 0;
    idleQueue_->Pop(index);
    auto addressDeleter = [this, index](const uint8_t *p) {
        this->RecycleMemoryBlock(index);
    };
    uint8_t *address = static_cast<uint8_t *>(blockBufferHead_) + index * (blockSize_ + DEFAULT_OFFSET_DVPP);
    std::shared_ptr<uint8_t> memoryBlock(address, addressDeleter);
    return memoryBlock;
}

void MemoryPool::RecycleMemoryBlock(const uint32_t &index)
{
    if (isDeInit_) {
        return;
    }
    idleQueue_->Push(index, true);
}
}
