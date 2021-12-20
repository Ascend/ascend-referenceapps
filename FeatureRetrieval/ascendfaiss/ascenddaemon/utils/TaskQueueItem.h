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

#ifndef ASCEND_TASK_QUEUE_ITEM_H
#define ASCEND_TASK_QUEUE_ITEM_H

namespace ascend {
namespace {
const int FLAG_ALIGN = 32;
const int FLAG_ALIGN_OFFSET = 16; // core 0 use first 16 flag, and core 1 use the second 16 flag.
}

struct QueueItem {
    QueueItem() : distPtr(nullptr), idPtr(nullptr), flagPtr(nullptr), size(0), executing(false) {}

    QueueItem(const QueueItem &item)
    {
        distPtr = item.distPtr;
        extremePtr = item.extremePtr;
        idPtr = item.idPtr;
        flagPtr = item.flagPtr;
        size = item.size;
        executing = item.executing ? true : false;
    }

    QueueItem &operator=(const QueueItem &item)
    {
        distPtr = item.distPtr;
        extremePtr = item.extremePtr;
        idPtr = item.idPtr;
        flagPtr = item.flagPtr;
        size = item.size;
        executing = item.executing ? true : false;
        return *this;
    }

    void SetExecuting(float16_t *dist, uint32_t *id, uint16_t *flag, int s)
    {
        ASCEND_ASSERT(dist != nullptr);
        ASCEND_ASSERT(id != nullptr);
        ASCEND_ASSERT(flag != nullptr);
        ASCEND_ASSERT(s != 0);

        distPtr = dist;
        idPtr = id;
        flagPtr = flag;
        flagPtrSec = flag + FLAG_ALIGN_OFFSET;
        size = s;
        executing = true;
    }

    void SetExecuting(float16_t *dist, float16_t *extreme, uint32_t *id, uint16_t *flag, int s)
    {
        ASCEND_ASSERT(dist != nullptr);
        ASCEND_ASSERT(id != nullptr);
        ASCEND_ASSERT(flag != nullptr);
        ASCEND_ASSERT(s != 0);

        distPtr = dist;
        extremePtr = extreme;
        idPtr = id;
        flagPtr = flag;
        flagPtrSec = flag + FLAG_ALIGN_OFFSET;
        size = s;
        executing = true;
    }

    inline bool IsExecuting()
    {
        return executing;
    }

    float16_t *distPtr;            // distance result mem pointer

    float16_t *extremePtr;         // extreme distance result mem pointer

    uint32_t *idPtr;               // ids mem pointer

    uint16_t *volatile flagPtr;    // flag mem pointer for aicore 0,
                                   // the first uint16_t will be setted to 1 when aicore finished calc

    uint16_t *volatile flagPtrSec; // flag mem pointer for aicore 1,
                                   // the first uint16_t will be setted to 1 when aicore finished calc

    int size;                      // size to idicate how many code to calc, and how many results to topk functor

    std::atomic<bool> executing;   // whether the item has beed added to stream for executing
};
}
#endif // ASCEND_TASK_QUEUE_ITEM_H
