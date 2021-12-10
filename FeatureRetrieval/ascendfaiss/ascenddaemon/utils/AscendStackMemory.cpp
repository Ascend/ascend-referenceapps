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

#include <ascenddaemon/utils/AscendStackMemory.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <sstream>

namespace ascend {
AscendStackMemory::Stack::Stack(size_t sz)
    : isOwner(true),
      start(nullptr),
      end(nullptr),
      size(sz),
      head(nullptr),
      mallocCurrent(0),
      highWaterMemoryUsed(0),
      highWaterMalloc(0),
      mallocWarning(true)
{
    if (size > 0) {
        allocMemorySpace(MemorySpace::DEVICE_HUGEPAGE, &start, size);
    }

    if (start != nullptr) {
        head = start;
        end = start + size;
    }
}

AscendStackMemory::Stack::Stack(void* p, size_t sz, bool isOwner)
    : isOwner(isOwner),
      start((char*)p),
      end(((char*)p) + sz),
      size(sz),
      head((char*)p),
      mallocCurrent(0),
      highWaterMemoryUsed(0),
      highWaterMalloc(0),
      mallocWarning(true)
{
}

AscendStackMemory::Stack::~Stack()
{
    reset();
}

bool AscendStackMemory::Stack::alloc(size_t sz)
{
    if (sz != size && sz > 0) {
        reset();

        size = sz;
        allocMemorySpace(MemorySpace::DEVICE_HUGEPAGE, &start, size);

        head = start;
        end = start + size;
    }
    
    if (sz == 0) {
        reset();
    }

    return true;
}

void AscendStackMemory::Stack::reset()
{
    if (isOwner && start != nullptr) {
        FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, start);
    }

    isOwner = true;
    start = nullptr;
    end = nullptr;
    size = 0;
    head = nullptr;
    mallocCurrent = 0;
    highWaterMemoryUsed = 0;
    highWaterMalloc = 0;
    mallocWarning = true;
}

size_t AscendStackMemory::Stack::getSizeAvailable() const
{
    return (end - head);
}

char* AscendStackMemory::Stack::getAlloc(size_t sz, aclrtStream stream)
{
    if (sz > (size_t)(end - head)) {
        if (mallocWarning) {
            // Print our requested size before we attempt the allocation
            ACL_APP_LOG(ACL_WARNING, "[ascendfaiss] increase temp memory to avoid aclrtMalloc, "
                "or decrease query/add size (alloc %zu B, highwater %zu B)\n", sz, highWaterMalloc);
        }

        char* p = nullptr;
        allocMemorySpace(MemorySpace::DEVICE_HUGEPAGE, &p, sz);
        mallocCurrent += sz;
        highWaterMalloc = std::max(highWaterMalloc, mallocCurrent);
        return p;
    } else {
        // We can make the allocation out of our stack
        // Find all the ranges that we overlap that may have been
        // previously allocated; our allocation will be [head, endAlloc)
        char* startAlloc = head;
        char* endAlloc = head + sz;

        while (lastUsers.size() > 0) {
            auto& prevUser = lastUsers.back();

            // Because there is a previous user, we must overlap it
            ASCEND_ASSERT(prevUser.start <= endAlloc &&
                          prevUser.end >= startAlloc);

            // stream != prevUser.stream never happened  [before 2021-03]
            // Synchronization required, never come here [before 2021-03]
            // After aclrtMalloc by some stream, memory can used for all device [2021-03]
            // No need to care stream of prev-user

            if (endAlloc < prevUser.end) {
                // Update the previous user info
                prevUser.start = endAlloc;
                break;
            }

            // If we're the exact size of the previous request, then we
            // don't need to continue
            bool done = (prevUser.end == endAlloc);
            lastUsers.pop_back();
            if (done) {
                break;
            }
        }

        head = endAlloc;
        ASCEND_ASSERT(head <= end);

        highWaterMemoryUsed = std::max(highWaterMemoryUsed, (size_t)(head - start));
        return startAlloc;
    }
}

void AscendStackMemory::Stack::returnAlloc(char* p, size_t sz, aclrtStream stream)
{
    if (p < start || p >= end) {
        // This is not on our stack; it was a one-off allocation
        FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, p);
        ASCEND_ASSERT(mallocCurrent >= sz);
        mallocCurrent -= sz;
    } else {
        // This is on our stack
        // Allocations should be freed in the reverse order they are made
        ASCEND_ASSERT(p + sz == head);

        head = p;
        lastUsers.push_back(Range(p, p + sz, stream));
    }
}

std::string AscendStackMemory::Stack::toString() const
{
    std::stringstream s;

    s << "Total memory " << size << " ["
      << (void*)start << ", " << (void*)end << ")\n";
    s << "     Available memory " << (size_t)(end - head)
      << " [" << (void*)head << ", " << (void*)end << ")\n";
    s << "     High water temp alloc " << highWaterMemoryUsed << "\n";
    s << "     High water aclrtMalloc " << highWaterMalloc << "\n";

    int i = lastUsers.size();
    for (auto it = lastUsers.rbegin(); it != lastUsers.rend(); ++it) {
        s << i-- << ": size " << (size_t)(it->end - it->start)
          << " stream " << it->stream
          << " [" << (void*)it->start << ", " << (void*)it->end << ")\n";
    }

    return s.str();
}

size_t AscendStackMemory::Stack::getHighWater() const
{
    return highWaterMalloc;
}

AscendStackMemory::AscendStackMemory()
    : stack(0), refCount(0)
{
}

bool AscendStackMemory::allocMemory(size_t sz)
{
    return stack.alloc(sz);
}

void AscendStackMemory::ref()
{
    ++refCount;
}

void AscendStackMemory::unRef()
{
    if (--refCount == 0) {
        stack.reset();
    }
}

AscendStackMemory::AscendStackMemory(size_t alloc)
    : stack(alloc), refCount(0)
{
}

AscendStackMemory::AscendStackMemory(void* p, size_t size, bool isOwner)
    : stack(p, size, isOwner), refCount(0)
{
}

AscendStackMemory::~AscendStackMemory()
{
}

void AscendStackMemory::setMallocWarning(bool flag)
{
    stack.mallocWarning = flag;
}

AscendMemoryReservation AscendStackMemory::getMemory(aclrtStream stream, size_t size)
{
    // We guarantee 32 byte alignment for allocations, so bump up `size`
    // to the next highest multiple of 32
    size = utils::roundUp(size, (size_t)32);
    return AscendMemoryReservation(this,
                                   stack.getAlloc(size, stream),
                                   size,
                                   stream);
}

size_t AscendStackMemory::getSizeAvailable() const
{
    return stack.getSizeAvailable();
}

std::string AscendStackMemory::toString() const
{
    return stack.toString();
}

size_t AscendStackMemory::getHighWater() const
{
    return stack.getHighWater();
}

void AscendStackMemory::returnAllocation(AscendMemoryReservation& m)
{
    ASCEND_ASSERT(m.get());

    stack.returnAlloc((char*) m.get(), m.size(), m.stream());
}
}  // namespace ascend