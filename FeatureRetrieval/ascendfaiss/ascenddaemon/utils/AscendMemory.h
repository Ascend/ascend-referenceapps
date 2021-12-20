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

#ifndef ASCEND_ASCENDMEMORY_INCLUDED
#define ASCEND_ASCENDMEMORY_INCLUDED

#include <ascenddaemon/utils/AscendUtils.h>

namespace ascend {
class AscendMemory;

class AscendMemoryReservation {
public:
    AscendMemoryReservation();

    AscendMemoryReservation(AscendMemory *mem, void *data, size_t dataSize, aclrtStream stream);

    AscendMemoryReservation(AscendMemoryReservation &&m) noexcept;
    ~AscendMemoryReservation();

    AscendMemoryReservation &operator = (AscendMemoryReservation &&m);

    void *get();
    size_t size();
    aclrtStream stream();

private:
    AscendMemory *parent;
    void *dataPtr;
    size_t dataSize;
    aclrtStream aclStream;
};

class AscendMemory {
public:
    virtual ~AscendMemory();

    // / Obtains a temporary memory allocation for our device,
    // / whose usage is ordered with respect to the given stream.
    virtual AscendMemoryReservation getMemory(aclrtStream stream, size_t size) = 0;

    // / Returns the current size available without calling aclrtMalloc
    virtual size_t getSizeAvailable() const = 0;

    // / Returns a string containing our current memory manager state
    virtual std::string toString() const = 0;

    // / Returns the high-water mark of aclrtMalloc allocations for our device
    virtual size_t getHighWater() const = 0;

protected:
    friend class AscendMemoryReservation;
    virtual void returnAllocation(AscendMemoryReservation &m) = 0;
};
} // namespace ascend

#endif // ASCEND_ASCENDMEMORY_H
