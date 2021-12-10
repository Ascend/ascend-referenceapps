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

#ifndef ASCEND_STACK_DEVICE_MEMORY_INCLUDED
#define ASCEND_STACK_DEVICE_MEMORY_INCLUDED

#include <ascenddaemon/utils/AscendMemory.h>
#include <list>

namespace ascend {
/// Not MT Safe
class AscendStackMemory : public AscendMemory {
public:
    static AscendStackMemory& getInstance()
    {
        static AscendStackMemory memory;
        return memory;
    }

    /// Allocate a new region of memory that we manage
    explicit AscendStackMemory(size_t alloc);

    /// For instantiating a singleton
    AscendStackMemory();

    /// Manage a region of memory for a particular device, with or
    /// without ownership
    AscendStackMemory(void* p, size_t size, bool isOwner);

    ~AscendStackMemory() override;

    // alloc memory after get singleton
    bool allocMemory(size_t sz);

    // holder bind to singleton
    void ref();

    // holder release from singleton
    void unRef();

    /// Enable or disable the warning about not having enough temporary memory
    /// when aclrtMalloc gets called
    void setMallocWarning(bool flag);

    AscendMemoryReservation getMemory(aclrtStream stream, size_t size) override;

    size_t getSizeAvailable() const override;
    std::string toString() const override;
    size_t getHighWater() const override;

protected:
    void returnAllocation(AscendMemoryReservation& m) override;

    /// Previous allocation ranges and the streams for which
    /// synchronization is required
    struct Range {
        inline Range(char* s, char* e, aclrtStream str)
            : start(s),
              end(e),
              stream(str)
        {
        }

        // References a memory range [start, end)
        char* start;
        char* end;
        aclrtStream stream;
    };

    struct Stack {
        /// Constructor that allocates memory via aclrtMalloc
        Stack(size_t sz);

        /// Constructor that references a pre-allocated region of memory
        Stack(void* p, size_t size, bool isOwner);

        ~Stack();

        bool alloc(size_t sz);

        void reset();

        /// Returns how much size is available for an allocation without
        /// calling aclrtMalloc
        size_t getSizeAvailable() const;

        /// Obtains an allocation; all allocations are guaranteed to be 8
        /// byte aligned
        char* getAlloc(size_t size, aclrtStream stream);

        /// Returns an allocation
        void returnAlloc(char* p, size_t size, aclrtStream stream);

        /// Returns the stack state
        std::string toString() const;

        /// Returns the high-water mark of aclrtMalloc activity
        size_t getHighWater() const;

        /// Do we own our region of memory?
        bool isOwner;

        /// Where our allocation begins and ends
        /// [start, end) is valid
        char* start;
        char* end;

        /// Total size end - start
        size_t size;

        /// Stack head within [start, end)
        char* head;

        /// List of previous last users of allocations on our stack, for
        /// possible synchronization purposes
        std::list<Range> lastUsers;

        /// How much aclrtMalloc memory is currently outstanding?
        size_t mallocCurrent;

        /// What's the high water mark in terms of memory used from the
        /// temporary buffer?
        size_t highWaterMemoryUsed;

        /// What's the high water mark in terms of memory allocated via
        /// aclrtMalloc?
        size_t highWaterMalloc;

        /// Whether or not a warning upon aclrtMalloc is generated
        bool mallocWarning;
    };

    /// Memory stack
    Stack stack;

    // refrence count for holder
    int refCount;
};
}  // namespace ascend

#endif

