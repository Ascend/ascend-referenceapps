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

#include <ascenddaemon/utils/AscendMemory.h>

namespace ascend {
AscendMemoryReservation::AscendMemoryReservation() : parent(nullptr), dataPtr(nullptr), dataSize(0), aclStream(0) {}

AscendMemoryReservation::AscendMemoryReservation(AscendMemory *mem, void *data, size_t size, aclrtStream stream)
    : parent(mem), dataPtr(data), dataSize(size), aclStream(stream)
{}

AscendMemoryReservation::AscendMemoryReservation(AscendMemoryReservation &&m) noexcept
{
    parent = m.parent;
    dataPtr = m.dataPtr;
    dataSize = m.dataSize;
    aclStream = m.aclStream;

    m.dataPtr = nullptr;
}

AscendMemoryReservation::~AscendMemoryReservation()
{
    if (dataPtr) {
        ASCEND_ASSERT(parent);
        parent->returnAllocation(*this);
    }

    dataPtr = nullptr;
}

AscendMemoryReservation &AscendMemoryReservation::operator = (ascend::AscendMemoryReservation &&m)
{
    if (dataPtr) {
        ASCEND_ASSERT(parent);
        parent->returnAllocation(*this);
    }

    parent = m.parent;
    dataPtr = m.dataPtr;
    dataSize = m.dataSize;
    aclStream = m.aclStream;

    m.dataPtr = nullptr;

    return *this;
}

void *AscendMemoryReservation::get()
{
    return dataPtr;
}

size_t AscendMemoryReservation::size()
{
    return dataSize;
}

aclrtStream AscendMemoryReservation::stream()
{
    return aclStream;
}

AscendMemory::~AscendMemory() {}
} // namespace ascend
