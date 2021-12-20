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

#ifndef ASCEND_DEVICEVECTORINL_INCLUDED
#define ASCEND_DEVICEVECTORINL_INCLUDED

#include <cstring>

#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/StaticUtils.h>

namespace ascend {
// if size < 2GB, use memcpy_s, else use memcpy
const size_t MEMCPY_S_THRESHOLD = 0x80000000;

template<typename T>
DeviceVector<T>::DeviceVector(MemorySpace space) : dataPtr(nullptr), num(0), vecCapacity(0), space(space)
{}

template<typename T> DeviceVector<T>::~DeviceVector()
{
    clear();
}

template<typename T> void DeviceVector<T>::clear()
{
    if (this->dataPtr != nullptr) {
        FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
        this->dataPtr = nullptr;
    }

    this->num = 0;
    this->vecCapacity = 0;
}

template<typename T> inline const T &DeviceVector<T>::operator[](size_t pos) const
{
    ASCEND_THROW_IF_NOT(pos >= 0 && pos < num);
    return *(dataPtr + pos);
}

template<typename T> inline T &DeviceVector<T>::operator[](size_t pos)
{
    ASCEND_THROW_IF_NOT(pos >= 0 && pos < num);
    return *(dataPtr + pos);
}

template<typename T> std::vector<T> DeviceVector<T>::copyToStlVector() const
{
    if (this->num == 0 || this->dataPtr == nullptr) {
        return std::vector<T>();
    }

    std::vector<T> out(this->num);
    ASCEND_ASSERT(this->num * sizeof(T) < MEMCPY_S_THRESHOLD);
    (void)memcpy_s(out.data(), this->num * sizeof(T), this->dataPtr, this->num * sizeof(T));

    return out;
}

template<typename T> void DeviceVector<T>::append(const T *d, size_t n, bool reserveExact)
{
    if (n <= 0) {
        return;
    }

    size_t reserveSize = this->num + n;
    if (!reserveExact) {
        reserveSize = getNewCapacity(reserveSize);
    }

    reserve(reserveSize);
    ASCEND_ASSERT(this->num * sizeof(T) < MEMCPY_S_THRESHOLD);
    (void)memcpy_s(this->dataPtr + this->num, n * sizeof(T), d, n * sizeof(T));

    this->num += n;
}

template<typename T> void DeviceVector<T>::resize(size_t newSize, bool reserveExact)
{
    if (this->num < newSize) {
        if (reserveExact) {
            reserve(newSize);
        } else {
            reserve(getNewCapacity(newSize));
        }
    }

    this->num = newSize;
}

template<typename T> size_t DeviceVector<T>::reclaim(bool exact)
{
    size_t free = this->vecCapacity - this->num;

    if (exact) {
        this->realloc(this->num);
        return free * sizeof(T);
    }

    // If more than 1/4th of the space is free, then we want to
    // truncate to only having 1/8th of the space free; this still
    // preserves some space for new elements, but won't force us to
    // double our size right away
    const int RECLAIM_PROPORTION = 4;
    const int TRUNCATE_PROPORTION = 8;
    if (free > (this->vecCapacity / RECLAIM_PROPORTION)) {
        size_t newFree = this->vecCapacity / TRUNCATE_PROPORTION;
        size_t newCapacity = this->num + newFree;

        size_t oldCapacity = this->vecCapacity;
        ASCEND_ASSERT(newCapacity < oldCapacity);

        this->realloc(newCapacity);

        return (oldCapacity - newCapacity) * sizeof(T);
    }

    return 0;
}

template<typename T>
void DeviceVector<T>::reserve(size_t newCapacity)
{
    if (newCapacity > this->vecCapacity) {
        this->realloc(newCapacity);
    }
}

template<typename T>
void DeviceVector<T>::realloc(size_t newCapacity)
{
    ASCEND_THROW_IF_NOT(this->num <= newCapacity);

    T *newData = nullptr;
    if (newCapacity) {
        allocMemorySpace(space, &newData, newCapacity * sizeof(T));

        if (this->dataPtr != nullptr) {
            ASCEND_ASSERT(this->num * sizeof(T) < MEMCPY_S_THRESHOLD);
            (void)memcpy_s(newData, newCapacity * sizeof(T), this->dataPtr, this->num * sizeof(T));

            FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
        }
    } else {
        FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
    }

    this->dataPtr = newData;
    this->vecCapacity = newCapacity;
}

template<typename T>
size_t DeviceVector<T>::getNewCapacity(size_t preferredSize)
{
    const int HALF_SCALE_THREADSHOULD = 512;
    const int QUARTER_SCALE_THREADSHOULD = 1024;
    const int EIGHTH_SCALE_THREADSHOULD = 2048;
    size_t tmpPrefer = utils::nextHighestPowerOf2(preferredSize);
    if (preferredSize >= HALF_SCALE_THREADSHOULD && preferredSize < QUARTER_SCALE_THREADSHOULD) {
        // scale 1/2 * preferredSize size more, to (3/2) 1.5 * preferredSize
        size_t tmp = preferredSize * 3 / 2;
        tmpPrefer = (tmpPrefer > tmp) ? tmp : tmpPrefer;
    } else if (preferredSize >= QUARTER_SCALE_THREADSHOULD && preferredSize < EIGHTH_SCALE_THREADSHOULD) {
        // scale 1/4 * preferredSize size more, to (5/4) 1.25 * preferredSize
        size_t tmp = preferredSize * 5 / 4;
        tmpPrefer = (tmpPrefer > tmp) ? tmp : tmpPrefer;
    } else if (preferredSize >= EIGHTH_SCALE_THREADSHOULD) {
        // scale 1/8 * preferredSize size more, to (9/8) 1.125 * preferredSize
        size_t tmp = preferredSize * 9 / 8;
        tmpPrefer = (tmpPrefer > tmp) ? tmp : tmpPrefer;
    }

    return tmpPrefer;
}
} // namespace ascend

#endif // ASCEND_DEVICEVECTORINL_H
