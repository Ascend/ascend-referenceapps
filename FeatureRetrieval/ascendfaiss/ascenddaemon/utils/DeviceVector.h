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

#ifndef ASCEND_DEVICEVECTOR_INCLUDED
#define ASCEND_DEVICEVECTOR_INCLUDED

#include <ascenddaemon/utils/AscendUtils.h>
#include <vector>

namespace ascend {
template<typename T>
class DeviceVector {
public:
    DeviceVector(MemorySpace space = MemorySpace::DEVICE);

    ~DeviceVector();

    void clear();

    inline size_t size() const
    {
        return num;
    }

    inline size_t capacity() const
    {
        return vecCapacity;
    }

    inline T* data()
    {
        return dataPtr;
    }

    inline T* data() const
    {
        return dataPtr;
    }

    inline T& operator[](size_t pos);

    inline const T& operator[](size_t pos) const;

    std::vector<T> copyToStlVector() const;

    void append(const T* d, size_t n, bool reserveExact = false);

    void resize(size_t newSize, bool reserveExact = false);

    size_t reclaim(bool exact);

    void reserve(size_t newCapacity);

private:
    void realloc(size_t newCapacity);

    size_t getNewCapacity(size_t preferredSize);

private:
    T* dataPtr;
    size_t num;
    size_t vecCapacity;
    MemorySpace space;
};
}  // namespace ascend
#include <ascenddaemon/utils/DeviceVectorInl.h>

#endif  // ASCEND_DEVICEVECTOR_H
