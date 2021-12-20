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

#include <ascenddaemon/utils/TopkOp.h>

#include <limits>
#include <omp.h>
#include <algorithm>

#include <ascenddaemon/utils/Limits.h>

// lambda expression captures unused variable
#define UNUSED(var) [&var]{}()

namespace ascend {
namespace {
int32_t GetDefaultVal(int32_t val, bool asc)
{
    UNUSED(val); // just for matching overloading function
    if (asc) {
        return std::numeric_limits<int32_t>::max();
    }
    return std::numeric_limits<int32_t>::min();
}

float16_t GetDefaultVal(float16_t val, bool asc)
{
    UNUSED(val);
    if (asc) {
        return Limits<float16_t>::getMax();
    }
    return Limits<float16_t>::getMin();
}
}

template<typename T, typename E, typename D, bool ASC>
TopkOp<T, E, D, ASC>::TopkOp() {}

template<typename T, typename E, typename D, bool ASC>
TopkOp<T, E, D, ASC>::~TopkOp() {}

template<typename T, typename E, typename D, bool ASC>
void TopkOp<T, E, D, ASC>::reorder(AscendTensor<D, DIMS_2> &topkDistance, AscendTensor<uint32_t, DIMS_2> &topkIndices)
{
    int n = topkIndices.getSize(0);

#pragma omp parallel for if (n > 1) num_threads(n < 6 ? n : 6)
    for (int i = 0; i < n; i++) {
        auto dist = topkDistance[i].view();
        auto id = topkIndices[i].view();
        reorder(dist, id);
    }
}

template<typename T, typename E, typename D, bool ASC>
void TopkOp<T, E, D, ASC>::reorder(AscendTensor<D, 1> &topkDistance, AscendTensor<uint32_t, 1> &topkIndices)
{
    D *dist = topkDistance.data();
    uint32_t *ids = topkIndices.data();
    size_t k = (size_t)topkIndices.getSize(0);

    size_t i, j;

    for (i = 0, j = 0; i < k; i++) {
        /* top element should be put at the end of the list */
        D val = dist[0];
        uint32_t id = ids[0];

        popHeap(k - i, dist, ids, compare);
        dist[k - j - 1] = val;
        ids[k - j - 1] = id;
        if (id != std::numeric_limits<uint32_t>::max()) {
            j++;
        }
    }

    auto err = memmove_s(dist, k * sizeof(*dist), dist + k - j, j * sizeof(*dist));
    ASCEND_THROW_IF_NOT_FMT(err == EOK, "An error occured when memmove in TopK reorder, %d!\n", err);
    err = memmove_s(ids, k * sizeof(*ids), ids + k - j, j * sizeof(*ids));
    ASCEND_THROW_IF_NOT_FMT(err == EOK, "An error occured when memmove in TopK reorder, %d!\n", err);

    for (; j < k; j++) {
        dist[j] = GetDefaultVal(dist[j], ASC);
        // invalid id should be -1 or max unsigned value
        ids[j] = std::numeric_limits<uint32_t>::max();
    }
}

template<typename T, typename E, typename D, bool ASC>
bool TopkOp<T, E, D, ASC>::exec(AscendTensor<D, DIMS_2> &distance,
                                AscendTensor<uint32_t, DIMS_2> &indices,
                                AscendTensor<D, DIMS_2> &topkDistance,
                                AscendTensor<uint32_t, DIMS_2> &topkIndices,
                                uint32_t indexOffset)
{
    if (!checkParams(distance, indices, topkDistance, topkIndices)) {
        return false;
    }

    D *disArray = distance.data();
    uint32_t *idArray = indices.data();
    D *disResult = topkDistance.data();
    uint32_t *idResult = topkIndices.data();
    const int batch = distance.getSize(0);
    const int numDist = distance.getSize(1);
    const int kVal = topkDistance.getSize(1);
    bool idFlag = idArray != nullptr;

#pragma omp parallel for if (batch > 1) num_threads(batch < 6 ? batch : 6)
    for (int i = 0; i < batch; i++) {
        int batchOffset = numDist * i;
        int heapOffset = kVal * i;
        for (int j = 0; j < numDist; j++) {
            if (compareEqual(disArray[batchOffset + j], disResult[heapOffset])) {
                continue;
            }
            int index = idFlag ? *(idArray + batchOffset + j) : (indexOffset + j);
            pushHeap(kVal, disResult + heapOffset, idResult + heapOffset, disArray[batchOffset + j], index, compare);
        }
    }

    return true;
}

template<typename T, typename E, typename D, bool ASC>
bool TopkOp<T, E, D, ASC>::exec(AscendTensor<D, DIMS_1> &distance,
                                AscendTensor<uint32_t, DIMS_1> &indices,
                                AscendTensor<D, DIMS_1> &topkDistance,
                                AscendTensor<uint32_t, DIMS_1> &topkIndices,
                                uint32_t realSize,
                                uint32_t indexOffset)
{
    if (!checkParams(distance, indices, topkDistance, topkIndices)) {
        return false;
    }

    if (distance.getSize(0) < (int)realSize) {
        return false;
    }

    D *disArray = distance.data();
    uint32_t *idArray = indices.data();
    D *disResult = topkDistance.data();
    uint32_t *idResult = topkIndices.data();
    const int kVal = topkDistance.getSize(0);
    bool idFlag = idArray != nullptr;

    for (uint32_t j = 0; j < realSize; j++) {
        if (compareEqual(disArray[j], disResult[0])) {
            continue;
        }
        int index = idFlag ? *(idArray + j) : (indexOffset + j);
        pushHeap(kVal, disResult, idResult, *(disArray + j), index, compare);
    }

    return true;
}

template<typename T, typename E, typename D, bool ASC>
bool TopkOp<T, E, D, ASC>::exec(std::tuple<D *, D *, uint32_t *> &opOutTp,
                                std::tuple<D *, uint32_t *, int> &topKHeapTp,
                                std::tuple<D *, uint32_t *> &tmpHeapTp,
                                const uint32_t realSize,
                                const int burstLen)
{
    D *extremeDisArray = std::get<1>(opOutTp);
    D *tmpDisResult = std::get<0>(tmpHeapTp);
    uint32_t *tmpIdResult = std::get<1>(tmpHeapTp);

    const int kVal = std::get<2>(topKHeapTp);
    const int extremeSize = (realSize + burstLen - 1) / burstLen * 2;
    const int halfMinBatch = burstLen / 2;

    // 2 reps skip index, only handle distance
    for (int i = 0; i < extremeSize; i += 2) {
        if (compareEqual(extremeDisArray[i], tmpDisResult[0])) {
            continue;
        }

        pushHeap(kVal, tmpDisResult, tmpIdResult, extremeDisArray[i], i * halfMinBatch, compare);
    }
    std::sort(tmpIdResult, tmpIdResult + kVal);

    D *disArray = std::get<0>(opOutTp);
    uint32_t *idArray = std::get<2>(opOutTp); // 2 reps second element
    D *disResult = std::get<0>(topKHeapTp);
    uint32_t *idResult = std::get<1>(topKHeapTp);

    const bool idFlag = idArray != nullptr;
    const uint32_t *tmpIdEnd = tmpIdResult + std::min(kVal, extremeSize / 2);

    for (uint32_t *pid = tmpIdResult; pid != tmpIdEnd; ++pid) {
        const int endIdx = std::min(pid[0] + burstLen, realSize);
        for (int j = pid[0]; j < endIdx; ++j) {
            if (compareEqual(disArray[j], disResult[0])) {
                continue;
            }

            uint32_t index = idFlag ? *(idArray + j) : j;
            pushHeap(kVal, disResult, idResult, disArray[j], index, compare);
        }
    }

    return true;
}

template<typename T, typename E, typename D, bool ASC>
bool TopkOp<T, E, D, ASC>::exec(std::tuple<D *, D *, uint32_t *> &opOutTp,
                                std::tuple<D *, uint32_t *, int> &topKHeapTp,
                                const uint32_t realSize,
                                const uint32_t indexOffset,
                                const int burstLen)
{
    D *disArray = std::get<0>(opOutTp);
    D *extremeDisArray = std::get<1>(opOutTp);
    uint32_t *idArray = std::get<2>(opOutTp); // 2 reps second element
    D *disResult = std::get<0>(topKHeapTp);
    uint32_t *idResult = std::get<1>(topKHeapTp);
    const int kVal = std::get<2>(topKHeapTp);
    bool idFlag = idArray != nullptr;

    // 2 reps skip index, only handle distance
    for (uint32_t i = 0, k = 0; i < realSize; i += burstLen, k += 2) {
        const int endIdx = std::min(i + burstLen, realSize);
        for (int j = i; j < endIdx; ++j) {
            if (compareEqual(extremeDisArray[k], disResult[0])) {
                break;
            }

            if (compareEqual(disResult[0], disArray[j])) {
                int index = idFlag ? *(idArray + j) : (indexOffset + j);
                pushHeap(kVal, disResult, idResult, disArray[j], index, compare);
            }
        }
    }

    return true;
}

template<typename T, typename E, typename D, bool ASC>
bool TopkOp<T, E, D, ASC>::exec(AscendTensor<D, 1> &distance, AscendTensor<uint32_t, 1> &indices,
    AscendTensor<D, 1> &topkDistance, AscendTensor<uint32_t, 1> &topkIndices)
{
    if (!checkParams(distance, indices, topkDistance, topkIndices)) {
        return false;
    }

    D *disArray = distance.data();
    uint32_t *idArray = indices.data();
    D *disResult = topkDistance.data();
    uint32_t *idResult = topkIndices.data();
    const int numDist = distance.getSize(0);
    const int kVal = topkDistance.getSize(0);
    bool idFlag = idArray != nullptr;

    for (int j = 0; j < numDist; j++) {
        if (compareEqual(disArray[j], disResult[0])) {
            continue;
        }
        int index = idFlag ? *(idArray + j) : j;
        pushHeap(kVal, disResult, idResult, *(disArray + j), index, compare);
    }

    return true;
}

template<typename T, typename E, typename D, bool ASC>
bool TopkOp<T, E, D, ASC>::checkParams(AscendTensor<D, 1> &distance, AscendTensor<uint32_t, 1> &indices,
    AscendTensor<D, 1> &topkDistance, AscendTensor<uint32_t, 1> &topkIndices)
{
    if (distance.data() == nullptr || topkDistance.data() == nullptr || topkIndices.data() == nullptr) {
        return false;
    }

    int outDistSize = topkDistance.getSize(0);
    int outIdSize = topkIndices.getSize(0);
    if (outIdSize != outDistSize) {
        return false;
    }

    return true;
}

template<typename T, typename E, typename D, bool ASC>
bool TopkOp<T, E, D, ASC>::checkParams(AscendTensor<D, DIMS_2> &distance, AscendTensor<uint32_t, DIMS_2> &indices,
    AscendTensor<D, DIMS_2> &topkDistance, AscendTensor<uint32_t, DIMS_2> &topkIndices)
{
    if (distance.data() == nullptr || topkDistance.data() == nullptr || topkIndices.data() == nullptr) {
        return false;
    }

    int inDistSize0 = distance.getSize(0);
    int outDistSize0 = topkDistance.getSize(0);
    int outDistSize1 = topkDistance.getSize(1);
    int outIdSize0 = topkIndices.getSize(0);
    int outIdSize1 = topkIndices.getSize(1);
    if (inDistSize0 != outDistSize0 || inDistSize0 != outIdSize0 || outIdSize1 != outDistSize1) {
        return false;
    }

    return true;
}

template class TopkOp<std::greater<>, std::greater_equal<>, float16_t>;
template class TopkOp<std::less<>, std::less_equal<>, float16_t, false>;
template class TopkOp<std::greater<>, std::greater_equal<>, int32_t>;
} // namespace ascend
