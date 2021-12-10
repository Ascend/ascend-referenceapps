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

#include <set>
#include <algorithm>

#include <ascenddaemon/impl/IndexInt8Flat.h>
#include <ascenddaemon/impl/Index.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/AscendAssert.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/DeviceVectorInl.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/AscendThreadPool.h>

namespace ascend {
namespace {
const int THREADS_CNT = 6;
const int CUBE_ALIGN_SIZE = 16;
const int CUBE_ALIGN_INT8_SIZE = 32;
}

template<typename P>
IndexInt8Flat<P>::IndexInt8Flat(int dim, MetricType metric, int resourceSize) : IndexInt8(dim, metric, resourceSize)
{
    ASCEND_THROW_IF_NOT(this->dims % CUBE_ALIGN_INT8_SIZE == 0);

    // supported batch size
    this->searchPageSizes = { 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1 };

    threadPool = new AscendThreadPool(THREADS_CNT);

    // no need train for flat index
    this->isTrained = true;

    int dim1 = utils::divUp(this->distComputeBatch, CUBE_ALIGN_SIZE);
    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8_SIZE);
    this->devVecCapacity = dim1 * dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_INT8_SIZE;
}

template<typename P>
IndexInt8Flat<P>::~IndexInt8Flat()
{
    if (threadPool) {
        delete threadPool;
    }
}

template<typename P>
void IndexInt8Flat<P>::addVectors(AscendTensor<int8_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);
    int dim = rawData.getSize(1);
    ASCEND_THROW_IF_NOT(num >= 0);
    ASCEND_THROW_IF_NOT(dim == this->dims);

    if (num == 0) {
        return;
    }

    // 1. resize(malloc enough memory)
    int vecSize = utils::divUp(this->ntotal, this->distComputeBatch);
    int addVecNum = utils::divUp(this->ntotal + num, distComputeBatch) - vecSize;
    for (int i = 0; i < addVecNum; ++i) {
        this->baseShaped.emplace_back(std::make_unique<DeviceVector<int8_t>>(MemorySpace::DEVICE_HUGEPAGE));
        this->normBase.emplace_back(std::make_unique<DeviceVector<P>>());

        this->baseShaped.at(vecSize + i)->resize(this->devVecCapacity, true);
        this->normBase.at(vecSize + i)->resize(this->distComputeBatch, true);
    }

    // 2. save the rawData to shaped data
    int dim2 = utils::divUp(dim, CUBE_ALIGN_INT8_SIZE);
    for (int i = 0; i < num; i++) {
        int seq = (this->ntotal + i) % distComputeBatch;

        int offset1 = (seq / CUBE_ALIGN_SIZE) * dim2;
        int offset2 = seq % CUBE_ALIGN_SIZE;
        int offset = offset1 * (CUBE_ALIGN_SIZE * CUBE_ALIGN_INT8_SIZE) + offset2 * CUBE_ALIGN_INT8_SIZE;
        int8_t *tmpData = baseShaped[(this->ntotal + i) / distComputeBatch]->data() + offset;

        for (int j = 0; j < dim2; j++) {
            int hpadding = (j == (dim2 - 1)) ? ((j + 1) * CUBE_ALIGN_INT8_SIZE - dim) : 0;
            (void)memcpy_s(tmpData, (CUBE_ALIGN_INT8_SIZE - hpadding) * sizeof(int8_t),
                rawData[i][j * CUBE_ALIGN_INT8_SIZE].data(), (CUBE_ALIGN_INT8_SIZE - hpadding) * sizeof(int8_t));

            if (hpadding > 0) {
                (void)memset_s(tmpData + (CUBE_ALIGN_INT8_SIZE - hpadding), sizeof(int8_t) * hpadding, 0x0,
                    sizeof(int8_t) * hpadding);
            }
            tmpData += (CUBE_ALIGN_SIZE * CUBE_ALIGN_INT8_SIZE);
        }
    }
}

template<typename P>
void IndexInt8Flat<P>::getVectors(uint32_t offset, uint32_t num, std::vector<int8_t> &vectors)
{
    uint32_t actualNum;
    if (offset >= this->ntotal) {
        actualNum = 0;
    } else if (offset + num >= this->ntotal) {
        actualNum = this->ntotal - offset;
    } else {
        actualNum = num;
    }

    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8_SIZE);
    vectors.resize(actualNum * this->dims);

    int vecIdx = offset / this->distComputeBatch;
    int dVecIdx = offset % this->distComputeBatch;

    for (uint32_t i = 0; i < actualNum; i++) {
        int idx = dVecIdx++;
        if (idx == this->distComputeBatch) {
            ++vecIdx;
            dVecIdx = 0;
            idx = dVecIdx++;
        }

        int8_t *dataPtr = baseShaped[vecIdx]->data() +
            (idx / CUBE_ALIGN_SIZE) * (dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_INT8_SIZE) +
            (idx % CUBE_ALIGN_SIZE) * (CUBE_ALIGN_INT8_SIZE);
        for (int j = 0; j < dim2; j++) {
            size_t vectorsOffset = i * this->dims + j * CUBE_ALIGN_INT8_SIZE;
            size_t cpyNum = CUBE_ALIGN_INT8_SIZE;
            if (j == dim2 - 1) {
                cpyNum = this->dims - j * CUBE_ALIGN_INT8_SIZE;
            }
            if (memcpy_s(vectors.data() + vectorsOffset, (vectors.size() - vectorsOffset) * sizeof(int8_t),
                dataPtr + j * CUBE_ALIGN_SIZE * CUBE_ALIGN_INT8_SIZE, cpyNum * sizeof(int8_t)) != EOK) {
                ASCEND_THROW_FMT("memcpy_s error, (i=%d,j=%d)target buf remains %d\n", i, j,
                    (vectors.size() - vectorsOffset) * sizeof(int8_t));
            }
        }
    }
}

template<typename P>
void IndexInt8Flat<P>::addImpl(int n, const int8_t *x, const idx_t *ids)
{
    // do nothing, use addVectors instead
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);
}

template<typename P>
void IndexInt8Flat<P>::reset()
{
    int dvSize = utils::divUp(this->ntotal, this->distComputeBatch);
    for (int i = 0; i < dvSize; ++i) {
        baseShaped.at(i)->clear();
        normBase.at(i)->clear();
    }
    this->ntotal = 0;
}

// the method is not used for IndexIntFlatCos
template<typename P>
void IndexInt8Flat<P>::computeNorm(AscendTensor<int8_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);
    int dim = rawData.getSize(1);
    ASCEND_THROW_IF_NOT(num >= 0);
    ASCEND_THROW_IF_NOT(dim == this->dims);

    if (num == 0) {
        return;
    }

    bool isFirst = true;
    int idx = 0;
    for (int i = 0; i < num; i++) {
        int idx1 = (this->ntotal + i) / distComputeBatch;
        int idx2 = (this->ntotal + i) % distComputeBatch;

        // if the baseShapedSlice is full or reach the last
        if (idx2 + 1 == distComputeBatch || i == num - 1) {
            P *pNormBaseSlice = normBase[idx1]->data();

            // calc y^2 (the first time is different)
            if (isFirst) {
                ivecNormsL2sqr(pNormBaseSlice + this->ntotal % distComputeBatch, rawData[idx][0].data(), dim, i + 1);
                idx += (i + 1);
                isFirst = false;
            } else {
                ivecNormsL2sqr(pNormBaseSlice, rawData[idx][0].data(), dim, idx2 + 1);
                idx += (idx2 + 1);
            }
        }
    }
}

template<typename P>
P IndexInt8Flat<P>::ivecNormL2sqr(const int8_t *x, size_t d)
{
    size_t i;
    P res = 0;
    for (i = 0; i < d; i++) {
        res += x[i] * x[i];
    }
    return res;
}

template<typename P>
void IndexInt8Flat<P>::ivecNormsL2sqr(P *nr, const int8_t *x, size_t d, size_t nx)
{
#pragma omp parallel for
    for (size_t i = 0; i < nx; i++) {
        nr[i] = static_cast<P>(ivecNormL2sqr(x + i * d, d));
    }
}

template<typename P>
void IndexInt8Flat<P>::moveNormForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_ASSERT(srcIdx >= dstIdx);
    int srcIdx1 = srcIdx / this->distComputeBatch;
    int srcIdx2 = srcIdx % this->distComputeBatch;
    int dstIdx1 = dstIdx / this->distComputeBatch;
    int dstIdx2 = dstIdx % this->distComputeBatch;

    (*normBase[dstIdx1])[dstIdx2] = (*normBase[srcIdx1])[srcIdx2];
}

template<typename P>
void IndexInt8Flat<P>::moveShapedForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_ASSERT(srcIdx >= dstIdx);
    int srcIdx1 = srcIdx / this->distComputeBatch;
    int srcIdx2 = srcIdx % this->distComputeBatch;

    int dstIdx1 = dstIdx / this->distComputeBatch;
    int dstIdx2 = dstIdx % this->distComputeBatch;

    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8_SIZE);

    int8_t *srcDataPtr = baseShaped[srcIdx1]->data() +
        (srcIdx2 / CUBE_ALIGN_SIZE) * (dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_INT8_SIZE) +
        (srcIdx2 % CUBE_ALIGN_SIZE) * (CUBE_ALIGN_INT8_SIZE);
    int8_t *dstDataPtr = baseShaped[dstIdx1]->data() +
        (dstIdx2 / CUBE_ALIGN_SIZE) * (dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_INT8_SIZE) +
        (dstIdx2 % CUBE_ALIGN_SIZE) * (CUBE_ALIGN_INT8_SIZE);

    for (int i = 0; i < dim2; i++) {
        auto err_ = memcpy_s(dstDataPtr, CUBE_ALIGN_INT8_SIZE * sizeof(int8_t), srcDataPtr,
            CUBE_ALIGN_INT8_SIZE * sizeof(int8_t));
        ASCEND_THROW_IF_NOT_FMT(err_ == EOK, "Memcpy error %d", static_cast<int>(err_));
        dstDataPtr += CUBE_ALIGN_SIZE * CUBE_ALIGN_INT8_SIZE;
        srcDataPtr += CUBE_ALIGN_SIZE * CUBE_ALIGN_INT8_SIZE;
    }
}

template<typename P>
size_t IndexInt8Flat<P>::removeIdsBatch(const std::vector<idx_t> &indices)
{
    // 1. filter the same id
    std::set<idx_t> filtered;
    for (auto idx : indices) {
        if (idx < this->ntotal) {
            filtered.insert(idx);
        }
    }

    // 2. sort by DESC, then delete from the big to small
    std::vector<idx_t> sortData(filtered.begin(), filtered.end());
    std::sort(sortData.begin(), sortData.end(), std::greater<idx_t>());

    // 3. move the end data to the locate of delete data
    idx_t oldTotal = this->ntotal;
    for (const auto index : sortData) {
        moveVectorForward(this->ntotal - 1, index);
        --this->ntotal;
    }

    // 4. release the space of unusage
    size_t removedCnt = filtered.size();
    removeInvalidData(oldTotal, removedCnt);

    return removedCnt;
}

template<typename P>
size_t IndexInt8Flat<P>::removeIdsRange(idx_t min, idx_t max)
{
    // 1. check param
    if (min >= max || min >= this->ntotal) {
        return 0;
    }

    if (max > this->ntotal) {
        max = this->ntotal;
    }

    // 2. move the end data to the locate of delete data(delete from back to front)
    int oldTotal = this->ntotal;
    for (idx_t i = 1; i <= max - min; ++i) {
        moveVectorForward(this->ntotal - 1, max - i);
        --this->ntotal;
    }

    // 3. release the space of unusage
    size_t removeCnt = max - min;
    removeInvalidData(oldTotal, removeCnt);

    return removeCnt;
}

template<typename P>
size_t IndexInt8Flat<P>::removeIdsImpl(const ascend::IDSelector &sel)
{
    size_t removeCnt = 0;

    try {
        const ascend::IDSelectorBatch &batch = dynamic_cast<const ascend::IDSelectorBatch &>(sel);
        std::vector<idx_t> buf(batch.set.begin(), batch.set.end());
        removeCnt = removeIdsBatch(buf);
    } catch (std::bad_cast &e) {
        // ignore
    }

    try {
        const ascend::IDSelectorRange &range = dynamic_cast<const ascend::IDSelectorRange &>(sel);
        removeCnt = removeIdsRange(range.imin, range.imax);
    } catch (std::bad_cast &e) {
        // ignore
    }

    return removeCnt;
}

template<typename P>
void IndexInt8Flat<P>::removeInvalidData(int oldTotal, int remove)
{
    int oldVecSize = utils::divUp(oldTotal, this->distComputeBatch);
    int vecSize = utils::divUp(oldTotal - remove, this->distComputeBatch);

    for (int i = oldVecSize - 1; i >= vecSize; --i) {
        this->baseShaped.at(i)->clear();
        this->normBase.at(i)->clear();
    }
}

template<typename P>
size_t IndexInt8Flat<P>::calcShapedBaseSize(idx_t totalNum)
{
    int numBatch = utils::divUp(totalNum, distComputeBatch);
    int dim1 = utils::divUp(distComputeBatch, CUBE_ALIGN_SIZE);
    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8_SIZE);
    return numBatch * (dim1 * dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_INT8_SIZE);
}

template<typename P>
size_t IndexInt8Flat<P>::calcNormBaseSize(idx_t totalNum)
{
    int numBatch = utils::divUp(totalNum, distComputeBatch);
    return numBatch * distComputeBatch;
}

template class IndexInt8Flat<int32_t>;
template class IndexInt8Flat<float16_t>;
} // namespace ascend
