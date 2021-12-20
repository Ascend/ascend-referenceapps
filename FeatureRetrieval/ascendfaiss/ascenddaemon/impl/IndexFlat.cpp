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


#include <ascenddaemon/impl/IndexFlat.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/DeviceVectorInl.h>
#include <ascenddaemon/utils/Limits.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/AscendThreadPool.h>

namespace ascend {
namespace {
const int DEFAULT_DIST_COMPUTE_BATCH = 16384 * 16;
const double TIMEOUT_MS = 50000;
const int TIMEOUT_CHECK_TICK = 5120;
const int FLAG_ALIGN_SIZE = 32;
const int FLAG_ALIGN_OFFSET = 16; // core 0 use first 16 flag, and core 1 use the second 16 flag.
const int SIZE_ALIGN_SIZE = 8;
const int THREADS_CNT = 6;
const int BURST_LEN = 64;
}

IndexFlat::IndexFlat(int dim, int resourceSize) : Index(dim, resourceSize), distComputeBatch(DEFAULT_DIST_COMPUTE_BATCH)
{
    ASCEND_THROW_IF_NOT(dims % CUBE_ALIGN_SIZE == 0);

    // IndexFlat does not need training
    isTrained = true;

    // supported batch size
    searchPageSizes = { 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1 };
    // Double the BURST_LEN after round up, hence here we multiply 2
    this->burstsOfComputeBatch = (this->distComputeBatch + BURST_LEN - 1) / BURST_LEN * 2;

    threadPool = new AscendThreadPool(THREADS_CNT);

    int dim1 = utils::divUp(distComputeBatch, CUBE_ALIGN_SIZE);
    int dim2 = utils::divUp(dims, CUBE_ALIGN_SIZE);
    this->devVecCapacity = dim1 * dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE;
}

IndexFlat::~IndexFlat()
{
    if (threadPool) {
        delete threadPool;
    }
}

void IndexFlat::addImpl(int n, const float16_t *x, const idx_t *ids)
{
    // do nothing, use addVectors instead
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);
}

size_t IndexFlat::calcShapedBaseSize(idx_t totalNum)
{
    int numBatch = utils::divUp(totalNum, distComputeBatch);
    int dim1 = utils::divUp(distComputeBatch, CUBE_ALIGN_SIZE);
    int dim2 = utils::divUp(dims, CUBE_ALIGN_SIZE);
    return numBatch * (dim1 * dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE);
}

void IndexFlat::addVectors(AscendTensor<float16_t, DIMS_2> &rawData)
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
    int addVecNum = utils::divUp(ntotal + num, distComputeBatch) - vecSize;
    for (int i = 0; i < addVecNum; ++i) {
        this->baseShaped.emplace_back(std::make_unique<DeviceVector<float16_t>>(MemorySpace::DEVICE_HUGEPAGE));
        this->baseShaped.at(vecSize + i)->resize(this->devVecCapacity, true);
    }

    // 2. save the rawData to shaped data
    int dim2 = utils::divUp(dim, CUBE_ALIGN_SIZE);
    for (int i = 0; i < num; i++) {
        int seq = (ntotal + i) % distComputeBatch;

        int offset1 = (seq / CUBE_ALIGN_SIZE) * dim2;
        int offset2 = seq % CUBE_ALIGN_SIZE;
        int offset = offset1 * (CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE) + offset2 * CUBE_ALIGN_SIZE;
        float16_t *tmpData = baseShaped[(ntotal + i) / distComputeBatch]->data() + offset;

        for (int j = 0; j < dim2; j++) {
            int hpadding = (j == (dim2 - 1)) ? ((j + 1) * CUBE_ALIGN_SIZE - dim) : 0;
            (void)memcpy_s(tmpData, (CUBE_ALIGN_SIZE - hpadding) * sizeof(float16_t),
                rawData[i][j * CUBE_ALIGN_SIZE].data(), (CUBE_ALIGN_SIZE - hpadding) * sizeof(float16_t));

            if (hpadding > 0) {
                (void)memset_s(tmpData + (CUBE_ALIGN_SIZE - hpadding), sizeof(float16_t) * hpadding, 0x0,
                    sizeof(float16_t) * hpadding);
            }
            tmpData += (CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE);
        }
    }
}

void IndexFlat::getVectors(uint32_t offset, uint32_t num, std::vector<float16_t> &vectors)
{
    uint32_t actualNum;
    if (offset >= ntotal) {
        actualNum = 0;
    } else if (offset + num >= ntotal) {
        actualNum = ntotal - offset;
    } else {
        actualNum = num;
    }

    int dim2 = utils::divUp(dims, CUBE_ALIGN_SIZE);
    vectors.resize(actualNum * dims);

    int vecIdx = offset / this->distComputeBatch;
    int dVecIdx = offset % this->distComputeBatch;

    for (uint32_t i = 0; i < actualNum; i++) {
        int idx = dVecIdx++;
        if (idx == this->distComputeBatch) {
            ++vecIdx;
            dVecIdx = 0;
            idx = dVecIdx++;
        }

        float16_t *dataPtr = baseShaped[vecIdx]->data() +
            (idx / CUBE_ALIGN_SIZE) * (dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE) +
            (idx % CUBE_ALIGN_SIZE) * (CUBE_ALIGN_SIZE);
        for (int j = 0; j < dim2; j++) {
            size_t vectorsOffset = i * dims + j * CUBE_ALIGN_SIZE;
            size_t cpyNum = CUBE_ALIGN_SIZE;
            if (j == dim2 - 1) {
                cpyNum = dims - j * CUBE_ALIGN_SIZE;
            }
            if (EOK != memcpy_s(vectors.data() + vectorsOffset, (vectors.size() - vectorsOffset) * sizeof(float16_t),
                dataPtr + j * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE, cpyNum * sizeof(float16_t))) {
                ASCEND_THROW_FMT("memcpy_s error, (i=%d,j=%d)target buf remains %d\n", i, j,
                    (vectors.size() - vectorsOffset) * sizeof(float16_t));
            }
        }
    }
}

void IndexFlat::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
{
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, dims });
    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<idx_t, DIMS_2> outIndices(labels, { n, k });
    outIndices.initValue(std::numeric_limits<idx_t>::max());
    return searchImpl(queries, k, outDistances, outIndices);
}

void IndexFlat::moveShapedForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_ASSERT(srcIdx >= dstIdx);
    int srcIdx1 = srcIdx / this->distComputeBatch;
    int srcIdx2 = srcIdx % this->distComputeBatch;

    int dstIdx1 = dstIdx / this->distComputeBatch;
    int dstIdx2 = dstIdx % this->distComputeBatch;

    int dim2 = utils::divUp(dims, CUBE_ALIGN_SIZE);

    float16_t *srcDataPtr = baseShaped[srcIdx1]->data() +
        (srcIdx2 / CUBE_ALIGN_SIZE) * (dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE) +
        (srcIdx2 % CUBE_ALIGN_SIZE) * (CUBE_ALIGN_SIZE);
    float16_t *dstDataPtr = baseShaped[dstIdx1]->data() +
        (dstIdx2 / CUBE_ALIGN_SIZE) * (dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE) +
        (dstIdx2 % CUBE_ALIGN_SIZE) * (CUBE_ALIGN_SIZE);

    for (int i = 0; i < dim2; i++) {
        (void)memcpy_s(dstDataPtr, CUBE_ALIGN_SIZE * sizeof(float16_t), srcDataPtr,
            CUBE_ALIGN_SIZE * sizeof(float16_t));
        dstDataPtr += CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE;
        srcDataPtr += CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE;
    }
}

size_t IndexFlat::removeIdsBatch(const std::vector<idx_t> &indices)
{
    // 1. filter the same id
    std::set<idx_t> filtered;
    for (auto idx : indices) {
        if (idx < ntotal) {
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

size_t IndexFlat::removeIdsRange(idx_t min, idx_t max)
{
    // 1. check param
    if (min >= max || min >= ntotal) {
        return 0;
    }

    if (max > ntotal) {
        max = ntotal;
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

size_t IndexFlat::removeIdsImpl(const ascend::IDSelector &sel)
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

void IndexFlat::removeInvalidData(int oldTotal, int remove)
{
    int oldVecSize = utils::divUp(oldTotal, this->distComputeBatch);
    int vecSize = utils::divUp(oldTotal - remove, this->distComputeBatch);

    for (int i = oldVecSize - 1; i >= vecSize; --i) {
        this->baseShaped.at(i)->clear();
    }
}

void IndexFlat::reset()
{
    int dvSize = utils::divUp(this->ntotal, this->distComputeBatch);
    for (int i = 0; i < dvSize; ++i) {
        baseShaped.at(i)->clear();
    }
    ntotal = 0;
}
} // namespace ascend
