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

#include <ascenddaemon/impl/IndexSQ.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/Limits.h>
#include <ascenddaemon/utils/StaticUtils.h>

namespace ascend {
namespace {
const int SQ8_DIST_COMPUTE_BATCH = 16384 * 16;
const int TIMEOUT_CHECK_TICK = 5120;
const double TIMEOUT_MS = 50000;
const int FLAG_ALIGN_SIZE = 32;
const int FLAG_ALIGN_OFFSET = 16; // core 0 use first 16 flag, and core 1 use the second 16 flag.
const int SIZE_ALIGN_SIZE = 8;
const int THREADS_CNT = 6;
const int BURST_LEN = 64;
}

IndexSQ::IndexSQ(int dim, int resourceSize)
    : Index(dim, resourceSize), distComputeBatch(SQ8_DIST_COMPUTE_BATCH), devVecCapacity(0), threadPool(nullptr)
{
    ASCEND_THROW_IF_NOT(dims % CUBE_ALIGN_SIZE == 0);

    isTrained = false;

    // supported batch size
    searchPageSizes = { 96, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1 };
    // Double the BURST_LEN after round up, hence here we multiply 2
    this->burstsOfComputeBatch = (this->distComputeBatch + BURST_LEN - 1) / BURST_LEN * 2;

    threadPool = new AscendThreadPool(THREADS_CNT);

    reset();

    int dim1 = utils::divUp(this->distComputeBatch, CUBE_ALIGN_SIZE);
    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_SIZE);
    this->devVecCapacity = dim1 * dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE;
}

IndexSQ::~IndexSQ()
{
    if (threadPool) {
        delete threadPool;
    }
}

void IndexSQ::reset()
{
    this->codes.clear();
    this->preCompute.clear();
    this->ntotal = 0;
}

void IndexSQ::addVectors(size_t numVecs, const uint8_t *data, const float *preCompute)
{
    if (numVecs == 0) {
        return;
    }

    int newVecSize = utils::divUp(this->ntotal + numVecs, this->distComputeBatch);
    int vecSize = utils::divUp(this->ntotal, this->distComputeBatch);
    ASCEND_ASSERT(vecSize == static_cast<int>(codes.size()));

    // 1. resize DeviceVector to save the data x
    for (int i = 0; i < newVecSize - vecSize; ++i) {
        this->codes.emplace_back(std::make_unique<DeviceVector<uint8_t>>(MemorySpace::DEVICE_HUGEPAGE));
        this->preCompute.emplace_back(std::make_unique<DeviceVector<float>>());

        this->codes.at(vecSize + i)->resize(this->devVecCapacity, true);
        this->preCompute.at(vecSize + i)->resize(this->distComputeBatch, true);
    }

    // 2. save codes
    int vecIndex = this->ntotal / this->distComputeBatch;
    int backDVecSize = this->ntotal % this->distComputeBatch;
    int dVecIndex = backDVecSize;
    saveCodes(vecIndex, dVecIndex, numVecs, data);

    // 3. save preCompute if not null
    if (preCompute != nullptr) {
        savePreCompute(vecIndex, dVecIndex, numVecs, preCompute);
    }

    // 4. modify ntotal
    this->ntotal += numVecs;
}

void IndexSQ::saveCodes(int vecIndex, int dVecIndex, int numVecs, const uint8_t *data)
{
    ASCEND_ASSERT(static_cast<int>(this->codes.size()) > vecIndex);
    ASCEND_ASSERT(this->distComputeBatch > dVecIndex);

    int cubeDim = utils::divUp(this->dims, CUBE_ALIGN_SIZE);

    AscendTensor<uint8_t, DIMS_2> dataTensor(const_cast<uint8_t *>(data), { numVecs, this->dims });
    for (int i = 0; i < numVecs; ++i, ++dVecIndex) {
        if (dVecIndex >= this->distComputeBatch) {
            ++vecIndex;
            dVecIndex = 0;
        }

        DeviceVector<uint8_t> *codesSlice = this->codes[vecIndex].get();

        int offset1 = (dVecIndex / CUBE_ALIGN_SIZE) * cubeDim;
        int offset2 = dVecIndex % CUBE_ALIGN_SIZE;
        int offset = offset1 * (CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE) + offset2 * CUBE_ALIGN_SIZE;
        uint8_t *tmpData = codesSlice->data() + offset;

        for (int j = 0; j < cubeDim; ++j) {
            int hpadding = (j == (cubeDim - 1)) ? ((j + 1) * CUBE_ALIGN_SIZE - dims) : 0;
            MEMCPY_S(tmpData, (CUBE_ALIGN_SIZE - hpadding) * sizeof(uint8_t), dataTensor[i][j * CUBE_ALIGN_SIZE].data(),
                (CUBE_ALIGN_SIZE - hpadding) * sizeof(uint8_t));

            if (hpadding > 0) {
                MEMCPY_S(tmpData + (CUBE_ALIGN_SIZE - hpadding), sizeof(uint8_t) * hpadding, 0x0,
                    sizeof(uint8_t) * hpadding);
            }
            tmpData += (CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE);
        }
    }
}

void IndexSQ::savePreCompute(int vecIndex, int dVecIndex, int numVecs, const float *preCompute)
{
    ASCEND_ASSERT(static_cast<int>(this->preCompute.size()) > vecIndex);
    ASCEND_ASSERT(this->distComputeBatch > dVecIndex);

    for (int i = 0; i < numVecs; ++i, ++dVecIndex, ++preCompute) {
        if (dVecIndex >= this->distComputeBatch) {
            ++vecIndex;
            dVecIndex = 0;
        }

        (*this->preCompute[vecIndex])[dVecIndex] = *preCompute;
    }
}

void IndexSQ::getVectors(uint32_t offset, uint32_t num, std::vector<uint8_t> &vectors)
{
    if (this->ntotal == 0) {
        return;
    }

    // 1. check param
    uint32_t actualNum;
    if (offset >= this->ntotal) {
        actualNum = 0;
    } else if (offset + num >= this->ntotal) {
        actualNum = this->ntotal - offset;
    } else {
        actualNum = num;
    }

    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_SIZE);
    vectors.resize(actualNum * this->dims);

    int vecIndex = offset / this->distComputeBatch;
    int dVecIndex = offset % this->distComputeBatch;

    ASCEND_ASSERT(static_cast<int>(this->codes.size()) > vecIndex);

    for (uint32_t i = 0; i < actualNum; i++) {
        int idx = dVecIndex++;
        if (idx == this->distComputeBatch) {
            ++vecIndex;
            dVecIndex = 0;
            idx = dVecIndex++;
        }

        uint8_t *dataPtr = this->codes[vecIndex]->data() +
            (idx / CUBE_ALIGN_SIZE) * (dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE) +
            (idx % CUBE_ALIGN_SIZE) * (CUBE_ALIGN_SIZE);
        for (int j = 0; j < dim2; j++) {
            size_t vectorsOffset = i * dims + j * CUBE_ALIGN_SIZE;
            size_t cpyNum = CUBE_ALIGN_SIZE;
            if (j == dim2 - 1) {
                cpyNum = dims - j * CUBE_ALIGN_SIZE;
            }
            if (EOK != memcpy_s(vectors.data() + vectorsOffset, (vectors.size() - vectorsOffset) * sizeof(uint8_t),
                dataPtr + j * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE, cpyNum * sizeof(uint8_t))) {
                ASCEND_THROW_FMT("memcpy_s error, (i=%d,j=%d)target buf remains %d\n", i, j,
                    (vectors.size() - vectorsOffset) * sizeof(uint8_t));
            }
        }
    }
}

void IndexSQ::getVectors(uint32_t offset, uint32_t num, uint8_t *vectors)
{
    if (this->ntotal == 0 || offset >= this->ntotal) {
        return;
    }

    // check param
    uint32_t actualNum = (offset + num >= this->ntotal) ? (this->ntotal - offset) : num;
    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_SIZE);
    size_t actualSize = actualNum * this->dims;

#pragma omp parallel for if (actualNum >= 100)
    for (uint32_t i = 0; i < actualNum; i++) {
        int idx = (offset + i) % this->distComputeBatch;
        int vecIndex = (offset + i) / this->distComputeBatch;

        uint8_t *dataPtr = this->codes[vecIndex]->data() +
            (idx / CUBE_ALIGN_SIZE) * (dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE) +
            (idx % CUBE_ALIGN_SIZE) * (CUBE_ALIGN_SIZE);
        for (int j = 0; j < dim2; j++) {
            size_t vectorsOffset = i * dims + j * CUBE_ALIGN_SIZE;
            size_t cpyNum = CUBE_ALIGN_SIZE;
            if (j == dim2 - 1) {
                cpyNum = dims - j * CUBE_ALIGN_SIZE;
            }
            if (EOK != memcpy_s(vectors + vectorsOffset, (actualSize - vectorsOffset) * sizeof(uint8_t),
                dataPtr + j * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE, cpyNum * sizeof(uint8_t))) {
                ASCEND_THROW_FMT("memcpy_s error, (i=%d,j=%d)target buf remains %d\n", i, j,
                    (actualSize - vectorsOffset) * sizeof(uint8_t));
            }
        }
    }
}

void IndexSQ::updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin,
    AscendTensor<float16_t, DIMS_1> &trainedDiff)
{
    int dimMin = trainedMin.getSize(0);
    int dimDiff = trainedDiff.getSize(0);
    ASCEND_THROW_IF_NOT_FMT(dimMin == dimDiff && dimMin == this->dims,
        "sq trained data's shape invalid.(%d, %d) vs (dim:%d)", dimMin, dimDiff, this->dims);

    AscendTensor<float16_t, DIMS_1> minTensor({ dimMin });
    AscendTensor<float16_t, DIMS_1> diffTensor({ dimDiff });
    minTensor.copyFromSync(trainedMin);
    diffTensor.copyFromSync(trainedDiff);
    vMin = std::move(minTensor);
    vDiff = std::move(diffTensor);

    this->isTrained = true;
}

void IndexSQ::addImpl(int n, const float16_t *x, const idx_t *ids)
{
    // do nothing, use addVectors instead
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);
}

size_t IndexSQ::removeIdsImpl(const IDSelector &sel)
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

void IndexSQ::moveCodesForward(idx_t srcIdx, idx_t destIdx)
{
    ASCEND_ASSERT(srcIdx >= destIdx);
    if (srcIdx == destIdx) {
        return;
    }

    int srcVecIndex = srcIdx / this->distComputeBatch;
    int srcDVecIndex = srcIdx % this->distComputeBatch;
    int destVecIndex = destIdx / this->distComputeBatch;
    int destDVecIndex = destIdx % this->distComputeBatch;
    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_SIZE);

    ASCEND_ASSERT(static_cast<int>(this->codes.size()) > destVecIndex);
    ASCEND_ASSERT(static_cast<int>(this->codes.size()) > srcVecIndex);

    uint8_t *srcDataPtr = this->codes[srcVecIndex]->data() +
        (srcDVecIndex / CUBE_ALIGN_SIZE) * (dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE) +
        (srcDVecIndex % CUBE_ALIGN_SIZE) * (CUBE_ALIGN_SIZE);
    uint8_t *destDataPtr = this->codes[destVecIndex]->data() +
        (destDVecIndex / CUBE_ALIGN_SIZE) * (dim2 * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE) +
        (destDVecIndex % CUBE_ALIGN_SIZE) * (CUBE_ALIGN_SIZE);

    for (int i = 0; i < dim2; i++) {
        MEMCPY_S(destDataPtr, CUBE_ALIGN_SIZE * sizeof(uint8_t), srcDataPtr, CUBE_ALIGN_SIZE * sizeof(uint8_t));
        destDataPtr += CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE;
        srcDataPtr += CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE;
    }
}

void IndexSQ::movePreComputeForward(idx_t srcIdx, idx_t destIdx)
{
    ASCEND_ASSERT(srcIdx >= destIdx);
    if (srcIdx == destIdx) {
        return;
    }

    int srcVecIndex = srcIdx / this->distComputeBatch;
    int srcDVecIndex = srcIdx % this->distComputeBatch;
    int destVecIndex = destIdx / this->distComputeBatch;
    int destDVecIndex = destIdx % this->distComputeBatch;

    ASCEND_ASSERT(static_cast<int>(this->preCompute.size()) > destVecIndex);
    ASCEND_ASSERT(static_cast<int>(this->preCompute.size()) > srcVecIndex);

    (*this->preCompute[destVecIndex])[destDVecIndex] = (*this->preCompute[srcVecIndex])[srcDVecIndex];
}

void IndexSQ::releaseUnusageSpace(int oldTotal, int remove)
{
    int oldVecSize = utils::divUp(oldTotal, this->distComputeBatch);
    int vecSize = utils::divUp(oldTotal - remove, this->distComputeBatch);

    for (int i = 0; i < oldVecSize - vecSize; ++i) {
        this->codes.pop_back();
        this->preCompute.pop_back();
    }
}

size_t IndexSQ::removeIdsRange(idx_t min, idx_t max)
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
    releaseUnusageSpace(oldTotal, removeCnt);

    return removeCnt;
}

size_t IndexSQ::removeIdsBatch(const std::vector<idx_t> &indices)
{
    idx_t oldTotal = this->ntotal;
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
    for (const auto index : sortData) {
        moveVectorForward(this->ntotal - 1, index);
        --this->ntotal;
    }

    size_t removedCnt = filtered.size();
    // 4. release the space of unusage
    releaseUnusageSpace(oldTotal, removedCnt);

    return removedCnt;
}
} // ascend