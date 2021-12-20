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

#include <algorithm>
#include <atomic>
#include <ascenddaemon/impl/IndexIVFSQ.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/TaskQueueItem.h>
#include <ascenddaemon/utils/Limits.h>

namespace ascend {
template<typename T>
IndexIVFSQ<T>::IndexIVFSQ(int numList, int dim, bool encodeResidual, int nprobes, int resourceSize)
    : IndexIVF(numList, dim, dim, nprobes, resourceSize),
      threadPool(nullptr)
{
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN == 0);

    isTrained = false;

    // supported batch size
    searchPageSizes = {64, 32, 16, 8, 4, 2, 1};

    threadPool = new AscendThreadPool(THREADS_CNT);

    for (int i = 0; i < numLists; ++i) {
        preComputeData.push_back(
            std::make_unique<DeviceVector<float>>(MemorySpace::DEVICE_HUGEPAGE));
    }
}

template<typename T>
IndexIVFSQ<T>::~IndexIVFSQ()
{
    if (threadPool) {
        delete threadPool;
    }
}

template<typename T>
void IndexIVFSQ<T>::reset()
{
    // reset the database and precomputed, but trained values is maintained.
    IndexIVF::reset();

    preComputeData.clear();
    for (int i = 0; i < numLists; ++i) {
        preComputeData.push_back(
            std::make_unique<DeviceVector<float>>(MemorySpace::DEVICE_HUGEPAGE));
    }
}

template<typename T>
void IndexIVFSQ<T>::reserveMemory(size_t numVecs)
{
    size_t numVecsPerList = utils::divUp(numVecs, static_cast<size_t>(numLists));
    if (numVecsPerList < 1) {
        return;
    }

    numVecsPerList = utils::roundUp(numVecsPerList, static_cast<size_t>(CUBE_ALIGN));
    size_t tmpLen = numVecsPerList * static_cast<size_t>(numLists);
    IndexIVF::reserveMemory(tmpLen);

    for (auto &list : preComputeData) {
        list->reserve(numVecsPerList);
    }
}

template<typename T>
void IndexIVFSQ<T>::reserveMemory(int listId, size_t numVecs)
{
    if (numVecs < 1) {
        return;
    }

    numVecs = utils::roundUp(numVecs, static_cast<size_t>(CUBE_ALIGN));
    IndexIVF::reserveMemory(listId, numVecs);

    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    preComputeData[listId]->reserve(numVecs);
}

template<typename T>
size_t IndexIVFSQ<T>::reclaimMemory()
{
    size_t totalReclaimed = IndexIVF::reclaimMemory();

    for (auto &list : preComputeData) {
        totalReclaimed += list->reclaim(true);
    }

    return totalReclaimed;
}

template<typename T>
size_t IndexIVFSQ<T>::reclaimMemory(int listId)
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    size_t totalReclaimed = IndexIVF::reclaimMemory(listId);

    totalReclaimed += preComputeData[listId]->reclaim(true);

    return totalReclaimed;
}

template<typename T>
void IndexIVFSQ<T>::addVectors(int listId, size_t numVecs, const uint8_t *codes,
    const uint32_t *indices, const float *preCompute)
{
    ASCEND_THROW_IF_NOT(this->isTrained);
    ASCEND_THROW_IF_NOT(listId >= 0 && listId < numLists);

    if (numVecs == 0) {
        return;
    }

    // code need to be Zz format because of DistanceComputeSQ8 operator's limitation.
    //       origin code for example (shape n X dim). n=16, dim = 128
    //       |  0_0  0_1  0_2  0_3 ...  0_125  0_126  0_127 |
    //       |  1_0  1_1  1_2  1_3 ...  1_125  1_126  1_127 |
    //       |        .                          .          |
    //       |        .                          .          |
    //       | 14_0 14_1 14_2 14_3 ... 14_125 14_126 14_127 |
    //       | 15_0 15_1 15_2 15_3 ... 15_125 15_126 15_127 |
    //                              | shape dims 2: (dim/16 X n/16) X (16 X 16),
    //             after Zz format    dims4: (n/16) X (dim/16) X 16 X 16
    //       |   0_0   0_1 ...  0_14  0_15   1_0   1_1 ...  1_15 ...   15_15 |
    //       |  0_16  0_17 ...  0_30  0_31  1_16  1_17 ...  1_31 ...   15_31 |
    //       |        .                    .                  .         .    |
    //       |        .                    .                  .         .    |
    //       |  0_96  0_97 ... 0_110 0_111  1_96  1_97 ... 1_111 ...  15_111 |
    //       | 0_112 0_113 ... 0_126 0_127 1_112 1_113 ... 1_127 ...  15_127 |
    // n and dim must be 16 aligned, otherwise padding data is needed.

    // 1. save codes data
    AscendTensor<uint8_t, DIMS_2> codesData(const_cast<uint8_t *>(codes), { static_cast<int>(numVecs), this->dims });
    size_t originLen = getListLength(listId);
    size_t tmpLen = utils::roundUp((originLen + numVecs), static_cast<size_t>(CUBE_ALIGN));
    deviceListData[listId]->resize(tmpLen * this->dims);

    // dims is alignd with CUBE_ALIGN, no padding data in horizontal direction
    int dimShaped = utils::divUp(this->dims, CUBE_ALIGN);

// input codes are contigous(numVecs X dims), reconstruct the codes into Zz format.
#pragma omp parallel for if (numVecs >= 100)
    for (size_t i = 0; i < numVecs; i++) {
        int seq = static_cast<int>(originLen + i);
        uint8_t *tmpData = static_cast<uint8_t *>(deviceListData[listId]->data()) + getShapedDataOffset(seq);

        for (int j = 0; j < dimShaped; j++) {
            MEMCPY_S(tmpData, CUBE_ALIGN * sizeof(uint8_t),
                codesData[i][j * CUBE_ALIGN].data(), CUBE_ALIGN * sizeof(uint8_t));
            tmpData += (CUBE_ALIGN * CUBE_ALIGN);
        }
    }

    // 2. save pre compute data if not null
    if (preCompute != nullptr) {
        preComputeData[listId]->resize(tmpLen);
        float *precompData = preComputeData[listId]->data() + originLen;
        MEMCPY_S(precompData, numVecs * sizeof(float), preCompute, numVecs * sizeof(float));
    }
    deviceListIndices[listId]->append(indices, numVecs);

    maxListLength = std::max(maxListLength, static_cast<int>(getListLength(listId)));
    maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
    this->ntotal += numVecs;
}

template<typename T>
void IndexIVFSQ<T>::updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2> &coarseCentroidsData)
{
    // update coarse centroids for L1 search.
    IndexIVF::updateCoarseCentroidsData(coarseCentroidsData);

    // isTrained need to be set when all trained values are updated.
    // if vMin has been updated, set isTrained = true
    if (this->vMin.data()) {
        this->isTrained = true;
    }
}

template<typename T>
void IndexIVFSQ<T>::updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin,
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

    // isTrained need to be set when all trained values are updated.
    // if coarseCentroids has been updated, set isTrained = true
    if (this->coarseCentroids.data()) {
        this->isTrained = true;
    }
}

template<typename T>
void IndexIVFSQ<T>::calcResiduals(AscendTensor<float16_t, DIMS_1> &query,
                                  AscendTensor<uint32_t, DIMS_1> &nprobeIndices,
                                  AscendTensor<float16_t, DIMS_2> &residulas)
{
    int dim = query.getSize(0);
    int probes = nprobeIndices.getSize(0);
    ASCEND_ASSERT(probes == this->nprobe);
    ASCEND_ASSERT(dim == this->dims);

    // query - L1 coarse centroids
    for (int probeIdx = 0; probeIdx < probes; ++probeIdx) {
        int list = nprobeIndices[probeIdx].value();
        for (int k = 0; k < dim; k++) {
            residulas[probeIdx][k] = query[k].value() - coarseCentroids[list][k].value();
        }
    };
}

template<typename T>
void IndexIVFSQ<T>::addImpl(int n, const float16_t *x, const idx_t *ids)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);
}

template<typename T>
size_t IndexIVFSQ<T>::removeIdsImpl(const IDSelector &sel)
{
    //
    // | id0 id1 id2 ... id98 id99 id100 |
    //            ^                  |
    //            |__________________|
    // if id2 is to be removed, copy the last id(id100) to
    // the mem place of id2, and decrease the list size to size-1.
    size_t removeCnt = 0;

    // dims is alignd with CUBE_ALIGN, no padding data in horizontal direction
    int dimShaped = utils::divUp(this->dims, CUBE_ALIGN);

#pragma omp parallel for reduction(+ : removeCnt)
    for (int id = 0; id < numLists; id++) {
        DeviceScope device;
        auto &indicesList = deviceListIndices[id];
        auto &codeList = deviceListData[id];
        auto &precompList = preComputeData[id];

        uint32_t *indicesPtr = indicesList->data();
        float *precompPtr = precompList->data();
        uint8_t *codePtr = static_cast<uint8_t *>(codeList->data());
        bool hasMoved = false;
        int j = indicesList->size() - 1;
        int i = 0;
        while (i <= j) {
            if (!sel.is_member((*indicesList)[i])) {
                i++;
                continue;
            }

            MEMCPY_S(indicesPtr + i, sizeof(uint32_t), indicesPtr + j, sizeof(uint32_t));
            if (precompPtr != nullptr) {
                MEMCPY_S(precompPtr + i, sizeof(float), precompPtr + j, sizeof(float));
            }

            uint8_t *src = codePtr + getShapedDataOffset(j);
            uint8_t *dst = codePtr + getShapedDataOffset(i);
            for (int k = 0; k < dimShaped; k++) {
                MEMCPY_S(dst, CUBE_ALIGN * sizeof(uint8_t), src, CUBE_ALIGN * sizeof(uint8_t));
                src += (CUBE_ALIGN * CUBE_ALIGN);
                dst += (CUBE_ALIGN * CUBE_ALIGN);
            }

            j--;
            removeCnt++;
            hasMoved = true;
        }

        // if some code has been removed, list need to be resize and reclaim memory
        if (hasMoved) {
            size_t tmpLen = utils::roundUp((j + 1), CUBE_ALIGN);
            indicesList->resize(j + 1);
            codeList->resize(tmpLen * this->dims);
            precompList->resize(tmpLen);
            indicesList->reclaim(false);
            codeList->reclaim(false);
            precompList->reclaim(false);
        }
    }

    this->ntotal -= removeCnt;
    return removeCnt;
}

template<typename T>
int IndexIVFSQ<T>::getShapedDataOffset(int idx) const
{
    int offset = this->dims * utils::roundDown(idx, CUBE_ALIGN);
    offset += (idx % CUBE_ALIGN) * CUBE_ALIGN;
    return offset;
}

template<typename T>
DeviceVector<float> &IndexIVFSQ<T>::getListPrecompute(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return *preComputeData[listId];
}

template<typename T>
bool IndexIVFSQ<T>::listVectorsNeedReshaped() const
{
    return true;
}

template<typename T>
void IndexIVFSQ<T>::getListVectorsReshaped(int listId, std::vector<unsigned char> &reshaped) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    size_t size = getListLength(listId);
    auto &data = getListVectors(listId);
    int dimShaped = utils::divUp(this->dims, CUBE_ALIGN);
    reshaped.resize(size * this->dims);

// reshape code from Zz format data to contigous format.
#pragma omp parallel for if (size >= 100)
    for (size_t i = 0; i < size; i++) {
        int offset = getShapedDataOffset(i);
        auto srcPtr = data.data() + offset;
        auto dstPtr = reshaped.data() + i * this->dims * sizeof(unsigned char);
        for (int j = 0; j < dimShaped; j++) {
            MEMCPY_S(dstPtr + j * CUBE_ALIGN,
                     CUBE_ALIGN * sizeof(unsigned char),
                     srcPtr + j * CUBE_ALIGN * CUBE_ALIGN,
                     CUBE_ALIGN * sizeof(unsigned char));
        }
    }
}

template<typename T>
void IndexIVFSQ<T>::getListVectorsReshaped(int listId, unsigned char* reshaped) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    size_t size = getListLength(listId);
    auto &data = getListVectors(listId);
    int dimShaped = utils::divUp(this->dims, CUBE_ALIGN);

// reshape code from Zz format data to contigous format.
#pragma omp parallel for if (size >= 100)
    for (size_t i = 0; i < size; i++) {
        int offset = getShapedDataOffset(i);
        auto srcPtr = data.data() + offset;
        auto dstPtr = reshaped + i * this->dims * sizeof(unsigned char);
        for (int j = 0; j < dimShaped; j++) {
            MEMCPY_S(dstPtr + j * CUBE_ALIGN,
                     CUBE_ALIGN * sizeof(unsigned char),
                     srcPtr + j * CUBE_ALIGN * CUBE_ALIGN,
                     CUBE_ALIGN * sizeof(unsigned char));
        }
    }
}

template class IndexIVFSQ<float>;
} // ascend
