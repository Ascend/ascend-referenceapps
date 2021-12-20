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
#include <ascenddaemon/impl/IndexInt8IVFFlat.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/Limits.h>

namespace ascend {
namespace {
const int CUBE_ALIGN = 16;
const int CUBE_ALIGN_INT8 = 32;
const int THREADS_CNT = 4;
const int INT8IVF_DEFAULT_MEM = static_cast<int>(0x18000000);
}

template<typename P>
IndexInt8IVFFlat<P>::IndexInt8IVFFlat(int numList, int dim, int nprobes, MetricType metric, int resourceSize)
    : IndexInt8IVF(numList, dim, dim, nprobes, metric, INT8IVF_DEFAULT_MEM), threadPool(nullptr)
{
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN_INT8 == 0);

    this->isTrained = false;

    // supported batch size
    this->searchPageSizes = { 64, 32, 16, 8, 4, 2, 1 };

    threadPool = new AscendThreadPool(THREADS_CNT);

    for (int i = 0; i < this->numLists; ++i) {
        preComputeData.push_back(std::make_unique<DeviceVector<P>>(MemorySpace::DEVICE_HUGEPAGE));
    }
}

template<typename P>
IndexInt8IVFFlat<P>::~IndexInt8IVFFlat()
{
    if (threadPool) {
        delete threadPool;
    }
}

template<typename P>
void IndexInt8IVFFlat<P>::reset()
{
    // reset the database and precomputed, but trained values is maintained.
    IndexInt8IVF::reset();

    preComputeData.clear();
    for (int i = 0; i < this->numLists; ++i) {
        preComputeData.push_back(std::make_unique<DeviceVector<P>>(MemorySpace::DEVICE_HUGEPAGE));
    }
}

template<typename P>
void IndexInt8IVFFlat<P>::reserveMemory(size_t numVecs)
{
    size_t numVecsPerList = utils::divUp(numVecs, static_cast<size_t>(this->numLists));
    if (numVecsPerList < 1) {
        return;
    }

    numVecsPerList = utils::roundUp(numVecsPerList, static_cast<size_t>(CUBE_ALIGN));
    size_t tmpLen = numVecsPerList * static_cast<size_t>(this->numLists);
    IndexInt8IVF::reserveMemory(tmpLen);

    for (auto &list : preComputeData) {
        list->reserve(numVecsPerList);
    }
}

template<typename P>
void IndexInt8IVFFlat<P>::reserveMemory(int listId, size_t numVecs)
{
    if (numVecs < 1) {
        return;
    }

    numVecs = utils::roundUp(numVecs, static_cast<size_t>(CUBE_ALIGN));
    IndexInt8IVF::reserveMemory(listId, numVecs);

    ASCEND_THROW_IF_NOT((listId < this->numLists) && (listId >= 0));
    preComputeData[listId]->reserve(numVecs);
}

template<typename P>
size_t IndexInt8IVFFlat<P>::reclaimMemory()
{
    size_t totalReclaimed = 0;

    for (auto &list : preComputeData) {
        totalReclaimed += list->reclaim(true);
    }

    return totalReclaimed;
}

template<typename P>
size_t IndexInt8IVFFlat<P>::reclaimMemory(int listId)
{
    ASCEND_THROW_IF_NOT((listId < this->numLists) && (listId >= 0));

    size_t totalReclaimed = IndexInt8IVF::reclaimMemory(listId);

    totalReclaimed += preComputeData[listId]->reclaim(true);

    return totalReclaimed;
}

template<typename P>
void IndexInt8IVFFlat<P>::addVectors(int listId, size_t numVecs, const int8_t *codes, const uint32_t *indices)
{
    // coarse centroids need to be Zz format because of DistanceCompute operator's limitation.
    // origin code for example (shape n X dim). n=15, dim = 127. n and dim need to be 16 aligned,
    //    n aligned = 16, dim aligned = 128, the space for aligned need to been padded to 0x00
    //    |  0_0  0_1  0_2  0_3 ...  0_125  0_126 0x00 |
    //    |  1_0  1_1  1_2  1_3 ...  1_125  1_126 0x00 |
    //    |        .                          .        |
    //    |        .                          .        |
    //    | 14_0 14_1 14_2 14_3 ... 14_125 14_126 0x00 |
    //    | 0x00 0x00 0x00 0x00 ...   0x00   0x00 0x00 |
    //                          |
    //    after Zz format  (shape dims 2: n X dim, dims4: (n/16) X (dim/32) X 16 X 32)
    //    |   0_0   0_1 ...  0_30  0_31  1_0   1_1  ...  1_30  1_31  ...  15_30  15_31 |
    //    |  0_32  0_33 ...  0_62  0_63  1_32  1_33 ...  1_62  1_63  ...  15_62  15_63 |
    //    |  0_64  0_65 ...  0_94  0_95  1_64  1_65 ...  1_94  1_95  ...  15_94  15_95 |
    //    |  0_96  0_97 ... 0_126  0x00  1_96  1_97 ... 1_126  0x00  ... 15_126   0x00 |
    // n and dim must be 16 aligned, otherwise padding data is needed.

    // 1. resize the list data
    AscendTensor<int8_t, DIMS_2> codesData(const_cast<int8_t *>(codes), { static_cast<int>(numVecs), this->dims });
    size_t originLen = this->getListLength(listId);
    size_t tmpLen = utils::roundUp((originLen + numVecs), static_cast<size_t>(CUBE_ALIGN));
    this->deviceListData[listId]->resize(tmpLen * this->dims);
    // dims is alignd with CUBE_ALIGN_SIZE, no padding data in horizontal direction
    int dimShaped = utils::divUp(this->dims, CUBE_ALIGN_INT8);

    // 2. save code data. input codes are contigous(numVecs X dims), reconstruct the codes into Zz format.
#pragma omp parallel for if (numVecs >= 100)
    for (size_t i = 0; i < numVecs; i++) {
        int seq = static_cast<int>(originLen + i);
        int8_t *tmpData = static_cast<int8_t *>(this->deviceListData[listId]->data()) + getShapedDataOffset(seq);

        for (int j = 0; j < dimShaped; j++) {
            MEMCPY_S(tmpData, CUBE_ALIGN_INT8 * sizeof(int8_t), codesData[i][j * CUBE_ALIGN_INT8].data(),
                CUBE_ALIGN_INT8 * sizeof(int8_t));
            tmpData += (CUBE_ALIGN * CUBE_ALIGN_INT8);
        }
    }
    this->deviceListIndices[listId]->append(indices, numVecs);

    this->maxListLength = std::max(this->maxListLength, static_cast<int>(this->getListLength(listId)));
    this->maxListLength = utils::roundUp(this->maxListLength, CUBE_ALIGN);
    this->ntotal += numVecs;
}

template<typename P>
void IndexInt8IVFFlat<P>::addImpl(int n, const int8_t *x, const idx_t *ids)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);
}

template<typename P>
size_t IndexInt8IVFFlat<P>::removeIdsImpl(const IDSelector &sel)
{
    //
    // | id0 id1 id2 ... id98 id99 id100 |
    //            ^                  |
    //            |__________________|
    // if id2 is to be removed, copy the last id(id100) to
    // the mem place of id2, and decrease the list size to size-1.
    size_t removeCnt = 0;

    // dims is alignd with CUBE_ALIGN_SIZE, no padding data in horizontal direction
    int dimShaped = utils::divUp(this->dims, CUBE_ALIGN_INT8);

#pragma omp parallel for reduction(+ : removeCnt)
    for (int id = 0; id < this->numLists; id++) {
        DeviceScope device;
        auto &indicesList = this->deviceListIndices[id];
        auto &codeList = this->deviceListData[id];
        auto &precompList = preComputeData[id];

        uint32_t *indicesPtr = indicesList->data();
        P *precompPtr = precompList->data();
        int8_t *codePtr = static_cast<int8_t *>(codeList->data());
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
                MEMCPY_S(precompPtr + i, sizeof(P), precompPtr + j, sizeof(P));
            }

            int8_t *src = codePtr + getShapedDataOffset(j);
            int8_t *dst = codePtr + getShapedDataOffset(i);
            for (int k = 0; k < dimShaped; k++) {
                MEMCPY_S(dst, CUBE_ALIGN_INT8 * sizeof(int8_t), src, CUBE_ALIGN_INT8 * sizeof(int8_t));
                src += (CUBE_ALIGN * CUBE_ALIGN_INT8);
                dst += (CUBE_ALIGN * CUBE_ALIGN_INT8);
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

template<typename P>
int IndexInt8IVFFlat<P>::getShapedDataOffset(int idx) const
{
    int offset = this->dims * utils::roundDown(idx, CUBE_ALIGN);
    offset += (idx % CUBE_ALIGN) * CUBE_ALIGN_INT8;
    return offset;
}

template<typename P>
DeviceVector<P>& IndexInt8IVFFlat<P>::getListPrecompute(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < this->numLists) && (listId >= 0));
    return *preComputeData[listId];
}

template<typename P>
bool IndexInt8IVFFlat<P>::listVectorsNeedReshaped() const
{
    return true;
}

template<typename P>
void IndexInt8IVFFlat<P>::getListVectorsReshaped(int listId, std::vector<int8_t> &reshaped) const
{
    ASCEND_THROW_IF_NOT((listId < this->numLists) && (listId >= 0));

    size_t size = this->getListLength(listId);
    auto &data = this->getListVectors(listId);
    int dimShaped = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    reshaped.resize(size * this->dims);

// reshape code from Zz format data to contigous format.
#pragma omp parallel for if (size >= 100)
    for (size_t i = 0; i < size; i++) {
        int offset = getShapedDataOffset(i);
        auto srcPtr = data.data() + offset;
        auto dstPtr = reshaped.data() + i * this->dims * sizeof(int8_t);
        for (int j = 0; j < dimShaped; j++) {
            MEMCPY_S(dstPtr + j * CUBE_ALIGN_INT8, CUBE_ALIGN_INT8 * sizeof(int8_t),
                srcPtr + j * CUBE_ALIGN * CUBE_ALIGN_INT8, CUBE_ALIGN_INT8 * sizeof(int8_t));
        }
    }
}

template class IndexInt8IVFFlat<float16_t>;
template class IndexInt8IVFFlat<int32_t>;
} // ascend
