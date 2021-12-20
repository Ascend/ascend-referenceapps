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

#include <ascenddaemon/impl/IndexInt8IVF.h>
#include <ascenddaemon/utils/Limits.h>

namespace ascend {
namespace {
const int CUBE_ALIGN = 16;
const int CUBE_ALIGN_INT8 = 32;
}

IndexInt8IVF::IndexInt8IVF(int numList, int byteCntPerVector, int dim, int nprobes, MetricType metric, int resourceSize)
    : IndexInt8(dim, metric, resourceSize),
      numLists(numList),
      bytesPerVector(byteCntPerVector),
      nprobe(nprobes),
      maxListLength(0)
{
    ASCEND_THROW_IF_NOT(numList > 0);
    for (int i = 0; i < numLists; ++i) {
        deviceListData.push_back(
            std::make_unique<DeviceVector<int8_t>>(MemorySpace::DEVICE_HUGEPAGE));
        deviceListIndices.push_back(
            std::make_unique<DeviceVector<uint32_t>>(MemorySpace::DEVICE_HUGEPAGE));
    }
}

IndexInt8IVF::~IndexInt8IVF() {}

void IndexInt8IVF::reset()
{
    deviceListData.clear();
    deviceListIndices.clear();

    for (int i = 0; i < numLists; ++i) {
        deviceListData.push_back(
            std::make_unique<DeviceVector<int8_t>>(MemorySpace::DEVICE_HUGEPAGE));
        deviceListIndices.push_back(
            std::make_unique<DeviceVector<uint32_t>>(MemorySpace::DEVICE_HUGEPAGE));
    }

    maxListLength = 0;
    this->ntotal = 0;
}

int IndexInt8IVF::getDim() const
{
    return this->dims;
}

void IndexInt8IVF::reserveMemory(size_t numVecs)
{
    size_t numVecsPerList = utils::divUp(numVecs, static_cast<size_t>(numLists));
    if (numVecsPerList < 1) {
        return;
    }

    size_t bytesPerDataList = numVecsPerList * bytesPerVector;
    for (auto &list : deviceListData) {
        list->reserve(bytesPerDataList);
    }

    for (auto &list : deviceListIndices) {
        list->reserve(numVecsPerList);
    }
}

void IndexInt8IVF::reserveMemory(int listId, size_t numVecs)
{
    if (numVecs < 1) {
        return;
    }

    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    size_t bytesDataList = numVecs * bytesPerVector;
    deviceListData[listId]->reserve(bytesDataList);
    deviceListIndices[listId]->reserve(numVecs);
}

size_t IndexInt8IVF::reclaimMemory()
{
    size_t totalReclaimed = 0;

    for (auto &list : deviceListData) {
        totalReclaimed += list->reclaim(true);
    }

    for (auto &list : deviceListIndices) {
        totalReclaimed += list->reclaim(true);
    }

    return totalReclaimed;
}

size_t IndexInt8IVF::reclaimMemory(int listId)
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    size_t totalReclaimed = 0;
    totalReclaimed += deviceListData[listId]->reclaim(true);
    totalReclaimed += deviceListIndices[listId]->reclaim(true);

    return totalReclaimed;
}

void IndexInt8IVF::setNumProbes(int nprobes)
{
    this->nprobe = nprobes;
}

size_t IndexInt8IVF::getNumLists() const
{
    return numLists;
}

size_t IndexInt8IVF::getListLength(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return deviceListIndices[listId]->size();
}

DeviceVector<uint32_t> &IndexInt8IVF::getListIndices(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return *deviceListIndices[listId];
}

DeviceVector<int8_t> &IndexInt8IVF::getListVectors(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return *deviceListData[listId];
}


bool IndexInt8IVF::listVectorsNeedReshaped() const
{
    return false;
}

void IndexInt8IVF::getListVectorsReshaped(int listId, std::vector<int8_t> &reshaped) const
{
    VALUE_UNUSED(listId);
    VALUE_UNUSED(reshaped);
    ASCEND_THROW_MSG("getListVectorsReshaped not implemented for this type of index");
}

void IndexInt8IVF::updateCoarseCentroidsData(AscendTensor<int8_t, DIMS_2> &coarseCentroidsData)
{
    int numCoarseCents = coarseCentroidsData.getSize(0);
    int dimCoarseCents = coarseCentroidsData.getSize(1);
    ASCEND_THROW_IF_NOT_FMT(numCoarseCents == numLists && dimCoarseCents == this->dims,
        "coarse centroids data's shape invalid.(%d X %d) vs (%d X %d)", numCoarseCents, dimCoarseCents, numLists,
        this->dims);

    // coarse centroids need to be Zz format because of DistanceCompute operator's limitation.
    //    // origin code for example (shape n X dim). n=15, dim = 127. n and dim need to be 16 aligned,
    //    //    n aligned = 16, dim aligned = 128, the space for aligned need to been padded to 0x00
    //    //    |  0_0  0_1  0_2  0_3 ...  0_125  0_126 0x00 |
    //    //    |  1_0  1_1  1_2  1_3 ...  1_125  1_126 0x00 |
    //    //    |        .                          .        |
    //    //    |        .                          .        |
    //    //    | 14_0 14_1 14_2 14_3 ... 14_125 14_126 0x00 |
    //    //    | 0x00 0x00 0x00 0x00 ...   0x00   0x00 0x00 |
    //    //                          |
    //    //    after Zz format  (shape dims 2: n X dim, dims4: (n/16) X (dim/32) X 16 X 32)
    //    //    |   0_0   0_1 ...  0_30  0_31  1_0   1_1  ...  1_30  1_31  ...  15_30  15_31 |
    //    //    |  0_32  0_33 ...  0_62  0_63  1_32  1_33 ...  1_62  1_63  ...  15_62  15_63 |
    //    //    |  0_64  0_65 ...  0_94  0_95  1_64  1_65 ...  1_94  1_95  ...  15_94  15_95 |
    //    //    |  0_96  0_97 ... 0_126  0x00  1_96  1_97 ... 1_126  0x00  ... 15_126   0x00 |
    //    // n and dim must be 16 aligned, otherwise padding data is needed.
    AscendTensor<int8_t, DIMS_2> deviceCoarseCentroids({ numCoarseCents, dimCoarseCents });
    deviceCoarseCentroids.copyFromSync(coarseCentroidsData);
    coarseCentroids = std::move(deviceCoarseCentroids);

    int dim1 = utils::divUp(numCoarseCents, CUBE_ALIGN);
    int dim2 = utils::divUp(dimCoarseCents, CUBE_ALIGN_INT8);
    AscendTensor<int8_t, DIMS_4> tmpShapedCentroids({ dim1, dim2, CUBE_ALIGN, CUBE_ALIGN_INT8 });

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            int8_t *tmpData = tmpShapedCentroids[i][j].data();
            int hpadding = (j == (dim2 - 1)) ? ((j + 1) * CUBE_ALIGN_INT8 - dimCoarseCents) : 0;
            int vpadding = (i == (dim1 - 1)) ? ((i + 1) * CUBE_ALIGN - numCoarseCents) : 0;
            for (int v = 0; v < (CUBE_ALIGN - vpadding); v++) {
                (void)memcpy_s(tmpData, (CUBE_ALIGN_INT8 - hpadding) * sizeof(int8_t),
                    coarseCentroidsData[i * CUBE_ALIGN + v][j * CUBE_ALIGN_INT8].data(),
                    (CUBE_ALIGN_INT8 - hpadding) * sizeof(int8_t));
                tmpData += (CUBE_ALIGN_INT8 - hpadding);

                if (hpadding) {
                    (void)memset_s(tmpData, sizeof(int8_t) * hpadding, 0x0, sizeof(int8_t) * hpadding);
                    tmpData += hpadding;
                }
            }

            for (int vp = 0; vp < vpadding; vp++) {
                (void)memset_s(tmpData, sizeof(int8_t) * CUBE_ALIGN_INT8, 0x0, sizeof(int8_t) * CUBE_ALIGN_INT8);
                tmpData += CUBE_ALIGN_INT8;
            }
        }
    }

    coarseCentroidsShaped = std::move(tmpShapedCentroids);
}
} // namespace ascend
