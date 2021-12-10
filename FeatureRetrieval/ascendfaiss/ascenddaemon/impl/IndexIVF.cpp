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

#include <ascenddaemon/impl/IndexIVF.h>
#include <ascenddaemon/utils/Limits.h>

namespace ascend {
IndexIVF::IndexIVF(int numList, int byteCntPerVector, int dim, int nprobes, int resourceSize)
    : Index(dim, resourceSize), 
      numLists(numList), 
      bytesPerVector(byteCntPerVector), 
      nprobe(nprobes), 
      maxListLength(0)
{
    ASCEND_THROW_IF_NOT(numList > 0);
    for (int i = 0; i < numLists; ++i) {
        deviceListData.push_back(std::unique_ptr<DeviceVector<unsigned char>>(
            new DeviceVector<unsigned char>(MemorySpace::DEVICE_HUGEPAGE)));
        deviceListIndices.push_back(std::unique_ptr<DeviceVector<uint32_t>>(
            new DeviceVector<uint32_t>(MemorySpace::DEVICE_HUGEPAGE)));
    }
}

IndexIVF::~IndexIVF() {}

void IndexIVF::reset()
{
    deviceListData.clear();
    deviceListIndices.clear();
    for (int i = 0; i < numLists; ++i) {
        deviceListData.push_back(std::unique_ptr<DeviceVector<unsigned char>>(
            new DeviceVector<unsigned char>(MemorySpace::DEVICE_HUGEPAGE)));
        deviceListIndices.push_back(std::unique_ptr<DeviceVector<uint32_t>>(
            new DeviceVector<uint32_t>(MemorySpace::DEVICE_HUGEPAGE)));
    }

    maxListLength = 0;
    this->ntotal = 0;
}

int IndexIVF::getDim() const
{
    return dims;
}

void IndexIVF::reserveMemory(size_t numVecs)
{
    size_t numVecsPerList = utils::divUp(numVecs, static_cast<size_t>(numLists));
    if (numVecsPerList < 1) {
        return;
    }

    size_t bytesPerDataList = numVecsPerList * bytesPerVector;
    for (auto& list : deviceListData) {
        list->reserve(bytesPerDataList);
    }

    for (auto& list : deviceListIndices) {
        list->reserve(numVecsPerList);
    }
}

void IndexIVF::reserveMemory(int listId, size_t numVecs)
{
    if (numVecs < 1) {
        return;
    }

    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    size_t bytesDataList = numVecs * bytesPerVector;
    deviceListData[listId]->reserve(bytesDataList);
    deviceListIndices[listId]->reserve(numVecs);
}

size_t IndexIVF::reclaimMemory()
{
    size_t totalReclaimed = 0;

    for (auto& list : deviceListData) {
        totalReclaimed += list->reclaim(true);
    }

    for (auto& list : deviceListIndices) {
        totalReclaimed += list->reclaim(true);
    }

    return totalReclaimed;
}

size_t IndexIVF::reclaimMemory(int listId)
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    size_t totalReclaimed = 0;
    totalReclaimed += deviceListData[listId]->reclaim(true);
    totalReclaimed += deviceListIndices[listId]->reclaim(true);

    return totalReclaimed;
}

void IndexIVF::setNumProbes(int nprobes)
{
    this->nprobe = nprobes;
}

size_t IndexIVF::getNumLists() const
{
    return numLists;
}

size_t IndexIVF::getListLength(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return deviceListIndices[listId]->size();
}

size_t IndexIVF::getMaxListDataIndicesBufferSize() const
{
    return maxListLength * (this->dims * sizeof(unsigned char) + sizeof(uint32_t));
}

DeviceVector<uint32_t>& IndexIVF::getListIndices(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return *deviceListIndices[listId];
}

DeviceVector<unsigned char>& IndexIVF::getListVectors(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return *deviceListData[listId];
}

bool IndexIVF::listVectorsNeedReshaped() const
{
    return false;
}

void IndexIVF::getListVectorsReshaped(int listId, std::vector<unsigned char>& reshaped) const
{
    VALUE_UNUSED(listId);
    VALUE_UNUSED(reshaped);
    ASCEND_THROW_MSG("getListVectorsReshaped not implemented for this type of index");
}

void IndexIVF::getListVectorsReshaped(int listId, unsigned char* reshaped) const
{
    VALUE_UNUSED(listId);
    VALUE_UNUSED(reshaped);
    ASCEND_THROW_MSG("getListVectorsReshaped not implemented for this type of index");
}

void IndexIVF::updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2>& coarseCentroidsData)
{
    int numCoarseCents = coarseCentroidsData.getSize(0);
    int dimCoarseCents = coarseCentroidsData.getSize(1);
    ASCEND_THROW_IF_NOT_FMT(numCoarseCents == numLists && dimCoarseCents == dims,
                            "coarse centroids data's shape invalid.(%d X %d) vs (%d X %d)",
                            numCoarseCents, dimCoarseCents, numLists, dims);

    // coarse centroids need to be Zz format because of DistanceCompute operator's limitation.
    //       origin code for example (shape n X dim). n=15, dim = 127. n and dim need to be 16 aligned, 
    //         n aligned = 16, dim aligned = 128, the space for aligned need to been padded to 0x00
    //       |  0_0  0_1  0_2  0_3 ...  0_125  0_126 0x00 |
    //       |  1_0  1_1  1_2  1_3 ...  1_125  1_126 0x00 |
    //       |        .                          .        |
    //       |        .                          .        |
    //       | 14_0 14_1 14_2 14_3 ... 14_125 14_126 0x00 |
    //       | 0x00 0x00 0x00 0x00 ...   0x00   0x00 0x00 |
    //                              | 
    //             after Zz format  (shape dims 2: n X dim, dims4: (n/16) X (dim/16) X 16 X 16)
    //       |   0_0   0_1 ...  0_14  0_15   1_0   1_1 ...  1_15 ...   7_15 |
    //       |  0_16  0_17 ...  0_30  0_31  1_16  1_17 ...  1_31 ...   7_31 |
    //       |        .                    .                  .         .   |
    //       |        .                    .                  .         .   |
    //       |  0_96  0_97 ... 0_110 0_111  1_96  1_97 ... 1_111 ...  7_111 |
    //       | 0_112 0_113 ... 0_126  0x00 1_112 1_113 ...  0x00 ...   0x00 |
    //       |   8_0   8_1 ...  8_14  8_15   9_0   9_1 ...  9_15 ...   0x00 |
    //       |  8_16  8_17 ...  8_30  8_31  9_16  9_17 ...  9_31 ...   0x00 |
    //       |        .                    .                  .         .   |
    //       |        .                    .                  .         .   |
    //       |  8_96  8_97 ... 8_110 8_111  9_96  9_97 ... 9_111 ...   0x00 |
    //       | 8_112 8_113 ... 8_126  0x00 9_112 9_113 ...  0x00 ...   0x00 |
    AscendTensor<float16_t, DIMS_2> deviceCoarseCentroids(
        { numCoarseCents, dimCoarseCents });
    deviceCoarseCentroids.copyFromSync(coarseCentroidsData);
    coarseCentroids = std::move(deviceCoarseCentroids);

    int dim1 = utils::divUp(numCoarseCents, CUBE_ALIGN_SIZE);
    int dim2 = utils::divUp(dimCoarseCents, CUBE_ALIGN_SIZE);
    AscendTensor<float16_t, DIMS_4> tmpShapedCentroids(
        { dim1, dim2, CUBE_ALIGN_SIZE, CUBE_ALIGN_SIZE });

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            float16_t *tmpData = tmpShapedCentroids[i][j].data();
            int hpadding = (j == (dim2 - 1)) ?
                ((j + 1) * CUBE_ALIGN_SIZE - dimCoarseCents) : 0;
            int vpadding = (i == (dim1 - 1)) ?
                ((i + 1) * CUBE_ALIGN_SIZE - numCoarseCents) : 0;
            for (int v = 0; v < (CUBE_ALIGN_SIZE - vpadding); v++) {
                (void)memcpy_s(tmpData, (CUBE_ALIGN_SIZE - hpadding) * sizeof(float16_t),
                    coarseCentroidsData[i * CUBE_ALIGN_SIZE + v][j * CUBE_ALIGN_SIZE].data(),
                    (CUBE_ALIGN_SIZE - hpadding) * sizeof(float16_t));
                tmpData += (CUBE_ALIGN_SIZE - hpadding);

                if (hpadding) {
                    (void)memset_s(tmpData, sizeof(float16_t) * hpadding, 0x0, 
                        sizeof(float16_t) * hpadding);
                    tmpData += hpadding;
                }
            }

            for (int vp = 0; vp < vpadding; vp++) {
                (void)memset_s(tmpData, sizeof(float16_t) * CUBE_ALIGN_SIZE, 0x0, 
                    sizeof(float16_t) * CUBE_ALIGN_SIZE);
                tmpData += CUBE_ALIGN_SIZE;
            }
        }
    }

    coarseCentroidsShaped = std::move(tmpShapedCentroids);

    // update L2 norm of coarseCentroids
    AscendTensor<float16_t, 1> tmpNormTensor({ numCoarseCents });
    fvecNormsL2sqr(tmpNormTensor.data(), coarseCentroidsData.data(), 
        dimCoarseCents, numCoarseCents);
    normCoarseCentroids = std::move(tmpNormTensor);
}
}  // namespace ascend
