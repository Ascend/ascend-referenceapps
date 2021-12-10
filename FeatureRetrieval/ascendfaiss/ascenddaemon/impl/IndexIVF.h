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

#ifndef ASCEND_INDEXIVF_INCLUDED
#define ASCEND_INDEXIVF_INCLUDED

#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/impl/Index.h>
#include <vector>
#include <memory>

namespace ascend {
class IndexIVF : public Index {
public:
    IndexIVF(int numList, int byteCntPerVector, int dim, int nprobes, int resourceSize = -1);

    virtual ~IndexIVF();

    // Clear out all inverted lists
    void reset() override;

    // Return the number of dimension we are indexing
    int getDim() const;

    // Return the number of inverted lists
    size_t getNumLists() const;

    // Return the list length of a particular list
    size_t getListLength(int listId) const;

    // Return the list buffer length of code and id of a particular list
    size_t getMaxListDataIndicesBufferSize() const;

    // Return the list indices of a particular list in this device?
    DeviceVector<uint32_t>& getListIndices(int listId) const;

    // Return the encoded vectors of a particular list in this device?
    DeviceVector<unsigned char>& getListVectors(int listId) const;

    // whether the encoded vectors is shaped and need reconstruct when getListVectors
    virtual bool listVectorsNeedReshaped() const;

    // reconstruct the shaped code data to origin code when getListVectors
    virtual void getListVectorsReshaped(int listId, 
        std::vector<unsigned char>& reshaped) const;

    virtual void getListVectorsReshaped(int listId, unsigned char* reshaped) const;

    // Before adding vectors, one can call this to reserve device memory
    // to improve the performance of adding
    void reserveMemory(size_t numVecs) override;

    virtual void reserveMemory(int listId, size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;

    virtual size_t reclaimMemory(int listId);

    virtual void setNumProbes(int nprobes);

    virtual void updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2>& coarseCentroidsData);

protected:
    // Number of inverted files
    int numLists;

    // Number of bytes per vector in the list
    int bytesPerVector;

    // top nprobe for quantizer searching
    int nprobe;

    // Maximum list length seen
    int maxListLength;

    // tensor store L1 coarse centroids
    AscendTensor<float16_t, DIMS_2> coarseCentroids;

    // tensor store L1 coarse centroids(Zz shaped data)
    AscendTensor<float16_t, DIMS_4> coarseCentroidsShaped;

    // tensor store L1 coarse centroids precomputed norms
    AscendTensor<float16_t, DIMS_1> normCoarseCentroids;

    // code data list
    std::vector<std::unique_ptr<DeviceVector<unsigned char>>> deviceListData;

    // indices data list
    std::vector<std::unique_ptr<DeviceVector<uint32_t>>> deviceListIndices;
};
}  // namespace ascend

#endif  // ASCEND_INDEXIVF_INCLUDED
