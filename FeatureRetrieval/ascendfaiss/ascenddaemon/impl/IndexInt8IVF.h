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

#ifndef ASCEND_INDEX_INT8_IVF_INCLUDED
#define ASCEND_INDEX_INT8_IVF_INCLUDED

#include <ascenddaemon/impl/IndexInt8.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/AscendThreadPool.h>
#include <vector>
#include <memory>

namespace ascend {
class IndexInt8IVF : public IndexInt8 {
public:
    using idx_t = uint32_t;

    IndexInt8IVF(int numList, int byteCntPerVector, int dim, int nprobes, MetricType metric, int resourceSize = -1);

    virtual ~IndexInt8IVF();

    // Clear out all inverted lists
    void reset() override;

    // Return the number of dimension we are indexing
    int getDim() const;

    // Return the number of inverted lists
    size_t getNumLists() const;

    // Return the list length of a particular list
    size_t getListLength(int listId) const;

    // Return the list indices of a particular list in this device?
    DeviceVector<uint32_t> &getListIndices(int listId) const;

    virtual void addVectors(int listId, size_t numVecs, const int8_t *codes, const uint32_t *indices) = 0;

    // whether the encoded vectors is shaped and need reconstruct when getListVectors
    virtual bool listVectorsNeedReshaped() const;

    // Return the encoded vectors of a particular list in this device?
    DeviceVector<int8_t> &getListVectors(int listId) const;

    // reconstruct the shaped code data to origin code when getListVectors
    virtual void getListVectorsReshaped(int listId, std::vector<int8_t> &reshaped) const;

    // Before adding vectors, one can call this to reserve device memory
    // to improve the performance of adding
    void reserveMemory(size_t numVecs) override;

    virtual void reserveMemory(int listId, size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;

    virtual size_t reclaimMemory(int listId);

    virtual void setNumProbes(int nprobes);

    virtual void updateCoarseCentroidsData(AscendTensor<int8_t, DIMS_2> &coarseCentroidsData);

protected:
    // Number of inverted files
    int numLists;

    // Number of bytes per vector in the list
    int bytesPerVector;

    // top nprobe for quantizer searching
    int nprobe;

    // Maximum list length seen
    int maxListLength;

    // shared ops
    std::map<int, std::unique_ptr<AscendOperator>> distL1Ops;

    // tensor store L1 coarse centroids
    AscendTensor<int8_t, DIMS_2> coarseCentroids;

    // tensor store L1 coarse centroids(Zz shaped data)
    AscendTensor<int8_t, DIMS_4> coarseCentroidsShaped;

    // code data list
    std::vector<std::unique_ptr<DeviceVector<int8_t>>> deviceListData;

    // indices data list
    std::vector<std::unique_ptr<DeviceVector<uint32_t>>> deviceListIndices;
};
} // namespace ascend

#endif // ASCEND_INDEX_INT8_IVF_INCLUDED
