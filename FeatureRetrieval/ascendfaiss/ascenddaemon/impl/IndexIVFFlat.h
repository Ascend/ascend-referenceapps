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

#ifndef ASCEND_INDEXIVFFLAT_INCLUDED
#define ASCEND_INDEXIVFFLAT_INCLUDED

#include <ascenddaemon/impl/IndexIVF.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/AscendMemory.h>
#include <ascenddaemon/utils/AscendThreadPool.h>
#include <arm_fp16.h>

namespace ascend {
class IndexIVFFlat : public IndexIVF {
public:
    IndexIVFFlat(int numList, int dim, int nprobes, int resourceSize = -1);

    ~IndexIVFFlat();

    void reset() override;

    // Before adding vectors, one can call this to reserve device memory
    // to improve the performance of adding
    void reserveMemory(size_t numVecs) override;

    void reserveMemory(int listId, size_t numVecs) override;

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;

    size_t reclaimMemory(int listId) override;

    void addVectors(int listId, size_t numVecs, const float16_t *rawData, const uint32_t *indices);

    void updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2> &coarseCentroidsData) override;

    // for debugging purpose
    DeviceVector<float16_t> &getListPrecompute(int listId) const;

    // whether the encoded vectors is shaped and need reconstruct when getListVectors
    bool listVectorsNeedReshaped() const override;

    // reconstruct the shaped code data to origin code when getListVectors
    void getListVectorsReshaped(int listId, std::vector<unsigned char> &reshaped) const override;

private:
    void addImpl(int n, const float16_t *x, const idx_t *ids) override;

    void searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    void searchImplL1(AscendTensor<float16_t, DIMS_2> &queries,
                      AscendTensor<uint32_t, DIMS_2> &result,
                      aclrtStream stream);

    void searchImplL2(AscendTensor<float16_t, DIMS_2> &queries,
                      AscendTensor<uint32_t, DIMS_2> &l1Indices,
                      AscendTensor<float16_t, DIMS_2> &outDistances,
                      AscendTensor<uint32_t, DIMS_2> &outIndices);

    void calcResiduals(AscendTensor<float16_t, DIMS_2> &query,
        AscendTensor<uint32_t, DIMS_2> &nprobeIndices,
        AscendTensor<float16_t, DIMS_3> &residulas);

    int getShapedDataOffset(int idx) const;

    size_t removeIdsImpl(const IDSelector &sel) override;

    void resetDistCompOp(int numLists);

    void runDistCompute(AscendTensor<float16_t, DIMS_2>& queryVecs,
                        AscendTensor<float16_t, DIMS_4>& shapedData,
                        AscendTensor<float16_t, DIMS_1>& norms,
                        AscendTensor<float16_t, DIMS_2>& outDistances,
                        AscendTensor<uint16_t, DIMS_1>& flag,
                        aclrtStream stream);

private:
    AscendThreadPool *threadPool;

    // shared ops
    std::unique_ptr<AscendOperator> distCompOp;

    std::vector<std::unique_ptr<DeviceVector<float16_t>>> preComputeData;
};
} // namespace ascend

#endif // ASCEND_INDEXIVFFLAT_INCLUDED