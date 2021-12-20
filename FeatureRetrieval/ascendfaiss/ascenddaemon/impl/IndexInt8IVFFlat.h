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

#ifndef ASCEND_INDEX_INT8_IVFFLAT_INCLUDED
#define ASCEND_INDEX_INT8_IVFFLAT_INCLUDED

#include <ascenddaemon/impl/IndexInt8IVF.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/AscendMemory.h>
#include <ascenddaemon/utils/AscendThreadPool.h>
#include <arm_fp16.h>

namespace ascend {
template<typename P>
class IndexInt8IVFFlat : public IndexInt8IVF {
public:
    using idx_t = Index::idx_t;

    IndexInt8IVFFlat(int numList, int dim, int nprobes, MetricType metric, int resourceSize = -1);

    ~IndexInt8IVFFlat();

    void reset() override;

    // Before adding vectors, one can call this to reserve device memory
    // to improve the performance of adding
    void reserveMemory(size_t numVecs) override;

    void reserveMemory(int listId, size_t numVecs) override;

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;

    size_t reclaimMemory(int listId) override;

    void addVectors(int listId, size_t numVecs, const int8_t *codes, const uint32_t *indices);

    // for debugging purpose
    DeviceVector<P> &getListPrecompute(int listId) const;

    // whether the vectors is shaped and need reconstruct when getListVectors
    bool listVectorsNeedReshaped() const;

    // reconstruct the shaped code data to origin code when getListVectors
    void getListVectorsReshaped(int listId, std::vector<int8_t> &reshaped) const;

protected:
    // precompute l2 norm Data list
    std::vector<std::unique_ptr<DeviceVector<P>>> preComputeData;

    AscendThreadPool *threadPool;

private:
    void addImpl(int n, const int8_t *x, const idx_t *ids);

    size_t removeIdsImpl(const IDSelector &sel);

    int getShapedDataOffset(int idx) const;
};
} // namespace ascend

#endif // ASCEND_INDEX_INT8_IVFFLAT_INCLUDED
