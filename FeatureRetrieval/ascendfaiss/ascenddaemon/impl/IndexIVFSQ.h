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

#ifndef ASCEND_INDEXIVFSQ_INCLUDED
#define ASCEND_INDEXIVFSQ_INCLUDED

#include <ascenddaemon/impl/IndexIVF.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/AscendMemory.h>
#include <ascenddaemon/utils/AscendThreadPool.h>
#include <arm_fp16.h>

namespace ascend {
namespace {
const int CUBE_ALIGN = 16;
const int SEARCH_LIST_SIZE = 65536; // must be CUBE_ALIGN aligned
const int SEARCH_SHAPED_SIZE = SEARCH_LIST_SIZE / CUBE_ALIGN;
const int TIMEOUT_CHECK_TICK = 5120;
const double TIMEOUT_MS = 50000;
const int SIZE_ALIGN = 8;
const int THREADS_CNT = 4;
const int BURST_LEN = 32; // SEARCH_LIST_SIZE must align to BURST_LEN
}

template<typename T>
class IndexIVFSQ : public IndexIVF {
public:
    IndexIVFSQ<T>(int numList, int dim, bool encodeResidual, int nprobes, int resourceSize = -1);

    ~IndexIVFSQ<T>();

    void reset() override;

    // Before adding vectors, one can call this to reserve device memory
    // to improve the performance of adding
    void reserveMemory(size_t numVecs) override;

    void reserveMemory(int listId, size_t numVecs) override;

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;

    size_t reclaimMemory(int listId) override;

    void addVectors(int listId, size_t numVecs, const uint8_t *codes,
                    const uint32_t *indices, const float *preCompute);

    void updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2> &coarseCentroidsData);

    void updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin,
                            AscendTensor<float16_t, DIMS_1> &trainedDiff);

    void calcResiduals(AscendTensor<float16_t, DIMS_1> &query,
                       AscendTensor<uint32_t, DIMS_1> &nprobeIndices,
                       AscendTensor<float16_t, DIMS_2> &residulas);

    // for debugging purpose
    DeviceVector<float>& getListPrecompute(int listId) const;

    // whether the encoded vectors is shaped and need reconstruct when getListVectors
    bool listVectorsNeedReshaped() const override;

    // reconstruct the shaped code data to origin code when getListVectors
    void getListVectorsReshaped(int listId, std::vector<unsigned char>& reshaped) const override;

    void getListVectorsReshaped(int listId, unsigned char* reshaped) const override;

protected:
    void addImpl(int n, const float16_t* x, const idx_t* ids) override;

    size_t removeIdsImpl(const IDSelector& sel) override;

    int getShapedDataOffset(int idx) const;

protected:
    AscendTensor<float16_t, DIMS_1> vMin;
    AscendTensor<float16_t, DIMS_1> vDiff;

    std::unique_ptr<AscendOperator> distSqOp;

    std::vector<std::unique_ptr<DeviceVector<T>>> preComputeData;

    AscendThreadPool* threadPool;
};
}  // namespace ascend

#endif  // ASCEND_INDEXIVFPQ_INCLUDED

