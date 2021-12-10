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

#ifndef ASCEND_INDEXIVFPQ_INCLUDED
#define ASCEND_INDEXIVFPQ_INCLUDED

#include <ascenddaemon/impl/IndexIVF.h>
#include <ascenddaemon/utils/DistanceMatrixOp.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/AscendMemory.h>
#include <ascenddaemon/utils/AscendThreadPool.h>
#include <arm_fp16.h>

namespace ascend {
class IndexIVFPQ : public IndexIVF {
public:
    IndexIVFPQ(int numList, int dim, int subQuantizers, int bitCntPerSubQuantizer, int nprobes, int resourceSize = -1);

    ~IndexIVFPQ();

    void addVectors(int listId, const uint8_t* codes, const uint32_t* indices, size_t numVecs);

    void updatePQCentroidsData(AscendTensor<float16_t, DIMS_2>& pqCentroidsData);

    void updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2>& coarseCentroidsData) override;

private:
    void addImpl(int n, const float16_t* x, const idx_t* ids) override;

    void searchImpl(AscendTensor<float16_t, DIMS_2>& queries, int k,
                    AscendTensor<float16_t, DIMS_2>& outDistance, AscendTensor<uint32_t, DIMS_2>& outIndices);

    void searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t* labels) override;

    size_t removeIdsImpl(const IDSelector& sel) override;

    void resetDistTableOperator();

    void resetDistAccumOperator();

    void runDistanceTableBuild(AscendTensor<float16_t, DIMS_2>& queries,
                               AscendTensor<float16_t, DIMS_2>& PQCentroids,
                               AscendTensor<int32_t, DIMS_2>& listNos,
                               AscendTensor<float16_t, DIMS_2>& l1Centroids,
                               AscendTensor<float16_t, DIMS_4>& distTables,
                               AscendTensor<uint16_t, DIMS_1>& flag,
                               aclrtStream stream);

    void runDistanceMatrixAccum(AscendTensor<float16_t, DIMS_2>& distMatrix,
                                AscendTensor<float16_t, DIMS_1>& distAccumResult,
                                AscendTensor<uint16_t, DIMS_1>& distAccumFlag,
                                aclrtStream stream);

private:
    AscendTensor<float16_t, DIMS_2> pqCentroids;

    /// Number of sub-quantizers per vector
    const int numSubQuantizers;

    /// Number of sub-quantizers per vector aligned with 16
    const int numSubQuantizersAligned;

    /// Number of bits per sub-quantizer
    const int bitsPerSubQuantizer;

    /// Number of per sub-quantizer codes (2^bits)
    const int numSubQuantizerCodes;

    /// Number of dimensions per each sub-quantizer
    const int dimPerSubQuantizer;

    std::unique_ptr<AscendOperator> distTableBuildOp;
    std::unique_ptr<AscendOperator> distAccumOp;
    DistanceMatrixOp distMatrixOp;

    AscendThreadPool threadPool;
};
}  // namespace ascend

#endif  // ASCEND_INDEXIVFPQ_INCLUDED
