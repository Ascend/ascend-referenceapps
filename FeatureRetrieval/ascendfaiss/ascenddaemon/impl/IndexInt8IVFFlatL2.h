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

#ifndef ASCEND_INDEX_INT8_IVFFLAT_L2_INCLUDED
#define ASCEND_INDEX_INT8_IVFFLAT_L2_INCLUDED

#include <ascenddaemon/impl/IndexInt8IVFFlat.h>
#include <ascenddaemon/utils/TaskQueueItem.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/AscendMemory.h>
#include <ascenddaemon/utils/AscendThreadPool.h>
#include <arm_fp16.h>

namespace ascend {
class IndexInt8IVFFlatL2 : public IndexInt8IVFFlat<int32_t> {
public:
    using idx_t = Index::idx_t;

    IndexInt8IVFFlatL2(int numList, int dim, int nprobes, int resourceSize = -1);

    ~IndexInt8IVFFlatL2();

    void addVectors(int listId, size_t numVecs, const int8_t *codes, const uint32_t *indices) override;

    void updateCoarseCentroidsData(AscendTensor<int8_t, DIMS_2> &coarseCentroidsData) override;

private:
    void addImpl(int n, const int8_t *x, const idx_t *ids);

    void searchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels);

    void resetL1DistOperator();

    void resetL2DistOperator();

    void runL1DistOperator(AscendTensor<int8_t, DIMS_2> &queries, 
                           AscendTensor<int8_t, DIMS_4> &codes,
                           AscendTensor<int32_t, DIMS_1> &precomp,
                           AscendTensor<float16_t, DIMS_2> &result, 
                           AscendTensor<uint16_t, DIMS_1> &flag)
    {
        AscendOperator *op = nullptr;
        int batch = queries.getSize(0);
        if (distL1Ops.find(batch) != distL1Ops.end()) {
            op = distL1Ops[batch].get();
        }
        ASCEND_ASSERT(op);

        // prepare for input data's buffer
        l1OpInput[0] = aclCreateDataBuffer(queries.data(), queries.getSizeInBytes());           // input 0
        l1OpInput[1] = aclCreateDataBuffer(codes.data(), codes.getSizeInBytes());               // input 1
        l1OpInput[2] = aclCreateDataBuffer(precomp.data(), precomp.getSizeInBytes());           // input 2

        // prepare for output data's buffer
        l1OpOutput[0] = aclCreateDataBuffer(result.data(), result.getSizeInBytes());            // output 0
        l1OpOutput[1] = aclCreateDataBuffer(flag.data(), flag.getSizeInBytes());                // output 1

        // async executing operator
        aclrtStream stream = resources.getDefaultStream();
        op->exec(l1OpInput, l1OpOutput, stream);

        for (auto &item : l1OpInput) {
            ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
        }

        for (auto &item : l1OpOutput) {
            ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
        }
    }                       

    void runL2DistOperator(AscendTensor<int8_t, DIMS_2> &queries, 
                          AscendTensor<int8_t, DIMS_4> &codes,
                          AscendTensor<int32_t, DIMS_1> &preComp, 
                          AscendTensor<uint32_t, DIMS_1> &size,
                          AscendTensor<float16_t, DIMS_2> &result, 
                          AscendTensor<float16_t, DIMS_2> &minResult, 
                          AscendTensor<uint16_t, DIMS_1> &flag)
    {
        // prepare for input data's buffer
        l2OpInput[0] = aclCreateDataBuffer(queries.data(), queries.getSizeInBytes());           // input 0
        l2OpInput[1] = aclCreateDataBuffer(codes.data(), codes.getSizeInBytes());               // input 1
        l2OpInput[2] = aclCreateDataBuffer(preComp.data(), preComp.getSizeInBytes());           // input 2
        l2OpInput[3] = aclCreateDataBuffer(size.data(), size.getSizeInBytes());                 // input 3

        // prepare for output data's buffer
        l2OpOutput[0] = aclCreateDataBuffer(result.data(), result.getSizeInBytes());            // output 0
        l2OpOutput[1] = aclCreateDataBuffer(minResult.data(), minResult.getSizeInBytes());      // output 1
        l2OpOutput[2] = aclCreateDataBuffer(flag.data(), flag.getSizeInBytes());                // output 2

        // async executing operator
        aclrtStream stream = resources.getDefaultStream();
        distL2Op->exec(l2OpInput, l2OpOutput, stream);

        for (auto &item : l2OpInput) {
            ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
        }

        for (auto &item : l2OpOutput) {
            ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
        }
    }

    void searchImplL1(AscendTensor<int8_t, DIMS_2> &queries, 
                      AscendTensor<float16_t, DIMS_2> &distances,
                      aclrtStream stream);

    void searchImplL2(AscendTensor<int8_t, DIMS_2> &queries, 
                      AscendTensor<float16_t, DIMS_2> &l1Distances,
                      AscendTensor<float16_t, DIMS_2> &outDistances, 
                      AscendTensor<uint32_t, DIMS_2> &outIndices);

    int32_t computeInt8VecSquare(const int8_t *x, size_t d);

    void computeInt8VecsSquare(int32_t *nr, const int8_t *x, size_t d, size_t nx);

private:
    int burstSize;

    std::unique_ptr<AscendOperator> distL2Op;

    std::vector<const aclDataBuffer *> l1OpInput;
    std::vector<aclDataBuffer *> l1OpOutput;
    std::vector<const aclDataBuffer *> l2OpInput;
    std::vector<aclDataBuffer *> l2OpOutput;

    TopkOp<std::greater<>, std::greater_equal<>, float16_t> topKMinOp;

    // tensor store L1 coarse centroids precomputed norms
    AscendTensor<int32_t, DIMS_1> normCoarseCentroids;
};
} // namespace ascend

#endif // ASCEND_INDEX_INT8_IVFFLAT_L2_INCLUDED
