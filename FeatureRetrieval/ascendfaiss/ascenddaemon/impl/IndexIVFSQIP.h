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

#ifndef ASCEND_INDEXIVFSQ_IP_INCLUDED
#define ASCEND_INDEXIVFSQ_IP_INCLUDED

#include <ascenddaemon/impl/IndexIVFSQ.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/AscendMemory.h>
#include <ascenddaemon/utils/AscendThreadPool.h>
#include <arm_fp16.h>

namespace ascend {
class IndexIVFSQIP : public IndexIVFSQ<float> {
public:
    IndexIVFSQIP(int numList, int dim, bool encodeResidual, int nprobes, int resourceSize = -1);

    ~IndexIVFSQIP();

private:
    void searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    void searchImplL1(AscendTensor<float16_t, DIMS_2> &queries, 
                      AscendTensor<float16_t, DIMS_2> &distances,
                      aclrtStream stream);

    void searchImplL2(AscendTensor<float16_t, DIMS_2> &queries, 
                      AscendTensor<float16_t, DIMS_2> &l1Distances,
                      AscendTensor<float16_t, DIMS_2> &outDistances, 
                      AscendTensor<uint32_t, DIMS_2> &outIndices);

    void resetSqDistOperator();

    void runSqDistOperator(AscendTensor<float16_t, DIMS_2> &queries, 
                           AscendTensor<uint8_t, DIMS_4> &codes,
                           AscendTensor<uint32_t, DIMS_1> &size, 
                           AscendTensor<float16_t, DIMS_2> &result,
                           AscendTensor<float16_t, DIMS_2> &maxResult,
                           AscendTensor<uint16_t, DIMS_1> &flag);

private:
    bool byResidual;
    int burstLen;
    int bursts;
    std::vector<const aclDataBuffer *> distSqOpInput;
    std::vector<aclDataBuffer *> distSqOpOutput;

    TopkOp<std::less<>, std::less_equal<>, float16_t, false> topKMaxOp;
    TopkOp<std::greater<>, std::greater_equal<>, float16_t> topKMinOp;
};
} // namespace ascend
#endif // ASCEND_INDEXIVFPQ_IP_INCLUDED
