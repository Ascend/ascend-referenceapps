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

#ifndef ASCEND_INDEX_INT8_FLAT_L2_INCLUDED
#define ASCEND_INDEX_INT8_FLAT_L2_INCLUDED

#include <map>
#include <vector>

#include <ascenddaemon/impl/IndexInt8Flat.h>
#include <ascenddaemon/utils/TopkOp.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/AscendTensor.h>

namespace ascend {
class IndexInt8FlatL2 : public IndexInt8Flat<int32_t> {
public:
    using idx_t = Index::idx_t;

    IndexInt8FlatL2(int dim, int resourceSize = -1);

    ~IndexInt8FlatL2();

    void addVectors(AscendTensor<int8_t, DIMS_2> &rawData) override;

protected:
    void searchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels) override;

    void runDistCompute(AscendTensor<int8_t, DIMS_2> &queryVecs,
                        AscendTensor<uint8_t, DIMS_2> &mask,
                        AscendTensor<int8_t, DIMS_4> &shapedData,
                        AscendTensor<int32_t, DIMS_1> &norms,
                        AscendTensor<uint32_t, DIMS_2> &size,
                        AscendTensor<float16_t, DIMS_2> &outDistances,
                        AscendTensor<float16_t, DIMS_2> &minDistances,
                        AscendTensor<uint16_t, DIMS_2> &flag, aclrtStream stream);

    void resetDistCompOp(int codeNum);

protected:
    int burstsOfComputeBatch;

    // shared ops
    std::map<int, std::unique_ptr<AscendOperator>> distComputeOps;
    TopkOp<std::greater<>, std::greater_equal<>, float16_t> topkOp;
};
} // namespace ascend

#endif // ASCEND_INDEX_INT8_FLAT_L2_INCLUDED
