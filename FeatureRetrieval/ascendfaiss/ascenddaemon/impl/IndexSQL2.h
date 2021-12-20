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

#ifndef ASCEND_INDEXSQ_L2_INCLUDED
#define ASCEND_INDEXSQ_L2_INCLUDED

#include <ascenddaemon/impl/IndexSQ.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/AscendThreadPool.h>
#include <memory>

namespace ascend {
class IndexSQL2 : public IndexSQ {
public:
    using idx_t = Index::idx_t;

    IndexSQL2(int dim, int resource = -1);

    ~IndexSQL2();

private:
    void searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    void runSqDistOperator(AscendTensor<float16_t, DIMS_2> &queries,
                            AscendTensor<uint8_t, DIMS_4> &codes,
                            AscendTensor<float, DIMS_1> &precomp,
                            AscendTensor<float16_t, DIMS_1> &vdiff,
                            AscendTensor<float16_t, DIMS_1> &vmin,
                            AscendTensor<uint32_t, DIMS_1> &size,
                            AscendTensor<float16_t, DIMS_2> &result,
                            AscendTensor<float16_t, DIMS_2> &minResult,
                            AscendTensor<uint16_t, DIMS_1> &flag,
                            aclrtStream stream);

    void resetSqDistOperator();

private:
    TopkOp<std::greater<>, std::greater_equal<>, float16_t> topkMinOp;
};
} // namespace ascend

#endif // ASCEND_INDEXSQ_L2_INCLUDED
