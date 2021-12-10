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

#ifndef ASCEND_INDEXFLAT_L2_INCLUDED
#define ASCEND_INDEXFLAT_L2_INCLUDED

#include <vector>

#include <ascenddaemon/impl/IndexFlat.h>
#include <ascenddaemon/utils/TopkOp.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/DeviceVectorInl.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/AscendThreadPool.h>

namespace ascend {
class IndexFlatL2 : public IndexFlat {
public:
    IndexFlatL2(int dim, int resourceSize = -1);

    ~IndexFlatL2();

    void reset();

    void addVectors(AscendTensor<float16_t, DIMS_2>& rawData) override;

    inline const std::vector<std::unique_ptr<DeviceVector<float16_t>>>& getNormBase() const
    {
        return normBase;
    }

private:
    void searchImpl(AscendTensor<float16_t, DIMS_2>& queries, int k,
                    AscendTensor<float16_t, DIMS_2>& outDistance, AscendTensor<uint32_t, DIMS_2>& outIndices) override;

    void runDistCompute(AscendTensor<float16_t, DIMS_2> &queryVecs,
                        AscendTensor<float16_t, DIMS_4> &shapedData,
                        AscendTensor<float16_t, DIMS_1> &norms,
                        AscendTensor<uint32_t, DIMS_1> &size,
                        AscendTensor<float16_t, DIMS_2> &outDistances,
                        AscendTensor<float16_t, DIMS_2> &maxDistances,
                        AscendTensor<uint16_t, DIMS_1> &flag,
                        aclrtStream stream);

    void resetDistCompOp(int numLists);

    void moveNormForward(idx_t srcIdx, idx_t dstIdx);
    inline void moveVectorForward(idx_t srcIdx, idx_t dstIdx) override
    {
        moveNormForward(srcIdx, dstIdx);
        moveShapedForward(srcIdx, dstIdx);
    }

    void removeInvalidData(int oldTotal, int remove);

    size_t calcNormBaseSize(idx_t totalNum);
private:
    std::vector<std::unique_ptr<DeviceVector<float16_t>>> normBase;
    TopkOp<std::greater<>, std::greater_equal<>, float16_t> topkOp;
};
}  // namespace ascend

#endif  // ASCEND_INDEXFLAT_INCLUDED
