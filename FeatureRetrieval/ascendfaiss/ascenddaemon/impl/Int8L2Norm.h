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

#ifndef ASCEND_INT8_L2_NORM_INCLUDED
#define ASCEND_INT8_L2_NORM_INCLUDED

#include <map>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <arm_fp16.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/AscendOperator.h>

namespace ascend {
struct IDSelector;
class IndexPreTransform;

class Int8L2Norm {
public:
    using idx_t = uint32_t;

    explicit Int8L2Norm(idx_t d = 0);

    ~Int8L2Norm();

    void dispatchL2NormTask(AscendTensor<int8_t, DIMS_2> &codesData, 
                            AscendTensor<float16_t, DIMS_1> &normData,
                            AscendTensor<uint32_t, DIMS_2> &actualNum,
                            aclrtStream stream);

protected:
    // vector dimension
    int dims;

    std::vector<int> l2NormPageSizes;
    std::unique_ptr<AscendOperator> l2NormOp;
    std::vector<const aclDataBuffer *> l2NormOpInput;
    std::vector<aclDataBuffer *> l2NormOpOutput;

    std::vector<float16_t> transfer;

private:
    void resetL2NormOperator();

    void runL2NormOperator(AscendTensor<int8_t, DIMS_2> &vectors,
                           AscendTensor<float16_t, DIMS_2> &transfer,
                           AscendTensor<uint32_t, DIMS_1> &actualNum,
                           AscendTensor<float16_t, DIMS_1> &result,
                           aclrtStream stream);
};
}  // namespace ascend

#endif  // ASCEND_INT8_L2_NORM_INCLUDED