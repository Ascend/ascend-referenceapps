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

#ifndef DISTANCE_MATRIX_OP_INCLUDED
#define DISTANCE_MATRIX_OP_INCLUDED

#include <arm_fp16.h>
#include <ascenddaemon/utils/AscendTensor.h>

namespace ascend {
class DistanceMatrixOp {
public:
    DistanceMatrixOp();
    virtual ~DistanceMatrixOp();

    bool exec(AscendTensor<unsigned char, DIMS_2>& code,
              AscendTensor<float16_t, DIMS_2>& distTable,
              AscendTensor<float16_t, DIMS_2>& distMatrix);

private:
    bool checkParams(AscendTensor<unsigned char, DIMS_2>& code,
                     AscendTensor<float16_t, DIMS_2>& distTable,
                     AscendTensor<float16_t, DIMS_2>& distMatrix);

    void computePqCodeSize2(AscendTensor<unsigned char, DIMS_2>& code,
                            AscendTensor<float16_t, DIMS_2>& distTable,
                            AscendTensor<float16_t, DIMS_2>& distMatrix);

    void computePqCodeSize4(AscendTensor<unsigned char, DIMS_2>& code,
                            AscendTensor<float16_t, DIMS_2>& distTable,
                            AscendTensor<float16_t, DIMS_2>& distMatrix);

    void computePqCodeSize16(AscendTensor<unsigned char, DIMS_2>& code,
                             AscendTensor<float16_t, DIMS_2>& distTable,
                             AscendTensor<float16_t, DIMS_2>& distMatrix);
};
}  // namespace ascend

#endif
