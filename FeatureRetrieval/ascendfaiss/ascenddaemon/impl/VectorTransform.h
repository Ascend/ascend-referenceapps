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

#ifndef ASCEND_VECTORTRANSFORM_INCLUDED
#define ASCEND_VECTORTRANSFORM_INCLUDED

#include <map>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <arm_fp16.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/AscendOperator.h>

namespace ascend {
class VectorTransform {
public:
    explicit VectorTransform(int dimIn, int dimOut);

    virtual ~VectorTransform();

    // Perform training on a representative set of vectors
    virtual void train(int n, const float16_t *x);

    void apply(int n, const float16_t *x, float16_t *xt, aclrtStream stream);

    void applyAsync(int n, const float16_t *x, float16_t *xt, aclrtStream stream);

public:
    int dimIn;
    int dimOut;
    // set if the Index does not require training, or if training is done already
    bool isTrained;

protected:
    virtual void applyImpl(int n, const float16_t *x, float16_t *xt, aclrtStream stream) = 0;

protected:
    // support apply batch sizes, default is no paging
    std::vector<int> applyPageSizes;
};

class LinearTransform : public VectorTransform {
public:
    explicit LinearTransform(int dimIn, int dimOut, bool haveBias);

    explicit LinearTransform(int dimIn, int dimOut);

    void updateTrainedValue(AscendTensor<float16_t, DIMS_2> &matrix, AscendTensor<float, DIMS_1> &bias);

    virtual ~LinearTransform();

protected:
    void applyImpl(int n, const float16_t *x, float16_t *xt, aclrtStream stream) override;
private:
    void resetLinearTransOperator();

    void runLinearTransOperator(AscendTensor<float16_t, DIMS_2> &src, AscendTensor<float16_t, DIMS_3> &matrix,
        AscendTensor<float, DIMS_1> &bias, AscendTensor<float16_t, DIMS_2> &dst, aclrtStream stream);

private:
    AscendTensor<float16_t, DIMS_3> matrix;
    AscendTensor<float, DIMS_1> bias;
    // shared ops
    std::map<int, std::unique_ptr<AscendOperator>> linearTransOps;
};
}

#endif // ASCEND_VECTORTRANSFORM_INCLUDED
