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

#include <ascenddaemon/impl/VectorTransform.h>
#include <ascenddaemon/utils/Limits.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/utils/AscendAssert.h>
namespace ascend {
VectorTransform::VectorTransform(int dimIn, int dimOut) : dimIn(dimIn), dimOut(dimOut), isTrained(false) {}

VectorTransform::~VectorTransform() {}

void VectorTransform::train(int n, const float16_t *x)
{
    VALUE_UNUSED(x);
    VALUE_UNUSED(n);
}

void VectorTransform::apply(int n, const float16_t *x, float16_t *xt, aclrtStream stream)
{
    applyAsync(n, x, xt, stream);
    ACL_REQUIRE_OK(aclrtSynchronizeStream(stream));
}

void VectorTransform::applyAsync(int n, const float16_t *x, float16_t *xt, aclrtStream stream)
{
    ASCEND_THROW_IF_NOT(isTrained);
    ASCEND_THROW_IF_NOT(n > 0);
    ASCEND_THROW_IF_NOT(x != nullptr);
    ASCEND_THROW_IF_NOT(xt != nullptr);
    size_t size = applyPageSizes.size();
    if (n > 1 && size > 0) {
        int applied = 0;
        for (size_t i = 0; i < size; i++) {
            int pageSize = applyPageSizes[i];
            if ((n - applied) < pageSize) {
                continue;
            }

            int page = (n - applied) / pageSize;
            for (int j = 0; j < page; j++) {
                applyImpl(pageSize, x + applied * this->dimIn, xt + applied * this->dimOut, stream);
                applied += pageSize;
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            applyImpl(1, x + i * this->dimIn, xt + i * this->dimOut, stream);
        }
    }
}

LinearTransform::LinearTransform(int dimIn, int dimOut) : VectorTransform(dimIn, dimOut)
{
    ASCEND_THROW_IF_NOT(dimIn > 0);
    ASCEND_THROW_IF_NOT(dimOut > 0);
    ASCEND_THROW_IF_NOT_FMT(dimOut % CUBE_ALIGN_SIZE == 0, "dimOut should be divisible by 16, but here is %d", dimOut);
    isTrained = false;
    applyPageSizes = { 256, 128, 64, 32, 16, 8, 4, 2, 1 };
    resetLinearTransOperator();
}

void LinearTransform::applyImpl(int n, const float16_t *x, float16_t *xt, aclrtStream stream)
{
    AscendTensor<float16_t, DIMS_2> src(const_cast<float16_t *>(x), { n, this->dimIn });
    AscendTensor<float16_t, DIMS_2> dst(xt, { n, this->dimOut });
    runLinearTransOperator(src, this->matrix, this->bias, dst, stream);
}

void LinearTransform::resetLinearTransOperator()
{
    auto linearTransOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("LinearTransform");
        std::vector<int64_t> srcShape({ batch, dimIn });
        std::vector<int64_t> matrixShape({ utils::divUp(dimIn, CUBE_ALIGN_SIZE), dimOut, (int64_t)CUBE_ALIGN_SIZE });
        std::vector<int64_t> biasShape({ dimOut });
        std::vector<int64_t> dstShape({ batch, dimOut });

        desc.addInputTensorDesc(ACL_FLOAT16, srcShape.size(), srcShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, matrixShape.size(), matrixShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, biasShape.size(), biasShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, dstShape.size(), dstShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = std::make_unique<AscendOperator>(desc);
    };

    for (auto batch : applyPageSizes) {
        linearTransOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        linearTransOpReset(linearTransOps[batch], batch);
    }
}

void LinearTransform::runLinearTransOperator(AscendTensor<float16_t, DIMS_2> &src,
    AscendTensor<float16_t, DIMS_3> &matrix, AscendTensor<float, DIMS_1> &bias, AscendTensor<float16_t, DIMS_2> &dst,
    aclrtStream stream)
{
    AscendOperator *linearTransOp = nullptr;
    int batch = src.getSize(0);
    if (linearTransOps.find(batch) != linearTransOps.end()) {
        linearTransOp = linearTransOps[batch].get();
    }
    ASCEND_ASSERT(linearTransOp);
    std::vector<const aclDataBuffer *> linearTransOpInput;
    linearTransOpInput.emplace_back(aclCreateDataBuffer(src.data(), src.getSizeInBytes()));
    linearTransOpInput.emplace_back(aclCreateDataBuffer(matrix.data(), matrix.getSizeInBytes()));
    linearTransOpInput.emplace_back(aclCreateDataBuffer(bias.data(), bias.getSizeInBytes()));

    std::vector<aclDataBuffer *> linearTransOpOutput;
    linearTransOpOutput.emplace_back(aclCreateDataBuffer(dst.data(), dst.getSizeInBytes()));

    linearTransOp->exec(linearTransOpInput, linearTransOpOutput, stream);

    for (auto &item : linearTransOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    linearTransOpInput.clear();

    for (auto &item : linearTransOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    linearTransOpOutput.clear();
}


void LinearTransform::updateTrainedValue(AscendTensor<float16_t, DIMS_2> &matrix, AscendTensor<float, DIMS_1> &bias)
{
    int dimN = matrix.getSize(0);
    int dimK = matrix.getSize(1);
    int dimBias = bias.getSize(0);
    ASCEND_THROW_IF_NOT_FMT(dimN == this->dimOut && dimK == this->dimIn && dimN == dimBias,
        "LinearTransform trained data's shape invalid.(dimA, dimb: (%d, %d), %d) vs (dimIn, dimOut: %d, %d)", dimN,
        dimK, dimBias, this->dimIn, this->dimOut);
    ASCEND_THROW_IF_NOT_FMT(dimN % CUBE_ALIGN_SIZE == 0,
        "LinearTransform matrix shape invalid. dimN should be divisible by 16, but here is %d", dimN);
    // matrix needs to be zN shaped
    AscendTensor<float16_t, DIMS_3> matrixFractal({ utils::divUp(dimK, CUBE_ALIGN_SIZE), dimN, (int)CUBE_ALIGN_SIZE });
    matrixFractal.initValue(0);
    // reshape code from zN shaped data.
    for (int i = 0; i < utils::divUp(dimK, CUBE_ALIGN_SIZE); i++) {
        int rsize = dimK - i * CUBE_ALIGN_SIZE;
        rsize = (rsize < CUBE_ALIGN_SIZE) ? rsize : CUBE_ALIGN_SIZE;
        auto dstPtr = matrixFractal.data() + i * dimN * CUBE_ALIGN_SIZE;
        auto srcPtr = matrix.data() + i * CUBE_ALIGN_SIZE;
        for (int j = 0; j < dimN; j++) {
            MEMCPY_S(dstPtr + j * CUBE_ALIGN_SIZE, rsize * sizeof(float16_t), srcPtr + j * dimK,
                rsize * sizeof(float16_t));
        }
    }
    this->matrix = matrixFractal;
    this->bias = bias;
    this->isTrained = true;
}

LinearTransform::~LinearTransform() {}
}
