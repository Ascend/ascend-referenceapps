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

#include <ascenddaemon/impl/Int8L2Norm.h>
#include <ascenddaemon/utils/Limits.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/utils/AscendAssert.h>

namespace ascend {
namespace {
const int CUBE_ALIGN = 16;
const int SIZE_ALIGN = 8;
const int TRANSFER_SIZE = 256;
const int COMPUTE_BATCH = 16384;
const int L2NORM_OP_INPUT_NUM = 2;
const int L2NORM_OP_OUTPUT_NUM = 1;
}

Int8L2Norm::Int8L2Norm(idx_t d) : dims(d), 
    l2NormOpInput(L2NORM_OP_INPUT_NUM, nullptr),
    l2NormOpOutput(L2NORM_OP_OUTPUT_NUM, nullptr)
{
    transfer.resize(TRANSFER_SIZE * CUBE_ALIGN, 0);
    for (int i = 0; i < TRANSFER_SIZE / CUBE_ALIGN; ++i) {
        for (int j = 0; j < CUBE_ALIGN; ++j) {
            transfer[i * CUBE_ALIGN * CUBE_ALIGN + j * CUBE_ALIGN + j] = 1;
        }
    }

    resetL2NormOperator();
}

Int8L2Norm::~Int8L2Norm() {}

void Int8L2Norm::runL2NormOperator(AscendTensor<int8_t, DIMS_2> &vectors, 
                                   AscendTensor<float16_t, DIMS_2> &transfer,
                                   AscendTensor<uint32_t, DIMS_1> &actualNum,
                                   AscendTensor<float16_t, DIMS_1> &result, 
                                   aclrtStream stream)
{
    ASCEND_ASSERT(l2NormOp);
    
    // prepare for input data's buffer
    l2NormOpInput[0] = aclCreateDataBuffer(vectors.data(), vectors.getSizeInBytes()); // input 0
    l2NormOpInput[1] = aclCreateDataBuffer(transfer.data(), transfer.getSizeInBytes());     // input 1
    l2NormOpInput[2] = aclCreateDataBuffer(actualNum.data(), actualNum.getSizeInBytes());     // input 2

    // prepare for output data's buffer
    l2NormOpOutput[0] = aclCreateDataBuffer(result.data(), result.getSizeInBytes());  // output 0
    
    // async executing operator
    l2NormOp->exec(l2NormOpInput, l2NormOpOutput, stream);

    for (auto &item : l2NormOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }

    for (auto &item : l2NormOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
}


void Int8L2Norm::resetL2NormOperator()
{
    auto l2NormOpReset = [&](std::unique_ptr<AscendOperator> &l2NormOp, int vectorNum) {
        AscendOpDesc desc("Int8L2Norm");
        std::vector<int64_t> vectorShape({ vectorNum, dims });
        std::vector<int64_t> transferShape({ TRANSFER_SIZE, CUBE_ALIGN });
        std::vector<int64_t> actualNumShape({ SIZE_ALIGN });
        std::vector<int64_t> resultShape({ vectorNum });

        desc.addInputTensorDesc(ACL_INT8, vectorShape.size(), vectorShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, transferShape.size(), transferShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, actualNumShape.size(), actualNumShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, resultShape.size(), resultShape.data(), ACL_FORMAT_ND);

        l2NormOp.reset();
        l2NormOp = std::make_unique<AscendOperator>(desc);
    };

    l2NormOp = std::unique_ptr<AscendOperator>(nullptr);
    l2NormOpReset(l2NormOp, COMPUTE_BATCH);
}

void Int8L2Norm::dispatchL2NormTask(AscendTensor<int8_t, DIMS_2> &codesData, 
                                    AscendTensor<float16_t, DIMS_1> &normData,
                                    AscendTensor<uint32_t, DIMS_2> &actualNum,
                                    aclrtStream stream)
{
    ASCEND_ASSERT(normData.getSize(0) % CUBE_ALIGN == 0);

    // dispatch the l2 norm task
    int offset = 0;
    int codeSize = codesData.getSize(0);
    int times = utils::divUp(codeSize, COMPUTE_BATCH);
    for (int t = 0; t < times; ++t) {
        int size = std::min(codeSize - offset, COMPUTE_BATCH);
        int8_t* pCodes = codesData[offset].data();
        AscendTensor<int8_t, DIMS_2> vectorData(pCodes, { COMPUTE_BATCH, dims });
        AscendTensor<float16_t, DIMS_2> transferData(transfer.data(), { TRANSFER_SIZE, CUBE_ALIGN });
        auto actualSize = actualNum[t].view();
        actualSize[0] = static_cast<uint32_t>(size);
        AscendTensor<float16_t, DIMS_1> result(normData.data() + offset, { COMPUTE_BATCH });

        runL2NormOperator(vectorData, transferData, actualSize, result, stream);
        offset += COMPUTE_BATCH;
    }
}
} // namespace ascend