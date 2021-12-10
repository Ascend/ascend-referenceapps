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

#include <ascenddaemon/utils/AscendOperator.h>

namespace ascend {
AscendOperator::AscendOperator(AscendOpDesc &desc)
    : opDesc(std::move(desc)),
      handle(nullptr),
      numInputs(opDesc.inputDesc.size()),
      numOutputs(opDesc.outputDesc.size())
{
    if (ACL_ERROR_NONE != aclopCreateHandle(opDesc.opType.c_str(),
        numInputs,
        opDesc.inputDesc.data(),
        numOutputs,
        opDesc.outputDesc.data(),
        opDesc.opAttr,
        &handle)) {
        ASCEND_THROW_FMT("Create op handle %s failed", opDesc.opType.c_str());
    }
}

AscendOperator::~AscendOperator()
{
    if (handle) {
        aclopDestroyHandle(handle);
        handle = nullptr;
    }
}

size_t AscendOperator::getInputNumDims(int index)
{
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.inputDesc.size()));
    return aclGetTensorDescNumDims(opDesc.inputDesc[index]);
}

int64_t AscendOperator::getInputDim(int index, int dimIndex)
{
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.inputDesc.size()));
    ASCEND_THROW_IF_NOT(dimIndex < static_cast<int>(getInputNumDims(index)));
#ifdef USE_ACL_INTERFACE_V2
    int64_t dimSize;
    ACL_REQUIRE_OK(aclGetTensorDescDimV2(opDesc.inputDesc[index], dimIndex, &dimSize));
    return dimSize;
#else
    return aclGetTensorDescDim(opDesc.outputDesc[index], dimIndex);
#endif
}

size_t AscendOperator::getInputSize(int index)
{
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.inputDesc.size()));
    return aclGetTensorDescSize(opDesc.inputDesc[index]);
}

size_t AscendOperator::getOutputNumDims(int index)
{
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.outputDesc.size()));
    return aclGetTensorDescNumDims(opDesc.outputDesc[index]);
}

int64_t AscendOperator::getOutputDim(int index, int dimIndex)
{
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.outputDesc.size()));
    ASCEND_THROW_IF_NOT(dimIndex < static_cast<int>(getOutputNumDims(index)));
#ifdef USE_ACL_INTERFACE_V2
    int64_t dimSize;
    ACL_REQUIRE_OK(aclGetTensorDescDimV2(opDesc.outputDesc[index], dimIndex, &dimSize));
    return dimSize;
#else
    return aclGetTensorDescDim(opDesc.outputDesc[index], dimIndex);
#endif
}

size_t AscendOperator::getOutputSize(int index)
{
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.outputDesc.size()));
    return aclGetTensorDescSize(opDesc.outputDesc[index]);
}

void AscendOperator::exec(std::vector<const aclDataBuffer *>& inputBuffers,
                          std::vector<aclDataBuffer *>& outputBuffers, aclrtStream stream)
{
    ACL_REQUIRE_OK(aclopExecWithHandle(handle,
                                       numInputs,
                                       inputBuffers.data(),
                                       numOutputs,
                                       outputBuffers.data(),
                                       stream));
}
}  // namespace ascend