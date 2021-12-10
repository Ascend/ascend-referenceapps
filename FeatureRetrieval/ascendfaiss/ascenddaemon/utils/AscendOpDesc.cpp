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

#include <ascenddaemon/utils/AscendOpDesc.h>

namespace ascend {
AscendOpDesc::AscendOpDesc(std::string opName)
    : opType(std::move(opName))
{
    opAttr = aclopCreateAttr();
}

AscendOpDesc::AscendOpDesc(AscendOpDesc &&desc)
{
    opType = std::move(desc.opType);
    opAttr = desc.opAttr;
    desc.opAttr = nullptr;

    inputDesc = std::move(desc.inputDesc);
    outputDesc = std::move(desc.outputDesc);
}

AscendOpDesc::~AscendOpDesc()
{
    for (auto desc : inputDesc) {
        aclDestroyTensorDesc(desc);
    }

    for (auto desc : outputDesc) {
        aclDestroyTensorDesc(desc);
    }

    if (opAttr) {
        aclopDestroyAttr(opAttr);
        opAttr = nullptr;
    }
}

AscendOpDesc &AscendOpDesc::addInputTensorDesc(aclDataType dataType,
                                               int numDims,
                                               const int64_t *dims,
                                               aclFormat format)
{
    inputDesc.push_back(aclCreateTensorDesc(dataType, numDims, dims, format));
    return *this;
}

AscendOpDesc &AscendOpDesc::addOutputTensorDesc(aclDataType dataType,
                                                int numDims,
                                                const int64_t *dims,
                                                aclFormat format)
{
    outputDesc.push_back(aclCreateTensorDesc(dataType, numDims, dims, format));
    return *this;
}
}  // namespace ascend