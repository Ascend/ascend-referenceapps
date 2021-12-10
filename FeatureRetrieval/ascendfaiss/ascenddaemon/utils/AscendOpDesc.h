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

#ifndef ASCEND_OP_DESC_INCLUDED
#define ASCEND_OP_DESC_INCLUDED

#include <ascenddaemon/utils/AscendUtils.h>
#include <string>
#include <vector>

namespace ascend {
class AscendOpDesc {
public:
    AscendOpDesc(std::string opName);

    AscendOpDesc(AscendOpDesc &&desc);

    ~AscendOpDesc();

    AscendOpDesc &addInputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);
    AscendOpDesc &addOutputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

    std::string opType;
    std::vector<const aclTensorDesc *> inputDesc;
    std::vector<const aclTensorDesc *> outputDesc;
    aclopAttr *opAttr;
};
}  // namespace ascend

#endif  // ASCEND_OP_DESC_INCLUDED
