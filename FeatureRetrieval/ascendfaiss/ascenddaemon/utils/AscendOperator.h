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

#ifndef ASCEND_OPERATOR_INCLUDED
#define ASCEND_OPERATOR_INCLUDED

#include <ascenddaemon/utils/AscendOpDesc.h>
#include <ascenddaemon/utils/AscendUtils.h>
#include <vector>

namespace ascend {
class AscendOperator {
public:
    explicit AscendOperator(AscendOpDesc &opDesc);

    ~AscendOperator();

    void exec(std::vector<const aclDataBuffer *>& inputBuffers,
              std::vector<aclDataBuffer *>& outputBuffers, aclrtStream stream);

    inline int getNumInputs() const
    {
        return numInputs;
    }

    inline int getNumOutputs() const
    {
        return numOutputs;
    }

    size_t getInputNumDims(int index);

    int64_t getInputDim(int index, int dimIndex);

    size_t getInputSize(int index);

    size_t getOutputNumDims(int index);

    int64_t getOutputDim(int index, int dimIndex);

    size_t getOutputSize(int index);

private:
    AscendOpDesc opDesc;
    aclopHandle *handle;
    int numInputs;
    int numOutputs;
};
}  // namespace ascend

#endif  // ASCEND_OPERATOR_INCLUDED
