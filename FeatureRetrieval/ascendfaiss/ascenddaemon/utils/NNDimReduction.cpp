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

#include <ascenddaemon/utils/NNDimReduction.h>
#include <ascenddaemon/utils/AscendAssert.h>
#include "acl/acl.h"

namespace ascend {
NNDimReduction::NNDimReduction(uint32_t ntotal, uint32_t dimIn, uint32_t dimOut, uint32_t batchSize,
    float *data)
    : num(ntotal), inputDim(dimIn), outputDim(dimOut), batchSize(batchSize), inputData(data)
{}

NNDimReduction::~NNDimReduction() {}

void NNDimReduction::Process(std::shared_ptr<ModelProcess> processModel)
{
    //  execute with batch
    uint32_t cnt = 0;
    uint32_t pos = 0;
    for (uint32_t index = 0; index < num; ++index) {
        cnt++;
        if (cnt % batchSize == 0) {
            void *dataDevBuffer = inputData + pos;
            uint32_t devDataBufferSize = sizeof(float) * inputDim * batchSize;
            ASCEND_THROW_IF_NOT_FMT(dataDevBuffer != nullptr, "get device buffer failed,index is %zu", index);

            // model process
            processModel->CreateInput(dataDevBuffer, devDataBufferSize);

            processModel->Execute();

            processModel->DestroyInput();

            const aclmdlDataset *modelOutput = processModel->GetModelOutputData();
            aclDataBuffer *modelOutputAdrr = aclmdlGetDatasetBuffer(modelOutput, 0);
            float *resultDataAdrr = (float *)aclGetDataBufferAddr(modelOutputAdrr);
            size_t resultDataSize = aclGetDataBufferSizeV2(modelOutputAdrr);

            outputData.insert(outputData.end(), resultDataAdrr, resultDataAdrr + resultDataSize / sizeof(float));
            ASCEND_THROW_IF_NOT_MSG(modelOutput != nullptr, "get model output data failed");

            pos += batchSize * inputDim;
        }
    }
    // If num can't be divisible, execute the remaining data by one batchSize
    if (cnt % batchSize != 0) {
        std::vector<float> lastData;
        float *dataDevBuffer = inputData + pos;
        lastData.insert(lastData.end(), dataDevBuffer, dataDevBuffer + inputDim * (cnt % batchSize));
        lastData.resize(batchSize * inputDim);
        ASCEND_THROW_IF_NOT_MSG(dataDevBuffer != nullptr, "get device buffer failed");

        // model process
        processModel->CreateInput(lastData.data(), lastData.size() * sizeof(float));

        processModel->Execute();

        processModel->DestroyInput();

        const aclmdlDataset *modelOutput = processModel->GetModelOutputData();
        aclDataBuffer *modelOutputAdrr = aclmdlGetDatasetBuffer(modelOutput, 0);
        float *resultDataAdrr = (float *)aclGetDataBufferAddr(modelOutputAdrr);
        // Only intercept valid data
        outputData.insert(outputData.end(), resultDataAdrr, resultDataAdrr + outputDim * (cnt % batchSize));
        ASCEND_THROW_IF_NOT_MSG(modelOutput != nullptr, "get model output data failed");
    }
}

const std::vector<float> &NNDimReduction::GetResultData()
{
    return outputData;
}
} // namespace ascend
