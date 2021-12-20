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

#ifndef ASCEND_NN_DIM_REDUCTION_INCLUDED
#define ASCEND_NN_DIM_REDUCTION_INCLUDED

#include <vector>
#include <memory>
#include "ascenddaemon/utils/ModelProcess.h"

namespace ascend {
class NNDimReduction {
public:
    NNDimReduction(uint32_t ntotal, uint32_t dimIn, uint32_t dimOut, uint32_t batchSize, 
                        float* data);

    ~NNDimReduction();

    // Call the chip to perform inference
    void Process(std::shared_ptr<ModelProcess> processModel);

    // Return the inference result
    const std::vector<float>& GetResultData();

private:
    // Number of inferences
    uint32_t num;

    // Input data dimension
    uint32_t inputDim;

    // Output data dimension
    uint32_t outputDim;

    // Batch size for inference 
    uint32_t batchSize;

    // Input data
    float *inputData;

    // Output data
    std::vector<float> outputData;
};
}
#endif // ASCEND_NN_DIM_REDUCTION_INCLUDED
