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
#include <string>
#include <unordered_map>
#include <faiss/ascend/utils/AscendThreadPool.h>

namespace faiss {
namespace ascend {
using rpcContext = void *;

class AscendNNDimReduction {
public:
    AscendNNDimReduction(std::vector<int> deviceList, int dimIn, int dimOut, int batchSize, std::string &modelPath);

    ~AscendNNDimReduction();

    // Handles paged infer if the infer set is too large
    void infer(int n, const std::vector<float> &inputData, std::vector<float> &outputData);

    void infer(int n, const float *inputData, std::vector<float> &outputData);

public:
    // Whether to print infer log
    bool verbose;

private:
    // Init rpc context
    void initRpcCtx();

    // Handles paged infer if the infer set is too large, passes to
    // inferImpl to actually perform the infer for the current page
    void inferPaged(int n, const float *x, std::vector<float> &outputData);

    // Actually perform the infer
    void inferImpl(int n, const float *x, std::vector<float> &outputData);

    // Get the size of memory every database vector needed to store
    int getElementSize() const;

    // Destroy rpc context
    void clearRpcCtx();

private:
    // Vector dimension before reduction dimension
    int dimIn;

    // Vector dimension after reduction dimension
    int dimOut;

    // The number of samples selected by the model for one inference
    int batchSize;

    // The infer model path
    std::string modelPath;

    // The chip ID for inferring
    std::vector<int> deviceList;

    // Device --> Context
    std::unordered_map<int, rpcContext> contextMap;

    // Thread pool for multithread processing
    AscendThreadPool *pool;
};
} // ascend
} // faiss
#endif // ASCEND_NN_DIM_REDUCTION_INCLUDED