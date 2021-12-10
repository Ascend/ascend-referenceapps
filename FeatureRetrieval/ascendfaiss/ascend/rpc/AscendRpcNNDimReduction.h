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

#ifndef ASCEND_FAISS_RPC_NN_DIM_REDUCTION_H
#define ASCEND_FAISS_RPC_NN_DIM_REDUCTION_H

#include <faiss/ascend/rpc/AscendRpcCommon.h>

namespace faiss {
namespace ascend {
// create rpc context
RpcError RpcCreateNNDimReduction(rpcContext ctx, std::string &modelPath);

// neural network infer 
RpcError RpcInferNNDimReduction(rpcContext ctx, int n, int dimIn, int dimOut, int batchSize,
                                const float* data, std::vector<float> &outputData);

// destroy rpc context                     
RpcError RpcDestroyNNDimReduction(rpcContext ctx);
} // namespace ascend
} // namespace faiss
#endif
