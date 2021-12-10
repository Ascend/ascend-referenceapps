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

#ifndef ASCEND_FAISS_RPC_INDEX_PRETRANSFORM_H
#define ASCEND_FAISS_RPC_INDEX_PRETRANSFORM_H

#include <faiss/ascend/rpc/AscendRpcCommon.h>

namespace faiss {
namespace ascend {
// Create/Destroy PreTransform Index
RpcError RpcCreateIndexPreTransform(rpcContext ctx, int &indexId, int subindexId);
RpcError RpcDestroyIndexPreTransform(rpcContext ctx, int indexId);

// Create/Destroy LinearTransform
RpcError RpcCreateLinearTransform(rpcContext ctx, int &transformId, int dimIn, 
    int dimOut, bool haveBias);
RpcError RpcDestroyLinearTransform(rpcContext ctx, int transformId);

// update trained value to device
RpcError RpcLinearTransformUpdateTrainedValue(rpcContext ctx, int transformId, 
    int dimIn, int dimOut, const uint16_t *matrix, const float *bias = nullptr);

// add vector transform to IndexPretransform
RpcError RpcIndexPreTransformPrepend(rpcContext ctx, int indexId,
    int transformId);
} // namespace ascend
} // namespace faiss
#endif
