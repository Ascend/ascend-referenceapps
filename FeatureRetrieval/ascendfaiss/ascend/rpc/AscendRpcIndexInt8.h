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

#ifndef ASCEND_FAISS_RPC_INDEX_INT8_H
#define ASCEND_FAISS_RPC_INDEX_INT8_H

#include <faiss/ascend/rpc/AscendRpcCommon.h>

namespace faiss {
namespace ascend {
// destory index
RpcError RpcDestroyIndexInt8(rpcContext ctx, int indexId);

// index search
RpcError RpcIndexInt8Search(rpcContext ctx, int indexId, int n, int dim, int k,
    const int8_t *query, uint16_t *distance, uint32_t *label);

// remove ids
RpcError RpcIndexInt8RemoveIds(rpcContext ctx, int indexId, int n, uint32_t *ids, uint32_t *numRemoved);

// remove range ids
RpcError RpcIndexInt8RemoveRangeIds(rpcContext ctx, int indexId, uint32_t min, uint32_t max, uint32_t *numRemoved);

// reset index
RpcError RpcIndexInt8Reset(rpcContext ctx, int indexId);

// reserve or reclaim device memory for database
RpcError RpcIndexInt8ReserveMemory(rpcContext ctx, int &indexId, uint32_t numVec);
RpcError RpcIndexInt8ReclaimMemory(rpcContext ctx, int &indexId, uint32_t &sizeMem);
} // namespace rpc
} // namespace ascend

#endif
