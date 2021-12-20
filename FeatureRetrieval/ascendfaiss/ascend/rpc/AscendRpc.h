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

#ifndef ASCEND_FAISS_RPC_CONTEXT_H
#define ASCEND_FAISS_RPC_CONTEXT_H

#include <faiss/ascend/rpc/AscendRpcCommon.h>
#include <faiss/ascend/rpc/AscendRpcIndexFlat.h>
#include <faiss/ascend/rpc/AscendRpcIndexSQ.h>
#include <faiss/ascend/rpc/AscendRpcIndexIVFFlat.h>
#include <faiss/ascend/rpc/AscendRpcIndexIVF.h>
#include <faiss/ascend/rpc/AscendRpcIndexPreTransform.h>
#include <faiss/ascend/rpc/AscendRpcIndexInt8.h>
#include <faiss/ascend/rpc/AscendRpcIndexInt8Flat.h>
#include <faiss/ascend/rpc/AscendRpcIndexInt8IVF.h>
#include <faiss/ascend/rpc/AscendRpcIndexInt8IVFFlat.h>

namespace faiss {
namespace ascend {
// create/destroy rpc context
RpcError RpcCreateContext(int deviceId, rpcContext *ctx);
RpcError RpcDestroyContext(rpcContext);

// index search
RpcError RpcIndexSearch(rpcContext ctx, int indexId, int n, int dim, int k,
    const uint16_t *query, uint16_t *distance, uint32_t *label);

// remove ids
RpcError RpcIndexRemoveIds(rpcContext ctx, int indexId, int n, uint32_t *ids, uint32_t *numRemoved);
// remove range ids
RpcError RpcIndexRemoveRangeIds(rpcContext ctx, int indexId, uint32_t min, uint32_t max, uint32_t *numRemoved);

// reset index
RpcError RpcIndexReset(rpcContext ctx, int indexId);

// reserve or reclaim device memory for database
RpcError RpcIndexReserveMemory(rpcContext ctx, int &indexId, uint32_t numVec);
RpcError RpcIndexReclaimMemory(rpcContext ctx, int &indexId, uint32_t &sizeMem);

// for test
RpcError RpcTestDataIntegrity(rpcContext ctx, const std::vector<uint8_t> &data);
} // namespace rpc
} // namespace ascend

#endif
