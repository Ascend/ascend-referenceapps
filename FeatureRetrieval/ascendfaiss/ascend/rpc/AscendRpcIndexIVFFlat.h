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

#ifndef ASCEND_FAISS_RPC_INDEX_IVFFLAT_H
#define ASCEND_FAISS_RPC_INDEX_IVFFLAT_H

#include <faiss/ascend/rpc/AscendRpcCommon.h>

namespace faiss {
namespace ascend {
// Create/Destroy IVFFlat Index
RpcError RpcCreateIndexIVFFlat(rpcContext ctx, int &indexId, int dim, int nlist, faiss::MetricType metric,
    int nProbe = 1, int resourceSize = -1);
RpcError RpcDestroyIndexIVFFlat(rpcContext ctx, int indexId);

// add dataset to device
RpcError RpcIndexIVFFlatAdd(rpcContext ctx, int indexId, int n, int listId, 
    uint16_t *data, int dim,  uint32_t *ids);
} // namespace ascend
} // namespace faiss
#endif