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

#ifndef ASCEND_FAISS_RPC_INDEX_INT8_IVF_FLAT_H
#define ASCEND_FAISS_RPC_INDEX_INT8_IVF_FLAT_H

#include <faiss/ascend/rpc/AscendRpcCommon.h>

namespace faiss {
namespace ascend {
// create int8 flat index
RpcError RpcCreateIndexInt8IVFFlat(rpcContext ctx, int &indexId, int dim, int nlist,
    faiss::MetricType metric, int nProbe = 1, int resourceSize = -1);

// add dataset to device
RpcError RpcIndexInt8IVFFlatAdd(rpcContext ctx, int indexId, int n, int listId,
    int8_t *data, int dim,  uint32_t *ids);
} // namespace ascend
} // namespace faiss
#endif
