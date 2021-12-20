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

#ifndef ASCEND_FAISS_RPC_INDEX_INT8_FLAT_H
#define ASCEND_FAISS_RPC_INDEX_INT8_FLAT_H

#include <faiss/ascend/rpc/AscendRpcCommon.h>

namespace faiss {
namespace ascend {
// create/destroy int8 flat index
RpcError RpcCreateIndexInt8Flat(rpcContext ctx, int &indexId, int dim, 
    faiss::MetricType metric, int resourceSize = -1);

// add dataset to device
RpcError RpcIndexInt8FlatAdd(rpcContext ctx, int indexId, int n, int dim, int8_t *data);

// get base data vector[offset, offset + num)
RpcError RpcIndexInt8FlatGetBase(rpcContext ctx, int indexId, uint32_t offset,
    uint32_t num, std::vector<int8_t> &vectors, faiss::MetricType metric);

// get dataset size
RpcError RpcIndexInt8FlatGetBaseSize(rpcContext ctx, int indexId, uint32_t &size, faiss::MetricType metric);
} // namespace ascend
} // namespace faiss
#endif
