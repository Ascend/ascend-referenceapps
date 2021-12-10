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

#ifndef ASCEND_FAISS_RPC_INDEX_INT8_IVF_H
#define ASCEND_FAISS_RPC_INDEX_INT8_IVF_H

#include <faiss/ascend/rpc/AscendRpcCommon.h>
#include <faiss/IndexScalarQuantizer.h>

namespace faiss {
namespace ascend {
// update trained value to device
RpcError RpcIndexInt8IVFUpdateCoarseCent(rpcContext ctx, int indexId, int8_t *data, int dim, int total);

// get specific list's code and indices
RpcError RpcIndexInt8IVFGetListCodes(rpcContext ctx, int indexId, int listId,
    std::vector<int8_t> &codes, std::vector<uint32_t> &ids);

// get specific list's length
RpcError RpcIndexInt8IVFGetListLength(rpcContext ctx, int indexId, int listId, uint32_t &len);

// change ivf nprobe
RpcError RpcIndexInt8IVFUpdateNprobe(rpcContext ctx, int indexId, int nprobe);
} // namespace ascend
} // namespace faiss
#endif
