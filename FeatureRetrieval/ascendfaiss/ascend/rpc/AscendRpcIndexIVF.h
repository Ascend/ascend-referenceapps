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

#ifndef ASCEND_FAISS_RPC_INDEX_IVF_H
#define ASCEND_FAISS_RPC_INDEX_IVF_H

#include <faiss/ascend/rpc/AscendRpcCommon.h>
#include <faiss/IndexScalarQuantizer.h>

namespace faiss {
namespace ascend {
// Create/Destroy IVFPQ Index
RpcError RpcCreateIndexIVFPQ(rpcContext ctx, int &indexId, int dim, int nlist, int subQuantizers, int bitsPerCode,
    faiss::MetricType metric, int nProbe = 1, int resourceSize = -1);
RpcError RpcDestroyIndexIVFPQ(rpcContext ctx, int indexId);

// Create/Destroy IVFSQ Index
RpcError RpcCreateIndexIVFSQ(rpcContext ctx, int &indexId, int dim, int nlist, bool byResidual,
    faiss::ScalarQuantizer::QuantizerType qtype, faiss::MetricType metric, int nProbe = 1, int resourceSize = -1);
RpcError RpcDestroyIndexIVFSQ(rpcContext ctx, int indexId);

// update trained value to device
RpcError RpcIndexIVFUpdateCoarseCent(rpcContext ctx, int indexId, uint16_t *data, int dim, int total);
RpcError RpcIndexIVFPQUpdatePQCent(rpcContext ctx, int indexId, uint16_t *data,
    int m, int ksub, int dsub);
RpcError RpcIndexSQUpdateTrainedValue(rpcContext ctx, int indexId, int dim, uint16_t *vmin, 
    uint16_t *vdiff, bool isIvfSQ = false);

// add dataset to device
RpcError RpcIndexIVFPQAdd(rpcContext ctx, int indexId, int n, int listId,
    uint8_t *codes, int codeSize, uint32_t *ids);
RpcError RpcIndexIVFSQAdd(rpcContext ctx, int indexId, int n, int listId, 
    const uint8_t *codes, int codeSize, const uint32_t *ids, const float *precomputedVal);

// get specific list's code and indices
RpcError RpcIndexIVFGetListCodes(rpcContext ctx, int indexId, int listId,
    std::vector<uint8_t> &codes, std::vector<uint32_t> &ids);

// get all lists data and add to ivfIndex 
RpcError RpcIndexIVFFastGetListCodes(rpcContext ctx, int indexId, int nlist, 
    InvertedLists *ivfIndex);

// get specific list's length
RpcError RpcIndexIVFGetListLength(rpcContext ctx, int indexId, int listId, uint32_t &len);

// change ivf nprobe
RpcError RpcIndexIVFUpdateNprobe(rpcContext ctx, int indexId, int nprobe);
} // namespace ascend
} // namespace faiss
#endif
