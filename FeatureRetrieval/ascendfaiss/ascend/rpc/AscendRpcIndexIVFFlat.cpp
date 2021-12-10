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

#include <faiss/ascend/rpc/AscendRpcIndexIVFFlat.h>
#include <faiss/ascend/rpc/HdcClient.h>

namespace faiss {
namespace ascend {
RpcError RpcCreateIndexIVFFlat(rpcContext ctx, int &indexId, int dim, int nlist, faiss::MetricType metric, int nProbe,
    int resourceSize)
{
    CreateIndexIVFFlatRequest req;
    CreateIndexResponse resp;
    req.set_dim(dim);
    req.set_nlist(nlist);
    req.set_metric(metric);
    req.set_nprobe(nProbe);
    req.set_resource(resourceSize);

    CALL_RPC(ctx, RPC_CREATE_INDEX_IVFFLAT, req, resp, resp.result().err());
    indexId = resp.indexid();
    return RPC_ERROR_NONE;
}

RpcError RpcDestroyIndexIVFFlat(rpcContext ctx, int indexId)
{
    return RpcDestroyIndex(ctx, indexId);
}

RpcError RpcIndexIVFFlatAdd(rpcContext ctx, int indexId, int n, int listId, uint16_t *data, int dim, uint32_t *ids)
{
    IndexIVFFlatAddRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_n(n);
    req.set_listid(listId);
    req.set_dim(dim);
    req.set_vectors(data, n * dim * sizeof(uint16_t));
    req.set_ids(ids, n * sizeof(uint32_t));

    CALL_RPC(ctx, RPC_INDEX_IVFFLAT_ADD, req, resp, resp.err());
    return RPC_ERROR_NONE;
}
} // namespace ascend
} // namespace faiss