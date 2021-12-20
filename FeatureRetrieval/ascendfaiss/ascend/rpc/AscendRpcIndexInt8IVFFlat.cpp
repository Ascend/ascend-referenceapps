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

#include <faiss/ascend/rpc/AscendRpcIndexInt8IVFFlat.h>
#include <faiss/ascend/rpc/HdcClient.h>

namespace faiss {
namespace ascend {
RpcError RpcCreateIndexInt8IVFFlat(rpcContext ctx, int &indexId, int dim, int nlist, faiss::MetricType metric,
    int nProbe, int resourceSize)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    CreateIndexInt8IVFFlatRequest req;
    CreateIndexResponse resp;
    req.set_dim(dim);
    req.set_nlist(nlist);
    req.set_metric(metric);
    req.set_nprobe(nProbe);
    req.set_resource(resourceSize);

    HdcRpcError ret = session->SendAndReceive(RPC_CREATE_INDEX_INT8_IVFFLAT, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    indexId = resp.indexid();
    return RPC_ERROR_NONE;
}

RpcError RpcIndexInt8IVFFlatAdd(rpcContext ctx, int indexId, int n, int listId, int8_t *data, int dim, uint32_t *ids)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);
    RPC_REQUIRE_NOT_NULL(data);

    IndexInt8IVFFlatAddRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_n(n);
    req.set_listid(listId);
    req.set_dim(dim);
    req.set_vectors(data, n * dim * sizeof(int8_t));
    req.set_ids(ids, n * sizeof(uint32_t));

    CALL_RPC(ctx, RPC_INDEX_INT8_IVFFLAT_ADD, req, resp, resp.err());
    return RPC_ERROR_NONE;
}
} // namespace ascend
} // namespace faiss