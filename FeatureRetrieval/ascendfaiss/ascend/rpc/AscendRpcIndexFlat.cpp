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

#include <faiss/ascend/rpc/AscendRpcIndexFlat.h>
#include <faiss/ascend/rpc/HdcClient.h>

namespace faiss {
namespace ascend {
RpcError RpcCreateIndexFlat(rpcContext ctx, int &indexId, int dim, faiss::MetricType metric, int resourceSize)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    CreateIndexFlatRequest req;
    CreateIndexResponse resp;
    req.set_dim(dim);
    req.set_metric(metric);
    req.set_resource(resourceSize);

    HdcRpcError ret = session->SendAndReceive(RPC_CREATE_INDEX_FLAT, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    indexId = resp.indexid();
    return RPC_ERROR_NONE;
}

RpcError RpcDestroyIndexFlat(rpcContext ctx, int indexId)
{
    return RpcDestroyIndex(ctx, indexId);
}

RpcError RpcIndexFlatAdd(rpcContext ctx, int indexId, int n, int dim, uint16_t *data)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);
    RPC_REQUIRE_NOT_NULL(data);

    IndexFlatAddRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_n(n);
    req.set_dim(dim);
    req.set_vectors(data, n * dim * sizeof(uint16_t));

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_FLAT_ADD, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}

RpcError RpcIndexFlatGetBase(rpcContext ctx, int indexId, uint32_t offset,
    uint32_t num, std::vector<uint16_t> &vectors)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    IndexFlatGetBaseRequest req;
    IndexFlatGetBaseResponse resp;
    req.set_indexid(indexId);
    req.set_offset(offset);
    req.set_num(num);

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_FLAT_GET_BASE, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    size_t totalNum = resp.data().size() / sizeof(uint16_t);
    auto respBase = reinterpret_cast<const uint16_t *>(resp.data().data());
    vectors.assign(respBase, respBase + totalNum);

    return RPC_ERROR_NONE;
}

RpcError RpcIndexFlatGetBaseSize(rpcContext ctx, int indexId, uint32_t &size)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    IndexFlatGetBaseLengthRequest req;
    IndexFlatGetBaseLengthResponse resp;
    req.set_indexid(indexId);

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_FLAT_GET_BASE_SIZE, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    size = resp.len();
    return RPC_ERROR_NONE;
}
} // namespace ascend
} // namespace faiss