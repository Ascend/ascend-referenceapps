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

#include <faiss/ascend/rpc/AscendRpc.h>
#include <faiss/ascend/rpc/HdcClient.h>

namespace faiss {
namespace ascend {
static RpcError HdcRpcTranslateErrorCode(HdcRpcError &err)
{
    return (err == HDC_RPC_ERROR_NONE) ? RPC_ERROR_NONE : RPC_ERROR_ERROR;
}

RpcError RpcCreateContext(int deviceId, rpcContext *ctx)
{
    RPC_LOG_INFO("create context on device %d\n", deviceId);
    RPC_REQUIRE_NOT_NULL(ctx);
    RPC_ASSERT(deviceId >= 0);

    HdcSession *session = nullptr;
    HdcClient &client = HdcClient::GetInstance();
    HdcRpcError ret = client.Connect(deviceId, session);
    if (ret != HDC_RPC_ERROR_NONE) {
        *ctx = nullptr;
        return RPC_ERROR_ERROR;
    }

    *ctx = static_cast<rpcContext>(session);
    return RPC_ERROR_NONE;
}

RpcError RpcDestroyContext(rpcContext ctx)
{
    RPC_LOG_INFO("destroy context %p\n", ctx);
    RPC_REQUIRE_NOT_NULL(ctx);

    HdcSession *session = static_cast<HdcSession *>(ctx);
    HdcRpcError ret = HdcClient::GetInstance().DisConnect(session);
    return HdcRpcTranslateErrorCode(ret);
}

RpcError RpcDestroyIndex(rpcContext ctx, int indexId)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    DestroyIndexRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);

    HdcRpcError ret = session->SendAndReceive(RPC_DESTROY_INDEX, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}

RpcError RpcIndexSearch(rpcContext ctx, int indexId, int n, int dim, int k, const uint16_t *query, uint16_t *distance,
    uint32_t *label)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);
    RPC_REQUIRE_NOT_NULL(query);
    RPC_REQUIRE_NOT_NULL(distance);
    RPC_REQUIRE_NOT_NULL(label);

    IndexSearchRequest req;
    IndexSearchResponse resp;
    req.set_indexid(indexId);
    req.set_n(n);
    req.set_dim(dim);
    req.set_k(k);
    req.set_query(query, n * dim * sizeof(uint16_t));

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_SEARCH, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    RPC_ASSERT(resp.distance().size() == n * k * sizeof(uint16_t));
    RPC_ASSERT(resp.label().size() == n * k * sizeof(uint32_t));
    errno_t err = memcpy_s(distance, resp.distance().size(), resp.distance().data(), resp.distance().size());
    if (err != EOK) {
        return RPC_ERROR_ERROR;
    }
    err = memcpy_s(label, resp.label().size(), resp.label().data(), resp.label().size());
    if (err != EOK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}

RpcError RpcIndexRemoveIds(rpcContext ctx, int indexId, int n, uint32_t *ids, uint32_t *numRemoved)
{
    RPC_LOG_INFO("remove %d vector(s) of index %d on ctx %p\n", n, indexId, ctx);
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);
    RPC_REQUIRE_NOT_NULL(ids);
    RPC_REQUIRE_NOT_NULL(numRemoved);

    IndexRemoveIdsRequest req;
    IndexRemoveIdsResponse resp;
    req.set_indexid(indexId);
    req.set_n(n);
    req.set_ids(ids, n * sizeof(uint32_t));

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_REMOVE_IDS, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    *numRemoved = resp.num();
    return RPC_ERROR_NONE;
}

RpcError RpcIndexRemoveRangeIds(rpcContext ctx, int indexId, uint32_t min, uint32_t max, uint32_t *numRemoved)
{
    RPC_LOG_INFO("remove vector(s) [%d, %d] of index %d on ctx %p\n", min, max, indexId, ctx);
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_ASSERT(min <= max);
    RPC_REQUIRE_NOT_NULL(session);
    RPC_REQUIRE_NOT_NULL(numRemoved);

    IndexRemoveRangeIdsRequest req;
    IndexRemoveIdsResponse resp;
    req.set_indexid(indexId);
    req.set_min(min);
    req.set_max(max);

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_REMOVE_RANGE_IDS, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    *numRemoved = resp.num();
    return RPC_ERROR_NONE;
}

RpcError RpcIndexReset(rpcContext ctx, int indexId)
{
    RPC_LOG_INFO("Reset index %d on ctx %p\n", indexId, ctx);
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    IndexResetRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_RESET, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}

RpcError RpcIndexReserveMemory(rpcContext ctx, int &indexId, uint32_t numVec)
{
    IndexReserveMemRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_n(numVec);

    CALL_RPC(ctx, RPC_INDEX_RESERVE_MEM, req, resp, resp.err());
    return RPC_ERROR_NONE;
}

RpcError RpcIndexReclaimMemory(rpcContext ctx, int &indexId, uint32_t &sizeMem)
{
    IndexReclaimMemRequest req;
    IndexReclaimMemResponse resp;
    req.set_indexid(indexId);

    CALL_RPC(ctx, RPC_INDEX_RECLAIM_MEM, req, resp, resp.result().err());
    sizeMem = resp.size();
    return RPC_ERROR_NONE;
}

RpcError RpcTestDataIntegrity(rpcContext ctx, const std::vector<uint8_t> &data)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    TestDataIntegrityRequest req;
    TestDataIntegrityResponse resp;

    req.set_len(data.size());
    req.set_data(data.data(), data.size());
    const uint8_t checksum = calcChecksum(data.data(), data.size());
    req.set_checksum(checksum);

    HdcRpcError ret = session->SendAndReceive(RPC_TEST_DATA_INTEGRITY, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    const std::string &receivedData = resp.data();
    if (receivedData.size() != data.size() || checksum != calcChecksum(receivedData.data(), receivedData.size())) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}
} // namespace ascend
} // namespace faiss
