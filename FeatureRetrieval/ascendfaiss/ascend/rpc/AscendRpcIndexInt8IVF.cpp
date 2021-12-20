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

#include <faiss/ascend/rpc/AscendRpcIndexIVF.h>
#include <faiss/ascend/rpc/HdcClient.h>

namespace faiss {
namespace ascend {
RpcError RpcIndexInt8IVFUpdateCoarseCent(rpcContext ctx, int indexId, int8_t *data,
    int dim, int total)
{
    RPC_LOG_INFO("update coarse cent of index %d on ctx %p\n", indexId, ctx);
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    IndexIVFUpdateCoarseCentRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_dim(dim);
    req.set_total(total);
    req.set_data(data, dim * total * sizeof(int8_t));

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_INT8_IVF_UPDATE_COARSE_CENT, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}

RpcError RpcIndexInt8IVFGetListCodes(rpcContext ctx, int indexId, int listId,
    std::vector<int8_t> &codes, std::vector<uint32_t> &ids)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    IndexIVFGetListInfoRequest req;
    IndexIVFGetListCodesResponse resp;
    req.set_indexid(indexId);
    req.set_listid(listId);

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_INT8_IVF_GET_LIST_CODES, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    size_t num = resp.ids().size() / sizeof(uint32_t);
    size_t codesLen = resp.codes().size();
    auto respCodes = reinterpret_cast<const int8_t *>(resp.codes().data());
    auto respIds = reinterpret_cast<const uint32_t *>(resp.ids().data());
    codes.assign(respCodes, respCodes + codesLen);
    ids.assign(respIds, respIds + num);

    return RPC_ERROR_NONE;
}

RpcError RpcIndexInt8IVFGetListLength(rpcContext ctx, int indexId, int listId, uint32_t &len)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    IndexIVFGetListInfoRequest req;
    IndexIVFGetListLengthResponse resp;
    req.set_indexid(indexId);
    req.set_listid(listId);

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_INT8_IVF_GET_LIST_LENGTH, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    len = resp.len();
    return RPC_ERROR_NONE;
}

RpcError RpcIndexInt8IVFUpdateNprobe(rpcContext ctx, int indexId, int nprobe)
{
    RPC_LOG_INFO("update nprobe(%d) of index %d on ctx %p\n", nprobe, indexId, ctx);
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    IndexIVFUpdateNprobeRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_nprobe(nprobe);

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_INT8_IVF_UPDATE_NPROBE, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}
} // namespace ascend
} // namespace faiss