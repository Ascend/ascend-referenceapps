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
#include <faiss/ascend/AscendIndexIVF.h>
#include <algorithm>
#include <mutex>

namespace faiss {
namespace ascend {
RpcError RpcCreateIndexIVFPQ(rpcContext ctx, int &indexId, int dim, int nlist,
    int subQuantizers, int bitsPerCode, faiss::MetricType metric, int nProbe, int resourceSize)
{
    RPC_LOG_INFO("create index on ctx %p\n", ctx);
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    CreateIndexIVFPQRequest req;
    CreateIndexResponse resp;
    req.set_dim(dim);
    req.set_nlist(nlist);
    req.set_subquantizers(subQuantizers);
    req.set_bitspercode(bitsPerCode);
    req.set_metric(metric);
    req.set_nprobe(nProbe);
    req.set_resource(resourceSize);

    HdcRpcError ret = session->SendAndReceive(RPC_CREATE_INDEX_IVFPQ, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    indexId = resp.indexid();
    return RPC_ERROR_NONE;
}

RpcError RpcDestroyIndexIVFPQ(rpcContext ctx, int indexId)
{
    return RpcDestroyIndex(ctx, indexId);
}

RpcError RpcIndexIVFUpdateCoarseCent(rpcContext ctx, int indexId, uint16_t *data,
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
    req.set_data(data, dim * total * sizeof(uint16_t));

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_IVF_UPDATE_COARSE_CENT, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}

RpcError RpcIndexIVFPQUpdatePQCent(rpcContext ctx, int indexId, uint16_t *data,
    int m, int ksub, int dsub)
{
    RPC_LOG_INFO("update PQ cent of index %d on ctx %p\n", indexId, ctx);
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);
    RPC_REQUIRE_NOT_NULL(data);

    IndexIVFPQUpdatePQCentRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_m(m);
    req.set_ksub(ksub);
    req.set_dsub(dsub);
    req.set_data(data, m * ksub * dsub * sizeof(uint16_t));

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_IVFPQ_UPDATE_PQ_CENT, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}

RpcError RpcIndexIVFPQAdd(rpcContext ctx, int indexId, int n, int listId,
    uint8_t *codes, int codeSize, uint32_t *ids)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);
    RPC_REQUIRE_NOT_NULL(codes);
    RPC_REQUIRE_NOT_NULL(ids);

    IndexIVFPQAddRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_n(n);
    req.set_listid(listId);
    req.set_codesize(codeSize);
    req.set_codes(codes, n * codeSize);
    req.set_ids(ids, n * sizeof(uint32_t));

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_IVFPQ_ADD, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}

RpcError RpcIndexIVFGetListCodes(rpcContext ctx, int indexId, int listId,
    std::vector<uint8_t> &codes, std::vector<uint32_t> &ids)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    IndexIVFGetListInfoRequest req;
    IndexIVFGetListCodesResponse resp;
    req.set_indexid(indexId);
    req.set_listid(listId);

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_IVF_GET_LIST_CODES, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    size_t num = resp.ids().size() / sizeof(uint32_t);
    size_t codesLen = resp.codes().size();
    auto respCodes = reinterpret_cast<const uint8_t *>(resp.codes().data());
    auto respIds = reinterpret_cast<const uint32_t *>(resp.ids().data());
    codes.assign(respCodes, respCodes + codesLen);
    ids.assign(respIds, respIds + num);

    return RPC_ERROR_NONE;
}

RpcError RpcIndexIVFFastGetListCodes(rpcContext ctx, int indexId, int nlist, InvertedLists *ivf)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    IndexIVFGetListInfoRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_listid(nlist);

    std::lock_guard<std::mutex> guard(session->GetSessionLock());
    HdcRpcError ret = session->SerializeAndSendMessage(RPC_INDEX_IVF_FAST_GET_LIST_CODES, true, req);
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    // prepare fast recv
    ret = session->HdcFastRecvPrepare();
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    void* dataBuf = nullptr;
    void* ctrlBuf = nullptr;

    const int idxListId = 0;
    const int idxListSize = 1;
    const int idxListCount = 2;

    int channelNumber = session->HdcGetFastSendChannel();

    for (int i = 0; i < nlist; i += channelNumber) {
        for (int j = i; j < i + channelNumber; j++) {
            if (j >= nlist) {
                continue;
            }
            int channelId = j % channelNumber;
            // fast recv
            ret = session->HdcFastRecv(&dataBuf, &ctrlBuf, channelId);
            RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

            auto codes = reinterpret_cast<uint8_t*>(dataBuf);
            int* recvCtrlBuf = reinterpret_cast<int*>(ctrlBuf);
            int listId = recvCtrlBuf[idxListId];
            int listSize = recvCtrlBuf[idxListSize];
            if (listSize > 0) {
                int codeSize = listSize * recvCtrlBuf[idxListCount];
                auto ids = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(dataBuf) + codeSize);
                std::vector<Index::idx_t> indice(listSize);
                std::transform(ids, ids + listSize, begin(indice), [](uint32_t x) { return Index::idx_t(x); });
                ivf->add_entries(listId, indice.size(), indice.data(), codes);
            }
        }
        // recv sync
        ret = session->HdcSendRecvSignal();
        RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);
    }

    // fast recv release
    ret = session->HdcFastRecvRelease();
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    ret = session->RecvAndParseResponse(resp);
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    return RPC_ERROR_NONE;
}

RpcError RpcIndexIVFGetListLength(rpcContext ctx, int indexId, int listId, uint32_t &len)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    IndexIVFGetListInfoRequest req;
    IndexIVFGetListLengthResponse resp;
    req.set_indexid(indexId);
    req.set_listid(listId);

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_IVF_GET_LIST_LENGTH, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    len = resp.len();
    return RPC_ERROR_NONE;
}

RpcError RpcIndexIVFUpdateNprobe(rpcContext ctx, int indexId, int nprobe)
{
    RPC_LOG_INFO("update nprobe(%d) of index %d on ctx %p\n", nprobe, indexId, ctx);
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    IndexIVFUpdateNprobeRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_nprobe(nprobe);

    HdcRpcError ret = session->SendAndReceive(RPC_INDEX_IVF_UPDATE_NPROBE, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}

RpcError RpcCreateIndexIVFSQ(rpcContext ctx, int &indexId, int dim, int nlist, bool byResidual, 
    faiss::ScalarQuantizer::QuantizerType qtype, faiss::MetricType metric, int nProbe, int resourceSize)
{
    CreateIndexIVFSQRequest req;
    CreateIndexResponse resp;
    req.set_dim(dim);
    req.set_nlist(nlist);
    req.set_byresidual(byResidual);
    req.set_qtype(qtype);
    req.set_metric(metric);
    req.set_nprobe(nProbe);
    req.set_resource(resourceSize);

    CALL_RPC(ctx, RPC_CREATE_INDEX_IVFSQ, req, resp, resp.result().err());
    indexId = resp.indexid();
    return RPC_ERROR_NONE;
}

RpcError RpcDestroyIndexIVFSQ(rpcContext ctx, int indexId)
{
    return RpcDestroyIndex(ctx, indexId);
}

RpcError RpcIndexSQUpdateTrainedValue(rpcContext ctx, int indexId, int dim,
    uint16_t *vmin, uint16_t *vdiff, bool isIvfSQ)
{
    IndexSQUpdateTrainedValueRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_isivfsq(isIvfSQ);
    req.set_dim(dim);
    req.set_vmin(vmin, dim * sizeof(uint16_t));
    req.set_vdiff(vdiff, dim * sizeof(uint16_t));

    CALL_RPC(ctx, RPC_INDEX_SQ_UPDATE_TRAINED_VALUE, req, resp, resp.err());
    return RPC_ERROR_NONE;
}

RpcError RpcIndexIVFSQAdd(rpcContext ctx, int indexId, int n, int listId, 
    const uint8_t *codes, int codeSize, const uint32_t *ids, const float *precomputedVal)
{
    IndexIVFSQAddRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_n(n);
    req.set_listid(listId);
    req.set_codesize(codeSize);
    req.set_codes(codes, n * codeSize);
    req.set_ids(ids, n * sizeof(uint32_t));
    req.set_precompute(precomputedVal, n * sizeof(float));

    CALL_RPC(ctx, RPC_INDEX_IVFSQ_ADD, req, resp, resp.err());
    return RPC_ERROR_NONE;
}
} // namespace ascend
} // namespace faiss