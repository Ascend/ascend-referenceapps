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

#include <faiss/ascend/rpc/AscendRpcIndexSQ.h>
#include <faiss/ascend/rpc/HdcClient.h>
#include <faiss/common/RpcUtils.h>
#include <mutex>

namespace faiss {
namespace ascend {
RpcError RpcCreateIndexSQ(rpcContext ctx, int &indexId, const IndexSQParameter &parameter)
{
    CreateIndexSQRequest req;
    CreateIndexResponse resp;
    req.set_dim(parameter.dim);
    req.set_qtype(parameter.qtype);
    req.set_metric(parameter.metric);
    req.set_resource(parameter.resourceSize);

    CALL_RPC(ctx, RPC_CREATE_INDEX_SQ, req, resp, resp.result().err());
    indexId = resp.indexid();
    return RPC_ERROR_NONE;
}

RpcError RpcDestroyIndexSQ(rpcContext ctx, int indexId)
{
    return RpcDestroyIndex(ctx, indexId);
}

RpcError RpcIndexSQAdd(rpcContext ctx, int indexId, int n, int dim, 
    const uint8_t *data, const float *precomputedVal)
{
    IndexSQAddRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_n(n);
    req.set_dim(dim);
    req.set_data(data, n * dim * sizeof(uint8_t));
    req.set_precompute(precomputedVal, n * sizeof(float));

    CALL_RPC(ctx, RPC_INDEX_SQ_ADD, req, resp, resp.err());
    return RPC_ERROR_NONE;
}

RpcError RpcIndexSQGetBase(rpcContext ctx, int indexId, uint32_t offset,
    uint32_t num, std::vector<uint8_t> &vectors)
{
    IndexSQGetBaseRequest req;
    IndexSQGetBaseResponse resp;
    req.set_indexid(indexId);
    req.set_offset(offset);
    req.set_num(num);

    CALL_RPC(ctx, RPC_INDEX_SQ_GET_BASE, req, resp, resp.result().err());

    size_t totalNum = resp.data().size() / sizeof(uint8_t);
    auto respBase = reinterpret_cast<const uint8_t *>(resp.data().data());
    vectors.assign(respBase, respBase + totalNum);

    return RPC_ERROR_NONE;
}

RpcError RpcIndexSQFastGetBase(rpcContext ctx, int indexId, uint32_t elementSize,
    uint32_t num, std::vector<uint8_t> &vectors)
{
    IndexSQGetBaseRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);

    HdcSession *session = static_cast<HdcSession *>(ctx);

    std::lock_guard<std::mutex> guard(session->GetSessionLock());
    HdcRpcError ret = session->SerializeAndSendMessage(RPC_INDEX_SQ_FAST_GET_BASE, true, req);
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    // prepare fast recv
    ret = session->HdcFastRecvPrepare();
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    void* dataBuf = nullptr;
    void* ctrlBuf = nullptr;

    uint32_t recvTotal = 0;

    const int idxListSize = 0;
    const int idxListLast = 1;
    const int idxChannelLast = 2;

    int channelCount = 0;
    while (true) {
        ret = session->HdcFastRecv(&dataBuf, &ctrlBuf, channelCount);
        RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

        int* recvCtrlBuf = reinterpret_cast<int*>(ctrlBuf);
        int listSize = recvCtrlBuf[idxListSize];
        int listLast = recvCtrlBuf[idxListLast];
        int channelLast = recvCtrlBuf[idxChannelLast];

        if (listSize > 0) {
            auto recvData = reinterpret_cast<uint8_t*>(dataBuf);
            vectors.insert(vectors.end(), recvData, recvData + listSize * elementSize);
            recvTotal += listSize;
        }

        channelCount++;
        if (channelLast != 0) {
            channelCount = 0;
            // recv sync
            ret = session->HdcSendRecvSignal();
            RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);
        }
        if (listLast != 0) {
            break;
        }
    }

    RPC_ASSERT(recvTotal == num);

    // fast recv release
    ret = session->HdcFastRecvRelease();
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    ret = session->RecvAndParseResponse(resp);
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    return RPC_ERROR_NONE;
}

RpcError RpcIndexSQGetBaseSize(rpcContext ctx, int indexId, uint32_t &size)
{
    IndexSQGetBaseLengthRequest req;
    IndexSQGetBaseLengthResponse resp;
    req.set_indexid(indexId);

    CALL_RPC(ctx, RPC_INDEX_SQ_GET_BASE_SIZE, req, resp, resp.result().err());

    size = resp.len();
    return RPC_ERROR_NONE;
}
} // namespace ascend
} // namespace faiss