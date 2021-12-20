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

#include "ascenddaemon/rpc/SessionHandler.h"


#include "ascenddaemon/impl/Index.h"
#include "ascenddaemon/impl/IndexIVFPQ.h"
#include "ascenddaemon/impl/IndexIVFSQ.h"
#include "ascenddaemon/impl/IndexIVFSQL2.h"
#include "ascenddaemon/impl/IndexIVFSQIP.h"
#include "ascenddaemon/impl/IndexIVFFlat.h"
#include "ascenddaemon/impl/IndexInt8IVFFlat.h"
#include "ascenddaemon/impl/IndexInt8IVFFlatL2.h"
#include "ascenddaemon/impl/IndexInt8IVFFlatCos.h"
#include "ascenddaemon/impl/IndexFlat.h"
#include "ascenddaemon/impl/IndexFlatL2.h"
#include "ascenddaemon/impl/IndexFlatIP.h"
#include "ascenddaemon/impl/IndexInt8.h"
#include "ascenddaemon/impl/IndexInt8Flat.h"
#include "ascenddaemon/impl/IndexInt8FlatL2.h"
#include "ascenddaemon/impl/IndexInt8FlatCos.h"
#include "ascenddaemon/impl/IndexSQ.h"
#include "ascenddaemon/impl/IndexSQL2.h"
#include "ascenddaemon/impl/IndexSQIP.h"
#include "ascenddaemon/impl/IndexPreTransform.h"
#include "ascenddaemon/impl/VectorTransform.h"
#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/utils/AscendAssert.h"
#include "ascenddaemon/utils/AscendException.h"
#include "ascenddaemon/utils/NNDimReduction.h"

using namespace ::ascend;
using namespace ::faiss::ascend;
using ::google::protobuf::Message;

namespace ascend {
#define LOG_IF_EXCEPT(EXPR)                                              \
    try {                                                                \
        EXPR;                                                            \
    } catch (const AscendException &e) {                                 \
        fprintf(stderr, "AscendException: %s", e.what());                \
        ACL_APP_LOG(ACL_ERROR, "[ascendfaiss] Exception: %s", e.what()); \
        result = CommonResponse_ErrorCode_ERROR;                         \
    } catch (const std::exception &e) {                                  \
        fprintf(stderr, "Exception: %s", e.what());                      \
        ACL_APP_LOG(ACL_ERROR, "[ascendfaiss] Exception: %s", e.what()); \
        result = CommonResponse_ErrorCode_ERROR;                         \
    }

SessionHandler::SessionHandler(HdcSession *session) : session(session), indexIdCounter(0),
                                                      transformsIdCounter(0), processModel(nullptr)
{
    RegisteRpcHandler();
}

SessionHandler::~SessionHandler()
{
    RPC_LOG_INFO("destruct session handler...\n");

    if (session != nullptr) {
        delete session;
    }

    for (auto &kv : indices) {
        RPC_LOG_INFO("index %d not released, destroy it\n", kv.first);
        delete kv.second;
    }
}

#define HANDLE_RPC(reqType, respType, handleFunc)                         \
    do {                                                                  \
        reqType req;                                                      \
        respType resp;                                                    \
        ret = session->ParseMessage(req, msgLen);                         \
        if (ret == HDC_RPC_ERROR_NONE) {                                  \
            handleFunc(&req, &resp);                                      \
            ret = session->SerializeAndSendMessage(rpcType, false, resp); \
        }                                                                 \
    } while (false)

#define CHECK_AND_ADD_INDEX(index, result, resp, indexIdCounter, indices)    \
    do {                                                                     \
        if (index == nullptr || result == CommonResponse_ErrorCode_ERROR) {  \
            resp->mutable_result()->set_err(CommonResponse_ErrorCode_ERROR); \
            return 0;                                                        \
        }                                                                    \
        index_id_t indexId = indexIdCounter++;                               \
        indices[indexId] = static_cast<Index *>(index);                      \
        resp->set_indexid(indexId);                                          \
        resp->mutable_result()->set_err(result);                             \
    } while (false)

void SessionHandler::RegisteRpcHandler()
{
    for (HdcRpcServiceType type = RPC_CREATE_CLIENT; type < RPC_TEST_DATA_INTEGRITY + 1;
        type = static_cast<HdcRpcServiceType>(type + 1)) {
        rpcHandlers[type] = &SessionHandler::HandleRpcCommon;
    }

    for (HdcRpcServiceType type = RPC_DESTROY_INDEX_INT8; type < RPC_INDEX_INT8_RESET + 1;
         type = static_cast<HdcRpcServiceType>(type + 1)) {
        rpcHandlers[type] = &SessionHandler::HandleRpcIndexInt8;
    }

    for (HdcRpcServiceType type = RPC_INDEX_IVF_UPDATE_COARSE_CENT; type < RPC_INDEX_RECLAIM_MEM + 1;
        type = static_cast<HdcRpcServiceType>(type + 1)) {
        rpcHandlers[type] = &SessionHandler::HandleRpcIndexIVFCommon;
    }

    for (HdcRpcServiceType type = RPC_INDEX_INT8_IVF_UPDATE_COARSE_CENT; type < RPC_INDEX_INT8_RECLAIM_MEM + 1;
         type = static_cast<HdcRpcServiceType>(type + 1)) {
        rpcHandlers[type] = &SessionHandler::HandleRpcIndexInt8IVF;
    }

    for (HdcRpcServiceType type = RPC_CREATE_INDEX_IVFPQ; type < RPC_INDEX_IVFPQ_ADD + 1;
        type = static_cast<HdcRpcServiceType>(type + 1)) {
        rpcHandlers[type] = &SessionHandler::HandleRpcIndexIVFPQ;
    }

    for (HdcRpcServiceType type = RPC_CREATE_INDEX_FLAT; type < RPC_INDEX_IVFFLAT_ADD + 1;
        type = static_cast<HdcRpcServiceType>(type + 1)) {
        rpcHandlers[type] = &SessionHandler::HandleRpcIndexFlat;
    }

    for (HdcRpcServiceType type = RPC_CREATE_INDEX_SQ; type < RPC_INDEX_IVFSQ_ADD + 1;
        type = static_cast<HdcRpcServiceType>(type + 1)) {
        rpcHandlers[type] = &SessionHandler::HandleRpcIndexSq8;
    }

    for (HdcRpcServiceType type = RPC_CREATE_INDEX_INT8_FLAT; type < RPC_INDEX_INT8_FLAT_GET_BASE_SIZE + 1;
         type = static_cast<HdcRpcServiceType>(type + 1)) {
        rpcHandlers[type] = &SessionHandler::HandleRpcIndexInt8Flat;
    }

    for (HdcRpcServiceType type = RPC_CREATE_INDEX_INT8_IVFFLAT; type < RPC_INDEX_INT8_IVFFLAT_ADD + 1;
         type = static_cast<HdcRpcServiceType>(type + 1)) {
        rpcHandlers[type] = &SessionHandler::HandleRpcIndexInt8IVFFlat;
    }

    for (HdcRpcServiceType type = RPC_CREATE_INDEX_PRETRANSFORM; type < RPC_DESTROY_TRANSFORM + 1;
        type = static_cast<HdcRpcServiceType>(type + 1)) {
        rpcHandlers[type] = &SessionHandler::HandleRpcIndexTransform;
    }
    for (HdcRpcServiceType type = RPC_CREATE_NN_DIM_REDUCTION; type < RPC_DESTROY_NN_DIM_REDUCTION + 1;
        type = static_cast<HdcRpcServiceType>(type + 1)) {
        rpcHandlers[type] = &SessionHandler::HandleRpcNNDimReduction;
    }
}

int SessionHandler::HandleRpcCommon(int rpcType, int msgLen)
{
    int ret = 0;
    switch (rpcType) {
        case RPC_DESTROY_INDEX:
            HANDLE_RPC(DestroyIndexRequest, CommonResponse, DestroyIndex);
            break;
        case RPC_INDEX_SEARCH:
            HANDLE_RPC(IndexSearchRequest, IndexSearchResponse, IndexSearch);
            break;
        case RPC_INDEX_REMOVE_IDS:
            HANDLE_RPC(IndexRemoveIdsRequest, IndexRemoveIdsResponse, IndexRemoveIds);
            break;
        case RPC_INDEX_REMOVE_RANGE_IDS:
            HANDLE_RPC(IndexRemoveRangeIdsRequest, IndexRemoveIdsResponse, IndexRemoveRangeIds);
            break;
        case RPC_INDEX_RESET:
            HANDLE_RPC(IndexResetRequest, CommonResponse, IndexReset);
            break;
        case RPC_TEST_DATA_INTEGRITY:
            HANDLE_RPC(TestDataIntegrityRequest, TestDataIntegrityResponse, TestDataIntegrity);
            break;
        default:
            RPC_UNREACHABLE();
    }
    return ret;
}

int SessionHandler::HandleRpcIndexInt8(int rpcType, int msgLen)
{
    int ret = 0;
    switch (rpcType) {
        case RPC_DESTROY_INDEX_INT8:
            HANDLE_RPC(DestroyIndexRequest, CommonResponse, DestroyIndexInt8);
            break;
        case RPC_INDEX_INT8_SEARCH:
            HANDLE_RPC(IndexInt8SearchRequest, IndexInt8SearchResponse, IndexInt8Search);
            break;
        case RPC_INDEX_INT8_REMOVE_IDS:
            HANDLE_RPC(IndexRemoveIdsRequest, IndexRemoveIdsResponse, IndexInt8RemoveIds);
            break;
        case RPC_INDEX_INT8_REMOVE_RANGE_IDS:
            HANDLE_RPC(IndexRemoveRangeIdsRequest, IndexRemoveIdsResponse, IndexInt8RemoveRangeIds);
            break;
        case RPC_INDEX_INT8_RESET:
            HANDLE_RPC(IndexResetRequest, CommonResponse, IndexInt8Reset);
            break;
        default:
            RPC_UNREACHABLE();
    }
    return ret;
}

int SessionHandler::HandleRpcIndexIVFCommon(int rpcType, int msgLen)
{
    int ret = 0;
    switch (rpcType) {
        case RPC_INDEX_IVF_UPDATE_COARSE_CENT:
            HANDLE_RPC(IndexIVFUpdateCoarseCentRequest, CommonResponse, IndexIVFUpdateCoarseCent);
            break;
        case RPC_INDEX_IVF_UPDATE_NPROBE:
            HANDLE_RPC(IndexIVFUpdateNprobeRequest, CommonResponse, IndexIVFUpdateNprobe);
            break;
        case RPC_INDEX_IVF_GET_LIST_LENGTH:
            HANDLE_RPC(IndexIVFGetListInfoRequest, IndexIVFGetListLengthResponse, IndexIVFGetListLength);
            break;
        case RPC_INDEX_IVF_GET_LIST_CODES:
            HANDLE_RPC(IndexIVFGetListInfoRequest, IndexIVFGetListCodesResponse, IndexIVFGetListCodes);
            break;
        case RPC_INDEX_IVF_FAST_GET_LIST_CODES:
            HANDLE_RPC(IndexIVFGetListInfoRequest, CommonResponse, IndexIVFFastGetListCodes);
            break;
        case RPC_INDEX_RESERVE_MEM:
            HANDLE_RPC(IndexReserveMemRequest, CommonResponse, IndexReserveMemory);
            break;
        case RPC_INDEX_RECLAIM_MEM:
            HANDLE_RPC(IndexReclaimMemRequest, IndexReclaimMemResponse, IndexReclaimMemory);
            break;
        default:
            RPC_UNREACHABLE();
    }
    return ret;
}

int SessionHandler::HandleRpcIndexInt8IVF(int rpcType, int msgLen)
{
    int ret = 0;
    switch (rpcType) {
        case RPC_INDEX_INT8_IVF_UPDATE_COARSE_CENT:
            HANDLE_RPC(IndexIVFUpdateCoarseCentRequest, CommonResponse, IndexInt8IVFUpdateCoarseCent);
            break;
        case RPC_INDEX_INT8_IVF_UPDATE_NPROBE:
            HANDLE_RPC(IndexIVFUpdateNprobeRequest, CommonResponse, IndexInt8IVFUpdateNprobe);
            break;
        case RPC_INDEX_INT8_IVF_GET_LIST_LENGTH:
            HANDLE_RPC(IndexIVFGetListInfoRequest, IndexIVFGetListLengthResponse, IndexInt8IVFGetListLength);
            break;
        case RPC_INDEX_INT8_IVF_GET_LIST_CODES:
            HANDLE_RPC(IndexIVFGetListInfoRequest, IndexIVFGetListCodesResponse, IndexInt8IVFGetListCodes);
            break;
        case RPC_INDEX_INT8_RESERVE_MEM:
            HANDLE_RPC(IndexReserveMemRequest, CommonResponse, IndexInt8ReserveMemory);
            break;
        case RPC_INDEX_INT8_RECLAIM_MEM:
            HANDLE_RPC(IndexReclaimMemRequest, IndexReclaimMemResponse, IndexInt8ReclaimMemory);
            break;
        default:
            RPC_UNREACHABLE();
    }
    return ret;
}

int SessionHandler::HandleRpcIndexFlat(int rpcType, int msgLen)
{
    int ret = 0;
    switch (rpcType) {
        case RPC_CREATE_INDEX_IVFFLAT:
            HANDLE_RPC(CreateIndexIVFFlatRequest, CreateIndexResponse, CreateIndexIVFFlat);
            break;
        case RPC_CREATE_INDEX_FLAT:
            HANDLE_RPC(CreateIndexFlatRequest, CreateIndexResponse, CreateIndexFlat);
            break;
        case RPC_INDEX_IVFFLAT_ADD:
            HANDLE_RPC(IndexIVFFlatAddRequest, CommonResponse, IndexIVFFlatAdd);
            break;
        case RPC_INDEX_FLAT_ADD:
            HANDLE_RPC(IndexFlatAddRequest, CommonResponse, IndexFlatAdd);
            break;
        case RPC_INDEX_FLAT_GET_BASE:
            HANDLE_RPC(IndexFlatGetBaseRequest, IndexFlatGetBaseResponse, IndexFlatGetBase);
            break;
        case RPC_INDEX_FLAT_GET_BASE_SIZE:
            HANDLE_RPC(IndexFlatGetBaseLengthRequest, IndexFlatGetBaseLengthResponse, IndexFlatGetBaseLength);
            break;
        default:
            RPC_UNREACHABLE();
    }
    return ret;
}

int SessionHandler::HandleRpcIndexInt8IVFFlat(int rpcType, int msgLen)
{
    int ret = 0;
    switch (rpcType) {
        case RPC_CREATE_INDEX_INT8_IVFFLAT:
            HANDLE_RPC(CreateIndexInt8IVFFlatRequest, CreateIndexResponse, CreateIndexInt8IVFFlat);
            break;
        case RPC_INDEX_INT8_IVFFLAT_ADD:
            HANDLE_RPC(IndexInt8IVFFlatAddRequest, CommonResponse, IndexInt8IVFFlatAdd);
            break;
        default:
            RPC_UNREACHABLE();
    }
    return ret;
}

int SessionHandler::HandleRpcIndexInt8Flat(int rpcType, int msgLen)
{
    int ret = 0;
    switch (rpcType) {
        case RPC_CREATE_INDEX_INT8_FLAT:
            HANDLE_RPC(CreateIndexInt8FlatRequest, CreateIndexResponse, CreateIndexInt8Flat);
            break;
        case RPC_INDEX_INT8_FLAT_ADD:
            HANDLE_RPC(IndexInt8FlatAddRequest, CommonResponse, IndexInt8FlatAdd);
            break;
        case RPC_INDEX_INT8_FLAT_GET_BASE:
            HANDLE_RPC(IndexFlatGetBaseRequest, IndexFlatGetBaseResponse, IndexInt8FlatGetBase);
            break;
        case RPC_INDEX_INT8_FLAT_GET_BASE_SIZE:
            HANDLE_RPC(IndexFlatGetBaseLengthRequest, IndexFlatGetBaseLengthResponse, IndexInt8FlatGetBaseLength);
            break;
        default:
            RPC_UNREACHABLE();
    }
    return ret;
}

int SessionHandler::HandleRpcIndexSq8(int rpcType, int msgLen)
{
    int ret = 0;
    switch (rpcType) {
        case RPC_CREATE_INDEX_IVFSQ:
            HANDLE_RPC(CreateIndexIVFSQRequest, CreateIndexResponse, CreateIndexIVFSQ);
            break;
        case RPC_CREATE_INDEX_SQ:
            HANDLE_RPC(CreateIndexSQRequest, CreateIndexResponse, CreateIndexSQ);
            break;
        case RPC_INDEX_SQ_UPDATE_TRAINED_VALUE:
            HANDLE_RPC(IndexSQUpdateTrainedValueRequest, CommonResponse, IndexSQUpdateTrainedValue);
            break;
        case RPC_INDEX_IVFSQ_ADD:
            HANDLE_RPC(IndexIVFSQAddRequest, CommonResponse, IndexIVFSQAdd);
            break;
        case RPC_INDEX_SQ_ADD:
            HANDLE_RPC(IndexSQAddRequest, CommonResponse, IndexSQAdd);
            break;
        case RPC_INDEX_SQ_GET_BASE:
            HANDLE_RPC(IndexSQGetBaseRequest, IndexSQGetBaseResponse, IndexSQGetBase);
            break;
        case RPC_INDEX_SQ_FAST_GET_BASE:
            HANDLE_RPC(IndexSQGetBaseRequest, CommonResponse, IndexSQFastGetBase);
            break;
        case RPC_INDEX_SQ_GET_BASE_SIZE:
            HANDLE_RPC(IndexSQGetBaseLengthRequest, IndexSQGetBaseLengthResponse, IndexSQGetBaseLength);
            break;
        default:
            RPC_UNREACHABLE();
    }
    return ret;
}

int SessionHandler::HandleRpcIndexIVFPQ(int rpcType, int msgLen)
{
    int ret = 0;
    switch (rpcType) {
        case RPC_CREATE_INDEX_IVFPQ:
            HANDLE_RPC(CreateIndexIVFPQRequest, CreateIndexResponse, CreateIndexIVFPQ);
            break;
        case RPC_INDEX_IVFPQ_UPDATE_PQ_CENT:
            HANDLE_RPC(IndexIVFPQUpdatePQCentRequest, CommonResponse, IndexIVFPQUpdatePQCent);
            break;
        case RPC_INDEX_IVFPQ_ADD:
            HANDLE_RPC(IndexIVFPQAddRequest, CommonResponse, IndexIVFPQAdd);
            break;
        default:
            RPC_UNREACHABLE();
    }
    return ret;
}

int SessionHandler::HandleRpcIndexTransform(int rpcType, int msgLen)
{
    int ret = 0;
    switch (rpcType) {
        case RPC_CREATE_INDEX_PRETRANSFORM:
            HANDLE_RPC(CreateIndexPreTransformRequest, CreateIndexResponse, CreateIndexPreTransform);
            break;
        case RPC_TRANSFORM_LINEAR_UPDATE_TRAINED_VALUE:
            HANDLE_RPC(LinearTransformUpdateTrainedValueRequest, CommonResponse, LinearTransformUpdateTrainedValue);
            break;
        case RPC_INDEX_PRETRANSFORM_PREPEND:
            HANDLE_RPC(IndexPreTransformPrependRequest, CommonResponse, IndexPreTransformPrepend);
            break;
        case RPC_CREATE_TRANSFORM_LINEAR:
            HANDLE_RPC(CreateLinearTransformRequest, CreateTransformResponse, CreateLinearTransform);
            break;
        case RPC_DESTROY_TRANSFORM:
            HANDLE_RPC(DestroyTransformRequest, CommonResponse, DestroyTransform);
            break;
        default:
            RPC_UNREACHABLE();
    }
    return ret;
}

int SessionHandler::HandleRpcNNDimReduction(int rpcType, int msgLen)
{
    int ret = 0;
    switch (rpcType) {
        case RPC_CREATE_NN_DIM_REDUCTION:
            HANDLE_RPC(NNDimReductionCreateRequest, NNDimReductionCreateResponse, CreateNNDimReduction);
            break;
        case RPC_INFER_NN_DIM_REDUCTION:
            HANDLE_RPC(NNDimReductionInferRequest, NNDimReductionInferResponse, InferNNDimReduction);
            break;
        case RPC_DESTROY_NN_DIM_REDUCTION:
            HANDLE_RPC(NNDimReductionDestroyRequest, NNDimReductionDestroyResponse, DestroyNNDimReduction);
            break;
        default:
            RPC_UNREACHABLE();
    }
    return ret;
}

void SessionHandler::Handle()
{
    int ret;
    int rpcType;
    bool isRequest = false;
    int msgLen;

    while (true) {
        ret = session->RecvMessage(rpcType, isRequest, msgLen);
        if (ret != HDC_RPC_ERROR_NONE) {
            RPC_LOG_ERROR("recv msg error\n");
            break;
        }
        RPC_ASSERT(isRequest);
        RPC_ASSERT(msgLen >= 0);
        RPC_ASSERT(rpcType >= static_cast<int>(RPC_DESTROY_INDEX));
        RPC_ASSERT(rpcType < static_cast<int>(RPC_SERVICE_TYPE_MAX));

        ret = (this->*rpcHandlers[rpcType])(rpcType, msgLen);
        if (ret != HDC_RPC_ERROR_NONE) {
            break;
        }
    }
}

int SessionHandler::CreateIndexIVFPQ(const CreateIndexIVFPQRequest *req, CreateIndexResponse *resp)
{
    RPC_TIME_LOG("CreateIndexIVFPQ, dim=%d, nlist=%d, M=%d, bitsPerCode=%d, nprobe=%d, resource=%d\n", req->dim(),
        req->nlist(), req->subquantizers(), req->bitspercode(), req->nprobe(), req->resource());

    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    IndexIVFPQ *index = nullptr;
    LOG_IF_EXCEPT(index = new IndexIVFPQ(req->nlist(), req->dim(), req->subquantizers(), req->bitspercode(),
        req->nprobe(), req->resource()));

    CHECK_AND_ADD_INDEX(index, result, resp, indexIdCounter, indices);
    return 0;
}

int SessionHandler::CreateIndexIVFFlat(const CreateIndexIVFFlatRequest *req, CreateIndexResponse *resp)
{
    RPC_TIME_LOG("CreateIndexIVFFlat, dim=%d, nlist=%d, nprobe=%d, resource=%d\n", req->dim(), req->nlist(),
        req->nprobe(), req->resource());

    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    IndexIVFFlat *index = nullptr;
    LOG_IF_EXCEPT(index = new IndexIVFFlat(req->nlist(), req->dim(), req->nprobe(), req->resource()));

    CHECK_AND_ADD_INDEX(index, result, resp, indexIdCounter, indices);
    return 0;
}

int SessionHandler::CreateIndexFlat(const CreateIndexFlatRequest *req, CreateIndexResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    IndexFlat *index = nullptr;
    switch (req->metric()) {
        case MetricType::METRIC_L2: 
            LOG_IF_EXCEPT(index = new IndexFlatL2(req->dim(), req->resource()));
            break;
        case MetricType::METRIC_INNER_PRODUCT: 
            LOG_IF_EXCEPT(index = new IndexFlatIP(req->dim(), req->resource()));
            break;
    }

    CHECK_AND_ADD_INDEX(index, result, resp, indexIdCounter, indices);
    return 0;
}

int SessionHandler::CreateIndexIVFSQ(const CreateIndexIVFSQRequest *req, CreateIndexResponse *resp)
{
    RPC_TIME_LOG("CreateIndexIVFSQ, dim=%d, nlist=%d, qtype=%d, byResidual=%d, nprobe=%d, resource=%d\n", req->dim(),
        req->nlist(), req->qtype(), req->byresidual(), req->nprobe(), req->resource());

    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    IndexIVFSQ<float> *index = nullptr;
    switch (req->metric()) {
        case MetricType::METRIC_L2: {
            LOG_IF_EXCEPT(index = new IndexIVFSQL2(req->nlist(), req->dim(), req->byresidual(),
                req->nprobe(), req->resource()));
            break;
        }
        case MetricType::METRIC_INNER_PRODUCT: {
            LOG_IF_EXCEPT(index = new IndexIVFSQIP(req->nlist(), req->dim(), req->byresidual(),
                req->nprobe(), req->resource()));
            break;
        }
    }

    CHECK_AND_ADD_INDEX(index, result, resp, indexIdCounter, indices);
    return 0;
}

int SessionHandler::CreateIndexSQ(const CreateIndexSQRequest *req, CreateIndexResponse *resp)
{
    RPC_TIME_LOG("CreateIndexSQ, dim=%d, qtype=%d, resource=%d\n", req->dim(), req->qtype(), req->resource());
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    IndexSQ *index = nullptr;
    switch (req->metric()) {
        case MetricType::METRIC_L2: {
            LOG_IF_EXCEPT(index = new IndexSQL2(req->dim(), req->resource()));
            CHECK_AND_ADD_INDEX(index, result, resp, indexIdCounter, indices);
            break;
        }
        case MetricType::METRIC_INNER_PRODUCT: {
            LOG_IF_EXCEPT(index = new IndexSQIP(req->dim(), req->resource()));
            CHECK_AND_ADD_INDEX(index, result, resp, indexIdCounter, indices);
            break;
        }
    }
    return 0;
}

int SessionHandler::CreateIndexInt8IVFFlat(const CreateIndexInt8IVFFlatRequest *req, CreateIndexResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    IndexInt8IVF *index = nullptr;
    switch (req->metric()) {
        case MetricType::METRIC_L2: {
            LOG_IF_EXCEPT(index = new IndexInt8IVFFlatL2(req->nlist(), req->dim(), req->nprobe(), req->resource()));
            break;
        }
        case MetricType::METRIC_INNER_PRODUCT: {
            LOG_IF_EXCEPT(index = new IndexInt8IVFFlatCos(req->nlist(), req->dim(), req->nprobe(), req->resource()));
            break;
        }
    }

    if (index == nullptr || result == CommonResponse_ErrorCode_ERROR) {
        resp->mutable_result()->set_err(CommonResponse_ErrorCode_ERROR);
        return 0;
    }

    index_id_t indexId = indexIdCounter++;
    indices[indexId] = reinterpret_cast<Index *>(index);
    resp->set_indexid(indexId);
    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::CreateIndexInt8Flat(const CreateIndexInt8FlatRequest *req, CreateIndexResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    IndexInt8 *index = nullptr;
    switch (req->metric()) {
        case MetricType::METRIC_L2: {
            LOG_IF_EXCEPT(index = new IndexInt8FlatL2(req->dim(), req->resource()));
            break;
        }
        case MetricType::METRIC_INNER_PRODUCT: {
            LOG_IF_EXCEPT(index = new IndexInt8FlatCos(req->dim(), req->resource()));
            break;
        }
    }

    if (index == nullptr || result == CommonResponse_ErrorCode_ERROR) {
        resp->mutable_result()->set_err(CommonResponse_ErrorCode_ERROR);
        return 0;
    }

    index_id_t indexId = indexIdCounter++;
    indices[indexId] = reinterpret_cast<Index *>(index);
    resp->set_indexid(indexId);
    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::CreateIndexPreTransform(const CreateIndexPreTransformRequest *req, CreateIndexResponse *resp)
{
    RPC_TIME_LOG("CreateIndexPreTransform, mainIndexId=%d\n", req->subindexid());
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t subindexId = req->subindexid();
    RPC_ASSERT_FMT(indices.find(subindexId) != indices.end(), "!!!Invalid index id: %d\n", subindexId);
    Index *subindex = indices[subindexId];
    RPC_REQUIRE_NOT_NULL(subindex);

    IndexPreTransform *index = nullptr;
    LOG_IF_EXCEPT(index = new IndexPreTransform(subindex));
    CHECK_AND_ADD_INDEX(index, result, resp, indexIdCounter, indices);
    return 0;
}

int SessionHandler::DestroyIndex(const DestroyIndexRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    RPC_TIME_LOG("DestroyIndex %d\n", indexId);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    LOG_IF_EXCEPT(delete indices[indexId]);
    RPC_TIME_LOG("DestroyIndex end\n");
    indices.erase(indexId);

    resp->set_err(result);
    return 0;
}

int SessionHandler::DestroyIndexInt8(const faiss_ascend::DestroyIndexRequest *req, faiss_ascend::CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    RPC_TIME_LOG("DestroyIndex %d\n", indexId);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = reinterpret_cast<IndexInt8 *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    LOG_IF_EXCEPT(delete pIndex);
    RPC_TIME_LOG("Destroy IndexInt8 end\n");
    indices.erase(indexId);

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexIVFUpdateCoarseCent(const IndexIVFUpdateCoarseCentRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t total = req->total();
    int32_t dim = req->dim();
    const std::string &data = req->data();
    RPC_TIME_LOG("index %d: UpdateCoarseCent, nlist=%d, dim=%d\n", indexId, total, dim);
    RPC_ASSERT(data.size() == total * dim * sizeof(uint16_t));
    const float16_t *tmp = reinterpret_cast<const float16_t *>(data.data());
    AscendTensor<float16_t, DIMS_2> coarseCent(const_cast<float16_t *>(tmp), { total, dim });

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexIVF *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    RPC_TIME_LOG("index %d: UpdateCoarseCent actual start\n", indexId);
    LOG_IF_EXCEPT(pIndex->updateCoarseCentroidsData(coarseCent));
    RPC_TIME_LOG("index %d: UpdateCoarseCent end\n", indexId);

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexInt8IVFUpdateCoarseCent(const IndexIVFUpdateCoarseCentRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t total = req->total();
    int32_t dim = req->dim();
    const std::string &data = req->data();
    RPC_TIME_LOG("index %d: UpdateCoarseCent, nlist=%d, dim=%d\n", indexId, total, dim);
    RPC_ASSERT(data.size() == total * dim * sizeof(int8_t));
    const int8_t *tmp = reinterpret_cast<const int8_t *>(data.data());
    AscendTensor<int8_t, DIMS_2> coarseCent(const_cast<int8_t *>(tmp), { total, dim });

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = reinterpret_cast<IndexInt8IVF *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    RPC_TIME_LOG("index %d: UpdateCoarseCent actual start\n", indexId);
    LOG_IF_EXCEPT(pIndex->updateCoarseCentroidsData(coarseCent));
    RPC_TIME_LOG("index %d: UpdateCoarseCent end\n", indexId);

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexIVFPQUpdatePQCent(const IndexIVFPQUpdatePQCentRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t m = req->m();
    int32_t ksub = req->ksub();
    int32_t dsub = req->dsub();
    const std::string &data = req->data();
    RPC_TIME_LOG("index %d: UpdatePQCent, M=%d, ksub=%d, dsub=%d\n", indexId, m, ksub, dsub);
    RPC_ASSERT(data.size() == m * ksub * dsub * sizeof(uint16_t));
    auto tmp = const_cast<char *>(data.data());
    AscendTensor<float16_t, DIMS_2> pqCent(reinterpret_cast<float16_t *>(tmp), { m, ksub * dsub });

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexIVFPQ *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    RPC_TIME_LOG("index %d: UpdatePQCent actual start\n", indexId);
    LOG_IF_EXCEPT(pIndex->updatePQCentroidsData(pqCent));
    RPC_TIME_LOG("index %d: UpdatePQCent end\n", indexId);

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexSQUpdateTrainedValue(const IndexSQUpdateTrainedValueRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t dim = req->dim();
    bool isIvfSQ = req->isivfsq();
    const std::string &vmin = req->vmin();
    const std::string &vdiff = req->vdiff();
    RPC_TIME_LOG("index %d: UpdateSQTrainedValue, total=%d, isIvfSQ=%d\n", indexId, dim, isIvfSQ);

    RPC_ASSERT(vmin.size() == dim * sizeof(uint16_t));
    RPC_ASSERT(vdiff.size() == dim * sizeof(uint16_t));
    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);

    auto tmpVmin = const_cast<char *>(vmin.data());
    auto tmpVdiff = const_cast<char *>(vdiff.data());
    AscendTensor<float16_t, DIMS_1> vminTensor(reinterpret_cast<float16_t *>(tmpVmin), { dim });
    AscendTensor<float16_t, DIMS_1> vdiffTensor(reinterpret_cast<float16_t *>(tmpVdiff), { dim });

    if (isIvfSQ) {
        auto pIndex = dynamic_cast<IndexIVFSQ<float> *>(indices[indexId]);
        RPC_REQUIRE_NOT_NULL(pIndex);
        LOG_IF_EXCEPT(pIndex->updateTrainedValue(vminTensor, vdiffTensor));
    } else {
        auto pIndex = dynamic_cast<IndexSQ *>(indices[indexId]);
        RPC_REQUIRE_NOT_NULL(pIndex);
        LOG_IF_EXCEPT(pIndex->updateTrainedValue(vminTensor, vdiffTensor));
    }

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexIVFPQAdd(const IndexIVFPQAddRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t n = req->n();
    int32_t listId = req->listid();
    int32_t codeSize = req->codesize();
    const std::string &codes = req->codes();
    const std::string &ids = req->ids();
    RPC_ASSERT(codes.size() == static_cast<size_t>(n * codeSize));
    RPC_ASSERT(ids.size() == static_cast<size_t>(n * sizeof(uint32_t)));

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexIVFPQ *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    LOG_IF_EXCEPT(pIndex->addVectors(listId, reinterpret_cast<const uint8_t *>(codes.data()),
        reinterpret_cast<const uint32_t *>(ids.data()), n));

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexIVFFlatAdd(const IndexIVFFlatAddRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t n = req->n();
    int32_t listId = req->listid();
    int32_t dim = req->dim();

    const std::string &vectors = req->vectors();
    const std::string &ids = req->ids();
    RPC_ASSERT(vectors.size() == static_cast<size_t>(n * dim) * sizeof(uint16_t));
    RPC_ASSERT(ids.size() == static_cast<size_t>(n * sizeof(uint32_t)));

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexIVFFlat *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    LOG_IF_EXCEPT(pIndex->addVectors(listId, n, reinterpret_cast<const float16_t *>(vectors.data()),
        reinterpret_cast<const uint32_t *>(ids.data())));

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexFlatAdd(const IndexFlatAddRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t n = req->n();
    int32_t dim = req->dim();
    const std::string &vectors = req->vectors();
    ASCEND_THROW_IF_NOT(vectors.size() == static_cast<size_t>(n * dim * sizeof(float16_t)));

    ASCEND_THROW_IF_NOT_FMT(indices.find(indexId) != indices.end(), "Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexFlat *>(indices[indexId]);
    ASCEND_THROW_IF_NOT(pIndex != nullptr);
    auto tmpPtr = reinterpret_cast<const float16_t *>(vectors.data());
    AscendTensor<float16_t, DIMS_2> vec(const_cast<float16_t *>(tmpPtr), { n, dim });
    LOG_IF_EXCEPT(pIndex->addVectors(vec));

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexInt8IVFFlatAdd(const IndexInt8IVFFlatAddRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t n = req->n();
    int32_t listId = req->listid();
    int32_t dim = req->dim();

    const std::string &vectors = req->vectors();
    const std::string &ids = req->ids();
    RPC_ASSERT(vectors.size() == static_cast<size_t>(n * dim) * sizeof(int8_t));
    RPC_ASSERT(ids.size() == static_cast<size_t>(n * sizeof(uint32_t)));
    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);

    IndexInt8IVF* pIndex = reinterpret_cast<IndexInt8IVF *>(indices[indexId]);
    ASCEND_THROW_IF_NOT(pIndex != nullptr);
    LOG_IF_EXCEPT(pIndex->addVectors(listId, n, reinterpret_cast<const int8_t *>(vectors.data()),
                                     reinterpret_cast<const uint32_t *>(ids.data())));
    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexInt8FlatAdd(const IndexInt8FlatAddRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t n = req->n();
    int32_t dim = req->dim();
    const std::string &vectors = req->data();
    ASCEND_THROW_IF_NOT(vectors.size() == static_cast<size_t>(n * dim * sizeof(int8_t)));

    ASCEND_THROW_IF_NOT_FMT(indices.find(indexId) != indices.end(), "Invalid index id: %d\n", indexId);
    auto tmpPtr = reinterpret_cast<const int8_t *>(vectors.data());
    AscendTensor<int8_t, DIMS_2> vec(const_cast<int8_t *>(tmpPtr), { n, dim });

    IndexInt8* pIndex = reinterpret_cast<IndexInt8 *>(indices[indexId]);
    ASCEND_THROW_IF_NOT(pIndex != nullptr);
    LOG_IF_EXCEPT(pIndex->addVectors(vec));
    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexIVFSQAdd(const IndexIVFSQAddRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t n = req->n();
    int32_t listId = req->listid();
    int32_t codeSize = req->codesize();
    const std::string &codes = req->codes();
    const std::string &ids = req->ids();
    const std::string &preCompute = req->precompute();
    RPC_ASSERT(codes.size() == static_cast<size_t>(n * codeSize));
    RPC_ASSERT(ids.size() == static_cast<size_t>(n * sizeof(uint32_t)));
    RPC_ASSERT(preCompute.size() == static_cast<size_t>(n * sizeof(float)));

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);

    auto pIndex = dynamic_cast<IndexIVFSQ<float> *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    LOG_IF_EXCEPT(pIndex->addVectors(listId, n, reinterpret_cast<const uint8_t *>(codes.data()),
        reinterpret_cast<const uint32_t *>(ids.data()), reinterpret_cast<const float *>(preCompute.data())));

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexSQAdd(const IndexSQAddRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t n = req->n();
    int32_t dim = req->dim();
    const std::string &data = req->data();
    const std::string &preCompute = req->precompute();
    RPC_ASSERT(data.size() == static_cast<size_t>(n * dim * sizeof(uint8_t)));
    RPC_ASSERT(preCompute.size() == static_cast<size_t>(n * sizeof(float)));

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexSQ *>(indices[indexId]);
    ASCEND_THROW_IF_NOT(pIndex != nullptr);
    auto codePtr = reinterpret_cast<const uint8_t *>(data.data());
    auto precompPtre = reinterpret_cast<const float *>(preCompute.data());
    LOG_IF_EXCEPT(pIndex->addVectors(n, codePtr, precompPtre));

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexPreTransformPrepend(const IndexPreTransformPrependRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    index_id_t transformId = req->transformid();

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    RPC_ASSERT_FMT(transforms.find(transformId) != transforms.end(), "!!!Invalid transform id: %d\n", transformId);
    auto pIndex = dynamic_cast<IndexPreTransform *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    auto pTrans = dynamic_cast<LinearTransform *>(transforms[transformId]);
    RPC_REQUIRE_NOT_NULL(pTrans);
    LOG_IF_EXCEPT(pIndex->prependTransform(pTrans));
    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexIVFGetListLength(const IndexIVFGetListInfoRequest *req, IndexIVFGetListLengthResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t listId = req->listid();

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexIVF *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    LOG_IF_EXCEPT({
        uint32_t listLen = pIndex->getListLength(listId);
        resp->set_len(listLen);
    });

    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::IndexInt8IVFGetListLength(const IndexIVFGetListInfoRequest *req,
                                              IndexIVFGetListLengthResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t listId = req->listid();

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = reinterpret_cast<IndexInt8IVF *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    LOG_IF_EXCEPT({
        uint32_t listLen = pIndex->getListLength(listId);
        resp->set_len(listLen);
    });

    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::IndexIVFGetListCodes(const IndexIVFGetListInfoRequest *req, IndexIVFGetListCodesResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t listId = req->listid();

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexIVF *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);

    LOG_IF_EXCEPT({
        if (pIndex->listVectorsNeedReshaped()) {
            std::vector<unsigned char> codes;
            pIndex->getListVectorsReshaped(listId, codes);
            resp->set_codes(codes.data(), codes.size());
        } else {
            DeviceVector<unsigned char> &codes = pIndex->getListVectors(listId);
            resp->set_codes(codes.data(), codes.size());
        }

        DeviceVector<uint32_t> &ids = pIndex->getListIndices(listId);
        resp->set_ids(ids.data(), ids.size() * sizeof(uint32_t));
    });

    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::IndexIVFFastGetListCodes(const IndexIVFGetListInfoRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t nlist = req->listid();

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexIVF *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);

    size_t maxDataBufSize = pIndex->getMaxListDataIndicesBufferSize();
    // prepare fast send: malloc and exchange addr
    HdcRpcError ret = session->HdcFastSendPrepare(maxDataBufSize);
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    int channelNumber = session->HdcGetFastSendChannel();

    const int idxListId = 0;
    const int idxListSize = 1;
    const int idxListCount = 2;

    for (int i = 0; i < nlist; i += channelNumber) {
        for (int j = i; j < i + channelNumber; j++) {
            if (j >= nlist) {
                continue;
            }
            int channelId = j % channelNumber;
            void* dataBuf = nullptr;
            void* ctrlBuf = nullptr;
            ret = session->HdcGetFastSendAddr(&dataBuf, &ctrlBuf, channelId);
            RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

            size_t listSize = pIndex->getListLength(j);
            int* sendCtrlBuf = reinterpret_cast<int*>(ctrlBuf);
            sendCtrlBuf[idxListId] = j; // list id
            sendCtrlBuf[idxListSize] = listSize;
            sendCtrlBuf[idxListCount] = pIndex->getDim();
            
            if (listSize > 0) {
                // codes
                size_t codeSize = pIndex->getDim() * listSize * sizeof(unsigned char);
                pIndex->getListVectorsReshaped(j, reinterpret_cast<unsigned char*>(dataBuf)); // codes must be reshaped

                // indices
                DeviceVector<uint32_t> &ids = pIndex->getListIndices(j);
                uint32_t* idPos = reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(dataBuf) + codeSize);
                MEMCPY_S(reinterpret_cast<unsigned char*>(idPos),
                    maxDataBufSize - codeSize, ids.data(), ids.size() * sizeof(uint32_t));
            }
            // fast send
            ret = session->HdcFastSend(dataBuf, ctrlBuf, channelId);
            RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);
        }
        // wait sync
        ret = session->HdcWaitRecvSignal();
        RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);
    }
    // fast release: free fast memory
    ret = session->HdcFastSendRelease();
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexInt8IVFGetListCodes(const IndexIVFGetListInfoRequest *req, IndexIVFGetListCodesResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t listId = req->listid();

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = reinterpret_cast<IndexInt8IVF *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);

    LOG_IF_EXCEPT({
        if (pIndex->listVectorsNeedReshaped()) {
            std::vector<int8_t> codes;
            pIndex->getListVectorsReshaped(listId, codes);
            resp->set_codes(codes.data(), codes.size());
        } else {
            DeviceVector<int8_t> &codes = pIndex->getListVectors(listId);
            resp->set_codes(codes.data(), codes.size());
        }

        DeviceVector<uint32_t> &ids = pIndex->getListIndices(listId);
        resp->set_ids(ids.data(), ids.size() * sizeof(uint32_t));
    });

    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::IndexFlatGetBaseLength(const IndexFlatGetBaseLengthRequest *req,
    IndexFlatGetBaseLengthResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();

    ASCEND_THROW_IF_NOT_FMT(indices.find(indexId) != indices.end(), "Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexFlat *>(indices[indexId]);
    ASCEND_THROW_IF_NOT(pIndex != nullptr);

    LOG_IF_EXCEPT(resp->set_len(pIndex->getSize()));
    resp->mutable_result()->set_err(result);

    return 0;
}

int SessionHandler::IndexInt8FlatGetBaseLength(const IndexFlatGetBaseLengthRequest *req,
    IndexFlatGetBaseLengthResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();

    ASCEND_THROW_IF_NOT_FMT(indices.find(indexId) != indices.end(), "Invalid index id: %d\n", indexId);
    switch (req->metric()) {
        case MetricType::METRIC_L2: {
            auto pIndex = reinterpret_cast<IndexInt8FlatL2 *>(indices[indexId]);
            ASCEND_THROW_IF_NOT(pIndex != nullptr);
            LOG_IF_EXCEPT(resp->set_len(pIndex->getSize()));
            break;
        }
        case MetricType::METRIC_INNER_PRODUCT: {
            auto pIndex = reinterpret_cast<IndexInt8FlatCos *>(indices[indexId]);
            ASCEND_THROW_IF_NOT(pIndex != nullptr);
            LOG_IF_EXCEPT(resp->set_len(pIndex->getSize()));
            break;
        }
    }

    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::IndexSQGetBaseLength(const IndexSQGetBaseLengthRequest *req, IndexSQGetBaseLengthResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();

    ASCEND_THROW_IF_NOT_FMT(indices.find(indexId) != indices.end(), "Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexSQ *>(indices[indexId]);
    ASCEND_THROW_IF_NOT(pIndex != nullptr);

    LOG_IF_EXCEPT(resp->set_len(pIndex->getSize()));
    resp->mutable_result()->set_err(result);

    return 0;
}

int SessionHandler::IndexFlatGetBase(const IndexFlatGetBaseRequest *req, IndexFlatGetBaseResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();

    ASCEND_THROW_IF_NOT_FMT(indices.find(indexId) != indices.end(), "Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexFlat *>(indices[indexId]);
    ASCEND_THROW_IF_NOT(pIndex != nullptr);

    LOG_IF_EXCEPT({
        std::vector<float16_t> vectors;
        pIndex->getVectors(req->offset(), req->num(), vectors);
        resp->set_data(vectors.data(), vectors.size() * sizeof(float16_t));
    });
    resp->mutable_result()->set_err(result);

    return 0;
}

int SessionHandler::IndexInt8FlatGetBase(const IndexFlatGetBaseRequest *req, IndexFlatGetBaseResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();

    ASCEND_THROW_IF_NOT_FMT(indices.find(indexId) != indices.end(), "Invalid index id: %d\n", indexId);
    switch (req->metric()) {
        case MetricType::METRIC_L2: {
            auto pIndex = reinterpret_cast<IndexInt8FlatL2 *>(indices[indexId]);
            ASCEND_THROW_IF_NOT(pIndex != nullptr);
            LOG_IF_EXCEPT({
                std::vector<int8_t> vectors;
                pIndex->getVectors(req->offset(), req->num(), vectors);
                resp->set_data(vectors.data(), vectors.size() * sizeof(uint8_t));
            });
        }
        case MetricType::METRIC_INNER_PRODUCT: {
            auto pIndex = reinterpret_cast<IndexInt8FlatCos *>(indices[indexId]);
            ASCEND_THROW_IF_NOT(pIndex != nullptr);
            LOG_IF_EXCEPT({
                std::vector<int8_t> vectors;
                pIndex->getVectors(req->offset(), req->num(), vectors);
                resp->set_data(vectors.data(), vectors.size() * sizeof(uint8_t));
            });
        }
    }

    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::IndexSQGetBase(const IndexSQGetBaseRequest *req, IndexSQGetBaseResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();

    ASCEND_THROW_IF_NOT_FMT(indices.find(indexId) != indices.end(), "Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexSQ *>(indices[indexId]);
    ASCEND_THROW_IF_NOT(pIndex != nullptr);

    LOG_IF_EXCEPT({
        std::vector<uint8_t> vectors;
        pIndex->getVectors(req->offset(), req->num(), vectors);
        resp->set_data(vectors.data(), vectors.size() * sizeof(uint8_t));
    });
    resp->mutable_result()->set_err(result);

    return 0;
}

int SessionHandler::IndexSQFastGetBase(const IndexSQGetBaseRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexSQ *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);

    int channelNumber = session->HdcGetFastSendChannel();

    size_t sendBatch =  pIndex->getSendBatch();
    size_t maxDataBufSize = sendBatch * pIndex->getDim(); 
    int loop = (pIndex->getSize() + sendBatch - 1) / sendBatch;

    // prepare fast send: malloc and exchange addr
    HdcRpcError ret = session->HdcFastSendPrepare(maxDataBufSize);
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    void* dataBuf = nullptr;
    void* ctrlBuf = nullptr;

    const int idxListSize = 0;
    const int idxListLast = 1;
    const int idxChannelLast = 2;

    int offset = 0;
    for (int i = 0; i < loop; i += channelNumber) {
        for (int j = i; j < i + channelNumber; j++) {
            if (j >= loop) {
                continue;
            }
            size_t num = (j < loop - 1) ? sendBatch : (pIndex->getSize() - j * sendBatch);
            int channelId = j % channelNumber;
            ret = session->HdcGetFastSendAddr(&dataBuf, &ctrlBuf, channelId);
            RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

            if (num > 0) {
                // codes
                pIndex->getVectors(offset, num, reinterpret_cast<uint8_t*>(dataBuf));
            }
            offset += num;

            int* sendCtrlBuf = reinterpret_cast<int*>(ctrlBuf);
            sendCtrlBuf[idxListSize] = num; // send num
            sendCtrlBuf[idxListLast] = (offset == pIndex->getSize()) ? 1 : 0;
            if (channelId == channelNumber - 1 || j == loop - 1) {
                sendCtrlBuf[idxChannelLast] = 1;
            } else {
                sendCtrlBuf[idxChannelLast] = 0;
            }
            // fast send
            ret = session->HdcFastSend(dataBuf, ctrlBuf, channelId);
            RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);
        }
        // wait sync
        ret = session->HdcWaitRecvSignal();
        RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);
    }
    // fast release: free fast memory
    ret = session->HdcFastSendRelease();
    RPC_ASSERT(ret == HDC_RPC_ERROR_NONE);

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexIVFUpdateNprobe(const IndexIVFUpdateNprobeRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t nprobe = req->nprobe();
    RPC_TIME_LOG("index %d update nprobe to %d\n", indexId, nprobe);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = dynamic_cast<IndexIVF *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    LOG_IF_EXCEPT(pIndex->setNumProbes(nprobe));

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexInt8IVFUpdateNprobe(const IndexIVFUpdateNprobeRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t nprobe = req->nprobe();
    RPC_TIME_LOG("index %d update nprobe to %d\n", indexId, nprobe);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    auto pIndex = reinterpret_cast<IndexInt8IVF *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    LOG_IF_EXCEPT(pIndex->setNumProbes(nprobe));

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexSearch(const IndexSearchRequest *req, IndexSearchResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t n = req->n();
    int32_t dim = req->dim();
    int32_t k = req->k();
    const std::string &query = req->query();
    RPC_ASSERT(query.size() == n * dim * sizeof(uint16_t));

    std::vector<uint16_t> distance(n * k);
    std::vector<uint32_t> label(n * k);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    RPC_TIME_LOG("index %d search actual start\n", indexId);
    LOG_IF_EXCEPT(indices[indexId]->search(n, reinterpret_cast<const float16_t *>(query.data()), k,
        reinterpret_cast<float16_t *>(distance.data()), reinterpret_cast<Index::idx_t *>(label.data())));
    RPC_TIME_LOG("index %d search end\n", indexId);

    resp->set_distance(distance.data(), distance.size() * sizeof(uint16_t));
    resp->set_label(label.data(), label.size() * sizeof(uint32_t));
    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::IndexInt8Search(const IndexInt8SearchRequest *req, IndexInt8SearchResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t n = req->n();
    int32_t dim = req->dim();
    int32_t k = req->k();
    const std::string &query = req->query();
    RPC_ASSERT(query.size() == n * dim * sizeof(int8_t));

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    RPC_TIME_LOG("index %d search actual start\n", indexId);
    std::vector<uint32_t> label(n * k);
    std::vector<float16_t> distance(n * k);
    IndexInt8* pIndex = reinterpret_cast<IndexInt8 *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    LOG_IF_EXCEPT(pIndex->search(n, reinterpret_cast<const int8_t *>(query.data()), k,
                                 distance.data(), reinterpret_cast<Index::idx_t *>(label.data())));
    RPC_TIME_LOG("index %d search end\n", indexId);

    resp->set_distance(distance.data(), distance.size() * sizeof(float16_t));
    resp->set_label(label.data(), label.size() * sizeof(uint32_t));
    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::IndexRemoveIds(const IndexRemoveIdsRequest *req, IndexRemoveIdsResponse *resp)
{
    return RemoveIdsInternal(req, resp);
}

int SessionHandler::IndexRemoveRangeIds(const IndexRemoveRangeIdsRequest *req, IndexRemoveIdsResponse *resp)
{
    return RemoveRangeIdsInternal(req, resp);
}

int SessionHandler::IndexInt8RemoveIds(const IndexRemoveIdsRequest *req, IndexRemoveIdsResponse *resp)
{
    return RemoveIdsInternal(req, resp, true);
}

int SessionHandler::IndexInt8RemoveRangeIds(const IndexRemoveRangeIdsRequest *req, IndexRemoveIdsResponse *resp)
{
    return RemoveRangeIdsInternal(req, resp, true);
}

int SessionHandler::IndexReset(const IndexResetRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    RPC_TIME_LOG("index %d reset\n", indexId);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    RPC_TIME_LOG("index %d reset actual start\n", indexId);
    LOG_IF_EXCEPT(indices[indexId]->reset());
    RPC_TIME_LOG("index %d reset end\n", indexId);

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexInt8Reset(const IndexResetRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    RPC_TIME_LOG("index %d reset\n", indexId);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    RPC_TIME_LOG("index %d reset actual start\n", indexId);
    IndexInt8* pIndex = reinterpret_cast<IndexInt8 *>(indices[indexId]);
    RPC_REQUIRE_NOT_NULL(pIndex);
    LOG_IF_EXCEPT(pIndex->reset());
    RPC_TIME_LOG("index %d reset end\n", indexId);

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexReserveMemory(const IndexReserveMemRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    size_t numVec = static_cast<size_t>(req->n());
    RPC_TIME_LOG("index %d reset\n", indexId);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    RPC_TIME_LOG("index %d reserve mem actual start\n", indexId);
    LOG_IF_EXCEPT(indices[indexId]->reserveMemory(numVec));
    RPC_TIME_LOG("index %d reserve mem end\n", indexId);

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexInt8ReserveMemory(const IndexReserveMemRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    size_t numVec = static_cast<size_t>(req->n());
    RPC_TIME_LOG("index %d reset\n", indexId);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    RPC_TIME_LOG("index %d reserve mem actual start\n", indexId);
    IndexInt8* pIndex = reinterpret_cast<IndexInt8 *>(indices[indexId]);
    LOG_IF_EXCEPT(pIndex->reserveMemory(numVec));
    RPC_TIME_LOG("index %d reserve mem end\n", indexId);

    resp->set_err(result);
    return 0;
}

int SessionHandler::IndexReclaimMemory(const IndexReclaimMemRequest *req, IndexReclaimMemResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    RPC_TIME_LOG("index %d reset\n", indexId);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    RPC_TIME_LOG("index %d reclaim mem actual start\n", indexId);
    LOG_IF_EXCEPT({
        size_t reclaimed = indices[indexId]->reclaimMemory();
        resp->set_size(static_cast<uint32_t>(reclaimed));
    });
    RPC_TIME_LOG("index %d reclaim mem end\n", indexId);

    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::IndexInt8ReclaimMemory(const IndexReclaimMemRequest *req, IndexReclaimMemResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    RPC_TIME_LOG("index %d reset\n", indexId);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    RPC_TIME_LOG("index %d reclaim mem actual start\n", indexId);
    LOG_IF_EXCEPT({
        IndexInt8* pIndex = reinterpret_cast<IndexInt8 *>(indices[indexId]);
        size_t reclaimed = pIndex->reclaimMemory();
        resp->set_size(static_cast<uint32_t>(reclaimed));
    });
    RPC_TIME_LOG("index %d reclaim mem end\n", indexId);

    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::CreateLinearTransform(const CreateLinearTransformRequest *req, CreateTransformResponse *resp)
{
    RPC_TIME_LOG("CreateLinearTransform, dimIn=%d, dimOut=%d, haveBias=%d\n", req->dimin(), req->dimout(),
        req->havebias());

    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;

    VectorTransform *transform = nullptr;
    LOG_IF_EXCEPT(transform = new LinearTransform(req->dimin(), req->dimout()));
    if (transform == nullptr || result == CommonResponse_ErrorCode_ERROR) {
        resp->mutable_result()->set_err(CommonResponse_ErrorCode_ERROR);
        return 0;
    }

    index_id_t transformsId = transformsIdCounter++;
    transforms[transformsId] = static_cast<VectorTransform *>(transform);
    resp->set_transformid(transformsId);
    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::DestroyTransform(const DestroyTransformRequest *req, CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t transformId = req->transformid();
    RPC_TIME_LOG("DestroyTransform %d\n", transformId);

    RPC_ASSERT_FMT(transforms.find(transformId) != transforms.end(), "!!!Invalid transformId id: %d\n", transformId);
    LOG_IF_EXCEPT(delete transforms[transformId]);
    RPC_TIME_LOG("DestroyTransform end\n");
    transforms.erase(transformId);

    resp->set_err(result);
    return 0;
}

int SessionHandler::LinearTransformUpdateTrainedValue(const LinearTransformUpdateTrainedValueRequest *req,
    CommonResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t transformsId = req->transformid();
    int32_t dimIn = req->dimin();
    int32_t dimOut = req->dimout();
    const std::string &matrix = req->matrix();
    const std::string &bias = req->bias();
    RPC_TIME_LOG("index %d: UpdateLinearTransformTrainedValue, dimIn=%d, dimOut=%d\n", transformsId, dimIn, dimOut);
    RPC_ASSERT(matrix.size() == dimIn * dimOut * sizeof(uint16_t));
    RPC_ASSERT(bias.size() == 0 || bias.size() == dimOut * sizeof(float));
    auto tmpMatrix = const_cast<char *>(matrix.data());
    auto tmpBias = const_cast<char *>(bias.data());
    AscendTensor<float16_t, DIMS_2> matrixTensor(reinterpret_cast<float16_t *>(tmpMatrix), { dimOut, dimIn });
    AscendTensor<float, DIMS_1> biasTensor(reinterpret_cast<float *>(tmpBias), { dimOut });

    RPC_ASSERT_FMT(transforms.find(transformsId) != transforms.end(), "!!!Invalid transforms id: %d\n", transformsId);
    auto pTransform = dynamic_cast<LinearTransform *>(transforms[transformsId]);
    RPC_REQUIRE_NOT_NULL(pTransform);
    LOG_IF_EXCEPT(pTransform->updateTrainedValue(matrixTensor, biasTensor));

    resp->set_err(result);
    return 0;
}

int SessionHandler::CreateNNDimReduction(const NNDimReductionCreateRequest *req, NNDimReductionCreateResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    const std::string &modelPath = req->modelpath();
    if (access(modelPath.c_str(), F_OK) == -1) {
        printf("%s does not exit\n", modelPath.c_str());
        result = CommonResponse_ErrorCode_ERROR;
        resp->mutable_result()->set_err(result);
        return 0;
    }

    processModel = std::make_shared<ModelProcess>();
    processModel->LoadModelFromFileWithMem(modelPath);
    processModel->CreateDesc();
    processModel->CreateOutput();

    resp->mutable_result()->set_err(result);

    return 0;
}

int SessionHandler::InferNNDimReduction(const NNDimReductionInferRequest *req, NNDimReductionInferResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    uint32_t n = req->n();
    uint32_t dimIn = req->dimin();
    uint32_t dimOut = req->dimout();
    uint32_t batchSize = req->batchsize();
    const std::string &data = req->data();
    if (n * dimIn != data.size() / sizeof(float)) {
        result = CommonResponse_ErrorCode_ERROR;
    }
    if (processModel == nullptr) {
        result = CommonResponse_ErrorCode_ERROR;
    }

    if (result == CommonResponse_ErrorCode_OK) {
        auto tmp = const_cast<char *>(data.data());
        auto inputData = reinterpret_cast<float *>(tmp);

        NNDimReduction DimReduction(n, dimIn, dimOut, batchSize, inputData);
        DimReduction.Process(processModel);
        
        const std::vector<float> &outputData = DimReduction.GetResultData();
        if (n * dimOut != outputData.size()) {
            result = CommonResponse_ErrorCode_ERROR;
        } else {
            resp->set_data(outputData.data(), outputData.size() * sizeof(float));
        }
    }

    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::DestroyNNDimReduction(const NNDimReductionDestroyRequest *req, NNDimReductionDestroyResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    if (processModel != nullptr) {
        processModel.reset();
    }
    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::TestDataIntegrity(const TestDataIntegrityRequest *req, TestDataIntegrityResponse *resp)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    const std::string &data = req->data();
    uint8_t checksum = calcChecksum(data.data(), data.size());
    if (req->len() != data.size() || static_cast<uint32_t>(checksum) != req->checksum()) {
        result = CommonResponse_ErrorCode_ERROR;
    }

    resp->mutable_result()->set_err(result);
    if (result == CommonResponse_ErrorCode_OK) {
        resp->set_data(data.data(), data.size());
    }

    return 0;
}

int SessionHandler::RemoveIdsInternal(const faiss_ascend::IndexRemoveIdsRequest *req,
                                      faiss_ascend::IndexRemoveIdsResponse *resp, bool isInt8Index)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    int32_t n = req->n();
    const std::string &ids = req->ids();
    RPC_TIME_LOG("index %d remove %d ids\n", indexId, n);
    RPC_ASSERT(ids.size() == n * sizeof(IDSelector::idx_t));
    IDSelectorBatch batch(n, reinterpret_cast<const IDSelector::idx_t *>(ids.data()));

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    RPC_TIME_LOG("index %d removeIds actual start\n", indexId);
    LOG_IF_EXCEPT({
        size_t numRemoved = isInt8Index ? reinterpret_cast<IndexInt8 *>(indices[indexId])->removeIds(batch) :
            indices[indexId]->removeIds(batch);
        resp->set_num(static_cast<uint32_t>(numRemoved));
    });
    RPC_TIME_LOG("index %d removeIds end\n", indexId);

    resp->mutable_result()->set_err(result);
    return 0;
}

int SessionHandler::RemoveRangeIdsInternal(const faiss_ascend::IndexRemoveRangeIdsRequest *req,
                                           faiss_ascend::IndexRemoveIdsResponse *resp, bool isInt8Index)
{
    CommonResponse_ErrorCode result = CommonResponse_ErrorCode_OK;
    index_id_t indexId = req->indexid();
    uint32_t minId = req->min();
    uint32_t maxId = req->max();
    RPC_TIME_LOG("index %d remove ids range(%d, %d)\n", indexId, minId, maxId);
    IDSelectorRange range(minId, maxId);

    RPC_ASSERT_FMT(indices.find(indexId) != indices.end(), "!!!Invalid index id: %d\n", indexId);
    RPC_TIME_LOG("index %d removeRangeIds actual start\n", indexId);
    LOG_IF_EXCEPT({
        size_t numRemoved = isInt8Index ? reinterpret_cast<IndexInt8 *>(indices[indexId])->removeIds(range) :
            indices[indexId]->removeIds(range);
        resp->set_num(static_cast<uint32_t>(numRemoved));
    });
    RPC_TIME_LOG("index %d removeRangeIds end\n", indexId);

    resp->mutable_result()->set_err(result);
    return 0;
}
} // ascend

