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

#include <map>
#include <atomic>
#include <unistd.h>

#include <google/protobuf/message.h>
#include "common/AscendIndex.pb.h"

#include "common/HdcBase.h"
#include "common/RpcUtils.h"
#include "ascenddaemon/impl/Index.h"
#include "ascenddaemon/impl/VectorTransform.h"
#include "ascenddaemon/utils/ModelProcess.h"

namespace faiss_ascend = ::faiss::ascend;

namespace ascend {
class SessionHandler {
public:
    using RpcHandler = int (SessionHandler::*)(int, int);
    explicit SessionHandler(faiss::ascend::HdcSession *session);
    ~SessionHandler();
    void Handle();

    void RegisteRpcHandler();
    int HandleRpcCommon(int rpcType, int msgLen);
    int HandleRpcIndexIVFCommon(int rpcType, int msgLen);
    int HandleRpcIndexFlat(int rpcType, int msgLen);
    int HandleRpcIndexInt8(int rpcType, int msgLen);
    int HandleRpcIndexInt8IVF(int rpcType, int msgLen);
    int HandleRpcIndexInt8IVFFlat(int rpcType, int msgLen);
    int HandleRpcIndexInt8Flat(int rpcType, int msgLen);
    int HandleRpcIndexSq8(int rpcType, int msgLen);
    int HandleRpcIndexIVFPQ(int rpcType, int msgLen);
    int HandleRpcIndexTransform(int rpcType, int msgLen);
    int HandleRpcNNDimReduction(int rpcType, int msgLen);

    int CreateIndexIVFPQ(const faiss_ascend::CreateIndexIVFPQRequest *req,
                         faiss_ascend::CreateIndexResponse *resp);
    int CreateIndexIVFFlat(const faiss_ascend::CreateIndexIVFFlatRequest *req,
                           faiss_ascend::CreateIndexResponse *resp);
    int CreateIndexFlat(const faiss_ascend::CreateIndexFlatRequest *req,
                        faiss_ascend::CreateIndexResponse *resp);
    int CreateIndexInt8IVFFlat(const faiss_ascend::CreateIndexInt8IVFFlatRequest *req,
                               faiss_ascend::CreateIndexResponse *resp);
    int CreateIndexInt8Flat(const faiss_ascend::CreateIndexInt8FlatRequest *req,
                            faiss_ascend::CreateIndexResponse *resp);
    int CreateIndexIVFSQ(const faiss_ascend::CreateIndexIVFSQRequest *req,
                         faiss_ascend::CreateIndexResponse *resp);
    int CreateIndexSQ(const faiss_ascend::CreateIndexSQRequest *req,
                      faiss_ascend::CreateIndexResponse *resp);
    int CreateIndexPreTransform(const faiss_ascend::CreateIndexPreTransformRequest *req,
                                faiss_ascend::CreateIndexResponse *resp);

    int DestroyIndex(const faiss_ascend::DestroyIndexRequest *req,
                     faiss_ascend::CommonResponse *resp);
    int DestroyIndexInt8(const faiss_ascend::DestroyIndexRequest *req,
                         faiss_ascend::CommonResponse *resp);

    int IndexIVFUpdateCoarseCent(const faiss_ascend::IndexIVFUpdateCoarseCentRequest *req,
                                 faiss_ascend::CommonResponse *resp);
    int IndexInt8IVFUpdateCoarseCent(const faiss_ascend::IndexIVFUpdateCoarseCentRequest *req,
                                     faiss_ascend::CommonResponse *resp);
    int IndexIVFPQUpdatePQCent(const faiss_ascend::IndexIVFPQUpdatePQCentRequest *req,
                               faiss_ascend::CommonResponse *resp);
    int IndexSQUpdateTrainedValue(const faiss_ascend::IndexSQUpdateTrainedValueRequest *req,
                                  faiss_ascend::CommonResponse *resp);

    int IndexIVFPQAdd(const faiss_ascend::IndexIVFPQAddRequest *req,
                      faiss_ascend::CommonResponse *resp);

    int IndexIVFFlatAdd(const faiss_ascend::IndexIVFFlatAddRequest *req,
                        faiss_ascend::CommonResponse *resp);
    int IndexFlatAdd(const faiss_ascend::IndexFlatAddRequest *req,
                     faiss_ascend::CommonResponse *resp);
    int IndexInt8IVFFlatAdd(const faiss_ascend::IndexInt8IVFFlatAddRequest *req,
                            faiss_ascend::CommonResponse *resp);
    int IndexInt8FlatAdd(const faiss_ascend::IndexInt8FlatAddRequest *req,
                         faiss_ascend::CommonResponse *resp);
    int IndexIVFSQAdd(const faiss_ascend::IndexIVFSQAddRequest *req,
                      faiss_ascend::CommonResponse *resp);
    int IndexSQAdd(const faiss_ascend::IndexSQAddRequest *req,
                   faiss_ascend::CommonResponse *resp);
    int IndexPreTransformPrepend(const faiss_ascend::IndexPreTransformPrependRequest *req,
                                 faiss_ascend::CommonResponse *resp);

    int IndexIVFGetListLength(const faiss_ascend::IndexIVFGetListInfoRequest *req,
                              faiss_ascend::IndexIVFGetListLengthResponse *resp);
    int IndexInt8IVFGetListLength(const faiss_ascend::IndexIVFGetListInfoRequest *req,
                                  faiss_ascend::IndexIVFGetListLengthResponse *resp);
    int IndexIVFGetListCodes(const faiss_ascend::IndexIVFGetListInfoRequest *req,
                             faiss_ascend::IndexIVFGetListCodesResponse *resp);
    int IndexIVFFastGetListCodes(const faiss_ascend::IndexIVFGetListInfoRequest *req,
                                 faiss_ascend::CommonResponse *resp);
    int IndexInt8IVFGetListCodes(const faiss_ascend::IndexIVFGetListInfoRequest *req,
                                 faiss_ascend::IndexIVFGetListCodesResponse *resp);
    int IndexFlatGetBaseLength(const faiss_ascend::IndexFlatGetBaseLengthRequest *req,
                               faiss_ascend::IndexFlatGetBaseLengthResponse *resp);
    int IndexInt8FlatGetBaseLength(const faiss_ascend::IndexFlatGetBaseLengthRequest *req,
                               faiss_ascend::IndexFlatGetBaseLengthResponse *resp);
    int IndexSQGetBaseLength(const faiss_ascend::IndexSQGetBaseLengthRequest *req,
                             faiss_ascend::IndexSQGetBaseLengthResponse *resp);
    int IndexFlatGetBase(const faiss_ascend::IndexFlatGetBaseRequest *req,
                         faiss_ascend::IndexFlatGetBaseResponse *resp);
    int IndexInt8FlatGetBase(const faiss_ascend::IndexFlatGetBaseRequest *req,
                         faiss_ascend::IndexFlatGetBaseResponse *resp);
    int IndexSQGetBase(const faiss_ascend::IndexSQGetBaseRequest *req,
                       faiss_ascend::IndexSQGetBaseResponse *resp);
    int IndexSQFastGetBase(const faiss_ascend::IndexSQGetBaseRequest *req,
                           faiss_ascend::CommonResponse *resp);
    int IndexIVFUpdateNprobe(const faiss_ascend::IndexIVFUpdateNprobeRequest *req,
                             faiss_ascend::CommonResponse *resp);
    int IndexInt8IVFUpdateNprobe(const faiss_ascend::IndexIVFUpdateNprobeRequest *req,
                                 faiss_ascend::CommonResponse *resp);
    int IndexSearch(const faiss_ascend::IndexSearchRequest *req,
                    faiss_ascend::IndexSearchResponse *resp);
    int IndexInt8Search(const faiss_ascend::IndexInt8SearchRequest *req,
                        faiss_ascend::IndexInt8SearchResponse *resp);
    int IndexInt8SearchFilter(const faiss_ascend::IndexInt8SearchFilterRequest *req,
                        faiss_ascend::IndexInt8SearchResponse *resp);
    int IndexRemoveIds(const faiss_ascend::IndexRemoveIdsRequest *req,
                       faiss_ascend::IndexRemoveIdsResponse *resp);
    int IndexInt8RemoveIds(const faiss_ascend::IndexRemoveIdsRequest *req,
                           faiss_ascend::IndexRemoveIdsResponse *resp);
    int IndexRemoveRangeIds(const faiss_ascend::IndexRemoveRangeIdsRequest *req,
                            faiss_ascend::IndexRemoveIdsResponse *resp);
    int IndexInt8RemoveRangeIds(const faiss_ascend::IndexRemoveRangeIdsRequest *req,
                            faiss_ascend::IndexRemoveIdsResponse *resp);
    int IndexReset(const faiss_ascend::IndexResetRequest *req,
                   faiss_ascend::CommonResponse *resp);
    int IndexInt8Reset(const faiss_ascend::IndexResetRequest *req,
                       faiss_ascend::CommonResponse *resp);
    int IndexReserveMemory(const faiss_ascend::IndexReserveMemRequest *req,
                           faiss_ascend::CommonResponse *resp);
    int IndexInt8ReserveMemory(const faiss_ascend::IndexReserveMemRequest *req,
                               faiss_ascend::CommonResponse *resp);
    int IndexReclaimMemory(const faiss_ascend::IndexReclaimMemRequest *req,
                           faiss_ascend::IndexReclaimMemResponse *resp);
    int IndexInt8ReclaimMemory(const faiss_ascend::IndexReclaimMemRequest *req,
                               faiss_ascend::IndexReclaimMemResponse *resp);
    int CreateLinearTransform(const faiss_ascend::CreateLinearTransformRequest *req,
                              faiss_ascend::CreateTransformResponse *resp);
    int DestroyTransform(const faiss_ascend::DestroyTransformRequest *req,
                         faiss_ascend::CommonResponse *resp);
    int LinearTransformUpdateTrainedValue(const faiss_ascend::LinearTransformUpdateTrainedValueRequest *req,
                                          faiss_ascend::CommonResponse *resp);
    int CreateNNDimReduction(const faiss_ascend::NNDimReductionCreateRequest *req, 
                             faiss_ascend::NNDimReductionCreateResponse *resp);
    int InferNNDimReduction(const faiss_ascend::NNDimReductionInferRequest *req,
                            faiss_ascend::NNDimReductionInferResponse *resp);
    int DestroyNNDimReduction(const faiss_ascend::NNDimReductionDestroyRequest *req, 
                             faiss_ascend::NNDimReductionDestroyResponse *resp);
    // for test
    int TestDataIntegrity(const faiss_ascend::TestDataIntegrityRequest *req,
                          faiss_ascend::TestDataIntegrityResponse *resp);

private:
    int RemoveIdsInternal(const faiss_ascend::IndexRemoveIdsRequest *req,
                          faiss_ascend::IndexRemoveIdsResponse *resp, bool isInt8Index = false);
    int RemoveRangeIdsInternal(const faiss_ascend::IndexRemoveRangeIdsRequest *req,
                               faiss_ascend::IndexRemoveIdsResponse *resp, bool isInt8Index = false);

private:
    faiss::ascend::HdcSession *session;
    std::atomic<faiss_ascend::index_id_t> indexIdCounter;
    std::map<faiss_ascend::index_id_t, ascend::Index *> indices;
    std::atomic<faiss_ascend::index_id_t> transformsIdCounter;
    std::map<faiss_ascend::index_id_t, ascend::VectorTransform *> transforms;
    std::map<int, RpcHandler> rpcHandlers;
    std::shared_ptr<ModelProcess> processModel;
};
} // namespace ascend
