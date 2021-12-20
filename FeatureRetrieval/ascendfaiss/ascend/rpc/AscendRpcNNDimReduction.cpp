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

#include <faiss/ascend/rpc/AscendRpcNNDimReduction.h>
#include <faiss/ascend/rpc/HdcClient.h>

namespace faiss {
namespace ascend {
RpcError RpcCreateNNDimReduction(rpcContext ctx, std::string &modelPath)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    NNDimReductionCreateRequest req;
    NNDimReductionCreateResponse resp;

    req.set_modelpath(modelPath.data(), modelPath.size());

    HdcRpcError ret = session->SendAndReceive(RPC_CREATE_NN_DIM_REDUCTION, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}

RpcError RpcInferNNDimReduction(rpcContext ctx, int n, int dimIn, int dimOut, int batchSize, 
                                const float* data, std::vector<float> &outputData)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    NNDimReductionInferRequest req;
    NNDimReductionInferResponse resp;
    req.set_n(n);
    req.set_dimin(dimIn);
    req.set_dimout(dimOut);
    req.set_batchsize(batchSize);
    req.set_data(data, n * dimIn * sizeof(float));
    HdcRpcError ret = session->SendAndReceive(RPC_INFER_NN_DIM_REDUCTION, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    const std::string &receivedData = resp.data();
    auto tmp = const_cast<char *>(receivedData.data());
    auto receivedFloat = reinterpret_cast<float *>(tmp);
    outputData.insert(outputData.end(), receivedFloat, receivedFloat + receivedData.size() / sizeof(float));
    return RPC_ERROR_NONE;
}

RpcError RpcDestroyNNDimReduction(rpcContext ctx)
{
    HdcSession *session = static_cast<HdcSession *>(ctx);
    RPC_REQUIRE_NOT_NULL(session);

    NNDimReductionDestroyRequest req;
    NNDimReductionDestroyResponse resp;

    HdcRpcError ret = session->SendAndReceive(RPC_DESTROY_NN_DIM_REDUCTION, req, resp);
    if (ret != HDC_RPC_ERROR_NONE || resp.result().err() != CommonResponse_ErrorCode_OK) {
        return RPC_ERROR_ERROR;
    }

    return RPC_ERROR_NONE;
}
} // namespace ascend
} // namespace faiss