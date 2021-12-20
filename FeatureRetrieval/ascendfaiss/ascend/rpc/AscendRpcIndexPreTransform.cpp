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

#include <faiss/ascend/rpc/AscendRpcIndexPreTransform.h>
#include <faiss/ascend/rpc/HdcClient.h>

namespace faiss {
namespace ascend {
static RpcError RpcDestroyTransform(rpcContext ctx, int transformId)
{
    DestroyTransformRequest req;
    CommonResponse resp;
    req.set_transformid(transformId);

    CALL_RPC(ctx, RPC_DESTROY_TRANSFORM, req, resp, resp.err());
    return RPC_ERROR_NONE;
}

RpcError RpcCreateIndexPreTransform(rpcContext ctx, int &indexId, int subindexId)
{
    CreateIndexPreTransformRequest req;
    CreateIndexResponse resp;
    req.set_subindexid(subindexId);

    CALL_RPC(ctx, RPC_CREATE_INDEX_PRETRANSFORM, req, resp, resp.result().err());
    indexId = resp.indexid();
    return RPC_ERROR_NONE;
}

RpcError RpcDestroyIndexPreTransform(rpcContext ctx, int indexId)
{
    return RpcDestroyIndex(ctx, indexId);
}

RpcError RpcCreateLinearTransform(rpcContext ctx, int &transformId, int dimIn, 
    int dimOut, bool haveBias)
{
    CreateLinearTransformRequest req;
    CreateTransformResponse resp;
    req.set_dimin(dimIn);
    req.set_dimout(dimOut);
    req.set_havebias(haveBias);

    CALL_RPC(ctx, RPC_CREATE_TRANSFORM_LINEAR, req, resp, resp.result().err());
    transformId = resp.transformid();
    return RPC_ERROR_NONE;
}

RpcError RpcDestroyLinearTransform(rpcContext ctx, int transformId)
{
    return RpcDestroyTransform(ctx, transformId);
}

RpcError RpcLinearTransformUpdateTrainedValue(rpcContext ctx, int transformId, 
    int dimIn, int dimOut, const uint16_t *matrix, const float *bias)
{
    LinearTransformUpdateTrainedValueRequest req;
    CommonResponse resp;
    req.set_transformid(transformId);
    req.set_dimin(dimIn);
    req.set_dimout(dimOut);
    req.set_matrix(matrix, dimIn * dimOut * sizeof(uint16_t));
    if (bias) {
        req.set_bias(bias, dimOut * sizeof(float));
    }

    CALL_RPC(ctx, RPC_TRANSFORM_LINEAR_UPDATE_TRAINED_VALUE, req, resp, resp.err());
    return RPC_ERROR_NONE;
}

RpcError RpcIndexPreTransformPrepend(rpcContext ctx, int indexId,
    int transformId)
{
    IndexPreTransformPrependRequest req;
    CommonResponse resp;
    req.set_indexid(indexId);
    req.set_transformid(transformId);
    CALL_RPC(ctx, RPC_INDEX_PRETRANSFORM_PREPEND, req, resp, resp.err());
    return RPC_ERROR_NONE;
}
} // namespace ascend
} // namespace faiss