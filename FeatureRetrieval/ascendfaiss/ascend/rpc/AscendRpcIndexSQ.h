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

#ifndef ASCEND_FAISS_RPC_INDEX_SQ_H
#define ASCEND_FAISS_RPC_INDEX_SQ_H

#include <faiss/ascend/rpc/AscendRpcCommon.h>
#include <faiss/IndexScalarQuantizer.h>

namespace faiss {
namespace ascend {
// Create/Destroy SQ Index

struct IndexSQParameter {
    IndexSQParameter(int d, faiss::ScalarQuantizer::QuantizerType quantizerType, 
        faiss::MetricType metricType, int resource) 
        : dim(d), qtype(quantizerType), metric(metricType), resourceSize(resource)
    {}

    int dim;
    faiss::ScalarQuantizer::QuantizerType qtype;
    faiss::MetricType metric;
    int resourceSize;
};

RpcError RpcCreateIndexSQ(rpcContext ctx, int &indexId, const IndexSQParameter &parameter);
RpcError RpcDestroyIndexSQ(rpcContext ctx, int indexId);

// add dataset to device
RpcError RpcIndexSQAdd(rpcContext ctx, int indexId, int n, int dim, 
    const uint8_t *data, const float *precomputedVal);

// Get the vector [offset, offset + num] in the SQ base
RpcError RpcIndexSQGetBase(rpcContext ctx, int indexId, uint32_t offset,
    uint32_t num, std::vector<uint8_t> &vectors);

// Fast Get the vector in the SQ base
RpcError RpcIndexSQFastGetBase(rpcContext ctx, int indexId, uint32_t elementSize,
    uint32_t num, std::vector<uint8_t> &vectors);

// Get SQ base size
RpcError RpcIndexSQGetBaseSize(rpcContext ctx, int indexId, uint32_t &size);
} // namespace ascend
} // namespace faiss
#endif
