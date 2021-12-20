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

#include <faiss/ascend/AscendNNDimReduction.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/ascend/utils/fp16.h>
#include <faiss/ascend/rpc/AscendRpc.h>
#include <faiss/ascend/rpc/AscendRpcNNDimReduction.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <algorithm>
#include <omp.h>
#include <sys/time.h>
#include <faiss/ascend/utils/AscendUtils.h>

namespace faiss {
namespace ascend {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
// get pagesize must be less than 32M, becauseof rpc limitation
const size_t PAGE_SIZE = 32U * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of infer
const size_t VEC_SIZE = 512U * KB;
} // namespace

AscendNNDimReduction::AscendNNDimReduction(std::vector<int> deviceList, int dimIn, int dimOut,
                                           int batchSize, std::string &modelPath)
{
    FAISS_THROW_IF_NOT_MSG(dimIn > 0, "Invalid number of input dimensions");
    FAISS_THROW_IF_NOT_MSG(dimOut > 0, "Invalid number of output dimensions");
    FAISS_THROW_IF_NOT_MSG(batchSize > 0, "Invalid number of batchSize");

    this->verbose = true;
    this->deviceList = deviceList;
    this->dimIn = dimIn;
    this->dimOut = dimOut;
    this->batchSize = batchSize;
    this->modelPath = modelPath;
    this->pool = new AscendThreadPool(deviceList.size());
    initRpcCtx();
}

AscendNNDimReduction::~AscendNNDimReduction()
{
    if (pool != nullptr) {
        delete pool;
        pool = nullptr;
    }
    clearRpcCtx();
}

void AscendNNDimReduction::initRpcCtx()
{
    for (uint32_t i = 0; i < deviceList.size(); i++) {
        rpcContext ctx;
        RpcError ret = RpcCreateContext(deviceList[i], &ctx);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create context failed(%d).", ret);
        contextMap[deviceList[i]] = ctx;
        ret = RpcCreateNNDimReduction(contextMap[deviceList[i]], this->modelPath);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create IndexFlat failed(%d).", ret);
    }
}

void AscendNNDimReduction::infer(int n, const std::vector<float> &inputData, std::vector<float> &outputData)
{
    return inferPaged(n, inputData.data(), outputData);
}

void AscendNNDimReduction::infer(int n, const float* inputData, std::vector<float> &outputData)
{
    return inferPaged(n, inputData, outputData);
}

void AscendNNDimReduction::inferPaged(int n, const float* x, std::vector<float> &outputData)
{
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(outputData.size() == 0, "The initial size must be 0");
    FAISS_THROW_IF_NOT_FMT(n <= std::numeric_limits<int>::max(),
                           "ntotal exceeds max(%d)", std::numeric_limits<int>::max());
    if (n <= 0) {
        return;
    }

    size_t totalSize = static_cast<size_t>(n) * static_cast<size_t>(getElementSize());
    if (totalSize > PAGE_SIZE || static_cast<size_t>(n) > VEC_SIZE) {
        // How many vectors fit into kInferPageSize?
        size_t maxNumVecsForPageSize =
            PAGE_SIZE / getElementSize();
        // Always add at least 1 vector, if we have huge vectors
        maxNumVecsForPageSize = std::max(
            maxNumVecsForPageSize, (size_t)1);

        size_t tileSize = std::min((size_t)n, maxNumVecsForPageSize);

        for (size_t i = 0; i < (size_t)n; i += tileSize) {
            size_t curNum = std::min(tileSize, n - i);
            if (this->verbose) {
                printf("AscendNNDimReduction::infer: inferring %ld:%ld / %d\n", i, i + curNum, n);
            }
            inferImpl(curNum, x + i * (size_t)this->dimIn, outputData);
        }
    } else {
        if (this->verbose) {
            printf("AscendNNDimReduction::infer: inferring 0:%d / %d\n", n, n);
        }
        inferImpl(n, x, outputData);
    }
}

void AscendNNDimReduction::inferImpl(int n, const float *x, std::vector<float> &outputData)
{
    FAISS_ASSERT(n > 0);
    size_t deviceCnt = deviceList.size();
    std::vector<int> inferMap(deviceCnt, 0);

    // Allocate the total to each chip
    for (size_t i = 0; i < deviceCnt; i++) {
        inferMap[i] += n / deviceCnt;
    }
    for (size_t i = 0; i < n % deviceCnt; i++) {
        inferMap[i] += 1;
    }
    int offsum = 0;
    std::vector<int> offsumMap(deviceCnt, 0);
    for (size_t i = 0; i < deviceCnt; i++) {
        offsumMap[i] = offsum;
        int num = inferMap.at(i);
        offsum += num;
    }
    
    std::vector<std::vector<float>> inferResult(deviceCnt, std::vector<float>());
    auto inferFunctor = [&](int idx) {
        int num = inferMap.at(idx);
        if (num != 0) {
            int deviceId = deviceList[idx];
            rpcContext ctx = contextMap.at(deviceId);
            RpcError ret = RpcInferNNDimReduction(ctx, num, this->dimIn, this->dimOut, this->batchSize,
                x + offsumMap[idx] * this->dimIn, inferResult[idx]);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "infer failed(%d).", ret);
        }
    };
    // Call multi-thread and multi-chip parallel inference
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, inferFunctor);
    for (size_t i = 0; i < deviceCnt; i++) {
        int num = inferMap.at(i);
        if (num == 0) {
            continue;
        }
        outputData.insert(outputData.end(), inferResult[i].begin(), inferResult[i].end());
    }
}

int AscendNNDimReduction::getElementSize() const
{
    return this->dimIn * sizeof(float);
}

void AscendNNDimReduction::clearRpcCtx()
{
    for (uint32_t i = 0; i < deviceList.size(); i++) {
        rpcContext ctx = contextMap[deviceList[i]];
        RpcError ret = RpcDestroyNNDimReduction(ctx);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy context failed(%d).", ret);
        ret = RpcDestroyContext(ctx);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy context failed(%d).", ret);
    }
    contextMap.clear();
}
} // ascend
} // faiss