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

#include <faiss/utils/distances.h>

#include <faiss/ascend/AscendIndexIVFSQ.h>
#include <faiss/ascend/rpc/AscendRpc.h>
#include <faiss/ascend/utils/AscendUtils.h>
#include <faiss/ascend/utils/AscendThreadPool.h>
#include <faiss/ascend/utils/fp16.h>
#include <algorithm>
#include <omp.h>

namespace faiss {
namespace ascend {
struct IVFSQAddInfo {
    IVFSQAddInfo(int idx, int deviceNum, size_t size) : addDeviceIdx(idx), deviceAddNum(deviceNum, 0), codeSize(size)
    {
        FAISS_ASSERT(idx < deviceNum);
    }

    inline void Add(uint8_t *code, const Index::idx_t *id, float *precompute)
    {
        deviceAddNum[addDeviceIdx] += 1;
        addDeviceIdx = (addDeviceIdx + 1) % deviceAddNum.size();
        codes.insert(codes.end(), code, code + codeSize);
        ids.emplace_back((uint32_t)(*id));
        precomputes.emplace_back((float)(*precompute));
    }

    inline int GetOffSet(int idx) const
    {
        int ret = 0;
        FAISS_THROW_IF_NOT_MSG(idx < (int)deviceAddNum.size(), "idx error");
        for (int i = 0; i < idx; i++) {
            ret += deviceAddNum[i];
        }

        return ret;
    }

    inline void GetCodeAndIdPtr(int idx, uint8_t **codePtr, uint32_t **idPtr, float **precompute)
    {
        FAISS_THROW_IF_NOT_MSG(idx < (int)deviceAddNum.size(), "idx error");
        int off = GetOffSet(idx);
        *codePtr = codes.data() + off * codeSize;
        *idPtr = ids.data() + off;
        *precompute = precomputes.data() + off;
        return;
    }

    inline int GetAddNum(int idx) const
    {
        FAISS_THROW_IF_NOT_MSG(idx < (int)deviceAddNum.size(), "idx error");
        return deviceAddNum[idx];
    }

    int addDeviceIdx;
    std::vector<int> deviceAddNum;
    size_t codeSize;
    std::vector<uint8_t> codes;
    std::vector<uint32_t> ids;
    std::vector<float> precomputes;
};

AscendIndexIVFSQ::AscendIndexIVFSQ(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQConfig config)
    : AscendIndexIVF(index->d, index->metric_type, index->nlist, config),
      sq(index->sq),
      ivfsqConfig(config),
      byResidual(index->by_residual)
{
    FAISS_THROW_IF_NOT_MSG(config.deviceList.size(), "device list can not be empty");
    copyFrom(index);
}

AscendIndexIVFSQ::AscendIndexIVFSQ(int dims, int nlist,
    faiss::ScalarQuantizer::QuantizerType qtype, faiss::MetricType metric,
    bool encodeResidual, AscendIndexIVFSQConfig config)
    : AscendIndexIVF(dims, metric, nlist, config),
      sq(dims, qtype),
      ivfsqConfig(config),
      byResidual(encodeResidual)
{
    FAISS_THROW_IF_NOT_MSG(config.deviceList.size(), "device list can not be empty");

    checkParams();

    initRpcCtx();

    initDeviceAddNumMap();

    this->is_trained = false;
}

AscendIndexIVFSQ::~AscendIndexIVFSQ()
{
    auto functor = [&](rpcContext ctx, int indexId) {
        RpcError ret = RpcDestroyIndexIVFSQ(ctx, indexId);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);

        ret = RpcDestroyContext(ctx);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);
    };

    // parallel deconstruct to every device
    CALL_PARALLEL_FUNCTOR_INDEXMAP(indexMap, pool, functor);

    indexMap.clear();
    contextMap.clear();
}

void AscendIndexIVFSQ::copyFrom(const faiss::IndexIVFScalarQuantizer *index)
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    AscendIndexIVF::copyFrom(index);

    sq = index->sq;
    byResidual = index->by_residual;
    checkParams();
    clearRpcCtx();
    initRpcCtx();
    initDeviceAddNumMap();

    // The other index might not be trained
    if (!index->is_trained) {
        return;
    }

    this->is_trained = true;

    // Copy our lists as well. The product quantizer must have data in it
    FAISS_THROW_IF_NOT_MSG(sq.trained.size() > 0, "index is trained but trained is empty.");

    updateDeviceCoarseCenter();
    updateDeviceSQTrainedValue();

    // copy cpu index's codes to ascend device
    copyCodes(index);
}

void AscendIndexIVFSQ::copyTo(faiss::IndexIVFScalarQuantizer *index) const
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    AscendIndexIVF::copyTo(index);
    index->by_residual = byResidual;
    index->sq = sq;
    index->code_size = sq.code_size;

    InvertedLists *ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (this->is_trained && this->ntotal > 0) {
        // use for(deviceList) rather than for(auto& index : indexMap),
        // to ensure merged codes and ids in sequence
        for (size_t i = 0; i < indexConfig.deviceList.size(); i++) {
            int deviceId = indexConfig.deviceList[i];
            rpcContext ctx = contextMap.at(deviceId);
            int indexId = indexMap.at(ctx);

            RpcError ret = RpcIndexIVFFastGetListCodes(ctx, indexId, nlist, ivf);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Get list of IndexIVFSQ failed(%d).", ret);
        }
    }
}

void AscendIndexIVFSQ::copyCodes(const faiss::IndexIVFScalarQuantizer *index)
{
    const InvertedLists *ivf = index->invlists;
    size_t nlist = ivf ? ivf->nlist : 0;
    size_t deviceCnt = indexConfig.deviceList.size();

    // seperate codes to every device
    for (size_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);

        // Ascend index can only support max int entries per list
        FAISS_THROW_IF_NOT_FMT(listSize <= (size_t)std::numeric_limits<int>::max(),
                               "Ascend inverted list can only support %zu entries; %zu found",
                               (size_t)std::numeric_limits<int>::max(), listSize);

        int listNumPerDevice = listSize / deviceCnt;
        int listNumLast = listSize % deviceCnt;
        for (size_t j = 0; j < deviceCnt; j++) {
            deviceAddNumMap[i][j] += listNumPerDevice;
        }

        for (int j = 0; j < listNumLast; j++) {
            deviceAddNumMap[i][j] += 1;
        }
    }

// update basecode to device
#pragma omp parallel for
    for (size_t i = 0; i < nlist; ++i) {
        uint32_t offsum = 0;
        size_t listSize = ivf->list_size(i);

        // precompute term 2 from codes
        std::vector<float> precomputeVals(listSize, 0.0f);
        calcPrecompute(ivf->get_codes(i), precomputeVals.data(), listSize);

        for (size_t j = 0; j < deviceCnt; j++) {
            int n = deviceAddNumMap[i][j];
            if (n == 0) {
                continue;
            }
            int deviceId = indexConfig.deviceList[j];
            rpcContext ctx = contextMap.at(deviceId);
            int indexId = indexMap.at(ctx);

            // get codes data's ptr and precompute data's ptr and indices's ptr
            uint8_t *offsetCodes = const_cast<uint8_t *>(ivf->get_codes(i)) + offsum * ivf->code_size;
            float *precompute = const_cast<float *>(precomputeVals.data()) + offsum;
            idx_t *offsetIds = const_cast<idx_t *>(ivf->get_ids(i)) + offsum;

            // indices stored in devices in uint32_t format, trandform idx_t to uint32
            std::vector<uint32_t> indice(offsetIds, offsetIds + n);
            transform(offsetIds, offsetIds + n, begin(indice), [](idx_t x) { return uint32_t(x); });

            RpcError ret = RpcIndexIVFSQAdd(ctx, indexId, n, i, offsetCodes, ivf->code_size, indice.data(), precompute);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add with ids failed(%d).", ret);
            offsum += n;
        }
    }
}

void AscendIndexIVFSQ::train(Index::idx_t n, const float *x)
{
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n should be a positive value.");

    if (this->is_trained) {
        FAISS_ASSERT(cpuQuantizer->is_trained);
        FAISS_ASSERT(cpuQuantizer->ntotal == nlist);
        FAISS_THROW_IF_NOT_MSG(indexMap.size() > 0, "indexMap.size must be >0");
        return;
    }

    // train L1 IVF quantizer
    trainQuantizer(n, x);
    updateDeviceCoarseCenter();

    // train L2 quantizer
    trainResidualQuantizer(n, x);

    this->is_trained = true;
}

void AscendIndexIVFSQ::addImpl(int n, const float *x, const Index::idx_t *ids)
{
    //    v_i = (code_i + 0.5) / 255 * vdiff_i + vmin_i (reconstruct code to float)
    //    distance = || x - v ||^2
    //             = x_i^2 - 2 * (x_i * v_i) + v_i^2
    //               --------------------------    -------
    //                       term 1                 term 2
    //    term2 can be precomputed when adding database
    const float *xi = x;
    size_t deviceCnt = indexConfig.deviceList.size();

    // calculate which list the vectors belongs to
    Index::idx_t *assign = new Index::idx_t[n];
    faiss::ScopeDeleter<Index::idx_t> del1(assign);
    cpuQuantizer->assign(n, x, assign);

    faiss::ScopeDeleter<float> del2;
    if (byResidual) {
        // get the residuals with the l1 coarse centroids
        float *residuals = new float[n * this->d];
        del2.set(residuals);
        cpuQuantizer->compute_residual_n(n, x, residuals, assign);
        xi = residuals;
    }

    // compute the sq codes
    uint8_t *codes = new uint8_t[n * sq.code_size];
    faiss::ScopeDeleter<uint8_t> del3(codes);
    sq.compute_codes(xi, codes, n);

    // precompute values(term2), reuse residuals mem if byResidual
    float *precomputeVals = new float[n];
    faiss::ScopeDeleter<float> del4(precomputeVals);
    calcPrecompute(codes, precomputeVals, n, const_cast<float *>(del2.ptr));

    // list id -> # being added,
    // combine the codes(codes assigned to the same IVFList) together
    std::unordered_map<int, IVFSQAddInfo> assignCounts;
    for (int i = 0; i < n; i++) {
        Index::idx_t listId = assign[i];
        FAISS_ASSERT(listId >= 0 && listId < this->nlist);
        auto it = assignCounts.find(listId);
        if (it != assignCounts.end()) {
            it->second.Add(codes + i * sq.code_size, ids + i, precomputeVals + i);
            continue;
        }

        int devIdx = 0;
        for (size_t j = 1; j < deviceCnt; j++) {
            if (deviceAddNumMap[listId][j] < deviceAddNumMap[listId][devIdx]) {
                devIdx = j;
                break;
            }
        }

        assignCounts.emplace(listId, IVFSQAddInfo(devIdx, deviceCnt, sq.code_size));
        assignCounts.at(listId).Add(codes + i * sq.code_size, ids + i, precomputeVals + i);
    }

    auto addFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        rpcContext ctx = contextMap[deviceId];
        int indexId = indexMap[ctx];
        for (auto &centroid : assignCounts) {
            int listId = centroid.first;
            int num = centroid.second.GetAddNum(idx);
            if (num == 0) {
                continue;
            }

            uint8_t *codePtr = nullptr;
            uint32_t *idPtr = nullptr;
            float *precompPtr = nullptr;
            centroid.second.GetCodeAndIdPtr(idx, &codePtr, &idPtr, &precompPtr);
            RpcError ret = RpcIndexIVFSQAdd(ctx, indexId, num, listId, codePtr, sq.code_size, idPtr, precompPtr);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add with ids failed(%d).", ret);
            deviceAddNumMap[listId][idx] += num;
        }
    };

    // parallel adding codes to every device
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, addFunctor);

    this->ntotal += n;
}

void AscendIndexIVFSQ::checkParams()
{
    // only support L2
    FAISS_THROW_IF_NOT_MSG(this->metric_type == MetricType::METRIC_L2 ||
        this->metric_type == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");

    // only support SQ8
    FAISS_THROW_IF_NOT_MSG(sq.qtype == faiss::ScalarQuantizer::QT_8bit ||
        sq.qtype == faiss::ScalarQuantizer::QT_8bit_uniform,
        "Unsupported qtype");

    // Must have some number of lists
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be > 0");

    // Diamension must be 16 aligned
    const int alignedNum = 16;
    FAISS_THROW_IF_NOT_FMT(this->d % alignedNum == 0, "Number of diamension (%d) must be 16 aligned.", this->d);
}

void AscendIndexIVFSQ::initRpcCtx()
{
    indexMap.clear();
    contextMap.clear();
    for (size_t i = 0; i < indexConfig.deviceList.size(); i++) {
        int indexId;
        rpcContext ctx;
        int deviceId = indexConfig.deviceList[i];

        RpcError ret = RpcCreateContext(deviceId, &ctx);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create context failed(%d).", ret);
        contextMap[deviceId] = ctx;

        ret = RpcCreateIndexIVFSQ(ctx, indexId, this->d, nlist, byResidual, sq.qtype, this->metric_type, nprobe,
            indexConfig.resourceSize);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create IndexIVFSQ failed(%d).", ret);
        indexMap[ctx] = indexId;
    }
}

void AscendIndexIVFSQ::clearRpcCtx()
{
    for (auto &index : indexMap) {
        RpcError ret = RpcDestroyIndexIVFSQ(index.first, index.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy IndexIVFSQ failed(%d).", ret);

        ret = RpcDestroyContext(index.first);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy context failed(%d).", ret);
    }

    indexMap.clear();
    contextMap.clear();
}

void AscendIndexIVFSQ::trainResidualQuantizer(Index::idx_t n, const float *x)
{
    if (this->verbose) {
        printf("training scalar quantizer on %ld vectors in %dD\n", n, this->d);
    }

    // The input is already guaranteed to be on the CPU
    sq.train_residual(n, x, this->cpuQuantizer, this->byResidual, this->verbose);
    updateDeviceSQTrainedValue();
}

void AscendIndexIVFSQ::updateDeviceSQTrainedValue()
{
    // convert trained value to fp16, *2 means vmin and vdiff
    std::vector<uint16_t> trainedFp16(this->d * 2);
    uint16_t *vmin = trainedFp16.data();
    uint16_t *vdiff = trainedFp16.data() + this->d;

    switch (sq.qtype) {
        case faiss::ScalarQuantizer::QT_8bit:
            transform(begin(sq.trained), end(sq.trained), begin(trainedFp16),
                [](float temp) { return fp16(temp).data; });
            break;
        case faiss::ScalarQuantizer::QT_8bit_uniform:
            for (int i = 0; i < this->d; i++) {
                *(vmin + i) = fp16(sq.trained[0]).data;
                *(vdiff + i) = fp16(sq.trained[1]).data;
            }
            break;
        default:
            FAISS_THROW_FMT("not supportted qtype(%d).", sq.qtype);
            break;
    }

    for (auto &index : indexMap) {
        RpcError ret = RpcIndexSQUpdateTrainedValue(index.first, index.second, this->d, vmin, vdiff, true);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Update trained value failed(%d).", ret);
    }
}

void AscendIndexIVFSQ::calcPrecompute(const uint8_t *codes, float *compute, size_t n, float *xMem)
{
    faiss::ScopeDeleter<float> del;
    float *x = xMem;
    if (!x) {
        x = new float[n * this->d];
        del.set(x);
    }
    sq.decode(codes, x, n);

    fvec_norms_L2sqr(compute, x, this->d, n);
}

int AscendIndexIVFSQ::getElementSize() const
{
    // element size: codesize + sizeof(ID) + sizeof(preCompute)
    return sq.code_size + sizeof(uint32_t) + sizeof(float);
}
} // ascend
} // faiss
