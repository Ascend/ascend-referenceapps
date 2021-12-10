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

#include <faiss/ascend/AscendIndexIVFFlat.h>
#include <faiss/ascend/rpc/AscendRpc.h>
#include <faiss/ascend/utils/AscendUtils.h>
#include <faiss/ascend/utils/AscendThreadPool.h>
#include <faiss/ascend/utils/fp16.h>
#include <algorithm>
#include <omp.h>

namespace faiss {
namespace ascend {
namespace {
const int DIM_MAX = 512;
const int FP16_2_FP32 = 2;

struct AddItem {
    AddItem(int idx, int deviceNum, int d) : deviceId(idx), dim(d), deviceAddNum(deviceNum, 0)
    {
        FAISS_ASSERT(idx < deviceNum);
    }

    inline void Add(float *data, const Index::idx_t *id)
    {
        deviceAddNum[deviceId] += 1;
        deviceId = (deviceId + 1) % deviceAddNum.size();

        std::vector<uint16_t> data16(dim);
        transform(data, data + dim, data16.begin(), [](float x) { return fp16(x).data; });

        dataset.insert(dataset.end(), data16.data(), data16.data() + dim);
        ids.push_back(static_cast<uint32_t>(*id));
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

    inline void GetDataAndIdPtr(int idx, uint16_t** pData, uint32_t** pId)
    {
        FAISS_THROW_IF_NOT_MSG(idx < (int)deviceAddNum.size(), "idx error");
        int off = GetOffSet(idx);
        *pData = dataset.data() + off * dim;
        *pId = ids.data() + off;
    }

    inline int GetAddNum(int idx) const
    {
        FAISS_THROW_IF_NOT_MSG(idx < (int)deviceAddNum.size(), "idx error");
        return deviceAddNum[idx];
    }

    int deviceId;
    int dim;
    std::vector<int> deviceAddNum;
    std::vector<uint16_t> dataset;
    std::vector<uint32_t> ids;
};
}

AscendIndexIVFFlat::AscendIndexIVFFlat(const faiss::IndexIVFFlat *index, AscendIndexIVFFlatConfig config)
    : AscendIndexIVF(index->d, index->metric_type, index->nlist, config), ivfflatconfig(config)
{
    FAISS_THROW_IF_NOT_MSG(config.deviceList.size(), "device list can not be empty");
    copyFrom(index);
}

AscendIndexIVFFlat::AscendIndexIVFFlat(int dims, int nlist, faiss::MetricType metric, AscendIndexIVFFlatConfig config)
    : AscendIndexIVF(dims, metric, nlist, config), ivfflatconfig(config)
{
    FAISS_THROW_IF_NOT_MSG(config.deviceList.size(), "device list can not be empty");

    checkParams();

    initRpcCtx();

    initDeviceAddNumMap();

    this->is_trained = false;
}

AscendIndexIVFFlat::~AscendIndexIVFFlat()
{
    auto functor = [&](rpcContext ctx, int indexId) {
        RpcError ret = RpcDestroyIndexIVFFlat(ctx, indexId);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);

        ret = RpcDestroyContext(ctx);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);
    };

    // parallel deconstruct to every device
    CALL_PARALLEL_FUNCTOR_INDEXMAP(this->indexMap, this->pool, functor);

    this->indexMap.clear();
    this->contextMap.clear();
}

void AscendIndexIVFFlat::copyFrom(const faiss::IndexIVFFlat *index)
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    AscendIndexIVF::copyFrom(index);

    checkParams();
    clearRpcCtx();
    initRpcCtx();
    initDeviceAddNumMap();

    // The other index might not be trained
    if (!index->is_trained) {
        return;
    }

    this->is_trained = true;

    updateDeviceCoarseCenter();

    // copy cpu index's codes to ascend device
    copyCodes(index);
}

void AscendIndexIVFFlat::copyTo(faiss::IndexIVFFlat *index) const
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    AscendIndexIVF::copyTo(index);

    index->code_size = d * sizeof(float);

    InvertedLists *ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (this->is_trained && this->ntotal > 0) {
        // use for(deviceList) rather than for(auto& index : indexMap),
        // to ensure merged codes and ids in sequence
        for (size_t i = 0; i < indexConfig.deviceList.size(); i++) {
            int deviceId = indexConfig.deviceList[i];
            rpcContext ctx = contextMap.at(deviceId);
            int indexId = indexMap.at(ctx);

#pragma omp parallel for if (nlist > 100)
            for (int j = 0; j < nlist; ++j) {
                std::vector<uint8_t> code;
                std::vector<uint32_t> ids;

                RpcError ret = RpcIndexIVFGetListCodes(ctx, indexId, j, code, ids);
                FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Get list of IndexIVFFlat failed(%d).", ret);

                std::vector<idx_t> indice(ids.size());
                transform(ids.begin(), ids.end(), begin(indice), [](uint32_t x) { return idx_t(x); });

                uint16_t* codefp16 = reinterpret_cast<uint16_t *>(code.data());
                std::vector<float> dist(code.size() / FP16_2_FP32);
                transform(codefp16, codefp16 + code.size() / FP16_2_FP32, dist.begin(),
                    [](uint16_t temp) { return (float)fp16(temp); });

                // the code save as uint8
                ivf->add_entries(j, indice.size(), indice.data(), (const uint8_t*)dist.data());
            }
        }
    }
}

void AscendIndexIVFFlat::train(Index::idx_t n, const float *x)
{
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n should be a positive value.");

    if (this->is_trained) {
        FAISS_ASSERT(cpuQuantizer->is_trained);
        FAISS_ASSERT(cpuQuantizer->ntotal == nlist);
        FAISS_THROW_IF_NOT_MSG(indexMap.size() > 0, "indexMap.size must be >0");
        return;
    }

    trainQuantizer(n, x);
    updateDeviceCoarseCenter();
    this->is_trained = true;
}

void AscendIndexIVFFlat::addImpl(int n, const float *x, const Index::idx_t *ids)
{
    float *xi = const_cast<float *>(x);
    size_t deviceCnt = indexConfig.deviceList.size();

    // calculate which list the vectors belongs to
    Index::idx_t *assign = new Index::idx_t[n];
    faiss::ScopeDeleter<Index::idx_t> del1(assign);
    cpuQuantizer->assign(n, x, assign);

    // list id -> # being added,
    // combine the codes(codes assigned to the same IVFList) together
    std::unordered_map<int, AddItem> assignCounts;
    for (int i = 0; i < n; ++i) {
        Index::idx_t listId = assign[i];
        FAISS_ASSERT(listId >= 0 && listId < this->nlist);
        auto it = assignCounts.find(listId);
        if (it != assignCounts.end()) {
            it->second.Add(xi + i * this->d, ids + i);
            continue;
        }

        int devIdx = 0;
        for (size_t j = 1; j < deviceCnt; j++) {
            if (deviceAddNumMap[listId][j] < deviceAddNumMap[listId][devIdx]) {
                devIdx = j;
                break;
            }
        }

        assignCounts.emplace(listId, AddItem(devIdx, deviceCnt, this->d));
        assignCounts.at(listId).Add(xi + i * this->d, ids + i);
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

            uint16_t* pData = nullptr;
            uint32_t* pId = nullptr;
            centroid.second.GetDataAndIdPtr(idx, &pData, &pId);
            RpcError ret = RpcIndexIVFFlatAdd(ctx, indexId, num, listId, pData, this->d, pId);

            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add with ids failed(%d).", ret);
            deviceAddNumMap[listId][idx] += num;
        }
    };

    // parallel adding codes to every device
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, addFunctor);
    
    this->ntotal += n;
}

void AscendIndexIVFFlat::checkParams()
{
    // only support L2
    FAISS_THROW_IF_NOT_MSG(this->metric_type == MetricType::METRIC_L2, "Unsupported metric type");

    // Must have some number of lists
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be > 0");

    // Dimension must be 16 aligned
    const int alignedNum = 16;
    FAISS_THROW_IF_NOT_FMT(this->d % alignedNum == 0, "Number of diamension (%d) must be 16 aligned.", this->d);

    // Dimension must be > 0 and <=512
    FAISS_THROW_IF_NOT_FMT((this->d > 0) && (this->d <= DIM_MAX), "Number of diamension (%d) must be > 0 and <= 512",
        this->d);
}

void AscendIndexIVFFlat::initRpcCtx()
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

        ret = RpcCreateIndexIVFFlat(ctx, indexId, this->d, nlist, this->metric_type, nprobe, indexConfig.resourceSize);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create IndexIVFFlat failed(%d).", ret);
        indexMap[ctx] = indexId;
    }
}

void AscendIndexIVFFlat::clearRpcCtx()
{
    for (auto &index : indexMap) {
        RpcError ret = RpcDestroyIndexIVFFlat(index.first, index.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy IndexIVFFlat failed(%d).", ret);

        ret = RpcDestroyContext(index.first);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy context failed(%d).", ret);
    }

    indexMap.clear();
    contextMap.clear();
}

int AscendIndexIVFFlat::getElementSize() const
{
    // element size: sizeof(vector) + sizeof(ID) + sizeof(preCompute)
    return this->d * sizeof(uint16_t) + sizeof(uint32_t) + sizeof(uint16_t);
}

void AscendIndexIVFFlat::copyCodes(const faiss::IndexIVFFlat *index)
{
    const InvertedLists *ivf = index->invlists;
    size_t nlist = ivf ? ivf->nlist : 0;
    size_t deviceCnt = indexConfig.deviceList.size();

    // seperate codes to every device
    for (size_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);

        // Ascend index can only support max int entries per list
        FAISS_THROW_IF_NOT_FMT(listSize <= (size_t)std::numeric_limits<int>::max(),
            "Ascend inverted list can only support %zu entries; %zu found", (size_t)std::numeric_limits<int>::max(),
            listSize);

        int listNumPerDevice = listSize / deviceCnt;
        int listNumLast = listSize % deviceCnt;
        for (size_t j = 0; j < deviceCnt; j++) {
            deviceAddNumMap[i][j] += listNumPerDevice;
        }

        for (int j = 0; j < listNumLast; j++) {
            deviceAddNumMap[i][j] += 1;
        }
    }

// update basecode(vector) to device
#pragma omp parallel for
    for (size_t i = 0; i < nlist; ++i) {
        uint32_t offsum = 0;

        for (size_t j = 0; j < deviceCnt; j++) {
            int n = deviceAddNumMap[i][j];
            if (n == 0) {
                continue;
            }

            int deviceId = indexConfig.deviceList[j];
            rpcContext ctx = contextMap.at(deviceId);
            int indexId = indexMap.at(ctx);

            // convert the code to uint32_t
            float *offsetData = reinterpret_cast<float *>(const_cast<uint8_t *>(
                ivf->get_codes(i))) + offsum * this->d;
            idx_t *offsetIds = const_cast<idx_t *>(ivf->get_ids(i)) + offsum;

            // convert uint32_t to the uint16_t, because the data in device is fp16
            std::vector<uint16_t> data(n * this->d);
            transform(offsetData, offsetData + n * this->d, data.begin(), 
                [](float temp) { return fp16(temp).data; });

            // indices stored in devices in uint32_t format, trandform idx_t to uint32
            std::vector<uint32_t> indice(offsetIds, offsetIds + n);
            transform(offsetIds, offsetIds + n, begin(indice), [](idx_t x) { return uint32_t(x); });

            RpcError ret = RpcIndexIVFFlatAdd(ctx, indexId, n, i, data.data(), this->d, indice.data());
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add with ids failed(%d).", ret);
            offsum += n;
        }
    }
}
} // ascend
} // faiss
