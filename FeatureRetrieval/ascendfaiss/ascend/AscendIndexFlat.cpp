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

#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/ascend/utils/AscendThreadPool.h>
#include <faiss/ascend/utils/fp16.h>
#include <faiss/ascend/rpc/AscendRpc.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <algorithm>
#include <omp.h>

namespace faiss {
namespace ascend {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
// get pagesize must be less than 32M, becauseof rpc limitation
const size_t PAGE_SIZE = 32U * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t VEC_SIZE = 512U * KB;
} // namespace

// implementation of AscendIndexFlat
AscendIndexFlat::AscendIndexFlat(const faiss::IndexFlat *index, AscendIndexFlatConfig config)
    : AscendIndex(index->d, index->metric_type, config), flatConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(index->metric_type == MetricType::METRIC_L2 ||
        index->metric_type == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");

    // Flat index doesn't need training
    this->is_trained = true;

    copyFrom(index);
}

AscendIndexFlat::AscendIndexFlat(int dims, faiss::MetricType metric, AscendIndexFlatConfig config)
    : AscendIndex(dims, metric, config), flatConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(metric == MetricType::METRIC_L2 || metric == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");

    // Flat index doesn't need training
    this->is_trained = true;

    initRpcCtx();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(config.deviceList.size());
}

AscendIndexFlat::~AscendIndexFlat()
{
    for (auto &index : indexMap) {
        RpcError ret = RpcDestroyIndexFlat(index.first, index.second);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);

        ret = RpcDestroyContext(index.first);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);
    }

    indexMap.clear();
    contextMap.clear();
}

void AscendIndexFlat::copyFrom(const faiss::IndexFlat *index)
{
    FAISS_THROW_IF_NOT_MSG(index->metric_type == MetricType::METRIC_L2 ||
        index->metric_type == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");

    this->d = index->d;
    this->metric_type = index->metric_type;

    initRpcCtx();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(flatConfig.deviceList.size());

    if (index->xb.size() > 0) {
        add(index->ntotal, index->xb.data());
    }
}


void AscendIndexFlat::copyTo(faiss::IndexFlat *index) const
{
    FAISS_THROW_IF_NOT(index != nullptr);
    index->d = this->d;
    index->ntotal = this->ntotal;
    index->metric_type = this->metric_type;

    if (this->is_trained && this->ntotal > 0) {
        std::vector<float> baseData;
        for (auto &device : flatConfig.deviceList) {
            size_t size = getBaseSize(device);
            std::vector<float> xb(size * this->d);
            getBase(device, xb);
            baseData.insert(baseData.end(), xb.begin(), xb.end());
        }
        index->xb = std::move(baseData);
    }
}

void AscendIndexFlat::initRpcCtx()
{
    indexMap.clear();
    contextMap.clear();
    for (size_t i = 0; i < flatConfig.deviceList.size(); i++) {
        int indexId;
        rpcContext ctx;
        int deviceId = flatConfig.deviceList[i];

        RpcError ret = RpcCreateContext(deviceId, &ctx);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create context failed(%d).", ret);
        contextMap[deviceId] = ctx;

        ret = RpcCreateIndexFlat(ctx, indexId, this->d, this->metric_type, flatConfig.resourceSize);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create IndexFlat failed(%d).", ret);
        indexMap[ctx] = indexId;
    }
}

void AscendIndexFlat::clearRpcCtx()
{
    for (auto &index : indexMap) {
        RpcError ret = RpcDestroyIndexFlat(index.first, index.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy IndexFlat failed(%d).", ret);

        ret = RpcDestroyContext(index.first);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy context failed(%d).", ret);
    }

    indexMap.clear();
    contextMap.clear();
}

size_t AscendIndexFlat::getBaseSize(int deviceId) const
{
    uint32_t size = 0;
    rpcContext ctx = contextMap.at(deviceId);
    int indexId = indexMap.at(ctx);
    RpcError ret = RpcIndexFlatGetBaseSize(ctx, indexId, size);
    FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Get base size failed(%d).", ret);
    return (size_t)size;
}

void AscendIndexFlat::getBase(int deviceId, std::vector<float> &xb) const
{
    size_t size = getBaseSize(deviceId);
    FAISS_THROW_IF_NOT_FMT(size > 0, "Get base size is (%d).", size);
    getPaged(deviceId, size, xb);
}

void AscendIndexFlat::getIdxMap(int deviceId, std::vector<Index::idx_t> &idxMap) const
{
    int deviceNum = flatConfig.deviceList.size();
    for (int i = 0; i < deviceNum; i++) {
        if (deviceId == flatConfig.deviceList[i]) {
            idxMap = idxDeviceMap.at(i);
            break;
        }
    }
}

void AscendIndexFlat::getPaged(int deviceId, int n, std::vector<float> &xb) const
{
    if (n > 0) {
        size_t totalSize = (size_t)n * this->d * sizeof(float);
        size_t offsetNum = 0;
        if (totalSize > PAGE_SIZE || (size_t)n > VEC_SIZE) {
            size_t maxNumVecsForPageSize = PAGE_SIZE / ((size_t)this->d * sizeof(float));

            // Always add at least 1 vector, if we have huge vectors
            maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t)1);

            size_t tileSize = std::min((size_t)n, maxNumVecsForPageSize);

            for (size_t i = 0; i < (size_t)n; i += tileSize) {
                size_t curNum = std::min(tileSize, n - i);

                getImpl(deviceId, offsetNum, curNum, xb.data() + offsetNum * (size_t)this->d);
                offsetNum += curNum;
            }
        } else {
            getImpl(deviceId, offsetNum, n, xb.data());
        }
    }
}

void AscendIndexFlat::getImpl(int deviceId, int offset, int n, float *x) const
{
    rpcContext ctx = contextMap.at(deviceId);
    int indexId = indexMap.at(ctx);
    std::vector<uint16_t> baseData(n);
    RpcError ret = RpcIndexFlatGetBase(ctx, indexId, offset, n, baseData);
    FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Get vectors of IndexFlat failed(%d).", ret);
    transform(baseData.begin(), baseData.end(), x, [](uint16_t temp) { return static_cast<float>(fp16(temp)); });
}

void AscendIndexFlat::reset()
{
    for (auto &data : indexMap) {
        RpcError ret = RpcIndexReset(data.first, data.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Reset IndexFlat failed(%d).", ret);
    }

    int deviceNum = flatConfig.deviceList.size();
    idxDeviceMap.clear();
    idxDeviceMap.resize(deviceNum);

    this->ntotal = 0;
}

void AscendIndexFlat::train(Index::idx_t n, const float *x)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    // do nothing
}

bool AscendIndexFlat::addImplRequiresIDs() const
{
    return true;
}

void AscendIndexFlat::addImpl(int n, const float *x, const Index::idx_t *ids)
{
    FAISS_ASSERT(n > 0);
    int devIdx = 0;
    size_t deviceCnt = flatConfig.deviceList.size();
    std::vector<int> addMap(deviceCnt, 0);
    for (size_t i = 1; i < deviceCnt; i++) {
        if (idxDeviceMap[i].size() < idxDeviceMap[devIdx].size()) {
            devIdx = i;
            break;
        }
    }
    for (size_t i = 0; i < deviceCnt; i++) {
        addMap[i] += n / deviceCnt;
    }
    for (size_t i = 0; i < n % deviceCnt; i++) {
        addMap[devIdx % deviceCnt] += 1;
        devIdx += 1;
    }

    uint32_t offsum = 0;
    for (size_t i = 0; i < deviceCnt; i++) {
        int num = addMap.at(i);
        if (num == 0) {
            continue;
        }
        int deviceId = flatConfig.deviceList[i];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);

        std::vector<uint16_t> xb(num * this->d);
        transform(x + offsum * this->d, x + (offsum + num) * this->d, std::begin(xb),
            [](float temp) { return fp16(temp).data; });
        RpcError ret = RpcIndexFlatAdd(ctx, indexId, num, this->d, xb.data());
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add with ids failed(%d).", ret);

        // record ids of new adding vector
        idxDeviceMap[i].insert(idxDeviceMap[i].end(), ids + offsum, ids + num + offsum);
        offsum += num;
    }
    this->ntotal += n;
}

void AscendIndexFlat::removeSingle(std::vector<std::vector<uint32_t>> &removes, int deviceNum, uint32_t idx)
{
    for (int i = 0; i < deviceNum; i++) {
        auto it = find(idxDeviceMap[i].begin(), idxDeviceMap[i].end(), idx);
        if (it != idxDeviceMap[i].end()) {
            int pos = distance(idxDeviceMap[i].begin(), it);
            removes[i].push_back(pos);
            break;
        }
    }
}

void AscendIndexFlat::searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
    std::vector<std::vector<uint32_t>> &label, int n, int k, float *distances, Index::idx_t *labels) const
{
    // transmit idx per device to referenced value
    size_t deviceCnt = this->flatConfig.deviceList.size();
    std::vector<std::vector<uint32_t>> transLabel(deviceCnt, std::vector<uint32_t>(n * k, 0));
    for (size_t i = 0; i < deviceCnt; i++) {
        transform(begin(label[i]), end(label[i]), begin(transLabel[i]), [&](uint32_t temp) {
            return (temp != std::numeric_limits<uint32_t>::max()) ? idxDeviceMap[i].at(temp) :
                                                                    std::numeric_limits<uint32_t>::max();
        });
    }

    mergeSearchResult(devices, dist, transLabel, n, k, distances, labels);
}

int AscendIndexFlat::getElementSize() const
{
    return this->d * sizeof(uint16_t);
}

size_t AscendIndexFlat::removeImpl(const IDSelector &sel)
{
    int deviceCnt = flatConfig.deviceList.size();
    uint32_t removeCnt = 0;

    // 1. remove vector from device, and removeMaps save the id(not index) of pre-removing
    std::vector<std::vector<uint32_t>> removeMaps(deviceCnt, std::vector<uint32_t>());
    if (auto rangeSel = dynamic_cast<const IDSelectorBatch *>(&sel)) {
        std::vector<uint32_t> removeBatch(rangeSel->set.size());
        transform(begin(rangeSel->set), end(rangeSel->set), begin(removeBatch),
            [](idx_t temp) { return (uint32_t)temp; });

        for (auto idx : removeBatch) {
            removeSingle(removeMaps, deviceCnt, idx);
        }
    } else if (auto rangeSel = dynamic_cast<const IDSelectorRange *>(&sel)) {
        for (auto idx = rangeSel->imin; idx < rangeSel->imax; ++idx) {
            removeSingle(removeMaps, deviceCnt, idx);
        }
    }

    // 2. remove the vector in the device
#pragma omp parallel for reduction(+ : removeCnt)
    for (int i = 0; i < deviceCnt; i++) {
        if (removeMaps[i].size() == 0) {
            continue;
        }
        uint32_t removeCnt_tmp = 0;
        int deviceId = flatConfig.deviceList[i];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        RpcError ret = RpcIndexRemoveIds(ctx, indexId, removeMaps[i].size(), removeMaps[i].data(), &removeCnt_tmp);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Remove ID selector failed(%d).", ret);
        removeCnt += removeCnt_tmp;
    }

    // 3. remove the index save in the host
    removeIdx(removeMaps);

    this->ntotal -= removeCnt;
    return (size_t)removeCnt;
}

void AscendIndexFlat::removeIdx(std::vector<std::vector<uint32_t>> &removeMaps)
{
    int deviceCnt = flatConfig.deviceList.size();
#pragma omp parallel for if (deviceCnt > 1) num_threads(deviceCnt)
    for (int i = 0; i < deviceCnt; ++i) {
        // 1. sort by DESC, then delete from the big to small
        std::sort(removeMaps[i].rbegin(), removeMaps[i].rend());

        for (auto pos : removeMaps[i]) {
            int lastIdx = this->idxDeviceMap[i].size() - 1;
            this->idxDeviceMap[i][pos] = this->idxDeviceMap[i][lastIdx];
            this->idxDeviceMap[i].pop_back();
        }
    }
}

// implementation of AscendIndexFlatL2
AscendIndexFlatL2::AscendIndexFlatL2(faiss::IndexFlatL2 *index, AscendIndexFlatConfig config)
    : AscendIndexFlat(index, config)
{}

AscendIndexFlatL2::AscendIndexFlatL2(int dims, AscendIndexFlatConfig config)
    : AscendIndexFlat(dims, faiss::METRIC_L2, config)
{}

void AscendIndexFlatL2::copyFrom(faiss::IndexFlat *index)
{
    FAISS_THROW_IF_NOT_MSG(index->metric_type == metric_type,
        "Cannot copy a AscendIndexFlatL2 from an index of different metric_type");
    AscendIndexFlat::copyFrom(index);
}

void AscendIndexFlatL2::copyTo(faiss::IndexFlat *index)
{
    FAISS_THROW_IF_NOT_MSG(index->metric_type == metric_type,
        "Cannot copy a AscendIndexFlatL2 to an index of different metric_type");
    AscendIndexFlat::copyTo(index);
}
} // ascend
} // faiss