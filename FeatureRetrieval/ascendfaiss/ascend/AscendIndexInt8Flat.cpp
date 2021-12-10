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

#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <faiss/ascend/utils/AscendThreadPool.h>
#include <faiss/ascend/rpc/AscendRpc.h>
#include <faiss/ascend/utils/AscendUtils.h>
#include <faiss/ascend/rpc/AscendRpcIndexInt8Flat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <algorithm>
#include <securec.h>
#include <omp.h>

namespace faiss {
namespace ascend {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
// get pagesize must be less than 32M, because of rpc limitation
const size_t PAGE_SIZE = 32U * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t VEC_SIZE = 512U * KB;

// Default size(64MB) for which we page add or search
const size_t ADD_PAGE_SIZE = 64U * KB * KB - RETAIN_SIZE;

// Or, maximum number(512K) of vectors to consider per page of add
const size_t ADD_VEC_SIZE = 512U * KB;

const size_t UNIT_PAGE_SIZE = 64;
const size_t UNIT_VEC_SIZE = 512;
} // namespace

// implementation of AscendIndexFlat
AscendIndexInt8Flat::AscendIndexInt8Flat(int dims, faiss::MetricType metric, AscendIndexInt8FlatConfig config)
    : AscendIndexInt8(dims, metric, config), int8FlatConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(metric == MetricType::METRIC_L2 || metric == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");
    FAISS_THROW_IF_NOT_MSG(dims > 0, "Invalid number of dimensions");

    pool = new AscendThreadPool(int8FlatConfig.deviceList.size());

    // Flat index doesn't need training
    this->is_trained = true;

    initRpcCtx();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(config.deviceList.size());
}

AscendIndexInt8Flat::AscendIndexInt8Flat(const faiss::IndexScalarQuantizer *index,
    AscendIndexInt8FlatConfig config)
    : AscendIndexInt8(index->d, index->metric_type, config), int8FlatConfig(config)
{
    copyFrom(index);
}

AscendIndexInt8Flat::~AscendIndexInt8Flat()
{
    for (auto &index : indexMap) {
        RpcError ret = RpcDestroyIndexInt8(index.first, index.second);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);

        ret = RpcDestroyContext(index.first);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);
    }

    indexMap.clear();
    contextMap.clear();

    if (pool != nullptr) {
        delete pool;
        pool = nullptr;
    }
}

void AscendIndexInt8Flat::initRpcCtx()
{
    indexMap.clear();
    contextMap.clear();
    for (size_t i = 0; i < int8FlatConfig.deviceList.size(); i++) {
        int indexId;
        rpcContext ctx;
        int deviceId = int8FlatConfig.deviceList[i];

        RpcError ret = RpcCreateContext(deviceId, &ctx);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create context failed(%d).", ret);
        contextMap[deviceId] = ctx;

        ret = RpcCreateIndexInt8Flat(ctx, indexId, this->d, this->metric_type, int8FlatConfig.resourceSize);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create IndexFlat failed(%d).", ret);
        indexMap[ctx] = indexId;
    }
}

void AscendIndexInt8Flat::clearRpcCtx()
{
    for (auto &index : indexMap) {
        RpcError ret = RpcDestroyIndexInt8(index.first, index.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy IndexFlat failed(%d).", ret);

        ret = RpcDestroyContext(index.first);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy context failed(%d).", ret);
    }

    indexMap.clear();
    contextMap.clear();
}

size_t AscendIndexInt8Flat::getBaseSize(int deviceId) const
{
    uint32_t size = 0;
    rpcContext ctx = contextMap.at(deviceId);
    int indexId = indexMap.at(ctx);
    RpcError ret = RpcIndexInt8FlatGetBaseSize(ctx, indexId, size, metric_type);
    FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Get base size failed(%d).", ret);
    return (size_t)size;
}

void AscendIndexInt8Flat::getBase(int deviceId, std::vector<int8_t> &xb) const
{
    size_t size = getBaseSize(deviceId);
    FAISS_THROW_IF_NOT_FMT(size > 0, "Get base size is (%d).", size);
    getPaged(deviceId, size, xb);
}

void AscendIndexInt8Flat::getIdxMap(int deviceId, std::vector<Index::idx_t> &idxMap) const
{
    int deviceNum = int8FlatConfig.deviceList.size();
    for (int i = 0; i < deviceNum; i++) {
        if (deviceId == int8FlatConfig.deviceList[i]) {
            idxMap = idxDeviceMap.at(i);
            break;
        }
    }
}

void AscendIndexInt8Flat::getPaged(int deviceId, int n, std::vector<int8_t> &xb) const
{
    if (n > 0) {
        size_t totalSize = (size_t)n * this->d * sizeof(int8_t);
        size_t offsetNum = 0;
        if (totalSize > PAGE_SIZE || (size_t)n > VEC_SIZE) {
            size_t maxNumVecsForPageSize = PAGE_SIZE / ((size_t)this->d * sizeof(int8_t));

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

void AscendIndexInt8Flat::getImpl(int deviceId, int offset, int n, int8_t *x) const
{
    rpcContext ctx = contextMap.at(deviceId);
    int indexId = indexMap.at(ctx);

    std::vector<int8_t> baseData(n);
    RpcError ret = RpcIndexInt8FlatGetBase(ctx, indexId, offset, n, baseData, metric_type);
    FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Get vectors of IndexInt8Flat failed(%d).", ret);
    (void)memcpy_s(x, n * this->d * sizeof(int8_t), baseData.data(), n * this->d * sizeof(int8_t));
}

void AscendIndexInt8Flat::reset()
{
    for (auto &data : indexMap) {
        RpcError ret = RpcIndexInt8Reset(data.first, data.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Reset IndexFlat failed(%d).", ret);
    }

    int deviceNum = int8FlatConfig.deviceList.size();
    idxDeviceMap.clear();
    idxDeviceMap.resize(deviceNum);

    this->ntotal = 0;
}

void AscendIndexInt8Flat::copyFrom(const faiss::IndexScalarQuantizer* index)
{
    initRpcCtx();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(int8FlatConfig.deviceList.size());

    // The other index might not be trained
    if (!index->is_trained) {
        this->is_trained = false;
        this->ntotal = 0;
        return;
    }

    this->is_trained = true;

    // copy cpu index's codes and preCompute to ascend index
    copyCode(index);
}

void AscendIndexInt8Flat::copyCode(const faiss::IndexScalarQuantizer *index)
{
    if (index->codes.size() > 0) {
        // set ids
        std::vector<Index::idx_t> ids(index->ntotal);
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = this->ntotal + i;
        }

        copyByPage(index->ntotal, reinterpret_cast<const int8_t*>(index->codes.data()), ids.data());
    }
}

void AscendIndexInt8Flat::copyByPage(int n, const int8_t *codes, const Index::idx_t *ids)
{
    if (n <= 0 || codes == nullptr) {
        return;
    }

    size_t totalSize = static_cast<size_t>(n) * static_cast<size_t>(getElementSize());
    if (totalSize > ADD_PAGE_SIZE || (size_t)n > ADD_VEC_SIZE) {
        // How many vectors fit into kAddPageSize?
        size_t maxNumVecsForPageSize = ADD_PAGE_SIZE / getElementSize();
        // Always add at least 1 vector, if we have huge vectors
        maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t)1);

        size_t tileSize = std::min((size_t)n, maxNumVecsForPageSize);

        for (int i = 0; i < n; i += tileSize) {
            size_t curNum = std::min(tileSize, (size_t)n - i);
            if (this->verbose) {
                printf("AscendIndexSQ::add: adding %d:%zu / %d\n", i, i + curNum, n);
            }
            copyImpl(curNum, codes + i * (size_t)this->d, (ids != nullptr) ? (ids + i) : nullptr);
        }
    } else {
        if (this->verbose) {
            printf("AscendIndexSQ::add: adding 0:%d / %d\n", n, n);
        }
        copyImpl(n, codes, ids);
    }
}

void AscendIndexInt8Flat::copyImpl(int n, const int8_t *codes, const Index::idx_t *ids)
{
    FAISS_ASSERT(n > 0);

    // 1. compute addMap
    size_t deviceCnt = int8FlatConfig.deviceList.size();
    std::vector<int> addMap(deviceCnt, 0);
    calcAddMap(n, addMap);

    // 2. transfer the codes to the device
    add2Device(n, const_cast<int8_t *>(codes), ids, addMap);
}

void AscendIndexInt8Flat::calcAddMap(int n, std::vector<int> &addMap)
{
    int devIdx = 0;
    size_t deviceCnt = int8FlatConfig.deviceList.size();
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
}

void AscendIndexInt8Flat::add2Device(int n, int8_t *codes, const Index::idx_t *ids, std::vector<int> &addMap)
{
    size_t deviceCnt = int8FlatConfig.deviceList.size();
    uint32_t offsum = 0;
    for (size_t i = 0; i < deviceCnt; i++) {
        int num = addMap.at(i);
        if (num == 0) {
            continue;
        }
        int deviceId = int8FlatConfig.deviceList[i];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);

        std::vector<int8_t> codeb(num * this->d);
        codeb.assign(codes + offsum * this->d, codes + (offsum + num) * this->d);
        RpcError ret = RpcIndexInt8FlatAdd(ctx, indexId, num, this->d, codeb.data());
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add with ids failed(%d).", ret);

        // record ids of new adding vector
        idxDeviceMap[i].insert(idxDeviceMap[i].end(), ids + offsum, ids + num + offsum);
        offsum += num;
    }
    this->ntotal += n;
}

void AscendIndexInt8Flat::copyTo(faiss::IndexScalarQuantizer* index) const
{
    FAISS_THROW_IF_NOT(index != nullptr);
    index->metric_type = this->metric_type;
    index->d = this->d;
    index->ntotal = this->ntotal;
    index->is_trained = this->is_trained;

    if (this->is_trained && this->ntotal > 0) {
        std::vector<uint8_t> baseData;
        for (auto &device : int8FlatConfig.deviceList) {
            size_t size = getBaseSize(device);
            if (size == 0) {
                continue;
            }

            std::vector<int8_t> codes(size * this->d);
            getBase(device, codes);
            uint8_t* codesUint8 = reinterpret_cast<uint8_t*>(codes.data());
            baseData.insert(baseData.end(), codesUint8, codesUint8 + size * this->d);
        }
        index->codes = std::move(baseData);
    }
}

bool AscendIndexInt8Flat::addImplRequiresIDs() const
{
    return true;
}

void AscendIndexInt8Flat::addImpl(int n, const int8_t *x, const Index::idx_t *ids)
{
    FAISS_ASSERT(n > 0);
    int devIdx = 0;
    size_t deviceCnt = int8FlatConfig.deviceList.size();
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
        int deviceId = int8FlatConfig.deviceList[i];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);

        RpcError ret = RpcIndexInt8FlatAdd(ctx, indexId, num, this->d, const_cast<int8_t *>(x + offsum * this->d));
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add with ids failed(%d).", ret);

        // record ids of new adding vector
        idxDeviceMap[i].insert(idxDeviceMap[i].end(), ids + offsum, ids + num + offsum);
        offsum += num;
    }
    this->ntotal += n;
}

void AscendIndexInt8Flat::removeSingle(std::vector<std::vector<uint32_t>> &removes, int deviceNum, uint32_t idx)
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

void AscendIndexInt8Flat::searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
    std::vector<std::vector<uint32_t>> &label, int n, int k, float *distances, Index::idx_t *labels) const
{
    // transmit idx per device to referenced value
    size_t deviceCnt = this->int8FlatConfig.deviceList.size();
    std::vector<std::vector<uint32_t>> transLabel(deviceCnt, std::vector<uint32_t>(n * k, 0));
    for (size_t i = 0; i < deviceCnt; i++) {
        transform(begin(label[i]), end(label[i]), begin(transLabel[i]), [&](uint32_t temp) {
            return (temp != std::numeric_limits<uint32_t>::max()) ? idxDeviceMap[i].at(temp) :
                                                                    std::numeric_limits<uint32_t>::max();
        });
    }

    mergeSearchResult(devices, dist, transLabel, n, k, distances, labels);
}

int AscendIndexInt8Flat::getElementSize() const
{
    return this->d * sizeof(int8_t);
}

size_t AscendIndexInt8Flat::removeImpl(const IDSelector &sel)
{
    int deviceCnt = int8FlatConfig.deviceList.size();
    uint32_t removeCnt = 0;

    // 1. remove vector from device, and removeMaps save the id(not index) of pre-removing
    std::vector<std::vector<uint32_t>> removeMaps(deviceCnt, std::vector<uint32_t>());
    if (auto rangeSel = dynamic_cast<const IDSelectorBatch *>(&sel)) {
        std::vector<uint32_t> removeBatch(rangeSel->set.size());
        transform(begin(rangeSel->set), end(rangeSel->set), begin(removeBatch),
            [](Index::idx_t temp) { return (uint32_t)temp; });

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
        int deviceId = int8FlatConfig.deviceList[i];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        RpcError ret =
            RpcIndexInt8RemoveIds(ctx, indexId, removeMaps[i].size(), removeMaps[i].data(), &removeCnt_tmp);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Remove ID selector failed(%d).", ret);
        removeCnt += removeCnt_tmp;
    }

    // 3. remove the index save in the host
    removeIdx(removeMaps);

    this->ntotal -= removeCnt;
    return (size_t)removeCnt;
}

void AscendIndexInt8Flat::removeIdx(std::vector<std::vector<uint32_t>> &removeMaps)
{
    int deviceCnt = int8FlatConfig.deviceList.size();
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
} // ascend
} // faiss