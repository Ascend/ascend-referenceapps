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
#include <algorithm>

#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/utils/AscendThreadPool.h>
#include <faiss/ascend/utils/fp16.h>
#include <faiss/utils/distances.h>
#include <faiss/ascend/rpc/AscendRpc.h>

namespace faiss {
namespace ascend {
namespace {
const int DIM_ALIGN_SIZE = 16;
const int DIM_MAX = 512;

const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;

// Default size(64MB) for which we page add or search
const size_t ADD_PAGE_SIZE = 64U * KB * KB - RETAIN_SIZE;

// Or, maximum number(512K) of vectors to consider per page of add
const size_t ADD_VEC_SIZE = 512U * KB;

// get pagesize must be less than 32M, becauseof rpc limitation
const size_t PAGE_SIZE = 32U * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t VEC_SIZE = 512U * KB;
} // namespace

AscendIndexSQ::AscendIndexSQ(const faiss::IndexScalarQuantizer *index, AscendIndexSQConfig config)
    : AscendIndex(index->d, index->metric_type, config), sq(index->sq), sqConfig(config)
{
    copyFrom(index);
}

AscendIndexSQ::AscendIndexSQ(int dims, faiss::ScalarQuantizer::QuantizerType qType, faiss::MetricType metric,
    AscendIndexSQConfig config)
    : AscendIndex(dims, metric, config), sq(dims, qType), sqConfig(config)
{
    checkParams();
    clearRpcCtx();
    initRpcCtx();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(sqConfig.deviceList.size());

    this->is_trained = false;
}

AscendIndexSQ::~AscendIndexSQ()
{
    for (auto &index : indexMap) {
        RpcError ret = RpcDestroyIndexSQ(index.first, index.second);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);

        ret = RpcDestroyContext(index.first);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);
    }

    indexMap.clear();
    contextMap.clear();
}

void AscendIndexSQ::checkParams()
{
    // only support L2
    FAISS_THROW_IF_NOT_MSG(this->metric_type == MetricType::METRIC_L2 ||
        this->metric_type == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");

    // only support SQ8
    FAISS_THROW_IF_NOT_MSG(this->sq.qtype == faiss::ScalarQuantizer::QT_8bit ||
        sq.qtype == faiss::ScalarQuantizer::QT_8bit_uniform,
        "Unsupported qtype");

    // Diamension must be 16 aligned
    FAISS_THROW_IF_NOT_FMT(this->d % DIM_ALIGN_SIZE == 0, "Number of diamension (%d) must be 16 aligned.", this->d);
    // Dimension must be > 0 and <=512
    FAISS_THROW_IF_NOT_FMT((this->d > 0) && (this->d <= DIM_MAX), "Number of diamension (%d) must be > 0 and <= 512",
        this->d);
}

void AscendIndexSQ::copyFrom(const faiss::IndexScalarQuantizer *index)
{
    checkParams();

    initRpcCtx();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(sqConfig.deviceList.size());

    // The other index might not be trained
    if (!index->is_trained) {
        this->is_trained = false;
        this->ntotal = 0;
        return;
    }

    this->is_trained = true;

    // copy cpu index's codes and preCompute to ascend index
    copyCode(index);

    // copy train param to ascend index
    updateDeviceSQTrainedValue();
}

void AscendIndexSQ::copyCode(const faiss::IndexScalarQuantizer *index)
{
    if (index->codes.size() > 0) {
        // set ids
        std::vector<Index::idx_t> ids(index->ntotal);
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = this->ntotal + i;
        }

        copyByPage(index->ntotal, index->codes.data(), ids.data());
    }
}

void AscendIndexSQ::copyByPage(int n, const uint8_t *codes, const Index::idx_t *ids)
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

void AscendIndexSQ::copyImpl(int n, const uint8_t *codes, const Index::idx_t *ids)
{
    FAISS_ASSERT(n > 0);

    // 1. compute addMap
    size_t deviceCnt = sqConfig.deviceList.size();
    std::vector<int> addMap(deviceCnt, 0);
    calcAddMap(n, addMap);

    // 2. compute the preCompute values
    float *preComputeVals = new float[n];
    faiss::ScopeDeleter<float> del2(preComputeVals);
    calcPreCompute(codes, preComputeVals, n);

    // 3. transfer the codes and preCompute to the device
    add2Device(n, const_cast<uint8_t *>(codes), ids, preComputeVals, addMap);
}

void AscendIndexSQ::copyTo(faiss::IndexScalarQuantizer *index) const
{
    FAISS_THROW_IF_NOT(index != nullptr);
    index->metric_type = this->metric_type;
    index->sq = this->sq;
    index->code_size = this->sq.code_size;
    index->d = this->d;
    index->ntotal = this->ntotal;
    index->is_trained = this->is_trained;

    if (this->is_trained && this->ntotal > 0) {
        std::vector<uint8_t> baseData;
        for (auto &device : sqConfig.deviceList) {
            size_t size = getBaseSize(device);
            if (size == 0) {
                continue;
            }

            std::vector<uint8_t> codes;
            getPagedFast(device, size, codes);
            baseData.insert(baseData.end(), codes.begin(), codes.end());
        }
        index->codes = std::move(baseData);
    }
}

void AscendIndexSQ::getBase(int deviceId, std::vector<uint8_t> &codes) const
{
    size_t size = getBaseSize(deviceId);
    FAISS_THROW_IF_NOT_FMT(size > 0, "Get base size is (%d).", size);
    getPaged(deviceId, size, codes);
}

size_t AscendIndexSQ::getBaseSize(int deviceId) const
{
    uint32_t size = 0;
    rpcContext ctx = contextMap.at(deviceId);
    int indexId = indexMap.at(ctx);
    RpcError ret = RpcIndexSQGetBaseSize(ctx, indexId, size);
    FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Get base size failed(%d).", ret);
    return (size_t)size;
}

void AscendIndexSQ::getIdxMap(int deviceId, std::vector<Index::idx_t> &idxMap) const
{
    int deviceNum = sqConfig.deviceList.size();
    for (int i = 0; i < deviceNum; i++) {
        if (deviceId == sqConfig.deviceList[i]) {
            idxMap = idxDeviceMap.at(i);
            break;
        }
    }
}

void AscendIndexSQ::reset()
{
    for (auto &data : indexMap) {
        RpcError ret = RpcIndexReset(data.first, data.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Reset IndexSQ failed(%d).", ret);
    }

    int deviceNum = sqConfig.deviceList.size();
    idxDeviceMap.clear();
    idxDeviceMap.resize(deviceNum);

    this->ntotal = 0;
}

void AscendIndexSQ::train(Index::idx_t n, const float *x)
{
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n should be a positive value.");

    if (this->is_trained) {
        FAISS_THROW_IF_NOT_MSG(indexMap.size() > 0, "indexMap.size must be >0");
        return;
    }

    // use the ScalarQuantizer to train data
    sq.train(n, x);

    // update the SQ param to device
    updateDeviceSQTrainedValue();

    this->is_trained = true;
}

void AscendIndexSQ::initRpcCtx()
{
    indexMap.clear();
    contextMap.clear();
    for (size_t i = 0; i < sqConfig.deviceList.size(); i++) {
        int indexId;
        rpcContext ctx;
        int deviceId = sqConfig.deviceList[i];

        RpcError ret = RpcCreateContext(deviceId, &ctx);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create context failed(%d).", ret);
        contextMap[deviceId] = ctx;

        IndexSQParameter sqParameter(this->d, sq.qtype, this->metric_type, sqConfig.resourceSize);
        ret = RpcCreateIndexSQ(ctx, indexId, sqParameter);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create IndexSQ failed(%d).", ret);
        indexMap[ctx] = indexId;
    }
}

void AscendIndexSQ::clearRpcCtx()
{
    for (auto &index : indexMap) {
        RpcError ret = RpcDestroyIndexSQ(index.first, index.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy IndexSQ failed(%d).", ret);

        ret = RpcDestroyContext(index.first);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy context failed(%d).", ret);
    }

    indexMap.clear();
    contextMap.clear();
}

void AscendIndexSQ::updateDeviceSQTrainedValue()
{
    // convert trained value to fp16, contain vmin and vdiff, so need *2
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
        RpcError ret = RpcIndexSQUpdateTrainedValue(index.first, index.second, this->d, vmin, vdiff, false);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Update trained value failed(%d).", ret);
    }
}

void AscendIndexSQ::getPaged(int deviceId, int n, std::vector<uint8_t> &codes) const
{
    if (n > 0) {
        size_t totalSize = (size_t)n * getElementSize();
        size_t offsetNum = 0;
        if (totalSize > PAGE_SIZE || (size_t)n > VEC_SIZE) {
            size_t maxNumVecsForPageSize = PAGE_SIZE / ((size_t)getElementSize());

            // Always add at least 1 vector, if we have huge vectors
            maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t)1);

            size_t tileSize = std::min((size_t)n, maxNumVecsForPageSize);

            for (int i = 0; i < n; i += tileSize) {
                size_t curNum = std::min(tileSize, size_t(n) - i);

                getImpl(deviceId, offsetNum, curNum, codes.data() + offsetNum * (size_t)this->sq.code_size);
                offsetNum += curNum;
            }
        } else {
            getImpl(deviceId, offsetNum, n, codes.data());
        }
    }
}

void AscendIndexSQ::getPagedFast(int deviceId, size_t n, std::vector<uint8_t>& codes) const
{
    if (n == 0) {
        return;
    }

    rpcContext ctx = contextMap.at(deviceId);
    int indexId = indexMap.at(ctx);
    size_t elementSize = this->sq.code_size * sizeof(uint8_t);
    RpcError ret = RpcIndexSQFastGetBase(ctx, indexId, elementSize, n, codes);
    FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Get vectors of IndexSQ failed(%d).", ret);
}

void AscendIndexSQ::getImpl(int deviceId, int offset, int n, uint8_t *code) const
{
    rpcContext ctx = contextMap.at(deviceId);
    int indexId = indexMap.at(ctx);
    std::vector<uint8_t> baseData(n);
    RpcError ret = RpcIndexSQGetBase(ctx, indexId, offset, n, baseData);
    FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Get vectors of IndexSQ failed(%d).", ret);
    transform(baseData.begin(), baseData.end(), code, [](uint8_t temp) { return temp; });
}

bool AscendIndexSQ::addImplRequiresIDs() const
{
    return true;
}

void AscendIndexSQ::calcAddMap(int n, std::vector<int> &addMap)
{
    int devIdx = 0;
    size_t deviceCnt = sqConfig.deviceList.size();
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

void AscendIndexSQ::add2Device(int n, uint8_t *codes, const Index::idx_t *ids, float *preCompute,
    std::vector<int> &addMap)
{
    size_t deviceCnt = sqConfig.deviceList.size();
    uint32_t offsum = 0;
    for (size_t i = 0; i < deviceCnt; i++) {
        int num = addMap.at(i);
        if (num == 0) {
            continue;
        }
        int deviceId = sqConfig.deviceList[i];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);

        std::vector<uint8_t> codeb(num * this->sq.code_size);
        codeb.assign(codes + offsum * this->sq.code_size, codes + (offsum + num) * this->sq.code_size);
        float *preComputeToAdd = preCompute + offsum;
        RpcError ret = RpcIndexSQAdd(ctx, indexId, num, this->sq.code_size, codeb.data(), preComputeToAdd);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add with ids failed(%d).", ret);

        // record ids of new adding vector
        idxDeviceMap[i].insert(idxDeviceMap[i].end(), ids + offsum, ids + num + offsum);
        offsum += num;
    }
    this->ntotal += n;
}

void AscendIndexSQ::addImpl(int n, const float *x, const Index::idx_t *ids)
{
    FAISS_ASSERT(n > 0);

    // 1. compute addMap
    size_t deviceCnt = sqConfig.deviceList.size();
    std::vector<int> addMap(deviceCnt, 0);
    calcAddMap(n, addMap);

    // 2. compute the sq codes
    uint8_t *codes = new uint8_t[n * this->sq.code_size];
    faiss::ScopeDeleter<uint8_t> del1(codes);
    sq.compute_codes(x, codes, n);

    // 3. compute the preCompute values
    float *preComputeVals = new float[n];
    faiss::ScopeDeleter<float> del2(preComputeVals);
    calcPreCompute(codes, preComputeVals, n);

    // 4. transfer the codes and preCompute to the device
    add2Device(n, codes, ids, preComputeVals, addMap);
}

void AscendIndexSQ::calcPreCompute(const uint8_t *codes, float *compute, size_t n, float *xMem)
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

void AscendIndexSQ::searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
    std::vector<std::vector<uint32_t>> &label, int n, int k, float *distances, Index::idx_t *labels) const
{
    // transmit idx per device to referenced value
    size_t deviceCnt = this->sqConfig.deviceList.size();
    std::vector<std::vector<uint32_t>> transLabel(deviceCnt, std::vector<uint32_t>(n * k, 0));
    for (size_t i = 0; i < deviceCnt; i++) {
        transform(begin(label[i]), end(label[i]), begin(transLabel[i]), [&](uint32_t temp) {
            return (temp != std::numeric_limits<uint32_t>::max()) ? idxDeviceMap[i].at(temp) :
                                                                    std::numeric_limits<uint32_t>::max();
        });
    }

    mergeSearchResult(devices, dist, transLabel, n, k, distances, labels);
}

void AscendIndexSQ::removeSingle(std::vector<std::vector<uint32_t>> &removes, int deviceNum, uint32_t idx)
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

int AscendIndexSQ::getElementSize() const
{
    // element size: codesize + sizeof(preCompute)
    return this->sq.code_size * sizeof(uint8_t) + sizeof(float);
}

size_t AscendIndexSQ::removeImpl(const IDSelector &sel)
{
    int deviceCnt = sqConfig.deviceList.size();
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

    // 2. remove the index save in the host
    removeIdx(removeMaps);

    // 3. remove the vector in the device
#pragma omp parallel for reduction(+ : removeCnt)
    for (int i = 0; i < deviceCnt; i++) {
        if (removeMaps[i].size() == 0) {
            continue;
        }
        uint32_t removeCnt_tmp = 0;
        int deviceId = sqConfig.deviceList[i];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        RpcError ret = RpcIndexRemoveIds(ctx, indexId, removeMaps[i].size(), removeMaps[i].data(), &removeCnt_tmp);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Remove ID selector failed(%d).", ret);
        removeCnt += removeCnt_tmp;
    }

    this->ntotal -= removeCnt;
    return (size_t)removeCnt;
}

void AscendIndexSQ::removeIdx(std::vector<std::vector<uint32_t>> &removeMaps)
{
    int deviceCnt = sqConfig.deviceList.size();
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