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

#include <faiss/IndexIVFPQ.h>
#include <faiss/ascend/utils/AscendThreadPool.h>
#include <faiss/ascend/utils/fp16.h>
#include <faiss/ascend/AscendIndexIVFPQ.h>
#include <faiss/ascend/rpc/AscendRpc.h>
#include <algorithm>
#include <omp.h>

namespace faiss {
namespace ascend {
    namespace {
        const int BITS_CNT_PER_CODE = 8;

        enum PQSize {
            PQ_SIZE_2 = 2,
            PQ_SIZE_4 = 4,
            PQ_SIZE_8 = 8,
            PQ_SIZE_12 = 12,
            PQ_SIZE_16 = 16,
            PQ_SIZE_20 = 20,
            PQ_SIZE_24 = 24,
            PQ_SIZE_32 = 32,
            PQ_SIZE_48 = 48,
            PQ_SIZE_64 = 64,
            PQ_SIZE_96 = 96,
            PQ_SIZE_128 = 128
        };

        enum PQSubDim {
            PQ_SUB_DIM_4 = 4,
            PQ_SUB_DIM_8 = 8,
            PQ_SUB_DIM_16 = 16,
            PQ_SUB_DIM_32 = 32,
            PQ_SUB_DIM_48 = 48,
            PQ_SUB_DIM_64 = 64,
            PQ_SUB_DIM_80 = 80,
            PQ_SUB_DIM_96 = 96,
            PQ_SUB_DIM_112 = 112,
            PQ_SUB_DIM_128 = 128
        };
    };

struct IVFPQAddInfo {
    IVFPQAddInfo(int idx, int deviceNum, size_t size)
        : addDeviceIdx(idx),
          deviceAddNum(deviceNum, 0),
          codeSize(size)
    {
        FAISS_ASSERT(idx < deviceNum);
    }

    void Add(uint8_t* code, const Index::idx_t* id)
    {
        deviceAddNum[addDeviceIdx] += 1;
        addDeviceIdx = (addDeviceIdx + 1) % deviceAddNum.size();
        codes.insert(codes.end(), code, code + codeSize);
        ids.emplace_back((uint32_t)(*id));
    }

    int GetOffSet(int idx) const
    {
        int ret = 0;
        FAISS_THROW_IF_NOT_MSG(idx < (int)deviceAddNum.size(), "idx error");
        for (int i = 0; i < idx; i++) {
            ret += deviceAddNum[i];
        }

        return ret;
    }

    void GetCodeAndIdPtr(int idx, uint8_t** codePtr, uint32_t** idPtr)
    {
        FAISS_THROW_IF_NOT_MSG(idx < (int)deviceAddNum.size(), "idx error");
        int off = GetOffSet(idx);
        *codePtr = codes.data() + off * codeSize;
        *idPtr = ids.data() + off;
        return;
    }

    int GetAddNum(int idx) const
    {
        FAISS_THROW_IF_NOT_MSG(idx < (int)deviceAddNum.size(), "idx error");
        return deviceAddNum[idx];
    }

    int addDeviceIdx;
    std::vector<int> deviceAddNum;
    size_t codeSize;
    std::vector<uint8_t> codes;
    std::vector<uint32_t> ids;
};

AscendIndexIVFPQ::AscendIndexIVFPQ(const faiss::IndexIVFPQ* index, AscendIndexIVFPQConfig config)
    : AscendIndexIVF(index->d, index->metric_type, index->nlist, config), ivfpqConfig(config), 
      subQuantizersCnt(0),
      bitsCntPerCode(0)
{
    FAISS_THROW_IF_NOT_MSG(config.deviceList.size(),
        "device list can not be empty");
    copyFrom(index);
}

AscendIndexIVFPQ::AscendIndexIVFPQ(int dims, int nlist, int subQuantizers, 
    int bitsPerCode, faiss::MetricType metric,
    AscendIndexIVFPQConfig config)
    : AscendIndexIVF(dims, metric, nlist, config),
      ivfpqConfig(config),
      subQuantizersCnt(subQuantizers),
      bitsCntPerCode(bitsPerCode)
{
    FAISS_THROW_IF_NOT_MSG(config.deviceList.size(),
        "device list can not be empty");

    checkParams();

    initRpcCtx();

    initDeviceAddNumMap();

    this->is_trained = false;
}

AscendIndexIVFPQ::~AscendIndexIVFPQ()
{
    for (auto& index : indexMap) {
        RpcError ret = RpcDestroyIndexIVFPQ(index.first, index.second);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);

        ret = RpcDestroyContext(index.first);
        FAISS_ASSERT(ret == RPC_ERROR_NONE);
    }

    indexMap.clear();
    contextMap.clear();
}

void AscendIndexIVFPQ::copyFrom(const faiss::IndexIVFPQ* index)
{
    AscendIndexIVF::copyFrom(index);
    clearRpcCtx();

    subQuantizersCnt = index->pq.M;
    bitsCntPerCode = index->pq.nbits;

    // We only support this
    const int nbits = 8;
    FAISS_THROW_IF_NOT_MSG(index->pq.nbits == nbits,
                           "Ascend: only pq.nbits == 8 is supported");
    FAISS_THROW_IF_NOT_MSG(index->by_residual,
                           "Ascend: only by_residual = true is supported");
    FAISS_THROW_IF_NOT_MSG(index->polysemous_ht == 0,
                           "Ascend: polysemous codes not supported");

    checkParams();
    initRpcCtx();
    
    // initial deviceAddNumMap mem space
    initDeviceAddNumMap();

    // The other index might not be trained
    if (!index->is_trained) {
        return;
    }

    this->is_trained = true;

    // Copy our lists as well. The product quantizer must have data in it
    FAISS_THROW_IF_NOT_MSG(index->pq.centroids.size() > 0, 
                           "Ascend: index is trained but centriods is empty.");

    pqData = index->pq;
    updateDeviceCoarseCenter();
    updateDevicePQCenter();

    // ͳ��ÿ��device��Ӻ����������
    const InvertedLists* ivf = index->invlists;
    size_t nlist = ivf ? ivf->nlist : 0;
    size_t deviceCnt = ivfpqConfig.deviceList.size();
    
    for (size_t i = 0; i < nlist; ++i) {
        size_t list_size = ivf->list_size(i);

        // Ascend index can only support max int entries per list
        FAISS_THROW_IF_NOT_FMT(list_size <=
                               (size_t)std::numeric_limits<int>::max(),
                               "Ascend inverted list can only support "
                               "%zu entries; %zu found",
                               (size_t)std::numeric_limits<int>::max(),
                               list_size);

        int listNumPerDevice = list_size / deviceCnt;
        int listNumLast = list_size % deviceCnt;
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
        for (size_t j = 0; j < deviceCnt; j++) {
            int n = deviceAddNumMap[i][j];  // listId������ÿ��device��������
            if (n == 0) {
                continue;
            }
            int deviceId = ivfpqConfig.deviceList[j];
            rpcContext ctx = contextMap.at(deviceId);
            int indexId = indexMap.at(ctx);
            uint8_t *offsetCodes = const_cast<uint8_t *>(ivf->get_codes(i)) + offsum * ivf->code_size;
            idx_t *offsetIds = const_cast<idx_t *>(ivf->get_ids(i)) + offsum;
            std::vector<uint32_t> indice(offsetIds, offsetIds + n);
            transform(offsetIds, offsetIds + n,
                      begin(indice), [](idx_t x) { return uint32_t(x); });

            RpcError ret = RpcIndexIVFPQAdd(ctx, indexId, n, i, offsetCodes,
                                            ivf->code_size, indice.data());
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add with ids failed(%d).", ret);
            offsum += n;
        }
    }
}
void AscendIndexIVFPQ::copyTo(faiss::IndexIVFPQ* index) const
{
    AscendIndexIVF::copyTo(index);
    // 
    // IndexIVFPQ information
    // 
    index->by_residual = true;
    index->use_precomputed_table = 0;
    index->code_size = subQuantizersCnt;
    if (this->is_trained) {
        index->pq = pqData;
    } else {
        index->pq = faiss::ProductQuantizer(this->d, subQuantizersCnt, bitsCntPerCode);
    }

    index->do_polysemous_training = false;
    index->polysemous_training = nullptr;

    index->scan_table_threshold = 0;
    index->max_codes = 0;
    index->polysemous_ht = 0;
    index->precomputed_table.clear();

    InvertedLists* ivf = new ArrayInvertedLists(
        nlist, index->code_size);

    index->replace_invlists(ivf, true);

    if (this->is_trained && this->ntotal > 0) {
        size_t deviceCnt = ivfpqConfig.deviceList.size();
        for (size_t i = 0; i < deviceCnt; ++i) {
            int deviceId = ivfpqConfig.deviceList[i];
            rpcContext ctx = contextMap.at(deviceId);
            int indexId = indexMap.at(ctx);
            for (int j = 0; j < nlist; ++j) {
                std::vector<uint8_t> code;
                std::vector<uint32_t> ids;
                RpcError ret = RpcIndexIVFGetListCodes(ctx, indexId, j, code, ids);
                FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Get list of IndexIVFPQ failed(%d).", ret);
                std::vector<idx_t> indice(ids.size());
                transform(ids.begin(), ids.end(),
                    begin(indice), [](uint32_t x) { return idx_t(x); });
                index->invlists->add_entries(j, indice.size(), indice.data(), code.data());
            }
        }
    }
}

void AscendIndexIVFPQ::initRpcCtx()
{
    indexMap.clear();
    contextMap.clear();
    for (size_t i = 0; i < ivfpqConfig.deviceList.size(); i++) {
        int indexId;
        rpcContext ctx;
        int deviceId = ivfpqConfig.deviceList[i];

        RpcError ret = RpcCreateContext(deviceId, &ctx);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create context failed(%d).", ret);
        contextMap[deviceId] = ctx;

        ret = RpcCreateIndexIVFPQ(ctx, indexId, this->d, nlist, subQuantizersCnt, bitsCntPerCode, this->metric_type,
            nprobe, ivfpqConfig.resourceSize);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create IndexIVFPQ failed(%d).", ret);
        indexMap[ctx] = indexId;
    }
}

void AscendIndexIVFPQ::clearRpcCtx()
{
    for (auto& index : indexMap) {
        RpcError ret = RpcDestroyIndexIVFPQ(index.first, index.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy IndexIVFPQ failed(%d).", ret);

        ret = RpcDestroyContext(index.first);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy context failed(%d).", ret);
    }

    indexMap.clear();
    contextMap.clear();
}

void AscendIndexIVFPQ::checkParams()
{
    FAISS_THROW_IF_NOT_MSG(this->metric_type == MetricType::METRIC_L2 ||
                           this->metric_type == MetricType::METRIC_INNER_PRODUCT,
                           "Unsupported metric type");

    // Must have some number of lists
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be >0");

    // up to a single byte per code
    FAISS_THROW_IF_NOT_FMT(bitsCntPerCode == BITS_CNT_PER_CODE,
                           "Bits per code must be 8 (passed %d)", bitsCntPerCode);

    // Diamension must be 16 aligned
    const int alignedNum = 16;
    FAISS_THROW_IF_NOT_FMT(this->d % alignedNum == 0,
                           "Number of diamension (%d) must be 16 aligned.", this->d);

    // Sub-quantizers must evenly divide dimensions available
    FAISS_THROW_IF_NOT_FMT(this->d % subQuantizersCnt == 0,
                           "Number of sub-quantizers (%d) must be an "
                           "even divisor of the number of dimensions (%d)",
                           subQuantizersCnt, this->d);

    // The number of bytes per encoded vector must be one we support
    FAISS_THROW_IF_NOT_FMT(isSupportedPQCodeLength(subQuantizersCnt),
                           "Number of bytes per encoded vector / sub-quantizers (%d) "
                           "is not supported", subQuantizersCnt);

    FAISS_THROW_IF_NOT_FMT(
        isSupportedPQSubDim(this->d / subQuantizersCnt),
        "Number of subdim: dim / sub-quantizers (%d) "
        "is not supported", this->d / subQuantizersCnt);
}

int AscendIndexIVFPQ::getNumSubQuantizers() const
{
    return subQuantizersCnt;
}

int AscendIndexIVFPQ::getBitsPerCode() const
{
    return bitsCntPerCode;
}

int AscendIndexIVFPQ::getCentroidsPerSubQuantizer() const
{
    const int clo = 2;
    return pow(clo, bitsCntPerCode);
}

bool AscendIndexIVFPQ::isSupportedPQCodeLength(int size)
{
    switch (PQSize(size)) {
        case PQ_SIZE_2:
        case PQ_SIZE_4:
        case PQ_SIZE_8:
        case PQ_SIZE_12:
        case PQ_SIZE_16:
        case PQ_SIZE_20:
        case PQ_SIZE_24:
        case PQ_SIZE_32:
        case PQ_SIZE_48:
        case PQ_SIZE_64:
        case PQ_SIZE_96:
        case PQ_SIZE_128:
            return true;
        default:
            return false;
    }
}

bool AscendIndexIVFPQ::isSupportedPQSubDim(int subDim)
{
    // PqSubDim is limited by DistanceTableBuild operator
    // subdim shoule be less than 128,
    // aligne with 16 or equal to 8,4
    switch (PQSubDim(subDim)) {
        case PQ_SUB_DIM_4:
        case PQ_SUB_DIM_8:
        case PQ_SUB_DIM_16:
        case PQ_SUB_DIM_32:
        case PQ_SUB_DIM_48:
        case PQ_SUB_DIM_64:
        case PQ_SUB_DIM_80:
        case PQ_SUB_DIM_96:
        case PQ_SUB_DIM_112:
        case PQ_SUB_DIM_128:
            return true;
        default:
            return false;
    }
}

int AscendIndexIVFPQ::getElementSize() const
{
    // element size: subQuantizersCnt * (bitsCntPerCode / 8) + sizeof(ID)
    return subQuantizersCnt + sizeof(uint32_t);
}

void AscendIndexIVFPQ::train(Index::idx_t n, const float* x)
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
    trainResidualQuantizer(n, x);
    this->is_trained = true;
}

uint32_t AscendIndexIVFPQ::getListLength(int listId, int deviceId) const
{
    uint32_t len = 0;
    rpcContext ctx = contextMap.at(deviceId);
    int indexId = indexMap.at(ctx);

    // get the list length from devices
    RpcError ret = RpcIndexIVFGetListLength(ctx, indexId, listId, len);
    FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "IVFPQ get list length failed(%d).", ret);
    return len;
}

void AscendIndexIVFPQ::getListCodesAndIds(int listId, int deviceId, std::vector<uint8_t>& codes,
                                          std::vector<uint32_t>& ids) const
{
    FAISS_ASSERT((listId >= 0) && (listId < nlist));

    rpcContext ctx = contextMap.at(deviceId);
    int indexId = indexMap.at(ctx);
    RpcError ret = RpcIndexIVFGetListCodes(ctx, indexId, listId, codes, ids);
    FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "GetList Code and Indices failed(%d).", ret);
}

void AscendIndexIVFPQ::addImpl(int n, const float* x, const Index::idx_t* ids)
{
    size_t deviceCnt = ivfpqConfig.deviceList.size();

    // calculate which list the vectors belongs to
    faiss::ScopeDeleter<Index::idx_t> del1;
    Index::idx_t* assign = new Index::idx_t[n];
    del1.set(assign);
    cpuQuantizer->assign(n, x, assign);

    // get the residuals with the l1 coarse centroids
    faiss::ScopeDeleter<float> del2;
    float* residuals = new float[n * d];
    del2.set(residuals);
    cpuQuantizer->compute_residual_n(n, x, residuals, assign);

    // compute the pq codes
    faiss::ScopeDeleter<uint8_t> del3;
    uint8_t* codes = new uint8_t[n * pqData.code_size];
    del3.set(codes);
    pqData.compute_codes(residuals, codes, n);

    // list id -> # being added
    std::unordered_map<int, IVFPQAddInfo> assignCounts;
    for (int i = 0; i < n; i++) {
        Index::idx_t listId = assign[i];
        FAISS_ASSERT(listId >= 0 && listId < this->nlist);
        auto it = assignCounts.find(listId);
        if (it != assignCounts.end()) {
            it->second.Add(codes + i * pqData.code_size, ids + i);
        } else {
            int devIdx = 0;
            for (size_t j = 1; j < deviceCnt; j++) {
                if (deviceAddNumMap[listId][j] <
                    deviceAddNumMap[listId][devIdx]) {
                    devIdx = j;
                    break;
                }
            }

            assignCounts.emplace(listId, IVFPQAddInfo(devIdx, deviceCnt, pqData.code_size));
            assignCounts.at(listId).Add(codes + i * pqData.code_size, ids + i);
        }
    }
    
    auto addFunctor = [&](int idx) {
        int deviceId = ivfpqConfig.deviceList[idx];
        rpcContext ctx = contextMap[deviceId];
        int indexId = indexMap[ctx];
        for (auto& centroid : assignCounts) {
            int listId = centroid.first;
            int num = centroid.second.GetAddNum(idx);
            uint8_t* codePtr = nullptr;
            uint32_t* idPtr = nullptr;
            centroid.second.GetCodeAndIdPtr(idx, &codePtr, &idPtr);
            RpcError ret = RpcIndexIVFPQAdd(ctx, indexId, num, listId, codePtr, 
                                            pqData.code_size, idPtr);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Add with ids failed(%d).", ret);
            deviceAddNumMap[listId][idx] += num;
        }
    };

    if (deviceCnt > 1) {
        std::vector<std::future<void>> addFunctorRets;
        for (size_t i = 0; i < deviceCnt; i++) {
            addFunctorRets.emplace_back(pool->Enqueue(addFunctor, i));
        }

        try {
            for (auto& ret : addFunctorRets) {
                ret.get();
            }
        } catch (std::exception &e) {
            FAISS_THROW_FMT("%s.", e.what());
        }
    } else {
        addFunctor(0);
    }
    this->ntotal += n;
}

void AscendIndexIVFPQ::updateDevicePQCenter()
{
    std::vector<uint16_t> pqCentroidFp16(pqData.centroids.size());
    transform(begin(pqData.centroids), end(pqData.centroids),
              begin(pqCentroidFp16), [](float temp) { return fp16(temp).data; });

    // update PQCent to device
    size_t deviceCnt = ivfpqConfig.deviceList.size();
    for (size_t i = 0; i < deviceCnt; i++) {
        int deviceId = ivfpqConfig.deviceList[i];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        RpcError ret = RpcIndexIVFPQUpdatePQCent(ctx, indexId, pqCentroidFp16.data(), 
                                                 pqData.M, pqData.ksub, pqData.dsub);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Update PQCent failed(%d).", ret);
    }
}

void AscendIndexIVFPQ::trainResidualQuantizer(Index::idx_t n, const float* x)
{
    const int subPqVecs = 64;
    n = std::min(n, (Index::idx_t)(1UL << static_cast<unsigned int>(bitsCntPerCode)) * subPqVecs);

    std::vector<Index::idx_t> assign(n);        
    cpuQuantizer->assign(n, x, assign.data());  

    std::vector<float> residuals(n * d);
    cpuQuantizer->compute_residual_n(n, x, residuals.data(), assign.data());  

    if (this->verbose) {
        printf("training %d x %d product quantizer on %ld vectors in %dD\n",
               subQuantizersCnt, getCentroidsPerSubQuantizer(), n, this->d);
    }

    // Just use the CPU product quantizer to determine sub-centroids
    pqData = faiss::ProductQuantizer(this->d, subQuantizersCnt, bitsCntPerCode);
    pqData.verbose = this->verbose;
    pqData.train(n, residuals.data());

    updateDevicePQCenter();
}
}  // namespace ascend
}  // namespace faiss
