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

#include <iterator>
#include <atomic>
#include <algorithm>

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>

#include <faiss/ascend/AscendIndexIVF.h>
#include <faiss/ascend/AscendClustering.h>
#include <faiss/ascend/utils/fp16.h>
#include <faiss/ascend/utils/AscendUtils.h>
#include <faiss/ascend/utils/AscendThreadPool.h>
#include <faiss/ascend/rpc/AscendRpc.h>

namespace faiss {
namespace ascend {
constexpr int DEFAULT_NPROBE = 64;

// implementation of AscendIndexIVF
AscendIndexIVF::AscendIndexIVF(int dims, faiss::MetricType metric, int listNum, AscendIndexIVFConfig config)
    : AscendIndex(dims, metric, config),
      nlist(listNum),
      nprobe(DEFAULT_NPROBE),
      cpuQuantizer(nullptr),
      ivfConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(this->metric_type == MetricType::METRIC_L2 ||
        this->metric_type == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");
    checkIVFParams();
    
    ivfConfig.cp.verbose = this->verbose;

    if (!cpuQuantizer) {
        // Construct an empty cpuQuantizer
        if (this->metric_type == faiss::METRIC_L2 || this->metric_type == MetricType::METRIC_INNER_PRODUCT) {
            cpuQuantizer = new IndexFlatL2(this->d);
        } else {
            // unknown metric type
            FAISS_THROW_IF_NOT_MSG(false, "unsupported metric type");
        }
    }
}

AscendIndexIVF::~AscendIndexIVF()
{
    if (cpuQuantizer) {
        delete cpuQuantizer;
        cpuQuantizer = nullptr;
    }
}

void AscendIndexIVF::checkIVFParams()
{   
    FAISS_THROW_IF_NOT_MSG(ivfConfig.cp.niter > 0, "ClusteringParameters niter must > 0");

    FAISS_THROW_IF_NOT_MSG(ivfConfig.cp.nredo > 0, "ClusteringParameters nredo must > 0");

    FAISS_THROW_IF_NOT_MSG(ivfConfig.cp.min_points_per_centroid > 0, 
        "ClusteringParameters min_points_per_centroid must > 0");

    FAISS_THROW_IF_NOT_MSG(ivfConfig.cp.max_points_per_centroid > 0, 
        "ClusteringParameters max_points_per_centroid must > 0");
}

// Copy what we need from the CPU equivalent
void AscendIndexIVF::copyFrom(const faiss::IndexIVF *index)
{
    this->d = index->d;
    FAISS_THROW_IF_NOT_MSG(index->metric_type == MetricType::METRIC_L2 ||
                           index->metric_type == MetricType::METRIC_INNER_PRODUCT,
                           "Unsupported metric type");

    this->metric_type = index->metric_type;

    FAISS_THROW_IF_NOT_FMT(index->nlist > 0, "Invalid number of nlist %zd", index->nlist);
    nlist = index->nlist;

    FAISS_THROW_IF_NOT_FMT(index->nprobe > 0, "Invalid number of nprobe %zd", index->nprobe);
    nprobe = index->nprobe;

    if (cpuQuantizer) {
        delete cpuQuantizer;
        cpuQuantizer = nullptr;
    }

    cpuQuantizer = new IndexFlatL2(this->d);

    if (!index->is_trained) {
        this->is_trained = false;
        this->ntotal = 0;
        return;
    }

    this->is_trained = true;
    this->ntotal = index->ntotal;

    auto flat = dynamic_cast<faiss::IndexFlat *>(index->quantizer);
    FAISS_THROW_IF_NOT_MSG(flat,
                           "Only IndexFlat is supported for the coarse quantizer "
                           "for copying from an IndexIVF into a AscendIndexIVF");

    cpuQuantizer->add(flat->ntotal, flat->xb.data());
}

// / Copy what we have to the CPU equivalent
void AscendIndexIVF::copyTo(faiss::IndexIVF *index) const
{
    index->ntotal = this->ntotal;
    index->d = this->d;
    index->metric_type = this->metric_type;
    index->is_trained = this->is_trained;
    index->nlist = nlist;
    index->nprobe = nprobe;

    faiss::IndexFlat *q = new faiss::IndexFlatL2(this->d);

    *q = *cpuQuantizer;

    if (index->own_fields) {
        delete index->quantizer;
    }

    index->quantizer = q;
    index->quantizer_trains_alone = 0;
    index->own_fields = true;
    index->cp = this->ivfConfig.cp;
    index->maintain_direct_map = false;
    index->direct_map.clear();
}

void AscendIndexIVF::reset()
{
    auto resetFunctor = [&](rpcContext ctx, int indexId) {
        RpcError ret = RpcIndexReset(ctx, indexId);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Reset IndexIVFSQ failed(%d).", ret);
    };

    // parallel reset to every device
    CALL_PARALLEL_FUNCTOR_INDEXMAP(indexMap, pool, resetFunctor);

    initDeviceAddNumMap();
    this->ntotal = 0;
}

// Sets the number of list probes per query
void AscendIndexIVF::setNumProbes(int nprobes)
{
    FAISS_THROW_IF_NOT_MSG(nprobes > 0, "Invalid number of nprobe");
    this->nprobe = nprobes;
    for (auto &index : indexMap) {
        RpcError ret = RpcIndexIVFUpdateNprobe(index.first, index.second, nprobes);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Set NumProbes failed(%d).", ret);
    }
}

bool AscendIndexIVF::addImplRequiresIDs() const
{
    // All IVF indices have storage for IDs
    return true;
}

void AscendIndexIVF::trainQuantizer(faiss::Index::idx_t n, const float *x)
{
    if (n == 0) {
        // nothing to do
        return;
    }

    if (cpuQuantizer->is_trained && (cpuQuantizer->ntotal == nlist)) {
        if (this->verbose) {
            printf("IVF quantizer does not need training.\n");
        }

        return;
    }

    if (this->verbose) {
        printf("Training IVF quantizer on %ld vectors in %dD to %d cluster\n", n, d, nlist);
    }

    cpuQuantizer->reset();
    if (this->ivfConfig.useKmeansPP) {
        AscendClustering clus(this->d, nlist, this->ivfConfig.cp);
        clus.verbose = this->verbose;
        clus.train(n, x, *cpuQuantizer);
    } else {
        Clustering clus(this->d, nlist, this->ivfConfig.cp);
        clus.verbose = this->verbose;
        clus.train(n, x, *cpuQuantizer);
    }
    cpuQuantizer->is_trained = true;

    FAISS_ASSERT(cpuQuantizer->ntotal == nlist);
}

uint32_t AscendIndexIVF::getListLength(int listId) const
{
    FAISS_ASSERT((listId >= 0) && (listId < nlist));

    uint32_t len = 0;
    for (auto &index : indexMap) {
        // get the list length from devices
        uint32_t tmpLen = 0;
        RpcError ret = RpcIndexIVFGetListLength(index.first, index.second, listId, tmpLen);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "IVFPQ get list length failed(%d).", ret);
        len += tmpLen;
    }
    return len;
}

void AscendIndexIVF::getListCodesAndIds(int listId, std::vector<uint8_t> &codes, std::vector<uint32_t> &ids) const
{
    FAISS_ASSERT((listId >= 0) && (listId < nlist));
    codes.clear();
    ids.clear();

    // use for(deviceList) rather than for(auto& index : indexMap),
    // to ensure merged codes and ids in sequence
    for (size_t j = 0; j < indexConfig.deviceList.size(); j++) {
        int deviceId = indexConfig.deviceList[j];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        std::vector<uint8_t> tmpCodes;
        std::vector<uint32_t> tmpIds;
        RpcError ret = RpcIndexIVFGetListCodes(ctx, indexId, listId, tmpCodes, tmpIds);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "GetList Code and Indices failed(%d).", ret);
        std::copy(tmpCodes.begin(), tmpCodes.end(), back_inserter(codes));
        std::copy(tmpIds.begin(), tmpIds.end(), back_inserter(ids));
    }
}

void AscendIndexIVF::reserveMemory(size_t numVecs)
{
    size_t deviceNum = indexConfig.deviceList.size();
    size_t numPerDev = (numVecs + deviceNum - 1) / deviceNum;
    uint32_t numReserve = static_cast<uint32_t>(numPerDev);

    auto reserveFunctor = [&](rpcContext ctx, int indexId) {
        RpcError ret = RpcIndexReserveMemory(ctx, indexId, numReserve);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "reserveMemory failed(%d).", ret);
    };

    // parallel reserveMemory to every device
    CALL_PARALLEL_FUNCTOR_INDEXMAP(indexMap, pool, reserveFunctor);
}

size_t AscendIndexIVF::reclaimMemory()
{
    std::atomic<size_t> tmpReclaimed(0);

    auto reclaimFunctor = [&](rpcContext ctx, int indexId) {
        uint32_t tmp = 0;
        RpcError ret = RpcIndexReclaimMemory(ctx, indexId, tmp);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "reclaimMemory failed(%d).", ret);
        tmpReclaimed += tmp;
    };

    // parallel reclaimMemory to every device
    CALL_PARALLEL_FUNCTOR_INDEXMAP(indexMap, pool, reclaimFunctor);
    return tmpReclaimed.load();
}

void AscendIndexIVF::initDeviceAddNumMap()
{
    deviceAddNumMap.clear();
    deviceAddNumMap.resize(nlist);
    for (int i = 0; i < nlist; i++) {
        deviceAddNumMap[i] = std::vector<int>(indexConfig.deviceList.size(), 0);
    }
}

void AscendIndexIVF::updateDeviceCoarseCenter()
{
    // coarse centroids have been stored on host in fp16 format
    std::vector<uint16_t> corseCentroidsFp16(cpuQuantizer->xb.size());
    transform(begin(cpuQuantizer->xb), end(cpuQuantizer->xb),
              begin(corseCentroidsFp16), [](float temp) { return fp16(temp).data; });

    for (auto &index : indexMap) {
        RpcError ret =
            RpcIndexIVFUpdateCoarseCent(index.first, index.second, corseCentroidsFp16.data(), this->d, this->nlist);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Update CoarseCent failed(%d).", ret);
    }
}

size_t AscendIndexIVF::removeImpl(const IDSelector &sel)
{
    int deviceCnt = indexConfig.deviceList.size();
    uint32_t removeCnt = 0;

    // remove vector from device
    if (auto rangeSel = dynamic_cast<const IDSelectorBatch *>(&sel)) {
        std::vector<uint32_t> removeBatch(rangeSel->set.size());
        transform(begin(rangeSel->set), end(rangeSel->set),
                  begin(removeBatch), [](idx_t temp) { return (uint32_t)temp; });

#pragma omp parallel for reduction(+ : removeCnt)
        for (int i = 0; i < deviceCnt; i++) {
            int deviceId = indexConfig.deviceList[i];
            rpcContext ctx = contextMap.at(deviceId);
            int indexId = indexMap.at(ctx);
            RpcError ret = RpcIndexRemoveIds(ctx, indexId, removeBatch.size(), removeBatch.data(), &removeCnt);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Remove ID selector failed(%d).", ret);
        }
    } else if (auto rangeSel = dynamic_cast<const IDSelectorRange *>(&sel)) {
#pragma omp parallel for reduction(+ : removeCnt)
        for (int i = 0; i < deviceCnt; i++) {
            int deviceId = indexConfig.deviceList[i];
            rpcContext ctx = contextMap[deviceId];
            int indexId = indexMap.at(ctx);
            RpcError ret = RpcIndexRemoveRangeIds(ctx, indexId, rangeSel->imin, rangeSel->imax, &removeCnt);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Remove ID range failed(%d).", ret);
        }
    }

    // update vector nums of deviceAddNumMap
#pragma omp parallel for if (deviceCnt > 1) num_threads(deviceCnt)
    for (int i = 0; i < deviceCnt; i++) {
        int deviceId = indexConfig.deviceList[i];
        rpcContext ctx = contextMap[deviceId];
        int indexId = indexMap.at(ctx);
        for (int listId = 0; listId < nlist; listId++) {
            uint32_t len = 0;
            RpcError ret = RpcIndexIVFGetListLength(ctx, indexId, listId, len);
            FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "IVFPQ get list length failed(%d).", ret);
            deviceAddNumMap[listId][i] = len;
        }
    }

    this->ntotal -= removeCnt;
    return (size_t)removeCnt;
}
} // namespace ascend
} // namespace faiss
