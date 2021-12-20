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

#include <limits>
#include <algorithm>

#include <faiss/impl/FaissAssert.h>
#include <faiss/ascend/AscendIndex.h>
#include <faiss/ascend/utils/AscendUtils.h>
#include <faiss/ascend/utils/AscendThreadPool.h>
#include <faiss/ascend/utils/fp16.h>
#include <faiss/ascend/rpc/AscendRpc.h>

namespace faiss {
namespace ascend {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 64;
const size_t UNIT_VEC_SIZE = 512;

// Default size for which we page add or search
const size_t ADD_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;

// Or, maximum number of vectors to consider per page of add
const size_t ADD_VEC_SIZE = UNIT_VEC_SIZE * KB;

// search pagesize must be less than 64M, becauseof rpc limitation
const size_t SEARCH_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t SEARCH_VEC_SIZE = UNIT_VEC_SIZE * KB;
}

AscendIndex::AscendIndex(int dims, faiss::MetricType metric,
                         AscendIndexConfig config)
    : Index(dims, metric), indexConfig(config), pool(nullptr)
{
    FAISS_THROW_IF_NOT_MSG(dims > 0, "Invalid number of dimensions");
    pool = new AscendThreadPool(indexConfig.deviceList.size());
}

AscendIndex::~AscendIndex()
{
    if (pool) {
        delete pool;
        pool = nullptr;
    }
}

void AscendIndex::add(idx_t n, const float* x)
{
    return add_with_ids(n, x, nullptr);
}

void AscendIndex::add_with_ids(Index::idx_t n, const float* x,
                               const Index::idx_t* ids)
{
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");
    FAISS_THROW_IF_NOT_FMT(n <= (Index::idx_t)std::numeric_limits<int>::max(),
                           "indices exceeds max(%d)", std::numeric_limits<int>::max());

    if (n == 0) {
        return;
    }

    std::vector<Index::idx_t> tmpIds;
    if (ids == nullptr && addImplRequiresIDs()) {
        tmpIds = std::vector<Index::idx_t>(n);

        for (Index::idx_t i = 0; i < n; ++i) {
            tmpIds[i] = this->ntotal + i;
        }

        ids = tmpIds.data();
    }

    return addPaged(n, x, ids);
}

void AscendIndex::addPaged(int n, const float* x, const Index::idx_t* ids)
{
    if (n <= 0) {
        return;
    }

    size_t totalSize = static_cast<size_t>(n) * static_cast<size_t>(getElementSize());
    if (totalSize > ADD_PAGE_SIZE || static_cast<size_t>(n) > ADD_VEC_SIZE) {
        // How many vectors fit into kAddPageSize?
        size_t maxNumVecsForPageSize =
            ADD_PAGE_SIZE / getElementSize();
        // Always add at least 1 vector, if we have huge vectors
        maxNumVecsForPageSize = std::max(
            maxNumVecsForPageSize, (size_t)1);

        size_t tileSize = std::min((size_t)n, maxNumVecsForPageSize);

        for (size_t i = 0; i < (size_t)n; i += tileSize) {
            size_t curNum = std::min(tileSize, n - i);
            if (this->verbose) {
                printf("AscendIndex::add: adding %ld:%ld / %d\n", i, i + curNum, n);
            }
            addImpl(curNum, x + i * (size_t)this->d,
                    ids ? (ids + i) : nullptr);
        }
    } else {
        if (this->verbose) {
            printf("AscendIndex::add: adding 0:%d / %d\n", n, n);
        }
        addImpl(n, x, ids);
    }
}

size_t AscendIndex::remove_ids(const faiss::IDSelector& sel)
{
    return removeImpl(sel);
}

void AscendIndex::search(Index::idx_t n, const float* x, Index::idx_t k,
                         float* distances, Index::idx_t* labels) const
{
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "distances can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "labels can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n must be > 0");
    FAISS_THROW_IF_NOT_MSG(k > 0, "k must be > 0");
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");
    FAISS_THROW_IF_NOT_FMT(n <= (Index::idx_t)std::numeric_limits<int>::max(),
                           "indices exceeds max(%d)", std::numeric_limits<int>::max());
    FAISS_THROW_IF_NOT_MSG(contextMap.size() > 0, "contextMap.size must be >0");
    FAISS_THROW_IF_NOT_MSG(indexMap.size() > 0, "indexMap.size must be >0");

    if (n == 0 || k == 0) {
        return;
    }

    return searchPaged(n, x, k, distances, labels);
}

void AscendIndex::searchPaged(int n, const float* x, int k,
                              float* distances, Index::idx_t* labels) const
{
    if (n > 0) {
        size_t totalSize = (size_t)n * this->d * sizeof(float);
        size_t totalOutSize = (size_t)n * k * (sizeof(uint16_t) + sizeof(uint32_t));

        if (totalSize > SEARCH_PAGE_SIZE || (size_t)n > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
            // How many vectors fit into searchPageSize?
            size_t maxNumVecsForPageSize =
                SEARCH_PAGE_SIZE / ((size_t)this->d * sizeof(float));

            size_t maxRetVecsForPageSize =
                SEARCH_PAGE_SIZE / (k * (sizeof(uint16_t) + sizeof(uint32_t)));

            maxNumVecsForPageSize = std::min(
                maxNumVecsForPageSize, maxRetVecsForPageSize);

            // Always add at least 1 vector, if we have huge vectors
            maxNumVecsForPageSize = std::max(
                maxNumVecsForPageSize, (size_t)1);

            size_t tileSize = std::min((size_t)n, maxNumVecsForPageSize);

            for (int i = 0; i < n; i += tileSize) {
                size_t curNum = std::min(tileSize, (size_t)(n) - i);
                searchImpl(curNum, x + i * (size_t)this->d, k,
                           distances + i * k, labels + i * k);
            }
        } else {
            searchImpl(n, x, k, distances, labels);
        }
    }
}

void AscendIndex::searchImpl(int n, const float* x, int k, float* distances,
    Index::idx_t* labels) const
{
    size_t deviceCnt = indexConfig.deviceList.size();

    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * this->d, 0);
    transform(x, x + n * this->d, begin(query), [](float temp) { return fp16(temp).data; });

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt, std::vector<uint16_t>(n * k, 0));
    std::vector<std::vector<uint32_t>> label(deviceCnt, std::vector<uint32_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        RpcError ret = RpcIndexSearch(ctx, indexId, n, this->d, k, query.data(),
            distHalf[idx].data(), label[idx].data());
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Search implement failed(%d).", ret);

        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]),
                  begin(dist[idx]), [](uint16_t temp) { return (float)fp16(temp); });
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);

    // post process: default is merge the topk results from devices
    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);
}

void AscendIndex::compute_residual(const float* x, float* residual,
                                   Index::idx_t key) const
{
    VALUE_UNUSED(x);
    VALUE_UNUSED(residual);
    VALUE_UNUSED(key);

    FAISS_THROW_MSG("compute_residual not implemented for this type of index");
}

void AscendIndex::compute_residual_n(Index::idx_t n, const float* xs,
                                     float* residuals, const Index::idx_t* keys) const
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(xs);
    VALUE_UNUSED(residuals);
    VALUE_UNUSED(keys);

    FAISS_THROW_MSG("compute_residual_n not implemented for this type of index");
}

void AscendIndex::reserveMemory(size_t numVecs)
{
    VALUE_UNUSED(numVecs);

    FAISS_THROW_MSG("reserveMemory not implemented for this type of index");
}

size_t AscendIndex::reclaimMemory()
{
    FAISS_THROW_MSG("reclaimMemory not implemented for this type of index");
    return 0;
}

void AscendIndex::mergeSearchResult(size_t devices, std::vector<std::vector<float>>& dist,
                                    std::vector<std::vector<uint32_t>>& label, int n, int k,
                                    float* distances, Index::idx_t* labels) const
{
    std::function<bool(float, float)> compFunc;
    switch (this->metric_type) {
        case faiss::METRIC_L2:
            std::less<float> lessComp;
            compFunc = lessComp;
            break;
        case faiss::METRIC_INNER_PRODUCT:
            std::greater<float> greaterComp;
            compFunc = greaterComp;
            break;
        default:
            FAISS_THROW_MSG("Unsupported metric type");
            break;
    }

    // merge several topk results into one topk results
    // every topk result need to be reodered in ascending order
#pragma omp parallel for if (n > 100)
    for (int i = 0; i < n; i++) {
        int num = 0;
        int offset = i * k;
        std::vector<int> posit(devices, 0);
        while (num < k) {
            size_t id = 0;
            float disMerged = dist[0][offset + posit[0]];
            uint32_t labelMerged = label[0][offset + posit[0]];
            for (size_t j = 1; j < devices; j++) {
                int pos = offset + posit[j];
                if (compFunc(dist[j][pos], disMerged)) {
                    disMerged = dist[j][pos];
                    labelMerged = label[j][pos];
                    id = j;
                }
            }

            *(distances + offset + num) = disMerged;
            *(labels + offset + num) = (Index::idx_t)labelMerged;
            posit[id]++;
            num++;
        }
    }
}

void AscendIndex::searchPostProcess(size_t devices, std::vector<std::vector<float>>& dist,
                                    std::vector<std::vector<uint32_t>>& label, int n, int k,
                                    float* distances, Index::idx_t* labels) const
{
    return mergeSearchResult(devices, dist, label, n, k, distances, labels);
}
}  // namespace ascend
}  // namespace faiss
