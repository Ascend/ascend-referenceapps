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

#ifndef ASCEND_INDEX_INT8_INCLUDED
#define ASCEND_INDEX_INT8_INCLUDED

#include <vector>
#include <unordered_map>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/ascend/utils/AscendThreadPool.h>

namespace faiss {
namespace ascend {
using rpcContext = void *;

const int INDEX_INT8_DEFAULT_MEM = static_cast<int>(0x2000000); // 0x2000000 mean 32M(resource mem pool's size)

struct AscendIndexInt8Config {
    inline AscendIndexInt8Config() : deviceList({ 0 }), resourceSize(INDEX_INT8_DEFAULT_MEM) {}

    inline AscendIndexInt8Config(std::initializer_list<int> devices, int resources = INDEX_INT8_DEFAULT_MEM)
        : deviceList(devices), resourceSize(resources)
    {
        FAISS_THROW_IF_NOT_MSG(deviceList.size() != 0, "device list can not be empty!");
    }

    inline AscendIndexInt8Config(std::vector<int> devices, int resources = INDEX_INT8_DEFAULT_MEM)
        : deviceList(devices), resourceSize(resources)
    {
        FAISS_THROW_IF_NOT_MSG(deviceList.size() != 0, "device list can not be empty!");
    }

    // Ascend devices mask on which the index is resident
    std::vector<int> deviceList;
    int resourceSize;
};

class AscendIndexInt8 {
public:
    AscendIndexInt8(int dims, faiss::MetricType metric, AscendIndexInt8Config config);

    virtual ~AscendIndexInt8();

    // Perform training on a representative set of vectors
    virtual void train(Index::idx_t n, const int8_t *x);

    // `x` need to be resident on CPU
    // update the centroids clustering center of the training result
    virtual void updateCentroids(Index::idx_t n, const int8_t *x);

    // `x` need to be resident on CPU
    // update the centroids clustering center of the training result
    virtual void updateCentroids(Index::idx_t n, const char *x);

    // `x` need to be resident on CPU
    // Handles paged adds if the add set is too large;
    void add(Index::idx_t n, const int8_t *x);

    // `x` and `ids` need to be resident on the CPU;
    // Handles paged adds if the add set is too large;
    void add_with_ids(Index::idx_t n, const int8_t *x, const Index::idx_t *ids);

    // `x` need to be resident on CPU
    // Handles paged adds if the add set is too large;
    void add(Index::idx_t n, const char *x);

    // `x` and `ids` need to be resident on the CPU;
    // Handles paged adds if the add set is too large;
    void add_with_ids(Index::idx_t n, const char *x, const Index::idx_t *ids);

    // removes IDs from the index. Not supported by all
    // indexes. Returns the number of elements removed.
    size_t remove_ids(const faiss::IDSelector &sel);

    void assign(Index::idx_t n, const int8_t *x, Index::idx_t *labels, Index::idx_t k = 1);

    // `x`, `distances` and `labels` need to be resident on the CPU
    void search(Index::idx_t n, const int8_t *x, Index::idx_t k, float *distances, Index::idx_t *labels) const;

    // `x`, `distances` and `labels` need to be resident on the CPU
    void search(Index::idx_t n, const char *x, Index::idx_t k, float *distances, Index::idx_t *labels) const;

    // reserve memory for the database.
    virtual void reserveMemory(size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    virtual size_t reclaimMemory();

public:
    // vector dimension
    int d = 512;

    // verbose level
    bool verbose = false;

    // total nb of indexed vectors
    faiss::Index::idx_t ntotal = 0;

    // set if the Index does not require training, or if training is done already
    bool is_trained = true;

    // type of metric this index uses for search
    faiss::MetricType metric_type = faiss::METRIC_L2;

protected:
    // Does addImpl_ require IDs? If so, and no IDs are provided, we will
    // generate them sequentially based on the order in which the IDs are added
    virtual bool addImplRequiresIDs() const = 0;

    // Overridden to actually perform the add
    virtual void addImpl(int n, const int8_t *x, const Index::idx_t *ids) = 0;

    // Overridden to actually perform the search
    virtual void searchImpl(int n, const int8_t *x, int k, float *distances, Index::idx_t *labels) const;

    // Overridden to actually perform the remove_ids
    virtual size_t removeImpl(const IDSelector &sel) = 0;

    // Handles paged adds if the add set is too large, passes to
    // addImpl to actually perform the add for the current page
    void addPaged(int n, const int8_t *x, const Index::idx_t *ids);

    // Handles paged search if the search set is too large, passes to
    // searchImpl to actually perform the search for the current page
    void searchPaged(int n, const int8_t *x, int k, float *distances, Index::idx_t *labels) const;

    // get the size of memory every database vector needed to store.
    virtual int getElementSize() const = 0;

    // merge topk results from all devices used in search process
    virtual void mergeSearchResult(size_t devices, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<uint32_t>> &label, int n, int k, float *distances, Index::idx_t *labels) const;

    // post process after search results got from all devices
    virtual void searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<uint32_t>> &label, int n, int k, float *distances, Index::idx_t *labels) const;

protected:
    AscendIndexInt8Config indexConfig;

    // thread pool for multithread processing
    AscendThreadPool *pool;

    // device --> context
    std::unordered_map<int, rpcContext> contextMap;

    // context --> index * n
    std::unordered_map<rpcContext, int> indexMap;
};

#define VALUE_UNUSED(x) (void)x;
} // namespace ascend
} // namespace faiss

#endif