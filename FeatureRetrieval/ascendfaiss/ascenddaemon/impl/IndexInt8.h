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
#include <arm_fp16.h>
#include <ascenddaemon/impl/Index.h>
#include <ascenddaemon/StandardAscendResources.h>

namespace ascend {
struct IDSelector;

class IndexInt8 {
public:
    using idx_t = Index::idx_t;

    // index constructor, resourceSize = -1 means using default config of resource
    explicit IndexInt8(idx_t d = 0, MetricType metric = MetricType::METRIC_L2, int resourceSize = -1);

    inline StandardAscendResources *getResources()
    {
        return &resources;
    }

    virtual ~IndexInt8();

    // Perform training on a representative set of vectors
    virtual void train(idx_t n, const int8_t *x);

    virtual void add(idx_t n, const int8_t *x);

    virtual void addWithIds(idx_t n, const int8_t *x, const idx_t *xids);

    virtual void addVectors(AscendTensor<int8_t, DIMS_2> &rawData);

    // removes all elements from the database.
    virtual void reset() = 0;

    // reserve memory for the database.
    virtual void reserveMemory(size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    virtual size_t reclaimMemory();

    // remove IDs from the index. Not supported by all indexes.
    //  Returns the number of elements removed.
    virtual size_t removeIds(const IDSelector &sel);

    // query n vectors of dimension d to the index
    // return at most k vectors. If there are not enough results for a query,
    // the result array is padded with -1s.
    void search(idx_t n, const int8_t *x, idx_t k, float16_t *distances, idx_t *labels, uint8_t *mask = nullptr);

protected:
    virtual void addImpl(int n, const int8_t *x, const idx_t *ids) = 0;

    virtual bool addImplRequiresIDs() const;

    virtual void searchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels) = 0;

    virtual size_t removeIdsImpl(const IDSelector &sel) = 0;

    void addPaged(int n, const int8_t *x, const idx_t *ids);

    void searchPaged(int n, const int8_t *x, int k, float16_t *distance, idx_t *labels);

protected:
    // Manage resources on ascend
    StandardAscendResources resources;

    // vector dimension
    int dims;

    // total nb of indexed vectors
    idx_t ntotal;

    // metric type
    MetricType metricType;

    // set if the Index does not require training, or if training is done already
    bool isTrained;

    uint8_t *maskData;
    uint32_t maskSearchedOffset;

    // support search batch sizes, default is no paging
    std::vector<int> searchPageSizes;
};
} // namespace ascend

#endif // ASCEND_INDEX_INT8_INCLUDED
