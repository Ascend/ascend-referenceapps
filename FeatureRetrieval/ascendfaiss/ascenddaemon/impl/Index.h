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

#ifndef ASCEND_INDEX_INCLUDED
#define ASCEND_INDEX_INCLUDED

#include <map>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <arm_fp16.h>
#include <ascenddaemon/StandardAscendResources.h>
#include <ascenddaemon/utils/TopkOp.h>
#include <ascenddaemon/utils/AscendOperator.h>

namespace ascend {
struct IDSelector;
class IndexPreTransform;

enum MetricType {
    METRIC_INNER_PRODUCT = 0,  // maximum inner product search
    METRIC_L2 = 1,             // squared L2 search
    METRIC_COSINE = 2,         // consine search
};

class Index {
public:
    using idx_t = uint32_t;

    // index constructor, resourceSize = -1 means using default config of resource
    explicit Index(idx_t d = 0, int resourceSize = -1);

    inline StandardAscendResources* getResources()
    {
        return &resources;
    }

    virtual ~Index();

    // Perform training on a representative set of vectors
    virtual void train(idx_t n, const float16_t* x);

    virtual void add(idx_t n, const float16_t* x);

    virtual void addWithIds(idx_t n, const float16_t* x, const idx_t* xids);

    // removes all elements from the database.
    virtual void reset() = 0;

    // reserve memory for the database.
    virtual void reserveMemory(size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    virtual size_t reclaimMemory();

    // remove IDs from the index. Not supported by all indexes.
    //  Returns the number of elements removed.
    virtual size_t removeIds(const IDSelector& sel);

    // query n vectors of dimension d to the index
    // return at most k vectors. If there are not enough results for a query,
    // the result array is padded with -1s.
    void search(idx_t n, const float16_t* x, idx_t k,
                float16_t* distances, idx_t* labels);

protected:
    virtual void addImpl(int n, const float16_t* x, const idx_t* ids) = 0;

    virtual bool addImplRequiresIDs() const;

    virtual void searchImpl(int n, const float16_t* x, int k,
                            float16_t* distances, idx_t* labels) = 0;

    virtual size_t removeIdsImpl(const IDSelector& sel) = 0;

    void addPaged(int n, const float16_t* x, const idx_t* ids);

    void searchPaged(int n, const float16_t* x, int k,
                     float16_t* distance, idx_t* labels);

    void resetDistCompOperator(int numLists);

    void runDistanceCompute(AscendTensor<float16_t, DIMS_2>& queryVecs,
                            AscendTensor<float16_t, DIMS_4>& shapedData,
                            AscendTensor<float16_t, DIMS_1>& norms,
                            AscendTensor<float16_t, DIMS_2>& outDistances,
                            AscendTensor<uint16_t, DIMS_1>& flag,
                            aclrtStream stream);

    float16_t fvecNormL2sqr(const float16_t* x, size_t d);
    void fvecNormsL2sqr(float16_t* nr, const float16_t* x, size_t d, size_t nx);

protected:
    // Manage resources on ascend
    StandardAscendResources resources;

    // vector dimension
    int dims;

    // total nb of indexed vectors
    idx_t ntotal;

    // set if the Index does not require training, or if training is done already
    bool isTrained;

    // support search batch sizes, default is no paging
    std::vector<int> searchPageSizes;

    // shared ops
    std::map<int, std::unique_ptr<AscendOperator>> distComputeOps;
    TopkOp<std::greater<>, std::greater_equal<>, float16_t> topkOp;

    friend class IndexPreTransform;
};
}  // namespace ascend

#endif  // ASCEND_INDEX_INCLUDED
