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

#ifndef ASCEND_INDEX_INT8_FLAT_INCLUDED
#define ASCEND_INDEX_INT8_FLAT_INCLUDED

#include <vector>

#include <ascenddaemon/impl/IndexInt8.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/AscendThreadPool.h>

namespace ascend {
const int FLAT_DEFAULT_DIST_COMPUTE_BATCH = 16384 * 16;

template<typename P>
class IndexInt8Flat : public IndexInt8 {
public:
    using idx_t = Index::idx_t;

    IndexInt8Flat(int dim, MetricType metric = MetricType::METRIC_L2, int resourceSize = -1);

    ~IndexInt8Flat();

    // unused
    void addImpl(int n, const int8_t *x, const idx_t *ids) override;

    void addVectors(AscendTensor<int8_t, DIMS_2> &rawData) override;

    void getVectors(uint32_t offset, uint32_t num, std::vector<int8_t> &vectors);

    void reset() override;

    inline int getSize() const
    {
        return ntotal;
    }

    inline int getDim() const
    {
        return dims;
    }

    inline int getDistComputeBatch() const
    {
        return distComputeBatch;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<int8_t>>> &getBaseShaped() const
    {
        return baseShaped;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<P>>> &getNormBase() const
    {
        return normBase;
    }

protected:
    void computeNorm(AscendTensor<int8_t, DIMS_2> &rawData);
    P ivecNormL2sqr(const int8_t *x, size_t d);
    void ivecNormsL2sqr(P *nr, const int8_t *x, size_t d, size_t nx);

    void moveNormForward(idx_t srcIdx, idx_t dstIdx);
    void moveShapedForward(idx_t srcIdx, idx_t dstIdx);

    inline void moveVectorForward(idx_t srcIdx, idx_t dstIdx)
    {
        moveNormForward(srcIdx, dstIdx);
        moveShapedForward(srcIdx, dstIdx);
    }

    size_t removeIdsImpl(const IDSelector &sel);
    size_t removeIdsBatch(const std::vector<idx_t> &indices);
    size_t removeIdsRange(idx_t min, idx_t max);
    void removeInvalidData(int oldTotal, int remove);

    size_t calcShapedBaseSize(idx_t totalNum);
    size_t calcNormBaseSize(idx_t totalNum);

protected:
    int distComputeBatch = FLAT_DEFAULT_DIST_COMPUTE_BATCH;
    int devVecCapacity = 0;

    std::vector<std::unique_ptr<DeviceVector<int8_t>>> baseShaped;
    std::vector<std::unique_ptr<DeviceVector<P>>> normBase;

    AscendThreadPool *threadPool = nullptr;
};
} // namespace ascend

#endif // ASCEND_INDEX_INT8_FLAT_INCLUDED
