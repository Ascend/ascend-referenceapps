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

#ifndef ASCEND_INDEXFLAT_INCLUDED
#define ASCEND_INDEXFLAT_INCLUDED

#include <vector>

#include <ascenddaemon/impl/Index.h>
#include <ascenddaemon/utils/TopkOp.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/DeviceVectorInl.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/AscendThreadPool.h>

namespace ascend {
class IndexFlat : public Index {
public:
    IndexFlat(int dim, int resourceSize = -1);

    ~IndexFlat();

    // unused
    void addImpl(int n, const float16_t *x, const idx_t *ids) override;

    virtual void addVectors(AscendTensor<float16_t, DIMS_2> &rawData);

    void searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    void reset();

    inline int getSize() const
    {
        return ntotal;
    }

    void getVectors(uint32_t offset, uint32_t num, std::vector<float16_t> &vectors);

    // for debug
    inline int getDim() const
    {
        return dims;
    }

    inline int getDistComputeBatch() const
    {
        return distComputeBatch;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<float16_t>>> &getBaseShaped() const
    {
        return baseShaped;
    }

protected:
    virtual void searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
        AscendTensor<float16_t, DIMS_2> &outDistance, AscendTensor<uint32_t, DIMS_2> &outIndices) = 0;

    void moveShapedForward(idx_t srcIdx, idx_t dstIdx);
    inline virtual void moveVectorForward(idx_t srcIdx, idx_t dstIdx)
    {
        moveShapedForward(srcIdx, dstIdx);
    }

    size_t removeIdsImpl(const IDSelector &sel) override;
    size_t removeIdsBatch(const std::vector<idx_t> &indices);
    size_t removeIdsRange(idx_t min, idx_t max);
    virtual void removeInvalidData(int oldTotal, int remove);

    size_t calcShapedBaseSize(idx_t totalNum);

protected:
    int distComputeBatch;
    int devVecCapacity;
    int burstsOfComputeBatch;

    std::vector<std::unique_ptr<DeviceVector<float16_t>>> baseShaped;

    AscendThreadPool *threadPool;
};
} // namespace ascend

#endif // ASCEND_INDEXFLAT_INCLUDED
