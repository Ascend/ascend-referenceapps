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

#ifndef ASCEND_INDEXSQ_INCLUDED
#define ASCEND_INDEXSQ_INCLUDED

#include <ascenddaemon/impl/Index.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/AscendThreadPool.h>
#include <memory>

namespace ascend {
class IndexSQ : public Index {
public:
    IndexSQ(int dim, int resource = -1);

    ~IndexSQ();

    void reset() override;

    void addVectors(size_t numVecs, const uint8_t *data, const float *preCompute);

    inline int getSize() const
    {
        return ntotal;
    }

    void getVectors(uint32_t offset, uint32_t num, std::vector<uint8_t> &vectors);
    void getVectors(uint32_t offset, uint32_t num, uint8_t *vectors);

    inline int getDim() const
    {
        return dims;
    }

    inline uint32_t getSendBatch() const
    {
        return SEND_BATCH;
    }

    inline int getDistComputeBatch() const
    {
        return distComputeBatch;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<uint8_t>>> &getCodes() const
    {
        return codes;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<float>>> &getPreCompute() const
    {
        return preCompute;
    }

    void updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin, AscendTensor<float16_t, DIMS_1> &trainedDiff);

protected:
    void addImpl(int n, const float16_t *x, const idx_t *ids) override;

    virtual void searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t* labels) = 0;

    void saveCodes(int vecIndex, int dVecIndex, int numVecs, const uint8_t *data);

    void savePreCompute(int vecIndex, int dVecIndex, int numVecs, const float *preCompute);

    size_t removeIdsImpl(const IDSelector &sel) override;
    size_t removeIdsBatch(const std::vector<idx_t> &indices);
    size_t removeIdsRange(idx_t min, idx_t max);

    void moveCodesForward(idx_t srcIdx, idx_t destIdx);
    void movePreComputeForward(idx_t srcIdx, idx_t destIdx);
    inline void moveVectorForward(idx_t srcIdx, idx_t destIdx)
    {
        moveCodesForward(srcIdx, destIdx);
        movePreComputeForward(srcIdx, destIdx);
    }
    void releaseUnusageSpace(int oldTotal, int remove);

protected:
    static const uint32_t SEND_BATCH = 16384;          // sq send batch for each channel

    int distComputeBatch;
    int devVecCapacity;
    int burstsOfComputeBatch;

    AscendTensor<float16_t, DIMS_1> vMin;
    AscendTensor<float16_t, DIMS_1> vDiff;

    AscendThreadPool *threadPool;

    std::vector<std::unique_ptr<DeviceVector<uint8_t>>> codes;
    // precompute Data listc
    std::vector<std::unique_ptr<DeviceVector<float>>> preCompute;

    TopkOp<std::less<>, std::less_equal<>, float16_t, false> topkOp;
};
} // namespace ascend

#endif // ASCEND_INDEXSQ_INCLUDED
