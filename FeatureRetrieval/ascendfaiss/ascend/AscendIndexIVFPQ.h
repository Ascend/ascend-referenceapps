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

#ifndef ASCEND_INDEX_IVFPQ_INCLUDED
#define ASCEND_INDEX_IVFPQ_INCLUDED

#include <faiss/ascend/AscendIndexIVF.h>
#include <faiss/impl/ProductQuantizer.h>

namespace faiss {
struct IndexIVFPQ;
}

struct fp16;

namespace faiss {
namespace ascend {
const int IVFPQ_DEFAULT_MEM = static_cast<int>(0x8000000); // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexIVFPQConfig : public AscendIndexIVFConfig {
    inline AscendIndexIVFPQConfig() : AscendIndexIVFConfig({ 0, 1, 2, 3 }, IVFPQ_DEFAULT_MEM) {}

    inline AscendIndexIVFPQConfig(std::initializer_list<int> devices, int resourceSize = IVFPQ_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize)
    {}

    inline AscendIndexIVFPQConfig(std::vector<int> devices, int resourceSize = IVFPQ_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize)
    {}
};

class AscendIndexIVFPQ : public AscendIndexIVF {
public:
    // Construct an index from CPU IndexIVFPQ
    AscendIndexIVFPQ(const faiss::IndexIVFPQ *index, AscendIndexIVFPQConfig config = AscendIndexIVFPQConfig());

    // Construct an empty index
    AscendIndexIVFPQ(int dims, int nlist, int subQuantizers, int bitsPerCode, faiss::MetricType metric,
        AscendIndexIVFPQConfig config = AscendIndexIVFPQConfig());

    virtual ~AscendIndexIVFPQ();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexIVFPQ *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexIVFPQ *index) const;

    // Return the number of sub-quantizers we are using
    int getNumSubQuantizers() const;

    // Return the number of bits per PQ code
    int getBitsPerCode() const;

    // Return the number of centroids per PQ code (2^bits per code)
    int getCentroidsPerSubQuantizer() const;

    void train(Index::idx_t n, const float *x) override;

    // For debugging purposes, return the list length of a particular
    // list
    uint32_t getListLength(int listId, int deviceId) const;

    // For debugging purposes, return the list codes of a particular
    // list
    void getListCodesAndIds(int listId, int deviceId, std::vector<uint8_t> &codes, std::vector<uint32_t> &ids) const;

protected:
    // Called from AscendIndex for add/add_with_ids
    void addImpl(int n, const float *x, const Index::idx_t *ids) override;

private:
    void checkParams();

    void initRpcCtx();

    void clearRpcCtx();

    void trainResidualQuantizer(Index::idx_t n, const float *x);

    void updateDevicePQCenter();

    bool isSupportedPQCodeLength(int size);

    bool isSupportedPQSubDim(int subDim);

    int getElementSize() const override;

private:
    AscendIndexIVFPQConfig ivfpqConfig;

    // Number of sub-quantizers per encoded vector
    int subQuantizersCnt;

    // Bits count per sub-quantizer code
    int bitsCntPerCode;

    // cpu version for pqcentroids training and endcoding
    faiss::ProductQuantizer pqData;
};
} // namespace ascend
} // namespace faiss

#endif
