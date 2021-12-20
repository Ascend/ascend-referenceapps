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

#ifndef ASCEND_INDEX_IVFSQ_INCLUDED
#define ASCEND_INDEX_IVFSQ_INCLUDED

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/ascend/AscendIndexIVF.h>

struct fp16;

namespace faiss {
namespace ascend {
const int IVFSQ_DEFAULT_TEMP_MEM = static_cast<int>(0x18000000);

struct AscendIndexIVFSQConfig : public AscendIndexIVFConfig {
    inline AscendIndexIVFSQConfig() : AscendIndexIVFConfig({ 0, 1, 2, 3 }, IVFSQ_DEFAULT_TEMP_MEM)
    {
        SetDefaultIVFSQConfig();
    }

    inline AscendIndexIVFSQConfig(std::initializer_list<int> devices, int resourceSize = IVFSQ_DEFAULT_TEMP_MEM)
        : AscendIndexIVFConfig(devices, resourceSize)
    {
        SetDefaultIVFSQConfig();
    }

    inline AscendIndexIVFSQConfig(std::vector<int> devices, int resourceSize = IVFSQ_DEFAULT_TEMP_MEM)
        : AscendIndexIVFConfig(devices, resourceSize)
    {
        SetDefaultIVFSQConfig();
    }

    inline void SetDefaultIVFSQConfig()
    {
        // increase iteration to 16 for better convergence
        // increase max_points_per_centroid to 512 for getting more data to train
        cp.niter = 16; // 16 iterator
        cp.max_points_per_centroid = 512; // 512 points per centroid
    }
};

class AscendIndexIVFSQ : public AscendIndexIVF {
public:
    // Construct an index from CPU IndexIVFSQ
    AscendIndexIVFSQ(const faiss::IndexIVFScalarQuantizer *index,
        AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());

    // Construct an empty index
    AscendIndexIVFSQ(int dims, int nlist,
        faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true,
        AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());

    virtual ~AscendIndexIVFSQ();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexIVFScalarQuantizer *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexIVFScalarQuantizer *index) const;

    void train(Index::idx_t n, const float *x) override;

public:
    // cpu version for training and endcoding
    faiss::ScalarQuantizer sq;

protected:
    // Called from AscendIndex for add/add_with_ids
    void addImpl(int n, const float *x, const Index::idx_t *ids) override;

private:
    void checkParams();

    void initRpcCtx();

    void clearRpcCtx();

    void trainResidualQuantizer(Index::idx_t n, const float *x);

    void updateDeviceSQTrainedValue();

    void copyCodes(const faiss::IndexIVFScalarQuantizer *index);

    void calcPrecompute(const uint8_t *codes, float *compute, size_t n, float *xMem = nullptr);

    int getElementSize() const override;

private:
    AscendIndexIVFSQConfig ivfsqConfig;

    // whether to encode code by residual
    bool byResidual;
};
} // ascend
} // faiss
#endif