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

#ifndef ASCEND_INDEX_IVFFLAT_INCLUDED
#define ASCEND_INDEX_IVFFLAT_INCLUDED

#include <faiss/ascend/AscendIndexIVF.h>
#include <faiss/IndexIVFFlat.h>

struct fp16;

namespace faiss {
namespace ascend {
const int IVFFLAT_DEFAULT_MEM = static_cast<int>(0x8000000); // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexIVFFlatConfig : public AscendIndexIVFConfig {
    inline AscendIndexIVFFlatConfig() : AscendIndexIVFConfig({ 0, 1, 2, 3 }, IVFFLAT_DEFAULT_MEM) {}

    inline AscendIndexIVFFlatConfig(std::initializer_list<int> devices, int resourceSize = IVFFLAT_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize)
    {}

    inline AscendIndexIVFFlatConfig(std::vector<int> devices, int resourceSize = IVFFLAT_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize)
    {}
};

class AscendIndexIVFFlat : public AscendIndexIVF {
public:
    AscendIndexIVFFlat(const faiss::IndexIVFFlat *index, AscendIndexIVFFlatConfig config = AscendIndexIVFFlatConfig());

    AscendIndexIVFFlat(int dims, int nlist, faiss::MetricType metric = MetricType::METRIC_L2,
        AscendIndexIVFFlatConfig config = AscendIndexIVFFlatConfig());

    virtual ~AscendIndexIVFFlat();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexIVFFlat *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexIVFFlat *index) const;

    void train(Index::idx_t n, const float *x) override;

protected:
    // Called from AscendIndex for add/add_with_ids
    void addImpl(int n, const float *x, const Index::idx_t *ids) override;

private:
    void checkParams();

    void initRpcCtx();

    void clearRpcCtx();

    int getElementSize() const override;

    void copyCodes(const faiss::IndexIVFFlat *index);

private:
    AscendIndexIVFFlatConfig ivfflatconfig;
};
} // ascend
} // faiss

#endif // ASCEND_INDEX_IVFFLAT_INCLUDED
