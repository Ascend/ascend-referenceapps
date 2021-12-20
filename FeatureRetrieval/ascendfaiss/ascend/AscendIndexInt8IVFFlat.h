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

#ifndef ASCEND_INDEX_INT8_IVFFLAT_INCLUDED
#define ASCEND_INDEX_INT8_IVFFLAT_INCLUDED

#include <faiss/ascend/AscendIndexInt8IVF.h>
#include <faiss/IndexScalarQuantizer.h>

struct fp16;

namespace faiss {
namespace ascend {
const int INT8_IVFFLAT_DEFAULT_MEM = static_cast<int>(0x8000000); // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexInt8IVFFlatConfig : public AscendIndexInt8IVFConfig {
    inline AscendIndexInt8IVFFlatConfig() : AscendIndexInt8IVFConfig({ 0, 1, 2, 3 }, INT8_IVFFLAT_DEFAULT_MEM) {}

    inline AscendIndexInt8IVFFlatConfig(std::initializer_list<int> devices, int resourceSize = INT8_IVFFLAT_DEFAULT_MEM)
        : AscendIndexInt8IVFConfig(devices, resourceSize)
    {}

    inline AscendIndexInt8IVFFlatConfig(std::vector<int> devices, int resourceSize = INT8_IVFFLAT_DEFAULT_MEM)
        : AscendIndexInt8IVFConfig(devices, resourceSize)
    {}
};

class AscendIndexInt8IVFFlat : public  AscendIndexInt8IVF {
public:
    AscendIndexInt8IVFFlat(int dims, int nlist, faiss::MetricType metric = MetricType::METRIC_L2,
        AscendIndexInt8IVFFlatConfig config = AscendIndexInt8IVFFlatConfig());

    // Construct an index from CPU IndexIVFSQ
    AscendIndexInt8IVFFlat(const faiss::IndexIVFScalarQuantizer *index,
        AscendIndexInt8IVFFlatConfig config = AscendIndexInt8IVFFlatConfig());

    virtual ~AscendIndexInt8IVFFlat();

    void updateCentroids(Index::idx_t n, const int8_t *x) override;

    void updateCentroids(Index::idx_t n, const char *x) override;

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexIVFScalarQuantizer* index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexIVFScalarQuantizer* index) const;

    void copyCodes(const faiss::IndexIVFScalarQuantizer *index);

protected:
    // Called from AscendIndex for add
    void addImpl(int n, const int8_t *x, const Index::idx_t *ids);

protected:
    AscendIndexInt8IVFFlatConfig ivfflatconfig;

private:
    void checkParams();

    void initRpcCtx();

    void clearRpcCtx();

    // get the size of memory every database vector needed to store.
    int getElementSize() const override;
};
} // ascend
} // faiss

#endif // ASCEND_INDEX_INT8_IVFFLAT_INCLUDED
