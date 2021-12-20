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

#include <faiss/ascend/AscendIndexInt8.h>
#include <faiss/IndexScalarQuantizer.h>

namespace faiss {
namespace ascend {
const int INT8_FLAT_DEFAULT_MEM = static_cast<int>(0x8000000); // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexInt8FlatConfig : public AscendIndexInt8Config {
    inline AscendIndexInt8FlatConfig() {}

    inline AscendIndexInt8FlatConfig(std::initializer_list<int> devices, int resourceSize = INT8_FLAT_DEFAULT_MEM)
        : AscendIndexInt8Config(devices, resourceSize)
    {}

    inline AscendIndexInt8FlatConfig(std::vector<int> devices, int resourceSize = INT8_FLAT_DEFAULT_MEM)
        : AscendIndexInt8Config(devices, resourceSize)
    {}
};

class AscendIndexInt8Flat : public AscendIndexInt8 {
public:
    // Construct an empty instance that can be added to
    AscendIndexInt8Flat(int dims, faiss::MetricType metric = faiss::METRIC_L2,
        AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());

    // Construct an index from CPU IndexSQ
    AscendIndexInt8Flat(const faiss::IndexScalarQuantizer *index,
        AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());

    virtual ~AscendIndexInt8Flat();

    // Returns the number of vectors we contain
    size_t getBaseSize(int deviceId) const;

    // Returns the vectors of we contain
    void getBase(int deviceId, std::vector<int8_t> &xb) const;

    // Returns the index of vector we contain
    void getIdxMap(int deviceId, std::vector<Index::idx_t> &idxMap) const;

    // Clears all vectors from this index
    void reset();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexScalarQuantizer* index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexScalarQuantizer* index) const;

protected:
    // Flat index does not require IDs as there is no storage available for them
    bool addImplRequiresIDs() const;

    // Called from AscendIndex for add
    void addImpl(int n, const int8_t *x, const Index::idx_t *ids);

    size_t removeImpl(const IDSelector &sel);

    void copyCode(const faiss::IndexScalarQuantizer* index);
    void copyByPage(int n, const int8_t *codes, const Index::idx_t *ids);
    void copyImpl(int n, const int8_t *codes, const Index::idx_t *ids);
    void calcAddMap(int n, std::vector<int> &addMap);
    void add2Device(int n, int8_t *codes, const Index::idx_t *ids, std::vector<int> &addMap);

protected:
    void initRpcCtx();

    void clearRpcCtx();

    void getPaged(int deviceId, int n, std::vector<int8_t> &xb) const;

    void getImpl(int deviceId, int offset, int n, int8_t *x) const;

    void searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<uint32_t>> &label, int n, int k, float *distances, Index::idx_t *labels) const;

    void removeSingle(std::vector<std::vector<uint32_t>> &removes, int deviceNum, uint32_t idx);

    void removeIdx(std::vector<std::vector<uint32_t>> &removeMaps);

    int getElementSize() const;

protected:
    AscendIndexInt8FlatConfig int8FlatConfig;
    
    // recorder assign index of each vector
    std::vector<std::vector<Index::idx_t>> idxDeviceMap;
};
} // ascend
} // faiss
#endif // ASCEND_INDEX_INT8_FLAT_INCLUDED