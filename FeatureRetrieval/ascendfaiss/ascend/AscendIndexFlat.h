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

#ifndef ASCEND_INDEX_FLAT_INCLUDED
#define ASCEND_INDEX_FLAT_INCLUDED

#include <faiss/ascend/AscendIndex.h>

namespace faiss {
struct IndexFlat;
struct IndexFlatL2;
} // faiss

namespace faiss {
namespace ascend {
class AscendFlatIndex;

const int FLAT_DEFAULT_MEM = static_cast<int>(0x8000000); // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexFlatConfig : public AscendIndexConfig {
    inline AscendIndexFlatConfig() {}

    inline AscendIndexFlatConfig(std::initializer_list<int> devices, int resourceSize = FLAT_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}

    inline AscendIndexFlatConfig(std::vector<int> devices, int resourceSize = FLAT_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}
};

class AscendIndexFlat : public AscendIndex {
public:
    // Construct from a pre-existing faiss::IndexFlat instance
    AscendIndexFlat(const faiss::IndexFlat *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());

    // Construct an empty instance that can be added to
    AscendIndexFlat(int dims, faiss::MetricType metric, AscendIndexFlatConfig config = AscendIndexFlatConfig());

    virtual ~AscendIndexFlat();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexFlat *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexFlat *index) const;

    // Returns the number of vectors we contain
    size_t getBaseSize(int deviceId) const;

    // Returns the vectors of we contain
    void getBase(int deviceId, std::vector<float> &xb) const;

    // Returns the index of vector we contain
    void getIdxMap(int deviceId, std::vector<Index::idx_t> &idxMap) const;

    // Clears all vectors from this index
    void reset() override;

    // This index is not trained, so this does nothing
    void train(Index::idx_t n, const float *x) override;

protected:
    // Flat index does not require IDs as there is no storage available for them
    bool addImplRequiresIDs() const override;

    // Called from AscendIndex for add
    void addImpl(int n, const float *x, const Index::idx_t *ids) override;

    size_t removeImpl(const IDSelector &sel) override;

private:
    void initRpcCtx();

    void clearRpcCtx();

    void getPaged(int deviceId, int n, std::vector<float> &xb) const;

    void getImpl(int deviceId, int offset, int n, float *x) const;

    void searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<uint32_t>> &label, int n, int k, float *distances, Index::idx_t *labels) const override;

    void removeSingle(std::vector<std::vector<uint32_t>> &removes, int deviceNum, uint32_t idx);

    void removeIdx(std::vector<std::vector<uint32_t>> &removeMaps);

    int getElementSize() const override;

private:
    AscendIndexFlatConfig flatConfig;

    // recorder assign index of each vector
    std::vector<std::vector<Index::idx_t>> idxDeviceMap;
};

// Wrapper around the Ascend implementation that looks like
// faiss::IndexFlatL2; copies over centroid data from a given
// faiss::IndexFlat
class AscendIndexFlatL2 : public AscendIndexFlat {
public:
    // Construct from a pre-existing faiss::IndexFlatL2 instance, copying
    // data over to the given Ascend
    AscendIndexFlatL2(faiss::IndexFlatL2 *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());

    // Construct an empty instance that can be added to
    AscendIndexFlatL2(int dims, AscendIndexFlatConfig config = AscendIndexFlatConfig());

    // Destructor
    ~AscendIndexFlatL2() {}

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(faiss::IndexFlat *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexFlat *index);
};
} // ascend
} // faiss
#endif // ASCEND_INDEX_FLAT_INCLUDED