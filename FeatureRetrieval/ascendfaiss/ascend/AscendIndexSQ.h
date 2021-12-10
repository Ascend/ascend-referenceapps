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

#ifndef ASCEND_INDEX_SQ_INCLUDED
#define ASCEND_INDEX_SQ_INCLUDED

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/ascend/AscendIndex.h>

namespace faiss {
namespace ascend {
const int SQ_DEFAULT_MEM = static_cast<int>(0x8000000); // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexSQConfig : public AscendIndexConfig {
    inline AscendIndexSQConfig() : AscendIndexConfig({ 0, 1, 2, 3 }, SQ_DEFAULT_MEM) {}

    inline AscendIndexSQConfig(std::initializer_list<int> devices, int resourceSize = SQ_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}

    inline AscendIndexSQConfig(std::vector<int> devices, int resourceSize = SQ_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}
};

class AscendIndexSQ : public AscendIndex {
public:
    // Construct an index from CPU IndexSQ
    AscendIndexSQ(const faiss::IndexScalarQuantizer* index,
        AscendIndexSQConfig config = AscendIndexSQConfig());

    AscendIndexSQ(int dims, 
        faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_L2,
        AscendIndexSQConfig config = AscendIndexSQConfig());

    virtual ~AscendIndexSQ();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexScalarQuantizer* index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexScalarQuantizer* index) const;

    // Returns the codes of we contain
    void getBase(int deviceId, std::vector<uint8_t>& xb) const;

    // Returns the number of codes we contain
    size_t getBaseSize(int deviceId) const;

    // Returns the index of vector we contain
    void getIdxMap(int deviceId, std::vector<Index::idx_t>& idxMap) const;

    // Clears all codes and preCompute from this index
    void reset() override;

    void train(Index::idx_t n, const float* x) override;

public:
    // cpu version for training and endcoding
    faiss::ScalarQuantizer sq;

protected:
    /// SQ index does not require IDs as there is no storage available for them
    bool addImplRequiresIDs() const override;

    /// Called from AscendIndex for add
    void addImpl(int n, const float* x, const Index::idx_t* ids) override;

    void searchPostProcess(size_t devices, std::vector<std::vector<float>>& dist, 
                                    std::vector<std::vector<uint32_t>>& label, int n, int k, 
                                    float* distances, Index::idx_t* labels) const override;

    size_t removeImpl (const IDSelector &sel) override;

private:
    void checkParams();

    void initRpcCtx();

    void clearRpcCtx();

    void updateDeviceSQTrainedValue();

    void copyCode(const faiss::IndexScalarQuantizer* index);

    void copyByPage(int n, const uint8_t* codes, const Index::idx_t* ids);

    void copyImpl(int n, const uint8_t* codes, const Index::idx_t* ids);

    void calcAddMap(int n, std::vector<int> &addMap);

    void add2Device(int n, uint8_t* codes, const Index::idx_t* ids, float* preCompute, std::vector<int> &addMap); 

    void calcPreCompute(const uint8_t* codes, float* compute,
        size_t n, float* xMem = nullptr);

    void getPaged(int deviceId, int n, std::vector<uint8_t>& codes) const;

    void getPagedFast(int deviceId, size_t n, std::vector<uint8_t>& codes) const;

    void getImpl(int deviceId, int offset, int n, uint8_t* code) const;

    void removeSingle(std::vector<std::vector<uint32_t>>& removes,
        int deviceNum, uint32_t idx);

    int getElementSize() const override;

    // remove the idx, this need consistent with the removeImpl in the device
    void removeIdx(std::vector<std::vector<uint32_t>> &removeMaps);

private:
    AscendIndexSQConfig sqConfig;

    // recorder assign index of each vector
    std::vector<std::vector<Index::idx_t>> idxDeviceMap;
};
} // ascend
} // faiss
#endif