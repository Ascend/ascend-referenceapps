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

#ifndef ASCEND_INDEX_IVF_INCLUDED
#define ASCEND_INDEX_IVF_INCLUDED

#include <faiss/ascend/AscendIndex.h>
#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>

namespace faiss {
struct IndexIVF;
}  // namespace faiss

namespace faiss {
namespace ascend {
const int IVF_DEFAULT_MEM = static_cast<int>(0x8000000); // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexIVFConfig : public AscendIndexConfig {
    inline AscendIndexIVFConfig() : AscendIndexConfig({ 0, 1, 2, 3 }, IVF_DEFAULT_MEM), useKmeansPP(false)
    {
        SetDefaultClusteringConfig();
    }

    inline AscendIndexIVFConfig(std::initializer_list<int> devices, int resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize), useKmeansPP(false)
    {
        SetDefaultClusteringConfig();
    }

    inline AscendIndexIVFConfig(std::vector<int> devices, int resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize), useKmeansPP(false)
    {
        SetDefaultClusteringConfig();
    }

    inline void SetDefaultClusteringConfig()
    {
        // here we set a low # iterations because this is typically used
        // for large clusterings
        const int niter = 10;
        cp.niter = niter;
    }

    // Configuration for the coarse quantizer object
    AscendIndexConfig flatConfig;

    // whether to use kmeansPP
    bool useKmeansPP;
    
    // clustering parameters for trainQuantizer
    ClusteringParameters cp;
};

class AscendIndexIVF : public AscendIndex {
public:
    AscendIndexIVF(int dims, faiss::MetricType metric,
                   int nlist,
                   AscendIndexIVFConfig config = AscendIndexIVFConfig());

    virtual ~AscendIndexIVF();

    // Returns the number of inverted lists we're managing
    inline int getNumLists() const
    {
        return nlist;
    }

    // Copy what we need from the CPU equivalent
    void copyFrom(const faiss::IndexIVF* index);

    // Copy what we have to the CPU equivalent
    void copyTo(faiss::IndexIVF* index) const;

    // Clears out all inverted lists, but retains the trained information
    void reset() override;

    // Sets the number of list probes per query
    void setNumProbes(int nprobes);

    // Returns our current number of list probes per query
    inline int getNumProbes() const
    {
        return nprobe;
    }

    // reserve memory for the database.
    void reserveMemory(size_t numVecs) override;

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;
    
    // For debugging purposes, return the list length of a particular list
    uint32_t getListLength(int listId) const;

    // For debugging purposes, return the list codes of a particular list
    void getListCodesAndIds(int listId, std::vector<uint8_t>& codes,
                        std::vector<uint32_t>& ids) const;

protected:
    void checkIVFParams();
    
    bool addImplRequiresIDs() const override;

    void initDeviceAddNumMap();

    // train L1 IVF quantizer
    void trainQuantizer(Index::idx_t n, const float* x);

    // update coarse centroid to device
    void updateDeviceCoarseCenter();
    
    // Called from AscendIndex for remove
    size_t removeImpl(const IDSelector& sel) override;

protected:
    // Number of vectors the quantizer contains
    int nlist;

    // top nprobe for quantizer searching
    int nprobe;

    // where the quantizer data stored
    IndexFlat* cpuQuantizer;

    // config
    AscendIndexIVFConfig ivfConfig;

    // centroidId -> feature index number @ device
    // std::unordered_map <Index::idx_t, std::vector<int>> deviceAddNumMap;
    std::vector<std::vector<int>> deviceAddNumMap;
};
}  // namespace ascend
}  // namespace faiss
#endif
