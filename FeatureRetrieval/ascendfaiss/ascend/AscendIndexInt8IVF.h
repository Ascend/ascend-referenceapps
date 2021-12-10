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

#ifndef ASCEND_INDEX_INT8_IVF_INCLUDED
#define ASCEND_INDEX_INT8_IVF_INCLUDED

#include <faiss/ascend/AscendIndexInt8.h>
#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>
#include <faiss/IndexScalarQuantizer.h>

namespace faiss {
struct IndexIVF;
}  // namespace faiss

namespace faiss {
namespace ascend {
const int INT8_IVF_DEFAULT_MEM = static_cast<int>(0x8000000); // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexInt8IVFConfig : public AscendIndexInt8Config {
    inline AscendIndexInt8IVFConfig() : AscendIndexInt8Config({ 0, 1, 2, 3 }, INT8_IVF_DEFAULT_MEM), useKmeansPP(false)
    {
        SetDefaultClusteringConfig();
    }

    inline AscendIndexInt8IVFConfig(std::initializer_list<int> devices, int resourceSize = INT8_IVF_DEFAULT_MEM)
        : AscendIndexInt8Config(devices, resourceSize), useKmeansPP(false)
    {
        SetDefaultClusteringConfig();
    }

    inline AscendIndexInt8IVFConfig(std::vector<int> devices, int resourceSize = INT8_IVF_DEFAULT_MEM)
        : AscendIndexInt8Config(devices, resourceSize), useKmeansPP(false)
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
    AscendIndexInt8Config flatConfig;

    // whether to use kmeansPP
    bool useKmeansPP;
    
    // clustering parameters for trainQuantizer
    ClusteringParameters cp;
};

class AscendIndexInt8IVF : public AscendIndexInt8 {
public:
    AscendIndexInt8IVF(int dims, faiss::MetricType metric, int nlist,
        AscendIndexInt8IVFConfig config = AscendIndexInt8IVFConfig());

    virtual ~AscendIndexInt8IVF();

    // Returns the number of inverted lists we're managing
    inline int getNumLists() const
    {
        return nlist;
    }

    // Clears out all inverted lists, but retains the trained information
    void reset();

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
    void getListCodesAndIds(int listId, std::vector<int8_t>& codes,
                        std::vector<uint32_t>& ids) const;

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexIVFScalarQuantizer* index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexIVFScalarQuantizer* index) const;

protected:
    void checkIVFParams();
    
    bool addImplRequiresIDs() const override;

    void initDeviceAddNumMap();

    // train L1 IVF quantizer
    void trainQuantizer(Index::idx_t n, const float* x);

    // update coarse centroid to device
    void updateDeviceCoarseCenter(int8_t *x);
    
    // Called from AscendIndexInt8 for remove
    size_t removeImpl(const IDSelector& sel) override;

protected:
    // Number of vectors the quantizer contains
    int nlist;

    // top nprobe for quantizer searching
    int nprobe;

    // where the quantizer data stored
    IndexFlat* cpuQuantizer;

    // config
    AscendIndexInt8IVFConfig ivfConfig;

    // centroidId -> feature index number @ device
    // std::unordered_map <Index::idx_t, std::vector<int>> deviceAddNumMap;
    std::vector<std::vector<int>> deviceAddNumMap;
};
}  // namespace ascend
}  // namespace faiss
#endif
