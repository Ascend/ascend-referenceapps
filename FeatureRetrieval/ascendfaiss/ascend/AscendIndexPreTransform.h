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

#ifndef ASCEND_INDEX_PRETRANSFORM_INCLUDED
#define ASCEND_INDEX_PRETRANSFORM_INCLUDED

#include <faiss/ascend/AscendIndex.h>
#include <faiss/ascend/AscendVectorTransform.h>
namespace faiss {
struct IndexPreTransform;
}

namespace faiss {
namespace ascend {
/* * Index that applies a LinearTransform transform on vectors before
 * handing them over to a sub-index */
class AscendIndexPreTransform : public AscendIndex {
public:
    // Construct an index from CPU IndexPreTransform
    AscendIndexPreTransform(const faiss::IndexPreTransform *index, AscendIndexConfig config = AscendIndexConfig());

    AscendIndexPreTransform(AscendIndex *index);

    virtual ~AscendIndexPreTransform();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexPreTransform *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexPreTransform *index) const;

    // reserve memory for the database.
    void reserveMemory(size_t numVecs) override;

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;

    // prepend vector transform to the PreTransform chain
    template<class Transform, class... Args>
    void prependTransform(Args... args);

    void train(idx_t n, const float *x) override;

    const float *applyChain(idx_t n, const float *x) const;

    void reset() override;

    // get the subindex
    inline const Index *getIndex()
    {
        return index;
    }

    // get the subindex
    inline const Index *getIndex() const
    {
        return index;
    }

public:
    bool ownFields; // whether pointers are deleted in destructor

protected:
    // Called from AscendIndex for add/add_with_ids
    void addImpl(int n, const float *x, const Index::idx_t *ids) override;

    // Does addImpl_ require IDs? If so, and no IDs are provided, we will
    // generate them sequentially based on the order in which the IDs are added
    bool addImplRequiresIDs() const override;

    void searchPostProcess(size_t devices,
                           std::vector<std::vector<float>> &dist,
                           std::vector<std::vector<uint32_t>> &label,
                           int n,
                           int k,
                           float *distances,
                           Index::idx_t *labels) const override;

    // Called from AscendIndex for remove
    size_t removeImpl(const IDSelector &sel) override;

    int getElementSize() const override;

private:
    void initRpcCtx();
    void clearRpcCtx();
    void prependRpcChain(AscendVectorTransform *ltrans);

private:
    AscendIndex *index; // the sub-index
    std::vector<AscendVectorTransform *> chain; // chain of transforms
};

template<class Transform, class... Args> 
void AscendIndexPreTransform::prependTransform(Args... args)
{
    AscendVectorTransform *ltrans = new Transform(args..., indexConfig.deviceList);
    FAISS_THROW_IF_NOT(ltrans->d_out == this->d);
    this->is_trained = this->is_trained && ltrans->is_trained;
    this->chain.insert(this->chain.begin(), ltrans);
    this->d = ltrans->d_in;
    this->prependRpcChain(ltrans);
}
}
}
#endif
