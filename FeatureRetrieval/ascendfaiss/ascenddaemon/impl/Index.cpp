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

#include <ascenddaemon/impl/Index.h>
#include <ascenddaemon/utils/Limits.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/utils/AscendAssert.h>
#include <ascenddaemon/StandardAscendResources.h>

namespace ascend {
namespace {
const int KB = 1024;
// Default size for which we page add or search
const int ADD_PAGE_SIZE_MB = 256;
const int ADD_PAGE_SIZE = ADD_PAGE_SIZE_MB * KB * KB;
// Or, maximum number of vectors to consider per page of add or search
const int ADD_VEC_SIZE = 0x80000;
const int FLAG_ALIGN_SIZE = 32;
}

Index::Index(ascend::Index::idx_t d, int resourceSize) : dims(d), ntotal(0), isTrained(false)
{
    if (resourceSize == 0) {
        resources.noTempMemory();
    } else if (resourceSize > 0) {
        resources.setTempMemory(resourceSize);
    }

    // resourceSize < 0 means use default mem configure
    resources.initialize();
}

Index::~Index() {}

void Index::train(idx_t n, const float16_t *x)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
}

void Index::add(idx_t n, const float16_t *x)
{
    return addWithIds(n, x, nullptr);
}

void Index::addWithIds(Index::idx_t n, const float16_t *x, const Index::idx_t *xids)
{
    ASCEND_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    ASCEND_THROW_IF_NOT_MSG(this->isTrained, "Index not trained");
    ASCEND_THROW_IF_NOT_FMT(n <= std::numeric_limits<Index::idx_t>::max(),
                            "indices exceeds max(%d)", std::numeric_limits<Index::idx_t>::max());
    if (n == 0) {
        return;
    }

    std::vector<Index::idx_t> tmpIds;
    if (xids == nullptr && addImplRequiresIDs()) {
        tmpIds = std::vector<Index::idx_t>(n);

        for (Index::idx_t i = 0; i < n; ++i) {
            tmpIds[i] = this->ntotal + i;
        }

        xids = tmpIds.data();
    }

    return addPaged(n, x, xids);
}

bool Index::addImplRequiresIDs() const
{
    return true;
}

size_t Index::removeIds(const IDSelector &sel)
{
    return removeIdsImpl(sel);
}

void Index::search(idx_t n, const float16_t *x, idx_t k, float16_t *distances, idx_t *labels)
{
    ASCEND_THROW_IF_NOT_MSG(x, "x can not be nullptr");
    ASCEND_THROW_IF_NOT_MSG(distances, "distance can not be nullptr");
    ASCEND_THROW_IF_NOT_MSG(labels, "labels can not be nullptr.");
    ASCEND_THROW_IF_NOT_MSG(this->isTrained, "Index not trained");
    ASCEND_THROW_IF_NOT_FMT(n <= (Index::idx_t)std::numeric_limits<int>::max(),
                            "indices exceeds max(%d)", std::numeric_limits<int>::max());

    if (n == 0 || k == 0) {
        return;
    }

    return searchPaged(n, x, k, distances, labels);
}

void Index::addPaged(int n, const float16_t *x, const Index::idx_t *ids)
{
    if (n > 0) {
        size_t totalSize = (size_t)n * this->dims * sizeof(float16_t);

        if (totalSize > ADD_PAGE_SIZE || n > ADD_VEC_SIZE) {
            // How many vectors fit into ADD_PAGE_SIZE?
            int maxNumVecsForPageSize = ADD_PAGE_SIZE / (this->dims * sizeof(float16_t));

            // always add at least 1 vector, if we have huge vectors
            maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, 1);

            int tileSize = std::min(n, maxNumVecsForPageSize);

            for (int i = 0; i < n; i += tileSize) {
                int curNum = std::min(tileSize, n - i);
                addImpl(curNum, x + i * this->dims, ids ? (ids + i) : nullptr);
            }
        } else {
            addImpl(n, x, ids);
        }
    }
}

void Index::searchPaged(int n, const float16_t *x, int k, float16_t *distance, idx_t *labels)
{
    size_t size = searchPageSizes.size();
    if (n > 1 && size > 0) {
        int searched = 0;
        for (size_t i = 0; i < size; i++) {
            int pageSize = searchPageSizes[i];
            if ((n - searched) >= pageSize) {
                int page = (n - searched) / pageSize;
                for (int j = 0; j < page; j++) {
                    searchImpl(pageSize, x + searched * this->dims, k, distance + searched * k, labels + searched * k);
                    searched += pageSize;
                }
            }
        }

        for (int i = searched; i < n; i++) {
            searchImpl(1, x + i * this->dims, k, distance + i * k, labels + i * k);
        }
    } else {
        searchImpl(n, x, k, distance, labels);
    }
}

void Index::runDistanceCompute(
    AscendTensor<float16_t, DIMS_2>& queryVecs,
    AscendTensor<float16_t, DIMS_4>& shapedData,
    AscendTensor<float16_t, DIMS_1>& norms,
    AscendTensor<float16_t, DIMS_2>& outDistances,
    AscendTensor<uint16_t, DIMS_1>& flag,
    aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = queryVecs.getSize(0);
    if (distComputeOps.find(batch) != distComputeOps.end()) {
        op = distComputeOps[batch].get();
    }
    ASCEND_ASSERT(op);

    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(queryVecs.data(), queryVecs.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(shapedData.data(), shapedData.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(norms.data(), norms.getSizeInBytes()));

    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    op->exec(distOpInput, distOpOutput, stream);

    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();

    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
}

void Index::resetDistCompOperator(int numLists)
{
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceComputeFlat");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(numLists, CUBE_ALIGN_SIZE),
            utils::divUp(dims, CUBE_ALIGN_SIZE), CUBE_ALIGN_SIZE, (int64_t)CUBE_ALIGN_SIZE });
        std::vector<int64_t> preNormsShape({ numLists });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> flagShape({ FLAG_ALIGN_SIZE });
        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, preNormsShape.size(), preNormsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = std::make_unique<AscendOperator>(desc);
    };

    for (auto batch : searchPageSizes) {
        distComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        distCompOpReset(distComputeOps[batch], batch);
    }
}

float16_t Index::fvecNormL2sqr(const float16_t *x, size_t d)
{
    size_t i;
    double res = 0;
    for (i = 0; i < d; i++) {
        res += x[i] * x[i];
    }
    return res;
}

void Index::fvecNormsL2sqr(float16_t *nr, const float16_t *x, size_t d, size_t nx)
{
#pragma omp parallel for
    for (size_t i = 0; i < nx; i++) {
        nr[i] = fvecNormL2sqr(x + i * d, d);
    }
}

void Index::reserveMemory(size_t numVecs)
{
    VALUE_UNUSED(numVecs);
    ASCEND_THROW_MSG("reserveMemory not implemented for this type of index");
}

size_t Index::reclaimMemory()
{
    ASCEND_THROW_MSG("reclaimMemory not implemented for this type of index");
    return 0;
}
} // namespace ascend
