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

#include <ascenddaemon/impl/IndexInt8.h>
#include <ascenddaemon/utils/Limits.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/utils/AscendAssert.h>

namespace ascend {
namespace {
const int KB = 1024;
// Default size for which we page add or search
const int ADD_PAGE_SIZE_MB = 256;
const int ADD_PAGE_SIZE = ADD_PAGE_SIZE_MB * KB * KB;
// Or, maximum number of vectors to consider per page of add or search
const int ADD_VEC_SIZE = 0x80000;
}

IndexInt8::IndexInt8(idx_t d, MetricType metric, int resourceSize)
    : dims(d), 
      ntotal(0), 
      metricType(metric),
      isTrained(false),
      maskData(nullptr),
      maskSearchedOffset(0)
{
    if (resourceSize == 0) {
        resources.noTempMemory();
    } else if (resourceSize > 0) {
        resources.setTempMemory(resourceSize);
    }

    // resourceSize < 0 means use default mem configure
    resources.initialize();
}

IndexInt8::~IndexInt8() {}

void IndexInt8::train(idx_t n, const int8_t *x)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
}

void IndexInt8::add(idx_t n, const int8_t *x)
{
    return addWithIds(n, x, nullptr);
}

void IndexInt8::addWithIds(IndexInt8::idx_t n, const int8_t *x, const IndexInt8::idx_t *xids)
{
    ASCEND_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    ASCEND_THROW_IF_NOT_MSG(this->isTrained, "Index not trained");
    ASCEND_THROW_IF_NOT_FMT(n <= std::numeric_limits<IndexInt8::idx_t>::max(),
                            "indices exceeds max(%d)", std::numeric_limits<IndexInt8::idx_t>::max());
    if (n == 0) {
        return;
    }

    std::vector<IndexInt8::idx_t> tmpIds;
    if (xids == nullptr && addImplRequiresIDs()) {
        tmpIds = std::vector<IndexInt8::idx_t>(n);

        for (IndexInt8::idx_t i = 0; i < n; ++i) {
            tmpIds[i] = this->ntotal + i;
        }

        xids = tmpIds.data();
    }

    return addPaged(n, x, xids);
}

void IndexInt8::addVectors(AscendTensor<int8_t, DIMS_2> &rawData)
{
    VALUE_UNUSED(rawData);
}

bool IndexInt8::addImplRequiresIDs() const
{
    return true;
}

size_t IndexInt8::removeIds(const IDSelector &sel)
{
    return removeIdsImpl(sel);
}

void IndexInt8::search(idx_t n, const int8_t *x, idx_t k, float16_t *distances, idx_t *labels, uint8_t *mask)
{
    ASCEND_THROW_IF_NOT_MSG(x, "x can not be nullptr");
    ASCEND_THROW_IF_NOT_MSG(distances, "distance can not be nullptr");
    ASCEND_THROW_IF_NOT_MSG(labels, "labels can not be nullptr.");
    ASCEND_THROW_IF_NOT_MSG(this->isTrained, "Index not trained");
    ASCEND_THROW_IF_NOT_FMT(n <= (IndexInt8::idx_t)std::numeric_limits<int>::max(),
                            "indices exceeds max(%d)", std::numeric_limits<int>::max());

    if (n == 0 || k == 0) {
        return;
    }

    this->maskData = mask;
    this->maskSearchedOffset = 0;

    return searchPaged(n, x, k, distances, labels);
}

void IndexInt8::addPaged(int n, const int8_t *x, const IndexInt8::idx_t *ids)
{
    if (n > 0) {
        size_t totalSize = (size_t)n * this->dims * sizeof(int8_t);

        if (totalSize > ADD_PAGE_SIZE || n > ADD_VEC_SIZE) {
            // How many vectors fit into ADD_PAGE_SIZE?
            int maxNumVecsForPageSize = ADD_PAGE_SIZE / (this->dims * sizeof(int8_t));

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

void IndexInt8::searchPaged(int n, const int8_t *x, int k, float16_t *distance, idx_t *labels)
{
    size_t size = searchPageSizes.size();
    if (n > 1 && size > 0) {
        int searched = 0;
        for (size_t i = 0; i < size; i++) {
            int pageSize = searchPageSizes[i];
            if ((n - searched) < pageSize) {
                continue;
            }

            int page = (n - searched) / pageSize;
            for (int j = 0; j < page; j++) {
                searchImpl(pageSize, x + searched * this->dims, k, distance + searched * k, labels + searched * k);
                searched += pageSize;
            }
        }

        for (int i = searched; i < n; i++) {
            searchImpl(1, x + i * this->dims, k, distance + i * k, labels + i * k);
        }
    } else {
        searchImpl(n, x, k, distance, labels);
    }
}

void IndexInt8::reserveMemory(size_t numVecs)
{
    VALUE_UNUSED(numVecs);
    ASCEND_THROW_MSG("reserveMemory not implemented for this type of index");
}

size_t IndexInt8::reclaimMemory()
{
    ASCEND_THROW_MSG("reclaimMemory not implemented for this type of index");
    return 0;
}
} // namespace ascend
