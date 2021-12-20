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

#include <algorithm>
#include <ascenddaemon/impl/IndexPreTransform.h>

namespace ascend {
IndexPreTransform::IndexPreTransform(Index *subIndex) 
    : Index(subIndex->*&IndexPreTransform::dims, subIndex->getResources()->getResourceSize()),
      index(subIndex)
{
    isTrained = true;
}

IndexPreTransform::~IndexPreTransform() {}

void IndexPreTransform::prependTransform(VectorTransform *ltrans)
{
    ASCEND_THROW_IF_NOT(dims == ltrans->dimOut);
    chain.insert(chain.begin(), ltrans);
    dims = ltrans->dimIn;
}

void IndexPreTransform::reset() {}

void IndexPreTransform::addImpl(int n, const float16_t *x, const idx_t *ids)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);
}

void IndexPreTransform::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
{
    size_t chainSize = chain.size();
    if (chainSize < 1) {
        index->search(n, x, k, distances, labels);
        return;
    }

    int maxDim = 0;
    for_each(chain.begin(), chain.end(), [&maxDim](auto& item) {
        maxDim = std::max(maxDim, item->dimIn);
        maxDim = std::max(maxDim, item->dimOut);
    });
    
    auto stream = index->resources.getDefaultStream();
    auto& mem = index->resources.getMemoryManager();
    AscendTensor<float16_t, DIMS_1> src(mem, { n * maxDim }, stream);
    AscendTensor<float16_t, DIMS_1> dst(mem, { n * maxDim }, stream);

    float16_t *xt = src.data();
    float16_t *xtt = dst.data();

    chain[0]->applyAsync(n, x, xtt, stream);
    for (size_t i = 1; i < chainSize; i++) {
        std::swap(xt, xtt);
        chain[i]->applyAsync(n, xt, xtt, stream);
    }
    index->search(n, xtt, k, distances, labels);
}

size_t IndexPreTransform::removeIdsImpl(const IDSelector &sel)
{
    return 0;
}
}
