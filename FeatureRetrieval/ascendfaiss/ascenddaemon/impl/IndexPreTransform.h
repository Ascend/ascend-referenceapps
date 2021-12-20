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

#ifndef ASCEND_INDEXPRETRANSFORM_INCLUDED
#define ASCEND_INDEXPRETRANSFORM_INCLUDED

#include <ascenddaemon/impl/Index.h>
#include <ascenddaemon/impl/VectorTransform.h>

namespace ascend {
class IndexPreTransform : public Index {
public:
    IndexPreTransform(Index *subIndex);

    ~IndexPreTransform();

    void reset() override;

    void prependTransform(VectorTransform *ltrans);

private:
    void addImpl(int n, const float16_t *x, const idx_t *ids) override;

    void searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    size_t removeIdsImpl(const IDSelector &sel) override;

private:
    std::vector<VectorTransform *> chain; // chain of transforms
    Index *index;
};
}

#endif // ASCEND_INDEXIVFPQ_INCLUDED
