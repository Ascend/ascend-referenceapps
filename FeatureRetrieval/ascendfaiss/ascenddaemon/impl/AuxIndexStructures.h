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

#ifndef ASCEND_AUXINDEXSTRUCTURES_INCLUDED
#define ASCEND_AUXINDEXSTRUCTURES_INCLUDED

#include <ascenddaemon/impl/Index.h>
#include <unordered_set>
#include <vector>

namespace ascend {
/// Encapsulates a set of ids to remove.
struct IDSelector {
    typedef Index::idx_t idx_t;
    virtual bool is_member(idx_t id) const = 0;
    virtual ~IDSelector()
    {
    }
};

/// remove ids between [imin, imax).
struct IDSelectorRange : IDSelector {
    idx_t imin, imax;

    IDSelectorRange(idx_t imin, idx_t imax);
    bool is_member(idx_t id) const override;
    ~IDSelectorRange() override
    {
    }
};

/** Remove ids from a set. Repetitions of ids in the indices set  passed to the
 * constructor does not hurt performanc. The hash function used for the bloom filter
 * and GCC's implementation of unordered_set are just the least significant bits of
 * the id. This works fine for random ids or ids in sequences but will produce many
 * hash collisions if lsb's are always the same */
struct IDSelectorBatch : IDSelector {
    std::unordered_set<idx_t> set;

    typedef unsigned char uint8_t;
    std::vector<uint8_t> bloom;
    int nbits;
    idx_t mask;

    IDSelectorBatch(size_t n, const idx_t *indices);
    bool is_member(idx_t id) const override;
    ~IDSelectorBatch() override
    {
    }
};
}  // namespace ascend
#endif  // ASCEND_AUXINDEXSTRUCTURES_INCLUDED
