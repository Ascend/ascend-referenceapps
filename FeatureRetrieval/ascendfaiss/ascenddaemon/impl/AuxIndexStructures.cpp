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

#include <ascenddaemon/impl/AuxIndexStructures.h>

namespace ascend {
namespace {
    const uint32_t OFFSET_BIT = 5;
    const uint32_t SHIFT_BIT = 3;
    const uint32_t ID_MASK = 7;
}

IDSelectorRange::IDSelectorRange(idx_t imin, idx_t imax)
    : imin(imin),
      imax(imax)
{
}

bool IDSelectorRange::is_member(idx_t id) const
{
    return id >= imin && id < imax;
}

IDSelectorBatch::IDSelectorBatch(size_t n, const idx_t *indices)
{
    nbits = 0;
    while (n > (1UL << nbits)) nbits++;
    nbits += OFFSET_BIT;

    mask = (1L << nbits) - 1;
    bloom.resize(1UL << (nbits - SHIFT_BIT), 0);
    for (size_t i = 0; i < n; i++) {
        Index::idx_t id = indices[i];
        set.insert(id);
        id &= mask;
        bloom[id >> SHIFT_BIT] |= 1 << (id & ID_MASK);
    }
}

bool IDSelectorBatch::is_member(idx_t i) const
{
    long im = i & mask;
    if (!(bloom[im >> SHIFT_BIT] & (1 << (im & ID_MASK)))) {
        return 0;
    }
    return set.count(i);
}
}  // namespace ascend