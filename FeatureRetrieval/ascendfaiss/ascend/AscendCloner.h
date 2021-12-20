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

#ifndef ASCEND_CLONER_INCLUDED
#define ASCEND_CLONER_INCLUDED

#include <faiss/Index.h>
#include <faiss/clone_index.h>
#include <faiss/ascend/AscendClonerOptions.h>
#include <faiss/ascend/AscendIndexInt8.h>
#include <initializer_list>
#include <vector>

namespace faiss {
namespace ascend {
// Cloner specialized for Ascend -> CPU
struct ToCPUCloner : faiss::Cloner {
    void merge_index(Index *dst, Index *src, bool successive_ids);
    Index *clone_Index(const Index *index) override;
    Index *clone_IndexInt8(const AscendIndexInt8 *index);
};

// Cloner specialized for CPU -> 1 Ascend
struct ToAscendCloner : faiss::Cloner, AscendClonerOptions {
    std::vector<int> devices;

    ToAscendCloner(std::initializer_list<int> devs, const AscendClonerOptions &options);
    ToAscendCloner(std::vector<int> devs, const AscendClonerOptions &options);
    Index *clone_Index(const Index *index) override;
    AscendIndexInt8 *clone_IndexInt8(const Index *index);
};

faiss::Index *index_ascend_to_cpu(const faiss::Index *ascend_index);

faiss::Index *index_cpu_to_ascend(std::initializer_list<int> devices,
                                  const faiss::Index *index, const AscendClonerOptions *options = nullptr);
faiss::Index *index_cpu_to_ascend(std::vector<int> devices,
                                  const faiss::Index *index, const AscendClonerOptions *options = nullptr);

// handle index int8 which not inherited from faiss::Index
faiss::Index *index_int8_ascend_to_cpu(const AscendIndexInt8 *index);

AscendIndexInt8 *index_int8_cpu_to_ascend(std::initializer_list<int> devices, const faiss::Index *index, 
                                          const AscendClonerOptions *options = nullptr);
AscendIndexInt8 *index_int8_cpu_to_ascend(std::vector<int> devices, const faiss::Index *index,
                                          const AscendClonerOptions *options = nullptr);
}  // namespace ascend
}  // namespace faiss
#endif  // ASCEND_CLONER_INCLUDED
