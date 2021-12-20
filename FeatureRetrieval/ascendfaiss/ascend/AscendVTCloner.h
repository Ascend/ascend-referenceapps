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

#ifndef ASCEND_VECTOR_TRANSFORM_CLONER_INCLUDED
#define ASCEND_VECTOR_TRANSFORM_CLONER_INCLUDED

#include <faiss/VectorTransform.h>
#include <faiss/clone_index.h>
#include <faiss/ascend/AscendClonerOptions.h>
#include <vector>

namespace faiss {
namespace ascend {
// Cloner specialized for Ascend -> CPU
struct ToCPUVTCloner : faiss::Cloner {
    VectorTransform *clone_VectorTransform(const VectorTransform *vtrans) override;
};

// Cloner specialized for CPU -> 1 Ascend
struct ToAscendVTCloner : faiss::Cloner, AscendClonerOptions {
    std::vector<int> devices;
    ToAscendVTCloner(std::initializer_list<int> devs, const AscendClonerOptions &options);
    ToAscendVTCloner(std::vector<int> devs, const AscendClonerOptions &options);
    VectorTransform *clone_VectorTransform(const faiss::VectorTransform *vtrans) override;
};

faiss::VectorTransform *vtrans_ascend_to_cpu(const faiss::VectorTransform *ascend_vtrans);

faiss::VectorTransform *vtrans_cpu_to_ascend(std::initializer_list<int> devices, const faiss::VectorTransform *vtrans,
    const AscendClonerOptions *options = nullptr);
faiss::VectorTransform *vtrans_cpu_to_ascend(std::vector<int> devices, const faiss::VectorTransform *vtrans,
    const AscendClonerOptions *options = nullptr);
}
}
#endif // ASCEND_VECTOR_TRANSFORM_CLONER_INCLUDED
