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

#include <faiss/ascend/AscendAutoTune.h>
#include <faiss/ascend/AscendIndexPreTransform.h>
#include <faiss/ascend/AscendIndexIVF.h>

namespace faiss {
namespace ascend {
// auto tune k limitation
const size_t MAX_K_SELECTION = 1000;

// max shift bits
const size_t NPROBE_SHIFT_MASK_MAX = 12;


#define DC_CONST(classname) \
    auto ix = dynamic_cast<const classname *>(index)

#define DC(classname) \
    auto ix = dynamic_cast<classname *>(index)

void AscendParameterSpace::initialize(const Index* index)
{
    if (DC_CONST(AscendIndexPreTransform)) {
        index = ix->getIndex();
    }
    
    if (DC_CONST(AscendIndexIVF)) {
        ParameterRange& pr = add_range("nprobe");
        for (size_t i = 0; i < NPROBE_SHIFT_MASK_MAX; i++) {
            size_t nprobe = 1 << i;
            if (static_cast<int>(nprobe) >= ix->getNumLists() || 
                nprobe > MAX_K_SELECTION) {
                break;
            }
            
            pr.values.push_back(nprobe);
        }
    }
}

void AscendParameterSpace::set_index_parameter(Index* index, 
    const std::string& name, double val) const
{
    if (DC (AscendIndexPreTransform)) {
        set_index_parameter (const_cast<Index *>(ix->getIndex()), name, val);
        return;
    }
    
    if (name == "nprobe") {
        if (DC(AscendIndexIVF)) {
            ix->setNumProbes(int(val));
            return;
        }
    }

    ParameterSpace::set_index_parameter(index, name, val);
}
}
}