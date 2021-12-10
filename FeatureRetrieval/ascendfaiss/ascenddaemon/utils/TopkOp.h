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
 
#ifndef TOPK_OP_INCLUDED
#define TOPK_OP_INCLUDED

#include "HeapSort.h"
#include <arm_fp16.h>
#include <tuple>
#include <ascenddaemon/utils/AscendTensor.h>

namespace ascend {
template<typename T, typename E, typename D, bool ASC = true>
class TopkOp {
public:
    TopkOp<T, E, D, ASC>();
    virtual ~TopkOp<T, E, D, ASC>();

    bool exec(AscendTensor<D, DIMS_2>& distance,
              AscendTensor<uint32_t, DIMS_2>& indices,
              AscendTensor<D, DIMS_2>& topkDistance,
              AscendTensor<uint32_t, DIMS_2>& topkIndices,
              uint32_t indexOffset = 0);

    bool exec(AscendTensor<D, DIMS_1> &distance,
                  AscendTensor<uint32_t, DIMS_1> &indices,
                  AscendTensor<D, DIMS_1> &topkDistance,
                  AscendTensor<uint32_t, DIMS_1> &topkIndices,
                  uint32_t realSize,
                  uint32_t indexOffset);

    bool exec(AscendTensor<D, DIMS_1>& distance,
              AscendTensor<uint32_t, DIMS_1>& indices,
              AscendTensor<D, DIMS_1>& topkDistance,
              AscendTensor<uint32_t, DIMS_1>& topkIndices);

    bool exec(std::tuple<D*, D*, uint32_t*> &opOutTp,
              std::tuple<D*, uint32_t*, int> &topKHeapTp,
              std::tuple<D*, uint32_t*> &tmpHeapTp,
              const uint32_t realSize,
              const int burstLen);

    bool exec(std::tuple<D*, D*, uint32_t*> &opOutTp,
              std::tuple<D*, uint32_t*, int> &topKHeapTp,
              const uint32_t realSize,
              const uint32_t indexOffset,
              const int burstLen);

    void reorder(AscendTensor<D, DIMS_2>& topkDistance,
                 AscendTensor<uint32_t, DIMS_2>& topkIndices);

    void reorder(AscendTensor<D, DIMS_1>& topkDistance,
                 AscendTensor<uint32_t, DIMS_1>& topkIndices);

private:
    bool checkParams(AscendTensor<D, DIMS_2>& distance,
                     AscendTensor<uint32_t, DIMS_2>& indices,
                     AscendTensor<D, DIMS_2>& topkDistance,
                     AscendTensor<uint32_t, DIMS_2>& topkIndices);

    bool checkParams(AscendTensor<D, DIMS_1>& distance,
                     AscendTensor<uint32_t, DIMS_1>& indices,
                     AscendTensor<D, DIMS_1>& topkDistance,
                     AscendTensor<uint32_t, DIMS_1>& topkIndices);

private:
    T compare;
    E compareEqual;
};
}  // namespace ascend

#endif
