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
 
#ifndef HEAP_SORT_INCLUDED
#define HEAP_SORT_INCLUDED

#include <tuple>
#include <arm_fp16.h>
#include <ascenddaemon/utils/AscendTensor.h>

namespace ascend {
    template<typename D, typename __Compare>
    void pushHeap(const size_t size, D* heapDist, uint32_t* heapId,
                  const D pushDist, const uint32_t pushId, __Compare compare)
    {
        size_t i = 1;
        size_t leftChild, rightChild;

        heapDist--;
        heapId--;
        while (1) {
            leftChild = i << 1;
            rightChild = leftChild + 1;
            if (leftChild > size) {
                break;
            }

            if (rightChild == (size + 1) || compare(heapDist[leftChild], heapDist[rightChild])) {
                if (compare(pushDist, heapDist[leftChild])) {
                    break;
                }

                heapDist[i] = heapDist[leftChild];
                heapId[i] = heapId[leftChild];
                i = leftChild;
            } else {
                if (compare(pushDist, heapDist[rightChild])) {
                    break;
                }

                heapDist[i] = heapDist[rightChild];
                heapId[i] = heapId[rightChild];
                i = rightChild;
            }
        }

        heapDist[i] = pushDist;
        heapId[i] = pushId;
    }

    template<typename DP, typename __Compare>
    void popHeap(size_t k, DP heapDist, uint32_t* heapId, __Compare compare)
    {
        heapDist--; /* Use 1-based indexing for easier node->child translation */
        heapId--;
        float16_t val = heapDist[k];
        size_t i = 1;
        size_t i1;
        size_t i2;
        while (1) {
            i1 = i << 1;
            i2 = i1 + 1;
            if (i1 > k) {
                break;
            }

            if (i2 == k + 1 || compare(heapDist[i1], heapDist[i2])) {
                if (compare(val, heapDist[i1])) {
                    break;
                }
                heapDist[i] = heapDist[i1];
                heapId[i] = heapId[i1];
                i = i1;
            } else {
                if (compare(val, heapDist[i2])) {
                    break;
                }
                heapDist[i] = heapDist[i2];
                heapId[i] = heapId[i2];
                i = i2;
            }
        }
        heapDist[i] = heapDist[k];
        heapId[i] = heapId[k];
    }
}  // namespace ascend

#endif
