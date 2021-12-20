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

#ifndef ASCEND_LIMITS_H
#define ASCEND_LIMITS_H

#include <arm_fp16.h>

const int CUBE_ALIGN_SIZE = 16;

template<typename T>
struct Limits {
};

template<>
struct Limits<float16_t> {
    static inline float16_t getMin()
    {
        uint16_t val = 0xfbffU;
        return *((float16_t *)(&val));
    }
    static inline float16_t getMax()
    {
        uint16_t val = 0x7bffU;
        return *((float16_t *)(&val));
    }
};

#endif