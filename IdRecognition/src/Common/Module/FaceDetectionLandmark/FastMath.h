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
#ifndef FAST_MATH_H
#define FAST_MATH_H

#include <algorithm>
#include <cmath>

class FastMath {
public:
    FastMath()
    {
    }
    void Init()
    {
        for (auto i = 0; i < MASK_LEN; i++) {
            negCoef[0][i] = std::exp(-float(i) / QUANT_VALUE);
            negCoef[1][i] = std::exp(-float(i) * MASK_LEN / QUANT_VALUE);
            posCoef[0][i] = std::exp(float(i) / QUANT_VALUE);
            posCoef[1][i] = std::exp(float(i) * MASK_LEN / QUANT_VALUE);
        }
    }
    ~FastMath() {}

    inline float Fexp(const float x) const
    {
        int quantX = std::max(std::min(x, float(QUANT_BOUND)), -float(QUANT_BOUND)) * QUANT_VALUE;
        bool flag = quantX & 0x80000000;
        int index = flag ? (~quantX + 0x00000001) : quantX;
        return flag ? (negCoef[0][(index)&MASK_VALUE] * negCoef[1][(index >> MASK_BITS) & MASK_VALUE]) :
                      (posCoef[0][(index)&MASK_VALUE] * posCoef[1][(index >> MASK_BITS) & MASK_VALUE]);
    }

    inline float Sigmoid(float x) const
    {
        return 1.0f / (1.0f + Fexp(-x));
    }

private:
    static const int MASK_BITS = 12;
    static const int MASK_LEN = (1 << MASK_BITS);
    static const int MASK_VALUE = MASK_LEN - 1;
    static const int QUANT_BITS = 16;
    static const int QUANT_VALUE = (1 << QUANT_BITS);
    static const int CONSTANT_2 = 2;
    static const int QUANT_BOUND = (1 << (CONSTANT_2 * MASK_BITS - QUANT_BITS)) - 1;

    float negCoef[CONSTANT_2][MASK_LEN] = { {0} };
    float posCoef[CONSTANT_2][MASK_LEN] = { {0} };
};

#endif