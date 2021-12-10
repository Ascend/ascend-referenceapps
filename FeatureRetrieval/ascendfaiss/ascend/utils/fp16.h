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

#ifndef ASCEND_FP16_INCLUDED
#define ASCEND_FP16_INCLUDED

#include <math.h>
#include <string>

namespace faiss { 
namespace ascend {
/**
 * @ingroup fp16
 * @brief   Half precision float
 *         bit15:       1 bit SIGN      +---+-----+------------+
 *         bit14-10:    5 bit EXP       | S |EEEEE|MM MMMM MMMM|
 *         bit0-9:      10bit MAN       +---+-----+------------+
 */
struct fp16 {
    union {
        uint16_t data;  // All bits
        struct {
            uint16_t man : 10;  // mantissa
            uint16_t exp : 5;   // exponent
            uint16_t sign : 1;  // sign
        } Bits;
    };

public:
    fp16();

    fp16(const uint16_t &val);

    fp16(const int16_t &val);

    fp16(const fp16 &fp);

    fp16(const int32_t &val);

    fp16(const uint32_t &val);

    fp16(const float &val);

    bool operator==(const fp16 &fp) const;

    bool operator!=(const fp16 &fp) const;

    bool operator>(const fp16 &fp) const;

    bool operator>=(const fp16 &fp) const;

    bool operator<(const fp16 &fp) const;

    bool operator<=(const fp16 &fp) const;

    fp16 &operator=(const fp16 &fp);

    fp16 &operator=(const float &val);

    operator float() const;

    static fp16 min();
    static fp16 max();
};
} // ascend
} // faiss

#endif