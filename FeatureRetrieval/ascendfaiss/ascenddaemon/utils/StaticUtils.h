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

#ifndef ASCEND_STATICUTILS_H
#define ASCEND_STATICUTILS_H

#include <sys/time.h>

namespace ascend {
namespace utils {
template<typename U, typename V>
constexpr auto divDown(U a, V b) -> decltype(a + b)
{
    return (a / b);
}

template<typename U, typename V>
constexpr auto divUp(U a, V b) -> decltype(a + b)
{
    return (a + b - 1) / b;
}

template<typename U, typename V>
constexpr auto roundDown(U a, V b) -> decltype(a + b)
{
    return divDown(a, b) * b;
}

template<typename U, typename V>
constexpr auto roundUp(U a, V b) -> decltype(a + b)
{
    return divUp(a, b) * b;
}

template<class T>
constexpr T pow(T n, T power)
{
    return (power > 0) ? (n * pow(n, power - 1)) : 1;
}

template<class T>
constexpr T pow2(T n)
{
    const int power = 2;
    return pow(power, (T)n);
}

template<typename T>
constexpr int log2(T n, int p = 0)
{
    const int divisor = 2;
    return (n <= 1) ? p : log2(n / divisor, p + 1);
}

template<typename T>
constexpr bool isPowerOf2(T v)
{
    return (v && !(v & (v - 1)));
}

template<typename T>
constexpr T nextHighestPowerOf2(T v)
{
    const int scale = 2;
    return (isPowerOf2(v) ? (T)scale * v : ((T)1 << (log2(v) + 1)));
}

inline double getMillisecs()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    const double sec2msec = 1e3;
    const double usec2msec = 1e-3;
    return tv.tv_sec * sec2msec + tv.tv_usec * usec2msec;
}
}  // namespace utils
}  // namespace ascend
#endif  // ASCEND_STATICUTILS_H
