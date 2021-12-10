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

#ifndef ASCEND_ASSERT_INCLUDED
#define ASCEND_ASSERT_INCLUDED

#include <ascenddaemon/utils/AscendException.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <securec.h>

// Assertions abort if condition not satisfied
#define ASCEND_ASSERT(X)                                          \
    do {                                                          \
        if (!(X)) {                                               \
            fprintf(stderr, "Ascend assertion '%s' failed in %s " \
                    "at %s:%d\n",                                 \
                    #X, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
            abort();                                              \
        }                                                         \
    } while (false)

#define ASCEND_ASSERT_MSG(X, MSG)                                 \
    do {                                                          \
        if (!(X)) {                                               \
            fprintf(stderr, "Ascend assertion '%s' failed in %s " \
                    "at %s:%d; details: " MSG "\n",               \
                    #X, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
            abort();                                              \
        }                                                         \
    } while (false)

#define ASCEND_ASSERT_FMT(X, FMT, ...)                                         \
    do {                                                                       \
        if (!(X)) {                                                            \
            fprintf(stderr, "Ascend assertion '%s' failed in %s "              \
                    "at %s:%d; details: " FMT "\n",                            \
                    #X, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); \
            abort();                                                           \
        }                                                                      \
    } while (false)


// Exceptions for returning user errors
#define ASCEND_THROW_MSG(MSG)                                                        \
    do {                                                                             \
        throw ascend::AscendException(MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)

#define ASCEND_THROW_FMT(FMT, ...)                                                   \
    do {                                                                             \
        int size = 1024;                                                             \
        std::string __s(size, '\0');                                                 \
        while (snprintf_s(&__s[0], __s.size(), __s.size(), FMT, __VA_ARGS__) < 0) {  \
            size = size * 2;                                                         \
            __s.resize(size, '\0');                                                  \
        }                                                                            \
        throw ascend::AscendException(__s, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)


// Exceptions thrown upon a conditional failure
#define ASCEND_THROW_IF_NOT(X)                          \
    do                                                  \
    {                                                   \
        if (!(X)) {                                     \
            ASCEND_THROW_FMT("Error: '%s' failed", #X); \
        }                                               \
    } while (false)

#define ASCEND_THROW_IF_NOT_CODE(X, CODE)               \
    do                                                  \
    {                                                   \
        if (!(X)) {                                     \
            CODE;                                       \
            ASCEND_THROW_FMT("Error: '%s' failed", #X); \
        }                                               \
    } while (false)

#define ASCEND_THROW_IF_NOT_MSG(X, MSG)                       \
    do                                                        \
    {                                                         \
        if (!(X)) {                                           \
            ASCEND_THROW_FMT("Error: '%s' failed: " MSG, #X); \
        }                                                     \
    } while (false)

#define ASCEND_THROW_IF_NOT_FMT(X, FMT, ...)                               \
    do                                                                     \
    {                                                                      \
        if (!(X)) {                                                        \
            ASCEND_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__); \
        }                                                                  \
    } while (false)

#endif
