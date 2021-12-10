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

#ifndef ASCEND_RPC_UTILS_H
#define ASCEND_RPC_UTILS_H

#include <string>
#include <cstdio>
#include <cstdint>
#include <syslog.h>
#include <sys/time.h>

namespace faiss {
namespace ascend {
namespace {
    const int SEC_CONVER = 1000;
}

/* -------------------------- Shared types -------------------------- */
using client_id_t = int32_t;
using index_id_t = int32_t;

/* ------------------------------ Utils ------------------------------ */
#ifdef DEBUG
#define RPC_DEBUG
#define RPC_LOG_ENABLE
#endif

#define RPC_VALUE_UNUSED(x) \
    static_cast<void>(x);

#ifdef RPC_DEBUG
#define RPC_ASSERT(X) do { \
    if (!(X)) {                                               \
        fprintf(stderr,                                       \
                "[RPC] assertion '%s' failed in %s "          \
                "at %s:%d\n",                                 \
                #X, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
        abort();                                              \
    }                                                         \
} while (false)

#define RPC_ASSERT_MSG(X, MSG) do { \
    if (!(X)) {                                               \
        fprintf(stderr,                                       \
                "[RPC] assertion '%s' failed in %s "          \
                "at %s:%d; details: " MSG "\n",               \
                #X, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
        abort();                                              \
    }                                                         \
} while (false)

#define RPC_ASSERT_FMT(X, FMT, ...) do { \
    if (!(X)) {                                                            \
        fprintf(stderr,                                                    \
                "[RPC] assertion '%s' failed in %s "                       \
                "at %s:%d; details: " FMT "\n",                            \
                #X, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); \
        abort();                                                           \
    }                                                                      \
} while (false)

#else
#define RPC_ASSERT(X) RPC_VALUE_UNUSED(X)
#define RPC_ASSERT_MSG(X, MSG) RPC_VALUE_UNUSED(X)
#define RPC_ASSERT_FMT(X, FMT, ...) RPC_VALUE_UNUSED(X)
#endif  // RPC_DEBUG

#ifdef RPC_LOG_ENABLE
#define RPC_LOG_INFO(...) do { \
    fprintf(stdout, "[%s:%d--%s] ",            \
            __FILE__, __LINE__, __FUNCTION__); \
    fprintf(stdout, __VA_ARGS__);              \
    fflush(stdout);                            \
} while (false)
#else
#define RPC_LOG_INFO(...)
#endif  // RPC_LOG_ENABLE

#define RPC_LOG_ERROR(...) do { \
    syslog(LOG_ERR, __VA_ARGS__);              \
    fprintf(stdout, "[%s:%d--%s] ",            \
            __FILE__, __LINE__, __FUNCTION__); \
    fprintf(stdout, __VA_ARGS__);              \
    fflush(stdout);                            \
} while (false)

#ifdef RPC_LOG_ENABLE
#define RPC_TIME_LOG(...) do { \
    fprintf(stdout, "[%.6f][%s:%d-%s] ", ::faiss::ascend::getTimeSecondElapsed(), \
            __FILE__, __LINE__, __FUNCTION__);                                    \
    fprintf(stdout, __VA_ARGS__);                                                 \
    fflush(stdout);                                                               \
} while (false)
#else
#define RPC_TIME_LOG(...)
#endif

#define RPC_UNIMPLEMENTED() RPC_ASSERT_MSG(false, "UNIMPLEMENTED!!!")
#define RPC_UNREACHABLE() RPC_ASSERT_MSG(false, "UNREACHABLE!!!")
#define RPC_REQUIRE_NOT_NULL(ptr) RPC_ASSERT_MSG((ptr) != nullptr, "Unexpected nullptr")
#define RPC_REQUIRE_OK(ret) RPC_ASSERT_FMT(((ret) == 0), "Unexpected error %d", ret)

inline uint64_t getTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * SEC_CONVER * SEC_CONVER + tv.tv_usec;
}

inline double getTimeSecondElapsed()
{
    static double startTimeUs = getTimeUs();
    return (getTimeUs() - startTimeUs) / static_cast<double>(SEC_CONVER * SEC_CONVER);
}

inline uint8_t calcChecksum(const void *data, int len)
{
    RPC_REQUIRE_NOT_NULL(data);
    RPC_ASSERT(len > 0);

    uint8_t tmp = 0;
    for (int i = 0; i < len; i++) {
        tmp += static_cast<const uint8_t *>(data)[i];
    }

    return ~tmp;
}
}  // namespace ascend
}  // namespace faiss

#endif  // ASCEND_RPC_UTILS_H
