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

#ifndef ASCEND_UTILS_INCLUDED
#define ASCEND_UTILS_INCLUDED

#include <ascenddaemon/utils/MemorySpace.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/utils/AscendAssert.h>
#include <initializer_list>
#include "acl/acl.h"
#include "securec.h"
#include <string>

namespace ascend {
class AscendUtils {
public:
    // Sets the current thread-local Ascend device
    static void setCurrentDevice(int device);

    // Reset the current thread-local Ascend device
    static void resetCurrentDevice(int device);

    // Sets the current thread-local Ascend device
    static void setCurrentContext(aclrtContext ctx);

    // Returns the current thread-local Ascend context
    static aclrtContext getCurrentContext();

    // Returns the maximum k-selection value supported
    static int getMaxKSelection();

    static void attachToCpu(int cpuId);

    static void attachToCpus(std::initializer_list<uint8_t> cpus);
};

class AscendOperatorManager {
public:
    static void init(std::string path);
    virtual ~AscendOperatorManager();

private:
    AscendOperatorManager() = delete;
};

class DeviceScope {
public:
    explicit DeviceScope();
    ~DeviceScope();
};

#define CALL_PARALLEL_FUNCTOR(pcounter, pmin, forCounter, threadPool, functor) \
    do {                                                                       \
        if (pcounter >= pmin) {                                                \
            std::vector<std::future<void>> functorRets;                        \
            for (int i = 0; i < forCounter; i++) {                             \
                functorRets.emplace_back(threadPool->Enqueue(functor, i));     \
            }                                                                  \
                                                                               \
            try {                                                              \
                for (auto &ret : functorRets) {                                \
                    ret.get();                                                 \
                }                                                              \
            } catch (std::exception & e) {                                     \
                ASCEND_THROW_FMT("%s.", e.what());                             \
            }                                                                  \
        } else {                                                               \
            for (int i = 0; i < forCounter; i++) {                             \
                functor(i);                                                    \
            }                                                                  \
        }                                                                      \
    } while (false)

#define CALL_PARALLEL_FUNCTOR_NOEXCEPTION(pcounter, pmin, forCounter, threadPool, functor) \
    do {                                                                                   \
        if (pcounter >= pmin) {                                                            \
            std::vector<std::future<void>> functorRets;                                    \
            for (int i = 0; i < forCounter; i++) {                                         \
                functorRets.emplace_back(threadPool->Enqueue(functor, i));                 \
            }                                                                              \
                                                                                           \
            for (auto &ret : functorRets) {                                                \
                ret.get();                                                                 \
            }                                                                              \
        } else {                                                                           \
            for (int i = 0; i < forCounter; i++) {                                         \
                functor(i);                                                                \
            }                                                                              \
        }                                                                                  \
    } while (false)

#define WAITING_FLAG_READY(flag, checkTicks, timeout)                                                  \
    do {                                                                                               \
        int waitTicks_ = 0;                                                                            \
        double startWait_ = utils::getMillisecs();                                                     \
        while (!(flag)) {                                                                              \
            waitTicks_++;                                                                              \
            if (!(waitTicks_ % (checkTicks)) && ((utils::getMillisecs() - startWait_) >= (timeout))) { \
                ASCEND_THROW_MSG("wait timeout.");                                                     \
            }                                                                                          \
        }                                                                                              \
    } while (false)

// Wrapper to test return status of ACL functions
#define ACL_REQUIRE_OK(x)                                                           \
    do {                                                                            \
        auto err_ = (x);                                                            \
        ASCEND_THROW_IF_NOT_FMT(err_ == ACL_ERROR_NONE, "ACL error %d", (int)err_); \
    } while (0)

#define ACL_ASSERT_OK(x)                                                      \
    do {                                                                      \
        auto err_ = (x);                                                      \
        ASCEND_ASSERT_FMT(err_ == ACL_ERROR_NONE, "ACL error %d", (int)err_); \
    } while (0)


#define MEMCPY_S(dst, dstLen, src, srcLen)                                  \
    do {                                                                    \
        auto err_ = memcpy_s(dst, dstLen, src, srcLen);                     \
        ASCEND_THROW_IF_NOT_FMT(err_ == EOK, "Memcpy error %d", (int)err_); \
    } while (0)

#define VALUE_UNUSED(x) (void)x;
} // ascend

#endif