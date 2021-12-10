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

#include <ascenddaemon/utils/AscendUtils.h>
#include <mutex>
#include <thread>

namespace ascend {
namespace {
const int MAX_K_SELECTION = 1000;
}

void AscendUtils::setCurrentDevice(int device)
{
    ACL_REQUIRE_OK(aclrtSetDevice(device));
}

void AscendUtils::resetCurrentDevice(int device)
{
    ACL_REQUIRE_OK(aclrtResetDevice(device));
}

aclrtContext AscendUtils::getCurrentContext()
{
    aclrtContext ctx = 0;
    aclrtGetCurrentContext(&ctx);
    return ctx;
}

void AscendUtils::setCurrentContext(aclrtContext ctx)
{
    ACL_REQUIRE_OK(aclrtSetCurrentContext(ctx));
}

int AscendUtils::getMaxKSelection()
{
    return MAX_K_SELECTION;
}

void AscendUtils::attachToCpu(int cpuId)
{
    int cpu = cpuId % std::thread::hardware_concurrency();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void AscendUtils::attachToCpus(std::initializer_list<uint8_t> cpus)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (auto cpuId : cpus) {
        int cpu = cpuId % std::thread::hardware_concurrency();
        CPU_SET(cpu, &cpuset);
    }
    
    (void)pthread_setaffinity_np(pthread_self(),
                                 sizeof(cpu_set_t), &cpuset);
}

void AscendOperatorManager::init(std::string path)
{
    static bool isInited = false;
    static std::mutex mtx;

    if (!isInited) {
        std::unique_lock<std::mutex> lock(mtx);
        if (!isInited) {
            ACL_REQUIRE_OK(aclopSetModelDir(path.c_str()));
            isInited = true;
        }
    }

    return;
}

AscendOperatorManager::~AscendOperatorManager()
{
}

DeviceScope::DeviceScope()
{
    AscendUtils::setCurrentDevice(0);
}

DeviceScope::~DeviceScope()
{
    AscendUtils::resetCurrentDevice(0);
}
} // namespace ascend
