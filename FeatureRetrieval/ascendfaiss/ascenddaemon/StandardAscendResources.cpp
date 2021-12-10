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

#include <ascenddaemon/StandardAscendResources.h>

namespace ascend {
// How many streams we allocate by default (for multi-streaming)
const int NUM_STREAMS = 2;

// Default temporary memory allocation
const size_t BIT_SHIFT_FOR_TEMP_MEM = 27;
const size_t DEFAULT_TEMP_MEM = (size_t)(1 << BIT_SHIFT_FOR_TEMP_MEM);

// Max temporary memory allocation 1 << 30 means 1GB mem
const size_t MAX_TEMP_MEM = (size_t)(1 << 30);
const size_t INVALID_TEMP_MEM = (size_t)-1;

StandardAscendResources::StandardAscendResources(std::string modelPath)
    : defaultStream(0),
      tempMemSize(INVALID_TEMP_MEM)
{
    AscendOperatorManager::init(modelPath);
    AscendStackMemory::getInstance().ref();
}

StandardAscendResources::~StandardAscendResources()
{
    if (defaultStream) {
        ACL_ASSERT_OK(aclrtSynchronizeStream(defaultStream));
        ACL_ASSERT_OK(aclrtDestroyStream(defaultStream));
    }

    for (auto& stream : alternateStreams) {
        ACL_ASSERT_OK(aclrtSynchronizeStream(stream));
        ACL_ASSERT_OK(aclrtDestroyStream(stream));
    }

    AscendStackMemory::getInstance().unRef();
}

void StandardAscendResources::noTempMemory()
{
    setTempMemory(0);
    setAscendMallocWarning(false);
}

void StandardAscendResources::setTempMemory(size_t size)
{
    if (tempMemSize != size) {
        tempMemSize = getDefaultTempMemSize(size);
        AscendStackMemory::getInstance().allocMemory(tempMemSize);
    }
}

void StandardAscendResources::setDefaultTempMemory()
{
    setTempMemory(DEFAULT_TEMP_MEM);
}

void StandardAscendResources::setAscendMallocWarning(bool flag)
{
    AscendStackMemory::getInstance().setMallocWarning(flag);
}

void StandardAscendResources::initialize()
{
    if (isInitialized()) {
        return;
    }

    // Create streams
    ACL_REQUIRE_OK(aclrtCreateStream(&defaultStream));

    for (int j = 0; j < NUM_STREAMS; ++j) {
        aclrtStream stream = 0;
        ACL_REQUIRE_OK(aclrtCreateStream(&stream));
        alternateStreams.push_back(stream);
    }

    if (tempMemSize == INVALID_TEMP_MEM) {
        setDefaultTempMemory();
    }
}

aclrtStream StandardAscendResources::getDefaultStream()
{
    initialize();
    return defaultStream;
}

void StandardAscendResources::syncDefaultStream()
{
    ACL_REQUIRE_OK(aclrtSynchronizeStream(getDefaultStream()));
}

std::vector<aclrtStream> StandardAscendResources::getAlternateStreams()
{
    initialize();
    return alternateStreams;
}

AscendMemory& StandardAscendResources::getMemoryManager()
{
    initialize();
    return AscendStackMemory::getInstance();
}

bool StandardAscendResources::isInitialized() const
{
    return defaultStream != 0;
}

size_t StandardAscendResources::getDefaultTempMemSize(size_t requested)
{
    return (requested > MAX_TEMP_MEM) ? MAX_TEMP_MEM : requested;
}
}  // namespace ascend