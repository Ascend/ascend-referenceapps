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
#include <ascenddaemon/utils/AscendUtils.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <gtest/gtest.h>
#include <vector>

namespace {
TEST(TestAscendResources, Init)
{
    ascend::StandardAscendResources res;

    aclrtStream stream = res.getDefaultStream();
    EXPECT_NE(stream, nullptr);
}

TEST(TestAscendResources, Stream)
{
    ascend::StandardAscendResources res;

    aclrtStream stream = res.getDefaultStream();
    EXPECT_NE(stream, nullptr);

    std::vector<aclrtStream> vecStr = res.getAlternateStreams();
    EXPECT_EQ(vecStr.size(), 2);

    for (auto &str : vecStr) {
        EXPECT_NE(str, nullptr);
    }
}

TEST(TestAscendResources, Mem)
{
    ascend::StandardAscendResources res;
    res.initialize();
    aclrtStream stream = res.getDefaultStream();
    EXPECT_NE(stream, nullptr);

    ascend::AscendMemory &memMan = res.getMemoryManager();

    int defaultMemSize = 0x8000000; // default memory size allocated
    int byteAligned = 32;  // byte Aligned in Ascend
    EXPECT_EQ(memMan.getSizeAvailable(), (size_t)defaultMemSize);
    EXPECT_EQ(memMan.getHighWater(), (size_t)0);

    {
        const int testSize = 512;
        auto mem = memMan.getMemory(stream, testSize);
        EXPECT_EQ(mem.size(), (size_t)testSize);
        EXPECT_EQ(mem.stream(), stream);

        EXPECT_EQ(memMan.getSizeAvailable(), (size_t)(defaultMemSize - testSize));
        EXPECT_EQ(memMan.getHighWater(), (size_t)0);
    }

    EXPECT_EQ(memMan.getSizeAvailable(), (size_t)defaultMemSize);
    EXPECT_EQ(memMan.getHighWater(), (size_t)0);

    {
        auto mem = memMan.getMemory(stream, defaultMemSize + 8);
        EXPECT_EQ(mem.size(), defaultMemSize + ascend::utils::roundUp(8, byteAligned));
        EXPECT_EQ(mem.stream(), stream);

        EXPECT_EQ(memMan.getSizeAvailable(), defaultMemSize);
        EXPECT_EQ(memMan.getHighWater(), defaultMemSize + ascend::utils::roundUp(8, byteAligned));
    }
    std::cout << memMan.toString() << std::endl;
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}
