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

#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/DeviceVectorInl.h>
#include <gtest/gtest.h>
#include <vector>

namespace {
TEST(DEVICE_VECTOR, ALL)
{
    int size = 512;
    ascend::DeviceVector<int32_t> dVec1;
    std::vector<int32_t> stdVec(size);
    for (int i = 0; i < size; i++) {
        stdVec[i] = (int32_t)(rand() % 100);
    }

    // append, size, capacity
    dVec1.append(stdVec.data(), stdVec.size());
    EXPECT_EQ(dVec1.size(), size);
    EXPECT_EQ(dVec1.capacity(), size * 3 / 2);

    // operator[]
    for (int i = 0; i < size; i++) {
        EXPECT_EQ(stdVec[i], dVec1[i]);
    }

    // reclaim
    auto freeSize = dVec1.reclaim(true);
    EXPECT_EQ(freeSize, (size * 3 / 2 - size) * sizeof(int32_t));

    // resize
    dVec1.resize(200);
    EXPECT_EQ(dVec1.size(), 200);

    // cpy to stl vector
    std::vector<int32_t> toVec = dVec1.copyToStlVector();
    EXPECT_EQ(memcmp(toVec.data(), dVec1.data(), dVec1.size()), 0);

    // clear
    dVec1.clear();
    EXPECT_EQ(dVec1.size(), 0);
    EXPECT_EQ(dVec1.capacity(), 0);
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}
