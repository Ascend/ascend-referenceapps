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
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <gtest/gtest.h>

namespace {
TEST(TestAscendTensor, Tensor)
{
    ascend::AscendTensor<float, 1> tensor();
    ascend::StandardAscendResources res;

    ascend::AscendMemory& memManager = res.getMemoryManager();
    aclrtStream stream = res.getDefaultStream();
    size_t s = memManager.getSizeAvailable();
    {
        ascend::AscendTensor<float, 3> tmpTensor(memManager, {3, 4, 5}, stream);
        int byteAligned = 32;  // byte Aligned in Ascend
        EXPECT_EQ(memManager.getSizeAvailable(), s - 
            ascend::utils::roundUp(tmpTensor.getSizeInBytes(), byteAligned));
    }
    EXPECT_EQ(memManager.getSizeAvailable(), s);

    ascend::AscendTensor<float, 3> deviceTensor(memManager, {3, 4, 5}, stream);
    EXPECT_EQ(deviceTensor.NUM_DIM, 3);
    EXPECT_NE(deviceTensor.data(), nullptr);
    EXPECT_NE(deviceTensor.end(), nullptr);
    EXPECT_EQ(deviceTensor.getSize(0), 3);
    EXPECT_EQ(deviceTensor.getSize(1), 4);
    EXPECT_EQ(deviceTensor.getSize(2), 5);
    EXPECT_EQ(deviceTensor.getStride(0), 20);
    EXPECT_EQ(deviceTensor.getStride(1), 5);
    EXPECT_EQ(deviceTensor.getStride(2), 1);
    EXPECT_EQ(deviceTensor.numElements(), 60);
    EXPECT_EQ(deviceTensor.getSizeInBytes(), 60 * sizeof(float));
    EXPECT_EQ(deviceTensor.getSizeInBytes(), sizeof(float) * (deviceTensor.end() - deviceTensor.data()));

    EXPECT_EQ(deviceTensor[0].data(), deviceTensor.data());

    auto viewTensor = deviceTensor[1].view();
    EXPECT_EQ(deviceTensor[1].data(), viewTensor.data());
    EXPECT_EQ(viewTensor.numElements(), 4 * 5);

    // Constructor: given size
    ascend::AscendTensor<float, 1> sizeTensor1({4});
    int32_t sizes[2] = {3, 4};
    ascend::AscendTensor<float, 2> sizeTensor2(sizes);
    EXPECT_EQ(sizeTensor1.numElements(), 4);
    EXPECT_EQ(sizeTensor2.numElements(), 12);

    // copy constructor
    ascend::AscendTensor<float, 2> copyTensor2(sizeTensor2);
    ascend::AscendTensor<float, 1> copyTensor1 = sizeTensor1;
    EXPECT_EQ(sizeTensor1.numElements(), copyTensor1.numElements());
    EXPECT_EQ(sizeTensor2.numElements(), copyTensor2.numElements());

    // move constructor
    ascend::AscendTensor<float, 1> moveTensor1(std::move(sizeTensor1));
    ascend::AscendTensor<float, 2> moveTensor2 = std::move(sizeTensor2);
    EXPECT_EQ(moveTensor1.numElements(), 4);
    EXPECT_EQ(moveTensor2.numElements(), 12);

    // copyFunctions
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}
