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

#include <gtest/gtest.h>
#include <faiss/ascend/AscendNNDimReduction.h>
#include <sys/time.h>

namespace {
inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

TEST(TestAscendNNDimReduction, All)
{
    int dimIn = 128;
    int dimOut = 32;
    int batchSize = 32;
    int ntotal = 1000000;
    std::vector<int> deviceList = { 0 };
    std::string modelPath = "./modelpath/nnDimReduction.om"; // Converted from neural network model
    int maxSize = ntotal * dimIn;

    std::vector<float> data(maxSize);
    std::vector<float> outputData;
    for (int i = 0; i < maxSize; i++) {
        data[i] = drand48();
    }

    ASSERT_EQ(outputData.size(), 0);

    faiss::ascend::AscendNNDimReduction dimReduction(deviceList, dimIn, dimOut, batchSize, modelPath);
    double ts = GetMillisecs();
    dimReduction.infer(ntotal, data.data(), outputData);
    double te = GetMillisecs();
    printf("deviceCount:%d, ntotal:%d, dimIn:%d, dimOut:%d, batchSize:%d, QPS:%.4f\n", deviceList.size(), ntotal, 
        dimIn, dimOut, batchSize, 1000.0 * ntotal / (te - ts));
    ASSERT_EQ(outputData.size(), ntotal * dimOut);

    ASSERT_EQ(outputData.size() / ntotal, dimOut);
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
