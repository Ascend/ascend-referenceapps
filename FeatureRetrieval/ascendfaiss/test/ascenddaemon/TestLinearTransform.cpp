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

#include <ascenddaemon/impl/VectorTransform.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>
#include <cmath>

namespace {
TEST(TestLinearTranform, UpdateAndApply)
{
    // create LinearTransform
    const int dimIn = 512;
    const int dimOut = 128;

    ascend::StandardAscendResources resources;
    resources.initialize();
    ascend::LinearTransform ltrans(dimIn, dimOut);

    unsigned int seed = time(nullptr);
    srand(seed);
    std::vector<float16_t> matrix(dimIn * dimOut);
#pragma omp parallel for
    for (size_t i = 0; i < matrix.size(); i++) {
        matrix[i] = (float16_t)i / 100;
    }
    std::vector<float> bias(dimOut);
#pragma omp parallel for
    for (size_t i = 0; i < bias.size(); i++) {
        bias[i] = (float)(512 - i);
    }
    ascend::AscendTensor<float16_t, ascend::DIMS_2> mm(matrix.data(), { dimOut, dimIn });
    ascend::AscendTensor<float, ascend::DIMS_1> bb(bias.data(), { dimOut });
    ltrans.updateTrainedValue(mm, bb);
    std::cout << "update trained value success" << std::endl;

    // apply
    const std::vector<int> nums { 1, 3, 7, 15, 31, 63, 127, 255 };
    for (auto num : nums) {
        ascend::DeviceScope device;
        auto stream = resources.getDefaultStream();
        auto &mem = resources.getMemoryManager();
        ascend::AscendTensor<float16_t, ascend::DIMS_2> x(mem, { num, dimIn }, stream);
        x.initValue(1.0);
        ascend::AscendTensor<float16_t, ascend::DIMS_2> xt(mem, { num, dimOut }, stream);

        ltrans.apply(num, x.data(), xt.data(), stream);
        std::cout << "----- start apply, num_vec is " << num << "-----" << std::endl;
        double cost = 0.0;
        double min = std::numeric_limits<double>::max();
        double max = 0.0;

        const int loopSearch = 100;
        for (int i = 0; i < loopSearch; i++) {
            double start = ascend::utils::getMillisecs();
            ltrans.apply(num, x.data(), xt.data(), stream);
            double end = ascend::utils::getMillisecs();
            double cur_cost = end - start;
            min = std::min(min, cur_cost);
            max = std::max(max, cur_cost);
            cost += cur_cost;
        }

        std::cout << "apply cost time: " << cost / (loopSearch * num) << "ms in average, "
                  << "max:" << max / num << "ms, min:" << min / num << "ms." << std::endl;
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}
