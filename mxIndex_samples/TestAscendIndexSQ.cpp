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

#include <numeric>
#include <random>
#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>

#include <sys/time.h>

namespace {
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;

inline void FastSrand(int seed)
{
    g_seed = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline int FastRand(void)
{
    const int mutipliyNum = 214013;
    const int addNum = 2531011;
    const int rshiftNum = 16;
    g_seed = (mutipliyNum * g_seed + addNum);
    return (g_seed >> rshiftNum) & FAST_RAND_MAX;
}

inline double GetMillisecs()
{
    struct timeval tv {
        0, 0
    };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline void AssertEqual(std::vector<uint8_t> &gt, std::vector<uint8_t> &data)
{
    ASSERT_EQ(gt.size(), data.size());
    for (size_t i = 0; i < gt.size(); i++) {
        ASSERT_EQ(gt[i], data[i]) << "i: " << i << " gt: " << gt[i] << " data: " << data[i] << std::endl;
    }
}

inline void Norm(float *data, int n, int dim)
{
#pragma omp parallelf for if(n > 100)
    for (size_t i = 0; i < n; ++i){
        float l2norm = 0;
        for (int j = 0; j < dim; ++j){
            l2norm += data[i * dim + j] * data[i * dim +j];
        }
        l2norm = std::sqrt(l2norm);
        for (int j = 0; j < dim; ++j){
            data[i * dim + j] = data[i * dim + j] / l2norm;
        }
    }
}


TEST(TestAscendIndexSQ, QPS)
{
    std::vector<int> dim = { 256,512 };
    std::vector<size_t> ntotal = { 7000000 };
    std::vector<int> searchNum = { 1, 2, 4, 8, 16, 32, 48, 64, 96 };

    size_t maxSize = ntotal.back() * dim.back();
    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
    }
    
    Norm(data.data(), ntotal.back(), dim.back());

    for (size_t i = 0; i < dim.size(); i++) {
        faiss::ascend::AscendIndexSQConfig conf({ 0 }, 1024 * 1024 * 1500);
        faiss::ascend::AscendIndexSQ index(dim[i], faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2,
            conf);

        for (size_t j = 0; j < ntotal.size(); j++) {
            index.reset();
            for (auto deviceId : conf.deviceList) {
                int len = index.getBaseSize(deviceId);
                ASSERT_EQ(len, 0);
            }

            index.train(ntotal[j], data.data());
            index.add(ntotal[j], data.data());
            {
                int getTotal = 0;
                for (size_t k = 0; k < conf.deviceList.size(); k++) {
                    int tmpTotal = index.getBaseSize(conf.deviceList[k]);
                    getTotal += tmpTotal;
                }
                EXPECT_EQ(getTotal, ntotal[j]);
            }

            {
                for (size_t n = 0; n < searchNum.size(); n++) {
                    int k = 100;
                    int loopTimes = 100;
                    std::vector<float> dist(searchNum[n] * k, 0);
                    std::vector<faiss::Index::idx_t> label(searchNum[n] * k, 0);
                    double ts = GetMillisecs();
                    for (int l = 0; l < loopTimes; l++) {
                        index.search(searchNum[n], data.data(), k, dist.data(), label.data());
                    }
                    double te = GetMillisecs();
                    int cases = i * ntotal.size() * searchNum.size() + j * searchNum.size() + n;
                    printf("case[%d]: base:%zu, dim:%d, search num:%d, QPS:%.4f\n", cases, ntotal[j], dim[i],
                        searchNum[n], 1000 * searchNum[n] * loopTimes / (te - ts));
                }
            }
        }
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}