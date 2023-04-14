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
#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexFlat.h>
#include <cstring>
#include <sys/time.h>
#include <faiss/index_io.h>
#include <cstdlib>


namespace {
inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline void AssertEqual(const std::vector<float> &gt, const std::vector<float> &data)
{
    const float epson = 1e-3;
    ASSERT_EQ(gt.size(), data.size());
    for (size_t i = 0; i < gt.size(); i++) {
        ASSERT_TRUE(fabs(gt[i] - data[i]) <= epson) << "i: " << i << " gt: " << gt[i] << " data: " << data[i] <<
            std::endl;
    }
}




namespace {
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;
}

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

TEST(TestAscendIndexFlat, QPS)
{
    int dim = 512;
    size_t ntotal = 7000000;
    size_t maxSize = ntotal * dim;

    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
    }

    faiss::ascend::AscendIndexFlatConfig conf({ 0 },1024 * 1024 *1500);
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);
    index.verbose = true;
    
    // 标准化
    Norm(data.data(), ntotal, dim);

    index.add(ntotal, data.data());
    {
        int getTotal = 0;
        for (size_t i = 0; i < conf.deviceList.size(); i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            getTotal += tmpTotal;
        }
        EXPECT_EQ(getTotal, ntotal);
    }
    
        int warmUpTimes = 10 ;
        std::vector<float> distw(127 * 10, 0);
        std::vector<faiss::Index::idx_t> labelw(127 * 10, 0);
        for (int i = 0; i < warmUpTimes; i++) {
            index.search(127, data.data(), 10, distw.data(), labelw.data());
        }

    std::vector<int> searchNum = {  8, 16, 32, 64, 128 ,256};
    for (size_t n = 0; n < searchNum.size(); n++) {
        int k = 10;
        int loopTimes = 100;
        std::vector<float> dist(searchNum[n] * k, 0);
        std::vector<faiss::Index::idx_t> label(searchNum[n] * k, 0);
        double ts = GetMillisecs();
        for (int i = 0; i < loopTimes; i++) {
            index.search(searchNum[n], data.data(), k, dist.data(), label.data());
        }
        double te = GetMillisecs();
        printf("case[%zu]: base:%zu, dim:%d, search num:%d, QPS:%.4f\n", n, ntotal, dim, searchNum[n],
            1000 * searchNum[n] * loopTimes / (te - ts));
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
