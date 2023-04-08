/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include <random>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexBinaryFlat.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/Clustering.h>
#include <faiss/index_io.h>

namespace {

 std::independent_bits_engine<std::mt19937, 8, uint8_t> engine(1);   

void FeatureGenerator(std::vector<uint8_t> &features)
{
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i) {
        features[i] = engine();
    }
}

inline double GetMillisecs()
{
    struct timeval tv {
        0, 0
    };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}





TEST(TestAscendIndexBinaryFlat, QPS)
{
    int dim = 512;
    size_t ntotal = 10000000;
    std::vector<int> searchNum = { 8, 16, 32, 64, 128,256 };

    faiss::ascend::AscendIndexBinaryFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexBinaryFlat index(dim, conf);
    index.verbose = true;

    printf("generate data\n");
    std::vector<uint8_t> base(ntotal * index.code_size,0);
    FeatureGenerator(base);

    printf("add data\n");
    index.add(ntotal, base.data());
    int warmUpTimes = 10 ;
    std::vector<int32_t> distw(127 * 10, 0);
    std::vector<faiss::Index::idx_t> labelw(127 * 10, 0);
    for (int i = 0; i < warmUpTimes; i++) {
        index.search(127, base.data(), 10, distw.data(), labelw.data());
    }
    
    for (size_t n = 0; n < searchNum.size(); n++) {
        int k = 128;
        int loopTimes = 10;
        std::vector<int32_t> dist(searchNum[n] * k, 0);
        std::vector<faiss::Index::idx_t> label(searchNum[n] * k, 0);
        double ts = GetMillisecs();
        for (int l = 0; l < loopTimes; l++) {
            index.search(searchNum[n], base.data(), k, dist.data(), label.data());
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
