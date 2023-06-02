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
 
#include <bitset>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <faiss/ascend/AscendIndexTS.h>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <queue>
#include <random>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

namespace {
using idx_t = int64_t;
using FeatureAttr = faiss::ascend::FeatureAttr;
using AttrFilter = faiss::ascend::AttrFilter;

std::independent_bits_engine<std::mt19937, 8, uint8_t> engine(1);

void FeatureGenerator(std::vector<int8_t> &features)
{
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i) {
        features[i] = engine()-128;
    }
}

void FeatureAttrGenerator(std::vector<FeatureAttr> &attrs)
{
    size_t n = attrs.size();
    for (size_t i = 0; i < n; ++i) {
        attrs[i].time = int32_t(i % 4);
        attrs[i].tokenId = int32_t(i % 4);
    }
}

inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}
} //end of namespace

TEST(TestAscendIndexTS_int8Cos, Init)
{
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    auto ts = GetMillisecs();
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    int res = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(res, 0);
    auto te = GetMillisecs();
    printf("init cost %f ms\n", te - ts);
    delete tsIndex;
}

TEST(TestAscendIndexTS_int8Cos, add)
{
    idx_t ntotal = 1000000;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    faiss::ascend::AscendIndexTS * tsIndex = new faiss::ascend::AscendIndexTS();
    tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);

    std::vector<int8_t> features(ntotal * dim);
    printf("[---add-----------]\n");
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (int i = 0; i < ntotal; ++i) {
        labels.push_back(i);
    }
    std::vector<FeatureAttr>attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto ts = GetMillisecs();
    auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);
    auto te = GetMillisecs();
    printf("add %ld cost %f ms\n", ntotal, te - ts);
 
    delete tsIndex;
}

TEST(TestAscendIndexTS_int8Cos, GetFeatureByLabel)
{
    int dim = 512;
    int maxTokenId = 2500;
    int ntotal = 100000;
    std::vector<int8_t> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(0, dim, maxTokenId, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    std::vector<int8_t>getBase(ntotal * dim);
    auto ts = GetMillisecs();
    ret = index->GetFeatureByLabel(ntotal, label.data(), getBase.data());
    auto te = GetMillisecs();
    printf("GetFeatureByLabel cost total %f ms\n", te - ts);
    EXPECT_EQ(ret, 0);

#pragma omp parallel for if (ntotal > 100)
    for (int i = 0; i < ntotal * dim; i++) {
        EXPECT_EQ(base[i], getBase[i]);
    }
}

TEST(TestAscendIndexTS_int8Cos, DeleteFeatureByLabel)
{
    int dim = 512;
    int maxTokenId = 2500;
    int ntotal = 1000000;
    std::vector<int8_t> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index  = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(0, dim, maxTokenId, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    int64_t validNum = 0;
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);
    int delCount = 1000;
    std::vector<int64_t>delLabel(delCount);
    delLabel.assign(label.begin(), label.begin() + delCount);
    auto ts = GetMillisecs();
    index->DeleteFeatureByLabel(delCount, delLabel.data());
    auto te = GetMillisecs();
    printf("DeleteFeatureByLabel delete cost totoal %f ms\n", te - ts);
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal - delCount);

    index->DeleteFeatureByLabel(delCount, delLabel.data());
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal - delCount);
}

TEST(TestAscendIndexTS_int8Cos, DeleteFeatureByToken)
{
    int dim = 512;
    int maxTokenId = 2500;
    int ntotal = 1000000;
    std::vector<int8_t> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index  = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(0, dim, maxTokenId, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    int64_t validNum = 0;
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);
    std::vector<uint32_t> delToken{0, 1};
    auto ts = GetMillisecs();
    index->DeleteFeatureByToken(2, delToken.data());
    auto te = GetMillisecs();
    printf("DeleteFeatureByToken delete cost totoal %f ms\n", te - ts);
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal / 2);
}

TEST(TestAscendIndexTS_int8Cos, Acc)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
    std::vector<int> topks = { 10 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);

    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;

        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        auto ts0 = GetMillisecs();
        tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        auto te0 = GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }

    for (auto k :topks) {
        int warmupTimes = 3;
        int loopTimes = 2;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, 10);
            std::vector<uint32_t> validnum(queryNum, 0);
            uint32_t size = queryNum * dim;
            std::vector<uint8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
            std::vector<uint8_t> bitSet(setlen, 0);
            bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 3;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;
            
            std::vector<AttrFilter> queryFilters(queryNum, filter);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                if (i == warmupTimes) {
                    continue;
                }
                tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                    distances.data(), validnum.data());
            }
            for (int i = 0; i < queryNum; i++) {
                ASSERT_TRUE(labelRes[i * k] == i);
                ASSERT_TRUE(distances[i * k] > float(0.99) && distances[i * k] < float(1.01));
            }

            bitSet[0] = 0x1 << 0 | 0x1 << 1;
            filter.timesStart = 1;
            filter.timesEnd = 3;

            queryFilters.clear();
            queryFilters.insert(queryFilters.begin(), queryNum, filter);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                if (i == warmupTimes) {
                    continue;
                }
                tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                    distances.data(), validnum.data());
            }
            for (int i = 0; i < queryNum; i++) {
                if (i % 4 == 1) {
                    ASSERT_TRUE(labelRes[i * k] == i);
                    ASSERT_TRUE(distances[i * k] > float(0.99) && distances[i * k] < float(1.01));
                }
                else {
                    ASSERT_TRUE(labelRes[i * k] != i);
                    ASSERT_TRUE(distances[i * k] <= float(0.3));
                }
            }
        }
    }
    delete tsIndex;
}

TEST(TestAscendIndexTS_int8Cos, SearchNoShareQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 10;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
    std::vector<int> topks = { 1024 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);

    std::vector<int8_t> features(ntotal * dim);
    printf("[add -----------\n]");
    FeatureGenerator(features);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;

        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        auto ts0 = GetMillisecs();
        auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        EXPECT_EQ(res, 0);
        auto te0 = GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }
    long double ts { 0. };
    long double te { 0. };
    for (auto k : topks) {
        int warmupTimes = 3;
        int loopTimes = 2;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, -1);
            std::vector<uint32_t> validnum(queryNum, 0);
            uint32_t size = queryNum * dim;
            std::vector<uint8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
            std::vector<uint8_t> bitSet(setlen, 0);

            bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2;

            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 100;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;

            std::vector<AttrFilter> queryFilters(queryNum, filter);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                if (i == warmupTimes) {
                    ts = GetMillisecs();
                }
                tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                    distances.data(), validnum.data());
            }
            te = GetMillisecs();

            printf("base: %ld, dim: %d, batch: %4d, top%d, QPS:%7.2Lf\n", ntotal * addNum, dim, queryNum, k,
                (long double)1000.0 * queryNum * loopTimes / (te - ts));
        }
    }

    delete tsIndex;
}

TEST(TestAscendIndexTS_int8Cos, SearchShareQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 10;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
    std::vector<int> topks = { 1024 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);

    EXPECT_EQ(ret, 0);

    std::vector<int8_t> features(ntotal * dim);
    printf("[add -----------]\n");
    FeatureGenerator(features);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;

        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
         auto ts0 = GetMillisecs();
        auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        EXPECT_EQ(res, 0);
        auto te0 = GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }

    long double ts { 0. };
    long double te { 0. };
    for (auto k : topks) {
        int warmupTimes = 3;
        int loopTimes = 2;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, -1);
            std::vector<uint32_t> validnum(queryNum, 1);
            uint32_t size = queryNum * dim;
            std::vector<int8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
            std::vector<uint8_t> bitSet(setlen, 0);

            bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2;

            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 100;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;

            std::vector<AttrFilter> queryFilters(queryNum, filter);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                if (i == warmupTimes) {
                    ts = GetMillisecs();
                }
                tsIndex->Search(queryNum, querys.data(), queryFilters.data(), true, k, labelRes.data(),
                    distances.data(), validnum.data());
            }
            te = GetMillisecs();
            printf("base: %ld, dim: %d, batch: %4d, top%d, QPS:%7.2Lf\n", ntotal * addNum, dim, queryNum, k,
                (long double)1000.0 * queryNum * loopTimes / (te - ts));
        }
    }

    delete tsIndex;
}

TEST(TestAscendIndexTS_int8Cos, SearchShareWithExtraQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 10;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
    std::vector<int> topks = { 100 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);

    EXPECT_EQ(ret, 0);
	
	printf("[add -----------]\n");
    std::vector<int8_t> features(ntotal * dim);   
    FeatureGenerator(features);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;

        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        auto ts0 = GetMillisecs();
        auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        EXPECT_EQ(res, 0);
        auto te0 = GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }

    long double ts { 0. };
    long double te { 0. };
    for (auto k : topks) {
        int warmupTimes = 3;
        int loopTimes = 10;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, -1);
            std::vector<uint32_t> validnum(queryNum, 1);
            uint32_t size = queryNum * dim;
            std::vector<int8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
            std::vector<uint8_t> bitSet(setlen, ~0);
            //  00000111  means index lable  0,1,2 has been chosed;
            // bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2;

            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 3;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;


            int extraMaskLen = 12;
            std::vector<uint8_t> extraMask(queryNum * extraMaskLen, 0);
            int ind = 1;
            for (int i = 0; i < queryNum; i++) {
            	for (int j = 0; j < extraMaskLen; j++) {
            		extraMask[i * extraMaskLen + j] = 0x1 << (ind % 8);
            		ind++;
            	}
            }

            std::vector<AttrFilter> queryFilters(queryNum, filter);
            for (int i = 0; i < warmupTimes; i++) {
            	tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), true, k, 
							            	 extraMask.data(), extraMaskLen, false,
							            	 labelRes.data(), distances.data(), validnum.data());
            }

            ts = GetMillisecs();
            for (int i = 0; i < loopTimes; i++) {
            	tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), true, k, 
							            	 extraMask.data(), extraMaskLen, false,
							            	 labelRes.data(), distances.data(), validnum.data());
            }
            te = GetMillisecs();

            printf("base: %ld, dim: %d, batch: %4d, top%d, QPS:%7.2Lf\n", ntotal * addNum, dim, queryNum, k,
                   (long double)1000.0 * queryNum * loopTimes / (te - ts));
        }
    }

    delete tsIndex;
}




int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
