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

#include <random>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/index_io.h>


namespace {
typedef std::unordered_map<int,float> recallMap;
inline void GenerateCodes(int8_t *codes, int total, int dim, int seed = -1)
{
    std::default_random_engine e((seed > 0) ? seed : time(nullptr));
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
    for (int i = 0; i < total; i++) {
        for (int j = 0; j < dim; j++) {
            // uint8's max value is 255
            codes[i * dim + j] = static_cast<int8_t>(255 * rCode(e) - 128);
        }
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

inline void AssertInt8Equal(size_t count, const int8_t *gt, const int8_t *data)
{
    for (size_t i = 0; i < count; i++) {
        ASSERT_TRUE(gt[i] == data[i]) << "i: " << i << " gt: " << int(gt[i]) << " data: " << int(data[i]) << std::endl;
    }
}

inline void sqEncode(const faiss::ScalarQuantizer sq, const float *base, int8_t *baseInt8, const size_t n)
{
    std::vector<uint8_t> queryUint8(n * sq.code_size);
    sq.compute_codes(base, queryUint8.data(), n);
    for(size_t i = 0; i < n * sq.code_size; i++){
        baseInt8[i] = queryUint8[i] - 128;
    }
}

template<class T> recallMap calRecall(std::vector<T> label, int64_t *gt, int queryNum)
{
    recallMap Map;
    Map[1] = 0;
    Map[10] = 0;
    Map[100] = 0;
    int k = label.size() / queryNum;

    for(int i = 0; i < queryNum; i++){
        std::set<int> labelSet(label.begin() + i * k, label.begin() + i * k + k);
        if (labelSet.size() != k) {
            printf("current query have duplicated labels!!! \n");
        }
         for(int j = 0; j < k; j++){
            if(gt[i * k] == label[i * k + j]){
                Map[100]++;
                switch (j){
                    case 0:
                        Map[1]++;
                        Map[10]++;
                        break;
                    case 1 ... 9: 
                        Map[10]++;
                        break;
                    default: break;

                }
                break;
            }
         }
    }
    Map[1] = Map[1] / queryNum * 100;
    Map[10] = Map[10] / queryNum * 100;
    Map[100] = Map[100] / queryNum * 100;
    return Map;
}

TEST(TestAscendIndexInt8Flat, QPS)
{
    int dim = 512;
    size_t ntotal = 7000000;
    std::vector<int> searchNum = { 8, 16, 32, 64, 128,256 };

    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 },1024 * 1024 * 1024);
    faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_INNER_PRODUCT, conf);
    index.verbose = true;

    printf("start generate data\n");
    std::vector<int8_t> base(ntotal * dim);
    std::vector<float> baseFp(ntotal * dim);
     for (size_t i = 0; i < ntotal * dim; i++) {
        baseFp[i] = drand48();
     }
    printf("generate data finished\n");
    faiss::ScalarQuantizer sq = faiss::ScalarQuantizer(dim,faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    sq.train(ntotal,baseFp.data());
    sqEncode(sq,baseFp.data(),base.data(),ntotal);
    printf("add data\n");
    index.add(ntotal, base.data());
    int warmUpTimes = 10 ;
    std::vector<float> distw(127 * 10, 0);
    std::vector<faiss::Index::idx_t> labelw(127 * 10, 0);
    for (int i = 0; i < warmUpTimes; i++) {
        index.search(127, base.data(), 10, distw.data(), labelw.data());
    }
    

    for (size_t n = 0; n < searchNum.size(); n++) {
        int k = 100;
        int loopTimes = 10;

        std::vector<float> dist(searchNum[n] * k, 0);
        std::vector<faiss::Index::idx_t> label(searchNum[n] * k, 0);
        double ts = GetMillisecs();
        // start to quantize query data
        std::vector<int8_t> query(searchNum[n] * dim);
        sqEncode(sq,baseFp.data(),query.data(),searchNum[n]);
        
        for (int l = 0; l < loopTimes; l++) {
            index.search(searchNum[n], query.data(), k, dist.data(), label.data());
        }
        double te = GetMillisecs();
        printf("case[%zu]: base:%zu, dim:%d, search num:%d, QPS:%.4f\n", n, ntotal, dim, searchNum[n],
            1000 * searchNum[n] * loopTimes / (te - ts));
    }
    
    
}

TEST(TestAscendIndexInt8Flat, Recall)
{
    int dim = 512;
    size_t ntotal = 7000000;
    int searchNum = 8;

    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 },1024 * 1024 * 1024);
    faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_INNER_PRODUCT, conf);
    index.verbose = true;

    printf("start generate data\n");
    std::vector<int8_t> base(ntotal * dim);
    std::vector<float> baseFp(ntotal * dim);
     for (size_t i = 0; i < ntotal * dim; i++) {
        baseFp[i] = drand48();
     }
    printf("generate data finished\n");
    faiss::ScalarQuantizer sq = faiss::ScalarQuantizer(dim,faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    sq.train(ntotal,baseFp.data());
    sqEncode(sq,baseFp.data(),base.data(),ntotal);
    printf("add data\n");
    index.add(ntotal, base.
    data());

    int k = 100;
    std::vector<faiss::Index::idx_t> gt(searchNum * k, 0);
    for (int i = 0; i < searchNum; i++) {
        gt[i * k] = i;
    }
    std::vector<float> dist(searchNum * k, 0);
    std::vector<faiss::Index::idx_t> label(searchNum * k, 0);
    // start to quantize query data
    std::vector<int8_t> query(searchNum * dim);
    sqEncode(sq,baseFp.data(),query.data(),searchNum);
    index.search(searchNum, query.data(), k, dist.data(), label.data());
    recallMap Top = calRecall(label, gt.data(), searchNum);
    printf("TOPK %d: t1 = %.2f,  t10 = %.2f,t100 = %.2f\n",k,Top[1], Top[10], Top[100]);

}

} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
