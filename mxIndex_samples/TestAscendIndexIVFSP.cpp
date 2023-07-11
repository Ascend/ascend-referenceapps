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

#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include <map>
#include <string>
#include <fstream>
#include <random>
#include <ctime>
#include <climits>
#include <cstdio>
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <gtest/gtest.h>
#include <faiss/index_io.h>
#include <faiss/ascend/AscendIndexIVFSP.h>
#include <faiss/ascend/AscendMultiIndexSearch.h>
#include <securec.h>

namespace {

inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

using recallMap = std::unordered_map<int, float>;

// camera id 简写为 cid， 7位数， 取值范围0~127

const int32_t K_MAX_CAMERA_NUM = 128;
const int MASK_LEN = 8;

struct IDFilter {
    IDFilter()
    {
        memset_s(cameraIdMask, sizeof(cameraIdMask) / sizeof(cameraIdMask[0]),
            static_cast<uint8_t>(0), K_MAX_CAMERA_NUM / MASK_LEN);
        timeRange[0] = 0;
        timeRange[1] = -1;
    }

    // 一个IDFilter对象是可以涵盖处理所有cid in [0, 127] 共128个camera
    uint8_t cameraIdMask[K_MAX_CAMERA_NUM / MASK_LEN] = {0};
    uint32_t timeRange[2] = {0};
};

// batch即searchNum， 一条被检索的特征向量，传递一个IDFilter对象
// std::vector<int> &cids, 是一个固定的128元素的向量，其值从0到127
void ConstructCidFilter(IDFilter *idFilters, int batch, const std::vector<int> &cids,
    const std::vector<uint32_t> &timestamps)
{
    for (int i = 0; i < batch; ++i) {
        for (auto current_cid : cids) {
            int g = current_cid / MASK_LEN;
            int k = current_cid % MASK_LEN;
            idFilters[i].cameraIdMask[g] += (1 << k);
        }
        idFilters[i].timeRange[0] = timestamps[0]; // start
        idFilters[i].timeRange[1] = timestamps[1]; // end
    }
}

void ConstructCidFilter2(IDFilter *idFilters, int batch, const std::vector<int> &cids,
    const std::vector<uint32_t> &timestamps)
{
    for (int i = 0; i < batch; ++i) {
        for (auto current_cid : cids) {
            int g = current_cid / MASK_LEN;
            int k = current_cid % MASK_LEN;
        if (k % 2 == 0) {
            idFilters[i].cameraIdMask[g] += (1 << k);
        } else {
            idFilters[i].cameraIdMask[g] += 0;
        }
        }
        idFilters[i].timeRange[0] = timestamps[0]; // start
        idFilters[i].timeRange[1] = timestamps[1]; // end
    }
}

void printDistAndLabel(std::vector<faiss::Index::idx_t> &labels, std::vector<float> &dist,
    int queryNum, int printqn, int k, int printk)
{
    auto lbl = labels.data();
    auto dst = dist.data();
    for (int i = 0; i < printqn; i++) {
        for (int j = 0; j < printk; j++) {
            int64_t id = *(lbl + i * k +j);
            int64_t cid = (id >> 42) & 0x7F;
            int64_t id_ = (id >> 10) & 0xFFFFFFFF;
            if (j == 0) {
                printf("query: %d <%ld, %ld, %.4f>\n", i, cid, id_, *(dst + i * k + j));
            } else {
                printf("<%ld, %ld, %.4f>\n", cid, id_, *(dst + i * k + j));
            }
        }
        printf("\n");
    }
}

void ConstructCidFilter(IDFilter *idFilters, int staIdx, int batch, const std::vector<int> &cids,
    const std::vector<uint32_t> &timestamps)
{
    for (int i = 0; i < batch; ++i) {
        for (auto current_cid : cids) {
            int g = current_cid / MASK_LEN;
            int k = current_cid % MASK_LEN;
            idFilters[staIdx + i].cameraIdMask[g] += (1 << k);
        }
        idFilters[staIdx + i].timeRange[0] = timestamps[0]; // start
        idFilters[staIdx + i].timeRange[1] = timestamps[1]; // end
    }
}

//  Calculate Recall

template<class T>
recallMap calRecall(std::vector<T> &label, int64_t* gt, int shape)
{
    recallMap Map;
    Map[1] = 0;
    Map[10] = 0;
    Map[100] = 0;
    int k = label.size() / shape;

    for (int i = 0; i < shape; i++) {
        std::set<int> labelSet(label.begin() + i * k, label.begin() + i * k + k);
        if (labelSet.size()  != static_cast<size_t>(k)) {
            std::cout << "Current Query has duplicated labels!!!\n" << std::endl;
        }

        for (int j = 0; j < k; j++) {
            if (gt[i * k] == label[i * k + j])  { // 被检索的query[i]这条向量， 如果在检索得到的topk条结果中有任意一条
                                                  // label[i * k + j] 等于 gt[i * k], (topk 命中)， 则表示命中
                Map[100]++;
                switch (j) {
                    case 0:
                        Map[1]++;
                        Map[10]++;
                        break;
                    case 1 ... 9:
                        Map[10]++;
                        break;
                    default:
                        break;
                }
                break;
            }
        }
    }
    Map[1] = Map[1] / shape * 100;
    Map[10] = Map[10] / shape * 100;
    Map[100] = Map[100] / shape * 100;
    return Map;
}

template<class T>
double calNewRecallHelper(std::vector<T> &label, int64_t* gt, int shape, int p)
{
    int k = label.size() /shape;
    double result = 0.0;
    for (int i = 0;i < shape; ++i) {
        int hit = 0;
        int validGtNum = 0;
        auto it = label.begin() + i * k;
        for (int j = 0; j < k; ++j) {
            int64_t oneGt = gt[i * k + j];
            if (oneGt == -1) {
                break;
            }
            validGtNum += 1;
            if (std::find(it, it + p, oneGt) != it + p) {
                hit += 1;
            }
        }
        if (validGtNum > 0) {
            result += static_cast<double>(hit) / validGtNum;
        }
    }
    return result / shape * 100;
}

template<class T>
double calNewRecallHelper2(std::vector<T> &label, int64_t* gt, int shape, int p)
{
    int k = label.size() /shape;
    double result = 0.0;
    for (int i = 0;i < shape; i++) {
        int hit = 0;
        int validGtNum = 0;
        auto it = label.begin() + i * k;
        for (int j = 0; j < p; ++j) {
            int64_t oneGt = gt[i * k + j];
            if (oneGt == -1) {
                break;
            }
            validGtNum += 1;
            if (std::find(it, it + p, oneGt) != it + p) {
                hit += 1;
            }
        }
        if (validGtNum > 0) {
            result += static_cast<double>(hit) / validGtNum;
        }
    }
    return result / shape * 100;
}

//  Calculate New Recall

template<class T>
recallMap calNewRecall(std::vector<T> &label, int64_t* gt, int shape)
{
    recallMap Map;
    Map[1] = calNewRecallHelper(label, gt, shape, 1);
    Map[10] = calNewRecallHelper(label, gt, shape, 10);
    Map[100] = calNewRecallHelper(label, gt, shape, 100);
    return Map;
}

template<class T>
recallMap calNewRecall2(std::vector<T> &label, int64_t* gt, int shape)
{
    recallMap Map;
    Map[1] = calNewRecallHelper2(label, gt, shape, 1);
    Map[10] = calNewRecallHelper2(label, gt, shape, 10);
    Map[100] = calNewRecallHelper2(label, gt, shape, 100);
    return Map;
}

template<class T>
void printRecall(std::vector<T> &labels, int64_t* gt, int shape, int bs)
{
    std::cout << "-------------calRecall-------------------" << std::endl;
    recallMap Map = calRecall(labels, gt, shape);
    std::cout << "batchSize: " << bs << std::endl;
    std::cout << "recall 1@1: " << Map[1] << std::endl;
    std::cout << "recall 1@10: " << Map[10] << std::endl;
    std::cout << "recall 1@100: " << Map[100] << std::endl;
}

template<class T>
void printMultiRecall(int indexNum, std::vector<T> &labels, int64_t* gt, int bs, int k)
{
    recallMap Map;
    Map[1] = 0;
    Map[10] = 0;
    Map[100] = 0;
    for (int i = 0; i < indexNum; i++) {
        std::vector<T> subLabels(labels.cbegin() + i * bs * k, labels.cbegin() + (i + 1) * bs * k);
        recallMap subMap = calRecall(subLabels, gt, bs);
        Map[1] += subMap[1];
        Map[10] += subMap[10];
        Map[100] += subMap[100];
    }

    Map[1] = Map[1] / indexNum;
    Map[10] = Map[10] / indexNum;
    Map[100] = Map[100] / indexNum;

    std::cout << "-------------calRecall-------------------" << std::endl;
    std::cout << "batchSize: " << bs << std::endl;
    std::cout << "recall 1@1: " << Map[1] << std::endl;
    std::cout << "recall 1@10: " << Map[10] << std::endl;
    std::cout << "recall 1@100: " << Map[100] << std::endl;
}

template<class T>
void printMultiRecallNew(int indexNum, std::vector<T> &labels, int64_t* gt, int bs, int batchNum, int k)
{
    recallMap Map;
    Map[1] = 0;
    Map[10] = 0;
    Map[100] = 0;
    for (int batchIdx = 0; batchIdx < batchNum; batchIdx++) {
        for (int i = 0; i < indexNum; i++) {
        std::vector<T> subLabels(labels.cbegin() + batchIdx * bs * indexNum * k + i * bs * k,
        labels.cbegin() + batchIdx * bs * indexNum * k + (i + 1) * bs * k);
        recallMap subMap = calRecall(subLabels, gt + batchIdx * bs * k, bs);
        Map[1] += subMap[1];
        Map[10] += subMap[10];
        Map[100] += subMap[100];
        }
    }

    Map[1] = Map[1] / (batchNum * indexNum);
    Map[10] = Map[10] / (batchNum * indexNum);
    Map[100] = Map[100] / (batchNum * indexNum);

    std::cout << "-------------calRecall-------------------" << std::endl;
    std::cout << "batchSize: " << bs << std::endl;
    std::cout << "recall 1@1: " << Map[1] << std::endl;
    std::cout << "recall 1@10: " << Map[10] << std::endl;
    std::cout << "recall 1@100: " << Map[100] << std::endl;
}

TEST(TestAscendIndexIVFSP, Reset)
{
    int codeBookDim = 256;
    int dim = 256;
    int nonzeroNum = 64;
    int ncentroids = 256; // nlist
    int nprobe = 64;
    int codebookNum = ncentroids;
    size_t nCodeBook = codebookNum * nonzeroNum;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32};
    int searchListSize = 32768;
    int ntotal = 2000000;
    int k = 100;
    int queryNum = 5306;

    // 码本，训练数据， 查询数据路径。
    std::string basePath = " ";
    // 特征底库数据
    std::string dataPath = basePath + "base.bin";
    // 查询数据
    std::string queryPath = basePath + "query.bin";
    // （grouptruth） 测试召回的比较数据
    std::string gtsPath = basePath + "gt.bin";
    // codeBook 码本
    std::string codeBookPath = basePath + "codebook.bin";
    // ivfsp 数据，有：加载， 没有就添加并保存
    std::string indexPath = basePath + "myivfsp_base_data.bin";


    // code offset
    std::vector<faiss::Index::idx_t> offset(codebookNum, 0);
    std::iota(offset.begin(), offset.end(), 0);

    // code book
    std::vector<float> codeBookData(nCodeBook * codeBookDim);
    std::ifstream fin(codeBookPath.c_str(), std::ios::binary);
    fin.read((char*)(codeBookData.data()), sizeof(float) * nCodeBook * codeBookDim);
    fin.close();

    // base data
    std::vector<float> data(dim * ntotal);
    std::ifstream codesFin(dataPath.c_str(), std::ios::binary);
    codesFin.read((char*)(data.data()), sizeof(float) * dim * ntotal);
    codesFin.close();

    // ids
    std::vector<faiss::Index::idx_t> ids(ntotal, 0);
    std::iota(ids.begin(), ids.end(), 0);

    // query data
    std::vector<float> qData(queryNum * dim);
    std::ifstream queryFin(queryPath.c_str(), std::ios::binary);
    queryFin.read((char*)(qData.data()), sizeof(float) * queryNum * dim);
    queryFin.close();

    std::vector<int64_t> gts(queryNum * k, 0);
    std::ifstream gtsFin(gtsPath.c_str(), std::ios::binary);
    gtsFin.read((char*)(gts.data()), sizeof(int64_t) * queryNum * k);
    gtsFin.close();

    faiss::ascend::AscendIndexIVFSPConfig conf({0});
    conf.nprobe = nprobe; // 64 32 128    16的倍数，且0 < nprobe <= nlist
    conf.handleBatch = nprobe; // 和OM算子保持一致
    conf.searchListSize = searchListSize; // 大于等于512 且为2的幂次。
    conf.filterable = true;

    faiss::ascend::AscendIndexIVFSP index(dim, nonzeroNum, ncentroids,
            codeBookPath.c_str(),
            faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            faiss::MetricType::METRIC_L2, conf);
    index.setVerbose(true);
    index.add_with_ids(ntotal, data.data(), ids.data());
    EXPECT_EQ(index.ntotal, ntotal);
    std::cout << "add" << std::endl;
    std::cout << "index.ntotal: " << index.ntotal << std::endl;

    // reset
    index.reset();
    ASSERT_EQ(index.ntotal, 0);
    std::cout << "reset the data, now ntotal:" << index.ntotal << std::endl;

    index.add(ntotal, data.data());
    EXPECT_EQ(index.ntotal, ntotal);
    std::cout << "second add" << std::endl;
    std::cout << "Now, index.ntotal: " << index.ntotal << std::endl;
}

TEST(TestAscendIndexIVFSP, RemoveBatch)
{
    int codeBookDim = 256;
    int dim = 256;
    int nonzeroNum = 64;
    int ncentroids = 256; // nlist
    int nprobe = 64;
    int codebookNum = ncentroids;
    size_t nCodeBook = codebookNum * nonzeroNum;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32};
    int searchListSize = 32768;
    int ntotal = 2000000;
    int k = 100;
    int queryNum = 5306;

    // 码本，训练数据， 查询数据路径。
    std::string basePath = " ";
    // 特征底库数据
    std::string dataPath = basePath + "base.bin";
    // 查询数据
    std::string queryPath = basePath + "query.bin";
    // （grouptruth） 测试召回的比较数据
    std::string gtsPath = basePath + "gt.bin";
    // codeBook 码本
    std::string codeBookPath = basePath + "codebook.bin";
    // ivfsp 数据，有：加载， 没有就添加并保存
    std::string indexPath = basePath + "myivfsp_base_data.bin";


    // code offset
    std::vector<faiss::Index::idx_t> offset(codebookNum, 0);
    std::iota(offset.begin(), offset.end(), 0);

    // code book
    std::vector<float> codeBookData(nCodeBook * codeBookDim);
    std::ifstream fin(codeBookPath.c_str(), std::ios::binary);
    fin.read((char*)(codeBookData.data()), sizeof(float) * nCodeBook * codeBookDim);
    fin.close();

    // base data
    std::vector<float> data(dim * ntotal);
    std::ifstream codesFin(dataPath.c_str(), std::ios::binary);
    codesFin.read((char*)(data.data()), sizeof(float) * dim * ntotal);
    codesFin.close();

    // ids
    std::vector<faiss::Index::idx_t> ids(ntotal, 0);
    std::iota(ids.begin(), ids.end(), 0);

    // query data
    std::vector<float> qData(queryNum * dim);
    std::ifstream queryFin(queryPath.c_str(), std::ios::binary);
    queryFin.read((char*)(qData.data()), sizeof(float) * queryNum * dim);
    queryFin.close();

    std::vector<int64_t> gts(queryNum * k, 0);
    std::ifstream gtsFin(gtsPath.c_str(), std::ios::binary);
    gtsFin.read((char*)(gts.data()), sizeof(int64_t) * queryNum * k);
    gtsFin.close();

    faiss::ascend::AscendIndexIVFSPConfig conf({0});
    conf.nprobe = nprobe; // 64 32 128    16的倍数，且0 < nprobe <= nlist
    conf.handleBatch = nprobe; // 和OM算子保持一致
    conf.searchListSize = searchListSize; // 大于等于512 且为2的幂次。
    conf.filterable = true;

    faiss::ascend::AscendIndexIVFSP index(dim, nonzeroNum, ncentroids,
            codeBookPath.c_str(),
            faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            faiss::MetricType::METRIC_L2, conf);
    index.setVerbose(true);
    if (FILE *file = fopen(indexPath.c_str(), "r")) {
        fclose(file);
        index.loadAllData(indexPath.c_str());
        std::cout << "loadAllData from " << indexPath << std::endl;
        std::cout << "index.ntotal: " << index.ntotal << std::endl;
    } else {
        index.add_with_ids(ntotal, data.data(), ids.data());
        std::cout << "add" << std::endl;
        std::cout << "index.ntotal: " << index.ntotal << std::endl;
        EXPECT_EQ(index.ntotal, ntotal);
        index.saveAllData(indexPath.c_str());
        std::cout << "saveAllData to " << indexPath << std::endl;
    }

    // remove batch
    std::vector<faiss::Index::idx_t> delBatches = {500, 1000, 510, 2000, 10000};
    faiss::IDSelectorBatch del(delBatches.size(), delBatches.data());
    int rmCnt = 0;
    for (int i = 0; i < ntotal; i++) {
        rmCnt += del.is_member(ids[i]) ? 1 : 0;
    }
    int rmedCnt = index.remove_ids(del);
    ASSERT_EQ(rmCnt, rmedCnt);
    ASSERT_EQ(index.ntotal, (ntotal - rmedCnt));
    std::cout << "remove with ids:" << rmedCnt << std::endl;
    std::cout << "index.ntotal: " << index.ntotal << std::endl;
}

TEST(TestAscendIndexIVFSP, RemoveRange)
{
    int codeBookDim = 256;
    int dim = 256;
    int nonzeroNum = 64;
    int ncentroids = 256; // nlist
    int nprobe = 64;
    int codebookNum = ncentroids;
    size_t nCodeBook = codebookNum * nonzeroNum;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32};
    int searchListSize = 32768;
    int ntotal = 2000000;
    int k = 100;
    int queryNum = 5306;

    // 码本，训练数据， 查询数据路径。
    std::string basePath = " ";
    // 特征底库数据
    std::string dataPath = basePath + "base.bin";
    // 查询数据
    std::string queryPath = basePath + "query.bin";
    // （grouptruth） 测试召回的比较数据
    std::string gtsPath = basePath + "gt.bin";
    // codeBook 码本
    std::string codeBookPath = basePath + "codebook.bin";
    // ivfsp 数据，有：加载， 没有就添加并保存
    std::string indexPath = basePath + "myivfsp_base_data.bin";


    // code offset
    std::vector<faiss::Index::idx_t> offset(codebookNum, 0);
    std::iota(offset.begin(), offset.end(), 0);

    // code book
    std::vector<float> codeBookData(nCodeBook * codeBookDim);
    std::ifstream fin(codeBookPath.c_str(), std::ios::binary);
    fin.read((char*)(codeBookData.data()), sizeof(float) * nCodeBook * codeBookDim);
    fin.close();

    // base data
    std::vector<float> data(dim * ntotal);
    std::ifstream codesFin(dataPath.c_str(), std::ios::binary);
    codesFin.read((char*)(data.data()), sizeof(float) * dim * ntotal);
    codesFin.close();

    // ids
    std::vector<faiss::Index::idx_t> ids(ntotal, 0);
    std::iota(ids.begin(), ids.end(), 0);

    // query data
    std::vector<float> qData(queryNum * dim);
    std::ifstream queryFin(queryPath.c_str(), std::ios::binary);
    queryFin.read((char*)(qData.data()), sizeof(float) * queryNum * dim);
    queryFin.close();

    std::vector<int64_t> gts(queryNum * k, 0);
    std::ifstream gtsFin(gtsPath.c_str(), std::ios::binary);
    gtsFin.read((char*)(gts.data()), sizeof(int64_t) * queryNum * k);
    gtsFin.close();

    faiss::ascend::AscendIndexIVFSPConfig conf({0});
    conf.nprobe = nprobe; // 64 32 128    16的倍数，且0 < nprobe <= nlist
    conf.handleBatch = nprobe; // 和OM算子保持一致
    conf.searchListSize = searchListSize; // 大于等于512 且为2的幂次。
    conf.filterable = true;

    faiss::ascend::AscendIndexIVFSP index(dim, nonzeroNum, ncentroids,
            codeBookPath.c_str(),
            faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            faiss::MetricType::METRIC_L2, conf);
    index.setVerbose(true);
    if (FILE *file = fopen(indexPath.c_str(), "r")) {
        fclose(file);
        index.loadAllData(indexPath.c_str());
        std::cout << "loadAllData from " << indexPath << std::endl;
        std::cout << "index.ntotal: " << index.ntotal << std::endl;
    } else {
        index.add_with_ids(ntotal, data.data(), ids.data());
        std::cout << "add" << std::endl;
        std::cout << "index.ntotal: " << index.ntotal << std::endl;
        EXPECT_EQ(index.ntotal, ntotal);
        index.saveAllData(indexPath.c_str());
        std::cout << "saveAllData to " << indexPath << std::endl;
    }

    // remove range
    const int rangeDel = 20000;
    faiss::Index::idx_t idmin = static_cast<faiss::Index::idx_t>(50 * (rand() / (RAND_MAX +0.0f)));
    faiss::IDSelectorRange del(idmin, idmin + rangeDel);
    int rmCnt = 0;
    for (int i = 0; i < ntotal; i++) {
        rmCnt += del.is_member(ids[i]) ? 1 : 0;
    }
    int rmedCnt = index.remove_ids(del);
    ASSERT_EQ(rmCnt, rmedCnt);
    ASSERT_EQ(index.ntotal, (ntotal - rmedCnt));
    std::cout << "remove with ids:" << rmedCnt << std::endl;
    std::cout << "index.ntotal: " << index.ntotal << std::endl;
}

TEST(TestAscendIndexIVFSP, RecallAndRecallFilter)
{
    int codeBookDim = 256;
    int dim = 256;
    int nonzeroNum = 64;
    int ncentroids = 256; // nlist
    int nprobe = 64;
    int codebookNum = ncentroids;
    size_t nCodeBook = codebookNum * nonzeroNum;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32};
    int searchListSize = 32768;
    int ntotal = 2000000;
    int k = 100;
    int queryNum = 5306;

    // 码本，训练数据， 查询数据路径。
    std::string basePath = " ";
    // 特征底库数据
    std::string dataPath = basePath + "base.bin";
    // 查询数据
    std::string queryPath = basePath + "query.bin";
    // （grouptruth） 测试召回的比较数据
    std::string gtsPath = basePath + "gt.bin";
    // codeBook 码本
    std::string codeBookPath = basePath + "codebook.bin";
    // ivfsp 数据，有：加载， 没有就添加并保存
    std::string indexPath = basePath + "myivfsp_base_data.bin";


    // code offset
    std::vector<faiss::Index::idx_t> offset(codebookNum, 0);
    std::iota(offset.begin(), offset.end(), 0);

    // code book
    std::vector<float> codeBookData(nCodeBook * codeBookDim);
    std::ifstream fin(codeBookPath.c_str(), std::ios::binary);
    fin.read((char*)(codeBookData.data()), sizeof(float) * nCodeBook * codeBookDim);
    fin.close();

    // base data
    std::vector<float> data(dim * ntotal);
    std::ifstream codesFin(dataPath.c_str(), std::ios::binary);
    codesFin.read((char*)(data.data()), sizeof(float) * dim * ntotal);
    codesFin.close();

    // ids
    std::vector<faiss::Index::idx_t> ids(ntotal, 0);
    std::iota(ids.begin(), ids.end(), 0);

    // query data
    std::vector<float> qData(queryNum * dim);
    std::ifstream queryFin(queryPath.c_str(), std::ios::binary);
    queryFin.read((char*)(qData.data()), sizeof(float) * queryNum * dim);
    queryFin.close();

    std::vector<int64_t> gts(queryNum * k, 0);
    std::ifstream gtsFin(gtsPath.c_str(), std::ios::binary);
    gtsFin.read((char*)(gts.data()), sizeof(int64_t) * queryNum * k);
    gtsFin.close();

    faiss::ascend::AscendIndexIVFSPConfig conf({0});
    conf.nprobe = nprobe; // 64 32 128    16的倍数，且0 < nprobe <= nlist
    conf.handleBatch = nprobe; // 和OM算子保持一致
    conf.searchListSize = searchListSize; // 大于等于512 且为2的幂次。
    conf.filterable = true;

    faiss::ascend::AscendIndexIVFSP index(dim, nonzeroNum, ncentroids,
            codeBookPath.c_str(),
            faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            faiss::MetricType::METRIC_L2, conf);
    index.setVerbose(true);
    index.add(ntotal, data.data());
    std::cout << "add" << std::endl;
    std::cout << "index.ntotal: " << index.ntotal << std::endl;
    index.saveAllData(indexPath.c_str());
    std::cout << "saveAllData to " << indexPath << std::endl;

    std::vector<int> nprobeList = {nprobe, nprobe * 2, nprobe / 2};
    for (int tmpNprobe : nprobeList) {
        if (tmpNprobe != nprobe) {
            printf("-------------set nprobe: %d-------------------\n", tmpNprobe);
            index.setNumProbes(tmpNprobe);
        }
        printf("-------------search-------------------\n");
        for (int batch : batches) {
            int loop = queryNum / batch;
            std::vector<float> dist(loop * k * batch, 0);
            std::vector<faiss::Index::idx_t> labels(loop * k * batch, 0);
            double ts = GetMillisecs();
            for (int i = 0; i < loop; i++) {
                index.search(batch, qData.data() + i * batch * dim, k,
                    dist.data() + i * k * batch, labels.data() + i * k * batch);
            }
            double te = GetMillisecs();
            printf("search, k: %d, base: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n",
                k, ntotal, dim, batch, loop * batch, 1000 * loop * batch / (te -ts));
            printRecall (labels, gts.data(), batch * loop, batch);
        }

        std::vector<int> search_cid(K_MAX_CAMERA_NUM, 0);
        std::iota(search_cid.begin(), search_cid.end(), 0);
        std::vector<uint32_t> search_time = {0, static_cast<uint32_t>(ntotal)};

        printf("-------------Search with Filter-------------------\n");
        for (int batch : batches) {
            IDFilter idFilters[batch];
            void *pFilter = &idFilters[0];
            ConstructCidFilter(idFilters, batch, search_cid, search_time);
            int loop = queryNum / batch;
            std::vector<float> dist4filter(loop * k * batch, 0);
            std::vector<faiss::Index::idx_t> labels4filter(loop * k * batch, 0);
            double ts = GetMillisecs();
            for (int i = 0; i < loop; i++) {
                index.search_with_filter(batch, qData.data() + i * batch * dim, k,
                    dist4filter.data() + i * k * batch, labels4filter.data() + i * k * batch, pFilter);
            }
            double te = GetMillisecs();
            printf("search with filter, k: %d, base: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n",
                k, ntotal, dim, batch, loop * batch, 1000 * loop * batch / (te -ts));
            printRecall (labels4filter, gts.data(), batch * loop, batch);
        }
    }
}

TEST(TestAscendIndexIVFSP, MultiSearchAndMultiSearchFilterQPS)
{
    int codeBookDim = 256;
    int dim = 256;
    int nonzeroNum = 64;
    int ncentroids = 256; // nlist
    int nprobe = 64;
    int codebookNum = ncentroids;
    size_t nCodeBook = codebookNum * nonzeroNum;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32};
    int searchListSize = 32768;
    int ntotal = 2000000;
    int indexNum = 200;
    int k = 100;
    int queryNum = 5306;

    // 码本，训练数据， 查询数据路径。
    std::string basePath = " ";
    // 特征底库数据
    std::string dataPath = basePath + "base.bin";
    // 查询数据
    std::string queryPath = basePath + "query.bin";
    // （grouptruth） 测试召回的比较数据
    std::string gtsPath = basePath + "gt.bin";
    // codeBook 码本
    std::string codeBookPath = basePath + "codebook.bin";
    // ivfsp 数据，有：加载， 没有就添加并保存
    std::string indexPath = basePath + "myivfsp_base_data.bin";


    // code offset
    std::vector<faiss::Index::idx_t> offset(codebookNum, 0);
    std::iota(offset.begin(), offset.end(), 0);

    // code book
    std::vector<float> codeBookData(nCodeBook * codeBookDim);
    std::ifstream fin(codeBookPath.c_str(), std::ios::binary);
    fin.read((char*)(codeBookData.data()), sizeof(float) * nCodeBook * codeBookDim);
    fin.close();

    // base data
    std::vector<float> data(dim * ntotal);
    std::ifstream codesFin(dataPath.c_str(), std::ios::binary);
    codesFin.read((char*)(data.data()), sizeof(float) * dim * ntotal);
    codesFin.close();

    // ids
    std::vector<faiss::Index::idx_t> ids(ntotal, 0);
    std::iota(ids.begin(), ids.end(), 0);

    // query data
    std::vector<float> qData(queryNum * dim);
    std::ifstream queryFin(queryPath.c_str(), std::ios::binary);
    queryFin.read((char*)(qData.data()), sizeof(float) * queryNum * dim);
    queryFin.close();

    std::vector<int64_t> gts(queryNum * k, 0);
    std::ifstream gtsFin(gtsPath.c_str(), std::ios::binary);
    gtsFin.read((char*)(gts.data()), sizeof(int64_t) * queryNum * k);
    gtsFin.close();

    faiss::ascend::AscendIndexIVFSPConfig conf({0});
    conf.nprobe = nprobe; // 64 32 128    16的倍数，且0 < nprobe <= nlist
    conf.handleBatch = nprobe; // 和OM算子保持一致
    conf.searchListSize = searchListSize; // 大于等于512 且为2的幂次。
    conf.filterable = true;

    std::vector<faiss::ascend::AscendIndex*> indexes;
    for (int i = 0; i < indexNum; ++i) {
        auto index = new faiss::ascend::AscendIndexIVFSP(dim, nonzeroNum, ncentroids,
            i == 0 ? codeBookPath.c_str():nullptr,
            faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            faiss::MetricType::METRIC_L2, conf);
        ASSERT_FALSE(index == nullptr);
        index->setVerbose(true);
        indexes.emplace_back(index);
        printf("creat index:%d\n", i);
    }

    struct stat indexPathStat;
    if (lstat(indexPath.c_str(), &indexPathStat) == 0) {
        remove(indexPath.c_str());
    }
    for (int i = 0; i < indexNum; ++i) {
        faiss::ascend::AscendIndexIVFSP* index = dynamic_cast<faiss::ascend::AscendIndexIVFSP*>(indexes[i]);
        if (FILE *file = fopen(indexPath.c_str(), "r")) {
            fclose(file);
            index->loadAllData(indexPath.c_str());
            std::cout << "loadAllData from " << indexPath << std::endl;
            std::cout << "index.ntotal: " << index->ntotal << std::endl;
        } else {
            index->add(ntotal, data.data());
            std::cout << "add" << std::endl;
            std::cout << "index.ntotal: " << index->ntotal << std::endl;
            index->saveAllData(indexPath.c_str());
            std::cout << "saveAllData to " << indexPath << std::endl;
        }
    }
    int loopTimes = 100;
    printf("-------------MultiSearch-------------------\n");
    for (size_t j = 0; j < batches.size(); j++) {
        std::vector<float> dist(indexNum * k * batches[j], 0);
        std::vector<faiss::Index::idx_t> label(indexNum * k * batches[j], 0);
        double ts = GetMillisecs();
        for (int l = 0; l < loopTimes; l++) {
        Search(indexes, batches[j], qData.data(), k, dist.data(), label.data(), false);
        }
        double te = GetMillisecs();
        printf("multi search: true, index num: %d, k: %d, base: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n",
                indexNum, k, ntotal, dim, batches[j], queryNum, 1000 * batches[j] * loopTimes / (te -ts));
    }

    printf("-------------MultiSearchFilter-------------------\n");
    std::vector<int> search_cid(K_MAX_CAMERA_NUM, 0);
    std::iota(search_cid.begin(), search_cid.end(), 0);
    std::vector<uint32_t> search_time = {0, static_cast<uint32_t>(ntotal)};
    
    for (size_t j = 0; j < batches.size(); j++) {
        IDFilter idFilters[batches[j]];
        void *pFilter = &idFilters[0];
        ConstructCidFilter(idFilters, batches[j], search_cid, search_time);

        std::vector<float> dist(indexNum * k * batches[j], 0);
        std::vector<faiss::Index::idx_t> label(indexNum * k * batches[j], 0);
        double ts = GetMillisecs();
        for (int l = 0; l < loopTimes; l++) {
            SearchWithFilter(indexes, batches[j], qData.data(),
                k, dist.data(), label.data(), pFilter, false);
        }
        double te = GetMillisecs();
        printf("multi search filter: true, index num: %d, k: %d, base: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n",
                indexNum, k, ntotal, dim, batches[j], queryNum, 1000 * batches[j] * loopTimes / (te -ts));
    }

    printf("-------------MultiSearchFilter for different filters-------------------\n");
    for (size_t j = 0; j < batches.size(); j++) {
        void *pFilter[batches[j]];
        IDFilter idFilters[indexNum * batches[j]];
        for (int queryIdx = 0; queryIdx < batches[j]; queryIdx++) {
            for (int indexIdx = 0; indexIdx < indexNum; indexIdx++) {
                ConstructCidFilter(idFilters, indexIdx + queryIdx * indexNum, 1, search_cid, search_time);
            }
            pFilter[queryIdx] = &idFilters[indexNum * queryIdx];
        }
        std::vector<float> dist(indexNum * k * batches[j], 0);
        std::vector<faiss::Index::idx_t> label(indexNum * k * batches[j], 0);
        double ts = GetMillisecs();
        for (int l = 0; l < loopTimes; l++) {
            SearchWithFilter(indexes, batches[j], qData.data(),
                k, dist.data(), label.data(), pFilter, false);
        }
        double te = GetMillisecs();
        printf("multi search for different filters: true, index num: %d, k: %d, base: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n",
                indexNum, k, ntotal, dim, batches[j], queryNum, 1000 * batches[j] * loopTimes / (te -ts));
    }

    for (int i = 0; i < indexNum; ++i) {
        delete indexes[i];
    }
}

TEST(TestAscendIndexIVFSP, MultiSearchAndMultiSearchFilter)
{
    int codeBookDim = 256;
    int dim = 256;
    int nonzeroNum = 64;
    int ncentroids = 256; // nlist
    int nprobe = 64;
    int codebookNum = ncentroids;
    size_t nCodeBook = codebookNum * nonzeroNum;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32};
    int searchListSize = 32768;
    int ntotal = 2000000;
    int indexNum = 200;
    int k = 100;
    int queryNum = 5306;

    // 码本，训练数据， 查询数据路径。
    std::string basePath = " ";
    // 特征底库数据
    std::string dataPath = basePath + "base.bin";
    // 查询数据
    std::string queryPath = basePath + "query.bin";
    // （grouptruth） 测试召回的比较数据
    std::string gtsPath = basePath + "gt.bin";
    // codeBook 码本
    std::string codeBookPath = basePath + "codebook.bin";
    // ivfsp 数据，有：加载， 没有就添加并保存
    std::string indexPath = basePath + "myivfsp_base_data.bin";


    // code offset
    std::vector<faiss::Index::idx_t> offset(codebookNum, 0);
    std::iota(offset.begin(), offset.end(), 0);

    // code book
    std::vector<float> codeBookData(nCodeBook * codeBookDim);
    std::ifstream fin(codeBookPath.c_str(), std::ios::binary);
    fin.read((char*)(codeBookData.data()), sizeof(float) * nCodeBook * codeBookDim);
    fin.close();

    // base data
    std::vector<float> data(dim * ntotal);
    std::ifstream codesFin(dataPath.c_str(), std::ios::binary);
    codesFin.read((char*)(data.data()), sizeof(float) * dim * ntotal);
    codesFin.close();

    // ids
    std::vector<faiss::Index::idx_t> ids(ntotal, 0);
    std::iota(ids.begin(), ids.end(), 0);

    // query data
    std::vector<float> qData(queryNum * dim);
    std::ifstream queryFin(queryPath.c_str(), std::ios::binary);
    queryFin.read((char*)(qData.data()), sizeof(float) * queryNum * dim);
    queryFin.close();

    std::vector<int64_t> gts(queryNum * k, 0);
    std::ifstream gtsFin(gtsPath.c_str(), std::ios::binary);
    gtsFin.read((char*)(gts.data()), sizeof(int64_t) * queryNum * k);
    gtsFin.close();

    int64_t resourceSize = 1 * static_cast<int64_t>(1024 * 1024 * 1024);
    faiss::ascend::AscendIndexIVFSPConfig conf({0}, resourceSize);
    conf.nprobe = nprobe; // 64 32 128    16的倍数，且0 < nprobe <= nlist
    conf.handleBatch = nprobe; // 和OM算子保持一致
    conf.searchListSize = searchListSize; // 大于等于512 且为2的幂次。
    conf.filterable = true;

    std::vector<faiss::ascend::AscendIndex*> indexes;
    for (int i = 0; i < indexNum; ++i) {
        auto index = new faiss::ascend::AscendIndexIVFSP(dim, nonzeroNum, ncentroids,
            i == 0 ? codeBookPath.c_str():nullptr,
            faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            faiss::MetricType::METRIC_L2, conf);
        ASSERT_FALSE(index == nullptr);
        index->setVerbose(true);
        indexes.emplace_back(index);
        printf("create index:%d\n", i);
    }

    struct stat indexPathStat;
    if (lstat(indexPath.c_str(), &indexPathStat) == 0) {
        remove(indexPath.c_str());
    }
    for (int i = 0; i < indexNum; ++i) {
        faiss::ascend::AscendIndexIVFSP* index = dynamic_cast<faiss::ascend::AscendIndexIVFSP*>(indexes[i]);
        printf("add data index:%d\n", i);
        if (FILE *file = fopen(indexPath.c_str(), "r")) {
            fclose(file);
            index->loadAllData(indexPath.c_str());
            std::cout << "loadAllData from " << indexPath << std::endl;
            std::cout << "index.ntotal: " << index->ntotal << std::endl;
        } else {
            index->add(ntotal, data.data());
            std::cout << "add" << std::endl;
            std::cout << "index.ntotal: " << index->ntotal << std::endl;
            index->saveAllData(indexPath.c_str());
            std::cout << "saveAllData to " << indexPath << std::endl;
        }
    }

    std::vector<int> nprobeList = {nprobe};
    for (int tmpNprobe : nprobeList) {
        if (tmpNprobe != (nprobe * 1)) {
            printf("-------------set nprobe: %d-------------------\n", tmpNprobe);
            for (int i = 0; i < indexNum; ++i) {
                faiss::ascend::AscendIndexIVFSP* index = dynamic_cast<faiss::ascend::AscendIndexIVFSP*>(indexes[i]);
                index->setNumProbes(tmpNprobe);
            }
        }

        printf("-------------MultiSearch-------------------\n");
        for (size_t j = 0; j < batches.size(); j++) {
            int iloop = queryNum / batches[j];
            std::vector<float> dist(iloop * indexNum * k * batches[j], 0);
            std::vector<faiss::Index::idx_t> label(iloop * indexNum * k * batches[j], 0);
            double ts = GetMillisecs();
            int outLoop = 1;
            for (int oStep = 0; oStep < outLoop; oStep++) {
                for (int iStep = 0; iStep < iloop; iStep++) {
                    Search(indexes, batches[j], qData.data() + iStep * batches[j] * dim, k,
                        dist.data() + iStep * indexNum * k * batches[j],
                        label.data() + iStep * indexNum * k * batches[j], false);
                    if (iStep *batches[j] %512 == 0) {
                        printf("istep:%d\n", iStep);
                    }
                }
            }
            double te = GetMillisecs();
            printf("multi search: true, index num: %d, k: %d, base: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n",
                indexNum, k, ntotal, dim, batches[j], iloop * batches[j], 1000 * iloop * batches[j] * outLoop / (te -ts));
            printMultiRecallNew (indexNum, label, gts.data(), batches[j], iloop, k);
            sleep(30);
        }

        std::vector<int> search_cid(K_MAX_CAMERA_NUM, 0);
        std::iota(search_cid.begin(), search_cid.end(), 0);
        std::vector<uint32_t> search_time = {0, static_cast<uint32_t>(ntotal)};

        printf("-------------MultiSearchFilter for different filters-------------------\n");
        for (size_t j = 0; j < batches.size(); j++) {
            void *pFilters[batches[j]];
            IDFilter idFilters[indexNum * batches[j]];
            for (int queryIdx = 0; queryIdx < batches[j]; queryIdx++) {
                for (int indexIdx = 0; indexIdx < indexNum; indexIdx++) {
                    ConstructCidFilter(idFilters, indexIdx + queryIdx * indexNum, 1, search_cid, search_time);
                }
                pFilters[queryIdx] = &idFilters[indexNum * queryIdx];
            }
            int iloop = queryNum / batches[j];
            std::vector<float> dist(iloop * indexNum * k * batches[j], 0);
            std::vector<faiss::Index::idx_t> label(iloop * indexNum * k * batches[j], 0);
            double ts = GetMillisecs();
            int outLoop = 1;
            for (int oStep = 0; oStep < outLoop; oStep++) {
                for (int iStep = 0; iStep < iloop; iStep++) {
                    SearchWithFilter(indexes, batches[j], qData.data() + iStep * batches[j] * dim, k,
                        dist.data() + iStep * indexNum * k * batches[j],
                        label.data() + iStep * indexNum * k * batches[j], pFilters, false);
                    if (iStep *batches[j] %512 == 0) {
                        printf("istep:%d\n", iStep);
                    }
                }
            }
            double te = GetMillisecs();
            printf("multi search for different filters: true, index num: %d, k: %d, base: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n",
                indexNum, k, ntotal, dim, batches[j], iloop * batches[j], 1000 * iloop * batches[j] * outLoop / (te -ts));
            printMultiRecallNew (indexNum, label, gts.data(), batches[j], iloop, k);
            sleep(30);
        }
    }

    for (int i = 0; i < indexNum; ++i) {
        delete indexes[i];
    }
}

} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}