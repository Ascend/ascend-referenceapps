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

#include <map>
#include <set>
#include <string>
#include <fstream>
#include <random>
#include <cstdio>
#include <sys/time.h>
#include <sys/stat.h>
#include <faiss/ascend/AscendIndexIVFSP.h>
#include <faiss/ascend/AscendMultiIndexSearch.h>

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
        memset(cameraIdMask, static_cast<uint8_t>(0), K_MAX_CAMERA_NUM / MASK_LEN);
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
void printMultiRecall(int indexNum, std::vector<T> &labels, int64_t* gt, int bs, int batchNum, int k)
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

void RecallAndRecallFilter()
{
    // 数据集（特征数据、查询数据、groundtruth数据）、码本， 所在的目录。请根据实际情况填写。
    std::string basePath = " ";
    // 特征底库数据
    std::string dataPath = basePath + "base.bin";
    // 查询数据
    std::string queryPath = basePath + "query.bin";
    // （groundtruth） 测试召回的比较数据
    std::string gtsPath = basePath + "gt.bin";

    // codeBook 码本
    std::string codeBookPath = basePath + "codebook.bin";

    // 参数值 dim、nonzeroNum、nlist、searchListSize，应该和使用的codeBook 码本保持一致，即和训练码本时指定的参数保持一致。
    int dim = 256;
    int nonzeroNum = 64;
    int nlist = 256;
    int handleBatch = 64;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32, 64};
    int searchListSize = 32768;

    // nototal应该小于等于使用的"特征底库数据"集的特征向量的实际条数
    int ntotal = 2000000;
    int k = 100;
    // queryNum应该小于等于使用的"查询数据"集的特征向量的实际条数
    int queryNum = 5306;

    // base data
    std::vector<float> data(dim * ntotal);
    std::ifstream codesFin(dataPath.c_str(), std::ios::binary);
    codesFin.read((char*)(data.data()), sizeof(float) * dim * ntotal);
    codesFin.close();

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
    conf.handleBatch = handleBatch; // 和OM算子保持一致
    conf.nprobe = handleBatch; // 64 32 128    16的倍数，且0 < nprobe <= nlist
    conf.searchListSize = searchListSize; // 大于等于512 且为2的幂次。
    conf.filterable = true;

    faiss::ascend::AscendIndexIVFSP index(dim, nonzeroNum, nlist,
            codeBookPath.c_str(),
            faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            faiss::MetricType::METRIC_L2, conf);
    index.setVerbose(true);
    index.add(ntotal, data.data());
    std::cout << "index.ntotal: " << index.ntotal << std::endl;

    std::vector<int> nprobeList = {handleBatch, handleBatch * 2, handleBatch / 2};
    for (int tmpNprobe : nprobeList) {
        printf("-------------set nprobe: %d-------------------\n", tmpNprobe);
        index.setNumProbes(tmpNprobe);

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

void MultiSearchAndMultiSearchFilter()
{
    // 数据集（特征数据、查询数据、groundtruth数据）、码本， 所在的目录。请根据实际情况填写。
    std::string basePath = " ";
    // 特征底库数据
    std::string dataPath = basePath + "base.bin";
    // 查询数据
    std::string queryPath = basePath + "query.bin";
    // （groundtruth） 测试召回的比较数据
    std::string gtsPath = basePath + "gt.bin";

    // codeBook 码本
    std::string codeBookPath = basePath + "codebook.bin";
    // ivfsp索引数据落盘保存到的路径indexPath
    std::string indexPath = basePath + "myivfsp_base_data.bin";

    // 参数值 dim、nlist、nonzeroNum、searchListSize，应该和使用的codeBook 码本保持一致，即和训练码本时指定的参数保持一致。
    int dim = 256;
    int nonzeroNum = 64;
    int nlist = 256;
    int handleBatch = 64;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32, 64};
    int searchListSize = 32768;

    // nototal应该小于等于使用的"特征底库数据"集的特征向量的实际条数
    int ntotal = 2000000;
    int indexNum = 10;
    int k = 100;
    // queryNum应该小于等于使用的"查询数据"集的特征向量的实际条数
    int queryNum = 5306;

    // base data
    std::vector<float> data(dim * ntotal);
    std::ifstream codesFin(dataPath.c_str(), std::ios::binary);
    codesFin.read((char*)(data.data()), sizeof(float) * dim * ntotal);
    codesFin.close();

    // query data
    std::vector<float> qData(queryNum * dim);
    std::ifstream queryFin(queryPath.c_str(), std::ios::binary);
    queryFin.read((char*)(qData.data()), sizeof(float) * queryNum * dim);
    queryFin.close();

    // ground truth data
    std::vector<int64_t> gts(queryNum * k, 0);
    std::ifstream gtsFin(gtsPath.c_str(), std::ios::binary);
    gtsFin.read((char*)(gts.data()), sizeof(int64_t) * queryNum * k);
    gtsFin.close();

    int64_t resourceSize = 2 * static_cast<int64_t>(1024 * 1024 * 1024);
    faiss::ascend::AscendIndexIVFSPConfig conf({0}, resourceSize);
    conf.handleBatch = handleBatch; // 和OM算子保持一致
    conf.nprobe = handleBatch; // 64 32 128    16的倍数，且0 < nprobe <= nlist
    conf.searchListSize = searchListSize; // 大于等于512 且为2的幂次。
    conf.filterable = true;

    std::vector<faiss::ascend::AscendIndex*> indexes;
    for (int i = 0; i < indexNum; ++i) {
        auto index = new faiss::ascend::AscendIndexIVFSP(dim, nonzeroNum, nlist,
            i == 0 ? codeBookPath.c_str():nullptr,
            faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            faiss::MetricType::METRIC_L2, conf);
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

    std::vector<int> nprobeList = {handleBatch, handleBatch * 2, handleBatch / 2};
    for (int tmpNprobe : nprobeList) {
        printf("-------------set nprobe: %d-------------------\n", tmpNprobe);
        for (int i = 0; i < indexNum; ++i) {
            faiss::ascend::AscendIndexIVFSP* index = dynamic_cast<faiss::ascend::AscendIndexIVFSP*>(indexes[i]);
            index->setNumProbes(tmpNprobe);
        }

        printf("-------------MultiSearch-------------------\n");
        for (size_t j = 0; j < batches.size(); j++) {
            int iloop = queryNum / batches[j];
            std::vector<float> dist(iloop * indexNum * k * batches[j], 0);
            std::vector<faiss::Index::idx_t> label(iloop * indexNum * k * batches[j], 0);
            double ts = GetMillisecs();
            for (int iStep = 0; iStep < iloop; iStep++) {
                Search(indexes, batches[j], qData.data() + iStep * batches[j] * dim, k,
                    dist.data() + iStep * indexNum * k * batches[j],
                    label.data() + iStep * indexNum * k * batches[j], false);
                if (iStep *batches[j] %512 == 0) {
                    printf("istep:%d\n", iStep);
                }
            }
            double te = GetMillisecs();
            printf("multi search: true, index num: %d, k: %d, base: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n",
                indexNum, k, ntotal, dim, batches[j], iloop * batches[j], 1000 * iloop * batches[j] / (te -ts));
            printMultiRecall(indexNum, label, gts.data(), batches[j], iloop, k);
        }

        std::vector<int> search_cid(K_MAX_CAMERA_NUM, 0);
        std::iota(search_cid.begin(), search_cid.end(), 0);
        std::vector<uint32_t> search_time = {0, static_cast<uint32_t>(ntotal)};

        printf("-------------MultiSearchFilter for same filters-------------------\n");
        for (size_t j = 0; j < batches.size(); j++) {
            IDFilter idFilters[batches[j]];
            void *pFilters = &idFilters[0];
            ConstructCidFilter(idFilters, batches[j], search_cid, search_time);

            int iloop = queryNum / batches[j];
            std::vector<float> dist(iloop * indexNum * k * batches[j], 0);
            std::vector<faiss::Index::idx_t> label(iloop * indexNum * k * batches[j], 0);
            double ts = GetMillisecs();
            for (int iStep = 0; iStep < iloop; iStep++) {
                SearchWithFilter(indexes, batches[j], qData.data() + iStep * batches[j] * dim, k,
                    dist.data() + iStep * indexNum * k * batches[j],
                    label.data() + iStep * indexNum * k * batches[j], pFilters, false);
                if (iStep *batches[j] % 512 == 0) {
                    printf("istep:%d\n", iStep);
                }
            }
            double te = GetMillisecs();
            printf("multi search for same filters: true, index num: %d, k: %d, base: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n",
                indexNum, k, ntotal, dim, batches[j], iloop * batches[j], 1000 * iloop * batches[j] / (te -ts));
            printMultiRecall(indexNum, label, gts.data(), batches[j], iloop, k);
        }

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
            for (int iStep = 0; iStep < iloop; iStep++) {
                SearchWithFilter(indexes, batches[j], qData.data() + iStep * batches[j] * dim, k,
                    dist.data() + iStep * indexNum * k * batches[j],
                    label.data() + iStep * indexNum * k * batches[j], pFilters, false);
                if (iStep *batches[j] % 512 == 0) {
                    printf("istep:%d\n", iStep);
                }
            }
            double te = GetMillisecs();
            printf("multi search for different filters: true, index num: %d, k: %d, base: %d, dim: %d, batch size: %d, search num: %2d, QPS: %9.4f\n",
                indexNum, k, ntotal, dim, batches[j], iloop * batches[j], 1000 * iloop * batches[j] / (te -ts));
            printMultiRecall(indexNum, label, gts.data(), batches[j], iloop, k);
        }
    }

    for (int i = 0; i < indexNum; ++i) {
        delete indexes[i];
    }
}

} // namespace

int main(int argc, char **argv)
{
    RecallAndRecallFilter();
    MultiSearchAndMultiSearchFilter();
    return 0;
}