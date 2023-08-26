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

 #include <faiss/ascend/AscendIndexCluster.h>
 #include <random>

void Genarate(std::vector<float> &addnVec, std::vector<uint32_t> &ids, int ntotal, int dim)
{
    std::vector<float> normBase(ntotal);
    std::default_random_engine e(time(nullptr));
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
 
    for (uint32_t i = 0; i < ntotal * static_cast<uint32_t>(dim); i++) {
        addnVec[i] = static_cast<int8_t>(255 * rCode(e) - 128);
        size_t baseIdx = i / dim;
        normBase[baseIdx] += addnVec[i] * addnVec[i];
        if ((i + 1) % dim == 0) {
            normBase[baseIdx] = sqrt(normBase[baseIdx]);
        }
    }

     for (uint32_t i = 0; i < ntotal * static_cast<uint32_t>(dim); i++) {
        addnVec[i] /= normBase[i / dim];
     }

     std::iota(ids.begin(), ids.end(), 0);
}

int main (int argc, char **argv)
{
    int dim = 64;
    int ntotal = 100000;
    int capacity = 1200000;
    int64_t resourceSize = static_cast<int64_t>(2) * static_cast<int64_t>(1024 * 1024 * 1024);
    auto meticType = faiss::MetricType::METRIC_INNER_PRODUCT;
    faiss::ascend::AscendIndexCluster index;
    std::vector<int> deciveList = {0};
    auto ret = index.Init(dim, capacity, meticType, deciveList, resourceSize);
    if (ret != 0) {
        printf("[ERROR] Init fail ret = %d \r\n", ret);
        return 1;
    }

    std::vector<float> addVec(static_cast<int64_t>(ntotal) * static_cast<int64_t>(dim));
    std::vector<uint32_t> ids(ntotal);
    Genarate(addVec, ids, ntotal, dim);

    ret = index.AddFeatures(ntotal, addVec.data(), ids.data());
    if (ret != 0) {
        printf("[ERROR] AddFeatures fail ret = %d \r\n", ret);
        return 1;
    }

    uint32_t nq = 128;
    uint32_t start = 0;
    uint32_t codeStartIdx = 0;
    uint32_t codeNum =1000;
    float threshold = 0.75;
    std::vector<uint32_t> queryIdArr(nq);
    std::iota(queryIdArr.begin(), queryIdArr.end(), start);

    bool aboveFilter = true;
    std::vector<std::vector<float>> resDist(nq);
    std::vector<std::vector<uint32_t>> resIdx(nq);

    ret = index.ComputeDistanceByThreshold(queryIdArr, codeStartIdx, codeNum, threshold, aboveFilter, resDist, resIdx);
    if (ret != 0) {
        printf("[ERROR] ComputeDistanceByThreshold fail ret = %d \r\n", ret);
        return 1;
    }

    for(uint32_t i = 0; i < nq; i++) {
        uint32_t len = resDist[i].size();
        printf("queryFeature(%d/%d), %u feature dist greater than the threshold:\r\n", i, nq, len);
        for (uint32_t j = 0; j < len; j++ ){
            printf("   id: %u, dist: %.4lf\r\n", resIdx[i][j], resDist[i][j]);
        }
    }

    index.Finalize();
}
