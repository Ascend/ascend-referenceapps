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

#include <vector>
#include <numeric>
#include <faiss/index_io.h>
#include <random>
#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <faiss/ascend/AscendMultiIndexSearch.h>
#include <faiss/index_io.h>

namespace {

const auto METRIC_TYPE = faiss::METRIC_INNER_PRODUCT;

inline void Norm(float *data, int n, int dim)
{
#pragma omp parallelf for if(n > 100)
    for (int i = 0; i < n; ++i){
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

struct IDFilter {
    IDFilter()
    {
        memset(camera_id_mask, static_cast<uint8_t>(0xFF), 128 / 8);
        time_range[0] = 0;
        time_range[1] = -1;
    }

    uint8_t camera_id_mask[16] = { 0xFF };  //  cid
    uint32_t time_range[2] = { 0 };        // 时间戳
};

void AscendIndexSQMultiSearchFilter()
{
    int dim = 64;
    int searchNum = 2;
    int indexNum = 10;
    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    conf.filterable = true;

    int ntotal = 128;
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, dim);

    std::vector<int64_t> ids(ntotal, 0);
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e1(seed);
    std::uniform_int_distribution<int32_t> id(0, std::numeric_limits<int32_t>::max());
    std::uniform_int_distribution<uint8_t> searchCid(0, 127);

    for (int i = 0; i < ntotal; i++) {
        ids[i] = (static_cast<int64_t>(searchCid(e1)) << 42) + (static_cast<int64_t>(id(e1)) << 10);
    }

    std::cout << "AscendIndex SQ MultiSearch with different filter start" << std::endl;
    std::vector<faiss::ascend::AscendIndex *> indexes(indexNum);
    for (int i = 0; i < indexNum; i++) {
        indexes[i] = new faiss::ascend::AscendIndexSQ(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

        indexes[i]->train(ntotal, data.data());
        indexes[i]->add_with_ids(ntotal, data.data(), ids.data());
    }

    void *multiFilters[searchNum];
    IDFilter idFilters[indexNum * searchNum];
    for (int query = 0; query < searchNum; query++) {
        for (int indexId = 0; indexId < indexNum; indexId++) {
            IDFilter idFilter;
            idFilter.time_range[0] = 0;
            idFilter.time_range[1] = 0x7fffffff;
            // 16个uint8_t表示128位的cid
            for (int i = 0; i < 16; i++) {
                idFilter.camera_id_mask[i] = searchCid(e1);
            }
            idFilters[query * indexNum + indexId] = idFilter;
        }
        multiFilters[query] = &idFilters[query * indexNum];
    }
    int k = 10;
    std::vector<float> dist(indexNum * k * searchNum, 0);
    std::vector<faiss::Index::idx_t> label(indexNum * k * searchNum, 0);
    SearchWithFilter(indexes, searchNum, data.data(), k, dist.data(), label.data(), multiFilters, false);

    for (int i = 0; i < indexNum; i++) {
        delete indexes[i];
    }
    std::cout << "AscendIndex SQ MultiSearch with different filter end" << std::endl;
}

void IndexSQMultiSearchFilter()
{
    int dim = 64;
    int searchNum = 2;
    int indexNum = 10;
    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    conf.filterable = true;

    int ntotal = 128;
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, dim);

    std::vector<int64_t> ids(ntotal, 0);
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e1(seed);
    std::uniform_int_distribution<int32_t> id(0, std::numeric_limits<int32_t>::max());
    std::uniform_int_distribution<uint8_t> searchCid(0, 127);

    for (int i = 0; i < ntotal; i++) {
        ids[i] = (static_cast<int64_t>(searchCid(e1)) << 42) + (static_cast<int64_t>(id(e1)) << 10);
    }

    std::cout << "Index SQ MultiSearch with different filter start" << std::endl;
    std::vector<faiss::Index *> indexes(indexNum);
    for (int i = 0; i < indexNum; i++) {
        indexes[i] = new faiss::ascend::AscendIndexSQ(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

        indexes[i]->train(ntotal, data.data());
        indexes[i]->add_with_ids(ntotal, data.data(), ids.data());
    }

    void *multiFilters[searchNum];
    IDFilter idFilters[indexNum * searchNum];
    for (int query = 0; query < searchNum; query++) {
        for (int indexId = 0; indexId < indexNum; indexId++) {
            IDFilter idFilter;
            idFilter.time_range[0] = 0;
            idFilter.time_range[1] = 0x7fffffff;
            // 16个uint8_t表示128位的cid
            for (int i = 0; i < 16; i++) {
                idFilter.camera_id_mask[i] = searchCid(e1);
            }
            idFilters[query * indexNum + indexId] = idFilter;
        }
        multiFilters[query] = &idFilters[query * indexNum];
    }
    int k = 10;
    std::vector<float> dist(indexNum * k * searchNum, 0);
    std::vector<faiss::Index::idx_t> label(indexNum * k * searchNum, 0);
    faiss::ascend::SearchWithFilter(indexes, searchNum, data.data(), k, dist.data(), label.data(), multiFilters, false);

    for (int i = 0; i < indexNum; i++) {
        delete indexes[i];
    }
    std::cout << "Index SQ MultiSearch with different filter end" << std::endl;
}

void AscendIndexSQMultiSearch()
{
    size_t ntotal = 1000000;
    int k = 5;
    int dim = 64;
    int indexNum = 10;
    std::vector<int> searchNum = { 1, 2, 4 , 8 };
    size_t maxSize = ntotal * dim;
    std::vector<float> data(maxSize);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, dim);

    std::cout << "AscendIndex SQ MultiSearch start" << std::endl;
    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    std::vector<faiss::ascend::AscendIndex *> indexes(indexNum);
    for (int i = 0; i < indexNum; i++) {
        indexes[i] = new faiss::ascend::AscendIndexSQ(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

        indexes[i]->train(ntotal, data.data());
        indexes[i]->add(ntotal, data.data());
    }

    for (size_t i = 0; i < searchNum.size(); i++) {
        std::vector<float> dist(indexNum * searchNum[i] * k, 0);
        std::vector<faiss::Index::idx_t> label(indexNum * searchNum[i] * k, 0);

        Search(indexes, searchNum[i], data.data(), k, dist.data(), label.data(), false);
    }
    for (int i = 0; i < indexNum; i++) {
        delete indexes[i];
    }
    std::cout << "AscendIndex SQ MultiSearch end" << std::endl;
}

void AscendIndexInt8MultiSearch()
{
    size_t ntotal = 100000;
    int k = 5;
    int dim = 64;
    int indexNum = 10;
    std::vector<int> searchNum = { 1, 2, 4, 8 };
    
    std::vector<std::vector<int8_t>> data(indexNum, std::vector<int8_t>(dim * ntotal));
    for (int i = 0; i < indexNum; i++) {
        GenerateCodes(data[i].data(), ntotal, dim);
    }

    std::cout << "AscendIndex Int8 MultiSearch start" << std::endl;
    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 });
    std::vector<faiss::ascend::AscendIndexInt8 *> indexes(indexNum);
    
    for (int i = 0; i < indexNum; i++) {
        indexes[i] = new faiss::ascend::AscendIndexInt8Flat(dim, METRIC_TYPE, conf);

        indexes[i]->add(ntotal, data[i].data());
    }

    for (size_t i = 0; i < searchNum.size(); i++) {
        std::vector<float> dist(indexNum * searchNum[i] * k, 0);
        std::vector<faiss::Index::idx_t> label(indexNum * searchNum[i] * k, 0);
        std::vector<int8_t> query(dim * searchNum[i]);
        GenerateCodes(query.data(), searchNum[i], dim);

        Search(indexes, searchNum[i], query.data(), k, dist.data(), label.data(), false);
    }
    for (int i = 0; i < indexNum; i++) {
        delete indexes[i];
    }
    std::cout << "AscendIndex Int8 MultiSearch end" << std::endl;
}
}

int main(int argc, char **argv) 
{
    AscendIndexSQMultiSearchFilter();    // SQ算法多index批量检索带属性过滤 （AscendIndex）
    IndexSQMultiSearchFilter();          // SQ算法多index批量检索带属性过滤（Index）
    AscendIndexSQMultiSearch();          // SQ算法多index批量检索
    AscendIndexInt8MultiSearch();        // Int8算法多index批量检索
    return 0;
}