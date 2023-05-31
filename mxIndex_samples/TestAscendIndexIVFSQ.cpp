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

 // python3 ivfsq8_generate_model.py -d 64 -c 8192
 // python3 flat_at_generate_model.py -d 64 -c 8192
 // python3 aicpu_generate_model.py

#include <faiss/ascend/AscendIndexIVFSQ.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

void Norm(float *data, size_t n, size_t dim)
{
#pragma omp parallel for if (n>1)
    for (size_t i = 0; i < n; ++i) {
        float l2norm = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = std::sqrt(l2norm);

        for (size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = data[i * dim + j] / l2norm;
        }
    }
}

int main()
{
    size_t dim = 64;
    size_t ntotal = 1000000;
    int ncentroids = 8192;
    int nprobe = 64;

    printf("generate data\n");
    std::vector<float> data(dim * ntotal);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, dim);

    faiss::ascend::AscendIndexIVFSQ *index = nullptr;
    try {
        faiss::ascend::AscendIndexIVFSQConfig conf{{0}};
        conf.useKmeansPP = true;
        printf("create index\n");
        index = new faiss::ascend::AscendIndexIVFSQ(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
                                                    faiss::METRIC_INNER_PRODUCT, false, conf);
        index->verbose = true;
        index->setNumProbes(nprobe);

        printf("start train\n");
        index->train(ntotal, data.data());
        printf("start add\n");
        index->add(ntotal, data.data());

        size_t n = 10;
        size_t k = 10;
        std::vector<float> dist(n * k, 0.0);
        std::vector<faiss::Index::idx_t> label(n * k, 0);
        printf("start search\n");
        index->search(n, data.data(), k, dist.data(), label.data());
    } catch (std::exception &e) {
        printf("exceptin caught: %s\n", e.what());
        delete index;
        return -1;
    }
    delete index;
    printf("search success\n");
    return 0;
}
