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

#include <algorithm>
#include <math.h>
#include <iostream>
#include <vector>
#include <random>

#include <sys/time.h>

#include <faiss/ascend/custom/AscendIndexIVFSQT.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/index_io.h>

using namespace std;

// 提前生成算子
// python3 ivfsqt_generate_model.py -d 256 -c 16384 --cores 8 -t 310P
// python3 flat_at_generate_model.py -d 256 -c 16384 --cores 8 -t 310P
// python3 flat_at_int8_generate_model.py -d 256 -c 16384 --cores 8 -t 310P

void Norm(float *data, size_t n, size_t dim)
{
#pragma omp parallel for if(n > 100)
    for (size_t i = 0; i < n; ++i) {
        float l2norm = 0;
        for (size_t j = 0; j < dim; ++j) {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = sqrt(l2norm);

        for (size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = data[i * dim + j] / l2norm;
        }
    }
}


const int FAST_RAND_MAX = 0x7FFF;
unsigned int g_seed = 5678;
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

int main(int argc, char **argv)
{
    std::vector<int> devices {0};
    float threshold = 1.5;

    const size_t dimIn = 256;
    const size_t dimOut = 64;
    const size_t nlist = 16384;
    size_t k = 100;
    const size_t addTotal = 20000000;
    const size_t trainTotal = 2000000;
    const size_t queryNum = 500000;
    const int niter = 16;
    int fuzzyK = 3;

    std::vector<float> data(dimIn * addTotal);
    cout << "generate data" << endl;
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
    }

    Norm(data.data(), addTotal, dimIn);

    faiss::ascend::AscendIndexIVFSQT *index = nullptr;
    try {
        cout << "index start" << endl;
        faiss::ascend::AscendIndexIVFSQTConfig conf({devices});
        conf.cp.niter = niter;
        conf.useKmeansPP = true;
        conf.cp.max_points_per_centroid = 256;
        cout << "index init" << endl;

        index = new faiss::ascend::AscendIndexIVFSQT(dimIn, dimOut, nlist,
            faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);

        index->verbose = true;
        index->setFuzzyK(fuzzyK);
        index->setThreshold(threshold);

        cout << "train start" << endl;
        double start = GetMillisecs();
        index->train(trainTotal, data.data());
        double end = GetMillisecs();
        cout << "train time cost:" << end - start << " ms" << endl;

        cout << "add start" << endl;
        start = GetMillisecs();
        index->add(addTotal, data.data());
        end = GetMillisecs();
        cout << "add time cost:" << end - start << " ms" << endl;

        cout << "update start" << endl;
        start = GetMillisecs();
        index->update();
        end = GetMillisecs();
        cout << "update time cost:" << end - start << " ms" << endl;

        cout << "search start" << endl;
        std::shared_ptr<float> dist(new float[k * queryNum], std::default_delete<float[]>());
        std::shared_ptr<faiss::Index::idx_t> indices(
            new faiss::Index::idx_t[k * queryNum], std::default_delete<faiss::Index::idx_t[]>());

        start = GetMillisecs();
        index->search(queryNum, data.data(), k, dist.get(), indices.get());
        end = GetMillisecs();
        cout << "search finished successfully" << endl;
        cout << "search time cost:" << end - start << " ms" << endl;

        {
            const char *globalFileName = "IVFSQT.faiss";
            cout << "Test index_ascend_to_cpu For IVFSQT, result save to " << globalFileName << endl;
            faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(index);
            faiss::write_index(cpuIndex, globalFileName);
            delete cpuIndex;
            cout << "Test index_ascend_to_cpu For IVFSQT finished" << endl;

            cout << "Test index_cpu_to_ascend For IVFSQT, read from " << globalFileName << endl;
            faiss::Index *initIndex = faiss::read_index(globalFileName);
            faiss::ascend::AscendIndexIVFSQT *realIndex =
                dynamic_cast<faiss::ascend::AscendIndexIVFSQT *>(faiss::ascend::index_cpu_to_ascend(devices, initIndex));
            cout << "Test index_cpu_to_ascend For IVFSQT, finished" << endl;

            realIndex->search(queryNum, data.data(), k, dist.get(), indices.get());

            delete realIndex;
            delete initIndex;
        }
        delete index;
    } catch (faiss::FaissException & e) {
        cout << "Exception caught!" << e.what() << endl;
        if (index == nullptr) {
            delete index;
        }
        return -1;
    }

    return 0;
}
