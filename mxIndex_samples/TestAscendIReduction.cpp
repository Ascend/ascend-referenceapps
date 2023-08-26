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
#include <faiss/index_io.h>
#include <random>
#include <fstream>
#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/custom/IReduction.h>
#include <faiss/ascend/AscendNNInference.h>

namespace {
    int DIMIN = 256;
    int DIMOUT = 64;
    std::string nnom;
    std::string MetricTypeName = "INNER_PRODUCT";
    faiss::MetricType MetricType = MetricTypeName == "L2" ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;

    void ReadModel()
    {
        std::string modelPath = "./"; //导入对应的NN降维模型
        std::ifstream istrm(modelPath.c_str(), std::ios::binary);
        std::stringstream buffer;
        buffer << istrm.rdbuf();
        nnom = buffer.str();
        istrm.close();
    }

    void TestSampleNNInfer()
    {
        std::vector<int> deviceList = { 0 };
        int ntotal = 100000;
        int maxSize = ntotal * DIMIN;
        std::vector<float> data(maxSize);
        std::vector<float> outputData(ntotal * DIMOUT);
        for (int i = 0; i < maxSize; i++) {
            data[i] = drand48();
        }

        std::cout << "TestSampleNNInfer start " << std::endl;
        ReadModel();

        faiss::ascend::AscendNNInference dimInfer(deviceList, nnom.data(), nnom.size());
        dimInfer.infer(ntotal, (char *)data.data(), (char *)outputData.data());

        std::cout << "TestSampleNNInfer end " << std::endl;
    }

    void TestSampleNNReduce()
    {
        std::vector<int> deviceList = { 0 };
        int ntotal = 100000;
        int maxSize = ntotal * DIMIN;
        std::vector<float> data(maxSize);
        std::vector<float> outputData(ntotal * DIMOUT);
        for (int i = 0; i < maxSize; i++) {
            data[i] = drand48();
        }

        std::cout << "TestSampleNNReduce start " << std::endl;
        ReadModel();

        faiss::ascend::ReductionConfig reductionConfig(deviceList, nnom.data(), nnom.size());
        std::string method = "NN";
        faiss::ascend::IReduction* reduction = CreateReduction(method, reductionConfig);
        reduction->train(ntotal, data.data());
        reduction->reduce(ntotal, data.data(), outputData.data());

        std::cout << "TestSampleNNReduce end " << std::endl;
    }

    void TestSamplePcarReduce()
    {
        std::vector<int> deviceList = { 0 };
        int ntotal = 100000;
        int maxSize = ntotal * DIMIN;
        std::vector<float> data(maxSize);
        std::vector<float> outputData(ntotal * DIMOUT);
        for (int i = 0; i < maxSize; i++) {
            data[i] = drand48();
        }

        std::cout << "TestSamplePcarReduce start " << std::endl;
        // Pcar IReduction
        faiss::ascend::ReductionConfig reductionConfig(DIMIN, DIMOUT, 0, false);
        std::string method = "PCAR";
        faiss::ascend::IReduction* reduction = CreateReduction(method, reductionConfig);
        reduction->train(ntotal, data.data());
        reduction->reduce(ntotal, data.data(), outputData.data());

        std::cout << "TestSamplePcarReduce end " << std::endl;
    }
}

int main(int argc, char **argv)
{
    TestSampleNNInfer();
    TestSampleNNReduce();
    TestSamplePcarReduce();
    return 0;
}
