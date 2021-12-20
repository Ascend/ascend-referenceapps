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

#include <ascenddaemon/impl/IndexPreTransform.h>
#include <ascenddaemon/impl/IndexIVFSQL2.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <random>

namespace {
TEST(TestIndexPreTransform, Search)
{
    // create Index
    const float min = 1.0f;
    const float diff = 5.0f;
    const int numList = 16384;
    const int dim = 128;
    const int nprobe = 64;

    ascend::IndexIVFSQL2 indexIVFSQ(numList, dim, true, nprobe);
    ascend::IndexPreTransform indexPreTransform(&indexIVFSQ);

    std::default_random_engine e(time(nullptr));
    const int randIntMax = 0x7fff;  // 0x7fff is RAND_MAX
    std::uniform_int_distribution<> distrib(1, randIntMax);

    std::vector<float16_t> coarseCentsVec(numList * dim);
#pragma omp parallel for
    for (size_t i = 0; i < coarseCentsVec.size(); i++) {
        coarseCentsVec[i] = (float16_t)fmod(((float)distrib(e)) + min, diff + min);
    }

    std::vector<float16_t> vmin(dim);
    std::vector<float16_t> vdiff(dim);
    for (int i = 0; i < dim; i++) {
        vmin[i] = (float16_t)fmod((float)distrib(e), min);
        vdiff[i] = (float16_t)fmod((float)distrib(e), diff);
    }

    ascend::AscendTensor<float16_t, 2> coarseCents(coarseCentsVec.data(), { numList, dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedMin(vmin.data(), { dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedDiff(vdiff.data(), { dim });
    indexIVFSQ.updateCoarseCentroidsData(coarseCents);
    indexIVFSQ.updateTrainedValue(trainedMin, trainedDiff);
    std::cout << "indexivfsq update trained value and coarse centroids success" << std::endl;

    // addVectors
    int listLength = 1700; // num of vectors every list
    int idx = 0;
    std::vector<unsigned char> codes(listLength * dim);
    std::vector<float> precomp(listLength, 0);
#pragma omp parallel for
    for (size_t j = 0; j < codes.size(); j++) {
        codes[j] = static_cast<unsigned char>(255 * (distrib(e) / (randIntMax + 0.0f)));
    }

#pragma omp parallel for
    for (size_t i = 0; i < precomp.size(); i++) {
        int offset = i * dim;
        for (int j = 0; j < dim; j++) {
            float val = (codes[offset + j] + 0.5f) * (float)vdiff[j] / 255.0f + (float)vmin[j];
            precomp[i] += val * val;
        }
    }

    for (int i = 0; i < numList; i++) {
        std::vector<uint32_t> indices(listLength);
        for (size_t j = 0; j < indices.size(); j++) {
            indices[j] = idx;
            idx++;
        }

        indexIVFSQ.addVectors(i, listLength, codes.data(), indices.data(), precomp.data());
    }
    std::cout << "add vectors success" << std::endl;

    // create Index
    const int dimIn = 512;
    const int dimOut = dim;

    ascend::LinearTransform ltrans(dimIn, dimOut);
    std::vector<float16_t> matrix(dimIn * dimOut);
#pragma omp parallel for
    for (size_t i = 0; i < matrix.size(); i++) {
        // eye matrix
        if (i % (dimIn + 1) == 0) {
            matrix[i] = (float16_t)1.0;
        } else {
            matrix[i] = (float16_t)0;
        }
    }
    std::vector<float> bias(dimOut);
#pragma omp parallel for
    for (size_t i = 0; i < bias.size(); i++) {
        bias[i] = (float)0;
    }
    ascend::AscendTensor<float16_t, ascend::DIMS_2> mm(matrix.data(), { dimOut, dimIn });
    ascend::AscendTensor<float, ascend::DIMS_1> bb(bias.data(), { dimOut });
    ltrans.updateTrainedValue(mm, bb);
    std::cout << "lineartransform update trained value success" << std::endl;

    indexPreTransform.prependTransform(&ltrans);
    std::cout << "indexPretransform prepend success" << std::endl;

    // search
    const std::vector<int> numQueries { 1, 31, 101, 201, 401 };
    const int topK = 1000;
    for (auto qnum : numQueries) {
        std::vector<float16_t> queries(qnum * dimIn);
        std::vector<uint32_t> queryIds(qnum);
        for (int i = 0; i < qnum; i++) {
            int listId = distrib(e) % (numList - 1);
            int subId = distrib(e) % (indexIVFSQ.getListLength(listId) - 1);
            queryIds[i] = *(indexIVFSQ.getListIndices(listId).data() + subId);
            auto err = memcpy_s(queries.data() + i * dimIn, dim * sizeof(float16_t),
                coarseCentsVec.data() + listId * dim, dim * sizeof(float16_t));
            ASSERT_EQ(err, EOK);

            std::vector<unsigned char> reshapedData;
            indexIVFSQ.getListVectorsReshaped(listId, reshapedData);

            for (int j = 0; j < dim; j++) {
                float16_t val = (float16_t)((reshapedData[subId * dim + j] + 0.5f) * vdiff[j] / 255.0f + vmin[j]);
                queries[i * dimIn + j] += val;
            }
        }

        std::vector<float16_t> resDists(qnum * topK);
        std::vector<ascend::Index::idx_t> resIndices(qnum * topK);
        indexPreTransform.search(qnum, queries.data(), topK, resDists.data(), resIndices.data());

        std::cout << "----- start search, num_query is " << qnum << "-----" << std::endl;
        double cost = 0.0;
        double min = std::numeric_limits<double>::max();
        double max = 0.0;

        const int loopSearch = 100;
        for (int i = 0; i < loopSearch; i++) {
            double start = ascend::utils::getMillisecs();
            indexPreTransform.search(qnum, queries.data(), topK, resDists.data(), resIndices.data());
            double end = ascend::utils::getMillisecs();
            double cur_cost = end - start;
            min = std::min(min, cur_cost);
            max = std::max(max, cur_cost);
            cost += cur_cost;
        }

        for (int i = 0; i < qnum; i++) {
            std::cout << "query: " << queryIds[i] << ", results_id: " << resIndices[i * topK] << ", distance : " <<
                resDists[i * topK] << std::endl;

            EXPECT_EQ(queryIds[i], resIndices[i * topK]);
        }
        std::cout << "search cost time: " << cost / (loopSearch * qnum) << "ms in average, "
                  << "max:" << max / qnum << "ms, min:" << min / qnum << "ms." << std::endl;
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}