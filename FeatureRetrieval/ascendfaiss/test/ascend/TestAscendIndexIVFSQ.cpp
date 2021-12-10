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

#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexIVFSQ.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendAutoTune.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>

namespace {
void Norm(float* data, int n, int dim)
{
#pragma omp parallel for if(n > 1)
    for (int i = 0; i < n; ++i) {
        float l2norm = 0;
        for (int j = 0; j < dim; ++j) {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = sqrt(l2norm);

        for (int j = 0; j < dim; ++j) {
            data[i * dim + j] = data[i * dim + j] / l2norm;
        }
    }
}

TEST(TestAscendIndexIVFSQ, All)
{
    int dim = 64;
    int ntotal = 200000;
    int ncentroids = 1024;
    int nprobe = 64;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexIVFSQConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQ index(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::METRIC_L2, true, conf);
    index.verbose = true;

    faiss::ascend::AscendParameterSpace para;
    para.set_index_parameter(&index, "nprobe", (double)(nprobe / 2));
    EXPECT_EQ(index.getNumProbes(), nprobe / 2);

    index.setNumProbes(nprobe);
    EXPECT_EQ(index.getNumProbes(), nprobe);

    index.train(ntotal, data.data());
    for (int i = 0; i < ncentroids; i++) {
        int len = index.getListLength(i);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, data.data());
    ASSERT_EQ(index.ntotal, ntotal);
    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            tmpTotal += index.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, ntotal);
    }

    std::vector<std::vector<uint8_t>> codes(ncentroids, std::vector<uint8_t>());
    std::vector<std::vector<uint32_t>> indices(ncentroids, std::vector<uint32_t>());
    for (int i = 0; i < ncentroids; i++) {
        index.getListCodesAndIds(i, codes[i], indices[i]);
        ASSERT_EQ(codes[i].size(), indices[i].size() * index.d);
    }

    index.reset();
    for (int i = 0; i < ncentroids; i++) {
        int len = index.getListLength(i);
        ASSERT_EQ(len, 0);
    }

    index.reserveMemory(ntotal);
    index.add(ntotal, data.data());
    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            tmpTotal += index.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, ntotal);
    }

    for (int i = 0; i < ncentroids; i++) {
        std::vector<uint8_t> tmpCode;
        std::vector<uint32_t> tmpIndi;
        index.getListCodesAndIds(i, tmpCode, tmpIndi);
        ASSERT_EQ(tmpCode.size(), tmpIndi.size() * index.d) << "failure index:" << i;
        ASSERT_EQ(tmpCode.size(), codes[i].size()) << "failure index:" << i;
        ASSERT_EQ(tmpIndi.size(), indices[i].size()) << "failure index:" << i;
        ASSERT_EQ(memcmp(tmpCode.data(), codes[i].data(), indices[i].size()), 0) << "failure index:" << i;
        ASSERT_EQ(memcmp(tmpIndi.data(), indices[i].data(), indices[i].size() * sizeof(uint32_t)), 0) <<
            "failure index:" << i;
    }

    {
        int n = 1;
        int k = 1000;
        for (int i = 3; i < 10; i++) {
            std::vector<float> dist(k, 0);
            std::vector<faiss::Index::idx_t> label(k, 0);
            index.search(n, data.data() + i * dim, k, dist.data(), label.data());
            ASSERT_EQ(label[0], i);

            faiss::Index::idx_t assign;
            index.assign(1, data.data() + i * dim, &assign);
            ASSERT_EQ(assign, i);
        }
    }
}

TEST(TestAscendIndexIVFSQ, CloneAscend2CPU)
{
    int dim = 64;
    int ntotal = 2500000;
    int ncentroids = 1024;
    int nprobe = 64;

    srand48(0);
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    // ascend index
    faiss::ascend::AscendIndexIVFSQConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQ ascendIndex(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::METRIC_L2, true, conf);
    ascendIndex.verbose = true;
    ascendIndex.setNumProbes(nprobe);
    EXPECT_EQ(ascendIndex.getNumProbes(), nprobe);

    // real training data is divided by 10
    ascendIndex.train(ntotal / 10, data.data());

    // add data
    ascendIndex.reserveMemory(ntotal);
    ascendIndex.add(ntotal, data.data());

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&ascendIndex);
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./ivfsqtest.faiss";
    write_index(cpuIndex, outfilename);
    printf("write ascendIndex to file ok!\n");

    faiss::IndexIVFScalarQuantizer *cpuIvfSq = dynamic_cast<faiss::IndexIVFScalarQuantizer *>(cpuIndex);
    EXPECT_EQ(ascendIndex.metric_type, cpuIvfSq->metric_type);
    EXPECT_EQ(ascendIndex.d, cpuIvfSq->d);
    EXPECT_EQ(ascendIndex.ntotal, cpuIvfSq->ntotal);
    EXPECT_EQ(ascendIndex.is_trained, cpuIvfSq->is_trained);
    EXPECT_EQ(ascendIndex.getNumLists(), cpuIvfSq->nlist);
    EXPECT_EQ(ascendIndex.getNumProbes(), cpuIvfSq->nprobe);

    const faiss::InvertedLists *ivf = cpuIvfSq->invlists;
    EXPECT_NE(ivf, nullptr);

    int tmpTotal = 0;
    for (int i = 0; i < ncentroids; ++i) {
        size_t listSize = ivf->list_size(i);
        size_t listSize1 = ascendIndex.getListLength(i);
        EXPECT_EQ(listSize, listSize1);

        const uint8_t *codeCpu = ivf->get_codes(i);
        const faiss::Index::idx_t *idCpu = ivf->get_ids(i);

        std::vector<unsigned char> codes;
        std::vector<uint32_t> indices;
        ascendIndex.getListCodesAndIds(i, codes, indices);

        std::vector<uint32_t> indicesCpu(listSize, 0);
        transform(idCpu, idCpu + listSize, begin(indicesCpu), [](faiss::Index::idx_t x) { return uint32_t(x); });
        EXPECT_EQ(memcmp(codeCpu, codes.data(), codes.size()), 0);
        EXPECT_EQ(memcmp(indicesCpu.data(), indices.data(), indices.size() * sizeof(uint32_t)), 0);

        tmpTotal += listSize1;
    }
    EXPECT_EQ(tmpTotal, ntotal);

    for (int i = 0; i < 100; i++) {
        int k = 1000;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::Index::idx_t> label(k, 0);
        ascendIndex.search(1, data.data() + idx * dim, k, dist.data(), label.data());

        EXPECT_EQ(label[0], idx);
    }

    delete cpuIndex;
}

TEST(TestAscendIndexIVFSQ, CloneCPU2Ascend)
{
    int dim = 64;
    int ntotal = 2500000;
    int ncentroids = 1024;
    int nprobe = 64;

    srand48(0);
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    const char *indexfilename = "./ivfsqtest.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);
    faiss::IndexIVFScalarQuantizer *cpuIndex = dynamic_cast<faiss::IndexIVFScalarQuantizer *>(initIndex);

    // ascend index
    std::vector<int> devices = { 0, 1, 2, 3 };
    faiss::ascend::AscendIndexIVFSQ *ascendIndex =
        dynamic_cast<faiss::ascend::AscendIndexIVFSQ *>(faiss::ascend::index_cpu_to_ascend(devices, initIndex));
    ASSERT_FALSE(ascendIndex == nullptr);

    EXPECT_EQ(ascendIndex->d, dim);
    EXPECT_EQ(ascendIndex->ntotal, ntotal);
    EXPECT_EQ(ascendIndex->getNumLists(), ncentroids);
    EXPECT_EQ(ascendIndex->getNumProbes(), nprobe);
    EXPECT_EQ(ascendIndex->metric_type, cpuIndex->metric_type);
    EXPECT_EQ(ascendIndex->d, cpuIndex->d);
    EXPECT_EQ(ascendIndex->ntotal, cpuIndex->ntotal);
    EXPECT_EQ(ascendIndex->is_trained, cpuIndex->is_trained);
    EXPECT_EQ(ascendIndex->getNumLists(), cpuIndex->nlist);
    EXPECT_EQ(ascendIndex->getNumProbes(), cpuIndex->nprobe);

    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            int sizeCpuList = cpuIndex->get_list_size(i);
            int sizeAscendList = ascendIndex->getListLength(i);
            ASSERT_EQ(sizeCpuList, sizeAscendList) << "Failure idx:" << i;

            const uint8_t *codeCpu = cpuIndex->invlists->get_codes(i);
            const faiss::Index::idx_t *idCpu = cpuIndex->invlists->get_ids(i);

            std::vector<unsigned char> codes;
            std::vector<uint32_t> indices;
            ascendIndex->getListCodesAndIds(i, codes, indices);

            std::vector<uint32_t> indicesCpu(sizeCpuList, 0);
            transform(idCpu, idCpu + sizeCpuList, begin(indicesCpu), [](faiss::Index::idx_t x) { return uint32_t(x); });
            ASSERT_EQ(memcmp(codeCpu, codes.data(), codes.size()), 0) << "Failure idx:" << i;
            ASSERT_EQ(memcmp(indicesCpu.data(), indices.data(), indices.size() * sizeof(uint32_t)), 0) <<
                "Failure idx:" << i;

            tmpTotal += sizeAscendList;
        }
        EXPECT_EQ(tmpTotal, ntotal);
    }

    for (int i = 0; i < 100; i++) {
        int k = 1000;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::Index::idx_t> label(k, 0);
        ascendIndex->search(1, data.data() + idx * dim, k, dist.data(), label.data());

        EXPECT_EQ(label[0], idx);
    }

    delete ascendIndex;
    delete initIndex;
}

TEST(TestAscendIndexIVFSQ, remove)
{
    int dim = 64;
    int ntotal = 200000;
    int ncentroids = 1024;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexIVFSQConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQ index(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::METRIC_L2, true, conf);
    index.verbose = true;

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());

    faiss::IDSelectorRange del(0, 2);
    int rmCnt = 0;
    for (int i = 0; i < ncentroids; i++) {
        std::vector<uint8_t> code;
        std::vector<uint32_t> ids;

        index.getListCodesAndIds(i, code, ids);
        for (size_t k = 0; k < ids.size(); k++) {
            rmCnt += del.is_member((int64_t)ids[k]) ? 1 : 0;
        }
    }

    size_t rmedCnt = index.remove_ids(del);
    ASSERT_EQ(rmedCnt, rmCnt);
    ASSERT_EQ(index.ntotal, (ntotal - rmedCnt));

    int tmpTotal = 0;
    for (int i = 0; i < ncentroids; i++) {
        std::vector<uint8_t> code;
        std::vector<uint32_t> ids;
        index.getListCodesAndIds(i, code, ids);
        ASSERT_EQ(code.size(), ids.size() * index.d);
        for (size_t k = 0; k < ids.size(); k++) {
            ASSERT_FALSE(del.is_member((int64_t)ids[k])) << "failure index:" << i;
        }
        tmpTotal += index.getListLength(i);
    }

    EXPECT_EQ(tmpTotal, (ntotal - rmedCnt));
}

TEST(TestAscendIndexIVFSQ, search_IP)
{
    int dim = 64;
    int ntotal = 2500000;
    int ncentroids = 1024;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, dim);

    faiss::ascend::AscendIndexIVFSQConfig conf({ 0, 1 });
    faiss::ascend::AscendIndexIVFSQ index(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::METRIC_INNER_PRODUCT, false, conf);
    index.verbose = true;

    index.train(ntotal / 10, data.data());
    index.add(ntotal, data.data());

    {
        int n = 1;
        int k = 100;
        for (int i = 3; i < 10; i++) {
            std::vector<float> dist(k, 0);
            std::vector<faiss::Index::idx_t> label(k, 0);
            index.search(n, data.data() + i * dim, k, dist.data(), label.data());
            ASSERT_EQ(label[0], i);
        }
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
