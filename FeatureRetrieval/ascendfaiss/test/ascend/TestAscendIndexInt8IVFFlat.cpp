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
#include <faiss/ascend/AscendIndexInt8IVFFlat.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendAutoTune.h>
#include <faiss/ascend/AscendClustering.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/Clustering.h>
#include <faiss/index_io.h>

namespace {
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

struct Trainer {
    Trainer(int d, int numList, bool useKmeans, bool ver) : dim(d), nlist(numList), useKmeansPP(useKmeans), verbose(ver)
    {
        cpuQuantizer = new faiss::IndexFlatL2(dim);
        SetDefaultClusteringConfig();
    }

    ~Trainer()
    {
        delete cpuQuantizer;
        cpuQuantizer = nullptr;
    }

    inline void SetDefaultClusteringConfig()
    {
        // here we set a low # iterations because this is typically used
        // for large clusterings
        const int niter = 10;
        cp.niter = niter;
    }

    void Train(int n, int dim, const float *x)
    {
        cpuQuantizer->reset();
        if (this->useKmeansPP) {
            faiss::ascend::AscendClustering clus(dim, nlist, cp);
            clus.verbose = verbose;
            clus.train(n, x, *cpuQuantizer);
        } else {
            faiss::Clustering clus(dim, nlist, cp);
            clus.verbose = verbose;
            clus.train(n, x, *cpuQuantizer);
        }
    }

    void GetCoarseCenter(std::vector<int8_t> &data)
    {
        transform(begin(cpuQuantizer->xb), end(cpuQuantizer->xb), begin(data),
            [](float temp) { return static_cast<int8_t>(temp); });
    }

    faiss::ClusteringParameters cp;
    faiss::IndexFlat *cpuQuantizer;
    int dim;
    int nlist;
    bool useKmeansPP;
    bool verbose;
};

TEST(TestAscendIndexInt8IVFFlat, All)
{
    int dim = 128;
    int ntotal = 250000;
    int nlist = 1024;
    int nprobe = 4;

    faiss::ascend::AscendIndexInt8IVFFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexInt8IVFFlat index(dim, nlist, faiss::METRIC_L2, conf);
    index.verbose = true;

    std::vector<int8_t> base(dim * ntotal);
    GenerateCodes(base.data(), ntotal, dim);

    std::vector<float> train(ntotal / 10 * dim);
    transform(base.data(), base.data() + train.size(), train.begin(), [](int8_t x) { return static_cast<float>(x); });
    auto *trainer = new Trainer(dim, nlist, false, true);
    trainer->Train(ntotal / 10, dim, train.data());

    std::vector<int8_t> coarseCenter(nlist * dim);
    trainer->GetCoarseCenter(coarseCenter);

    index.setNumProbes(nprobe);
    index.updateCentroids(nlist, coarseCenter.data());
    EXPECT_EQ(index.getNumProbes(), nprobe);

    index.add(ntotal, base.data());
    ASSERT_EQ(index.ntotal, ntotal);
    {
        int tmpTotal = 0;
        for (int i = 0; i < nlist; i++) {
            tmpTotal += index.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, ntotal);
    }

    std::vector<std::vector<int8_t>> codes(nlist, std::vector<int8_t>());
    std::vector<std::vector<uint32_t>> indices(nlist, std::vector<uint32_t>());
    for (int i = 0; i < nlist; i++) {
        index.getListCodesAndIds(i, codes[i], indices[i]);
        ASSERT_EQ(codes[i].size(), indices[i].size() * index.d);
    }

    index.reset();
    for (int i = 0; i < nlist; i++) {
        int len = index.getListLength(i);
        ASSERT_EQ(len, 0);
    }

    index.reserveMemory(ntotal);
    index.add(ntotal, base.data());
    {
        int tmpTotal = 0;
        for (int i = 0; i < nlist; i++) {
            tmpTotal += index.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, ntotal);
    }

    for (int i = 0; i < nlist; i++) {
        std::vector<int8_t> tmpCode;
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
            index.search(n, base.data() + i * dim, k, dist.data(), label.data());
            ASSERT_EQ(label[0], i);

            faiss::Index::idx_t assign;
            index.assign(1, base.data() + i * dim, &assign);
            ASSERT_EQ(assign, i);
        }
    }
    delete trainer;
}

TEST(TestAscendIndexInt8IVFFlat, CopyTo)
{
    int dim = 128;
    int ntotal = 2500000;
    int nlist = 1024;
    int nprobe = 64;

    int seed = 1000;
    std::vector<int8_t> data(dim * ntotal);
    GenerateCodes(data.data(), ntotal, dim, seed);

    // ascend index
    faiss::ascend::AscendIndexInt8IVFFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexInt8IVFFlat ascendIndex(dim, nlist, faiss::METRIC_L2, conf);
    ascendIndex.verbose = true;
    ascendIndex.setNumProbes(nprobe);
    EXPECT_EQ(ascendIndex.getNumProbes(), nprobe);

    std::vector<float> train(ntotal / 10 * dim);
    transform(data.data(), data.data() + train.size(), train.begin(), [](int8_t x) { return static_cast<float>(x); });
    auto *trainer = new Trainer(dim, nlist, false, true);
    trainer->Train(ntotal / 10, dim, train.data());

    std::vector<int8_t> coarseCenter(nlist * dim);
    trainer->GetCoarseCenter(coarseCenter);

    ascendIndex.updateCentroids(nlist, coarseCenter.data());

    // add data
    ascendIndex.reserveMemory(ntotal);
    ascendIndex.add(ntotal, data.data());

    faiss::IndexIVFScalarQuantizer cpuIvfSq;
    ascendIndex.copyTo(&cpuIvfSq);

    delete trainer;
}

TEST(TestAscendIndexInt8IVFFlat, CloneAscend2CPU)
{
    int dim = 128;
    int ntotal = 2500000;
    int nlist = 1024;
    int nprobe = 64;

    int seed = 1000;
    std::vector<int8_t> data(dim * ntotal);
    GenerateCodes(data.data(), ntotal, dim, seed);

    for (int i = 0; i < 10; i++) {
        printf("%d\t", int(data[i]));
    }
    printf("\n");

    // ascend index
    faiss::ascend::AscendIndexInt8IVFFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexInt8IVFFlat ascendIndex(dim, nlist, faiss::METRIC_L2, conf);
    ascendIndex.verbose = true;
    ascendIndex.setNumProbes(nprobe);
    EXPECT_EQ(ascendIndex.getNumProbes(), nprobe);

    std::vector<float> train(ntotal / 10 * dim);
    transform(data.data(), data.data() + train.size(), train.begin(), [](int8_t x) { return static_cast<float>(x); });
    auto *trainer = new Trainer(dim, nlist, false, true);
    trainer->Train(ntotal / 10, dim, train.data());

    std::vector<int8_t> coarseCenter(nlist * dim);
    trainer->GetCoarseCenter(coarseCenter);

    ascendIndex.updateCentroids(nlist, coarseCenter.data());

    // add data
    ascendIndex.reserveMemory(ntotal);
    ascendIndex.add(ntotal, data.data());

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_int8_ascend_to_cpu(&ascendIndex);
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./int8ivfflattest.faiss";
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
    for (int i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);
        size_t listSize1 = ascendIndex.getListLength(i);
        EXPECT_EQ(listSize, listSize1);

        const uint8_t *codeCpu = ivf->get_codes(i);
        const faiss::Index::idx_t *idCpu = ivf->get_ids(i);

        std::vector<signed char> codes;
        std::vector<uint32_t> indices;
        ascendIndex.getListCodesAndIds(i, codes, indices);

        std::vector<uint32_t> indicesCpu(listSize, 0);
        transform(idCpu, idCpu + listSize, begin(indicesCpu), [](faiss::Index::idx_t x) { return uint32_t(x); });
        EXPECT_EQ(memcmp(reinterpret_cast<const int8_t *>(codeCpu), codes.data(), codes.size()), 0);
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
    delete trainer;
}

TEST(TestAscendIndexInt8IVFFlat, CloneCPU2Ascend)
{
    int dim = 128;
    int ntotal = 2500000;
    int ncentroids = 1024;
    int nprobe = 64;

    int seed = 1000;
    std::vector<int8_t> data(dim * ntotal);
    GenerateCodes(data.data(), ntotal, dim, seed);

    for (int i = 0; i < 10; i++) {
        printf("%d\t", int(data[i]));
    }
    printf("\n");

    const char *indexfilename = "./int8ivfflattest.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);
    faiss::IndexIVFScalarQuantizer *cpuIndex = dynamic_cast<faiss::IndexIVFScalarQuantizer *>(initIndex);

    // ascend index
    std::vector<int> devices = { 0 };
    faiss::ascend::AscendIndexInt8IVFFlat *ascendIndex = dynamic_cast<faiss::ascend::AscendIndexInt8IVFFlat *>(
        faiss::ascend::index_int8_cpu_to_ascend(devices, initIndex));
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

            std::vector<signed char> codes;
            std::vector<uint32_t> indices;
            ascendIndex->getListCodesAndIds(i, codes, indices);

            std::vector<uint32_t> indicesCpu(sizeCpuList, 0);
            transform(idCpu, idCpu + sizeCpuList, begin(indicesCpu), [](faiss::Index::idx_t x) { return uint32_t(x); });
            ASSERT_EQ(memcmp(reinterpret_cast<const int8_t *>(codeCpu), codes.data(), codes.size()), 0) <<
                "Failure idx:" << i;
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

TEST(TestAscendIndexInt8IVFFlat, QPS)
{
    int dim = 128;
    size_t ntotal = 2500000;
    int nlist = 1024;
    int nprobe = 4;
    std::vector<int> searchNum = { 1, 2, 4, 8, 16, 32 };

    faiss::ascend::AscendIndexInt8IVFFlatConfig conf({ 1 });
    faiss::ascend::AscendIndexInt8IVFFlat index(dim, nlist, faiss::METRIC_L2, conf);
    index.verbose = true;

    printf("generate data\n");
    std::vector<int8_t> base(ntotal * dim);
    GenerateCodes(base.data(), ntotal, dim);

    std::vector<float> train(ntotal / 100 * dim);
    transform(base.data(), base.data() + train.size(), train.begin(), [](int8_t x) { return static_cast<float>(x); });
    auto *trainer = new Trainer(dim, nlist, false, true);
    trainer->Train(ntotal / 100, dim, train.data());

    std::vector<int8_t> coarseCenter(nlist * dim);
    trainer->GetCoarseCenter(coarseCenter);

    index.setNumProbes(nprobe);
    index.updateCentroids(nlist, coarseCenter.data());
    printf("add data\n");
    index.add(ntotal, base.data());
    {
        for (size_t n = 0; n < searchNum.size(); n++) {
            int k = 128;
            int loopTimes = 10;
            std::vector<float> dist(searchNum[n] * k, 0);
            std::vector<faiss::Index::idx_t> label(searchNum[n] * k, 0);
            double ts = GetMillisecs();
            for (int l = 0; l < loopTimes; l++) {
                index.search(searchNum[n], base.data(), k, dist.data(), label.data());
            }
            double te = GetMillisecs();
            printf("case[%zu]: base:%zu, dim:%d, search num:%d, QPS:%.4f\n", n, ntotal, dim, searchNum[n],
                1000 * searchNum[n] * loopTimes / (te - ts));
        }
    }
    delete trainer;
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
