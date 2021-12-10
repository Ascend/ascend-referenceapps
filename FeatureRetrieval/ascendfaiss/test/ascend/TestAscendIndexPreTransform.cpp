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
#include <faiss/ascend/AscendIndexPreTransform.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/ascend/AscendIndexIVFSQ.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendVTCloner.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>

namespace {
TEST(TestAscendIndexPreTransform, All)
{
    int dimIn = 512;
    int dim = 128;
    int ntotal = 200000;
    int ncentroids = 16384;
    int nprobe = 64;

    std::vector<float> data(dimIn * ntotal);
    for (int i = 0; i < dimIn * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexIVFSQConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQ subIndex(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::METRIC_L2, true, conf);
    subIndex.verbose = true;
    subIndex.setNumProbes(nprobe);

    faiss::ascend::AscendIndexPreTransform index(&subIndex);
    index.verbose = true;
    index.prependTransform<faiss::ascend::AscendPCAMatrix>(dimIn, dim, 0.0f, true);

    index.train(ntotal, data.data());

    for (int i = 0; i < ncentroids; i++) {
        int len = subIndex.getListLength(i);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, data.data());
    ASSERT_EQ(index.ntotal, ntotal);
    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            tmpTotal += subIndex.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, ntotal);
    }

    std::vector<std::vector<uint8_t>> codes(ncentroids, std::vector<uint8_t>());
    std::vector<std::vector<uint32_t>> indices(ncentroids, std::vector<uint32_t>());
    for (int i = 0; i < ncentroids; i++) {
        subIndex.getListCodesAndIds(i, codes[i], indices[i]);
        ASSERT_EQ(codes[i].size(), indices[i].size() * subIndex.d);
    }

    index.reset();
    for (int i = 0; i < ncentroids; i++) {
        int len = subIndex.getListLength(i);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, data.data());
    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            tmpTotal += subIndex.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, ntotal);
    }

    for (int i = 0; i < ncentroids; i++) {
        std::vector<uint8_t> tmpCode;
        std::vector<uint32_t> tmpIndi;
        subIndex.getListCodesAndIds(i, tmpCode, tmpIndi);
        ASSERT_EQ(tmpCode.size(), tmpIndi.size() * subIndex.d) << "failure subindex:" << i;
        ASSERT_EQ(tmpCode.size(), codes[i].size()) << "failure subindex:" << i;
        ASSERT_EQ(tmpIndi.size(), indices[i].size()) << "failure subindex:" << i;
        ASSERT_EQ(memcmp(tmpCode.data(), codes[i].data(), indices[i].size()), 0) << "failure subindex:" << i;
        ASSERT_EQ(memcmp(tmpIndi.data(), indices[i].data(), indices[i].size() * sizeof(uint32_t)), 0) <<
            "failure subindex:" << i;
    }

    {
        const std::vector<int> nums { 1, 31, 100, 200, 400, 600 };
        int idx = 3;
        for (auto n : nums) {
            int k = 1000;
            std::vector<float> dist(n * k, 0);
            std::vector<faiss::Index::idx_t> label(n * k, 0);
            index.search(n, data.data() + idx * dimIn, k, dist.data(), label.data());
            for (int j = 0; j < n; j++) {
                ASSERT_EQ(label[j * k], j + idx);
            }
        }
        faiss::Index::idx_t assign;
        index.assign(1, data.data() + idx * dimIn, &assign);
        ASSERT_EQ(assign, idx);
    }
}


TEST(TestAscendIndexPreTransform, CloneAscend2CPU)
{
    int dimIn = 512;
    int dim = 128;
    int ntotal = 200000;
    int ncentroids = 16384;
    int nprobe = 64;

    srand48(0);
    std::vector<float> data(dimIn * ntotal);
    for (int i = 0; i < dimIn * ntotal; i++) {
        data[i] = drand48();
    }

    // ascend index
    faiss::ascend::AscendIndexIVFSQConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQ subIndex(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::METRIC_L2, true, conf);
    subIndex.verbose = true;
    subIndex.setNumProbes(nprobe);
    EXPECT_EQ(subIndex.getNumProbes(), nprobe);
    faiss::ascend::AscendIndexPreTransform ascendIndex(&subIndex);
    ascendIndex.verbose = true;
    ascendIndex.prependTransform<faiss::ascend::AscendPCAMatrix>(dimIn, dim, 0.0f, true);

    ascendIndex.train(ntotal, data.data());

    // add data
    ascendIndex.add(ntotal, data.data());

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&ascendIndex);
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./pttest.faiss";
    write_index(cpuIndex, outfilename);
    printf("write ascendIndex to file ok!\n");

    faiss::IndexPreTransform *cpuPt = dynamic_cast<faiss::IndexPreTransform *>(cpuIndex);
    EXPECT_EQ(ascendIndex.metric_type, cpuPt->metric_type);
    EXPECT_EQ(ascendIndex.d, cpuPt->d);
    EXPECT_EQ(ascendIndex.ntotal, cpuPt->ntotal);
    EXPECT_EQ(ascendIndex.is_trained, cpuPt->is_trained);

    auto cpuIvfSq = dynamic_cast<faiss::IndexIVFScalarQuantizer *>(cpuPt->index);
    EXPECT_EQ(subIndex.getNumLists(), cpuIvfSq->nlist);
    EXPECT_EQ(subIndex.getNumProbes(), cpuIvfSq->nprobe);
    const faiss::InvertedLists *ivf = cpuIvfSq->invlists;
    EXPECT_NE(ivf, nullptr);

    int tmpTotal = 0;
    for (int i = 0; i < ncentroids; ++i) {
        size_t listSize = ivf->list_size(i);
        size_t listSize1 = subIndex.getListLength(i);
        EXPECT_EQ(listSize, listSize1);

        const uint8_t *codeCpu = ivf->get_codes(i);
        const faiss::Index::idx_t *idCpu = ivf->get_ids(i);

        std::vector<unsigned char> codes;
        std::vector<uint32_t> indices;
        subIndex.getListCodesAndIds(i, codes, indices);

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
        ascendIndex.search(1, data.data() + idx * dimIn, k, dist.data(), label.data());

        EXPECT_EQ(label[0], idx);
    }

    delete cpuIndex;
}

TEST(TestAscendIndexPreTransform, CloneCPU2Ascend)
{
    int dimIn = 512;
    int ntotal = 200000;
    int ncentroids = 16384;
    int nprobe = 64;

    srand48(0);
    std::vector<float> data(dimIn * ntotal);
    for (int i = 0; i < dimIn * ntotal; i++) {
        data[i] = drand48();
    }

    const char *indexfilename = "./pttest.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);
    faiss::IndexPreTransform *cpuIndex = dynamic_cast<faiss::IndexPreTransform *>(initIndex);

    // ascend index
    std::vector<int> devices = { 0, 1, 2, 3 };
    faiss::ascend::AscendIndexPreTransform *ascendIndex =
        dynamic_cast<faiss::ascend::AscendIndexPreTransform *>(faiss::ascend::index_cpu_to_ascend(devices, initIndex));
    ASSERT_FALSE(ascendIndex == nullptr);

    EXPECT_EQ(ascendIndex->d, cpuIndex->d);
    EXPECT_EQ(ascendIndex->ntotal, ntotal);
    EXPECT_EQ(ascendIndex->metric_type, cpuIndex->metric_type);
    EXPECT_EQ(ascendIndex->d, cpuIndex->d);
    EXPECT_EQ(ascendIndex->ntotal, cpuIndex->ntotal);
    EXPECT_EQ(ascendIndex->is_trained, cpuIndex->is_trained);

    auto subIndex = dynamic_cast<const faiss::ascend::AscendIndexIVFSQ *>(ascendIndex->getIndex());
    EXPECT_EQ(subIndex->getNumLists(), ncentroids);
    EXPECT_EQ(subIndex->getNumProbes(), nprobe);
    auto cpuIvfSq = dynamic_cast<faiss::IndexIVFScalarQuantizer *>(cpuIndex->index);
    EXPECT_EQ(subIndex->getNumLists(), cpuIvfSq->nlist);
    EXPECT_EQ(subIndex->getNumProbes(), cpuIvfSq->nprobe);

    {
        int tmpTotal = 0;

        for (int i = 0; i < ncentroids; i++) {
            int sizeCpuList = cpuIvfSq->get_list_size(i);
            int sizeAscendList = subIndex->getListLength(i);
            ASSERT_EQ(sizeCpuList, sizeAscendList) << "Failure idx:" << i;

            const uint8_t *codeCpu = cpuIvfSq->invlists->get_codes(i);
            const faiss::Index::idx_t *idCpu = cpuIvfSq->invlists->get_ids(i);

            std::vector<unsigned char> codes;
            std::vector<uint32_t> indices;
            subIndex->getListCodesAndIds(i, codes, indices);

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
        ascendIndex->search(1, data.data() + idx * dimIn, k, dist.data(), label.data());

        EXPECT_EQ(label[0], idx);
    }

    delete ascendIndex;
    delete initIndex;
}

TEST(TestAscendIndexPreTransform, remove)
{
    int dimIn = 512;
    int dim = 128;
    int ntotal = 200000;
    int ncentroids = 16384;
    int nprobe = 64;

    std::vector<float> data(dimIn * ntotal);
    for (int i = 0; i < dimIn * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexIVFSQConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQ subIndex(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::METRIC_L2, true, conf);
    subIndex.verbose = true;
    subIndex.setNumProbes(nprobe);

    faiss::ascend::AscendIndexPreTransform index(&subIndex);
    index.verbose = true;
    index.prependTransform<faiss::ascend::AscendPCAMatrix>(dimIn, dim, 0.0f, true);

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());

    faiss::IDSelectorRange del(0, 2);
    int rmCnt = 0;
    for (int i = 0; i < ncentroids; i++) {
        std::vector<uint8_t> code;
        std::vector<uint32_t> ids;

        subIndex.getListCodesAndIds(i, code, ids);
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
        subIndex.getListCodesAndIds(i, code, ids);
        ASSERT_EQ(code.size(), ids.size() * subIndex.d);
        for (size_t k = 0; k < ids.size(); k++) {
            ASSERT_FALSE(del.is_member((int64_t)ids[k])) << "failure index:" << i;
        }
        tmpTotal += subIndex.getListLength(i);
    }

    EXPECT_EQ(tmpTotal, (ntotal - rmedCnt));
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
