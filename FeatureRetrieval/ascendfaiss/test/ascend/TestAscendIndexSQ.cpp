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

#include <numeric>
#include <random>
#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>

#include <sys/time.h>

namespace {
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;

inline void FastSrand(int seed)
{
    g_seed = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
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

inline void AssertEqual(std::vector<uint8_t> &gt, std::vector<uint8_t> &data)
{
    ASSERT_EQ(gt.size(), data.size());
    for (size_t i = 0; i < gt.size(); i++) {
        ASSERT_EQ(gt[i], data[i]) << "i: " << i << " gt: " << gt[i] << " data: " << data[i] << std::endl;
    }
}

TEST(TestAscendIndexSQ, All)
{
    int dim = 128;
    int ntotal = 200000;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    faiss::ascend::AscendIndexSQ index(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, conf);

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());
    EXPECT_EQ(index.ntotal, ntotal);

    int deviceCnt = conf.deviceList.size();
    std::vector<uint8_t> codes;
    {
        int totals = 0;
        for (int i = 0; i < deviceCnt; i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            std::vector<uint8_t> base(tmpTotal * dim);
            index.getBase(conf.deviceList[i], base);
            codes.insert(codes.end(), base.begin(), base.end());
            totals += tmpTotal;
        }
        EXPECT_EQ(totals, ntotal);
    }

    index.reset();

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());
    {
        int getTotal = 0;
        std::vector<uint8_t> baseData;
        for (int i = 0; i < deviceCnt; i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            std::vector<uint8_t> tmpBase(tmpTotal * dim);
            index.getBase(conf.deviceList[i], tmpBase);
            baseData.insert(baseData.end(), tmpBase.begin(), tmpBase.end());
            getTotal += tmpTotal;
        }
        EXPECT_EQ(getTotal, ntotal);
        AssertEqual(codes, baseData);
    }

    {
        int batch = 4;
        for (int i = ntotal - 40; i < ntotal; i += batch) {
            int k = 1000;
            std::vector<float> dist(k * batch, 0);
            std::vector<faiss::Index::idx_t> label(k * batch, 0);
            index.search(batch, data.data() + i * dim, k, dist.data(), label.data());
            ASSERT_EQ(label[0], i);
            ASSERT_EQ(label[k], i + 1);
            ASSERT_EQ(label[k * 2], i + 2);
            ASSERT_EQ(label[k * 3], i + 3);
            faiss::Index::idx_t assign;
            index.assign(1, data.data() + i * dim, &assign);
            ASSERT_EQ(assign, i);
        }
    }
}

TEST(TestAscendIndexSQ, AddWithIds)
{
    int dim = 128;
    int ntotal = 200000;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    std::vector<int64_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(ids), std::end(ids), rng);

    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    faiss::ascend::AscendIndexSQ index(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, conf);

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.train(ntotal, data.data());
    index.add_with_ids(ntotal, data.data(), ids.data());
    EXPECT_EQ(index.ntotal, ntotal);

    int deviceCnt = conf.deviceList.size();
    {
        int totals = 0;
        for (int i = 0; i < deviceCnt; i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            totals += tmpTotal;
        }
        EXPECT_EQ(totals, ntotal);
    }

    {
        for (int i = 135; i < 200; i++) {
            int k = 1000;
            std::vector<float> dist(k, 0);
            std::vector<faiss::Index::idx_t> label(k, 0);
            index.search(1, data.data() + i * dim, k, dist.data(), label.data());
            ASSERT_EQ(label[0], ids[i]);
            faiss::Index::idx_t assign;
            index.assign(1, data.data() + i * dim, &assign);
            ASSERT_EQ(assign, ids[i]);
        }
    }
}

TEST(TestAscendIndexSQ, CopyFrom)
{
    int dim = 128;
    int ntotal = 250000;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    faiss::IndexScalarQuantizer cpuIndex(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    cpuIndex.train(ntotal, data.data());
    cpuIndex.add(ntotal, data.data());

    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    faiss::ascend::AscendIndexSQ index(&cpuIndex, conf);

    EXPECT_EQ(index.d, dim);
    EXPECT_EQ(index.ntotal, ntotal);

    // only make sure the format of basedata is same
    faiss::IndexScalarQuantizer cpuIndexRef(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    index.copyTo(&cpuIndexRef);

    {
        int sizeAscend = 0;
        std::vector<uint8_t> codesAsend;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<uint8_t> base(size * dim);
            index.getBase(deviceId, base);
            codesAsend.insert(codesAsend.end(), base.begin(), base.end());
            sizeAscend += size;
        }
        ASSERT_EQ(ntotal, sizeAscend);
        AssertEqual(codesAsend, cpuIndexRef.codes);
    }
}

TEST(TestAscendIndexSQ, CopyTo)
{
    int dim = 128;
    int ntotal = 250000;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    faiss::ascend::AscendIndexSQ index(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, conf);

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());

    faiss::IndexScalarQuantizer cpuIndex(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    index.copyTo(&cpuIndex);

    EXPECT_EQ(cpuIndex.ntotal, ntotal);
    EXPECT_EQ(cpuIndex.d, dim);

    {
        int tmpTotal = 0;
        std::vector<uint8_t> codes;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<uint8_t> base(size * dim);
            index.getBase(deviceId, base);

            codes.insert(codes.end(), base.begin(), base.end());
            tmpTotal += size;
        }

        EXPECT_EQ(tmpTotal, ntotal);
        AssertEqual(codes, cpuIndex.codes);
    }
}

TEST(TestAscendIndexSQ, CloneAscend2CPU)
{
    int dim = 128;
    int n = 250000;
    int ntotal = n * 10;

    srand48(1000);
    std::vector<float> data(dim * n);
    for (int i = 0; i < dim * n; i++) {
        data[i] = drand48();
    }
    for (int i = 0; i < 20; i++) {
        printf("%.4f\t", data[i]);
    }
    // ascend index
    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    faiss::ascend::AscendIndexSQ ascendIndex(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2,
        conf);

    ascendIndex.train(n, data.data());
    // add ground truth
    ascendIndex.add(n, data.data());

    // add 2250w vector
    for (int i = 0; i < (ntotal / n - 1); i++) {
        std::vector<float> dataTmp(dim * n);
        for (int j = 0; j < dim * n; j++) {
            dataTmp[j] = drand48();
        }
        ascendIndex.add(n, dataTmp.data());
        printf("add %d times of data.\n", i);
    }

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&ascendIndex);
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./SQIndex_2500000.faiss";
    write_index(cpuIndex, outfilename);
    printf("write ascendIndex to file ok!\n");

    int lenall = 0;
    for (auto deviceId : conf.deviceList) {
        lenall += ascendIndex.getBaseSize(deviceId);
    }

    EXPECT_EQ(lenall, ntotal);

    for (int i = 0; i < 10; i++) {
        int k = 1000;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::Index::idx_t> label(k, 0);
        double ts = GetMillisecs();
        for (int j = 0; j < 500; j++) {
            ascendIndex.search(1, data.data() + idx * dim, k, dist.data(), label.data());
        }
        double te = GetMillisecs();
        printf("all %f, means %f.\n", te - ts, (te - ts) / 500);
        // printf("ascend idx %d dist %f %f %f %f %f lable %lu %lu %lu %lu %lu.\n", idx, dist[0],
        // 	dist[1], dist[2], dist[3], dist[4], label[0], label[1], label[2], label[3], label[4]);
        EXPECT_EQ(label[0], idx);
    }
    delete cpuIndex;
}

TEST(TestAscendIndexSQ, CloneCPU2Ascend)
{
    int dim = 128;
    int n = 250000;
    int ntotal = n * 10;
    srand48(1000);
    std::vector<float> data(dim * n);
    for (int i = 0; i < dim * n; i++) {
        data[i] = drand48();
    }
    for (int i = 0; i < 20; i++) {
        printf("%.4f\t", data[i]);
    }
    const char *indexfilename = "./SQIndex_2500000.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);

    // ascend index
    std::vector<int> devices = { 0 };
    faiss::ascend::AscendIndexSQ *ascendIndex =
        dynamic_cast<faiss::ascend::AscendIndexSQ *>(faiss::ascend::index_cpu_to_ascend(devices, initIndex));
    ASSERT_FALSE(ascendIndex == nullptr);

    int lenall = 0;
    for (auto deviceId : devices) {
        lenall += ascendIndex->getBaseSize(deviceId);
    }

    EXPECT_EQ(lenall, ntotal);

    for (int i = 0; i < 1; i++) {
        int k = 1000;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::Index::idx_t> label(k, 0);
        double ts = GetMillisecs();
        for (int i = 0; i < 500; i++) {
            ascendIndex->search(1, data.data() + idx * dim, k, dist.data(), label.data());
        }
        double te = GetMillisecs();
        printf("all %f, means %f.\n", te - ts, (te - ts) / 500);
        printf("ascend idx %d dist %f %f %f %f %f lable %lu %lu %lu %lu %lu.\n", idx, dist[0], dist[1], dist[2],
            dist[3], dist[4], label[0], label[1], label[2], label[3], label[4]);
        EXPECT_EQ(label[0], idx);
    }

    delete ascendIndex;
    delete initIndex;
}

TEST(TestAscendIndexSQ, removeRange)
{
    int dim = 128;
    int ntotal = 200000;
    int delRangeMin = 0;
    int delRangeMax = 4;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    faiss::ascend::AscendIndexSQ index(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, conf);

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());
    // define ids
    std::vector<int64_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    // save pre-delete basedata and index of vector
    std::vector<uint8_t> codePre;
    std::vector<faiss::Index::idx_t> idxMapPre;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<uint8_t> base(size * dim);
        index.getBase(deviceId, base);
        codePre.insert(codePre.end(), base.begin(), base.end());

        std::vector<faiss::Index::idx_t> idMap(size);
        index.getIdxMap(deviceId, idMap);
        idxMapPre.insert(idxMapPre.end(), idMap.begin(), idMap.end());
    }

    faiss::IDSelectorRange del(delRangeMin, delRangeMax);
    int rmCnt = 0;
    for (int i = 0; i < ntotal; i++) {
        rmCnt += del.is_member(ids[i]) ? 1 : 0;
    }

    size_t rmedCnt = index.remove_ids(del);
    ASSERT_EQ(rmedCnt, rmCnt);
    ASSERT_EQ(index.ntotal, (ntotal - rmedCnt));

    int tmpTotal = 0;
    for (auto deviceId : conf.deviceList) {
        tmpTotal += index.getBaseSize(deviceId);
    }
    EXPECT_EQ(tmpTotal, (ntotal - rmedCnt));

    std::vector<uint8_t> codes;
    std::vector<faiss::Index::idx_t> idxMap;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<uint8_t> base(size * dim);
        index.getBase(deviceId, base);
        codes.insert(codes.end(), base.begin(), base.end());

        std::vector<faiss::Index::idx_t> idMap(size);
        index.getIdxMap(deviceId, idMap);
        idxMap.insert(idxMap.end(), idMap.begin(), idMap.end());
    }
    EXPECT_EQ(idxMapPre.size(), idxMap.size() + rmedCnt);
    EXPECT_EQ(codePre.size(), codes.size() + rmedCnt * dim);
    {
        // check idx
        int offset = 0;
        for (size_t i = 0; i < idxMap.size(); i++) {
            if ((idxMapPre[i] >= delRangeMin) && (idxMapPre[i] < delRangeMax)) {
                // check idx
                EXPECT_EQ(idxMapPre[idxMap.size() + i], idxMap[i]);
                // check vector
                for (int j = 0; j < dim; j++) {
                    EXPECT_EQ(codePre[(idxMap.size() + i) * dim + j], codes[i * dim + j]);
                }
                offset += 1;
            } else {
                int ptr = i * dim;
                // check idx
                EXPECT_EQ(idxMapPre[i], idxMap[i]);
                // check vector
                for (int j = 0; j < dim; j++) {
                    EXPECT_EQ(codePre[ptr + j], codes[ptr + j]);
                }
            }
        }
        EXPECT_EQ(offset, delRangeMax - delRangeMin);
    }
}

TEST(TestAscendIndexSQ, removeBatch)
{
    int dim = 128;
    int ntotal = 200000;
    std::vector<faiss::Index::idx_t> delBatchs = { 1, 23, 50, 10000 };

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    faiss::ascend::AscendIndexSQ index(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, conf);

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());
    // define ids
    std::vector<int64_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    // save pre-delete basedata and index of vector
    std::vector<uint8_t> codesPre;
    std::vector<faiss::Index::idx_t> idxMapPre;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<uint8_t> base(size * dim);
        index.getBase(deviceId, base);
        codesPre.insert(codesPre.end(), base.begin(), base.end());

        std::vector<faiss::Index::idx_t> idMap(size);
        index.getIdxMap(deviceId, idMap);
        idxMapPre.insert(idxMapPre.end(), idMap.begin(), idMap.end());
    }

    faiss::IDSelectorBatch del(delBatchs.size(), delBatchs.data());
    int rmCnt = 0;
    for (int i = 0; i < ntotal; i++) {
        rmCnt += del.is_member(ids[i]) ? 1 : 0;
    }

    size_t rmedCnt = index.remove_ids(del);
    ASSERT_EQ(rmedCnt, rmCnt);
    ASSERT_EQ(index.ntotal, (ntotal - rmedCnt));

    int tmpTotal = 0;
    for (auto deviceId : conf.deviceList) {
        tmpTotal += index.getBaseSize(deviceId);
    }
    EXPECT_EQ(tmpTotal, (ntotal - rmedCnt));

    std::vector<uint8_t> codes;
    std::vector<faiss::Index::idx_t> idxMap;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<uint8_t> base(size * dim);
        index.getBase(deviceId, base);
        codes.insert(codes.end(), base.begin(), base.end());

        std::vector<faiss::Index::idx_t> idMap(size);
        index.getIdxMap(deviceId, idMap);
        idxMap.insert(idxMap.end(), idMap.begin(), idMap.end());
    }
    EXPECT_EQ(idxMapPre.size(), idxMap.size() + rmedCnt);
    EXPECT_EQ(codesPre.size(), codes.size() + rmedCnt * dim);
    {
        int offset = 0;
        for (size_t i = 0; i < idxMap.size(); i++) {
            if (del.set.find(idxMapPre[i]) != del.set.end()) {
                // check ids
                EXPECT_EQ(idxMapPre[idxMap.size() + offset], idxMap[i]);
                // check vector
                for (int j = 0; j < dim; j++) {
                    EXPECT_EQ(codesPre[(idxMap.size() + offset) * dim + j], codes[i * dim + j]);
                }

                offset += 1;
            } else {
                int ptr = i * dim;
                // check ids
                EXPECT_EQ(idxMapPre[i], idxMap[i]);
                // check vector
                for (int j = 0; j < dim; j++) {
                    EXPECT_EQ(codesPre[ptr + j], codes[ptr + j]);
                }
            }
        }
        EXPECT_EQ(offset, del.set.size());
    }
}

TEST(TestAscendIndexSQ, QPS)
{
    std::vector<int> dim = { 512 };
    std::vector<size_t> ntotal = { 1000000 };
    std::vector<int> searchNum = { 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 1000 };

    size_t maxSize = ntotal.back() * dim.back();
    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
    }
    for (size_t i = 0; i < dim.size(); i++) {
        faiss::ascend::AscendIndexSQConfig conf({ 0, 1, 2, 3 });
        faiss::ascend::AscendIndexSQ index(dim[i], faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2,
            conf);

        for (size_t j = 0; j < ntotal.size(); j++) {
            index.reset();
            for (auto deviceId : conf.deviceList) {
                int len = index.getBaseSize(deviceId);
                ASSERT_EQ(len, 0);
            }

            index.train(ntotal[j], data.data());
            index.add(ntotal[j], data.data());
            {
                int getTotal = 0;
                for (size_t k = 0; k < conf.deviceList.size(); k++) {
                    int tmpTotal = index.getBaseSize(conf.deviceList[k]);
                    getTotal += tmpTotal;
                }
                EXPECT_EQ(getTotal, ntotal[j]);
            }

            {
                for (size_t n = 0; n < searchNum.size(); n++) {
                    int k = 100;
                    int loopTimes = 100;
                    std::vector<float> dist(searchNum[n] * k, 0);
                    std::vector<faiss::Index::idx_t> label(searchNum[n] * k, 0);
                    double ts = GetMillisecs();
                    for (int l = 0; l < loopTimes; l++) {
                        index.search(searchNum[n], data.data(), k, dist.data(), label.data());
                    }
                    double te = GetMillisecs();
                    int cases = i * ntotal.size() * searchNum.size() + j * searchNum.size() + n;
                    printf("case[%d]: base:%zu, dim:%d, search num:%d, QPS:%.4f\n", cases, ntotal[j], dim[i],
                        searchNum[n], 1000 * searchNum[n] * loopTimes / (te - ts));
                }
            }
        }
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}