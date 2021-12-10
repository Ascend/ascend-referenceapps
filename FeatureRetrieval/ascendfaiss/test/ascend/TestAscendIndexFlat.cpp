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
#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <cstring>
#include <sys/time.h>
#include <faiss/index_io.h>
#include <cstdlib>
#include <ascenddaemon/utils/StaticUtils.h>

namespace {
inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline void AssertEqual(const std::vector<float> &gt, const std::vector<float> &data)
{
    const float epson = 1e-3;
    ASSERT_EQ(gt.size(), data.size());
    for (size_t i = 0; i < gt.size(); i++) {
        ASSERT_TRUE(fabs(gt[i] - data[i]) <= epson) << "i: " << i << " gt: " << gt[i] << " data: " << data[i] <<
            std::endl;
    }
}

TEST(TestAscendIndexFlat, All)
{
    int dim = 512;
    int ntotal = 200000;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexFlatConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, data.data());
    EXPECT_EQ(index.ntotal, ntotal);

    int deviceCnt = conf.deviceList.size();
    std::vector<float> xb;
    {
        int totals = 0;
        for (int i = 0; i < deviceCnt; i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            std::vector<float> base(tmpTotal * dim);
            index.getBase(conf.deviceList[i], base);
            xb.insert(xb.end(), base.begin(), base.end());
            totals += tmpTotal;
        }
        EXPECT_EQ(totals, ntotal);
    }

    index.reset();

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, data.data());
    {
        int getTotal = 0;
        std::vector<float> baseData;
        for (int i = 0; i < deviceCnt; i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            std::vector<float> tmpBase(tmpTotal * dim);
            index.getBase(conf.deviceList[i], tmpBase);
            baseData.insert(baseData.end(), tmpBase.begin(), tmpBase.end());
            getTotal += tmpTotal;
        }
        EXPECT_EQ(getTotal, ntotal);
        AssertEqual(xb, baseData);
    }

    {
        int batch = 4;
        for (int i = ntotal - 240; i < ntotal; i += batch) {
            int k = 100;
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

TEST(TestAscendIndexFlat, AddWithIds)
{
    int dim = 512;
    int ntotal = 200000;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    std::vector<int64_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(ids), std::end(ids), rng);

    faiss::ascend::AscendIndexFlatConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.add_with_ids(ntotal, data.data(), ids.data());
    EXPECT_EQ(index.ntotal, ntotal);

    int deviceCnt = conf.deviceList.size();
    std::vector<float> xb;
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

TEST(TestAscendIndexFlat, CopyFrom)
{
    int dim = 512;
    int ntotal = 250000;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(ntotal, data.data());

    faiss::ascend::AscendIndexFlatConfig conf({ 0, 1 });
    faiss::ascend::AscendIndexFlat index(&cpuIndex, conf);

    EXPECT_EQ(index.d, dim);
    EXPECT_EQ(index.ntotal, ntotal);

    // only make sure the format of basedata is same
    faiss::IndexFlatL2 cpuIndexRef(dim);
    index.copyTo(&cpuIndexRef);

    {
        int sizeAscend = 0;
        std::vector<float> xbAsend;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<float> base(size * dim);
            index.getBase(deviceId, base);
            xbAsend.insert(xbAsend.end(), base.begin(), base.end());
            sizeAscend += size;
        }
        ASSERT_EQ(ntotal, sizeAscend);
        AssertEqual(xbAsend, cpuIndexRef.xb);
    }
}

TEST(TestAscendIndexFlat, CopyTo)
{
    int dim = 512;
    int ntotal = 250000;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexFlatConfig conf({ 0, 1 });
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);

    index.add(ntotal, data.data());

    faiss::IndexFlatL2 cpuIndex(dim);
    index.copyTo(&cpuIndex);

    EXPECT_EQ(cpuIndex.ntotal, ntotal);
    EXPECT_EQ(cpuIndex.d, dim);

    {
        int tmpTotal = 0;
        std::vector<float> xb;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<float> base(size * dim);
            index.getBase(deviceId, base);

            xb.insert(xb.end(), base.begin(), base.end());
            tmpTotal += size;
        }

        EXPECT_EQ(tmpTotal, ntotal);
        AssertEqual(xb, cpuIndex.xb);
    }
}

TEST(TestAscendIndexFlat, CloneAscend2CPU)
{
    int dim = 512;
    int ntotal = 250000;
    int xbSize = ntotal * 4;

    srand48(1000);
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    for (int i = 0; i < 20; i++) {
        printf("%.4f\t", data[i]);
    }
    // ascend index
    faiss::ascend::AscendIndexFlatConfig conf({ 0, 1 });
    faiss::ascend::AscendIndexFlat ascendIndex(dim, faiss::METRIC_L2, conf);

    // add ground truth
    ascendIndex.add(ntotal, data.data());

    // add 2250w vector
    for (int i = 0; i < (xbSize / ntotal - 1); i++) {
        std::vector<float> dataTmp(dim * ntotal);
        for (int i = 0; i < dim * ntotal; i++) {
            dataTmp[i] = drand48();
        }
        ascendIndex.add(ntotal, dataTmp.data());
        printf("add %d times of ntotal data.\n", i);
    }

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&ascendIndex);
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./flatIndex_2500000.faiss";
    write_index(cpuIndex, outfilename);
    printf("write ascendIndex to file ok!\n");

    int lenall = 0;
    for (auto deviceId : conf.deviceList) {
        lenall += ascendIndex.getBaseSize(deviceId);
    }

    EXPECT_EQ(lenall, xbSize);

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

TEST(TestAscendIndexFlat, CloneCPU2Ascend)
{
    int dim = 512;
    int ntotal = 250000;
    int xbSize = ntotal * 4;
    srand48(1000);
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    for (int i = 0; i < 20; i++) {
        printf("%.4f\t", data[i]);
    }
    const char *indexfilename = "./flatIndex_2500000.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);

    // ascend index
    std::vector<int> devices = { 0 };
    faiss::ascend::AscendIndexFlat *ascendIndex =
        dynamic_cast<faiss::ascend::AscendIndexFlat *>(faiss::ascend::index_cpu_to_ascend(devices, initIndex));
    ASSERT_FALSE(ascendIndex == nullptr);

    int lenall = 0;
    for (auto deviceId : devices) {
        lenall += ascendIndex->getBaseSize(deviceId);
    }

    EXPECT_EQ(lenall, xbSize);

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

TEST(TestAscendIndexFlat, removeRange)
{
    int dim = 512;
    int ntotal = 200000;
    int delRangeMin = 0;
    int delRangeMax = 4;
    const float epson = 1e-3;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexFlatConfig conf({ 0, 1, 2 });
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);

    index.add(ntotal, data.data());
    // define ids
    std::vector<int64_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    // save pre-delete basedata and index of vector
    std::vector<float> xbPre;
    std::vector<faiss::Index::idx_t> idxMapPre;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<float> base(size * dim);
        index.getBase(deviceId, base);
        xbPre.insert(xbPre.end(), base.begin(), base.end());

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

    std::vector<float> xb;
    std::vector<faiss::Index::idx_t> idxMap;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<float> base(size * dim);
        index.getBase(deviceId, base);
        xb.insert(xb.end(), base.begin(), base.end());

        std::vector<faiss::Index::idx_t> idMap(size);
        index.getIdxMap(deviceId, idMap);
        idxMap.insert(idxMap.end(), idMap.begin(), idMap.end());
    }
    EXPECT_EQ(idxMapPre.size(), idxMap.size() + rmedCnt);
    EXPECT_EQ(xbPre.size(), xb.size() + rmedCnt * dim);
    {
        int ntotal0 = ascend::utils::divUp(ntotal, conf.deviceList.size());
        // check idx
        for (size_t i = 0; i < ntotal0 - rmedCnt; i++) {
            if ((idxMapPre[i] >= delRangeMin) && (idxMapPre[i] < delRangeMax)) {
                int fptr = (ntotal0 - rmedCnt + i) * dim;
                int bptr = i * dim;
                // check vector
                for (int j = 0; j < dim; j++) {
                    ASSERT_TRUE(fabs(xbPre[fptr + j] - xb[bptr + j]) <= epson);
                }
            } else {
                int ptr = i * dim;
                // check idx
                EXPECT_EQ(idxMapPre[i], idxMap[i]);
                // check vector
                for (int j = 0; j < dim; j++) {
                    ASSERT_TRUE(fabs(xbPre[ptr + j] - xb[ptr + j]) <= epson);
                }
            }
        }
    }
}

TEST(TestAscendIndexFlat, removeBatch)
{
    int dim = 512;
    int ntotal = 200000;
    std::vector<faiss::Index::idx_t> delBatchs = { 1, 23, 50, 10000 };
    const float epson = 1e-3;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexFlatConfig conf({ 0, 1, 2 });
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);

    index.add(ntotal, data.data());
    // define ids
    std::vector<int64_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    // save pre-delete basedata and index of vector
    std::vector<float> xbPre;
    std::vector<faiss::Index::idx_t> idxMapPre;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<float> base(size * dim);
        index.getBase(deviceId, base);
        xbPre.insert(xbPre.end(), base.begin(), base.end());

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

    std::vector<float> xb;
    std::vector<faiss::Index::idx_t> idxMap;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<float> base(size * dim);
        index.getBase(deviceId, base);
        xb.insert(xb.end(), base.begin(), base.end());

        std::vector<faiss::Index::idx_t> idMap(size);
        index.getIdxMap(deviceId, idMap);
        idxMap.insert(idxMap.end(), idMap.begin(), idMap.end());
    }
    EXPECT_EQ(idxMapPre.size(), idxMap.size() + rmedCnt);
    EXPECT_EQ(xbPre.size(), xb.size() + rmedCnt * dim);
    {
        int ntotal0 = ascend::utils::divUp(ntotal, conf.deviceList.size());
        for (size_t i = 0; i < ntotal0 - rmedCnt; i++) {
            if (del.set.find(idxMapPre[i]) != del.set.end()) {
            } else {
                int ptr = i * dim;
                // check ids
                EXPECT_EQ(idxMapPre[i], idxMap[i]);
                // check vector
                for (int j = 0; j < dim; j++) {
                    ASSERT_TRUE(fabs(xbPre[ptr + j] - xb[ptr + j]) <= epson);
                }
            }
        }
    }
}

namespace {
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;
}

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

TEST(TestAscendIndexFlat, QPS)
{
    int dim = 512;
    size_t ntotal = 1000000;
    size_t maxSize = ntotal * dim;

    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
    }

    faiss::ascend::AscendIndexFlatConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, data.data());
    {
        int getTotal = 0;
        for (size_t i = 0; i < conf.deviceList.size(); i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            getTotal += tmpTotal;
        }
        EXPECT_EQ(getTotal, ntotal);
    }

    std::vector<int> searchNum = { 1, 2, 4, 8, 16, 32, 64, 128, 1000 };
    for (size_t n = 0; n < searchNum.size(); n++) {
        int k = 100;
        int loopTimes = 100;
        std::vector<float> dist(searchNum[n] * k, 0);
        std::vector<faiss::Index::idx_t> label(searchNum[n] * k, 0);
        double ts = GetMillisecs();
        for (int i = 0; i < loopTimes; i++) {
            index.search(searchNum[n], data.data(), k, dist.data(), label.data());
        }
        double te = GetMillisecs();
        printf("case[%zu]: base:%zu, dim:%d, search num:%d, QPS:%.4f\n", n, ntotal, dim, searchNum[n],
            1000 * searchNum[n] * loopTimes / (te - ts));
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
