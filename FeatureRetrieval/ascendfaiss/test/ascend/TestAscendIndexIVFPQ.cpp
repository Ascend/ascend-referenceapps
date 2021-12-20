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
#include <faiss/ascend/AscendIndexIVFPQ.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <cstring>
#include <sys/time.h>
#include <faiss/index_io.h>

namespace {
const bool VERBOSE = false;

inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

TEST(TestAscendIndexIVFPQ, All)
{
    int dim = 128; // dim is 128
    int ntotal = 200000;
    int ncentroids = 2048; // nlist is 2048
    int m = 32;  // subquantizer is 32

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexIVFPQConfig conf({ 2 });
    faiss::ascend::AscendIndexIVFPQ index(dim, ncentroids, m, 8, faiss::METRIC_L2, conf);
    index.verbose = VERBOSE;
    index.setNumProbes(64);

    index.train(ntotal, data.data());
    for (int i = 0; i < ncentroids; i++) {
        for (auto deviceId : conf.deviceList) {
            int len = index.getListLength(i, deviceId);
            ASSERT_EQ(len, 0);
        }
    }

    index.add(ntotal, data.data());
    EXPECT_EQ(index.ntotal, ntotal);
    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            for (auto deviceId : conf.deviceList) {
                tmpTotal += index.getListLength(i, deviceId);
            }
        }
        EXPECT_EQ(tmpTotal, ntotal);
    }

    int cnt = 2000;
    int deviceCnt = conf.deviceList.size();
    std::vector<std::vector<std::vector<uint8_t>>> codes(cnt,
        std::vector<std::vector<uint8_t>>(deviceCnt, std::vector<uint8_t>()));
    std::vector<std::vector<std::vector<uint32_t>>> indices(cnt,
        std::vector<std::vector<uint32_t>>(deviceCnt, std::vector<uint32_t>()));
    {
        for (int i = 0; i < cnt; i++) {
            std::vector<uint8_t> code;
            std::vector<uint32_t> indice;
            for (int j = 0; j < deviceCnt; j++) {
                index.getListCodesAndIds(i, conf.deviceList[j], code, indice);
                codes[i][j] = code;
                indices[i][j] = indice;
            }
        }
    }

    index.reset();
    for (int i = 0; i < ncentroids; i++) {
        for (auto deviceId : conf.deviceList) {
            int len = index.getListLength(i, deviceId);
            ASSERT_EQ(len, 0);
        }
    }

    index.add(ntotal, data.data());
    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            for (auto deviceId : conf.deviceList) {
                tmpTotal += index.getListLength(i, deviceId);
            }
        }
        EXPECT_EQ(tmpTotal, ntotal);
    }

    for (int i = 0; i < cnt; i++) {
        for (int j = 0; j < deviceCnt; j++) {
            std::vector<uint8_t> tmpCode;
            std::vector<uint32_t> tmpIndi;
            index.getListCodesAndIds(i, conf.deviceList[j], tmpCode, tmpIndi);
            std::vector<uint8_t> code = codes[i][j];
            std::vector<uint32_t> indi = indices[i][j];
            ASSERT_EQ(tmpCode.size(), tmpIndi.size() * index.getNumSubQuantizers());
            ASSERT_EQ(code.size(), indi.size() * index.getNumSubQuantizers());
            ASSERT_EQ(tmpCode.size(), code.size()) << "failure index:" << i;
            ASSERT_EQ(tmpIndi.size(), indi.size()) << "failure index:" << i;

            ASSERT_EQ(memcmp(tmpCode.data(), code.data(), code.size()), 0);
            ASSERT_EQ(memcmp(tmpIndi.data(), indi.data(), indi.size() * sizeof(uint32_t)), 0);
        }
    }

    {
        int n = 1;
        int k = 1;
        for (int i = 3; i < 10; i++) {
            float dist = 20.0f;
            faiss::Index::idx_t label;
            // search 1 vector @ Top1
            index.search(n, data.data() + i * dim, k, &dist, &label);
            ASSERT_EQ(label, i);

            faiss::Index::idx_t assign;
            index.assign(1, data.data() + i * dim, &assign);
            ASSERT_EQ(assign, i);
        }
    }
}

TEST(TestAscendIndexIVFPQ, CloneAscend2CPU)
{
    int dim = 128;  // dim is 128
    int ntotal = 250000;
    int ncentroids = 2048;  // nlist is 2048
    int nprobe = 64;
    int xbSize = ntotal * 10;
    int m = 32;  // subquantizer is 32

    srand48(0);
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    // ascend index
    faiss::ascend::AscendIndexIVFPQConfig conf({ 0, 2 });
    faiss::ascend::AscendIndexIVFPQ ascendIndex(dim, ncentroids, m, 8, faiss::METRIC_L2, conf);
    ascendIndex.verbose = VERBOSE;
    ascendIndex.setNumProbes(nprobe);

    ascendIndex.train(ntotal, data.data());

    // add ground truth
    ascendIndex.add(ntotal, data.data());

    // add 2250w vector
    for (int i = 0; i < (xbSize / ntotal - 1); i++) {
        std::vector<float> _data(dim * ntotal);
        for (int j = 0; j < dim * ntotal; j++) {
            _data[j] = drand48();
        }
        ascendIndex.add(ntotal, _data.data());
        printf("add %d times of ntotal data.\n", i);
    }

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&ascendIndex);
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./ivfpq2500000.faiss";
    write_index(cpuIndex, outfilename);
    printf("write ascendIndex to file ok!\n");

    int lenall = 0;
    for (int i = 0; i < ncentroids; i++) {
        for (auto deviceId : conf.deviceList) {
            lenall += ascendIndex.getListLength(i, deviceId);
        }
    }
    EXPECT_EQ(lenall, xbSize);

    for (int i = 0; i < 1; i++) {
        int k = 1000;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::Index::idx_t> label(k, 0);
        double ts = GetMillisecs();
        for (int i = 0; i < 500; i++) {
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

TEST(TestAscendIndexIVFPQ, CloneCPU2Ascend)
{
    int dim = 128;  // dim is 128
    int ntotal = 250000;
    int ncentroids = 2048;  // nlist is 2048
    int xbSize = ntotal * 10;

    srand48(0);
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    const char *indexfilename = "./ivfpq2500000.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);

    // ascend index
    std::vector<int> devices = { 0, 2 };
    faiss::ascend::AscendIndexIVFPQ *ascendIndex =
        dynamic_cast<faiss::ascend::AscendIndexIVFPQ *>(faiss::ascend::index_cpu_to_ascend(devices, initIndex));
    ASSERT_FALSE(ascendIndex == nullptr);

    int lenall = 0;
    for (int i = 0; i < ncentroids; i++) {
        for (auto deviceId : devices) {
            lenall += ascendIndex->getListLength(i, deviceId);
        }
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
        // printf("ascend idx %d dist %f %f %f %f %f lable %lu %lu %lu %lu %lu.\n", idx, dist[0],
        // dist[1], dist[2], dist[3], dist[4], label[0], label[1], label[2], label[3], label[4]);
        EXPECT_EQ(label[0], idx);
    }

    delete ascendIndex;
    delete initIndex;
}

TEST(TestAscendIndexIVFPQ, CopyFrom)
{
    int dim = 128;  // dim is 128
    int ntotal = 200000;
    int ncentroids = 2048;  // nlist is 2048
    int m = 32;  // subquantizer is 32
    int nbits = 8;

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    faiss::IndexFlatL2 cpuQuantizer(dim);
    faiss::IndexIVFPQ cpuIndex(&cpuQuantizer, dim, ncentroids, m, nbits);
    cpuIndex.nprobe = 64;
    cpuIndex.verbose = VERBOSE;
    cpuIndex.train(ntotal, data.data());
    cpuIndex.add(ntotal, data.data());

    faiss::ascend::AscendIndexIVFPQConfig conf({ 0, 2 });
    faiss::ascend::AscendIndexIVFPQ index(&cpuIndex, conf);
    index.verbose = VERBOSE;

    EXPECT_EQ(index.d, dim);
    EXPECT_EQ(index.ntotal, ntotal);
    EXPECT_EQ(index.getNumSubQuantizers(), m);
    EXPECT_EQ(index.getBitsPerCode(), nbits);
    EXPECT_EQ(index.getCentroidsPerSubQuantizer(), pow(2, nbits));

    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            int sizeCpuList = cpuIndex.get_list_size(i);
            int sizeAscendList = 0;
            for (auto deviceId : conf.deviceList) {
                sizeAscendList += index.getListLength(i, deviceId);
            }
            ASSERT_EQ(sizeCpuList, sizeAscendList) << "Failure idx:" << i;

            const uint8_t *codeCpu = cpuIndex.invlists->get_codes(i);
            const faiss::Index::idx_t *idCpu = cpuIndex.invlists->get_ids(i);

            std::vector<unsigned char> codes;
            std::vector<uint32_t> indices;
            for (auto deviceId : conf.deviceList) {
                std::vector<uint8_t> code;
                std::vector<uint32_t> ids;
                index.getListCodesAndIds(i, deviceId, code, ids);

                codes.insert(codes.end(), code.begin(), code.end());
                indices.insert(indices.end(), ids.begin(), ids.end());
            }
            std::vector<uint32_t> indicesCpu(sizeCpuList, 0);
            transform(idCpu, idCpu + sizeCpuList, begin(indicesCpu), [](faiss::Index::idx_t x) { return uint32_t(x); });
            ASSERT_EQ(memcmp(codeCpu, codes.data(), codes.size()), 0);
            ASSERT_EQ(memcmp(indicesCpu.data(), indices.data(), indices.size() * sizeof(uint32_t)), 0);

            tmpTotal += sizeAscendList;
        }
        EXPECT_EQ(tmpTotal, ntotal);
    }
}
TEST(TestAscendIndexIVFPQ, CopyTo)
{
    int dim = 128;  // dim is 128
    int ntotal = 100000;
    int ncentroids = 2048;  // nlist is 2048
    int m = 32;  // subquantizer is 32
    int nbits = 8;  // 1 byte is 8 bits

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexIVFPQConfig conf({ 0, 2 });
    faiss::ascend::AscendIndexIVFPQ index(dim, ncentroids, m, nbits, faiss::METRIC_L2, conf);
    index.verbose = VERBOSE;

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());

    faiss::IndexFlatL2 cpuQuantizer(dim);
    faiss::IndexIVFPQ cpuIndex(&cpuQuantizer, dim, ncentroids, m, nbits);

    index.copyTo(&cpuIndex);

    EXPECT_EQ(cpuIndex.ntotal, ntotal);
    EXPECT_EQ(cpuIndex.d, dim);
    EXPECT_EQ(cpuIndex.nlist, ncentroids);
    EXPECT_EQ(cpuIndex.nprobe, index.getNumProbes());
    EXPECT_EQ(cpuIndex.pq.M, m);
    EXPECT_EQ(cpuIndex.pq.nbits, nbits);

    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            int sizeCpuList = cpuIndex.get_list_size(i);
            int sizeAscendList = 0;
            for (auto deviceId : conf.deviceList) {
                sizeAscendList += index.getListLength(i, deviceId);
            }
            ASSERT_EQ(sizeCpuList, sizeAscendList) << "Failure idx:" << i;

            const uint8_t *codeCpu = cpuIndex.invlists->get_codes(i);
            const faiss::Index::idx_t *idCpu = cpuIndex.invlists->get_ids(i);

            std::vector<unsigned char> codes;
            std::vector<uint32_t> indices;
            for (auto deviceId : conf.deviceList) {
                std::vector<uint8_t> code;
                std::vector<uint32_t> ids;
                index.getListCodesAndIds(i, deviceId, code, ids);

                codes.insert(codes.end(), code.begin(), code.end());
                indices.insert(indices.end(), ids.begin(), ids.end());
            }
            std::vector<uint32_t> indicesCpu(sizeCpuList, 0);
            transform(idCpu, idCpu + sizeCpuList, begin(indicesCpu), [](faiss::Index::idx_t x) { return uint32_t(x); });

            ASSERT_EQ(memcmp(codeCpu, codes.data(), codes.size()), 0);
            ASSERT_EQ(memcmp(indicesCpu.data(), indices.data(), indices.size() * sizeof(uint32_t)), 0);

            tmpTotal += sizeCpuList;
        }
        EXPECT_EQ(tmpTotal, ntotal);
    }
}

TEST(TestAscendIndexIVFPQ, remove)
{
    int dim = 128;  // dim is 128
    int ntotal = 200000;
    int ncentroids = 2048;  // nlist is 2048
    int m = 32;  // subquantizer is 32
    int nbits = 8;  // 1 byte is 8 bits

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexIVFPQConfig conf({ 0, 2 });
    faiss::ascend::AscendIndexIVFPQ index(dim, ncentroids, m, nbits, faiss::METRIC_L2, conf);
    index.verbose = VERBOSE;

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());

    faiss::IDSelectorRange del(0, 2);
    int rmCnt = 0;
    for (int i = 0; i < ncentroids; i++) {
        for (auto deviceId : conf.deviceList) {
            std::vector<uint8_t> code;
            std::vector<uint32_t> ids;

            index.getListCodesAndIds(i, deviceId, code, ids);
            for (size_t k = 0; k < ids.size(); k++) {
                rmCnt += del.is_member((int64_t)ids[k]) ? 1 : 0;
            }
        }
    }

    size_t rmedCnt = index.remove_ids(del);
    ASSERT_EQ(rmedCnt, rmCnt);
    ASSERT_EQ(index.ntotal, (ntotal - rmedCnt));

    int tmpTotal = 0;
    for (int i = 0; i < ncentroids; i++) {
        for (auto deviceId : conf.deviceList) {
            std::vector<uint8_t> code;
            std::vector<uint32_t> ids;
            index.getListCodesAndIds(i, deviceId, code, ids);
            ASSERT_EQ(code.size(), ids.size() * index.getNumSubQuantizers());
            for (size_t k = 0; k < ids.size(); k++) {
                ASSERT_FALSE(del.is_member((int64_t)ids[k])) << "failure index:" << i;
            }
            tmpTotal += index.getListLength(i, deviceId);
        }
    }
    printf("remove end\n");
    EXPECT_EQ(tmpTotal, (ntotal - rmedCnt));
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
