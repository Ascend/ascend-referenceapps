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
#include <atomic>
#include <ascenddaemon/impl/IndexIVFFlat.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/Limits.h>
#include <ascenddaemon/utils/TaskQueueItem.h>

namespace ascend {
namespace {
const int EVENT_SIZE = 500;        // because the max of event count is 1024, wo should retain some for another use
const int SEARCH_LIST_SIZE = 2048; // must be CUBE_ALIGN_SIZE aligned
const int SEARCH_SHAPED_SIZE = SEARCH_LIST_SIZE / CUBE_ALIGN_SIZE;
const int TIMEOUT_CHECK_TICK = 5120;
const double TIMEOUT_MS = 50000;
const int SIZE_OF_FP16 = 2;
const int THREADS_CNT = 4;
}

IndexIVFFlat::IndexIVFFlat(int numList, int dim, int nprobes, int resourceSize)
    : IndexIVF(numList, SIZE_OF_FP16 * dim, dim, nprobes, resourceSize), threadPool(nullptr)
{
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN_SIZE == 0);
    isTrained = false;

    // supported batch size
    searchPageSizes = { 4, 2, 1 };

    threadPool = new AscendThreadPool(THREADS_CNT);

    for (int i = 0; i < numLists; ++i) {
        preComputeData.emplace_back(std::make_unique<DeviceVector<float16_t>>(MemorySpace::DEVICE_HUGEPAGE));
    }

    resetDistCompOperator(numList);
    resetDistCompOp(SEARCH_LIST_SIZE);
}

IndexIVFFlat::~IndexIVFFlat()
{
    if (threadPool) {
        delete threadPool;
    }
}

void IndexIVFFlat::reset()
{
    // reset the database and precomputed, but trained values is maintained.
    IndexIVF::reset();

    preComputeData.clear();
    for (int i = 0; i < numLists; ++i) {
        preComputeData.push_back(std::make_unique<DeviceVector<float16_t>>(MemorySpace::DEVICE_HUGEPAGE));
    }
}

void IndexIVFFlat::reserveMemory(size_t numVecs)
{
    size_t numVecsPerList = utils::divUp(numVecs, static_cast<size_t>(this->numLists));
    if (numVecsPerList < 1) {
        return;
    }

    numVecsPerList = utils::roundUp(numVecsPerList, static_cast<size_t>(CUBE_ALIGN_SIZE));
    size_t tmpLen = numVecsPerList * static_cast<size_t>(numLists);
    IndexIVF::reserveMemory(tmpLen);

    for (auto &list : preComputeData) {
        list->reserve(numVecsPerList);
    }
}

void IndexIVFFlat::reserveMemory(int listId, size_t numVecs)
{
    if (numVecs < 1) {
        return;
    }

    numVecs = utils::roundUp(numVecs, static_cast<size_t>(CUBE_ALIGN_SIZE));
    IndexIVF::reserveMemory(listId, numVecs);

    ASCEND_THROW_IF_NOT((listId < this->numLists) && (listId >= 0));
    preComputeData[listId]->reserve(numVecs);
}

size_t IndexIVFFlat::reclaimMemory()
{
    size_t totalReclaimed = IndexIVF::reclaimMemory();

    for (auto &list : preComputeData) {
        totalReclaimed += list->reclaim(true);
    }

    return totalReclaimed;
}

size_t IndexIVFFlat::reclaimMemory(int listId)
{
    ASCEND_THROW_IF_NOT((listId < this->numLists) && (listId >= 0));

    size_t totalReclaimed = IndexIVF::reclaimMemory(listId);

    totalReclaimed += preComputeData[listId]->reclaim(true);

    return totalReclaimed;
}

void IndexIVFFlat::addVectors(int listId, size_t numVecs, const float16_t *raw, const uint32_t *indices)
{
    ASCEND_THROW_IF_NOT(this->isTrained);
    ASCEND_THROW_IF_NOT(listId >= 0 && listId < numLists);

    // rawData need to be Zz format because of DistanceCompute operator's limitation.
    //       origin rawData for example (shape n X dim). n=16, dim = 128
    //       |  0_0  0_1  0_2  0_3 ...  0_125  0_126  0_127 |
    //       |  1_0  1_1  1_2  1_3 ...  1_125  1_126  1_127 |
    //       |        .                          .          |
    //       |        .                          .          |
    //       | 14_0 14_1 14_2 14_3 ... 14_125 14_126 14_127 |
    //       | 15_0 15_1 15_2 15_3 ... 15_125 15_126 15_127 |
    //                              | shape dims 2: (dim/16 X n/16) X (16 X 16),
    //       after Zz format?dims4 (n/16) X (dim/16) X 16 X 16
    //       |   0_0   0_1 ...  0_14  0_15   1_0   1_1 ...  1_15 ...   15_15 |
    //       |  0_16  0_17 ...  0_30  0_31  1_16  1_17 ...  1_31 ...   15_31 |
    //       |        .                    .                  .         .    |
    //       |        .                    .                  .         .    |
    //       |  0_96  0_97 ... 0_110 0_111  1_96  1_97 ... 1_111 ...  15_111 |
    //       | 0_112 0_113 ... 0_126 0_127 1_112 1_113 ... 1_127 ...  15_127 |
    // n and dim must be 16 aligned, otherwise padding data is needed.

    AscendTensor<float16_t, DIMS_2> rawData(const_cast<float16_t *>(raw), { static_cast<int>(numVecs), this->dims });
    size_t originLen = getListLength(listId);
    size_t tmpLen = utils::roundUp((originLen + numVecs), static_cast<size_t>(SEARCH_LIST_SIZE));

    // float16_t need two uint_8 to store
    this->deviceListData[listId]->resize(tmpLen * (SIZE_OF_FP16 * this->dims));
    this->preComputeData[listId]->resize(tmpLen);

    // dims is alignd with CUBE_ALIGN_SIZE, no padding data in horizontal direction
    int dimShaped = utils::divUp(this->dims, CUBE_ALIGN_SIZE);

// input rawData are contigous(numVecs X dims), reconstruct the rawData into Zz format.
#pragma omp parallel for if (numVecs >= 100)
    for (size_t i = 0; i < numVecs; ++i) {
        int seq = static_cast<int>(originLen + i);
        float16_t *tmpData = reinterpret_cast<float16_t *>(deviceListData[listId]->data()) + getShapedDataOffset(seq);

        // the type of deviceListData
        for (int j = 0; j < dimShaped; j++) {
            MEMCPY_S(tmpData, CUBE_ALIGN_SIZE * sizeof(float16_t), rawData[i][j * CUBE_ALIGN_SIZE].data(),
                CUBE_ALIGN_SIZE * sizeof(float16_t));
            tmpData += (CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE);
        }
    }

    float16_t *precompData = preComputeData[listId]->data() + originLen;
    fvecNormsL2sqr(precompData, raw, this->dims, numVecs);
    deviceListIndices[listId]->append(indices, numVecs);

    maxListLength = std::max(maxListLength, static_cast<int>(getListLength(listId)));
    maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN_SIZE);
    this->ntotal += numVecs;
}

int IndexIVFFlat::getShapedDataOffset(int idx) const
{
    int offset = this->dims * utils::roundDown(idx, CUBE_ALIGN_SIZE);
    offset += (idx % CUBE_ALIGN_SIZE) * CUBE_ALIGN_SIZE;
    return offset;
}

void IndexIVFFlat::addImpl(int n, const float16_t *x, const idx_t *ids)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);
}

void IndexIVFFlat::updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2> &coarseCentroidsData)
{
    // update coarse centroids for L1 search.
    IndexIVF::updateCoarseCentroidsData(coarseCentroidsData);
    this->isTrained = true;
}

void IndexIVFFlat::searchImplL1(AscendTensor<float16_t, DIMS_2> &queries, AscendTensor<uint32_t, DIMS_2> &result,
    aclrtStream stream)
{
    int n = queries.getSize(0);
    auto &mem = resources.getMemoryManager();
    AscendTensor<float16_t, DIMS_2> distances(mem, { n, this->numLists }, stream);
    AscendTensor<float16_t, DIMS_2> kDistance(mem, { n, this->nprobe }, stream);
    AscendTensor<uint32_t, DIMS_2> indices;

    AscendTensor<uint16_t, DIMS_1> opFlag(mem, { FLAG_ALIGN }, stream);
    opFlag.zero();

    // init results to invalid data.
    kDistance.initValue(Limits<float16_t>::getMax());
    result.initValue(std::numeric_limits<uint32_t>::max());

    // run l1 distance calculation
    runDistanceCompute(queries, this->coarseCentroidsShaped, this->normCoarseCentroids, distances, opFlag, stream);
    uint16_t *volatile flagPtr1 = opFlag.data();
    uint16_t *volatile flagPtr2 = opFlag.data() + FLAG_ALIGN_OFFSET;
    WAITING_FLAG_READY((*flagPtr1) && (*flagPtr2), TIMEOUT_CHECK_TICK, TIMEOUT_MS);

    // get the kNN results, k = nprobe. reorder is not needed
    // because L2 search do not concern which list is the nearest.
    ASCEND_THROW_IF_NOT(topkOp.exec(distances, indices, kDistance, result));
}

void IndexIVFFlat::searchImplL2(AscendTensor<float16_t, DIMS_2> &queries, AscendTensor<uint32_t, DIMS_2> &l1Indices,
    AscendTensor<float16_t, DIMS_2> &outDistances, AscendTensor<uint32_t, DIMS_2> &outIndices)
{
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);
    int maxScanSeg = utils::divUp(maxListLength, SEARCH_LIST_SIZE);

    // tensor for operator outputing L2 distance
    int resultSize = utils::roundUp(maxListLength, SEARCH_LIST_SIZE);
    AscendTensor<float16_t, DIMS_3> distResult(mem, { n, nprobe, resultSize }, stream);

    // tensor for operator flags
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { n, nprobe, (maxScanSeg * FLAG_ALIGN) }, stream);
    (void)opFlag.zero();

    bool errorQuit = false;
    std::vector<std::vector<QueueItem>> topkQueue(n, std::vector<QueueItem>((maxScanSeg * nprobe)));
    std::vector<std::future<void>> topkFunctorRet;
    std::vector<std::pair<volatile bool, volatile int>> executeInfo(n, { false, 0 });

    auto topkFunctor = [&](int idx) {
        ASCEND_ASSERT(idx < n);
        // bind thread to fixed cpus for stablity time costing,
        // bind from cpu 0 to cpu 3, cpu 4-5 is for main thread.
        if (idx < THREADS_CNT) {
            AscendUtils::attachToCpu(idx);
        }

        DeviceScope device;
        auto outDistance = outDistances[idx].view();
        auto outIndice = outIndices[idx].view();
        int iter = 0;
        while (!errorQuit && (!executeInfo[idx].first || iter < executeInfo[idx].second)) {
            while (iter < executeInfo[idx].second) {
                auto &item = topkQueue[idx][iter];

                // waitting for the item operator to be added to the stream to run
                WAITING_FLAG_READY((item.IsExecuting()), TIMEOUT_CHECK_TICK, TIMEOUT_MS);

                AscendTensor<float16_t, DIMS_1> distance(item.distPtr, { item.size });
                AscendTensor<uint32_t, DIMS_1> ids(item.idPtr, { item.size });

                // waitting for the operator finishing running, while the event will be
                // produce when finishing running.
                WAITING_FLAG_READY((*item.flagPtr && *(item.flagPtrSec)), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
                ASCEND_THROW_IF_NOT(topkOp.exec(distance, ids, outDistance, outIndice));
                ++iter;
            }
        }

        // reorder the results in distance's ascending order
        if (!errorQuit) {
            topkOp.reorder(outDistance, outIndice);
        }
    };

    for (int i = 0; i < n; ++i) {
        // add topkFunctor task to threadpool for async executing
        topkFunctorRet.emplace_back(threadPool->Enqueue(topkFunctor, i));
    }

    // functor for waitting topk functor ready
    auto waitTopkReady = [&]() {
        try {
            for (auto &ret : topkFunctorRet) {
                ret.get();
            }
        } catch (std::exception &e) {
            errorQuit = true;
            ASCEND_THROW_MSG(e.what());
        }
    };

    // divid n to several batch(batch size is THREADS_CNT) to query, because
    // threadpool have THREADS_CNT threads, THREADS_CNT is the maximum paralles.
    int tSize = utils::divUp(n, THREADS_CNT);
    for (int t = 0; t < tSize; t++) {
        int tStart = t * THREADS_CNT;
        int tEnd = (tStart + THREADS_CNT) >= n ? n : (tStart + THREADS_CNT);

        // put nprobe in outer loop, while n in inner loop, in order for the balance
        // of every query added to stream for op executing.
        for (int j = 0; j < nprobe; ++j) {
            for (int i = tStart; i < tEnd; ++i) {
                AscendTensor<float16_t, DIMS_2> query(queries[i].data(), { 1, dims });
                int list = l1Indices[i][j].value();

                // if throw exception, errorQuit flag needs to be setted to true, and waitting topk functors quit
                ASCEND_THROW_IF_NOT_CODE(list >= 0 && list < this->numLists, {
                    errorQuit = true;
                    waitTopkReady();
                });

                // seperator list's data for several segs to run L2 distance,
                // because of fixed-shape limitation of aicore's operator.
                int segs = utils::divUp(this->deviceListIndices[list]->size(), SEARCH_LIST_SIZE);
                for (int m = 0; m < segs; ++m) {
                    int offset = m * SEARCH_LIST_SIZE;
                    uint32_t size = std::min(static_cast<uint32_t>(SEARCH_LIST_SIZE),
                        static_cast<uint32_t>((this->deviceListIndices[list]->size() - offset)));

                    // data is stored in Zz format, Zz format is 4 dims shaped. z's shape is
                    // 16 X 16(AiCore cube's matrix operation's size). Z's shape is (SEARCH_LIST_SIZE / 16) X (dims /
                    // Zz's 4 dims shape is ((SEARCH_LIST_SIZE / 16), (dims / 16), 16, 16)
                    AscendTensor<float16_t, DIMS_4> baseshaped(
                        reinterpret_cast<float16_t *>(this->deviceListData[list]->data()) + dims * offset,
                        { SEARCH_SHAPED_SIZE, this->dims / CUBE_ALIGN_SIZE, CUBE_ALIGN_SIZE, CUBE_ALIGN_SIZE });
                    AscendTensor<float16_t, DIMS_1> precomp(preComputeData[list]->data() + offset,
                        { SEARCH_LIST_SIZE });
                    AscendTensor<float16_t, DIMS_2> result(distResult[i][j][offset].data(), { 1, SEARCH_LIST_SIZE });
                    AscendTensor<uint16_t, DIMS_1> flag(opFlag[i][j][m * FLAG_ALIGN].data(), {FLAG_ALIGN});

                    runDistCompute(query, baseshaped, precomp, result, flag, stream);
                    topkQueue[i][executeInfo[i].second].SetExecuting(result.data(),
                        this->deviceListIndices[list]->data() + offset, flag.data(), size);
                    executeInfo[i].second++;
                }
            }
        }

        // set quit flags to true, the flags will be used by topk
        // functor to check whether all operators have been added to the stream.
        for (int i = tStart; i < tEnd; i++) {
            executeInfo[i].first = true;
        }
    }

    // waiting for topk functor to finish
    waitTopkReady();
}

void IndexIVFFlat::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
{
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, dims });
    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<uint32_t, DIMS_2> outIndices(labels, { n, k });

    // init results to invalid data.
    outDistances.initValue(Limits<float16_t>::getMax());
    outIndices.initValue(std::numeric_limits<uint32_t>::max());

    // for performance improving, bind the main thread to cpu4-5,
    // and bind the threadpool to cpu0-cpu3. when n == 1, attach
    // main thread to one cpu(cpu5) is better than multicpus.
    if (n > 1) {
        AscendUtils::attachToCpus({ 4, 5 });
    } else {
        AscendUtils::attachToCpus({ 5 });
    }

    // L1 search, to find nprobe IVF list
    AscendTensor<uint32_t, DIMS_2> l1result(mem, { n, nprobe }, stream);
    searchImplL1(queries, l1result, stream);

    // L2 search, search base in nprobe IVF list to find topk results
    searchImplL2(queries, l1result, outDistances, outIndices);
}

size_t IndexIVFFlat::removeIdsImpl(const IDSelector &sel)
{
    //
    // | id0 id1 id2 ... id98 id99 id100 |
    //            ^                  |
    //            |__________________|
    // if id2 is to be removed, copy the last id(id100) to
    // the mem place of id2, and decrease the list size to size-1.
    size_t removeCnt = 0;

    // dims is alignd with CUBE_ALIGN_SIZE, no padding data in horizontal direction
    int dimShaped = utils::divUp(this->dims, CUBE_ALIGN_SIZE);

#pragma omp parallel for reduction(+ : removeCnt)
    for (int id = 0; id < numLists; ++id) {
        DeviceScope device;
        auto &indicesList = this->deviceListIndices[id];
        auto &baseList = this->deviceListData[id];
        auto &precompList = this->preComputeData[id];

        uint32_t *indicesPtr = indicesList->data();
        float16_t *basePtr = reinterpret_cast<float16_t *>(baseList->data());
        float16_t *precompPtr = precompList->data();
        bool hasMoved = false;
        int j = indicesList->size() - 1;

        for (int i = 0; i <= j;) {
            if (sel.is_member((*indicesList)[i])) {
                MEMCPY_S(indicesPtr + i, sizeof(uint32_t), indicesPtr + j, sizeof(uint32_t));
                MEMCPY_S(precompPtr + i, sizeof(float16_t), precompPtr + j, sizeof(float16_t));

                float16_t *src = basePtr + getShapedDataOffset(j);
                float16_t *dst = basePtr + getShapedDataOffset(i);
                for (int k = 0; k < dimShaped; k++) {
                    MEMCPY_S(dst, CUBE_ALIGN_SIZE * sizeof(float16_t), src, CUBE_ALIGN_SIZE * sizeof(float16_t));
                    src += (CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE);
                    dst += (CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE);
                }

                j--;
                removeCnt++;
                hasMoved = true;
            } else {
                i++;
            }
        }

        // if some code has been removed, list need to be resize and reclaim memory
        if (hasMoved) {
            size_t tmpLen = utils::roundUp((j + 1), CUBE_ALIGN_SIZE);
            indicesList->resize(j + 1);
            baseList->resize(tmpLen * this->dims);
            precompList->resize(tmpLen);
        }
    }

    this->ntotal -= removeCnt;
    return removeCnt;
}

DeviceVector<float16_t> &IndexIVFFlat::getListPrecompute(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return *preComputeData[listId];
}

bool IndexIVFFlat::listVectorsNeedReshaped() const
{
    return true;
}

void IndexIVFFlat::getListVectorsReshaped(int listId, std::vector<unsigned char> &reshaped) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    size_t size = getListLength(listId);
    auto &data = getListVectors(listId);
    int dimShaped = utils::divUp(this->dims, CUBE_ALIGN_SIZE);
    reshaped.resize(SIZE_OF_FP16 * this->dims * size);

// reshape base from Zz format data to contigous format.
#pragma omp parallel for if (size >= 100)
    for (size_t i = 0; i < size; ++i) {
        int offset = getShapedDataOffset(i);
        auto srcPtr = reinterpret_cast<float16_t *>(data.data()) + offset;
        auto dstPtr = reinterpret_cast<float16_t *>(reshaped.data()) + i * this->dims;
        for (int j = 0; j < dimShaped; ++j) {
            MEMCPY_S(dstPtr + j * CUBE_ALIGN_SIZE, CUBE_ALIGN_SIZE * sizeof(float16_t),
                srcPtr + j * CUBE_ALIGN_SIZE * CUBE_ALIGN_SIZE, CUBE_ALIGN_SIZE * sizeof(float16_t));
        }
    }
}

void IndexIVFFlat::runDistCompute(AscendTensor<float16_t, DIMS_2> &queryVecs,
    AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<float16_t, DIMS_1> &norms,
    AscendTensor<float16_t, DIMS_2> &outDistances, AscendTensor<uint16_t, DIMS_1>& flag,
    aclrtStream stream)
{
    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(queryVecs.data(), queryVecs.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(shapedData.data(), shapedData.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(norms.data(), norms.getSizeInBytes()));

    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    distCompOp->exec(distOpInput, distOpOutput, stream);

    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();

    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
}

void IndexIVFFlat::resetDistCompOp(int numLists)
{
    AscendOpDesc desc("DistanceComputeFlat");
    std::vector<int64_t> queryShape({ 1, dims });
    std::vector<int64_t> coarseCentroidsShape({ utils::divUp(numLists, CUBE_ALIGN_SIZE),
        utils::divUp(dims, CUBE_ALIGN_SIZE), CUBE_ALIGN_SIZE, (int64_t)CUBE_ALIGN_SIZE });
    std::vector<int64_t> preNormsShape({ numLists });
    std::vector<int64_t> distResultShape({ 1, numLists });
    std::vector<int64_t> flagShape({ FLAG_ALIGN });
    desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, preNormsShape.size(), preNormsShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    distCompOp = std::move(std::make_unique<AscendOperator>(desc));
}
} // namespace ascend