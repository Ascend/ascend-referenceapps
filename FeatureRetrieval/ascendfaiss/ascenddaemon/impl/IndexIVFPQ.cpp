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

#include <ascenddaemon/impl/IndexIVFPQ.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/Limits.h>
#include <limits>
#include <mutex>
#include <queue>

namespace ascend {
namespace {
const double THREADS_TIMEOUT_MS = 50000;
const int THREADS_TIMEOUT_CHECK_TICK = 5120;
const int ACCUM_OP_FLAG_ALIGN_SIZE = 16;
const int SEARCH_LIST_SIZE = 2048;
const int DISTANCE_TABLE_SIZE = 8;
const int THREADS_FOR_MATRIX_TABLE = 5;
const double TIMEOUT_MS = 50000;
const int TIMEOUT_CHECK_TICK = 5120;
const int FLAG_ALIGN_SIZE = 32;
const int FLAG_ALIGN_OFFSET = 16; // core 0 use first 16 flag, and core 1 use the second 16 flag.
}

struct MatrixQueueDesc {
    MatrixQueueDesc(float16_t* matrix, float16_t* dist, uint32_t* id, uint16_t* flag, int s)
        : matrixPtr(matrix),
          distPtr(dist),
          idPtr(id),
          flagPtr(flag),
          size(s)
    {
    }

    float16_t *matrixPtr;
    float16_t *distPtr;
    uint32_t *idPtr;
    uint16_t* volatile flagPtr;
    int size;
};

struct TopkQueueDesc {
    TopkQueueDesc(float16_t* dist, uint32_t* id, uint16_t* flag, int s)
        : distPtr(dist),
          idPtr(id),
          flagPtr(flag),
          size(s)
    {
    }

    TopkQueueDesc(TopkQueueDesc &&) = default;
    TopkQueueDesc& operator=(TopkQueueDesc &&) = default;

    float16_t *distPtr;
    uint32_t *idPtr;
    uint16_t* volatile flagPtr;
    int size;
};

struct IdDesc {
    IdDesc(int i, int l)
        : idx(i), listId(l)
    {
    }

    int idx;
    int listId;
};

IndexIVFPQ::IndexIVFPQ(int numList, int dim,
                       int subQuantizers, int bitCntPerSubQuantizer, int nprobes, int resourceSize)
    : IndexIVF(numList, subQuantizers, dim, nprobes, resourceSize),
      numSubQuantizers(subQuantizers),
      numSubQuantizersAligned(utils::roundUp(subQuantizers, CUBE_ALIGN_SIZE)),
      bitsPerSubQuantizer(bitCntPerSubQuantizer),
      numSubQuantizerCodes(utils::pow2(bitCntPerSubQuantizer)),
      dimPerSubQuantizer(dim / subQuantizers),
      distTableBuildOp(nullptr),
      distAccumOp(nullptr),
      threadPool(THREADS_FOR_MATRIX_TABLE + 1)
{
    ASCEND_THROW_IF_NOT(dims % numSubQuantizers == 0);
    isTrained = false;
	
    // supported batch size
    searchPageSizes = {16, 8, 4, 2, 1};

    resetDistCompOperator(numList);
    resetDistTableOperator();
    resetDistAccumOperator();
}

IndexIVFPQ::~IndexIVFPQ() {}

void IndexIVFPQ::addImpl(int n, const float16_t* x, const idx_t* ids)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);
}

void IndexIVFPQ::addVectors(int listId, const uint8_t* codes,
                            const uint32_t* indices, size_t numVecs)
{
    ASCEND_THROW_IF_NOT(listId >= 0 && listId < numLists);

    if (numVecs == 0) {
        return;
    }

    size_t codeLenInBytes = numVecs * bytesPerVector;
    deviceListData[listId]->append((unsigned char*)codes,
                                   codeLenInBytes, true);
    deviceListIndices[listId]->append(indices, numVecs, true);

    maxListLength = std::max(maxListLength, static_cast<int>(deviceListIndices[listId]->size()));
    this->ntotal += numVecs;
}

void IndexIVFPQ::updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2>& coarseCentroidsData)
{
    IndexIVF::updateCoarseCentroidsData(coarseCentroidsData);
    
    // if pqCentroids has been updated, set isTrained = true
    if (this->pqCentroids.data()) {
        this->isTrained = true;
    }
}

void IndexIVFPQ::updatePQCentroidsData(AscendTensor<float16_t, DIMS_2>& pqCentroidsData)
{
    int pqM = pqCentroidsData.getSize(0);
    int pqkdsub = pqCentroidsData.getSize(1);
    ASCEND_THROW_IF_NOT_FMT(pqM == numSubQuantizers && 
        pqkdsub == (dimPerSubQuantizer * numSubQuantizerCodes), 
        "pq centroids data's shape invalid.(%d X %d) vs (%d X %d)", 
        pqM, pqkdsub, numSubQuantizers, (dimPerSubQuantizer * numSubQuantizerCodes));

    AscendTensor<float16_t, DIMS_2> devicePQCentroids(
        { pqCentroidsData.getSize(0), pqCentroidsData.getSize(1) });
    devicePQCentroids.copyFromSync(pqCentroidsData);
    pqCentroids = std::move(devicePQCentroids);

    // if coarseCentroids has been updated, set isTrained = true
    if (this->coarseCentroids.data()) {
        this->isTrained = true;
    }
}

void IndexIVFPQ::searchImpl(AscendTensor<float16_t, DIMS_2>& queries, int k,
                            AscendTensor<float16_t, DIMS_2>& outDistances,
                            AscendTensor<uint32_t, DIMS_2>& outIndices)
{
    auto stream = resources.getDefaultStream();
    auto& mem = resources.getMemoryManager();
    int n = queries.getSize(0);

    AscendTensor<float16_t, DIMS_2> l1Distances(mem, {n, numLists}, stream);
    AscendTensor<float16_t, DIMS_2> l1TopkDistance(mem, {n, nprobe}, stream);
    AscendTensor<uint32_t, DIMS_2> l1TopkIndices(mem, {n, nprobe}, stream);
    AscendTensor<uint32_t, DIMS_2> indices;
    AscendTensor<uint16_t, DIMS_1> opFlag(mem, { FLAG_ALIGN_SIZE }, stream);
    opFlag.zero();
    l1TopkDistance.initValue(Limits<float16_t>::getMax());
    l1TopkIndices.initValue(std::numeric_limits<uint32_t>::max());

    // run L1 IVF search, get top-nprobe result list for L2 search
    runDistanceCompute(queries, coarseCentroidsShaped,
                       normCoarseCentroids, l1Distances, opFlag, stream);
    uint16_t *volatile flagPtr1 = opFlag.data();
    uint16_t *volatile flagPtr2 = opFlag.data() + FLAG_ALIGN_OFFSET;
    WAITING_FLAG_READY((*flagPtr1) && (*flagPtr2), TIMEOUT_CHECK_TICK, TIMEOUT_MS);

    topkOp.exec(l1Distances, indices, l1TopkDistance, l1TopkIndices);

    // run L2 search, L2 search has four steps:
    // distance table build��distance matrix build��
    // accumulate distance��topk
    for (int i = 0; i < n; i++) {
        auto outDistance = outDistances[i].view();
        auto outIndice = outIndices[i].view();
        std::queue<TopkQueueDesc> topkQueue;
        std::queue<MatrixQueueDesc> matrixQueue;
        std::vector<std::future<void>> matrixTableRet;
        std::mutex topkQueueLock;
        std::mutex matrixQueueLock;
        std::condition_variable topkCondition;
        std::condition_variable matrixCondition;
        bool matrixThreadDone = false;
        bool distAccumThreadDone = false;
        bool errorQuit = false;
        int tables = utils::divUp(nprobe, DISTANCE_TABLE_SIZE);

        AscendTensor<float16_t, DIMS_2> query(queries[i].data(), { 1, dims });
        AscendTensor<float16_t, DIMS_5> distTables(mem, { tables, 1, DISTANCE_TABLE_SIZE, 
            numSubQuantizers, numSubQuantizerCodes }, stream);
        AscendTensor<int32_t, DIMS_3> listNos(mem, {tables, 1, DISTANCE_TABLE_SIZE}, stream);
        listNos.initValue(-1);

        // init lists for scan, threadIdx:tableIdx:listIdx
        std::vector<std::vector<std::vector<IdDesc>>> scansList(THREADS_FOR_MATRIX_TABLE,
            std::vector<std::vector<IdDesc>>(tables, std::vector<IdDesc>()));
        int maxTableSize = 0;
        int maxListScanLen = 0;
        
        {
            // divide all distance tables idx into tables
            int idx = 0;
            std::vector<int> idxs(tables, 0);
            for (int j = 0; j < nprobe; j++) {
                int idIdx = l1TopkIndices[i][j].value();
                ASCEND_ASSERT(idIdx < static_cast<int>(deviceListIndices.size()));
                
                if (deviceListIndices[idIdx]->size()) {
                    listNos[idx][0][idxs[idx]] = idIdx;
                    idxs[idx]++;
                    idx = (idx + 1) % tables;
                    maxListScanLen = std::max(maxListScanLen, 
                        static_cast<int>(utils::roundUp(deviceListIndices[idIdx]->size(), SEARCH_LIST_SIZE)));
                }
            }
            maxTableSize = idxs[0];

            // divide all list idx into thread scans
            int threadIdx = 0;
            for (int j = 0; j < tables; j++) {
                int size = idxs[j];
                for (int m = 0; m < size; m++) {
                    scansList[threadIdx][j].emplace_back(IdDesc(m, listNos[j][0][m].value()));
                    threadIdx = (threadIdx + 1) % THREADS_FOR_MATRIX_TABLE;
                }
            }
        }

        AscendTensor<float16_t, DIMS_4> distMatrix(mem, { tables, maxTableSize, 
            maxListScanLen, numSubQuantizersAligned }, stream);
        AscendTensor<float16_t, DIMS_3> distAccumResult(mem, 
            { tables, maxTableSize, maxListScanLen }, stream);
        AscendTensor<uint16_t, DIMS_3> distAddFlag(mem, { tables, maxTableSize, 
            (maxListScanLen / SEARCH_LIST_SIZE * ACCUM_OP_FLAG_ALIGN_SIZE) }, stream);
        (void)distAddFlag.zero();
        AscendTensor<uint16_t, DIMS_2> distTableFlag(mem, {tables, ACCUM_OP_FLAG_ALIGN_SIZE}, stream);
        (void)distTableFlag.zero();
        
        // only when numSubQuantizersAligned not equal numSubQuantizers, needs to zero distance matrix.
        // because there are some pending mem which need to be zero.
        if (numSubQuantizers != numSubQuantizersAligned) {
            (void)distMatrix.zero();
        }

        // functor execute for matrix table build
        auto disMatrixFunctor = [&] (int bindCpu, int idx) {
            AscendUtils::attachToCpu(bindCpu);
            for (int j = 0; j < tables; j++) {
                uint16_t* volatile flagPtr = distTableFlag[j].data();
                // wait for distance table for finish
                int waitTicks = 0;
                double startWait = utils::getMillisecs();
                while (!(*flagPtr)) {
                    waitTicks++;
                    if (!(waitTicks % THREADS_TIMEOUT_CHECK_TICK) &&
                        (utils::getMillisecs() - startWait) >= THREADS_TIMEOUT_MS) {
                        ASCEND_THROW_MSG("matrix table functor wait timeout.");
                    }
                }
                
                // scan for lists from start listid to end listid
                auto scans = scansList[idx][j];
                for (size_t l = 0; l < scans.size(); l++) {
                    int listId = scans[l].listId;
                    int tmpIdx = scans[l].idx;
                    if (listId < 0 || listId >= numLists) {
                        continue;
                    }
                    
                    auto table = distTables[j][0][tmpIdx].view();
                    // seperate every list into fixed len codes to scan
                    int segs = utils::divUp(deviceListIndices[listId]->size(), SEARCH_LIST_SIZE);
                    for (int m = 0; m < segs; m++) {
                        int offset = m * SEARCH_LIST_SIZE;
                        int offsetFlag = m * ACCUM_OP_FLAG_ALIGN_SIZE;
                        int size = std::min(static_cast<int>(SEARCH_LIST_SIZE), 
                            static_cast<int>((deviceListIndices[listId]->size() - offset)));
                        AscendTensor<unsigned char, DIMS_2> code(deviceListData[listId]->data() 
                            + numSubQuantizers * offset, {size, numSubQuantizers});
                        AscendTensor<float16_t, DIMS_2> matrix(distMatrix[j][tmpIdx][offset].data(), 
                            { SEARCH_LIST_SIZE, numSubQuantizersAligned });
                        ASCEND_THROW_IF_NOT(distMatrixOp.exec(code, table, matrix));

                        std::unique_lock<std::mutex> lock(matrixQueueLock);
                        matrixQueue.emplace(MatrixQueueDesc(matrix.data(), 
                            distAccumResult[j][tmpIdx][offset].data(), deviceListIndices[listId]->data() + offset, 
                            distAddFlag[j][tmpIdx][offsetFlag].data(), size));
                        matrixCondition.notify_one();
                    }
                }
            }
        };

        // functor execute for matrix table accum
        auto accumFunctor = [&] (int bindCpu) {
            AscendUtils::attachToCpu(bindCpu);
            DeviceScope device;
            while (1) {
                {
                    std::unique_lock<std::mutex> lock(matrixQueueLock);
                    matrixCondition.wait(lock, [&] { 
                        return matrixThreadDone || !matrixQueue.empty() || errorQuit; 
                    });
                    
                    if ((matrixThreadDone && matrixQueue.empty()) || errorQuit) {
                        break;
                    }
                }
            
                while (!matrixQueue.empty()) {
                    matrixQueueLock.lock();
                    MatrixQueueDesc desc = std::move(matrixQueue.front());
                    matrixQueue.pop();
                    matrixQueueLock.unlock();

                    AscendTensor<float16_t, DIMS_2> matrix(desc.matrixPtr, 
                        { SEARCH_LIST_SIZE, numSubQuantizersAligned });
                    AscendTensor<float16_t, DIMS_1> distAccumResult(desc.distPtr, 
                        { SEARCH_LIST_SIZE });
                    AscendTensor<uint16_t, DIMS_1> distAddFlag(desc.flagPtr, {ACCUM_OP_FLAG_ALIGN_SIZE});
                    runDistanceMatrixAccum(matrix, distAccumResult, distAddFlag, stream);

                    std::unique_lock<std::mutex> lock(topkQueueLock);
                    topkQueue.emplace(TopkQueueDesc(
                        desc.distPtr, desc.idPtr, desc.flagPtr, desc.size));
                    topkCondition.notify_one();
                }
            }
        };

        // functor execute for topk
        auto topkFunctor = [&] () {
            while (1) {
                {
                    std::unique_lock<std::mutex> lock(topkQueueLock);
                    topkCondition.wait(lock, [&] { 
                        return distAccumThreadDone || !topkQueue.empty() || errorQuit; 
                    });
                    
                    if ((distAccumThreadDone && topkQueue.empty()) || errorQuit) {
                        break;
                    }
                }
            
                while (!topkQueue.empty()) {
                    topkQueueLock.lock();
                    TopkQueueDesc desc = std::move(topkQueue.front());
                    topkQueue.pop();
                    topkQueueLock.unlock();

                    AscendTensor<float16_t, 1> distance(desc.distPtr, { desc.size });
                    AscendTensor<uint32_t, 1> ids(desc.idPtr, { desc.size });

                    // waiting for specific accum op to finish
                    int waitTicks = 0;
                    double startWait = utils::getMillisecs();
                    while (!(*desc.flagPtr)) {
                        waitTicks++;
                        if (!(waitTicks % THREADS_TIMEOUT_CHECK_TICK) && 
                            (utils::getMillisecs() - startWait) >= THREADS_TIMEOUT_MS) {
                            ASCEND_THROW_MSG("topk functor wait timeout.");
                        }
                    }

                    ASCEND_THROW_IF_NOT(topkOp.exec(distance, ids, outDistance, outIndice));
                }
            }
        };

        auto notifyDone = [](std::mutex& mtx, bool& flag, std::condition_variable& cond) {
            std::unique_lock<std::mutex> lock(mtx);
            flag = true;
            cond.notify_one();
        };

        // step 1: distance table build, aicore run distance table ops
        for (int j = 0; j < tables; j++) {
            auto listNo = listNos[j].view();
            auto table = distTables[j].view();
            auto flag = distTableFlag[j].view();
            runDistanceTableBuild(query, pqCentroids, listNo, coarseCentroids, table, flag, stream);
        }

        // step 2: distance matrix build, cpu run distance matrix tasks
        for (int j = 0; j < THREADS_FOR_MATRIX_TABLE; j++) {
            matrixTableRet.emplace_back(threadPool.Enqueue(disMatrixFunctor, (j + 1), j));
        }

        // step 3: accumulate distance
        auto accumFunctorRet = threadPool.Enqueue(accumFunctor, 0);

        // step 4: topk task
        auto topkFunctorRet = threadPool.Enqueue(topkFunctor);

        try {
            // waiting for matrix thread to finish
            for (auto& ret : matrixTableRet) {
                ret.get();
            }
            notifyDone(matrixQueueLock, matrixThreadDone, matrixCondition);
        
            // waiting for accum thread to finish
            accumFunctorRet.get();
            notifyDone(topkQueueLock, distAccumThreadDone, topkCondition);

            // waiting for topk thread to finish
            topkFunctorRet.get();
        } catch (std::exception &e) {
            errorQuit = true;
            notifyDone(matrixQueueLock, matrixThreadDone, matrixCondition);
            notifyDone(topkQueueLock, distAccumThreadDone, topkCondition);
            ASCEND_THROW_MSG(e.what());
        }
    }

    topkOp.reorder(outDistances, outIndices);
}

void IndexIVFPQ::searchImpl(int n, const float16_t* x, int k,
                            float16_t* distances, idx_t* labels)
{
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), {n, dims});
    AscendTensor<float16_t, DIMS_2> outDistances(distances, {n, k});
    AscendTensor<uint32_t, DIMS_2> outIndices(labels, {n, k});

    // init result mem to invalid data
    outDistances.initValue(Limits<float16_t>::getMax());
    outIndices.initValue(std::numeric_limits<uint32_t>::max());
    return searchImpl(queries, k, outDistances, outIndices);
}

void IndexIVFPQ::resetDistTableOperator()
{
    distTableBuildOp.reset();
    AscendOpDesc desc("DistanceTableBuild");
    std::vector<int64_t> queryShape({ 1, dims });
    std::vector<int64_t> pqCentroidsShape({ numSubQuantizers, numSubQuantizerCodes * dimPerSubQuantizer });
    std::vector<int64_t> listIdsShape({ 1, DISTANCE_TABLE_SIZE });
    std::vector<int64_t> coarseCentroidsShape({ numLists, dims });
    std::vector<int64_t> disTableShape({ 1, DISTANCE_TABLE_SIZE, numSubQuantizers, numSubQuantizerCodes });
    std::vector<int64_t> flagShape({ ACCUM_OP_FLAG_ALIGN_SIZE });
    desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, pqCentroidsShape.size(), pqCentroidsShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_INT32, listIdsShape.size(), listIdsShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
    
    desc.addOutputTensorDesc(ACL_FLOAT16, disTableShape.size(), disTableShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    distTableBuildOp = std::make_unique<AscendOperator>(desc);
}

void IndexIVFPQ::resetDistAccumOperator()
{
    distAccumOp.reset();
    AscendOpDesc desc("DistAccum");
    std::vector<int64_t> distMatrixShape({ SEARCH_LIST_SIZE, numSubQuantizersAligned });
    std::vector<int64_t> addResultShape({ SEARCH_LIST_SIZE });
    std::vector<int64_t> addFlagShape({ ACCUM_OP_FLAG_ALIGN_SIZE });
    desc.addInputTensorDesc(ACL_FLOAT16, distMatrixShape.size(), distMatrixShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, addResultShape.size(), addResultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, addFlagShape.size(), addFlagShape.data(), ACL_FORMAT_ND);

    distAccumOp = std::make_unique<AscendOperator>(desc);
}

void IndexIVFPQ::runDistanceTableBuild(
    AscendTensor<float16_t, DIMS_2>& queries,
    AscendTensor<float16_t, DIMS_2>& PQCentroids,
    AscendTensor<int32_t, DIMS_2>& listNos,
    AscendTensor<float16_t, DIMS_2>& l1Centroids,
    AscendTensor<float16_t, DIMS_4>& distTables,
    AscendTensor<uint16_t, DIMS_1>& flag,
    aclrtStream stream)
{
    ASCEND_ASSERT(distTableBuildOp.get());
    std::vector<const aclDataBuffer *> distTableOpInput;
    distTableOpInput.emplace_back(aclCreateDataBuffer(queries.data(), 
        queries.getSizeInBytes()));
    distTableOpInput.emplace_back(aclCreateDataBuffer(PQCentroids.data(), 
        PQCentroids.getSizeInBytes()));
    distTableOpInput.emplace_back(aclCreateDataBuffer(listNos.data(), 
        listNos.getSizeInBytes()));
    distTableOpInput.emplace_back(aclCreateDataBuffer(l1Centroids.data(), 
        l1Centroids.getSizeInBytes()));

    std::vector<aclDataBuffer *> distTableOpOutput;
    distTableOpOutput.emplace_back(aclCreateDataBuffer(distTables.data(), 
        distTables.getSizeInBytes()));
    distTableOpOutput.emplace_back(aclCreateDataBuffer(flag.data(), 
        flag.getSizeInBytes()));
    
    distTableBuildOp->exec(distTableOpInput, distTableOpOutput, stream);
    
    for (auto &item : distTableOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distTableOpInput.clear();

    for (auto &item : distTableOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distTableOpOutput.clear();
}

void IndexIVFPQ::runDistanceMatrixAccum(
    AscendTensor<float16_t, DIMS_2>& distMatrix,
    AscendTensor<float16_t, DIMS_1>& distAccumResult,
    AscendTensor<uint16_t, DIMS_1>& distAccumFlag,
    aclrtStream stream)
{
    ASCEND_ASSERT(distAccumOp.get());
    std::vector<const aclDataBuffer *> distAddOpInput;
    distAddOpInput.emplace_back(aclCreateDataBuffer(distMatrix.data(), 
        distMatrix.getSizeInBytes()));

    std::vector<aclDataBuffer *> distAddOpOutput;
    distAddOpOutput.emplace_back(aclCreateDataBuffer(distAccumResult.data(), 
        distAccumResult.getSizeInBytes()));
    distAddOpOutput.emplace_back(aclCreateDataBuffer(distAccumFlag.data(), 
        distAccumFlag.getSizeInBytes()));
    
    distAccumOp->exec(distAddOpInput, distAddOpOutput, stream);

    for (auto &item : distAddOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distAddOpInput.clear();

    for (auto &item : distAddOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distAddOpOutput.clear();
}

size_t IndexIVFPQ::removeIdsImpl(const ascend::IDSelector& sel)
{
    // 
    // | id0 id1 id2 ... id98 id99 id100 |
    //            ^                  |
    //            |__________________|
    // if id2 is to be removed, copy the last id(id100) to
    // the mem place of id2, and decrease the list size to size-1.
    size_t removeCnt = 0;

#pragma omp parallel for reduction(+: removeCnt)
    for (int id = 0; id < numLists; id++) {
        DeviceScope device;
        auto& indicesList = deviceListIndices[id];
        auto& codeList = deviceListData[id];

        uint32_t* indicesPtr = indicesList->data();
        unsigned char* codePtr = codeList->data();
        bool hasMoved = false;
        int j = indicesList->size() - 1;
        int i = 0;
        while (i <= j) {
            if (sel.is_member((*indicesList)[i])) {
                auto err = memcpy_s(indicesPtr + i, sizeof(uint32_t), indicesPtr + j, 
                    sizeof(uint32_t));
                ASCEND_THROW_IF_NOT_MSG(err == EOK, "An error occured when removing Index ID!");
                err = memcpy_s(codePtr + i * bytesPerVector, bytesPerVector,
                    codePtr + j * bytesPerVector, bytesPerVector);
                ASCEND_THROW_IF_NOT_MSG(err == EOK, "An error occured when removing Index ID!");
                j--;
                removeCnt++;
                hasMoved = true;
            } else {
                i++;
            }
        }

        if (hasMoved) {
            indicesList->resize(j + 1);
            codeList->resize((j + 1) * bytesPerVector);
            indicesList->reclaim(false);
            codeList->reclaim(false);
        }
    }

    this->ntotal -= removeCnt;
    return removeCnt;
}
}  // namespace ascend
