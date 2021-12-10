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

#include <algorithm>
#include <atomic>
#include <ascenddaemon/impl/IndexIVFSQL2.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/TaskQueueItem.h>
#include <ascenddaemon/utils/Limits.h>

namespace ascend {
namespace {
const int SQOP_INPUT_NUM = 6;
const int SQOP_OUTPUT_NUM = 3;
}

IndexIVFSQL2::IndexIVFSQL2(int numList, int dim, bool encodeResidual, int nprobes, int resourceSize)
    : IndexIVFSQ<float>(numList, dim, encodeResidual, nprobes, resourceSize),
      byResidual(encodeResidual),
      distSqOpInput(SQOP_INPUT_NUM, nullptr),
      distSqOpOutput(SQOP_OUTPUT_NUM, nullptr)
{
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN == 0);

    // the burstLen * dim must be less than BURST_LEN * DEFAULT_DIM
    this->burstLen = BURST_LEN;
    // if the dim bigger than 128 
    if (dim > 128) {
        // the burstLen only support 16, because of the ub size of the aicore
        this->burstLen = 16;
    }
    // the output of vcmin operator contain 2 value, the min and index
    this->bursts = SEARCH_LIST_SIZE / this->burstLen * 2;
    for (int i = 0; i < numLists; ++i) {
        preComputeData.push_back(std::make_unique<DeviceVector<float>>(MemorySpace::DEVICE_HUGEPAGE));
    }

    resetDistCompOperator(numList);
    resetSqDistOperator();
}

IndexIVFSQL2::~IndexIVFSQL2() {}

void IndexIVFSQL2::searchImplL2(AscendTensor<float16_t, DIMS_2> &queries,
                                AscendTensor<float16_t, DIMS_2> &l1Distances,
                                AscendTensor<float16_t, DIMS_2> &outDistances,
                                AscendTensor<uint32_t, DIMS_2> &outIndices)
{
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);
    int k = outDistances.getSize(1);
    int maxScanSeg = utils::divUp(maxListLength, SEARCH_LIST_SIZE);

    // tensor for operator flags
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { n, nprobe, (maxScanSeg * FLAG_ALIGN) }, stream);
    (void)opFlag.zero();
    // tensor for telling operator how many code to calculate
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { n, nprobe, (maxScanSeg * SIZE_ALIGN) }, stream);
    // tensor for operator outputing sq distance
    AscendTensor<float16_t, DIMS_3> distResult(mem, { n, nprobe, maxListLength }, stream);
    AscendTensor<float16_t, DIMS_3> minDistResult(mem, {n, nprobe, (maxScanSeg * bursts)}, stream);

    AscendTensor<float16_t, DIMS_2> minDistances(mem, {n, k}, stream);
    AscendTensor<uint32_t, DIMS_2> minIndices(mem, {n, k}, stream);
    minDistances.initValue(Limits<float16_t>::getMax());
    minIndices.initValue(std::numeric_limits<uint32_t>::max());

    AscendTensor<uint32_t, DIMS_2> l1KIndices(mem, { n, nprobe }, stream);
    AscendTensor<float16_t, DIMS_2> l1KDistances(mem, { n, nprobe }, stream);
    AscendTensor<float16_t, DIMS_3> residuals(mem, { n, nprobe, dims }, stream);
    l1KDistances.initValue(Limits<float16_t>::getMax());

    std::vector<std::vector<QueueItem>> topkQueue(n, std::vector<QueueItem>((maxScanSeg * nprobe)));
    std::vector<std::future<void>> topkFunctorRet;
    std::vector<std::pair<volatile bool, volatile int>> executeInfo(n, { false, 0 });
    bool errorQuit = false;

    // topk functor
    auto topkFunctor = [&](int idx) {
        // bind thread to fixed cpus for stablity time costing,
        // bind from cpu 0 to cpu 3, cpu 4-5 is for main thread.
        if (idx < THREADS_CNT) {
            AscendUtils::attachToCpu(idx);
        }

        auto outDistance = outDistances[idx].view();
        auto outIndice = outIndices[idx].view();
        int iter = 0;
        while (!errorQuit && (!executeInfo[idx].first || iter < executeInfo[idx].second)) {
            while (iter < executeInfo[idx].second) {
                auto &item = topkQueue[idx][iter];

                // waitting for the item operator to be added to the stream to run
                WAITING_FLAG_READY((item.IsExecuting()), TIMEOUT_CHECK_TICK, TIMEOUT_MS);

                std::tuple<float16_t *, float16_t *, uint32_t *> opOutTp(item.distPtr, item.extremePtr, item.idPtr);
                std::tuple<float16_t *, uint32_t *, int> topkHeapTp(outDistance.data(), outIndice.data(), k);
                std::tuple<float16_t *, uint32_t *> minHeapTp(minDistances[idx].data(), minIndices[idx].data());
                
                // waitting for the operator finishing running, while the flags will be
                // setted by operator when finishing running.
                WAITING_FLAG_READY((*item.flagPtr && *(item.flagPtrSec)), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
                if (iter == 0) {
                    ASCEND_THROW_IF_NOT(topKMinOp.exec(opOutTp, topkHeapTp, minHeapTp, item.size, this->burstLen));
                } else {
                    ASCEND_THROW_IF_NOT(topKMinOp.exec(opOutTp, topkHeapTp, item.size, 0, this->burstLen));
                }
                iter++;
            }
        }

        // reorder the results in distance's ascending order
        if (!errorQuit) {
            topKMinOp.reorder(outDistance, outIndice);
        }
    };

    for (int i = 0; i < n; i++) {
        // add topkFunctor task to threadpool for async executing
        topkFunctorRet.emplace_back(threadPool->Enqueue(topkFunctor, i));
    }

    for (int nIdx = 0; nIdx < n; ++nIdx) {
        AscendTensor<uint32_t, DIMS_1> l1KIndice(l1KIndices[nIdx].data(), { nprobe });
        AscendTensor<float16_t, DIMS_1> l1KDistance(l1KDistances[nIdx].data(), { nprobe });
        AscendTensor<float16_t, DIMS_1> l1Distance(l1Distances[nIdx].data(), { numLists });
        AscendTensor<uint32_t, DIMS_1> l1Indice;
        ASCEND_THROW_IF_NOT(topKMinOp.exec(l1Distance, l1Indice, l1KDistance, l1KIndice));

        // residual calculate, query - L1 coarse centroids
        AscendTensor<float16_t, DIMS_2> queryL2(queries[nIdx].data(), { 1, dims });
        if (byResidual) {
            AscendTensor<float16_t, DIMS_1> query(queries[nIdx].data(), { dims });
            AscendTensor<float16_t, DIMS_2> residual(residuals[nIdx].data(), { nprobe, dims });
            calcResiduals(query, l1KIndice, residual);
            queryL2 = std::move(residual);
        }

        for (int probeIdx = 0; probeIdx < nprobe; ++probeIdx) {
            int list = l1KIndice[probeIdx].value();

            // seperator list's code for several segs to run sqdistance,
            // because of fixed-shape limitation of aicore's operator.
            int segs = utils::divUp(deviceListIndices[list]->size(), SEARCH_LIST_SIZE);
            for (int m = 0; m < segs; m++) {
                int offset = m * SEARCH_LIST_SIZE;
                int minOffset = m * bursts;
                uint32_t size = std::min(static_cast<uint32_t>(SEARCH_LIST_SIZE),
                    static_cast<uint32_t>((deviceListIndices[list]->size() - offset)));

                // code is stored in Zz format, Zz format is 4 dims shaped. z's shape is
                // 16 X 16(AiCore cube's matrix operation's size). Z's shape is (SEARCH_LIST_SIZE / 16) X (dims / 16).
                // Zz's 4 dims shape is ((SEARCH_LIST_SIZE / 16), (dims / 16), 16, 16)
                AscendTensor<float16_t, DIMS_2> query(queryL2[(byResidual ? probeIdx : 0)].data(), { 1, dims });
                AscendTensor<uint8_t, DIMS_4> code(static_cast<uint8_t *>(deviceListData[list]->data()) + dims * offset,
                    { SEARCH_SHAPED_SIZE, dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
                AscendTensor<float, DIMS_1> precomp(preComputeData[list]->data() + offset, { SEARCH_LIST_SIZE });
                AscendTensor<uint16_t, DIMS_1> flag(opFlag[nIdx][probeIdx][m * FLAG_ALIGN].data(), { FLAG_ALIGN });
                AscendTensor<uint32_t, DIMS_1> actualSize(opSize[nIdx][probeIdx][m * SIZE_ALIGN].data(),
                    { SIZE_ALIGN });
                AscendTensor<float16_t, DIMS_2> result(distResult[nIdx][probeIdx][offset].data(),
                    { 1, SEARCH_LIST_SIZE });
                AscendTensor<float16_t, DIMS_2> minResult(minDistResult[nIdx][probeIdx][minOffset].data(),
                    { 1, bursts });
                actualSize[0] = size;
                runSqDistOperator(query, code, precomp, actualSize, result, minResult, flag);
                topkQueue[nIdx][executeInfo[nIdx].second].SetExecuting(result.data(),
                    minResult.data(), deviceListIndices[list]->data() + offset, flag.data(), size);
                executeInfo[nIdx].second++;
            }
        }
        // set quit flags to true, the flags will be used by topk
        // functor to check whether all operators have been added to the stream.
        executeInfo[nIdx].first = true;
    }

    // waiting for topk functor to finish
    int topkWaitIdx = 0;
    try {
        for_each(topkFunctorRet.begin(), topkFunctorRet.end(), [&](auto &ret) {
            topkWaitIdx++;
            ret.get();
        });
    } catch (AscendException &e) {
        // if exception occures, waitting for the rest topkFunctor to quit.
        errorQuit = true;
        for_each(topkFunctorRet.begin() + topkWaitIdx, topkFunctorRet.end(), [](auto &ret) { ret.wait(); });
        ASCEND_THROW_MSG(e.what());
    }
}

void IndexIVFSQL2::searchImplL1(AscendTensor<float16_t, DIMS_2> &queries, 
                                AscendTensor<float16_t, DIMS_2> &distances,
                                aclrtStream stream)
{
    auto &mem = resources.getMemoryManager();
    AscendTensor<uint16_t, DIMS_1> opFlag(mem, { FLAG_ALIGN }, stream);
    opFlag.zero();
    
    // run l1 distance calculation
    runDistanceCompute(queries, coarseCentroidsShaped, normCoarseCentroids, distances, opFlag, stream);
    uint16_t *volatile flagPtr1 = opFlag.data();
    uint16_t *volatile flagPtr2 = opFlag.data() + FLAG_ALIGN_OFFSET;
    WAITING_FLAG_READY((*flagPtr1) && (*flagPtr2), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
}

void IndexIVFSQL2::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
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
    AscendTensor<float16_t, DIMS_2> l1Distances(mem, { n, numLists }, stream);
    searchImplL1(queries, l1Distances, stream);

    // L2 search, search codes in nprobe IVF list to find topk results
    searchImplL2(queries, l1Distances, outDistances, outIndices);

    // reattach cpus to cpu set { 0, 1, 2, 3, 4, 5 }
    AscendUtils::attachToCpus({ 0, 1, 2, 3, 4, 5 });
}

void IndexIVFSQL2::resetSqDistOperator()
{
    distSqOp.reset();
    AscendOpDesc desc("DistanceIVFSQ8L2");
    std::vector<int64_t> queryShape({ 1, dims });
    std::vector<int64_t> codeShape({ SEARCH_SHAPED_SIZE, dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> precompShape({ SEARCH_LIST_SIZE });
    std::vector<int64_t> vdiffShape({ dims });
    std::vector<int64_t> vminShape({ dims });
    std::vector<int64_t> sizeShape({ SIZE_ALIGN });
    std::vector<int64_t> resultShape({ 1, SEARCH_LIST_SIZE });
    std::vector<int64_t> minResultShape({ 1, bursts });
    std::vector<int64_t> flagShape({ FLAG_ALIGN });
    desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT8, codeShape.size(), codeShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, precompShape.size(), precompShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, vdiffShape.size(), vdiffShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, vminShape.size(), vminShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, resultShape.size(), resultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, minResultShape.size(), minResultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    distSqOp = std::make_unique<AscendOperator>(desc);
}

void IndexIVFSQL2::runSqDistOperator(AscendTensor<float16_t, DIMS_2> &queries, 
                                     AscendTensor<uint8_t, DIMS_4> &codes,
                                     AscendTensor<float, DIMS_1> &precomp,
                                     AscendTensor<uint32_t, DIMS_1> &size,
                                     AscendTensor<float16_t, DIMS_2> &result,
                                     AscendTensor<float16_t, DIMS_2> &minResult,
                                     AscendTensor<uint16_t, DIMS_1> &flag)
{
    ASCEND_ASSERT(distSqOp.get());
    auto stream = resources.getDefaultStream();
    // prepare for input data's buffer
    distSqOpInput[0] = aclCreateDataBuffer(queries.data(), queries.getSizeInBytes());           // input 0
    distSqOpInput[1] = aclCreateDataBuffer(codes.data(), codes.getSizeInBytes());               // input 1
    distSqOpInput[2] = aclCreateDataBuffer(precomp.data(), precomp.getSizeInBytes());           // input 2
    distSqOpInput[3] = aclCreateDataBuffer(vDiff.data(), vDiff.getSizeInBytes());               // input 3
    distSqOpInput[4] = aclCreateDataBuffer(vMin.data(), vMin.getSizeInBytes());                 // input 4
    distSqOpInput[5] = aclCreateDataBuffer(size.data(), size.getSizeInBytes());                 // input 5

    // prepare for output data's buffer
    distSqOpOutput[0] = aclCreateDataBuffer(result.data(), result.getSizeInBytes());            // output 0
    distSqOpOutput[1] = aclCreateDataBuffer(minResult.data(), minResult.getSizeInBytes());      // output 1
    distSqOpOutput[2] = aclCreateDataBuffer(flag.data(), flag.getSizeInBytes());                // output 2

    // async executing operator
    distSqOp->exec(distSqOpInput, distSqOpOutput, stream);

    for (auto &item : distSqOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }

    for (auto &item : distSqOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
}
} // ascend
