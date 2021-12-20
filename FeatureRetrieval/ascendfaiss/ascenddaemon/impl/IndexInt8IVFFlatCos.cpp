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
#include <ascenddaemon/impl/IndexInt8IVFFlatCos.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/Limits.h>

namespace ascend {
namespace {
const int SEARCH_LIST_SIZE = 65536; // must be CUBE_ALIGN_SIZE and BURST_LEN aligned
const int L2NORM_COMPUTE_BATCH = 16384;
const int CUBE_ALIGN = 16;
const int CUBE_ALIGN_INT8 = 32;
const int SEARCH_SHAPED_SIZE = SEARCH_LIST_SIZE / CUBE_ALIGN;
const int TIMEOUT_CHECK_TICK = 5120;
const double TIMEOUT_MS = 50000;
const int SIZE_ALIGN = 8;
const int THREADS_CNT = 4;
const int BURST_LEN = 32;
const int L1_COS_OP_INPUT_NUM = 4;
const int L1_COS_OP_OUTPUT_NUM = 2;
const int L2_OP_INPUT_NUM = 5;
const int L2_OP_OUTPUT_NUM = 3;
}

IndexInt8IVFFlatCos::IndexInt8IVFFlatCos(int numList, int dim, int nprobes, int resourceSize)
    : IndexInt8IVFFlat<float16_t>(numList, dim, nprobes, MetricType::METRIC_COSINE, resourceSize),
      l1OpInput(L1_COS_OP_INPUT_NUM, nullptr),
      l1OpOutput(L1_COS_OP_OUTPUT_NUM, nullptr),
      l2OpInput(L2_OP_INPUT_NUM, nullptr),
      l2OpOutput(L2_OP_OUTPUT_NUM, nullptr)
{
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN_INT8 == 0);

    // the output of vcmax operator contain 2 value, the min and index
    this->burstSize = SEARCH_LIST_SIZE / BURST_LEN * 2;
    int8L2Norm = std::make_unique<Int8L2Norm>(dim);

    for (int i = 0; i < numLists; ++i) {
        preComputeData.push_back(std::make_unique<DeviceVector<float16_t>>(MemorySpace::DEVICE_HUGEPAGE));
    }
    resetL1DistOperator();
    resetL2DistOperator();
}

IndexInt8IVFFlatCos::~IndexInt8IVFFlatCos() {}

void IndexInt8IVFFlatCos::addVectors(int listId, size_t numVecs, const int8_t *codes, const uint32_t *indices)
{
    ASCEND_THROW_IF_NOT(this->isTrained);
    ASCEND_THROW_IF_NOT(listId >= 0 && listId < numLists);

    if (numVecs == 0) {
        return;
    }

    // 1. resize precompute data
    size_t originLen = getListLength(listId);
    size_t tmpLen = utils::roundUp((originLen + numVecs), static_cast<size_t>(CUBE_ALIGN));
    preComputeData[listId]->resize(tmpLen);

    // 2. compute the l2 norm of code data
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    AscendTensor<float16_t, 1> precompData(preComputeData[listId]->data() + originLen,
        { static_cast<int>(utils::roundUp(numVecs, CUBE_ALIGN)) });
    AscendTensor<int8_t, DIMS_2> codesData(const_cast<int8_t *>(codes), { static_cast<int>(numVecs), this->dims });
    AscendTensor<uint32_t, DIMS_2> actualNum(mem,
        { static_cast<int>(utils::divUp(numVecs, L2NORM_COMPUTE_BATCH)), SIZE_ALIGN }, stream);

    int8L2Norm->dispatchL2NormTask(codesData, precompData, actualNum, stream);

    // 3. add codes
    IndexInt8IVFFlat::addVectors(listId, numVecs, codes, indices);

    // 4. wait aicore complete the l2 norm of code data
    resources.syncDefaultStream();
}

void IndexInt8IVFFlatCos::updateCoarseCentroidsData(AscendTensor<int8_t, DIMS_2> &coarseCentroidsData)
{
    // update coarse centroids for L1 search.
    IndexInt8IVFFlat::updateCoarseCentroidsData(coarseCentroidsData);

    // update L2 norm of coarseCentroids
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    int numCoarseCents = coarseCentroidsData.getSize(0);

    AscendTensor<float16_t, 1> tmpNormData(mem, { utils::roundUp(numCoarseCents, CUBE_ALIGN) }, stream);
    AscendTensor<uint32_t, DIMS_2> actualNum(mem, { utils::divUp(numCoarseCents, L2NORM_COMPUTE_BATCH), SIZE_ALIGN },
        stream);
    int8L2Norm->dispatchL2NormTask(coarseCentroidsData, tmpNormData, actualNum, stream);
    resources.syncDefaultStream();

    AscendTensor<float16_t, 1> tmpNormCentroids({ numCoarseCents });
    tmpNormCentroids.copyFromSync(tmpNormData);
    normCoarseCentroids = std::move(tmpNormData);

    this->isTrained = true;
}


void IndexInt8IVFFlatCos::addImpl(int n, const int8_t *x, const idx_t *ids)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);
}

void IndexInt8IVFFlatCos::searchImplL2(AscendTensor<int8_t, DIMS_2> &queries,
                                       AscendTensor<float16_t, DIMS_1> &queriesNorm,
                                       AscendTensor<float16_t, DIMS_2> &l1Distances,
                                       AscendTensor<float16_t, DIMS_2> &outDistances, 
                                       AscendTensor<uint32_t, DIMS_2> &outIndices)
{
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);
    int k = outDistances.getSize(1);
    int maxScanSeg = utils::divUp(maxListLength, SEARCH_LIST_SIZE);

    // residual calculate, query - L1 coarse centroids
    AscendTensor<int8_t, DIMS_3> queriesL2(queries.data(), { n, 1, dims });

    // tensor for operator flags
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { n, nprobe, (maxScanSeg * FLAG_ALIGN) }, stream);
    (void)opFlag.zero();
    // tensor for telling operator how many code to calculate
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { n, nprobe, (maxScanSeg * SIZE_ALIGN) }, stream);
    // tensor for operator outputing cos distance
    AscendTensor<float16_t, DIMS_3> distResult(mem, { n, nprobe, maxListLength }, stream);
    AscendTensor<float16_t, DIMS_3> maxDistResult(mem, {n, nprobe, (maxScanSeg * burstSize)}, stream);

    AscendTensor<float16_t, DIMS_2> maxDistances(mem, {n, k}, stream);
    AscendTensor<uint32_t, DIMS_2> maxIndices(mem, {n, k}, stream);
    maxDistances.initValue(Limits<float16_t>::getMin());
    maxIndices.initValue(std::numeric_limits<uint32_t>::max());

    AscendTensor<uint32_t, DIMS_2> l1KIndices(mem, { n, nprobe }, stream);
    AscendTensor<float16_t, DIMS_2> l1KDistances(mem, { n, nprobe }, stream);
    l1KDistances.initValue(Limits<float16_t>::getMin());
    l1KIndices.initValue(std::numeric_limits<uint32_t>::max());

    std::vector<std::vector<QueueItem>> topkQueue(n, std::vector<QueueItem>(maxScanSeg * nprobe));
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
                std::tuple<float16_t *, uint32_t *> maxHeapTp(maxDistances[idx].data(), maxIndices[idx].data());

                // waitting for the operator finishing running, while the flags will be
                // setted by operator when finishing running.
                WAITING_FLAG_READY((*item.flagPtr && *(item.flagPtrSec)), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
                if (iter == 0) {
                    ASCEND_THROW_IF_NOT(topKMaxOp.exec(opOutTp, topkHeapTp, maxHeapTp, item.size, BURST_LEN));
                } else {
                    ASCEND_THROW_IF_NOT(topKMaxOp.exec(opOutTp, topkHeapTp, item.size, 0, BURST_LEN));
                }
                iter++;
            }
        }

        // reorder the results in distance's ascending order
        if (!errorQuit) {
            topKMaxOp.reorder(outDistance, outIndice);
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
        ASCEND_THROW_IF_NOT(topKMaxOp.exec(l1Distance, l1Indice, l1KDistance, l1KIndice));

        for (int probeIdx = 0; probeIdx < nprobe; ++probeIdx) {
            int list = l1KIndice[probeIdx].value();

            // seperator list's code for several segs to run cos distance,
            // because of fixed-shape limitation of aicore's operator.
            int segs = utils::divUp(deviceListIndices[list]->size(), SEARCH_LIST_SIZE);
            for (int m = 0; m < segs; m++) {
                int offset = m * SEARCH_LIST_SIZE;
                int maxOffset = m * burstSize;
                uint32_t size = std::min(static_cast<uint32_t>(SEARCH_LIST_SIZE),
                    static_cast<uint32_t>((deviceListIndices[list]->size() - offset)));

                // code is stored in Zz format, Zz format is 4 dims shaped. z's shape is
                // 16 X 16(AiCore cube's matrix operation's size). Z's shape is (SEARCH_LIST_SIZE / 16) X (dims / 16).
                // Zz's 4 dims shape is ((SEARCH_LIST_SIZE / 16), (dims / 16), 16, 16)
                AscendTensor<int8_t, DIMS_2> query(queriesL2[nIdx][0].data(), { 1, dims });
                AscendTensor<int8_t, DIMS_4> code(deviceListData[list]->data() + dims * offset,
                    { SEARCH_SHAPED_SIZE, dims / CUBE_ALIGN_INT8, CUBE_ALIGN, CUBE_ALIGN_INT8 });
                AscendTensor<float16_t, DIMS_1> precomp(preComputeData[list]->data() + offset, { SEARCH_LIST_SIZE });
                AscendTensor<uint16_t, DIMS_1> flag(opFlag[nIdx][probeIdx][m * FLAG_ALIGN].data(),
                    { FLAG_ALIGN });
                AscendTensor<uint32_t, DIMS_1> actualSize(opSize[nIdx][probeIdx][m * SIZE_ALIGN].data(),
                    { SIZE_ALIGN });
                AscendTensor<float16_t, DIMS_2> result(distResult[nIdx][probeIdx][offset].data(),
                    { 1, SEARCH_LIST_SIZE });
                AscendTensor<float16_t, DIMS_2> maxResult(maxDistResult[nIdx][probeIdx][maxOffset].data(),
                    { 1, burstSize });
                actualSize[0] = size;
                runL2DistOperator(query, code, queriesNorm, precomp, actualSize, result, maxResult, flag);
                topkQueue[nIdx][executeInfo[nIdx].second].SetExecuting(result.data(), maxResult.data(),
                    deviceListIndices[list]->data() + offset, flag.data(), size);
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

void IndexInt8IVFFlatCos::searchImplL1(AscendTensor<int8_t, DIMS_2> &queries,
                                       AscendTensor<float16_t, DIMS_1> queriesNorm,
                                       AscendTensor<float16_t, DIMS_2> &distances,
                                       aclrtStream stream)
{
    auto &mem = resources.getMemoryManager();
    AscendTensor<uint16_t, DIMS_1> opFlag(mem, { FLAG_ALIGN }, stream);
    opFlag.zero();
    
    // run l1 distance calculation
    runL1DistOperator(queries, queriesNorm, distances, opFlag);

    uint16_t *volatile flagPtr1 = opFlag.data();
    uint16_t *volatile flagPtr2 = opFlag.data() + FLAG_ALIGN_OFFSET;
    WAITING_FLAG_READY((*flagPtr1) && (*flagPtr2), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
}

void IndexInt8IVFFlatCos::searchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels)
{
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    AscendTensor<int8_t, DIMS_2> queries(const_cast<int8_t *>(x), { n, dims });
    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<uint32_t, DIMS_2> outIndices(labels, { n, k });

    // init results to invalid data.
    outDistances.initValue(Limits<float16_t>::getMin());
    outIndices.initValue(std::numeric_limits<uint32_t>::max());

    // for performance improving, bind the main thread to cpu4-5,
    // and bind the threadpool to cpu0-cpu3. when n == 1, attach
    // main thread to one cpu(cpu5) is better than multicpus.
    if (n > 1) {
        AscendUtils::attachToCpus({ 4, 5 });
    } else {
        AscendUtils::attachToCpus({ 5 });
    }

    // compute the L2 norm of queries
    AscendTensor<float16_t, DIMS_1> queriesNorm(mem, { utils::roundUp(n, CUBE_ALIGN) }, stream);
    AscendTensor<uint32_t, DIMS_2> actualNum(mem, { utils::divUp(n, L2NORM_COMPUTE_BATCH), SIZE_ALIGN }, stream);
    int8L2Norm->dispatchL2NormTask(queries, queriesNorm, actualNum, stream);
    resources.syncDefaultStream();

    // L1 search, to find nprobe IVF list
    AscendTensor<float16_t, DIMS_2> l1Distances(mem, { n, numLists }, stream);
    searchImplL1(queries, queriesNorm, l1Distances, stream);

    // L2 search, search codes in nprobe IVF list to find topk results
    searchImplL2(queries, queriesNorm, l1Distances, outDistances, outIndices);

    // reattach cpus to cpu set { 0, 1, 2, 3, 4, 5 }
    AscendUtils::attachToCpus({ 0, 1, 2, 3, 4, 5 });
}

void IndexInt8IVFFlatCos::resetL1DistOperator()
{
    auto l1DistOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("DistanceIVFInt8CosL1");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> codeShape({ numLists / CUBE_ALIGN, dims / CUBE_ALIGN_INT8, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        std::vector<int64_t> vQueriesL2NormShape({ (batch + CUBE_ALIGN - 1) / CUBE_ALIGN * CUBE_ALIGN });
        std::vector<int64_t> vCodesL2NormShape({ numLists });
        std::vector<int64_t> resultShape({ batch, numLists });
        std::vector<int64_t> flagShape({ FLAG_ALIGN });

        desc.addInputTensorDesc(ACL_INT8, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT8, codeShape.size(), codeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, vQueriesL2NormShape.size(), vQueriesL2NormShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, vCodesL2NormShape.size(), vCodesL2NormShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, resultShape.size(), resultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = std::make_unique<AscendOperator>(desc);
    };
    for (auto batch : searchPageSizes) {
        distL1Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        l1DistOpReset(distL1Ops[batch], batch);
    }
}

void IndexInt8IVFFlatCos::resetL2DistOperator()
{
    AscendOpDesc desc("DistanceIVFInt8Cos");
    std::vector<int64_t> queryShape({ 1, dims });
    std::vector<int64_t> codeShape({ 
        SEARCH_LIST_SIZE / CUBE_ALIGN, dims / CUBE_ALIGN_INT8, CUBE_ALIGN, CUBE_ALIGN_INT8 });
    std::vector<int64_t> vQueriesL2NormShape({ CUBE_ALIGN });
    std::vector<int64_t> vCodesL2NormShape({ SEARCH_LIST_SIZE });
    std::vector<int64_t> sizeShape({ SIZE_ALIGN });
    std::vector<int64_t> resultShape({ 1, SEARCH_LIST_SIZE });
    std::vector<int64_t> maxResultShape({1, burstSize});
    std::vector<int64_t> flagShape({ FLAG_ALIGN });

    desc.addInputTensorDesc(ACL_INT8, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_INT8, codeShape.size(), codeShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, vQueriesL2NormShape.size(), vQueriesL2NormShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, vCodesL2NormShape.size(), vCodesL2NormShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, resultShape.size(), resultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, maxResultShape.size(), maxResultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    distCosOp.reset();
    distCosOp = std::make_unique<AscendOperator>(desc);
}
} // ascend