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

#include <set>
#include <algorithm>

#include <ascenddaemon/impl/IndexInt8FlatCos.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/DeviceVectorInl.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/AscendThreadPool.h>
#include <ascenddaemon/utils/Limits.h>

namespace ascend {
namespace {
const int L2NORM_COMPUTE_BATCH = 16384;
const double TIMEOUT_MS = 50000;
const int TIMEOUT_CHECK_TICK = 5120;
const int FLAG_ALIGN_SIZE = 16;
const int FLAG_ALIGN_OFFSET = 16; // core 0 use first 16 flag, and core 1 use the second 16 flag.
const int SIZE_ALIGN_SIZE = 8;
const int THREADS_CNT = 6;
const int BURST_LEN = 64;
const int FP16_ALGIN = 16;
const int CUBE_ALIGN = 16;
const int CUBE_ALIGN_INT8 = 32;
const int CORE_NUM = 2;
const int FLAG_NUM = 16;
const int IDX_ACTUAL_NUM = 0;
const int IDX_COMP_OFFSET = 1;
const int IDX_MASK_LEN = 2;
const int IDX_USE_MASK = 3;
}

IndexInt8FlatCos::IndexInt8FlatCos(int dim, int resourceSize)
    : IndexInt8Flat<float16_t>(dim, MetricType::METRIC_L2, resourceSize)
{
    // align by 2
    this->burstsOfComputeBatch = (this->distComputeBatch + BURST_LEN - 1) / BURST_LEN * 2;
    int8L2Norm = std::make_unique<Int8L2Norm>(dim);
    resetDistCompOp(distComputeBatch);
}

IndexInt8FlatCos::~IndexInt8FlatCos() {}

void IndexInt8FlatCos::addVectors(AscendTensor<int8_t, DIMS_2> &rawData)
{
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();

    // 1. dispatch the task of compute the l2 norm of code data
    int num = rawData.getSize(0);
    AscendTensor<float16_t, 1> precompData(mem, { static_cast<int>(utils::roundUp(num, CUBE_ALIGN)) }, stream);
    AscendTensor<uint32_t, DIMS_2> actualNum(mem, { utils::divUp(num, L2NORM_COMPUTE_BATCH), SIZE_ALIGN_SIZE }, stream);
    int8L2Norm->dispatchL2NormTask(rawData, precompData, actualNum, stream);

    // 2. save code
    IndexInt8Flat::addVectors(rawData);

    // 3. wait all the L2Norm task compute
    aclrtSynchronizeStream(stream);

    // 4. save the pre compute data
    int vecSize = utils::divUp(this->ntotal, this->distComputeBatch);
    int newVecSize = utils::divUp(this->ntotal + num, this->distComputeBatch);
    int lastOffset = ntotal % this->distComputeBatch;
    int offset = 0;

    // if normBase[vecSize - 1] is not full
    if (lastOffset != 0) {
        int cpyNum = std::min(num, this->distComputeBatch * vecSize - static_cast<int>(ntotal));
        (void)memcpy_s(this->normBase.at(vecSize - 1)->data() + lastOffset, this->distComputeBatch * sizeof(float16_t),
            precompData.data(), cpyNum * sizeof(float16_t));
        offset += cpyNum;
    }

    for (int i = vecSize; i < newVecSize; ++i) {
        int cpyNum = std::min(num - offset, this->distComputeBatch);
        (void)memcpy_s(this->normBase.at(i)->data(), this->distComputeBatch * sizeof(float16_t),
            precompData.data() + offset, cpyNum * sizeof(float16_t));
        offset += cpyNum;
    }

    this->ntotal += num;
}

void IndexInt8FlatCos::searchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels)
{
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<int8_t, DIMS_2> queries(const_cast<int8_t *>(x), { n, dims });
    AscendTensor<float16_t, DIMS_1> queriesNorm(mem, { utils::roundUp(n, CUBE_ALIGN) }, stream);
    AscendTensor<uint32_t, DIMS_2> actualNum(mem, { utils::divUp(n, L2NORM_COMPUTE_BATCH), SIZE_ALIGN_SIZE }, stream);
    int8L2Norm->dispatchL2NormTask(queries, queriesNorm, actualNum, stream);

    int repeatTimes = utils::divUp(this->ntotal, this->distComputeBatch);
    AscendTensor<float16_t, DIMS_3> distResult(mem, { repeatTimes, n, distComputeBatch }, stream);
    AscendTensor<float16_t, DIMS_3> minDistResult(mem, { repeatTimes, n, this->burstsOfComputeBatch }, stream);
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { repeatTimes, CORE_NUM, SIZE_ALIGN_SIZE }, stream);
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { repeatTimes, FLAG_NUM, FLAG_ALIGN_SIZE }, stream);
    opFlag.zero();

    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<idx_t, DIMS_2> outIndices(labels, { n, k });
    outDistances.initValue(Limits<float16_t>::getMin());
    outIndices.initValue(std::numeric_limits<idx_t>::max());

    AscendTensor<float16_t, DIMS_2> minDistances(mem, { n, k }, stream);
    AscendTensor<uint32_t, DIMS_2> minIndices(mem, { n, k }, stream);
    minDistances.initValue(Limits<float16_t>::getMin());
    minIndices.initValue(std::numeric_limits<uint32_t>::max());

    // 2. wait computed
    resources.syncDefaultStream();

    // 3. run the topK operator to select the top K async
    bool errorQuit = false;
    auto topkFunctor = [&](int idx) {
        if (idx < THREADS_CNT) {
            AscendUtils::attachToCpu(idx);
        }

        AscendTensor<uint32_t, DIMS_1> indices;
        uint32_t offset = 0;
        for (int i = 0; i < repeatTimes && !errorQuit; ++i) {
            for (int j = 0; j < CORE_NUM; ++j) {
                uint16_t *volatile flagPtr = opFlag[i][j].data();
                WAITING_FLAG_READY(*flagPtr, TIMEOUT_CHECK_TICK, TIMEOUT_MS);
            }

            int size = (i == (repeatTimes - 1)) ? (ntotal - offset) : distComputeBatch;
            for (int j = idx; j < n; j += THREADS_CNT) {
                std::tuple<float16_t *, float16_t *, uint32_t *> opOutTp(distResult[i][j].data(),
                                                                         minDistResult[i][j].data(), indices.data());
                std::tuple<float16_t *, uint32_t *, int> topkHeapTp(outDistances[j].data(), outIndices[j].data(), k);
                std::tuple<float16_t *, uint32_t *> minHeapTp(minDistances[j].data(), minIndices[j].data());

                if (i == 0) {
                    ASCEND_THROW_IF_NOT(topkOp.exec(opOutTp, topkHeapTp, minHeapTp, size, BURST_LEN));
                } else {
                    ASCEND_THROW_IF_NOT(topkOp.exec(opOutTp, topkHeapTp, size, offset, BURST_LEN));
                }
            }
            offset += distComputeBatch;
        }
    };

    int functorSize = (n > THREADS_CNT) ? THREADS_CNT : n;
    std::vector<std::future<void>> topkFunctorRet;
    for (int i = 0; i < functorSize; i++) {
        topkFunctorRet.emplace_back(threadPool->Enqueue(topkFunctor, i));
    }

    // 4. run the disance operator to compute the distance
    const int dim1 = utils::divUp(this->distComputeBatch, CUBE_ALIGN);
    const int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    for (int i = 0; i < repeatTimes; ++i) {
        AscendTensor<int8_t, DIMS_4> shaped(baseShaped[i]->data(),
            { dim1, dim2, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        AscendTensor<float16_t, DIMS_1> codesNorm(normBase[i]->data(), { distComputeBatch });
        auto dist = distResult[i].view();
        auto minDist = minDistResult[i].view();
        auto actualSize = opSize[i].view();
        auto flag = opFlag[i].view();

        int offset = i * this->distComputeBatch;
        int maskSize = static_cast<int>(utils::divUp(this->distComputeBatch, 8));
        AscendTensor<uint8_t, DIMS_2> mask(this->maskData + this->maskSearchedOffset, { n, maskSize });
        actualSize[0][IDX_ACTUAL_NUM] = 
            std::min(static_cast<uint32_t>(this->ntotal - offset), static_cast<uint32_t>(distComputeBatch));        
        actualSize[0][IDX_COMP_OFFSET] = offset;
        actualSize[0][IDX_MASK_LEN] = static_cast<int>(utils::divUp(this->ntotal, 8));      
        actualSize[0][IDX_USE_MASK] = (this->maskData != nullptr) ? 1 : 0;
        
        runDistCompute(queries, mask, shaped, queriesNorm, codesNorm, actualSize, dist, minDist, flag, stream);
    }

    // 5. wait all the op task compute, avoid thread dispatch
    aclrtSynchronizeStream(stream);

    // 6. waiting for topk functor to finish
    int topkWaitIdx = 0;
    try {
        for (auto &ret : topkFunctorRet) {
            topkWaitIdx++;
            ret.get();
        }
    } catch (std::exception &e) {
        errorQuit = true;
        for_each(topkFunctorRet.begin() + topkWaitIdx, topkFunctorRet.end(), [](auto &ret) { ret.wait(); });
        ASCEND_THROW_MSG(e.what());
    }

    topkOp.reorder(outDistances, outIndices);
}

void IndexInt8FlatCos::runDistCompute(AscendTensor<int8_t, DIMS_2> &queryVecs,
                                      AscendTensor<uint8_t, DIMS_2> &mask,
                                      AscendTensor<int8_t, DIMS_4> &shapedData,
                                      AscendTensor<float16_t, DIMS_1> &queriesNorm,
                                      AscendTensor<float16_t, DIMS_1> &codesNorm,
                                      AscendTensor<uint32_t, DIMS_2> &size,
                                      AscendTensor<float16_t, DIMS_2> &outDistances,
                                      AscendTensor<float16_t, DIMS_2> &outMinDistances,
                                      AscendTensor<uint16_t, DIMS_2> &flag,
                                      aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = queryVecs.getSize(0);
    if (distComputeOps.find(batch) != distComputeOps.end()) {
        op = distComputeOps[batch].get();
    }
    ASCEND_ASSERT(op);

    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(queryVecs.data(), queryVecs.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(mask.data(), mask.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(shapedData.data(), shapedData.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(queriesNorm.data(), queriesNorm.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(codesNorm.data(), codesNorm.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(size.data(), size.getSizeInBytes()));

    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(outMinDistances.data(), outMinDistances.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    op->exec(distOpInput, distOpOutput, stream);

    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();

    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
}

void IndexInt8FlatCos::resetDistCompOp(int codeNum)
{
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceInt8CosMaxs");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> maskShape({ batch, utils::divUp(codeNum, 8) });
        std::vector<int64_t> codeShape({ codeNum / CUBE_ALIGN, dims / CUBE_ALIGN_INT8, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        std::vector<int64_t> queriesNormShape({ (batch + FP16_ALGIN - 1) / FP16_ALGIN * FP16_ALGIN });
        std::vector<int64_t> codesNormShape({ codeNum });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN_SIZE });
        std::vector<int64_t> resultShape({ batch, codeNum });
        std::vector<int64_t> minResultShape({ batch, this->burstsOfComputeBatch });
        std::vector<int64_t> flagShape({ FLAG_NUM, FLAG_ALIGN_SIZE });

        desc.addInputTensorDesc(ACL_INT8, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, maskShape.size(), maskShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT8, codeShape.size(), codeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, queriesNormShape.size(), queriesNormShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, codesNormShape.size(), codesNormShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, resultShape.size(), resultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, minResultShape.size(), minResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = std::make_unique<AscendOperator>(desc);
    };

    for (auto batch : searchPageSizes) {
        distComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        distCompOpReset(distComputeOps[batch], batch);
    }
}
} // namespace ascend
