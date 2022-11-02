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

#include <ascenddaemon/impl/IndexInt8FlatL2.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/DeviceVectorInl.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/AscendThreadPool.h>
#include <ascenddaemon/utils/Limits.h>

namespace ascend {
namespace {
const double TIMEOUT_MS = 50000;
const int TIMEOUT_CHECK_TICK = 5120;
const int FLAG_ALIGN_SIZE = 16;
const int FLAG_ALIGN_OFFSET = 16; // core 0 use first 16 flag, and core 1 use the second 16 flag.
const int SIZE_ALIGN_SIZE = 8;
const int THREADS_CNT = 6;
const int BURST_LEN = 64;
const int CUBE_ALIGN = 16;
const int CUBE_ALIGN_INT8 = 32;
const int CORE_NUM = 2;
const int FLAG_NUM = 16;
const int IDX_ACTUAL_NUM = 0;
const int IDX_COMP_OFFSET = 1;
const int IDX_MASK_LEN = 2;
const int IDX_USE_MASK = 3;
}

IndexInt8FlatL2::IndexInt8FlatL2(int dim, int resourceSize)
    : IndexInt8Flat<int32_t>(dim, MetricType::METRIC_L2, resourceSize)
{
    // align by 2
    this->burstsOfComputeBatch = (this->distComputeBatch + BURST_LEN - 1) / BURST_LEN * 2;

    resetDistCompOp(distComputeBatch);
}

IndexInt8FlatL2::~IndexInt8FlatL2() {}

void IndexInt8FlatL2::addVectors(AscendTensor<int8_t, DIMS_2> &rawData)
{
    // 1. save code
    IndexInt8Flat::addVectors(rawData);

    // 2. compute the precompute data
    computeNorm(rawData);

    this->ntotal += rawData.getSize(0);
}

void IndexInt8FlatL2::searchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels)
{
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<int8_t, DIMS_2> queries(const_cast<int8_t *>(x), { n, dims });
    int nq = queries.getSize(0);

    int repeatTimes = utils::divUp(this->ntotal, this->distComputeBatch);
    AscendTensor<float16_t, DIMS_3> distResult(mem, { repeatTimes, nq, distComputeBatch }, stream);
    AscendTensor<float16_t, DIMS_3> minDistResult(mem, { repeatTimes, nq, this->burstsOfComputeBatch }, stream);
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { repeatTimes, CORE_NUM, SIZE_ALIGN_SIZE }, stream);
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { repeatTimes, FLAG_NUM, FLAG_ALIGN_SIZE }, stream);
    opFlag.zero();

    AscendTensor<float16_t, DIMS_2> minDistances(mem, { nq, k }, stream);
    AscendTensor<uint32_t, DIMS_2> minIndices(mem, { nq, k }, stream);
    minDistances.initValue(Limits<float16_t>::getMax());
    minIndices.initValue(std::numeric_limits<uint32_t>::max());

    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<idx_t, DIMS_2> outIndices(labels, { n, k });
    outDistances.initValue(Limits<float16_t>::getMax());
    outIndices.initValue(std::numeric_limits<idx_t>::max());

    // 1. run the topK operator to select the top K async
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
            for (int j = idx; j < nq; j += THREADS_CNT) {
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

    int functorSize = (nq > THREADS_CNT) ? THREADS_CNT : nq;
    std::vector<std::future<void>> topkFunctorRet;
    for (int i = 0; i < functorSize; i++) {
        topkFunctorRet.emplace_back(threadPool->Enqueue(topkFunctor, i));
    }

    // 2. run the disance operator to compute the distance
    const int dim1 = utils::divUp(this->distComputeBatch, CUBE_ALIGN);
    const int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    for (int i = 0; i < repeatTimes; ++i) {
        AscendTensor<int8_t, DIMS_4> shaped(baseShaped[i]->data(),
            { dim1, dim2, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        AscendTensor<int32_t, DIMS_1> norm(normBase[i]->data(), { distComputeBatch });
        auto dist = distResult[i].view();
        auto minDist = minDistResult[i].view();
        auto actualSize = opSize[i].view();
        auto flag = opFlag[i].view();

        int offset = i * this->distComputeBatch;
        int maskSize = static_cast<int>(utils::divUp(this->distComputeBatch, 8));  /* uint8 has 8 bits */
        AscendTensor<uint8_t, DIMS_2> mask(this->maskData + this->maskSearchedOffset, { n, maskSize });
        
        actualSize[0][IDX_ACTUAL_NUM] =
            std::min(static_cast<uint32_t>(this->ntotal - offset), static_cast<uint32_t>(this->distComputeBatch));
        actualSize[0][IDX_COMP_OFFSET] = offset;

        // uint8 have 8 bits
        actualSize[0][IDX_MASK_LEN] = static_cast<int>(utils::divUp(this->ntotal, 8));
        actualSize[0][IDX_USE_MASK] = (this->maskData != nullptr) ? 1 : 0;

        runDistCompute(queries, mask, shaped, norm, actualSize, dist, minDist, flag, stream);
    }

    // 3. wait all the op task compute, avoid thread dispatch
    aclrtSynchronizeStream(stream);

    // 4. waiting for topk functor to finish
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

void IndexInt8FlatL2::runDistCompute(AscendTensor<int8_t, DIMS_2> &queryVecs,
                                     AscendTensor<uint8_t, DIMS_2> &mask,
                                     AscendTensor<int8_t, DIMS_4> &shapedData,
                                     AscendTensor<int32_t, DIMS_1> &norms,
                                     AscendTensor<uint32_t, DIMS_2> &size,
                                     AscendTensor<float16_t, DIMS_2> &outDistances,
                                     AscendTensor<float16_t, DIMS_2> &minDistances,
                                     AscendTensor<uint16_t, DIMS_2> &flag, aclrtStream stream)
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
    distOpInput.emplace_back(aclCreateDataBuffer(norms.data(), norms.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(size.data(), size.getSizeInBytes()));

    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(minDistances.data(), minDistances.getSizeInBytes()));
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

void IndexInt8FlatL2::resetDistCompOp(int codeNum)
{
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceInt8L2Mins");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> maskShape({ batch, utils::divUp(codeNum, 8) });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(codeNum, CUBE_ALIGN),
            utils::divUp(dims, CUBE_ALIGN_INT8), CUBE_ALIGN, (int64_t)CUBE_ALIGN_INT8 });
        std::vector<int64_t> preNormsShape({ codeNum });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN_SIZE });
        std::vector<int64_t> distResultShape({ batch, codeNum });
        std::vector<int64_t> minResultShape({ batch, this->burstsOfComputeBatch });
        std::vector<int64_t> flagShape({ FLAG_NUM, FLAG_ALIGN_SIZE });

        desc.addInputTensorDesc(ACL_INT8, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, maskShape.size(), maskShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT8, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, preNormsShape.size(), preNormsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
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
