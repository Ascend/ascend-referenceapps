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


#include <ascenddaemon/impl/IndexFlatL2.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/DeviceVectorInl.h>
#include <ascenddaemon/utils/Limits.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/AscendThreadPool.h>

namespace ascend {
namespace {
const double TIMEOUT_MS = 50000;
const int TIMEOUT_CHECK_TICK = 5120;
const int FLAG_ALIGN_SIZE = 32;
const int FLAG_ALIGN_OFFSET = 16; // core 0 use first 16 flag, and core 1 use the second 16 flag.
const int SIZE_ALIGN_SIZE = 8;
const int THREADS_CNT = 6;
const int BURST_LEN = 64;
}

IndexFlatL2::IndexFlatL2(int dim, int resourceSize) : IndexFlat(dim, resourceSize)
{
    ASCEND_THROW_IF_NOT(dims % CUBE_ALIGN_SIZE == 0);
    resetDistCompOp(distComputeBatch);
}

IndexFlatL2::~IndexFlatL2() {}

void IndexFlatL2::reset()
{
    int dvSize = utils::divUp(this->ntotal, this->distComputeBatch);
    for (int i = 0; i < dvSize; ++i) {
        normBase.at(i)->clear();
    }

    IndexFlat::reset();
}

void IndexFlatL2::addVectors(AscendTensor<float16_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);
    int dim = rawData.getSize(1);
    if (num == 0) {
        return;
    }

    // 1. save the rawData to shaped data
    IndexFlat::addVectors(rawData);

    // 2. resize normBase
    int vecSize = utils::divUp(this->ntotal, this->distComputeBatch);
    int addVecNum = utils::divUp(ntotal + num, distComputeBatch) - vecSize;
    for (int i = 0; i < addVecNum; ++i) {
        this->normBase.emplace_back(std::make_unique<DeviceVector<float16_t>>());
        this->normBase.at(vecSize + i)->resize(this->distComputeBatch, true);
    }

    // 3. compute the norm data
    bool isFirst = true;
    int idx = 0;
    for (int i = 0; i < num; i++) {
        int idx1 = (ntotal + i) / distComputeBatch;
        int idx2 = (ntotal + i) % distComputeBatch;

        // if the baseShapedSlice is full or reach the last
        if (idx2 + 1 == distComputeBatch || i == num - 1) {
            float16_t *pNormBaseSlice = normBase[idx1]->data();

            // calc y^2 (the first time is different)
            if (isFirst) {
                fvecNormsL2sqr(pNormBaseSlice + ntotal % distComputeBatch, rawData[idx][0].data(), dim, i + 1);
                idx += (i + 1);
                isFirst = false;
            } else {
                fvecNormsL2sqr(pNormBaseSlice, rawData[idx][0].data(), dim, idx2 + 1);
                idx += (idx2 + 1);
            }
        }
    }

    this->ntotal += num;
}

void IndexFlatL2::moveNormForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_ASSERT(srcIdx >= dstIdx);
    int srcIdx1 = srcIdx / this->distComputeBatch;
    int srcIdx2 = srcIdx % this->distComputeBatch;
    int dstIdx1 = dstIdx / this->distComputeBatch;
    int dstIdx2 = dstIdx % this->distComputeBatch;

    (*normBase[dstIdx1])[dstIdx2] = (*normBase[srcIdx1])[srcIdx2];
}

void IndexFlatL2::removeInvalidData(int oldTotal, int remove)
{
    IndexFlat::removeInvalidData(oldTotal, remove);

    int oldVecSize = utils::divUp(oldTotal, this->distComputeBatch);
    int vecSize = utils::divUp(oldTotal - remove, this->distComputeBatch);

    for (int i = oldVecSize - 1; i >= vecSize; --i) {
        this->normBase.at(i)->clear();
    }
}

void IndexFlatL2::searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
    AscendTensor<float16_t, DIMS_2> &outDistances, AscendTensor<uint32_t, DIMS_2> &outIndices)
{
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    int nq = queries.getSize(0);

    outDistances.initValue(Limits<float16_t>::getMax());

    int repeatTimes = utils::divUp(this->ntotal, this->distComputeBatch);
    AscendTensor<float16_t, DIMS_3> distResult(mem, { repeatTimes, nq, distComputeBatch }, stream);
    AscendTensor<float16_t, DIMS_3> minDistResult(mem, { repeatTimes, nq, this->burstsOfComputeBatch }, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, { repeatTimes, SIZE_ALIGN_SIZE }, stream);
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { repeatTimes, FLAG_ALIGN_SIZE }, stream);
    opFlag.zero();

    AscendTensor<float16_t, DIMS_2> minDistances(mem, { nq, k }, stream);
    AscendTensor<uint32_t, DIMS_2> minIndices(mem, { nq, k }, stream);
    minDistances.initValue(Limits<float16_t>::getMax());
    minIndices.initValue(std::numeric_limits<uint32_t>::max());

    // 1. run the topK operator to select the top K async
    bool errorQuit = false;
    auto topkFunctor = [&](int idx) {
        if (idx < THREADS_CNT) {
            AscendUtils::attachToCpu(idx);
        }

        AscendTensor<uint32_t, DIMS_1> indices;
        uint32_t offset = 0;
        for (int i = 0; i < repeatTimes && !errorQuit; ++i) {
            uint16_t *volatile flagPtr1 = opFlag[i].data();
            uint16_t *volatile flagPtr2 = opFlag[i].data() + FLAG_ALIGN_OFFSET;

            WAITING_FLAG_READY((*flagPtr1) && (*flagPtr2), TIMEOUT_CHECK_TICK, TIMEOUT_MS);

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
    const int dim1 = utils::divUp(this->distComputeBatch, CUBE_ALIGN_SIZE);
    const int dim2 = utils::divUp(this->dims, CUBE_ALIGN_SIZE);
    for (int i = 0; i < repeatTimes; ++i) {
        AscendTensor<float16_t, DIMS_4> shaped(baseShaped[i]->data(), { dim1, dim2, CUBE_ALIGN_SIZE, CUBE_ALIGN_SIZE });
        AscendTensor<float16_t, DIMS_1> norm(normBase[i]->data(), { distComputeBatch });
        auto dist = distResult[i].view();
        auto minDist = minDistResult[i].view();
        auto actualSize = opSize[i].view();
        auto flag = opFlag[i].view();

        int offset = i * this->distComputeBatch;
        actualSize[0] = std::min(static_cast<uint32_t>(this->ntotal - offset), static_cast<uint32_t>(distComputeBatch));

        runDistCompute(queries, shaped, norm, actualSize, dist, minDist, flag, stream);
    }

    // 3. wait all the op task compute, avoid thread dispatch
    aclrtSynchronizeStream(stream);

    // 4. waiting for topk functor to finish
    try {
        for (auto &ret : topkFunctorRet) {
            ret.get();
        }
    } catch (std::exception &e) {
        errorQuit = true;
        ASCEND_THROW_MSG(e.what());
    }

    topkOp.reorder(outDistances, outIndices);
}

void IndexFlatL2::runDistCompute(AscendTensor<float16_t, DIMS_2> &queryVecs,
    AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<float16_t, DIMS_1> &norms,
    AscendTensor<uint32_t, DIMS_1> &size, AscendTensor<float16_t, DIMS_2> &outDistances,
    AscendTensor<float16_t, DIMS_2> &minDistances, AscendTensor<uint16_t, DIMS_1> &flag, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = queryVecs.getSize(0);
    if (distComputeOps.find(batch) != distComputeOps.end()) {
        op = distComputeOps[batch].get();
    }
    ASCEND_ASSERT(op);

    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(queryVecs.data(), queryVecs.getSizeInBytes()));
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

void IndexFlatL2::resetDistCompOp(int numLists)
{
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceComputeFlatMin64");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(numLists, CUBE_ALIGN_SIZE),
            utils::divUp(dims, CUBE_ALIGN_SIZE), CUBE_ALIGN_SIZE, (int64_t)CUBE_ALIGN_SIZE });
        std::vector<int64_t> preNormsShape({ numLists });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN_SIZE });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> minResultShape({ batch, this->burstsOfComputeBatch });
        std::vector<int64_t> flagShape({ FLAG_ALIGN_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, preNormsShape.size(), preNormsShape.data(), ACL_FORMAT_ND);
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

size_t IndexFlatL2::calcNormBaseSize(idx_t totalNum)
{
    int numBatch = utils::divUp(totalNum, distComputeBatch);
    return numBatch * distComputeBatch;
}
} // namespace ascend
