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


#include <ascenddaemon/impl/IndexFlatIP.h>
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

IndexFlatIP::IndexFlatIP(int dim, int resourceSize) : IndexFlat(dim, resourceSize)
{
    ASCEND_THROW_IF_NOT(dims % CUBE_ALIGN_SIZE == 0);
    resetDistCompOp(distComputeBatch);
}

IndexFlatIP::~IndexFlatIP() {}

void IndexFlatIP::addVectors(AscendTensor<float16_t, DIMS_2> &rawData)
{
    IndexFlat::addVectors(rawData);

    int num = rawData.getSize(0);
    this->ntotal += num;
}

void IndexFlatIP::searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
    AscendTensor<float16_t, DIMS_2> &outDistances, AscendTensor<uint32_t, DIMS_2> &outIndices)
{
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    int nq = queries.getSize(0);

    outDistances.initValue(Limits<float16_t>::getMin());

    int repeatTimes = utils::divUp(this->ntotal, this->distComputeBatch);
    AscendTensor<float16_t, DIMS_3> distResult(mem, { repeatTimes, nq, distComputeBatch }, stream);
    AscendTensor<float16_t, DIMS_3> maxDistResult(mem, { repeatTimes, nq, this->burstsOfComputeBatch }, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, { repeatTimes, SIZE_ALIGN_SIZE }, stream);
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { repeatTimes, FLAG_ALIGN_SIZE }, stream);
    opFlag.zero();

    AscendTensor<float16_t, DIMS_2> maxDistances(mem, { nq, k }, stream);
    AscendTensor<uint32_t, DIMS_2> maxIndices(mem, { nq, k }, stream);
    maxDistances.initValue(Limits<float16_t>::getMin());
    maxIndices.initValue(std::numeric_limits<uint32_t>::max());

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
                    maxDistResult[i][j].data(), indices.data());
                std::tuple<float16_t *, uint32_t *, int> topkHeapTp(outDistances[j].data(), outIndices[j].data(), k);
                std::tuple<float16_t *, uint32_t *> maxHeapTp(maxDistances[j].data(), maxIndices[j].data());

                if (i == 0) {
                    ASCEND_THROW_IF_NOT(topkMaxOp.exec(opOutTp, topkHeapTp, maxHeapTp, size, BURST_LEN));
                } else {
                    ASCEND_THROW_IF_NOT(topkMaxOp.exec(opOutTp, topkHeapTp, size, offset, BURST_LEN));
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
        auto dist = distResult[i].view();
        auto maxDist = maxDistResult[i].view();
        auto actualSize = opSize[i].view();
        auto flag = opFlag[i].view();

        int offset = i * this->distComputeBatch;
        actualSize[0] = std::min(static_cast<uint32_t>(this->ntotal - offset), static_cast<uint32_t>(distComputeBatch));

        runDistCompute(queries, shaped, actualSize, dist, maxDist, flag, stream);
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

    topkMaxOp.reorder(outDistances, outIndices);
}

void IndexFlatIP::runDistCompute(AscendTensor<float16_t, DIMS_2> &queryVecs,
                                 AscendTensor<float16_t, DIMS_4> &shapedData,
                                 AscendTensor<uint32_t, DIMS_1> &size,
                                 AscendTensor<float16_t, DIMS_2> &outDistances,
                                 AscendTensor<float16_t, DIMS_2> &maxDistances,
                                 AscendTensor<uint16_t, DIMS_1> &flag, aclrtStream stream)
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
    distOpInput.emplace_back(aclCreateDataBuffer(size.data(), size.getSizeInBytes()));

    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(maxDistances.data(), maxDistances.getSizeInBytes()));
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

void IndexFlatIP::resetDistCompOp(int numLists)
{
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIPMaxs");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(numLists, CUBE_ALIGN_SIZE),
            utils::divUp(dims, CUBE_ALIGN_SIZE), CUBE_ALIGN_SIZE, (int64_t)CUBE_ALIGN_SIZE });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN_SIZE });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> maxResultShape({ batch, this->burstsOfComputeBatch });
        std::vector<int64_t> flagShape({ FLAG_ALIGN_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, maxResultShape.size(), maxResultShape.data(), ACL_FORMAT_ND);
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
