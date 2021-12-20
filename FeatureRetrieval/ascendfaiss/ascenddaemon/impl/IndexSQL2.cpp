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

#include <ascenddaemon/impl/IndexSQL2.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/Limits.h>
#include <ascenddaemon/utils/StaticUtils.h>

namespace ascend {
namespace {
const int SQ8_DIST_COMPUTE_BATCH = 16384 * 16;
const int TIMEOUT_CHECK_TICK = 5120;
const double TIMEOUT_MS = 50000;
const int FLAG_ALIGN_SIZE = 32;
const int FLAG_ALIGN_OFFSET = 16; // core 0 use first 16 flag, and core 1 use the second 16 flag.
const int SIZE_ALIGN_SIZE = 8;
const int THREADS_CNT = 6;
const int BURST_LEN = 64;
}

IndexSQL2::IndexSQL2(int dim, int resourceSize) : IndexSQ(dim, resourceSize)
{
    resetSqDistOperator();
}

IndexSQL2::~IndexSQL2() {}

void IndexSQL2::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
{
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    int batchNum = utils::divUp(this->ntotal, this->distComputeBatch);
    ASCEND_ASSERT(static_cast<int>(this->codes.size()) == batchNum);

    bool errorQuit = false;

    // 1. costruct the SQ distance operator param
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, dims });
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { batchNum, (n * FLAG_ALIGN_SIZE) }, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, { batchNum, (n * SIZE_ALIGN_SIZE) }, stream);
    AscendTensor<float16_t, DIMS_3> distResult(mem, { batchNum, n, this->distComputeBatch }, stream);
    AscendTensor<float16_t, DIMS_3> minDistResult(mem, { batchNum, n, this->burstsOfComputeBatch }, stream);
    opFlag.zero();

    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<uint32_t, DIMS_2> outIndices(labels, { n, k });
    outDistances.initValue(Limits<float16_t>::getMax());
    outIndices.initValue(std::numeric_limits<uint32_t>::max());

    AscendTensor<float16_t, DIMS_2> minDistances(mem, { n, k }, stream);
    AscendTensor<uint32_t, DIMS_2> minIndices(mem, { n, k }, stream);
    minDistances.initValue(Limits<float16_t>::getMax());
    minIndices.initValue(std::numeric_limits<uint32_t>::max());

    // 2. run the topK operator to select the top K async
    auto topkFunctor = [&](int idx) {
        if (idx < THREADS_CNT) {
            AscendUtils::attachToCpu(idx);
        }
        AscendTensor<uint32_t, DIMS_1> indices;
        uint32_t offset = 0;
        for (int i = 0; i < batchNum && !errorQuit; ++i) {
            uint16_t *volatile flagPtr1 = opFlag[i].data();
            uint16_t *volatile flagPtr2 = opFlag[i].data() + FLAG_ALIGN_OFFSET;

            // waitting for the operator finishing running, while the flags will be
            // setted by operator when finishing running.
            WAITING_FLAG_READY((*flagPtr1 && *(flagPtr2)), TIMEOUT_CHECK_TICK, TIMEOUT_MS);

            uint32_t size = opSize[i][0];
            for (int j = idx; j < n; j += THREADS_CNT) {
                std::tuple<float16_t *, float16_t *, uint32_t *> opOutTp(distResult[i][j].data(),
                    minDistResult[i][j].data(), indices.data());
                std::tuple<float16_t *, uint32_t *, int> topkHeapTp(outDistances[j].data(), outIndices[j].data(), k);
                std::tuple<float16_t *, uint32_t *> minHeapTp(minDistances[j].data(), minIndices[j].data());
                if (i == 0) {
                    ASCEND_THROW_IF_NOT(topkMinOp.exec(opOutTp, topkHeapTp, minHeapTp, size, BURST_LEN));
                } else {
                    ASCEND_THROW_IF_NOT(topkMinOp.exec(opOutTp, topkHeapTp, size, offset, BURST_LEN));
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

    // 3. run the disance operator to compute the distance
    for (int i = 0; i < batchNum; ++i) {
        AscendTensor<uint8_t, DIMS_4> batchCode(this->codes[i]->data(), { this->distComputeBatch / CUBE_ALIGN_SIZE,
            this->dims / CUBE_ALIGN_SIZE, CUBE_ALIGN_SIZE, CUBE_ALIGN_SIZE });
        AscendTensor<float, DIMS_1> batchPreComp(this->preCompute[i]->data(), { this->distComputeBatch });
        AscendTensor<uint16_t, DIMS_1> flag = opFlag[i].view();
        AscendTensor<uint32_t, DIMS_1> actualSize = opSize[i].view();
        AscendTensor<float16_t, DIMS_2> result = distResult[i].view();
        AscendTensor<float16_t, DIMS_2> minResult = minDistResult[i].view();

        int offset = i * this->distComputeBatch;
        actualSize[0] =
            std::min(static_cast<uint32_t>(this->ntotal - offset), static_cast<uint32_t>(this->distComputeBatch));

        runSqDistOperator(queries, batchCode, batchPreComp, this->vDiff, this->vMin, actualSize, result, minResult,
            flag, stream);
    }

    // 4. wait all the op task compute, avoid thread dispatch
    aclrtSynchronizeStream(stream);

    // 5. waiting for topk functor to finish
    try {
        for (auto &ret : topkFunctorRet) {
            ret.get();
        }
    } catch (std::exception &e) {
        errorQuit = true;
        ASCEND_THROW_MSG(e.what());
    }

    // 5. reorder the topk results in ascend order
    topkMinOp.reorder(outDistances, outIndices);
}

void IndexSQL2::runSqDistOperator(AscendTensor<float16_t, DIMS_2> &queries, AscendTensor<uint8_t, DIMS_4> &codes,
    AscendTensor<float, DIMS_1> &precomp, AscendTensor<float16_t, DIMS_1> &vdiff, AscendTensor<float16_t, DIMS_1> &vmin,
    AscendTensor<uint32_t, DIMS_1> &size, AscendTensor<float16_t, DIMS_2> &result,
    AscendTensor<float16_t, DIMS_2> &minResult, AscendTensor<uint16_t, DIMS_1> &flag, aclrtStream stream)
{
    AscendOperator *distSqOp = nullptr;
    int batch = queries.getSize(0);
    if (distComputeOps.find(batch) != distComputeOps.end()) {
        distSqOp = distComputeOps[batch].get();
    }
    ASCEND_ASSERT(distSqOp);

    std::vector<const aclDataBuffer *> distSqOpInput;
    distSqOpInput.emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    distSqOpInput.emplace_back(aclCreateDataBuffer(codes.data(), codes.getSizeInBytes()));
    distSqOpInput.emplace_back(aclCreateDataBuffer(precomp.data(), precomp.getSizeInBytes()));
    distSqOpInput.emplace_back(aclCreateDataBuffer(vdiff.data(), vdiff.getSizeInBytes()));
    distSqOpInput.emplace_back(aclCreateDataBuffer(vmin.data(), vmin.getSizeInBytes()));
    distSqOpInput.emplace_back(aclCreateDataBuffer(size.data(), size.getSizeInBytes()));

    std::vector<aclDataBuffer *> distSqOpOutput;
    distSqOpOutput.emplace_back(aclCreateDataBuffer(result.data(), result.getSizeInBytes()));
    distSqOpOutput.emplace_back(aclCreateDataBuffer(minResult.data(), minResult.getSizeInBytes()));
    distSqOpOutput.emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    distSqOp->exec(distSqOpInput, distSqOpOutput, stream);

    for (auto &item : distSqOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distSqOpInput.clear();

    for (auto &item : distSqOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distSqOpOutput.clear();
}

void IndexSQL2::resetSqDistOperator()
{
    auto sqDistOpReset = [&](std::unique_ptr<AscendOperator> &distSqOp, int batch) {
        distSqOp.reset();
        AscendOpDesc desc("DistanceComputeSQ8Min64");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> codeShape({ this->distComputeBatch / CUBE_ALIGN_SIZE, this->dims / CUBE_ALIGN_SIZE,
            CUBE_ALIGN_SIZE, CUBE_ALIGN_SIZE });
        std::vector<int64_t> precompShape({ this->distComputeBatch });
        std::vector<int64_t> vdiffShape({ this->dims });
        std::vector<int64_t> vminShape({ this->dims });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN_SIZE });
        std::vector<int64_t> resultShape({ batch, this->distComputeBatch });
        std::vector<int64_t> minResultShape({ batch, this->burstsOfComputeBatch });
        std::vector<int64_t> flagShape({ FLAG_ALIGN_SIZE });

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
    };

    for (auto batch : searchPageSizes) {
        distComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        sqDistOpReset(distComputeOps[batch], batch);
    }
}
} // ascend