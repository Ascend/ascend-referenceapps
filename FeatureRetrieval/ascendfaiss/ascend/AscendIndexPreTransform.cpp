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

#include <faiss/IndexPreTransform.h>
#include <faiss/ascend/AscendIndexPreTransform.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendVTCloner.h>
#include <faiss/ascend/utils/AscendThreadPool.h>
#include <faiss/ascend/utils/fp16.h>
#include <faiss/ascend/utils/AscendUtils.h>
#include <faiss/ascend/rpc/AscendRpc.h>

namespace faiss {
namespace ascend {
AscendIndexPreTransform::AscendIndexPreTransform(const faiss::IndexPreTransform *index, AscendIndexConfig config)
    : AscendIndex(index->d, index->metric_type, config), ownFields(index->own_fields), index(nullptr)
{
    FAISS_THROW_IF_NOT_MSG(config.deviceList.size(), "device list can not be empty");
    copyFrom(index);
}

AscendIndexPreTransform::AscendIndexPreTransform(AscendIndex *index)
    : AscendIndex(index->d, index->metric_type, index->indexConfig), ownFields(false), index(index)
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    is_trained = index->is_trained;
    ntotal = index->ntotal;
    initRpcCtx();
}

AscendIndexPreTransform::~AscendIndexPreTransform()
{
    for (size_t i = 0; i < chain.size(); i++) {
        delete chain[i];
    }
    chain.clear();

    if (index && ownFields) {
        delete index;
    }

    clearRpcCtx();
    indexMap.clear();
    contextMap.clear();
}

void AscendIndexPreTransform::copyFrom(const faiss::IndexPreTransform *index)
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    this->d = index->d;
    this->metric_type = index->metric_type;
    this->is_trained = index->is_trained;
    this->ntotal = index->ntotal;

    if (this->index && this->ownFields) {
        delete this->index;
    }
    for (size_t i = 0; i < chain.size(); i++) {
        delete this->chain[i];
    }
    this->chain.clear();

    this->index = dynamic_cast<faiss::ascend::AscendIndex *>(index_cpu_to_ascend(indexConfig.deviceList, index->index));
    FAISS_THROW_IF_NOT_MSG(this->index != nullptr, "index is nullptr.");

    this->ownFields = true;

    clearRpcCtx();
    initRpcCtx();
    for (int i = index->chain.size() - 1; i >= 0; i--) {
        AscendVectorTransform *vtrans = dynamic_cast<faiss::ascend::AscendVectorTransform *>(
            vtrans_cpu_to_ascend(indexConfig.deviceList, index->chain[i]));
        this->chain.insert(this->chain.begin(), vtrans);
        prependRpcChain(vtrans);
    }
}

void AscendIndexPreTransform::copyTo(faiss::IndexPreTransform *index) const
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    index->d = this->d;
    index->metric_type = this->metric_type;
    index->is_trained = this->is_trained;
    index->ntotal = this->ntotal;

    if (index->index && index->own_fields) {
        delete index->index;
    }

    index->index = index_ascend_to_cpu(this->index);
    for (size_t i = 0; i < this->chain.size(); i++) {
        VectorTransform *vtrans = vtrans_ascend_to_cpu(this->chain[i]);
        index->chain.push_back(vtrans);
    }
    index->own_fields = true;
}

void AscendIndexPreTransform::reserveMemory(size_t numVecs)
{
    index->reserveMemory(numVecs);
}

size_t AscendIndexPreTransform::reclaimMemory()
{
    return index->reclaimMemory();
}

void AscendIndexPreTransform::initRpcCtx()
{
    indexMap.clear();
    contextMap.clear();
    for (size_t i = 0; i < indexConfig.deviceList.size(); i++) {
        int indexId;
        rpcContext ctx;
        int deviceId = indexConfig.deviceList[i];

        RpcError ret = RpcCreateContext(deviceId, &ctx);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create context failed(%d).", ret);
        contextMap[deviceId] = ctx;

        ret = RpcCreateIndexPreTransform(ctx, indexId, index->indexMap[ctx]);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create IndexIVFPreTransform failed(%d).", ret);
        indexMap[ctx] = indexId;
    }
}

void AscendIndexPreTransform::clearRpcCtx()
{
    for (auto &index : indexMap) {
        RpcError ret = RpcDestroyIndexPreTransform(index.first, index.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy IndexIVFPreTransform failed(%d).", ret);

        ret = RpcDestroyContext(index.first);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy context failed(%d).", ret);
    }
    indexMap.clear();
    contextMap.clear();
}

void AscendIndexPreTransform::prependRpcChain(AscendVectorTransform *ltrans)
{
    FAISS_THROW_IF_NOT_MSG(ltrans != nullptr, "ltrans is nullptr.");
    for (auto &index : indexMap) {
        int vtransID = ltrans->vtransMap.at(index.first);
        RpcError ret = RpcIndexPreTransformPrepend(index.first, index.second, vtransID);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Prepend implement failed(%d).", ret);
    }
}

void AscendIndexPreTransform::train(idx_t n, const float *x)
{
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n should be a positive value.");
    size_t lastUntrained = 0;
    if (!index->is_trained) {
        lastUntrained = chain.size();
    } else {
        for (int i = chain.size() - 1; i >= 0; i--) {
            if (!chain[i]->is_trained) {
                lastUntrained = i;
                break;
            }
        }
    }

    if (verbose) {
        printf("AscendIndexPreTransform::train: training chain 0 to %zu\n", lastUntrained);
    }
    const float *prevX = x;
    faiss::ScopeDeleter<float> del;
    for (size_t i = 0; i <= lastUntrained; i++) {
        if (i < chain.size()) {
            AscendVectorTransform *ltrans = chain[i];
            if (!ltrans->is_trained) {
                if (verbose) {
                    printf("   Training chain component %zu/%zu\n", i, chain.size());
                }
                ltrans->train(n, prevX);
            }
        } else {
            if (verbose) {
                printf("   Training sub-index\n");
            }
            index->train(n, prevX);
        }
        if (i == lastUntrained) {
            break;
        }
        if (verbose) {
            printf("   Applying transform %zu/%zu\n", i, chain.size());
        }
        float *xt = chain[i]->apply(n, prevX);
        if (prevX != x) {
            delete[] prevX;
        }
        prevX = xt;
        del.set(xt);
    }
    is_trained = true;
}


const float *AscendIndexPreTransform::applyChain(idx_t n, const float *x) const
{
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n should be a positive value.");
    const float *prevX = x;
    faiss::ScopeDeleter<float> del;

    for (size_t i = 0; i < chain.size(); i++) {
        float *xt = chain[i]->apply(n, prevX);
        faiss::ScopeDeleter<float> del2(xt);
        del2.swap(del);
        prevX = xt;
    }
    del.release();
    return prevX;
}

void AscendIndexPreTransform::reset()
{
    index->reset();
    ntotal = 0;
}

void AscendIndexPreTransform::searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
    std::vector<std::vector<uint32_t>> &label, int n, int k, float *distances, Index::idx_t *labels) const
{
    index->searchPostProcess(devices, dist, label, n, k, distances, labels);
}

void AscendIndexPreTransform::addImpl(int n, const float *x, const Index::idx_t *ids)
{
    const float *xt = applyChain(n, x);
    faiss::ScopeDeleter<float> del((xt == x) ? nullptr : xt);
    index->add_with_ids(n, xt, ids);
    ntotal = index->ntotal;
}

size_t AscendIndexPreTransform::removeImpl(const IDSelector &sel)
{
    size_t nremove = index->remove_ids(sel);
    ntotal = index->ntotal;
    return nremove;
}

bool AscendIndexPreTransform::addImplRequiresIDs() const
{
    return index->addImplRequiresIDs();
}

int AscendIndexPreTransform::getElementSize() const
{
    return index->getElementSize();
}
}
}
