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

#include <faiss/ascend/AscendVectorTransform.h>
#include <faiss/ascend/utils/fp16.h>
#include <faiss/ascend/rpc/AscendRpc.h>
#include <algorithm>
#include <iostream>

namespace faiss {
namespace ascend {
namespace {
const int DIM_ALIGN_SIZE = 16;
}

AscendVectorTransform::AscendVectorTransform(int dimIn, int dimOut, AscendVectorTransformConfig config)
    : faiss::VectorTransform(dimIn, dimOut), vtransConfig(config)
{}

AscendVectorTransform::~AscendVectorTransform() {}


AscendLinearTransform::AscendLinearTransform(int dimIn, int dimOut, bool haveBias, AscendVectorTransformConfig config)
    : AscendVectorTransform(dimIn, dimOut, config), cpuLtrans(nullptr), haveBias(haveBias)
{
    FAISS_THROW_IF_NOT_MSG(dimIn > 0, "dimIn should be positive.");
    FAISS_THROW_IF_NOT_MSG(dimOut > 0, "dimOut should be positive.");
    // restriction of D micro-architecture
    FAISS_THROW_IF_NOT_MSG(dimIn % DIM_ALIGN_SIZE == 0, "dimIn should be divisible by 16.");
    FAISS_THROW_IF_NOT_MSG(dimOut % DIM_ALIGN_SIZE == 0, "dimOut should be divisible by 16.");
    is_trained = false; // will be trained when A and b are initialized
    createLinearTransform();
}

void AscendLinearTransform::copyFrom(const faiss::LinearTransform *cpuTrans)
{
    FAISS_THROW_IF_NOT_MSG(cpuTrans != nullptr, "cpuTrans is nullptr.");
    this->is_trained = cpuTrans->is_trained;
    this->cpuLtrans->is_trained = cpuTrans->is_trained;
    this->haveBias = cpuTrans->have_bias;
    this->cpuLtrans->have_bias = cpuTrans->have_bias;
    if (cpuTrans->is_trained) {
        this->A = cpuTrans->A;
        this->cpuLtrans->A = cpuTrans->A;
        if (cpuTrans->have_bias) {
            this->b = cpuTrans->b;
            this->cpuLtrans->b = cpuTrans->b;
        } else {
            this->b = std::vector<float>(this->d_out, 0);
        }
        updateMatrixAndBias();
    }
}

void AscendLinearTransform::copyTo(faiss::LinearTransform *cpuTrans) const
{
    FAISS_THROW_IF_NOT_MSG(cpuTrans != nullptr, "cpuTrans is nullptr.");
    cpuTrans->is_trained = this->is_trained;
    cpuTrans->have_bias = this->haveBias;
    if (this->is_trained) {
        cpuTrans->A = A;
        if (haveBias) {
            cpuTrans->b = b;
        }
    }
}

void AscendLinearTransform::train(idx_t n, const float *x)
{
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n should be a positive value.");
    if (cpuLtrans == nullptr) {
        return;
    }
    cpuLtrans->train(n, x);
    A = cpuLtrans->A;
    if (haveBias) {
        b = cpuLtrans->b;
    } else {
        b = std::vector<float>(this->d_out, 0);
    }
    is_trained = true;
    updateMatrixAndBias();
}

void AscendLinearTransform::apply_noalloc(idx_t n, const float *x, float *xt) const
{
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n should be a positive value.");
    FAISS_THROW_IF_NOT_MSG(cpuLtrans, "cpuLtrans can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(is_trained, "is_trained should be true.");
    return cpuLtrans->apply_noalloc(n, x, xt);
}

AscendLinearTransform::~AscendLinearTransform()
{
    destroyLinearTransform();
}

void AscendLinearTransform::createLinearTransform()
{
    vtransMap.clear();
    contextMap.clear();
    for (size_t i = 0; i < vtransConfig.deviceList.size(); i++) {
        int vtransID;
        rpcContext ctx;
        int deviceId = vtransConfig.deviceList[i];

        RpcError ret = RpcCreateContext(deviceId, &ctx);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create context failed(%d).", ret);
        contextMap[deviceId] = ctx;

        ret = RpcCreateLinearTransform(ctx, vtransID, this->d_in, this->d_out, this->haveBias);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Create LinearTransform failed(%d).", ret);
        vtransMap[ctx] = vtransID;
    }
}

void AscendLinearTransform::destroyLinearTransform()
{
    for (auto &vtrans : vtransMap) {
        RpcError ret = RpcDestroyLinearTransform(vtrans.first, vtrans.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy LinearTransform failed(%d).", ret);

        ret = RpcDestroyContext(vtrans.first);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Destroy context failed(%d).", ret);
    }

    vtransMap.clear();
    contextMap.clear();
    if (cpuLtrans != nullptr) {
        delete cpuLtrans;
        cpuLtrans = nullptr;
    }
}

void AscendLinearTransform::updateMatrixAndBias()
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "is_trained should be true.");
    std::vector<uint16_t> AFp16(A.size());
    std::transform(begin(A), end(A), begin(AFp16), [](float temp) { return fp16(temp).data; });
    for (auto &vtrans : vtransMap) {
        RpcError ret = RpcLinearTransformUpdateTrainedValue(vtrans.first, vtrans.second, this->d_in, this->d_out,
            AFp16.data(), this->b.data());
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Update A & b failed(%d).", ret);
    }
}

AscendPCAMatrix::AscendPCAMatrix(int dimIn, int dimOut, float eigenPower, bool randomRotation,
    std::initializer_list<int> devices)
    : AscendLinearTransform(dimIn, dimOut, true, AscendVectorTransformConfig(devices)),
      eigenPower(eigenPower),
      randomRotation(randomRotation)
{
    FAISS_THROW_IF_NOT_MSG(dimIn >= dimOut, "dimIn should be >= dimOut.");
    cpuLtrans = new faiss::PCAMatrix(dimIn, dimOut, eigenPower, randomRotation);
}

AscendPCAMatrix::AscendPCAMatrix(int dimIn, int dimOut, float eigenPower, bool randomRotation,
    AscendVectorTransformConfig config)
    : AscendLinearTransform(dimIn, dimOut, true, config), eigenPower(eigenPower), randomRotation(randomRotation)
{
    FAISS_THROW_IF_NOT_MSG(dimIn >= dimOut, "dimIn should be >= dimOut.");
    cpuLtrans = new faiss::PCAMatrix(dimIn, dimOut, eigenPower, randomRotation);
}
}
}
