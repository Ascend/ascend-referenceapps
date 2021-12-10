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

#ifndef ASCEND_VECTORTRANSFORM_INCLUDED
#define ASCEND_VECTORTRANSFORM_INCLUDED

#include <vector>
#include <stdint.h>
#include <faiss/VectorTransform.h>
#include <faiss/ascend/AscendIndex.h>

namespace faiss {
struct VectorTransform;
}

namespace faiss {
namespace ascend {
struct AscendVectorTransformConfig {
    inline AscendVectorTransformConfig() : deviceList({ 0 }) {}

    inline AscendVectorTransformConfig(std::initializer_list<int> devices) : deviceList(devices)
    {
        FAISS_THROW_IF_NOT_MSG(deviceList.size() != 0, "device list can not be empty!");
    }

    inline AscendVectorTransformConfig(std::vector<int> devices) : deviceList(devices)
    {
        FAISS_THROW_IF_NOT_MSG(deviceList.size() != 0, "device list can not be empty!");
    }

    // Ascend devices mask on which the index is resident
    std::vector<int> deviceList;
};

class AscendVectorTransform : public faiss::VectorTransform {
public:
    explicit AscendVectorTransform(int dimIn = 0, int dimOut = 0,
        AscendVectorTransformConfig config = AscendVectorTransformConfig());
    virtual ~AscendVectorTransform();

public:
    AscendVectorTransformConfig vtransConfig;

    std::unordered_map<int, rpcContext> contextMap;

    std::unordered_map<rpcContext, int> vtransMap;
};


/* * Generic linear transformation, with bias term applied on output
 * y = A * x + b
 */
class AscendLinearTransform : public AscendVectorTransform {
public:
    explicit AscendLinearTransform(int dimIn = 0, int dimOut = 0, bool haveBias = false,
        AscendVectorTransformConfig config = AscendVectorTransformConfig());

    // / Initialize ourselves from the given CPU index; will overwrite
    // / all data in ourselves
    void copyFrom(const faiss::LinearTransform *cpuTrans);

    // / Copy ourselves to the given CPU index; will overwrite all data
    // / in the index instance
    void copyTo(faiss::LinearTransform *cpuTrans) const;

    void train(idx_t n, const float *x) override;

    virtual ~AscendLinearTransform();

protected:
    // same as apply, but result is pre-allocated
    void apply_noalloc(idx_t n, const float *x, float *xt) const override;

protected:
    faiss::LinearTransform *cpuLtrans = nullptr;
    bool haveBias = false; // whether to use the bias term
    // Transformation matrix, size d_out * d_in
    std::vector<float> A;
    // bias vector, size d_out
    std::vector<float> b;

private:
    void createLinearTransform();

    void destroyLinearTransform();

    void updateMatrixAndBias();
};

/* * Applies a principal component analysis on a set of vectors,
 * with optionally whitening and random rotation. */
class AscendPCAMatrix : public AscendLinearTransform {
public:
    // the final matrix is computed after random rotation and/or whitening
    explicit AscendPCAMatrix(int dimIn, int dimOut, float eigenPower, bool randomRotation,
        std::initializer_list<int> devices = { 0 });

    explicit AscendPCAMatrix(int dimIn, int dimOut, float eigenPower, bool randomRotation,
        AscendVectorTransformConfig config);

    ~AscendPCAMatrix() {}

    /* * after transformation the components are multiplied by
     * eigenvalues^eigen_power
     *
     * =0: no whitening
     * =-0.5: full whitening
     */
    float eigenPower;

    // random rotation after PCA
    bool randomRotation;
};
}
}
#endif // ASCEND_VECTORTRANSFORM_INCLUDED
