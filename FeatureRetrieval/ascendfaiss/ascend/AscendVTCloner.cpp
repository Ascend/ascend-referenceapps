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

#include <faiss/ascend/AscendVTCloner.h>
#include <faiss/ascend/AscendVectorTransform.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace ascend {
/* *********************************************************
 * Cloning to CPU
 * ******************************************************** */
faiss::VectorTransform *ToCPUVTCloner::clone_VectorTransform(const faiss::VectorTransform *vtrans)
{
    if (auto pca = dynamic_cast<const AscendPCAMatrix *>(vtrans)) {
        PCAMatrix *res = new PCAMatrix(pca->d_in, pca->d_out, pca->eigenPower, pca->randomRotation);
        pca->copyTo(res);
        return res;
    } else {
        FAISS_THROW_MSG("clone from ascend to cpu not supported for this type of VectorTransform");
        return nullptr;
    }
}

faiss::VectorTransform *vtrans_ascend_to_cpu(const faiss::VectorTransform *ascend_vtrans)
{
    ToCPUVTCloner cl;
    return cl.clone_VectorTransform(ascend_vtrans);
}

/*
 * Cloning to Ascend
 */
ToAscendVTCloner::ToAscendVTCloner(std::initializer_list<int> devs, const AscendClonerOptions &options)
    : AscendClonerOptions(options), devices(devs)
{
    FAISS_THROW_IF_NOT_MSG(devs.size() != 0, "device list can not be empty!");
}

ToAscendVTCloner::ToAscendVTCloner(std::vector<int> devs, const AscendClonerOptions &options)
    : AscendClonerOptions(options), devices(devs)
{
    FAISS_THROW_IF_NOT_MSG(devs.size() != 0, "device list can not be empty!");
}

faiss::VectorTransform *ToAscendVTCloner::clone_VectorTransform(const faiss::VectorTransform *vtrans)
{
    if (auto pca = dynamic_cast<const PCAMatrix *>(vtrans)) {
        AscendVectorTransformConfig config;
        config.deviceList = devices;
        AscendPCAMatrix *res =
            new AscendPCAMatrix(pca->d_in, pca->d_out, pca->eigen_power, pca->random_rotation, config);
        res->copyFrom(pca);
        return res;
    } else {
        FAISS_THROW_MSG("clone from ascend to cpu not supported for this type of Index");
        return nullptr;
    }
}

faiss::VectorTransform *vtrans_cpu_to_ascend(std::initializer_list<int> devices, const faiss::VectorTransform *vtrans,
    const AscendClonerOptions *options)
{
    AscendClonerOptions defaults;
    ToAscendVTCloner cl(devices, options ? *options : defaults);
    return cl.clone_VectorTransform(vtrans);
}

faiss::VectorTransform *vtrans_cpu_to_ascend(std::vector<int> devices, const faiss::VectorTransform *vtrans,
    const AscendClonerOptions *options)
{
    AscendClonerOptions defaults;
    ToAscendVTCloner cl(devices, options ? *options : defaults);
    return cl.clone_VectorTransform(vtrans);
}
}
}
