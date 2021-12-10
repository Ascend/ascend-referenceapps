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

#include <ascenddaemon/utils/ModelProcess.h>
#include <string>
#include <unistd.h>
#include <ascenddaemon/utils/AscendAssert.h>

namespace ascend {
ModelProcess::ModelProcess()
    : modelId(0),
      modelMemSize(0),
      modelWeightSize(0),
      modelMemPtr(nullptr),
      modelWeightPtr(nullptr),
      loadFlag(false),
      modelDesc(nullptr),
      input(nullptr),
      output(nullptr)
{}

ModelProcess::~ModelProcess()
{
    Unload();
    DestroyDesc();
    DestroyInput();
    DestroyOutput();
}

void ModelProcess::LoadModelFromFileWithMem(const std::string &modelPath)
{
    ASCEND_THROW_IF_NOT_MSG(!loadFlag, "has already loaded a model");

    aclError ret = aclmdlQuerySize(modelPath.c_str(), &modelMemSize, &modelWeightSize);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "query model failed, model file is %s", modelPath.c_str());

    ret = aclrtMalloc(&modelMemPtr, modelMemSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "malloc buffer for mem failed, require size is %zu", modelMemSize);

    ret = aclrtMalloc(&modelWeightPtr, modelWeightSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "malloc buffer for weight failed, require size is %zu",
        modelWeightSize);

    ret = aclmdlLoadFromFileWithMem(modelPath.c_str(), &modelId, modelMemPtr, modelMemSize, modelWeightPtr,
        modelWeightSize);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "load model from file failed, model file is %s", modelPath.c_str());

    loadFlag = true;
}

void ModelProcess::CreateDesc()
{
    modelDesc = aclmdlCreateDesc();
    ASCEND_THROW_IF_NOT_MSG(modelDesc != nullptr, "create model description failed");

    aclError ret = aclmdlGetDesc(modelDesc, modelId);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_ERROR_NONE, "get model description failed");
}

void ModelProcess::DestroyDesc()
{
    if (modelDesc != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc);
        modelDesc = nullptr;
    }
}

void ModelProcess::CreateInput(void *inputDataBuffer, size_t bufferSize)
{
    input = aclmdlCreateDataset();
    ASCEND_THROW_IF_NOT_MSG(input != nullptr, "can't create dataset, create input failed");

    aclDataBuffer *inputData = aclCreateDataBuffer(inputDataBuffer, bufferSize);
    ASCEND_THROW_IF_NOT_MSG(inputData != nullptr, "can't create data buffer, create input failed");

    aclError ret = aclmdlAddDatasetBuffer(input, inputData);
    if (ret != ACL_ERROR_NONE) {
        aclDestroyDataBuffer(inputData);
        inputData = nullptr;
    }
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_ERROR_NONE, "add input dataset buffer failed");
}

void ModelProcess::DestroyInput()
{
    if (input == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input, i);
        aclDestroyDataBuffer(dataBuffer);
    }
    aclmdlDestroyDataset(input);
    input = nullptr;
}

void ModelProcess::CreateOutput()
{
    ASCEND_THROW_IF_NOT_MSG(modelDesc != nullptr, "no model description, create ouput failed");
    output = aclmdlCreateDataset();
    ASCEND_THROW_IF_NOT_MSG(output != nullptr, "can't create dataset, create output failed");

    size_t outputSize = aclmdlGetNumOutputs(modelDesc);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "can't malloc buffer, size is %zu, create output failed",
            bufferSize);

        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, bufferSize);
        if (outputData == nullptr) {
            aclrtFree(outputBuffer);
        }
        ASCEND_THROW_IF_NOT_MSG(outputData != nullptr, "can't create data buffer, create output failed");

        ret = aclmdlAddDatasetBuffer(output, outputData);
        if (ret != ACL_ERROR_NONE) {
            aclrtFree(outputBuffer);
            aclDestroyDataBuffer(outputData);
        }
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_ERROR_NONE, "can't add data buffer, create output failed");
    }
}

void ModelProcess::DestroyOutput()
{
    if (output == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output, i);
        void *data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(output);
    output = nullptr;
}

void ModelProcess::Execute()
{
    aclError ret = aclmdlExecute(modelId, input, output);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "execute model failed, modelId is %u", modelId);
}

void ModelProcess::Unload()
{
    if (!loadFlag) {
        return;
    }

    aclError ret = aclmdlUnload(modelId);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "unload model failed, modelId is %u", modelId);

    if (modelDesc != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc);
        modelDesc = nullptr;
    }

    if (modelMemPtr != nullptr) {
        aclrtFree(modelMemPtr);
        modelMemPtr = nullptr;
        modelMemSize = 0;
    }

    if (modelWeightPtr != nullptr) {
        aclrtFree(modelWeightPtr);
        modelWeightPtr = nullptr;
        modelWeightSize = 0;
    }

    loadFlag = false;
}

const aclmdlDataset *ModelProcess::GetModelOutputData()
{
    return output;
}
}
