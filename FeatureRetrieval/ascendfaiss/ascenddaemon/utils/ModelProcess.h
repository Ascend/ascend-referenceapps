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

#ifndef ASCEND_MODEL_PROCESS_INCLUDED
#define ASCEND_MODEL_PROCESS_INCLUDED

#include "acl/acl.h"
#include <string>

namespace ascend {
class ModelProcess {
public:
    ModelProcess();

    ~ModelProcess();

    // Load model from file
    void LoadModelFromFileWithMem(const std::string &modelPath);

    // Create model desc
    void CreateDesc();

    // Create model input
    void CreateInput(void *inputDataBuffer, size_t bufferSize);

    // Destroy input resource
    void DestroyInput();

    // Create output buffer
    void CreateOutput();

    // Model execute
    void Execute();

    // Get model output data
    const aclmdlDataset *GetModelOutputData();

private:
    // Unload model
    void Unload();

    // Destroy desc
    void DestroyDesc();

    // Destroy output resource
    void DestroyOutput();

private:
    uint32_t modelId;
    size_t modelMemSize;
    size_t modelWeightSize;
    void *modelMemPtr;
    void *modelWeightPtr;
    bool loadFlag;  // model load flag
    aclmdlDesc *modelDesc;
    aclmdlDataset *input;
    aclmdlDataset *output;
};
}
#endif // ASCEND_MODEL_PROCESS_INCLUDED
