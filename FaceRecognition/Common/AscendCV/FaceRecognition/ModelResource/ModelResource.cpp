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

#include "ModelResource.h"

namespace ascendFaceRecognition {
ModelResource::ModelResource() {}

ModelResource::~ModelResource()
{
    modelMap_.clear();
}

ModelResource &ModelResource::GetInstance()
{
    static ModelResource modelResource;
    return modelResource;
}

std::shared_ptr<ModelProcess> ModelResource::GetModelProcess(const std::string &modelPath, const int instanceId)
{
    std::lock_guard<std::mutex> guard(mtx);
    if (modelMap_.find(modelPath) != modelMap_.end() &&
        modelMap_[modelPath].find(instanceId) != modelMap_[modelPath].end()) {
        return modelMap_[modelPath][instanceId];
    }
    std::shared_ptr<ModelProcess> modelProcess = std::make_shared<ModelProcess>();
    APP_ERROR ret = modelProcess->Init(modelPath);
    if (ret != APP_ERR_OK) {
        LogError << "ModelProcess Init failed";
        return nullptr;
    }
    if (modelMap_.find(modelPath) != modelMap_.end()) {
        modelMap_[modelPath][instanceId] = modelProcess;
    } else {
        modelMap_[modelPath] = std::map<int, std::shared_ptr<ModelProcess>>();
        modelMap_[modelPath][instanceId] = modelProcess;
    }

    return modelProcess;
}

APP_ERROR ModelResource::ClearModelProcess(const std::string &modelPath)
{
    std::lock_guard<std::mutex> guard(mtx);
    if (modelMap_.find(modelPath) != modelMap_.end()) {
        modelMap_.erase(modelPath);
        return APP_ERR_OK;
    }
    return APP_ERR_COMM_NO_EXIST;
}

void ModelResource::Release()
{
    modelMap_.clear();
}
} // namespace ascendFaceRecognition