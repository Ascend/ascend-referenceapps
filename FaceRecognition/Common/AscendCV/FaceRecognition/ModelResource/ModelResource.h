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

#ifndef INC_MODEL_RESOURCE_H
#define INC_MODEL_RESOURCE_H

#include "Framework/ModelProcess/ModelProcess.h"
#include "Log/Log.h"

#include <ErrorCode/ErrorCode.h>
#include <memory>
#include <mutex>
#include <map>

namespace ascendFaceRecognition {
class ModelResource {
public:
    static ModelResource &GetInstance();
    std::shared_ptr<ModelProcess> GetModelProcess(const std::string &modelPath, const int instanceId = 0);
    APP_ERROR ClearModelProcess(const std::string &modelPath);
    void Release();
    
private:
    std::mutex mtx = {};
    ModelResource();
    ~ModelResource();
    std::map<std::string, std::map<int, std::shared_ptr<ModelProcess>>> modelMap_ = {};
};
} // namespace ascendFaceRecognition

#endif