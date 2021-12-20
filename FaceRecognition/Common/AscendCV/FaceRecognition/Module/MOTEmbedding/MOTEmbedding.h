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

#ifndef INC_FACEDEMO_MOTEMBEDDINGMODULE_BASE_H
#define INC_FACEDEMO_MOTEMBEDDINGMODULE_BASE_H

#include <thread>
#include "ModuleBase/ModuleBase.h"
#include "Framework/ModelProcess/ModelProcess.h"

#include "acl/acl.h"

namespace ascendFaceRecognition {
class MOTEmbedding : public ModuleBase {
public:
    MOTEmbedding();
    ~MOTEmbedding();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);
    APP_ERROR ProcessFaceObjectQueue();
    APP_ERROR ProcessEmptyFaceFrame(std::shared_ptr<FrameAiInfo> frameAiInfo);
    APP_ERROR ProcessMulitFaceFrame(std::shared_ptr<FrameAiInfo> frameAiInfo, std::shared_ptr<FaceObject> faceObject);

private:
    APP_ERROR PreParaData(std::vector<std::shared_ptr<FaceObject>> &faceObjectQueue);
    APP_ERROR PreParaData(std::shared_ptr<FaceObject> faceObject);
    APP_ERROR PostData(std::vector<std::shared_ptr<FaceObject>> &faceObjectQueue);
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR InitResource(void);
    float FeatureNorm(const float *feature, const uint32_t &featureSize);

private:
    int enable_ = 1;
    uint32_t batchSize_ = 1;
    uint32_t channel_ = 0;
    std::string modelPath_ = {};
    std::shared_ptr<ModelProcess> modelInfer_ = nullptr;
    std::vector<void *> inputBuffers_ = {};
    std::vector<size_t> inputSizes_ = {};
    std::vector<void *> outputBuffers_ = {};
    std::vector<size_t> outputSizes_ = {};
    std::vector<std::shared_ptr<FaceObject>> faceObjectQueue_ = {};
    uint32_t normalMode_ = 0;
};
}
#endif
