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

#ifndef INC_FACEDEMO_FACE_ATTRIBUTE_BASE_H
#define INC_FACEDEMO_FACE_ATTRIBUTE_BASE_H

#include <thread>

#include "ModuleBase/ModuleBase.h"
#include "Framework/ModelProcess/ModelProcess.h"


#include "acl/acl.h"

namespace ascendFaceRecognition {
class FaceAttribute : public ModuleBase {
public:
    FaceAttribute();
    ~FaceAttribute();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR InitDvpp(void);
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR PreFilter(std::shared_ptr<FaceObject> faceObject,
        std::vector<std::shared_ptr<FaceObject>> &faceObjectQueue);
    APP_ERROR PreParaData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue);
    APP_ERROR PostData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue);
    int GetAge(const int &batchIdx);
    int GetGender(const int &batchIdx);
    int IsMask(const int &batchIdx);

private:
    int batchSize_ = 1;
    int width_ = 0;
    uint32_t alignedWidth_ = 0;
    int height_ = 0;
    uint32_t alignedHeight_ = 0;
    int inputChn_ = 0;
    uint32_t maxFaceNumPerFrame_ = 20;
    uint32_t minOutputArea_ = 0;
    std::string modelPath_ = {};
    std::vector<void *> inputBuffers_ = {};
    std::vector<size_t> inputSizes_ = {};
    std::vector<void *> outputBuffers_ = {};
    std::vector<size_t> outputSizes_ = {};
#ifndef ASCEND_ACL_OPEN_VESION
    std::vector<std::shared_ptr<void>> outputBuffersHost_ = {};
#endif
    std::shared_ptr<ModelProcess> modelInfer_ = nullptr;
    std::vector<std::shared_ptr<FaceObject>> faceObjectVec_ = {};
};

const int FACE_ATTRIBUTE_WIDTH_ALIGN = 16;
const int FACE_ATTRIBUTE_HEIGHT_ALIGN = 2;

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) ((((x) + ((align)-1)) / (align)) * (align))
#endif
}

#endif
