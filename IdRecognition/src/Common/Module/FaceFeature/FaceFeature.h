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

#ifndef FACE_FEATURE_H
#define FACE_FEATURE_H

#include <thread>

#include "ModuleBase/ModuleBase.h"
#include "ModelProcess/ModelProcess.h"
#include "Statistic/Statistic.h"

namespace ascendFaceRecognition {
class FaceFeature : public ModuleBase {
public:
    ~FaceFeature();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);
    double GetRunTimeAvg();

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR PreParaData(std::vector<FaceObject *> &faceObjectQueue) const;
    APP_ERROR PostData(std::vector<FaceObject *> &faceObjectQueue) const;

private:
    int width_ = 0;
    uint32_t alignedWidth_ = 0;
    int height_ = 0;
    uint32_t alignedHeight_ = 0;
    int inputChn_ = 0;
    uint32_t dataSzie_  = 0;
    uint32_t featrueSize_  = 0;
    std::string modelPath_ = {};
    std::vector<void *> inputBuffers_ = {};
    std::vector<size_t> inputSizes_ = {};
    std::vector<void *> outputBuffers_ = {};
    std::vector<size_t> outputSizes_  = {};
    std::shared_ptr<ModelProcess> modelProcess_ = nullptr;
    std::vector<uint32_t> dynamicBatchSize_ = {};
    Statistic faceFeatureStatic_ = {};
    Statistic ffPreStatic_ = {};
    Statistic ffModelStatic_ = {};
    Statistic ffPostStatic_ = {};

    APP_ERROR Resnet18DynamicBatchSizeInputBufferMalloc();
    APP_ERROR Resnet18DynamicBatchSizeOutputBufferMalloc();
    APP_ERROR FindTheBestBatchSize(size_t &bestBatchSize, const size_t leftFaceNum) const;
};
const int FACE_FEATURE_WIDTH_ALIGN = 16;
const int FACE_FEATURE_HEIGHT_ALIGN = 2;

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) ((((x) + ((align)-1)) / (align)) * (align))
#endif
}


#endif
