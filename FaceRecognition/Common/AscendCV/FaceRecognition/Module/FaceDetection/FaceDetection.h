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

#ifndef INC_FACE_DETECTION_H
#define INC_FACE_DETECTION_H

#include <thread>

#include "ConfigParser/ConfigParser.h"
#include "DataType/DataType.h"
#include "ModuleBase/ModuleBase.h"
#include "Framework/ModelProcess/ModelProcess.h"
#include "acl/acl.h"
#include "Yolov3Batch.h"
#include "TestCV/TestCV.h"

namespace ascendFaceRecognition {
class FaceDetection : public ModuleBase {
public:
    FaceDetection();
    ~FaceDetection();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR PostProcess(void) const;
    APP_ERROR PostProcessCPU(const std::vector<void*> &tensors) const;
    APP_ERROR PostProcessAiCore(const std::vector<void*> &tensors) const;
    void GetResultAiCore(DetectInfo &detectInfo, const uint8_t *resultAddress, const uint32_t &boxIndex,
        const uint32_t &boxSize) const;
    APP_ERROR FaceSelection(std::vector<DetectInfo> &detectResult) const;
    APP_ERROR ModelInit();
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR PreProcess(std::shared_ptr<FrameAiInfo> frameAiInfo);

private:
    uint32_t batchSize_ = 1;
    uint32_t channelCount_ = 0;
    uint32_t maxFaceNumPerFrame_ = 20;
    uint32_t width_ = 0;
    uint32_t height_ = 0;

    uint32_t outWidthStride_ = 0;
    uint32_t outHeightStride_ = 0;

    std::string modelPath_ = {};
    std::shared_ptr<ModelProcess> modelInfer_ = nullptr;
    std::vector<void *> inputBufs_ = {};
    std::vector<size_t> inputSizes_ = {};
    std::vector<void *> outputBufs_ = {};
    std::vector<size_t> outputSizes_ = {};
    std::vector<std::shared_ptr<void>> outputBufsHost_ = {};
    std::shared_ptr<Yolov3Batch> yolov3 = nullptr;
    std::vector<std::shared_ptr<FrameAiInfo>> inputArgQueue_ = {};
};
} // namespace ascendFaceRecognition

#endif // INC_FACE_DETECTION_H
