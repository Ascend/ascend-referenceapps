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
#ifndef FACE_DETECTION_LANDMARK_H
#define FACE_DETECTION_LANDMARK_H

#include <thread>

#include "ConfigParser/ConfigParser.h"
#include "DataType/DataType.h"
#include "ModelProcess/ModelProcess.h"
#include "ModuleBase/ModuleBase.h"
#include "acl/acl.h"
#include "CenterFace.h"
#include "Statistic/Statistic.h"

namespace ascendFaceRecognition {
const int MAX_FACE_PER_FRAME = 20;

class FaceDetectionLandmark : public ModuleBase {
public:
    FaceDetectionLandmark();
    ~FaceDetectionLandmark();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);
    double GetRunTimeAvg();
    APP_ERROR GetFaceStatus() const;
    void SetFaceStatus();

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR PostProcessDetection(void);
    APP_ERROR ProcessCenterfaceResult(std::shared_ptr<FrameAiInfo> frameAiInfo, size_t singleSize,
        std::vector<void *> &outTensorAddrs);
    APP_ERROR ConstructDetectResult(std::shared_ptr<FrameAiInfo> frameAiInfo, std::vector<DetectInfo> &boxinfo) const;
    APP_ERROR FaceSelection(std::vector<DetectInfo> &detectResult) const;
    APP_ERROR DvppInit();
    APP_ERROR ModelInit();
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR GetNumInputs();
    APP_ERROR GetNumOutputs();
    void GetSizePaddingResize(float resizeWidth, float resizeHeight, float &oriWidth, float &oriHeight) const;

private:
    uint32_t batchSize_ = 0;
    uint32_t channelCount_ = 0;
    uint32_t maxFaceNumPerFrame_ = 0;
    int width_ = 0;
    int height_ = 0;

    uint32_t outWidthStride_ = 0;
    uint32_t outHeightStride_ = 0;
    uint32_t vpcOutBufferSize_ = 0;
    uint8_t *vpcOutBuffer_ = nullptr;
    uint32_t vpcBufferOffset_ = 0;
    void *inputBuffer_ = nullptr;
    void *outputBufsHost_ = nullptr;
    APP_ERROR isMultiFace_ = APP_ERR_OK;
    std::shared_ptr<Centerface::CenterFace> centerface_ = nullptr;

    std::string modelPath_ = "";
    ModelProcess modelInfer_ = {};
    std::vector<void *> outputBufs_ = {};
    std::vector<size_t> outputSizes_ = {};
    std::vector<void *> batchBufs_ = {};
    std::vector<std::shared_ptr<FrameAiInfo>> inputArgQueue_ = {};
    Statistic faceDecLMStatic_ = {};
    Statistic centerfacePreStatic_ = {};
    Statistic centerfaceStatic_ = {};
    Statistic centerfacePostStatic_ = {};
};
} // namespace ascendFaceRecognition

#endif