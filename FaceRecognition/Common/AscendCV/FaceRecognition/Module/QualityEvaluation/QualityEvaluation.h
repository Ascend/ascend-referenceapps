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

#ifndef INC_QUALITY_EVALUATION_H
#define INC_QUALITY_EVALUATION_H

#include <map>
#include <set>
#include "ModuleBase/ModuleBase.h"
#include "Framework/ModelProcess/ModelProcess.h"
#include "acl/acl.h"

namespace ascendFaceRecognition {
class QualityEvaluation : public ModuleBase {
public:
    QualityEvaluation();
    ~QualityEvaluation();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR PreParaData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue);
    APP_ERROR PreParaData(std::shared_ptr<FaceObject> faceObject);
    APP_ERROR PostData(std::vector<std::shared_ptr<FaceObject>> faceObjectQueue);
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR InitResource(void);

    APP_ERROR PreFilter(std::shared_ptr<FaceObject> faceObject,
        std::vector<std::shared_ptr<FaceObject>> &faceObjectQueue);

    float CalKeyPointScore(int pos);
    float CalEulerScore(int pos);
    float CalBigFaceScore(std::shared_ptr<FaceObject> faceObject);
    APP_ERROR CalKPCoordinate(int pos, std::shared_ptr<FaceObject> faceObject);
    APP_ERROR FaceCropResize(std::vector<std::shared_ptr<FaceObject>> &faceObjectQueue);
    void SendData();
    void SendThread();

private:
    int heatmapWidth_ = 0;
    int heatmapHeight_ = 0;
    float keyPointWeight_ = 0.0;
    float eulerWeight_ = 0.0;
    float bigFaceWeight_ = 0.0;
    float minOutputScore_ = 0.0;
    uint32_t batchSize_ = 1;
    double maxSendTime_ = 15.; // second

    std::string modelPath_ = {};
    std::shared_ptr<ModelProcess> modelInfer_ = nullptr;

    std::vector<void *> inputBuffers_ = {};
    std::vector<size_t> inputSizes_ = {};
    std::vector<void *> outputBuffers_ = {};
    std::vector<size_t> outputSizes_ = {};
#ifndef ASCEND_ACL_OPEN_VESION
    std::vector<std::shared_ptr<void>> outputBuffersHost_ = {};
#endif
    std::vector<std::shared_ptr<FaceObject>> faceObjectVec_ = {};
    std::set<int32_t> channels_ = {};
    std::thread sendThread_ = {};
    bool sendStop_ = false;
    const uint32_t sleepTime_ = 1; // second
};
}
#endif
