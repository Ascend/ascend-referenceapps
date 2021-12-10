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

#ifndef FACE_SEARCH_H
#define FACE_SEARCH_H

#include <thread>

#include "ErrorCode/ErrorCode.h"
#include "FaceFeatureLib/FaceFeatureLib.h"
#include "ModuleBase/ModuleBase.h"
#include "ConfigParser/ConfigParser.h"
#include "Statistic/Statistic.h"

namespace ascendFaceRecognition {
class FaceSearch : public ModuleBase {
public:
    FaceSearch();
    ~FaceSearch();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);
    double GetRunTimeAvg();
    int GetSuccessCount() const;
    void SetSuccessCount();
    APP_ERROR ResetFaceFeatureLib() const;

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser) const;
    void SaveConfigFile(unsigned int i, const FaceObject &face, const std::string &fileName) const;
    void SaveResult(std::shared_ptr<FrameAiInfo> frameAiInfo) const;

private:
    FaceFeatureLib *faceFeatureLib_ = nullptr;
    Statistic faceSearchStatic_ = {};
    int successCount_ = 0;
};
} // namespace ascendFaceRecognition

#endif
