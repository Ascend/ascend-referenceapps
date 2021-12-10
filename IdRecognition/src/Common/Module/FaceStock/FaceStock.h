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

#ifndef FACE_STOCK_H
#define FACE_STOCK_H

#include <thread>

#include "ModuleBase/ModuleBase.h"
#include "FaceFeatureLib/FaceFeatureLib.h"
#include "ConfigParser/ConfigParser.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace ascendFaceRecognition {
class FaceStock : public ModuleBase {
public:
    FaceStock() {};
    ~FaceStock() {};
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);
    APP_ERROR GetFaceFeatureStatus() const;
    void SetFaceFeatureStatus();

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);
    APP_ERROR ValidateParm(std::shared_ptr<FrameAiInfo> frameAiInfo, unsigned int faceIdx) const;

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR CalculateROI(ImageInfo &imgOrigin, FaceObject &faceObject, cv::Rect &roi) const;

    FaceFeatureLib *faceFeatureLib_ = nullptr;
    int widthStockImg_ = 0;
    int heightStockImg_ = 0;
    APP_ERROR featureIsExist_ = APP_ERROR_FACE_WEB_USE_SYSTEM_ERROR;
};
} // namespace ascendFaceRecognition
#endif
