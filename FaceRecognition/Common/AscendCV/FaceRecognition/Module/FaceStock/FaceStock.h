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

#ifndef INC_FACE_STOCK_H
#define INC_FACE_STOCK_H

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
    FaceStock();
    ~FaceStock() {};
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
private:
    FaceFeatureLib *faceFeatureLib_ = nullptr;
};
} // namespace ascendFaceRecognition
#endif
