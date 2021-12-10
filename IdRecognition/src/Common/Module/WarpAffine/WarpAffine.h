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
#ifndef WARP_AFFINE_H
#define WARP_AFFINE_H
#include "ModuleBase/ModuleBase.h"
#include "Statistic/Statistic.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace ascendFaceRecognition {
const int AFFINE_LEN = 6;

class WarpAffine : public ModuleBase {
public:
    WarpAffine(): widthOutput_(0), heightOutput_(0){};
    ~WarpAffine();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);
    double GetRunTimeAvg();

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR ApplyWarpAffine(FaceObject &face, cv::Mat &imgBGR888);
    APP_ERROR KeyPointConversion(std::shared_ptr<KeyPointInfo> keyPointInfo, float &deno0);
    APP_ERROR CalAffineMatrix(float *keyPointBefore, int keyPointBeforeSize, float *keyPointAfter,
                              int keyPointAfterSize, int affineMatrixSize);

private:
    int widthOutput_;
    int heightOutput_;
    float affineMatrix_[AFFINE_LEN] = {};
    Statistic waColorCvtStatic_ = {};
    Statistic waPostStatic_ = {};
    Statistic warpAffineStatic_ = {};
};
} // namespace ascendFaceRecognition

#endif