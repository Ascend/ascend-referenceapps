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

#ifndef INC_WARP_PERSPECTIVE_H
#define INC_WARP_PERSPECTIVE_H

#include "DvppCommon/DvppCommon.h"
#include "ConfigParser/ConfigParser.h"
#include "Framework/ModuleManager/ModuleManager.h"
#include "Statistic/Statistic.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "Common/CommonType.h"

class WarpPerspective : public ascendBaseModule::ModuleBase {
public:
    WarpPerspective();
    ~WarpPerspective();
    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR ApplyWarpPerspective(std::shared_ptr<SendInfo> sendData, cv::Mat &imgRGB888);
    APP_ERROR CropTextBox(std::shared_ptr<SendInfo> sendData);
    APP_ERROR WarpPerspectiveProcess(std::shared_ptr<SendInfo> sendData);
    void InitCoordinate(std::shared_ptr<SendInfo> sendData);
    APP_ERROR TransToHost(std::shared_ptr<SendInfo> sendData, std::shared_ptr<uint8_t> &hostSharedPtr);

    uint32_t leftTopX_ = 0;
    uint32_t leftTopY_ = 0;
    uint32_t rightBotX_ = 0;
    uint32_t rightBotY_ = 0;
    std::unique_ptr<DvppCommon> dvppObjPtr_ = nullptr;
    aclrtStream dvppStream_ = nullptr;
    Statistic warpAffineStatic_ = {};
    uint32_t dstWidth_ = 0;
    uint32_t dstHeight_ = 0;
    uint32_t debugMode_ = false;
    bool ignoreWarpPerspective_ = true;
};

MODULE_REGIST(WarpPerspective)
#endif
