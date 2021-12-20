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

#ifndef INC_TEXT_DETECTION_H
#define INC_TEXT_DETECTION_H

#include "DvppCommon/DvppCommon.h"
#include "ConfigParser/ConfigParser.h"
#include "Framework/ModelProcess/ModelProcess.h"
#include "Framework/ModuleManager/ModuleManager.h"
#include "Statistic/Statistic.h"
#include "Common/CommonType.h"

class TextDetection : public ascendBaseModule::ModuleBase {
public:
    TextDetection();
    ~TextDetection();
    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR PrepareModelBuffer(std::vector<void *> &inputBuffers, std::vector<size_t> &inputSizes,
                                 std::vector<void *> &outputBuffers, std::vector<size_t> &outputSizes) const;

    uint32_t deviceId_ = 0;
    std::string modelName_ = "";
    std::string modelPath_ = "";
    std::unique_ptr<ModelProcess> modelProcess_ = nullptr;
    Statistic textDetectStatic_ = {};
};

MODULE_REGIST(TextDetection)
#endif
