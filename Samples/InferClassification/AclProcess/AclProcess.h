/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ACLMANAGER_H
#define ACLMANAGER_H

#include <map>
#include <iostream>
#include "acl/acl.h"
#include "CommonDataType/CommonDataType.h"
#include "Log/Log.h"
#include "ErrorCode/ErrorCode.h"
#include "FileManager/FileManager.h"
#include "Framework/ModelProcess/ModelProcess.h"
#include "DvppCommon/DvppCommon.h"
#include "SingleOpProcess/SingleOpProcess.h"
#include "ResourceManager/ResourceManager.h"

class AclProcess {
public:

    AclProcess(int deviceId, ModelInfo modelInfo, std::string opModelPath, aclrtContext context);

    ~AclProcess() {};

    void Release();

    APP_ERROR InitResource();

    APP_ERROR Process(const std::string& imageFile);

private:

    APP_ERROR InitModule();

    APP_ERROR InitOpCastResource() const;

    APP_ERROR InitOpArgMaxResource() const;

    APP_ERROR LoadLabels(const std::string& labelPath);

    APP_ERROR Preprocess(const std::string& imageFile) const;

    APP_ERROR ModelInfer(std::vector<RawData> &modelOutput);

    APP_ERROR CastOpInfer(std::vector<RawData> modelOutput);

    APP_ERROR ArgMaxOpInfer();

    APP_ERROR PostProcess();

    APP_ERROR WriteResult(int index);

    int32_t deviceId_; // device id used
    ModelInfo modelInfo_; // info of input model
    std::string opModelPath_; // path of operator model
    aclrtContext context_;
    aclrtStream stream_;
    std::unique_ptr<ModelProcess> modelProcess_; // model inference object
    std::unique_ptr<DvppCommon> dvppCommon_; // dvpp jpegd object
    std::unique_ptr<SingleOpProcess> argMaxOp_; // ArgMax operator
    std::unique_ptr<SingleOpProcess> castOp_; // Cast operator
    std::map<int, std::string> labelMap_; // labels info
};

#endif
