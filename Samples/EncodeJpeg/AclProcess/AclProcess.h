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
#include "DvppCommon/DvppCommon.h"

class AclProcess {
public:
    AclProcess(int deviceId, aclrtContext context);
    ~AclProcess() {};
    // Release all the resource
    void Release();
    // Create resource for this sample
    APP_ERROR InitResource();
    APP_ERROR Process(std::string imageFile, int width, int height, acldvppPixelFormat format) const;
    APP_ERROR WriteResult(std::string& fileName, const std::shared_ptr<void>& outBuf,
            const uint32_t& encodeOutDataSize) const;

private:
    int32_t deviceId_; // device id used
    aclrtContext context_;
    aclrtStream stream_;
    std::shared_ptr<DvppCommon> dvppCommon_; // dvpp common object
};

#endif