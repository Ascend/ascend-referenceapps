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

#ifndef ACL_PROCESS_H
#define ACL_PROCESS_H

#include <iostream>
#include "VdecUtils.h"
#include "VideoReader.h"
#include "acl/acl.h"
#include "DvppCommon/DvppCommon.h"
#include "ConfigParser/ConfigParser.h"
#include "Log/Log.h"

class AclProcess {
public:
    AclProcess();
    ~AclProcess();
    void SetInputFilePath(const std::string &inputFilePath);
    APP_ERROR InitResource();
    APP_ERROR DoVdecProcess();
private:
    void DestroyResource();
private:
    int32_t deviceId_ = 0;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    pthread_t threadId_ = -1;
    const char *outFolder_ = "result/";
    PicDesc picDesc_ = {};
    std::string filePath_;
    acldvppPixelFormat format_ = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
};

#endif // #ifndef ACL_PROCESS_H