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

#ifndef VDEC_UTILS_H
#define VDEC_UTILS_H

#include <iostream>
#include <unistd.h>
#include "CommonDataType/CommonDataType.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"


class VdecUtils {
public:
    static APP_ERROR CheckFolder(const std::string &foldName);
    static bool WriteToFile(const std::string &fileName, const std::unique_ptr<void, decltype(acldvppFree)*>& dataDev,
        uint32_t dataSize);
};

#endif // #ifndef VDEC_UTILS_H