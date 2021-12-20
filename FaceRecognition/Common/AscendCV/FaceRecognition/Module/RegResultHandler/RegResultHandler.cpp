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

#include "RegResultHandler.h"
#include "Log/Log.h"
#include "RegistApi/RegistApi.h"
#include "DataTrans/DataTrans.pb.h"

#ifdef ASCEND_ACL_OPEN_VESION
#include "HdcChannel/HdcChannel.h"
#endif

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace ascendFaceRecognition {
RegResultHandler::RegResultHandler() {}

APP_ERROR RegResultHandler::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    // init parameters
    LogDebug << "RegResultHandler[" << instanceId_ << "]: begin to init jpeg reader instance.";
    AssignInitArgs(initArgs);
    return APP_ERR_OK;
}

APP_ERROR RegResultHandler::DeInit(void)
{
    return APP_ERR_OK;
}

APP_ERROR RegResultHandler::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "RegResultHandler[" << instanceId_ << "]: process.";
    APP_ERROR ret = APP_ERR_OK;
    std::shared_ptr<DataTrans> dataTrans = std::static_pointer_cast<DataTrans>(inputData);
    std::shared_ptr<RegistResult> regResult = std::make_shared<RegistResult>(dataTrans->regresult());

    RegistApi::GetInstance()->HandleRegResult(regResult);

    return ret;
}
} // namespace ascendFaceRecognition
