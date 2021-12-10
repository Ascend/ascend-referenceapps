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

#include <iostream>
#include <csignal>
#include <unistd.h>

#include "CommandLine.h"
#include "CommonFuction.h"

using namespace ascendFaceRecognition;

namespace {
APP_ERROR Init(CmdParams &cmdParams, FaceRecgData &faceRecgData)
{
    // init ACL
    LogDebug << "Init ACL.";
    aclError deviceRet = aclInit(cmdParams.aclConfig.c_str());
    if (deviceRet != ACL_ERROR_NONE) {
        LogError << "Fail to init AscendCL, ret[" << deviceRet << "]";
        return APP_ERR_COMM_FAILURE;
    }

    // create face lib and feature lib
    APP_ERROR ret = faceRecgData.faceFeature.FaceFeatureLibInit();
    if (ret != APP_ERR_OK) {
        LogError << "Fail to init feature lib.";
        return APP_ERR_COMM_FAILURE;
    }

    // init system manager of face search
    if (cmdParams.runMode == PIPELINE_MODE_SEARCH) {
        faceRecgData.initArgsSearch.pipelineMode = PIPELINE_MODE_SEARCH;
        faceRecgData.initArgsSearch.moduleTypeCount = MODULE_TYPE_COUNT_SEARCH;
        faceRecgData.initArgsSearch.pipelineDesc = g_moduleDescSearch;
        faceRecgData.initArgsSearch.faceFeature = &faceRecgData.faceFeature;
        LogInfo << "Init system manager of face search.";
        ret = InitSystemManager(cmdParams.searchConfig, faceRecgData.initArgsSearch, faceRecgData.managerSearch);
        if (ret != APP_ERR_OK) {
            LogFatal << "Fail to init system manager of face search.";
            return APP_ERR_COMM_FAILURE;
        }
    }

    // init system manager of face registration
    if (cmdParams.runMode == PIPELINE_MODE_REG) {
        faceRecgData.initArgsReg.pipelineMode = PIPELINE_MODE_REG;
        faceRecgData.initArgsReg.moduleTypeCount = MODULE_TYPE_COUNT_REG;
        faceRecgData.initArgsReg.pipelineDesc = g_moduleDescReg;
        faceRecgData.initArgsReg.faceFeature = &faceRecgData.faceFeature;
        LogInfo << "Init system manager of face registration.";
        ret = InitSystemManager(cmdParams.regConfig, faceRecgData.initArgsReg, faceRecgData.managerReg);
        if (ret != APP_ERR_OK) {
            LogFatal << "Fail to init system manager of face registration.";
            return APP_ERR_COMM_FAILURE;
        }
    }

    return ret;
}
}

int main(int argc, const char *argv[])
{
    CmdParams cmdParams;
    // parse command line and set log level
    if (ParseACommandLine(argc, argv, cmdParams)) {
        LogFatal << "Fail to parse command line.";
        return APP_ERR_COMM_FAILURE;
    }
    SetLogLevel(cmdParams.debugLevel);
    if (cmdParams.runMode == PIPELINE_MODE_SEARCH) {
        LogInfo << "IdRecognition demo start search process successfully" << std::endl;
    } else if (cmdParams.runMode == PIPELINE_MODE_REG) {
        LogInfo << "IdRecognition demo start registration process successfully" << std::endl;
    } else {
        LogFatal << "IdRecognition demo unkown the run mode" << std::endl;
        return APP_ERR_COMM_FAILURE;
    }

    FaceRecgData faceRecgData;

    APP_ERROR ret = Init(cmdParams, faceRecgData);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to Init pipeline.";
        return APP_ERR_COMM_FAILURE;
    }

    ret = Run(cmdParams, faceRecgData);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to Run pipeline.";
        return APP_ERR_COMM_FAILURE;
    }

    // wait for exit signal
    LogDebug << "wait for exit signal.";
    if (signal(SIGINT, SigHandler) == SIG_ERR) {
        LogError << "cannot catch SIGINT.";
    }
    while (!g_signalRecieved) {
        usleep(SIGNAL_CHECK_TIMESTEP);
    }

    ret = DeInit(cmdParams, faceRecgData);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to DeInit pipeline.";
        return APP_ERR_COMM_FAILURE;
    }
    LogInfo << "IdRecognition demo ended successfully" << std::endl;

    return 0;
}
