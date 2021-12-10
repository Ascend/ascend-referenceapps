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

#ifndef IDRECOGNITION_COMMONFUCTION_H
#define IDRECOGNITION_COMMONFUCTION_H

#include <iostream>
#include <csignal>
#include <unistd.h>

#include "FaceFeatureLib/FaceFeatureLib.h"
#include "Log/Log.h"
#include "SystemManager/SystemManager.h"
#include "CommandLine.h"

struct FaceRecgData {
    ascendFaceRecognition::FaceFeatureLib faceFeature;
    ascendFaceRecognition::SystemInitArgs initArgsSearch;
    ascendFaceRecognition::SystemInitArgs initArgsReg;
    ascendFaceRecognition::SystemManager managerSearch;
    ascendFaceRecognition::SystemManager managerReg;
};

const int MODULE_TYPE_COUNT_REG = 6;
const int MODULE_TYPE_COUNT_SEARCH = 6;

ascendFaceRecognition::ModuleDesc g_moduleDescSearch[MODULE_TYPE_COUNT_SEARCH] = {
    {-1, ascendFaceRecognition::MT_JPEG_READER,
     "JpegReader", ascendFaceRecognition::MT_IMAGE_DECODER},
    {-1, ascendFaceRecognition::MT_IMAGE_DECODER,
     "ImageDecoder", ascendFaceRecognition::MT_FACE_DETECTION_LANDMARK},
    {1, ascendFaceRecognition::MT_FACE_DETECTION_LANDMARK,
     "FaceDetectionLandmark", ascendFaceRecognition::MT_WARP_AFFINE},
    {1, ascendFaceRecognition::MT_WARP_AFFINE,
     "WarpAffine", ascendFaceRecognition::MT_FACE_FEATURE},
    {1, ascendFaceRecognition::MT_FACE_FEATURE,
     "FaceFeature", ascendFaceRecognition::MT_FACE_SEARCH},
    {-1, ascendFaceRecognition::MT_FACE_SEARCH,
     "FaceSearch", ascendFaceRecognition::MT_BOTTOM}
};

ascendFaceRecognition::ModuleDesc g_moduleDescReg[MODULE_TYPE_COUNT_REG] = {
    {1, ascendFaceRecognition::MT_JPEG_READER,
     "JpegReader", ascendFaceRecognition::MT_IMAGE_DECODER},
    {1, ascendFaceRecognition::MT_IMAGE_DECODER,
     "ImageDecoder", ascendFaceRecognition::MT_FACE_DETECTION_LANDMARK},
    {1, ascendFaceRecognition::MT_FACE_DETECTION_LANDMARK,
     "FaceDetectionLandmark", ascendFaceRecognition::MT_WARP_AFFINE},
    {1, ascendFaceRecognition::MT_WARP_AFFINE,
     "WarpAffine", ascendFaceRecognition::MT_FACE_FEATURE},
    {1, ascendFaceRecognition::MT_FACE_FEATURE,
     "FaceFeature", ascendFaceRecognition::MT_FACE_STOCK},
    {1, ascendFaceRecognition::MT_FACE_STOCK,
     "FaceStock", ascendFaceRecognition::MT_BOTTOM}
};

bool g_signalRecieved = false;
const int SIGNAL_CHECK_TIMESTEP = 10000;

void SigHandler(int signo)
{
    if (signo == SIGINT) {
        g_signalRecieved = true;
        std::cout << std::endl;
    }
}

APP_ERROR InitSystemManager(std::string &configPath, ascendFaceRecognition::SystemInitArgs &initArgs,
                            ascendFaceRecognition::SystemManager &systemManager)
{
    LogInfo << "Begin to init system manager.";

    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueueVec;
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueueVec;

    // 1. create queues of receivers and senders
    ConfigParser configParser;
    APP_ERROR ret = configParser.ParseConfig(configPath);
    if (ret != APP_ERR_OK) {
        LogFatal << "Cannot parse file.";
        return ret;
    }
    int channelCount = {};
    ret = configParser.GetIntValue(std::string("SystemConfig.channelCount"), channelCount);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    if (channelCount <= 0) {
        LogFatal << "Invalid channel count.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    for (int i = 0; i < channelCount; i++) {
        inputQueueVec.push_back(std::make_shared<BlockingQueue<std::shared_ptr<void>>>
        (ascendFaceRecognition::MODULE_QUEUE_SIZE));
        outputQueueVec.push_back(std::make_shared<BlockingQueue<std::shared_ptr<void>>>
        (ascendFaceRecognition::MODULE_QUEUE_SIZE));
    }

    // 2. init system manager
    initArgs.inputQueVec = inputQueueVec;
    initArgs.outputQueVec = outputQueueVec;
    ret = systemManager.Init(configPath, initArgs);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to init system manager.";
        return APP_ERR_COMM_FAILURE;
    }

    return ret;
}

APP_ERROR DeInitSystemManager(ascendFaceRecognition::SystemInitArgs &initArgs,
                              ascendFaceRecognition::SystemManager &systemManager)
{
    LogInfo << "Begin to deinit system manager.";

    // 1. Deinit system manager
    APP_ERROR ret = systemManager.DeInit();
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to deinit system manager.";
        return APP_ERR_COMM_FAILURE;
    }

    return ret;
}

APP_ERROR Run(CmdParams &cmdParams, FaceRecgData &faceRecgData)
{
    APP_ERROR ret = APP_ERR_OK;

    // run pipeline of face search
    if (cmdParams.runMode == ascendFaceRecognition::PIPELINE_MODE_SEARCH) {
        LogInfo << "Run pipeline of face search.";
        ret = faceRecgData.managerSearch.RunPipeline();
        if (ret != APP_ERR_OK) {
            LogFatal << "Fail to run pipeline of face search.";
            return APP_ERR_COMM_FAILURE;
        }
    }

    // run pipeline of face registration
    if (cmdParams.runMode == ascendFaceRecognition::PIPELINE_MODE_REG) {
        LogInfo << "Run pipeline of face registration.";
        ret = faceRecgData.managerReg.RunPipeline();
        if (ret != APP_ERR_OK) {
            LogFatal << "Fail to run pipeline of face registration.";
            return APP_ERR_COMM_FAILURE;
        }
    }

    return ret;
}

APP_ERROR DeInit(CmdParams &cmdParams, FaceRecgData &faceRecgData)
{
    APP_ERROR ret = APP_ERR_OK;

    // deinit system manager of face search
    if (cmdParams.runMode == ascendFaceRecognition::PIPELINE_MODE_SEARCH) {
        LogInfo << "deinit system manager of face registration.";
        ret = DeInitSystemManager(faceRecgData.initArgsSearch, faceRecgData.managerSearch);
        if (ret != APP_ERR_OK) {
            LogFatal << "Fail to deinit system manager of face registration.";
            return APP_ERR_COMM_FAILURE;
        }
    }

    // deinit system manager of face registration
    if (cmdParams.runMode == ascendFaceRecognition::PIPELINE_MODE_REG) {
        LogInfo << "deinit system manager of face registration.";
        ret = DeInitSystemManager(faceRecgData.initArgsReg, faceRecgData.managerReg);
        if (ret != APP_ERR_OK) {
            LogFatal << "Fail to deinit system manager of face registration.";
            return APP_ERR_COMM_FAILURE;
        }
    }

    return ret;
}

#endif //  IDRECOGNITION_COMMONFUCTION_H
