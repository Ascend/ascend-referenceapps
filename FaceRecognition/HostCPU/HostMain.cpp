/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
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
#include <cstring>
#include <unistd.h>

#include "CommandLine.h"
#include "FaceFeatureLib/FaceFeatureLib.h"
#include "Log/Log.h"
#include "AsynLog/AsynLog.h"
#include "SystemManager/SystemManager.h"
#include "Statistic/Statistic.h"
#include "DeliveryChannel/DeliveryChannel.h"

#include "ChannelStatus/ChannelStatus.h"

using namespace ascendFaceRecognition;

namespace {
const std::string PIPELINE_SEARCH = "Search";
const int SEARCH_MODULE_TYPE_COUNT = 14;
const int SEARCH_MODULE_CONNECT_COUNT = 13;

const std::string PIPELINE_REGIST = "Regist";
const int REGIST_MODULE_TYPE_COUNT = 9;
const int REGIST_MODULE_CONNECT_COUNT = 8;
const int SIGNAL_CHECK_TIMESTEP = 10000;
}

static bool signalRecieved = false;

static ModuleDesc searchModuleDesc[SEARCH_MODULE_TYPE_COUNT] = {
    { -1, MT_STREAM_PULLER, "StreamPuller"},
    { -1, MT_VIDEO_DECODER, "VideoDecoder" }, // 16 x 25/3 x 10
    { 4, MT_VIDEO_RESIZE, "VideoResize" }, // 16 x 25/3 x 10
    { 4, MT_FACE_DETECTION, "FaceDetection"},
    { 8, MT_FACE_RESIZE, "FaceResize"},
    { -1, MT_MOT_EMBEDDING, "MOTEmbedding"},
    { -1, MT_MOT_CONNECTION, "MOTConnection"},
    { -1, MT_QUALITY_EVALUATION, "QualityEvaluation"},
    { 8, MT_WARP_AFFINE, "WarpAffine"},
    { 4, MT_FACE_ATTRIBUTE, "FaceAttribute"},
    { 4, MT_FACE_FEATURE, "FaceFeature"},
    { 4, MT_FACE_SEARCH, "FaceSearch" },
    { 1, MT_FACE_DETAIL_INFO, "FaceDetailInfo"},
    {-1, MT_FRAME_ALIGN, "FrameAlign"},

};

static ModuleConnectDesc searchConnectDesc[SEARCH_MODULE_CONNECT_COUNT] = {
    {MT_STREAM_PULLER, MT_VIDEO_DECODER, MODULE_CONNECT_CHANNEL},
    {MT_VIDEO_DECODER, MT_VIDEO_RESIZE, MODULE_CONNECT_CHANNEL},
    {MT_VIDEO_RESIZE, MT_FACE_DETECTION, MODULE_CONNECT_CHANNEL},
    {MT_FACE_DETECTION, MT_FACE_RESIZE, MODULE_CONNECT_CHANNEL},
    {MT_FACE_RESIZE, MT_MOT_EMBEDDING, MODULE_CONNECT_CHANNEL},
    {MT_MOT_EMBEDDING, MT_MOT_CONNECTION, MODULE_CONNECT_CHANNEL},
    {MT_MOT_CONNECTION, MT_QUALITY_EVALUATION, MODULE_CONNECT_CHANNEL},
    {MT_QUALITY_EVALUATION, MT_WARP_AFFINE, MODULE_CONNECT_CHANNEL},
    {MT_WARP_AFFINE, MT_FACE_ATTRIBUTE, MODULE_CONNECT_CHANNEL},
    {MT_FACE_ATTRIBUTE, MT_FACE_FEATURE, MODULE_CONNECT_CHANNEL},
    {MT_FACE_FEATURE, MT_FACE_SEARCH, MODULE_CONNECT_CHANNEL},
    {MT_FACE_SEARCH, MT_FACE_DETAIL_INFO, MODULE_CONNECT_CHANNEL},
    {MT_MOT_CONNECTION, MT_FRAME_ALIGN, MODULE_CONNECT_CHANNEL},
};

static ModuleDesc registModuleDesc[REGIST_MODULE_TYPE_COUNT] = {
    { 1, MT_IMAGE_READER, "ImageReader"},
    { 1, MT_IMAGE_DECODER, "ImageDecoder" }, // 16 x 25/3 x 10
    { 1, MT_FACE_DETECTION, "FaceDetection"},
    { 1, MT_FACE_RESIZE, "FaceResize"},
    { 1, MT_FACE_LANDMARK, "FaceLandmark"},
    { 1, MT_WARP_AFFINE, "WarpAffine"},
    { 1, MT_FACE_FEATURE, "FaceFeature"},
    { 1, MT_FACE_STOCK, "FaceStock"},
    { 1, MT_REG_RESULT_HANDLER, "RegResultHandler"}
};

static ModuleConnectDesc registConnectDesc[REGIST_MODULE_CONNECT_COUNT] = {
    {MT_IMAGE_READER, MT_IMAGE_DECODER, MODULE_CONNECT_CHANNEL},
    {MT_IMAGE_DECODER, MT_FACE_DETECTION, MODULE_CONNECT_CHANNEL},
    {MT_FACE_DETECTION, MT_FACE_RESIZE, MODULE_CONNECT_CHANNEL},
    {MT_FACE_RESIZE, MT_FACE_LANDMARK, MODULE_CONNECT_CHANNEL},
    {MT_FACE_LANDMARK, MT_WARP_AFFINE, MODULE_CONNECT_CHANNEL},
    {MT_WARP_AFFINE, MT_FACE_FEATURE, MODULE_CONNECT_CHANNEL},
    {MT_FACE_FEATURE, MT_FACE_STOCK, MODULE_CONNECT_CHANNEL},
    {MT_FACE_STOCK, MT_REG_RESULT_HANDLER, MODULE_CONNECT_CHANNEL}
};

static void SigHandler(int signal)
{
    if (signal == SIGINT) {
        signalRecieved = true;
    }
}

static APP_ERROR LoadSearchPipeline(int channelCount)
{
    APP_ERROR ret = SystemManager::GetInstance()->RegisterModules(PIPELINE_SEARCH.c_str(), 
        searchModuleDesc, SEARCH_MODULE_TYPE_COUNT, channelCount);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to regist module.";
        return APP_ERR_COMM_FAILURE;
    }

    ret = SystemManager::GetInstance()->RegisterModuleConnects(PIPELINE_SEARCH.c_str(), 
        searchConnectDesc, SEARCH_MODULE_CONNECT_COUNT);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to connect module.";
        return APP_ERR_COMM_FAILURE;
    }
    return APP_ERR_OK;
}

static APP_ERROR LoadRegistPipeline(int channelCount)
{
    APP_ERROR ret = SystemManager::GetInstance()->RegisterModules(PIPELINE_REGIST.c_str(), 
        registModuleDesc, REGIST_MODULE_TYPE_COUNT, channelCount);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to regist module.";
        return APP_ERR_COMM_FAILURE;
    }

    ret = SystemManager::GetInstance()->RegisterModuleConnects(PIPELINE_REGIST.c_str(), 
        registConnectDesc, REGIST_MODULE_CONNECT_COUNT);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to connect module.";
        return APP_ERR_COMM_FAILURE;
    }

    return APP_ERR_OK;
}


static APP_ERROR ParseConfig(std::string &configPath, SystemInitArgs &initArgs, int &deviceId, uint32_t &channelCount)
{
    ConfigParser configParser;
    // 1. create queues of receivers and senders
    APP_ERROR ret = configParser.ParseConfig(configPath);
    if (ret != APP_ERR_OK) {
        LogFatal << "Cannot parse file.";
        return ret;
    }

    std::string itemCfgStr = std::string("SystemConfig.channelCount");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, channelCount);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    if (channelCount <= 0) {
        LogFatal << "Invalid channel count.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    // 2. create HDC sessions
    itemCfgStr = "SystemConfig.deviceId";
    ret = configParser.GetIntValue(itemCfgStr, deviceId);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to get device id.";
        return ret;
    }

    return ret;
}

static APP_ERROR InitSystemManager(std::string &configPath, std::string &aclConfigPath, SystemInitArgs &initArgs)
{
    LogDebug << "Begin to init system manager.";
    int deviceId = -1;
    uint32_t channelCount = 0;
    APP_ERROR ret = ParseConfig(configPath, initArgs, deviceId, channelCount);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to parse " << configPath << ".";
        return ret;
    }

    LogDebug << "Begin to init DeliveryChannel Regist.";
    for (uint32_t i = 1; i <= channelCount + 1; i++) {
        auto queue = std::make_shared<BlockingQueue<std::shared_ptr<void>>>(MODULE_QUEUE_SIZE);
        ret = DeliveryChannel::GetInstance()->Regist(DELIVERY_VIDEO_STREAM_INDEX + i, queue);
        if (ret != APP_ERR_OK) {
            return ret;
        }
    }
    auto queuePicture = std::make_shared<BlockingQueue<std::shared_ptr<void>>>(MODULE_QUEUE_SIZE);
    ret = DeliveryChannel::GetInstance()->Regist(DELIVERY_PICTURE_INDEX, queuePicture);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    if (ret != APP_ERR_OK) {
        return ret;
    }
    LogDebug << "Begin to init SystemManager Init.";
    // 3. init system manager
    ret = SystemManager::GetInstance()->Init(configPath, initArgs, aclConfigPath);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to init system manager.";
        return APP_ERR_COMM_FAILURE;
    }

    ret = LoadSearchPipeline(channelCount);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to LoadSearchPipeline.";
        return APP_ERR_COMM_FAILURE;
    }

    ret = LoadRegistPipeline(channelCount);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to LoadRegistPipeline.";
        return APP_ERR_COMM_FAILURE;
    }
    return ret;
}

static APP_ERROR DeInitSystemManager(SystemInitArgs &initArgs)
{
    LogInfo << "Begin to deinit system manager.";

    // 1. Deinit system manager
    APP_ERROR ret = SystemManager::GetInstance()->DeInit();
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to deinit system manager.";
        return APP_ERR_COMM_FAILURE;
    }
    return ret;
}

int main(int argc, const char *argv[])
{
    Statistic::statisticEnable = true;
    LogInfo << "Face Recognition demo running.";
    CmdParams cmdParams;
    SystemInitArgs initArgsSearch;
    // 0. parse command line and set log level
    if (ParseACommandLine(argc, argv, cmdParams) != APP_ERR_OK) {
        return -1;
    }
    SetLogLevel(cmdParams.debugLevel);

    // start log thread
    AsynLog::GetInstance().Run();

    // 1. create face lib and feature lib
    FaceFeatureLib faceFeature;
    APP_ERROR ret = faceFeature.FaceFeatureLibInit();
    if (ret != APP_ERR_OK) {
        return APP_ERR_COMM_FAILURE;
    }

    // 2. init system manager of face search
    initArgsSearch.faceFeature = &faceFeature;

    LogInfo << "Init System Manager.";
    ret = InitSystemManager(cmdParams.Config, cmdParams.aclConfig, initArgsSearch);
    if (ret != APP_ERR_OK) {
        LogFatal << "Fail to Init System Manager.";
        return APP_ERR_COMM_FAILURE;
    }

    // 3. run pipeline
    LogInfo << "Run pipeline.";
    ret = SystemManager::GetInstance()->RunPipeline();
    if (ret != APP_ERR_OK) {
        return APP_ERR_COMM_FAILURE;
    }

    // 4. wait for exit signal
    LogInfo << "wait for exit signal.";
    if (signal(SIGINT, SigHandler) == SIG_ERR) {
        LogFatal << "cannot catch SIGINT.";
    }
    while (!signalRecieved) {
        usleep(SIGNAL_CHECK_TIMESTEP);
    }

    // 5. deinit system manager
    // 5.1 deinit system manager of face search
    LogInfo << "deinit system manager of face registration.";
    ret = DeInitSystemManager(initArgsSearch);
    if (ret != APP_ERR_OK) {
        return APP_ERR_COMM_FAILURE;
    }

    LogInfo << "Demo ended successfully.";

    // 6. stop AsynLog
    // output all log data before main thread exit.
    while (!AsynLog::GetInstance().IsLogQueueEmpty()) {
        continue;
    }
    AsynLog::GetInstance().Stop();

    return 0;
}
