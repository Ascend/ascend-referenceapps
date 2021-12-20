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
#include "FaceSearch.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "FileManager/FileManager.h"
#include "Log/Log.h"
#include "ErrorCode/ErrorCode.h"
#include "Statistic/Statistic.h"

namespace ascendFaceRecognition {
const int LAND_MARK_SIZE = 10;
const int MOD2 = 2;
const int DEFAULT_LENGTH = 4;
const int CHANNEL = 3;

FaceSearch::FaceSearch() {}

FaceSearch::~FaceSearch() {}

APP_ERROR FaceSearch::ParseConfig(ConfigParser &configParser) const
{
    int ret = APP_ERR_OK;
    return ret;
}

APP_ERROR FaceSearch::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "FaceSearch[" << instanceId_ << "]: FaceSearch init start.";

    AssignInitArgs(initArgs);
    // get feature lib
    faceFeatureLib_ = (FaceFeatureLib *)initArgs.userData;

    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceSearch[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }

    isStop_ = false;
    LogInfo << "FaceSearch[" << instanceId_ << "]: FaceSearch init success.";
    return APP_ERR_OK;
}

APP_ERROR FaceSearch::DeInit()
{
    LogInfo << "FaceSearch[" << instanceId_ << "]: FaceSearch deinit start.";

    StopAndDestroyQueue();

    outputQueVec_[instanceId_]->Clear();

    LogInfo << "FaceSearch[" << instanceId_ << "]: FaceSearch deinit success.";

    return APP_ERR_OK;
}

void FaceSearch::SaveConfigFile(unsigned int i, const FaceObject &face, const std::string &fileName) const
{
    LogDebug << "FaceSearch[" << instanceId_ << "]: begin to save face search config result.";
    ConfigParser configParser;
    configParser.NewConfig(fileName);
    configParser.WriteUint32("format", face.imgSearch.format);
    configParser.WriteUint32("imgWidth", face.imgSearch.width);
    configParser.WriteUint32("imgHeight", face.imgSearch.height);
    configParser.WriteUint32("faceId", i);
    configParser.WriteString("uuid", face.personInfo.uuid);
    configParser.WriteString("name", face.personInfo.name);
    configParser.WriteString("gender", face.personInfo.gender);
    configParser.WriteUint32("age", face.personInfo.age);
    configParser.WriteFloat("similarity", face.personInfo.similarity);
    configParser.WriteUint32("classId", face.info.classId);
    configParser.WriteFloat("confidence", face.info.confidence);
    configParser.WriteFloat("minx", face.info.minx);
    configParser.WriteFloat("miny", face.info.miny);
    configParser.WriteFloat("height", face.info.height);
    configParser.WriteFloat("width", face.info.width);

    float *landmarks = static_cast<float *>((void *)face.landmarks.hostData.get());
    for (int j = 0; j < LAND_MARK_SIZE; ++j) {
        int index = j / MOD2;
        std::string key;
        if (j % MOD2 == 0) {
            key = "x" + std::to_string(index);
        } else {
            key = "y" + std::to_string(index);
        }
        configParser.WriteFloat(key, landmarks[j]);
    }
    configParser.SaveConfig();
    LogDebug << "FaceSearch[" << instanceId_ << "]: saving config file by overwrite: fileName=" << fileName;
}

void FaceSearch::SaveResult(std::shared_ptr<FrameAiInfo> frameAiInfo) const
{
    LogDebug << "FaceSearch[" << instanceId_ << "]: begin to save face search result.";
    std::string rootPath = "./result";
    std::string filePath = rootPath + "/ChannelId" + std::to_string(frameAiInfo->info.channelId) + "/FrameId" +
        std::to_string(frameAiInfo->info.frameId);
    CreateDirRecursively(filePath);
    std::string fileFix = "/ChannelId" + std::to_string(frameAiInfo->info.channelId) + "_FrameId" +
        std::to_string(frameAiInfo->info.frameId);
    std::string fileName = filePath + fileFix + ".jpg";
    LogDebug << "FaceSearch["
             << "]: fileName:" << fileName;
#ifdef HOST_CPU_SOLUTION
    mode_t newUmaskJpg = 0077;
    mode_t oldUmaskJpg = umask(newUmaskJpg);
    CopyFile(frameAiInfo->info.imagePath, fileName);
    umask(oldUmaskJpg);
#else
#ifdef  CTRL_CPU_SOLUTION
    mode_t newUmaskJpg = 0077;
    mode_t oldUmaskJpg = umask(newUmaskJpg);
    CopyFile(frameAiInfo->info.imagePath, fileName);
    umask(oldUmaskJpg);
#else
#endif
#endif
    for (unsigned int i = 0; i < frameAiInfo->face.size(); ++i) {
        fileName = filePath + fileFix + "_Face" + std::to_string(i) + ".config";
        mode_t newUmask = 0077;
        mode_t oldUmask = umask(newUmask);
        SaveConfigFile(i, frameAiInfo->face[i], fileName);
        fileName = filePath + fileFix + "_Face" + std::to_string(i) + ".bgr";
        int length = (int)frameAiInfo->face[i].imgSearch.height * (int)frameAiInfo->face[i].imgSearch.width * CHANNEL;
        char *streamS = reinterpret_cast<char *>(frameAiInfo->face[i].imgSearch.buf.hostData.get());
        std::string frameData(streamS, streamS + length);
        SaveFileOverwrite(fileName, frameData, length);
        umask(oldUmask);
    }
    LogDebug << "FaceSearch[" << instanceId_ << "]: It is successful to save face search result.";
}

APP_ERROR FaceSearch::Process(std::shared_ptr<void> inputData)
{
    faceSearchStatic_.RunTimeStatisticStart("FaceSearch_Excute_Time", instanceId_);
    LogDebug << "FaceSearch[" << instanceId_ << "]: Begin to process data.";
    APP_ERROR ret = APP_ERR_OK;
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::static_pointer_cast<FrameAiInfo>(inputData);
    if (frameAiInfo.get() == nullptr) {
        return APP_ERR_COMM_INVALID_POINTER;
    }

    LogDebug << "FaceSearch[" << instanceId_ << "]: width=" << frameAiInfo->imgOrigin.width
             << ",height=" << frameAiInfo->imgOrigin.height;

    for (unsigned int i = 0; i < frameAiInfo->face.size(); i++) {
        ret = faceFeatureLib_->GetPersonInfoByFeature(frameAiInfo->face[i].featureVector,
            frameAiInfo->face[i].personInfo, frameAiInfo->face[i].imgSearch);
        if (ret != APP_ERR_OK) {
            LogError << "FaceSearch[" << instanceId_ << "]: Fail to get person info.";
            return ret;
        }
    }

#ifdef HOST_CPU_SOLUTION
    SaveResult(frameAiInfo);
    Statistic::GlobalTimeStatisticStop();
#else
#ifdef  CTRL_CPU_SOLUTION
    outputQueVec_[instanceId_]->Push(frameAiInfo, true);
#else
    SaveResult(frameAiInfo);
    Statistic::GlobalTimeStatisticStop();
    outputQueVec_[instanceId_]->Push(frameAiInfo, true);
#endif
#endif
    successCount_++;
    faceSearchStatic_.RunTimeStatisticStop();
    return ret;
}

double FaceSearch::GetRunTimeAvg()
{
    return faceSearchStatic_.GetRunTimeAvg();
}

int FaceSearch::GetSuccessCount() const
{
    return successCount_;
}

void FaceSearch::SetSuccessCount()
{
    successCount_ = 0;
}

APP_ERROR FaceSearch::ResetFaceFeatureLib() const
{
    return faceFeatureLib_->FaceFeatureLibInit();
}
} // End ascendFaceRecognition
