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
#include "FaceFeatureLib.h"

#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <sstream>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <cassert>

#include "FileManager/FileManager.h"
#include "Log/Log.h"

namespace ascendFaceRecognition {
const int FACE_FEATURE_SIZE = 2048;
const int FACE_FEATURE_NUM = FACE_FEATURE_SIZE / sizeof(float);
const int CHANNEL = 3;
const int CONST_NUM_100 = 100;
const int CONST_NUM_40 = 40;
const int CONST_NUM_20 = 20;
const int CONST_NUM_10 = 10;
const int CONST_NUM_5 = 5;
const float COS_TO_SIMILARITY_MOD = 2.0;
const float COS_TO_SIMILARITY_OFFSET = 0.5;


APP_ERROR FaceFeatureLib::FaceFeatureLibInit()
{
    LogInfo << "FaceFeatureLib: begain to init FaceFeatureLib.";
    // init variable
    featureLib_.clear();
    files_.clear();
    feactureLibPath_ = "./featureLib";
    CreateDirRecursively(feactureLibPath_);
    // read feature binary files list
    int feactureNum = GetAllFormatFiles(feactureLibPath_, files_, ".feature");
    LogDebug << "FaceFeatureLib: feactureLib has " << feactureNum << " features.";
    for (size_t i = 0; i < files_.size(); ++i) {
        std::string uuid = std::to_string(i);
        std::shared_ptr<FaceFeatureInfo> featureInfo = std::make_shared<FaceFeatureInfo>();
        // construct feature map binary file path
        std::string fileName = feactureLibPath_ + "/" + uuid + ".feature";
        int sizeFeature = 0;
        int ret = ReadBinaryFile(fileName, featureInfo->featureValue, sizeFeature);
        if ((ret != APP_ERR_OK) || (sizeFeature != FACE_FEATURE_SIZE)) {
            LogError << "failed to read binary feature file " << fileName;
            continue;
        }
        // get features norm value
        featureInfo->norm = CalcVectorNorm((float *)featureInfo->featureValue.get());
        // push featureInfo to list
        featureLib_.push_back(featureInfo);
    }
    ShownAllFeatures();
    return APP_ERR_OK;
}


int FaceFeatureLib::GetAllFormatFiles(std::string path, std::vector<std::string> &files, std::string format) const
{
    int iFileCnt = 0;
    DIR *dirptr = nullptr;
    struct dirent *dirp = nullptr;

    if ((dirptr = opendir(path.c_str())) == nullptr) { // open directory
        LogError << "FaceFeatureLib: open path fail:" << path << ".";
        return 0;
    }
    while ((dirp = readdir(dirptr)) != nullptr) {
        if ((dirp->d_type == DT_REG) && (strcmp(strchr(dirp->d_name, '.'), format.c_str())) == 0) {
            files.push_back(dirp->d_name);
            iFileCnt++;
        }
    }
    closedir(dirptr);

    return iFileCnt;
}


APP_ERROR FaceFeatureLib::InsertFeatureToLib(const DataBuffer &featureVector, const PersonInfo &personInfo,
    const ImageInfo &imageInfo, bool isForce)
{
    LogDebug << "FaceFeatureLib: begain to insert feature to lib, name=" << personInfo.name << ".";
    if (featureVector.dataSize != FACE_FEATURE_SIZE) {
        LogError << "FaceFeatureLib: face feature value is not correct, dataSize=" << featureVector.dataSize << ".";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (!isForce) {
        float similarity = 0;
        int index = GetMaxSimilarity((float *)featureVector.hostData.get(), similarity);
        LogDebug << "FaceFeatureLib:similarity=" << similarity << ", index=" << index << ".";
        if (index >= 0 && similarity > REG_SIMILARITY_THRESHOLD) {
            LogDebug << "FaceFeatureLib: the same picture already in lib, index=" << index;
            return APP_ERROR_FACE_WEB_USE_REPEAT_REG;
        }
    }

    // get uuid
    std::string uuid = std::to_string(featureLib_.size());
    // Generates a feature vector file. If no file exists, an empty file is generated. If there is a file, clear it.
    std::string fileName = feactureLibPath_ + "/" + uuid + ".feature";
    char *streamS = reinterpret_cast<char *>(featureVector.hostData.get());
    std::string featureData(streamS, streamS + featureVector.dataSize);
    mode_t newUmask = 0077;
    mode_t oldUmask = umask(newUmask);
    SaveFileOverwrite(fileName, featureData, featureVector.dataSize);

    // save personInfo, write to json
    fileName = feactureLibPath_ + "/" + uuid + ".config";
    ConfigParser configParser;
    configParser.NewConfig(fileName);
    configParser.WriteString("uuid", uuid);
    configParser.WriteString("name", personInfo.name);
    configParser.WriteString("gender", personInfo.gender);
    configParser.WriteInt("age", personInfo.age);
    configParser.WriteUint32("format", imageInfo.format);
    configParser.WriteUint32("width", imageInfo.width);
    configParser.WriteUint32("height", imageInfo.height);
    configParser.SaveConfig();

    fileName = feactureLibPath_ + "/" + uuid + ".bgr";
    int length = (int)imageInfo.height * (int)imageInfo.width * CHANNEL;
    streamS = reinterpret_cast<char *>(imageInfo.buf.hostData.get());
    std::string imageData(streamS, streamS + length);
    SaveFileOverwrite(fileName, imageData, length);
    umask(oldUmask);

    // update featureLib
    std::shared_ptr<FaceFeatureInfo> featureInfo = std::make_shared<FaceFeatureInfo>();
    FeatureInfoCopy(featureInfo->featureValue, featureVector);
    featureInfo->norm = CalcVectorNorm((float *)featureInfo->featureValue.get());
    featureLib_.push_back(featureInfo);

    LogDebug << "FaceFeatureLib: It is successfull to insert " << personInfo.name << " information to lib.";
    return APP_ERR_OK;
}


void FaceFeatureLib::FeatureInfoCopy(std::shared_ptr<uint8_t> &dstFeatureInfo, const DataBuffer &srcFeatureInfo)
{
    if (srcFeatureInfo.dataSize != FACE_FEATURE_SIZE) {
        LogError << "srcFeatureInfo.dataSize=" << srcFeatureInfo.dataSize << "is not " << FACE_FEATURE_SIZE;
        return;
    }
    std::shared_ptr<uint8_t> featureValue = std::make_shared<uint8_t>();
    featureValue.reset(new uint8_t[FACE_FEATURE_SIZE], std::default_delete<uint8_t[]>());
    std::copy(srcFeatureInfo.hostData.get(), srcFeatureInfo.hostData.get() + srcFeatureInfo.dataSize,
              featureValue.get());
    dstFeatureInfo = featureValue;
    return;
}


int FaceFeatureLib::GetMaxSimilarity(float *featureData, float &similarity)
{
    float featureNorm = CalcVectorNorm(featureData);
    similarity = 0;
    int iRecord = -1;
    for (size_t i = 0; i < featureLib_.size(); ++i) {
        LogDebug << "FaceFeatureLib: to match " << i << " lib size=" << featureLib_.size() << ".";
        float vectorDot = CalcVectorDot(featureData, (float *)featureLib_[i]->featureValue.get());
        if ((featureNorm == 0) || (featureLib_[i]->norm == 0)) {
            LogError << "FaceFeatureLib: the norm is 0.";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        float tempSimilarity = vectorDot / featureNorm / featureLib_[i]->norm;
        tempSimilarity = tempSimilarity / COS_TO_SIMILARITY_MOD + COS_TO_SIMILARITY_OFFSET;
        LogDebug << "FaceFeatureLib: vectorDot=" << vectorDot << ", featureNorm=" << featureNorm
                 << ", norm=" << featureLib_[i]->norm << ", tempSimilarity=" << tempSimilarity;
        if (similarity < tempSimilarity) {
            similarity = tempSimilarity;
            iRecord = i;
        }
    }
    return iRecord;
}


APP_ERROR FaceFeatureLib::GetPersonInfoByFeature(const DataBuffer &featureVector, PersonInfo &personInfo,
    ImageInfo &imageInfo)
{
    LogDebug << "FaceFeatureLib: beagin to get bestH information by feature value, lib size=" << featureLib_.size();
    if (featureVector.dataSize != FACE_FEATURE_SIZE) {
        LogError << "FaceFeatureLib: face feature value is not correct, dataSize=" << featureVector.dataSize;
        return APP_ERR_COMM_INVALID_PARAM;
    }

    if (featureLib_.empty()) {
        LogWarn << "FaceFeatureLib: Face Feature Lib is empty, retrun unkown information";
        personInfo.uuid = "unkown";
        personInfo.name = "unkown";
        personInfo.gender = "unkown";
        personInfo.age = 0;
        personInfo.similarity = 0.0;
        imageInfo.buf.dataSize = 0;
        imageInfo.height = 0;
        imageInfo.width = 0;
        return APP_ERR_OK;
    }

    float similarity = 0;
    int bestPersonIndex = GetMaxSimilarity(reinterpret_cast<float *>(featureVector.hostData.get()), similarity);
    if (bestPersonIndex < 0) {
        LogError << "FaceFeatureLib: get humman information failed.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    LogDebug << "FaceFeatureLib: find the best " << bestPersonIndex << ", similarity=" << similarity;
    // config file
    std::string uuid = std::to_string(bestPersonIndex);
    APP_ERROR ret = GetConfigContext(uuid, personInfo, imageInfo);
    if (ret != APP_ERR_OK) {
        LogError << "get information from config file fail, uuid=" << uuid;
        return APP_ERR_COMM_FAILURE;
    }
    personInfo.similarity = similarity;
    if (similarity < SEARCH_SIMILARITY_THRESHOLD) {
        personInfo.name = personInfo.name + "-Uncertain";
    }

    // id image
    std::string fileName = feactureLibPath_ + "/" + uuid + ".bgr";
    int sizeFeature = 0;
    ret = ReadBinaryFile(fileName, imageInfo.buf.hostData, sizeFeature);
    imageInfo.buf.dataSize = static_cast<uint32_t>(sizeFeature);
    if (ret != APP_ERR_OK) {
        LogError << "failed to read binary file " << fileName;
        return ret;
    }

    LogDebug << "FaceFeatureLib: It is successful to get person information " << personInfo.name;
    return APP_ERR_OK;
}

APP_ERROR FaceFeatureLib::GetConfigContext(std::string &uuid, PersonInfo &personInfo, ImageInfo &imageInfo)
{
    std::string fileName = feactureLibPath_ + "/" + uuid + ".config";
    ConfigParser configParser;
    APP_ERROR ret = configParser.ParseConfig(fileName);
    if (ret != APP_ERR_OK) {
        LogError << "parse config file fail, fileName=" << fileName;
        return APP_ERR_COMM_FAILURE;
    }
    ret = configParser.GetStringValue("uuid", personInfo.uuid);
    if (ret != APP_ERR_OK) {
        LogError << "get uuid fail";
        return APP_ERR_COMM_FAILURE;
    }
    ret = configParser.GetStringValue("name", personInfo.name);
    if (ret != APP_ERR_OK) {
        LogError << "get name fail";
        return APP_ERR_COMM_FAILURE;
    }
    ret = configParser.GetStringValue("gender", personInfo.gender);
    if (ret != APP_ERR_OK) {
        LogError << "get gender fail";
        return APP_ERR_COMM_FAILURE;
    }
    ret = configParser.GetUnsignedIntValue("age", personInfo.age);
    if (ret != APP_ERR_OK) {
        LogError << "get age fail";
        return APP_ERR_COMM_FAILURE;
    }
    ret = configParser.GetUnsignedIntValue("format", imageInfo.format);
    if (ret != APP_ERR_OK) {
        LogError << "get format fail";
        return APP_ERR_COMM_FAILURE;
    }
    ret = configParser.GetUnsignedIntValue("width", imageInfo.width);
    if (ret != APP_ERR_OK) {
        LogError << "get width fail";
        return APP_ERR_COMM_FAILURE;
    }
    ret = configParser.GetUnsignedIntValue("height", imageInfo.height);
    if (ret != APP_ERR_OK) {
        LogError << "get height fail";
        return APP_ERR_COMM_FAILURE;
    }
    return APP_ERR_OK;
}


float FaceFeatureLib::CalcVectorNorm(const float *vector)
{
    if (vector == NULL) {
        LogError << "FaceFeatureLib: input pointer pVector is null.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    float norm = 0;
    for (int i = 0; i < FACE_FEATURE_NUM; ++i) {
        norm += vector[i] * vector[i];
    }
    return sqrt(norm);
}


float FaceFeatureLib::CalcVectorDot(const float *vector1, const float *vector2)
{
    if ((vector1 == NULL) or (vector2 == NULL)) {
        LogError << "input pointer pointer is null, vector1=" << vector1 << ", vector2=" << vector2;
        return APP_ERR_COMM_INVALID_PARAM;
    }

    float dot = 0;
    for (int i = 0; i < FACE_FEATURE_NUM; ++i) {
        dot += vector1[i] * vector2[i];
    }
    return dot;
}


void FaceFeatureLib::ShownAllFeatures()
{
    std::string splitString(CONST_NUM_100, '*');
    LogDebug << std::endl << std::endl;
    LogDebug << splitString;
    LogDebug << "there are " << featureLib_.size() << " features";
    std::string uuidTitle("uuid");
    std::string nameTitle("name");
    std::string genderTitle("gender");
    std::string ageTitle("age");
    std::string normTitle("norm");

    std::string uuidTitleSpace(CONST_NUM_40 - uuidTitle.size(), ' ');
    std::string nameSpace(CONST_NUM_20 - nameTitle.size(), ' ');
    std::string genderSpace(CONST_NUM_10 - genderTitle.size(), ' ');
    std::string ageSpace(CONST_NUM_5 - ageTitle.size(), ' ');
    LogDebug << uuidTitle << uuidTitleSpace << nameTitle << nameSpace << genderTitle << genderSpace << ageTitle
             << ageSpace << normTitle;
    for (size_t i = 0; i < featureLib_.size(); ++i) {
        std::string fileName = feactureLibPath_ + "/" + std::to_string(i) + ".config";
        ConfigParser configParser;
        configParser.ParseConfig(fileName);
        std::string uuid;
        std::string name;
        std::string gender;
        uint32_t age = {};
        configParser.GetStringValue("uuid", uuid);
        configParser.GetStringValue("name", name);
        configParser.GetStringValue("gender", gender);
        configParser.GetUnsignedIntValue("age", age);

        std::string uuidSpace(CONST_NUM_40 - uuid.size(), ' ');
        std::string nameSpace(CONST_NUM_20 - name.size(), ' ');
        std::string genderSpace(CONST_NUM_10 - gender.size(), ' ');
        std::string ageSpace(CONST_NUM_5 - std::to_string(age).size(), ' ');

        LogDebug << uuid << uuidSpace << name << nameSpace << gender << genderSpace << age << ageSpace
                 << featureLib_[i]->norm;
    }
    LogDebug << splitString << std::endl << std::endl;
    return;
}
} // namespace ascendFaceRecognition
