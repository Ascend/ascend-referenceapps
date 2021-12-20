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

#include "FileEx/FileEx.h"
#include "FileManager/FileManager.h"
#include "Log/Log.h"

static std::vector<std::string> Split(const std::string &src, const std::string &pattern)
{
    std::vector<std::string> res;
    if (src == "")
        return res;
    std::string strs = src + pattern;
    size_t pos = strs.find(pattern);

    while (pos != strs.npos) {
        std::string temp = strs.substr(0, pos);
        if (temp != "")
            res.push_back(temp);
        strs = strs.substr(pos + 1, strs.size());
        pos = strs.find(pattern);
    }
    return res;
}

namespace ascendFaceRecognition {
namespace {
const int FACE_FEATURE_SIZE = 1024;
const int FACE_FEATURE_NUM = FACE_FEATURE_SIZE / sizeof(float);
const float NORM_EPS = 1e-10;
}

/*
 * Description: initialize function
 */
APP_ERROR FaceFeatureLib::FaceFeatureLibInit()
{
    LogDebug << "FaceFeatureLib: begain to init FaceFeatureLib.";
    std::lock_guard<std::mutex> guard(mtx_);
    // Initialize Variables
    feactureLibPath_ = "./featureLib";
    mkdir(feactureLibPath_.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    // Reads the binary file list of feature values. The file name is the UUID
    int feactureNum = GetAllFormatFiles(feactureLibPath_, files_, ".feature");
    LogDebug << "FaceFeatureLib: feactureLib has " << feactureNum << " features.";
    for (size_t i = 0; i < files_.size(); ++i) {
        std::vector<std::string> stringList = Split(files_[i], ".feature");
        // feature information
        std::shared_ptr<FaceFeatureInfo> featureInfo = std::make_shared<FaceFeatureInfo>();
        // construct the binary file path of the feature vector
        std::string fileName = feactureLibPath_ + "/" + stringList[0] + ".feature";
        int sizeFeature = 0;
        int ret = FileEx::ReadBinaryFile(fileName, featureInfo->featureValue, sizeFeature);
        if ((ret != APP_ERR_OK) || (sizeFeature != FACE_FEATURE_SIZE)) {
            LogError << "failed to read binary feature file " << fileName;
            continue;
        }
        // get feature modulus information to reduce calculation workload
        featureInfo->norm = CalcVectorNorm((float *)featureInfo->featureValue.get());

        // get uuid
        std::string configFile = feactureLibPath_ + "/" + stringList[0] + ".config";
        ConfigParser configParser;
        ret = configParser.ParseConfig(configFile);
        if (ret != APP_ERR_OK) {
            LogError << "FaceFeatureLib: " << configFile << " is not existed\n";
            return APP_ERR_COMM_NO_EXIST;
        }
        std::string uuid;
        ret = configParser.GetStringValue("uuid", uuid);
        if (ret != APP_ERR_OK) {
            LogError << "FaceFeatureLib: get message error\n";
            return APP_ERR_COMM_NO_EXIST;
        }
        featureInfo->uuid = uuid;
        // insert to uuid set
        uuidBase_.insert(uuid);
        // push into list
        featureLib_.push_back(featureInfo);
    }

    ShownAllFeatures();
    return APP_ERR_OK;
}

/*
 * Description: get the file name in a specified format and returns the number of found files. The file name does not
 */
int FaceFeatureLib::GetAllFormatFiles(const std::string &path, std::vector<std::string> &files,
    const std::string &format) const
{
    int iFileCnt = 0;
    DIR *dirptr = nullptr;
    struct dirent *dirp = nullptr;

    if ((dirptr = opendir(path.c_str())) == nullptr) { // open content
        LogError << "FaceFeatureLib: open path fail:" << path << ".";
        return 0;
    }
    while ((dirp = readdir(dirptr)) != nullptr) {
        // check whether the file is a file and the file name extension
        if ((dirp->d_type == DT_REG) && (strcmp(strchr(dirp->d_name, '.'), format.c_str())) == 0) {
            files.push_back(dirp->d_name);
            iFileCnt++;
        }
    }
    closedir(dirptr);

    return iFileCnt;
}

/*
 * Description: Add library content and forcibly search for isForce
 */
APP_ERROR FaceFeatureLib::InsertFeatureToLib(std::shared_ptr<FaceObject> faceObject, bool isForce)
{
    std::lock_guard<std::mutex> guard(mtx_);
    if (faceObject->featureVector.dataSize != FACE_FEATURE_SIZE) {
        LogError << "FaceFeatureLib: face feature value is not correct, dataSize=" <<
            faceObject->featureVector.dataSize << ".";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    if (uuidBase_.find(faceObject->personInfo.uuid) != uuidBase_.end()) {
        LogError << "FaceFeatureLib: uuid=" << faceObject->personInfo.uuid << " is already in dataset\n";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    if (!isForce) {
        float similarity = 0;
        int index = GetMaxSimilarity(faceObject, similarity);
        LogDebug << "FaceFeatureLib:similarity=" << similarity << ", index=" << index << ".";
        if (index >= 0 && similarity > REG_SIMILARITY_THRESHOLD) {
            LogDebug << "FaceFeatureLib: the same picture already in lib, index=" << index;
            return APP_ERR_OK;
        }
    }

    // generating a feature vector file
    std::string fileName = feactureLibPath_ + "/" + faceObject->personInfo.uuid + ".feature";
    LogDebug << faceObject->personInfo.uuid << ".feature";
#ifdef ASCEND_ACL_OPEN_VESION
    FileEx::SaveBinaryFileOverwrite(fileName, faceObject->featureVector.deviceData.get(),
        faceObject->featureVector.dataSize);
#else
    FileEx::SaveBinaryFileOverwrite(fileName, faceObject->featureVector.hostData.get(),
        faceObject->featureVector.dataSize);
#endif

    // save the employee information and write it to the JSON file
    fileName = feactureLibPath_ + "/" + faceObject->personInfo.uuid + ".config";
    mode_t new_umask = 0077; // 0077 is for S_IRUSR | S_IWUSR
    mode_t old_umask = umask(new_umask);
    ConfigParser configParser;
    configParser.NewConfig(fileName);
    configParser.WriteString("uuid", faceObject->personInfo.uuid);
    configParser.WriteString("name", faceObject->personInfo.name);
    configParser.WriteString("gender", faceObject->personInfo.gender);
    configParser.WriteInt("age", faceObject->personInfo.age);
    configParser.SaveConfig();
    umask(old_umask);

    // update List
    std::shared_ptr<FaceFeatureInfo> featureInfo = std::make_shared<FaceFeatureInfo>();
    FeatureInfoCopy(featureInfo->featureValue, faceObject->featureVector);
    featureInfo->norm = faceObject->featureNorm;
    featureInfo->uuid = faceObject->personInfo.uuid;
    featureLib_.push_back(featureInfo);
    // insert person uuid into uuidBase set
    uuidBase_.insert(faceObject->personInfo.uuid);

    LogDebug << "FaceFeatureLib: It is successfull to insert " << faceObject->personInfo.uuid << " information to lib.";
    return APP_ERR_OK;
}

/*
 * Description: copy feature values and create feature value storage space
 */
void FaceFeatureLib::FeatureInfoCopy(std::shared_ptr<uint8_t> &dstFeatureInfo, const DataBuffer &srcFeatureInfo)
{
    if (srcFeatureInfo.dataSize != FACE_FEATURE_SIZE) {
        LogError << "srcFeatureInfo.dataSize=" << srcFeatureInfo.dataSize << "is not " << FACE_FEATURE_SIZE;
        return;
    }
    auto featureValue = std::make_shared<uint8_t>();
    featureValue.reset(new uint8_t[FACE_FEATURE_SIZE], std::default_delete<uint8_t[]>());
#ifdef ASCEND_ACL_OPEN_VESION
    std::copy(srcFeatureInfo.deviceData.get(), srcFeatureInfo.deviceData.get() + srcFeatureInfo.dataSize,
        featureValue.get());
#else
    std::copy(srcFeatureInfo.hostData.get(), srcFeatureInfo.hostData.get() + srcFeatureInfo.dataSize,
        featureValue.get());
#endif
    dstFeatureInfo = featureValue;
    return;
}

/*
 * Description: get the maximum similarity in the library and returns the corresponding subscript.
 */
int FaceFeatureLib::GetMaxSimilarity(std::shared_ptr<FaceObject> faceObject, float &similarity)
{
    float featureNorm = faceObject->featureNorm;
    similarity = 0;
    int iRecord = -1;
    for (size_t i = 0; i < featureLib_.size(); ++i) {
#ifdef ASCEND_ACL_OPEN_VESION
        float vectorDot = CalcVectorDot((float *)faceObject->featureVector.deviceData.get(),
            (float *)featureLib_[i]->featureValue.get());
#else
        float vectorDot = CalcVectorDot((float *)faceObject->featureVector.hostData.get(),
            (float *)featureLib_[i]->featureValue.get());
#endif
        vectorDot = vectorDot / ((double)featureLib_[i]->norm);
        LogDebug << "FaceFeatureLib: to match：" << featureLib_[i]->uuid << " lib size=" << featureLib_.size() <<
            " dot=" << vectorDot << "\n";
        if (similarity < vectorDot) {
            similarity = vectorDot;
            iRecord = i;
        }
    }
    if (iRecord >= 0) {
        if (featureNorm > NORM_EPS) {
            similarity /= static_cast<double>(featureNorm);
        } else {
            similarity = 0.f;
        }
    }
    return iRecord;
}

/*
 * Description: get the closest match
 */
APP_ERROR FaceFeatureLib::GetPersonInfoByFeature(std::shared_ptr<FaceObject> faceObject)
{
    LogDebug << "FaceFeatureLib: beagin to get bestH information by feature value, lib size=" << featureLib_.size();
    std::lock_guard<std::mutex> guard(mtx_);
    if (faceObject->featureVector.dataSize != FACE_FEATURE_SIZE) {
        LogError << "FaceFeatureLib: face feature value is not correct, dataSize=" <<
            faceObject->featureVector.dataSize;
        return APP_ERR_COMM_INVALID_PARAM;
    }

    if (featureLib_.empty()) {
        LogDebug << "FaceFeatureLib: Face Feature Lib is empty, retrun unkown information";
        faceObject->personInfo.uuid = "unknow";
        const float ZERO = 0.;
        faceObject->personInfo.similarity = ZERO;
        LogDebug << "FaceFeatureLib: get unkown humman information";
        return APP_ERR_OK;
    }

    // find the best match
    float similarity = 0;
    int32_t bestPersonIndex = GetMaxSimilarity(faceObject, similarity);
    faceObject->personInfo.similarity = similarity;
    if (bestPersonIndex >= 0 && (uint32_t)bestPersonIndex < featureLib_.size()) {
        faceObject->personInfo.uuid = featureLib_[bestPersonIndex]->uuid;
    } else {
        faceObject->personInfo.uuid = "unknow";
    }

    float libNorm = 0.f;
    if (bestPersonIndex >= 0) {
        libNorm = featureLib_[bestPersonIndex]->norm;
    }
    LogDebug << "FaceFeatureLib: find the best " << bestPersonIndex << " uuid " << faceObject->personInfo.uuid <<
        ", similarity=" << similarity << " face norm: " << faceObject->featureNorm << " lib norm: " << libNorm << "\n";
    return APP_ERR_OK;
}

/*
 * Description: vector modulus value calculation, two vector calculation
 */
float FaceFeatureLib::CalcVectorNorm(const float *vec)
{
    if (vec == NULL) {
        LogError << "FaceFeatureLib: input pointer pVector is null.";
        return 0.f;
    }

    float norm = 0;
    for (int i = 0; i < FACE_FEATURE_NUM; ++i) {
        norm += vec[i] * vec[i];
    }
    return sqrt(norm);
}

/*
 * Description: calculate the inner product of two vectors
 */
float FaceFeatureLib::CalcVectorDot(const float *vector1, const float *vector2)
{
    if ((vector1 == NULL) or (vector2 == NULL)) {
        LogError << "input pointer pointer is null, vector1=" << vector1 << ", vector2=" << vector2;
        return 0.f;
    }

    float dot = 0;
    for (int i = 0; i < FACE_FEATURE_NUM; ++i) {
        dot += vector1[i] * vector2[i];
    }
    return dot;
}

/*
 * Description: display searched user information
 */
void FaceFeatureLib::ShownAllFeatures()
{
    const uint16_t strSpaceLen = 100;
    std::string splitString(strSpaceLen, '*');
    LogDebug << splitString;
    LogDebug << "there are " << featureLib_.size() << " features";
    std::string uuidTitle("uuid");
    std::string nameTitle("name");
    std::string genderTitle("gender");
    std::string ageTitle("age");
    std::string normTitle("norm");

    const uint16_t uuidSecWidth = 40;
    std::string uuidTitleSpace(uuidSecWidth - uuidTitle.size(), ' ');
    const uint16_t nameSecWidth = 20;
    std::string nameSpace(nameSecWidth - nameTitle.size(), ' ');
    const uint16_t genderSecWidth = 10;
    std::string genderSpace(genderSecWidth - genderTitle.size(), ' ');
    const uint16_t ageSecWidth = 5;
    std::string ageSpace(ageSecWidth - ageTitle.size(), ' ');
    LogDebug << uuidTitle << uuidTitleSpace << nameTitle << nameSpace << genderTitle << genderSpace << ageTitle <<
        ageSpace << normTitle;
    for (size_t i = 0; i < featureLib_.size(); ++i) {
        std::string prefix = featureLib_[i]->uuid;
        std::string fileName = feactureLibPath_ + "/" + prefix + ".config";
        ConfigParser configParser;
        configParser.ParseConfig(fileName);
        std::string uuid;
        std::string name;
        std::string gender;
        uint32_t age = 0;
        configParser.GetStringValue("uuid", uuid);
        std::string uuidSpace(uuidSecWidth - uuid.size(), ' ');
        LogDebug << uuid << uuidSpace << name << nameSpace << gender << genderSpace << age << ageSpace <<
            featureLib_[i]->norm;
    }
    LogDebug << splitString << std::endl << std::endl;
    return;
}
} // namespace ascendFaceRecognition
