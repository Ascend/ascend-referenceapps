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

#ifndef FACE_FEATURE_LIB_H
#define FACE_FEATURE_LIB_H

#include <string>
#include <vector>
#include <set>

#include "ConfigParser/ConfigParser.h"
#include "DataType/DataType.h"
#include "ErrorCode/ErrorCode.h"

const float SEARCH_SIMILARITY_THRESHOLD = 0.4125; // 搜索相似度门限，大于这个门限，能匹配到库注册信息，否则没有匹配到
const float REG_SIMILARITY_THRESHOLD =
    0.9; // 注册相似度门限，大于这个门限，判定为重复注册，不能再入库，否则可以注册入库

namespace ascendFaceRecognition {
// 人脸特征库信息
struct FaceFeatureInfo {
    float norm = 0.0;                                // 保存计算好的特征值的模值
    std::shared_ptr<uint8_t> featureValue = nullptr; // 用户特征值
    std::string uuid = {};
};

// 人脸特征库
class FaceFeatureLib {
public:
    FaceFeatureLib() {};
    ~FaceFeatureLib() {};

    APP_ERROR FaceFeatureLibInit();
    APP_ERROR InsertFeatureToLib(std::shared_ptr<FaceObject> faceObject, bool isForce);
    APP_ERROR GetPersonInfoByFeature(std::shared_ptr<FaceObject> faceObject);
    void ShownAllFeatures();

private:
    std::vector<std::shared_ptr<FaceFeatureInfo>> featureLib_ = {};
    std::vector<std::string> files_ = {};
    std::string feactureLibPath_ = {}; // lib路径
    std::set<std::string> uuidBase_ = {};
    std::mutex mtx_ = {};

    int GetAllFormatFiles(const std::string &path, std::vector<std::string> &files, const std::string &format) const;
    void FeatureInfoCopy(std::shared_ptr<uint8_t> &dstFeatureInfo, const DataBuffer &srcFeatureInfo);
    int GetMaxSimilarity(std::shared_ptr<FaceObject> faceObject, float &similarity);
    float CalcVectorNorm(const float *vec);
    float CalcVectorDot(const float *vector1, const float *vector2);
};
} // namespace ascendFaceRecognition

#endif
