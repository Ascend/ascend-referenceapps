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

#include "ConfigParser/ConfigParser.h"
#include "DataType/DataType.h"
#include "ErrorCode/ErrorCode.h"

const float SEARCH_SIMILARITY_THRESHOLD = 0.4125; // Search similarity threshold
const float REG_SIMILARITY_THRESHOLD = 0.9;       // Registration similarity threshold

namespace ascendFaceRecognition {
// Face feature library information
struct FaceFeatureInfo {
    float norm = 0.f;                            // Save the calculated modulus of the eigenvalue.
    std::shared_ptr<uint8_t> featureValue = nullptr; // Person feature value
};

// Face feature library
class FaceFeatureLib {
public:
    APP_ERROR FaceFeatureLibInit();
    APP_ERROR InsertFeatureToLib(const DataBuffer &featureVector, const PersonInfo &personInfo,
        const ImageInfo &imageInfo, bool isForce);
    APP_ERROR GetPersonInfoByFeature(const DataBuffer &featureVector, PersonInfo &personInfo, ImageInfo &imageInfo);
    void ShownAllFeatures();

private:
    std::vector<std::shared_ptr<FaceFeatureInfo>> featureLib_ = {};
    std::vector<std::string> files_ = {};
    std::string feactureLibPath_ = ""; // Face feature library path

    int GetAllFormatFiles(std::string path, std::vector<std::string> &files, std::string format) const;
    void FeatureInfoCopy(std::shared_ptr<uint8_t> &dstFeatureInfo, const DataBuffer &srcFeatureInfo);
    int GetMaxSimilarity(float *vectoy, float &similarity);
    float CalcVectorNorm(const float *vector);
    float CalcVectorDot(const float *vector1, const float *vector2);
    APP_ERROR GetConfigContext(std::string &uuid, PersonInfo &personInfo, ImageInfo &imageInfo);
};
} // namespace ascendFaceRecognition

#endif
