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

#ifndef _INC_FACE_DETAIL_INFO_H
#define _INC_FACE_DETAIL_INFO_H

#include "ErrorCode/ErrorCode.h"
#include "ModuleBase/ModuleBase.h"
#include "ConfigParser/ConfigParser.h"
#include "DataTrans/DataTrans.pb.h"

namespace ascendFaceRecognition {
const float EPSINON = 0.000001;
const float SIMILAR_THRESHOLD = 0.3;

class FaceDetailInfo : public ModuleBase {
public:
    FaceDetailInfo();
    ~FaceDetailInfo();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
    int MatToByteArray(std::shared_ptr<FaceAttributeDataTrans> faceInfo, std::vector<unsigned char> &buff);

private:
    double similarityThreshold_ = 0.3;
};
} // namespace ascendFaceRecognition

#endif // INFEROFFLINEVIDEOBASE_STREAMPULLER_H
