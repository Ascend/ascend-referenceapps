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
#ifndef CENTER_FACE_H
#define CENTER_FACE_H

#include <cstdint>
#include <vector>

#include "FastMath.h"
#include "DataType/DataType.h"

namespace Centerface {
const int CLASS_NUM = 1;
const float POST_NMS_THRESHOLD = 0.45;
const float SIMILARITY_THRESHOLD = 0.35;

const int SCALE_HOST_DIM = 2;
const int OFFSET_HOST_DIM = 2;
const int RESIZE_FACTOR = 4;
const double ROUND_HALF = 0.5;

const int KEY_POINTER_IDX_FACTOR = 2;
const int KEY_POINTER_IDX_RANGE = ascendFaceRecognition::LANDMARK_NUM / KEY_POINTER_IDX_FACTOR;

const int TENSOR_INDEX_0 = 0;
const int TENSOR_INDEX_1 = 1;
const int TENSOR_INDEX_2 = 2;
const int TENSOR_INDEX_3 = 3;

struct BoxRect {
    float x;
    float y;
    float w;
    float h;
};

struct ImageInfoPostProcess {
    float widthModelInput;
    float heightModelInput;
    float widthOrigin;
    float heightOrigin;
};

struct DetectResult {
    BoxRect bbox = {0.f, 0.f, 0.f, 0.f};
    float score = 0.f;
    float keyPoints[ascendFaceRecognition::LANDMARK_NUM] = {};
};

class CenterFace {
public:
    int Process(const std::vector<void *> &outTensorAddrs, const ImageInfoPostProcess &imageInfo,
    std::vector<ascendFaceRecognition::DetectInfo> &detectResult);

private:
    FastMath fastMath_;
};
} // namespace Centerface

#endif
