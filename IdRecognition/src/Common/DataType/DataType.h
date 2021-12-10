/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DATA_TYPE_H
#define DATA_TYPE_H
#include <memory>
#include <stdint.h>
#include <vector>

#include "BlockingQueue/BlockingQueue.h"
#include "CommonDataType/CommonDataType.h"

namespace ascendFaceRecognition {
const int MODULE_QUEUE_SIZE = 100;
const int LANDMARK_NUM = 10;

enum ModuleType {
    MT_STREAM_PULLER = 0,
    MT_JPEG_READER,
    MT_IMAGE_DECODER,
    MT_VIDEO_DECODER,
    MT_FACE_DETECTION,
    MT_FACE_TRACK,
    MT_FACE_LANDMARK,
    MT_FACE_DETECTION_LANDMARK,
    MT_WARP_AFFINE,
    MT_FACE_FEATURE,
    MT_FACE_STOCK,
    MT_FACE_SEARCH,
    MT_FEATURE_MERGE,
    MT_RESULT_SENDER,
    MT_OTHERS,
    MT_BOTTOM
};

enum PipelineMode {
    PIPELINE_MODE_SEARCH = 0, // for face search
    PIPELINE_MODE_REG,        // for face registration
    PIPELINE_MODE_BOTTOM
};

enum FaceImageFormat { // 0:yuv 420P,1:NV12, 2:nv21, 3:bgr888
    FACE_IMAGE_FORMAT_YUV420P = 0,
    FACE_IMAGE_FORMAT_NV12,
    FACE_IMAGE_FORMAT_NV21,
    FACE_IMAGE_FORMAT_BGR888
};

struct PersonInfo {
    std::string uuid;
    std::string name;
    std::string gender;
    uint32_t age;
    float similarity;
};

struct FrameInfo {      // old:StreamInfo
    uint64_t frameId;   /* frame id, in video stream channel */
    uint64_t startTime; /* time of start */
    uint32_t mode;      /* Acltodo Operate mode: register, normal, decode H26* */
    uint32_t format;    /* raw data format: 0-jpg, 1-png, 2-h264, 3-h265 */
    uint32_t channelId; /* video stream channel, corresponding to camera */
    uint32_t isEos;     /* flag of video stream reception */
    uint32_t isDisplay;
    uint32_t width;        /* used for jpeg reader */
    uint32_t height;       /* used for jpeg reader */
    std::string imagePath; /* used for face registration */
    PersonInfo personInfo; /* used for face registration */
};

struct DataBuffer {
    std::shared_ptr<uint8_t> deviceData;
    std::shared_ptr<uint8_t> hostData;
    uint32_t dataSize; // buffer size
};

struct ImageInfo {
    uint32_t format; // 0:yuv 420P,1:NV12, 2:nv21, 3:bgr888
    uint32_t width;
    uint32_t height;
    uint32_t widthAligned;  // required by DVPP
    uint32_t heightAligned; // required by DVPP
    DataBuffer buf;         // data buffer for image
};

struct DetectInfo {
    int32_t classId;
    float confidence;
    float minx; // x value of left-top point
    float miny; // y value of left-top point
    float height;
    float width;
    float landmarks[LANDMARK_NUM];
};

struct FaceObject {
    DetectInfo info;
    DataBuffer landmarks;
    DataBuffer featureVector;
    ImageInfo imgCroped;
    ImageInfo imgAffine;
    ImageInfo imgSearch;
    PersonInfo personInfo;
};

struct StreamRawData {
    FrameInfo info;
    std::shared_ptr<uint8_t> deviceData;
    std::shared_ptr<uint8_t> hostData;
    uint32_t dataSize;
    int frameType; // 0:I frame,1:P frame,2:B frame;
    uint64_t picID;
};

struct FrameAiInfo {
    FrameInfo info;
    std::vector<DetectInfo> detectResult; // Acltodo detectInfo only
    std::vector<FaceObject> face;
    ImageInfo imgOrigin;
    ImageInfo detectImg; // scaled image for model input
    std::vector<int> detIdx;
    std::vector<int> trkIdx;
    int isKeyFrame;
    int isTracked;
    uint64_t trkFrameID;
    int isSplited;
    int isOutputStream;
    int isSequenceFrame;
};

struct KeyPointInfo {
    float *keyPointBefore;
    int keyPointBeforeSize;
    float *keyPointAfter;
    int keyPointAfterSize;
    int *kPBefore;
    int kPBeforeSize;
    int *kPAfter;
    int kPAftersize;
};
} // namespace ascendFaceRecognition

#endif
