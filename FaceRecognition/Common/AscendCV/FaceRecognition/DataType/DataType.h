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
#ifndef __INC_DATA_TYPE_H__
#define __INC_DATA_TYPE_H__
#include <memory>
#include <stdint.h>
#include <vector>

#include "BlockingQueue/BlockingQueue.h"
#include "Proto/vehicle.pb.h"

namespace ascendFaceRecognition {
#define DVPP_ALIGN_UP(x, align) ((((x) + ((align)-1)) / (align)) * (align))

const int MODULE_QUEUE_SIZE = 1000;

enum ModuleType {
    MT_STREAM_PULLER = 0,
    MT_IMAGE_DECODER,
    MT_IMAGE_READER,
    MT_VIDEO_DECODER,
    MT_VIDEO_RESIZE,
    MT_VIDEO_ENCODER,
    MT_VIDEO_CONCAT,
    MT_FACE_DETECTION,
    MT_FACE_TRACK,
    MT_FACE_LANDMARK,
    MT_WARP_AFFINE,
    MT_FACE_FEATURE,
    MT_FACE_STOCK,
    MT_FACE_SEARCH,
    MT_MOT_EMBEDDING,
    MT_MOT_CONNECTION,
    MT_QUALITY_EVALUATION,
    MT_FACE_ATTRIBUTE,
    MT_FEATURE_MERGE,
    MT_RESULT_SENDER,
    MT_FRAME_ALIGN,
    MT_FACE_DETAIL_INFO,
    MT_REG_RESULT_HANDLER,
    MT_HDC_SEND,
    MT_HDC_RECV,
    MT_FACE_RESIZE,
    MT_OTHERS,
    MT_BOTTOM
};

enum ModuleConnectType {
    MODULE_CONNECT_ONE = 0,
    MODULE_CONNECT_CHANNEL, //
    MODULE_CONNECT_PAIR,    //
    MODULE_CONNECT_RANDOM   //
};

enum PipelineMode {
    PIPELINE_MODE_SEARCH = 0, // for face search
    PIPELINE_MODE_REG,        // for face registration
    PIPELINE_MODE_BOTTOM
};

struct PersonInfo {
    std::string uuid;
    std::string name;
    std::string gender;
    uint32_t age;
    std::string mask;
    float similarity;
};

enum FrameMode {
    FRAME_MODE_SEARCH = 0,
    FRAME_MODE_REG
};

struct FrameInfo {      // old:StreamInfo
    uint64_t frameId;   /* frame id, in video stream channel */
    uint64_t startTime; /* time of start */
    FrameMode mode;     /* Acltodo Operate mode: register, normal, decode H26* */
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
    DataBuffer() : deviceData(nullptr), hostData(nullptr), dataSize(0) {}
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
};

enum TraceFlag {
    NEW_FACE = 0,
    TRACkED_FACE,
    LOST_FACE
};
struct FaceTraceInfo {
    int32_t id;
    TraceFlag flag;
    int32_t survivalTime; // How long is it been since the first time, unit: detection period
    int32_t detectedTime; // How long is the face detected, unit: detection period
    std::chrono::time_point<std::chrono::high_resolution_clock> createTime;
};

struct TrackLet {
    FaceTraceInfo info;
    // reserved:  kalman status parameter
    int32_t lostTime;                     // undetected time for tracked face
    std::vector<DataBuffer> shortFeature; // nearest 10 frame
};


struct FaceQuality {
    float score;
};
struct Coordinate2D {
    uint32_t x;
    uint32_t y;
};

struct FaceObject {
    FrameInfo frameInfo;
    DetectInfo info;
    FaceTraceInfo trackInfo;
    FaceQuality faceQuality;
    DataBuffer landmarks;
    DataBuffer featureVector;
    float featureNorm;
    DataBuffer embedding;
    float embeddingNorm;
    DataBuffer attributeBuffer;
    ImageInfo imgEmbedding; // for extact the embeding feature
    ImageInfo imgCroped;
    ImageInfo imgAffine;
    ImageInfo imgOrigin;
    ImageInfo imgSearch;
    ImageInfo imgQuality;
    PersonInfo personInfo;
};

struct StreamRawData {
    FrameInfo info;
    std::shared_ptr<uint8_t> deviceData;
    std::shared_ptr<uint8_t> hostData;
    uint32_t dataSize;
    int frameType; // 0:I frame,1:P frame,2:B frame;
};

struct FrameAiInfo {
    FrameInfo info;
    std::vector<DetectInfo> detectResult; // Acltodo detectInfo only
    std::vector<FaceObject> face;
    ImageInfo imgOrigin;
    ImageInfo detectImg; // scaled image for model input
    ImageInfo resizeImg;
    std::vector<int> detIdx;
    std::vector<int> trkIdx;
    int isKeyFrame;
    int isTracked;
    int trkFrameID;
    int isSplited;
    int isOutputStream;
    int isSequenceFrame;
    int embeddingCount;
};

// define the output parameter:
struct FaceDetectOutput {
    FrameInfo info;
    std::vector<DetectInfo> detectResult;
};

struct FaceInfoOutput {
    FrameInfo frameInfo;
    PersonInfo personInfo;
    ImageInfo imgOrigin;
};
} // namespace ascendFaceRecognition

struct AttrT {
    AttrT(std::string name, std::string value) : name(std::move(name)), value(std::move(value)) {}

    std::string name = {};
    std::string value = {};
};

struct PictureDataT {
    std::string data;
    uint32_t channelId;
    std::vector<AttrT> attrList;
};

struct VideoDataT {
    uint32_t channelId;
    std::shared_ptr<VehicleData> data;
};

#endif
