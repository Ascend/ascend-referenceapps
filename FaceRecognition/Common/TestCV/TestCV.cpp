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
#include "TestCV.h"
#include "acl/acl.h"
using namespace cv;

namespace ascendFaceRecognition {
namespace {
const uint32_t YUV_BYTES_NU = 3; // Numerator of yuv image, H x W x 3 / 2
const uint32_t YUV_BYTES_DE = 2; // Denominator of yuv image, H x W x 3 / 2
const uint32_t BGR_CHANNELS = 3; // bgr image channels
}
TestCV::TestCV() {}
TestCV::~TestCV() {}

TestCV &TestCV::GetInstance()
{
    static TestCV testUtils;
    return testUtils;
}

void TestCV::AddRectIntoImage(ImageInfo &image, std::vector<DetectInfo> &detectResult)
{
    int height = image.heightAligned;
    int width = image.widthAligned;
    int step = image.widthAligned;
    cv::Mat frameYUV(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, (void *)image.buf.deviceData.get(), step);
    cv::Mat frameBGR;
    cv::cvtColor(frameYUV, frameBGR, cv::COLOR_YUV2BGR_NV12);
    for (const auto &rect : detectResult) {
        int x1 = (int)rect.minx;
        int y1 = (int)rect.miny;
        int width = (int)rect.width;
        int height = (int)rect.height;
        cv::Rect temp(x1, y1, width, height);
        cv::rectangle(frameBGR, temp, blue, rectThickness);
    }
    static int imgCount = 0;
    cv::imwrite("test/" + std::to_string(imgCount++) + ".jpg", frameBGR);
}

void TestCV::AddTrackIntoImage(FrameAiInfo &aiInfo)
{
    int height = aiInfo.imgOrigin.heightAligned;
    int width = aiInfo.imgOrigin.widthAligned;
    int step = aiInfo.imgOrigin.widthAligned;
    cv::Mat frameYUV(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1,
        (void *)aiInfo.imgOrigin.buf.deviceData.get(), step);
    cv::Mat frameBGR;
    cv::cvtColor(frameYUV, frameBGR, cv::COLOR_YUV2BGR_NV12);
    for (const auto &faceT : aiInfo.face) {
        int x1 = (int)faceT.info.minx;
        int y1 = (int)faceT.info.miny;
        int width = (int)faceT.info.width;
        int height = (int)faceT.info.height;
        cv::Rect temp(x1, y1, width, height);
        cv::rectangle(frameBGR, temp, green, thickness);
        cv::putText(frameBGR, std::to_string(faceT.trackInfo.id), Point(x1 + xOffset, y1 + yOffset),
            FONT_HERSHEY_SIMPLEX, fontScale, blue, thickness, lineType);
    }
    static int imgCount = 0;
    cv::imwrite("test/Track" + std::to_string(imgCount++) + ".jpg", frameBGR);
}

void TestCV::AddQEIntoImage(FrameAiInfo &aiInfo)
{
    int height = aiInfo.imgOrigin.heightAligned;
    int width = aiInfo.imgOrigin.widthAligned;
    int step = aiInfo.imgOrigin.widthAligned;
    cv::Mat frameYUV(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1,
        (void *)aiInfo.imgOrigin.buf.deviceData.get(), step);
    cv::Mat frameBGR;
    cv::cvtColor(frameYUV, frameBGR, cv::COLOR_YUV2BGR_NV12);
    for (const auto &faceT : aiInfo.face) {
        int x1 = (int)faceT.info.minx;
        int y1 = (int)faceT.info.miny;
        int width = (int)faceT.info.width;
        int height = (int)faceT.info.height;
        cv::Rect temp(x1, y1, width, height);
        cv::rectangle(frameBGR, temp, blue, rectThickness);
        cv::putText(frameBGR, std::to_string(faceT.trackInfo.id), Point(x1 + xOffset, y1 + yOffset),
            FONT_HERSHEY_SIMPLEX, fontScale, green, thickness, lineType);
    }
    static int imgCount = 0;
    cv::imwrite("test/qe/qe" + std::to_string(imgCount++) + ".jpg", frameBGR);
}

void TestCV::AddKPIntoImage(void *buf, std::vector<Coordinate2D> &keyPoints)
{
    const int inputSize = 96;
    cv::Mat frameYUV(inputSize * YUV_BYTES_NU / YUV_BYTES_DE, inputSize, CV_8UC1, buf, inputSize);
    cv::Mat frameBGR;
    cv::cvtColor(frameYUV, frameBGR, cv::COLOR_YUV2BGR_NV12);
    for (int i = 0; i < numLandmark; i++) {
        int x1 = (int)keyPoints[i].x;
        int y1 = (int)keyPoints[i].y;
        cv::putText(frameBGR, std::to_string(i), Point(x1, y1), FONT_HERSHEY_SIMPLEX, fontScale, blue, thickness,
            lineType);
    }
    static int imgCount = 0;
    cv::imwrite("./test/qe/qe" + std::to_string(imgCount++) + ".jpg", frameBGR);
}

void TestCV::WrapAffineImage(void *srcBuf, void *waBuf, const std::vector<cv::Point2f> &landmarks)
{
    const int inputSize = 112;
    cv::Mat frameYUV(inputSize * YUV_BYTES_NU / YUV_BYTES_DE, inputSize, CV_8UC1, srcBuf, inputSize);
    cv::Mat frameYUV2BGR;
    cv::cvtColor(frameYUV, frameYUV2BGR, cv::COLOR_YUV2BGR_NV12);
    cv::Mat frameBGR(inputSize, inputSize, CV_8UC3, waBuf);
    cv::Mat retMat = Mat::zeros(inputSize + inputSize, inputSize, CV_8UC3);
    static int imgCount = 0;
    std::string fileName = "./temp/face_" + std::to_string(imgCount) + "_raw";
    for (uint32_t i = 0; i < landmarks.size(); i++) {
        float x1 = landmarks[i].x;
        float y1 = landmarks[i].y;
        cv::circle(frameYUV2BGR, Point(x1, y1), radius, red);
        fileName += ("_" + std::to_string(x1) + "_" + std::to_string(y1));
    }
    fileName += "_.jpg";
    cv::imwrite(fileName, frameYUV2BGR);
    LogDebug << "file name:" << fileName << "";
    fileName = "./temp/face_" + std::to_string(imgCount) + ".jpg";
    cv::imwrite(fileName, frameBGR);
    imgCount++;
}


void TestCV::WrapAffineImage(void *srcBuf, void *waBuf, const float *landmarks)
{
    const int inputSize = 112;
    cv::Mat frameYUV(inputSize * YUV_BYTES_NU / YUV_BYTES_DE, inputSize, CV_8UC1, srcBuf, inputSize);
    cv::Mat frameYUV2BGR;
    cv::cvtColor(frameYUV, frameYUV2BGR, cv::COLOR_YUV2BGR_NV12);
    cv::Mat frameBGR(inputSize, inputSize, CV_8UC3, waBuf);
    cv::Mat retMat = Mat::zeros(inputSize + inputSize, inputSize, CV_8UC3);
    static int imgCount = 0;
    std::string fileName = "./temp/face_" + std::to_string(imgCount) + "_raw.jpg";
    const int pairSize = 2;
    const int xPairIndex = 0;
    const int yPairIndex = 1;
    if (landmarks != nullptr) {
        fileName = "./temp/face_" + std::to_string(imgCount) + "_raw";
        for (int i = 0; i < numLandmark; i++) {
            float x1 = landmarks[pairSize * i + xPairIndex];
            float y1 = landmarks[pairSize * i + yPairIndex];
            cv::circle(frameYUV2BGR, Point(x1, y1), radius, red);
            fileName += ("_" + std::to_string(x1) + "_" + std::to_string(y1));
        }
        fileName += "_.jpg";
    }
    cv::imwrite(fileName, frameYUV2BGR);
    LogDebug << "file name:" << fileName << "";
    fileName = "./temp/face_" + std::to_string(imgCount) + ".jpg";
    cv::imwrite(fileName, frameBGR);
    imgCount++;
}

void TestCV::WrapAffineImage(const cv::Mat &warpImage)
{
    static int imgCount = 0;
    imgCount++;
    std::string fileName = "./temp/face_" + std::to_string(imgCount) + ".jpg";
    cv::imwrite(fileName, warpImage);
}

void TestCV::WAAddKPIntoImage(void *buf, std::vector<Coordinate2D> &keyPoints)
{
    const int inputSize = 112;
    cv::Mat frameYUV(inputSize * YUV_BYTES_NU / YUV_BYTES_DE, inputSize, CV_8UC1, buf, inputSize);
    cv::Mat frameBGR;
    cv::cvtColor(frameYUV, frameBGR, cv::COLOR_YUV2BGR_NV12);
    for (int i = 0; i < numLandmark; i++) {
        int x1 = (int)keyPoints[i].x;
        int y1 = (int)keyPoints[i].y;
        cv::circle(frameBGR, Point(x1, y1), radius, red);
    }
    static int imgCount = 0;
    cv::imwrite("./temp/was" + std::to_string(imgCount++) + ".jpg", frameBGR);
}
#ifndef ASCEND_ACL_OPEN_VESION
std::shared_ptr<void> TestCV::DeviceCopyToHost(const uint32_t size, std::shared_ptr<uint8_t> data) const
{
    if (size <= 0) {
        return nullptr;
    }

    auto host = std::make_shared<uint8_t>();
    host.reset(new uint8_t[size], std::default_delete<uint8_t[]>());
    if (host == nullptr) {
        return nullptr;
    }

    std::shared_ptr<void> hostData = host;
    APP_ERROR ret = aclrtMemcpy(host.get(), size, data.get(), size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != APP_ERR_OK) {
        return nullptr;
    }
    return hostData;
}
#endif
void TestCV::SaveDetectResult(std::shared_ptr<FrameAiInfo> frameAiInfo)
{
    uint32_t height = frameAiInfo->imgOrigin.heightAligned;
    uint32_t width = frameAiInfo->imgOrigin.widthAligned;
    uint32_t size = width * height * YUV_BYTES_NU / YUV_BYTES_DE;
#ifdef ASCEND_ACL_OPEN_VESION
    cv::Mat srcN21 =
        cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, frameAiInfo->imgOrigin.buf.deviceData.get());
#else
    std::shared_ptr<void> hostData = DeviceCopyToHost(size, frameAiInfo->imgOrigin.buf.deviceData);
    cv::Mat srcN21 = cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, hostData.get());
#endif
    cv::Mat dstBgr = cv::Mat(frameAiInfo->imgOrigin.heightAligned, frameAiInfo->imgOrigin.widthAligned, CV_8UC3);
    cv::cvtColor(srcN21, dstBgr, cv::COLOR_YUV2BGR_NV12);
    std::string fileName = "temp/channel_" + std::to_string(frameAiInfo->info.channelId) + "frame_result_" +
        std::to_string(frameAiInfo->info.frameId) + ".jpg";

    for (uint32_t i = 0; i < frameAiInfo->detectResult.size(); i++) {
        cv::rectangle(dstBgr,
            cv::Rect(int(frameAiInfo->detectResult[i].minx), int(frameAiInfo->detectResult[i].miny),
                int(frameAiInfo->detectResult[i].width), int(frameAiInfo->detectResult[i].height)),
            green, rectThickness);
    }
    cv::imwrite(fileName, dstBgr);
}

void TestCV::SaveTrackImage(std::shared_ptr<FrameAiInfo> frameAiInfo)
{
    uint32_t height = frameAiInfo->imgOrigin.heightAligned;
    uint32_t width = frameAiInfo->imgOrigin.widthAligned;
    uint32_t size = width * height * YUV_BYTES_NU / YUV_BYTES_DE;
#ifdef ASCEND_ACL_OPEN_VESION
    cv::Mat srcN21 = cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width,
#else
    std::shared_ptr<void> hostData = DeviceCopyToHost(size, frameAiInfo->imgOrigin.buf.deviceData);
    cv::Mat srcN21 = cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, hostData.get());
#endif
    cv::Mat dstBgr = cv::Mat(frameAiInfo->imgOrigin.heightAligned, frameAiInfo->imgOrigin.widthAligned, CV_8UC3);
    cv::cvtColor(srcN21, dstBgr, cv::COLOR_YUV2BGR_NV12);
    for (uint32_t i = 0; i < frameAiInfo->face.size(); i++) {
        if (frameAiInfo->face[i].trackInfo.flag == LOST_FACE || frameAiInfo->face[i].trackInfo.flag == NEW_FACE)
            continue;

        int id = frameAiInfo->face[i].trackInfo.id;
        cv::rectangle(dstBgr,
            cv::Rect(int(frameAiInfo->face[i].info.minx), int(frameAiInfo->face[i].info.miny),
                int(frameAiInfo->face[i].info.width), int(frameAiInfo->face[i].info.height)),
            green, rectThickness);

        cv::putText(dstBgr, std::to_string(id),
            Point(int(frameAiInfo->face[i].info.minx) + xOffset, int(frameAiInfo->face[i].info.miny) + yOffset),
            FONT_HERSHEY_SIMPLEX, fontScale, green, thickness, lineType);
    }

    std::string fileName = "./temp/track_channel_" + std::to_string(frameAiInfo->info.channelId) + "_frame_idx_" +
        std::to_string(frameAiInfo->info.frameId) + ".jpg";
    cv::imwrite(fileName, dstBgr);
}

void TestCV::SaveQEImage(std::shared_ptr<FaceObject> faceObject, int batchIdx)
{
    uint32_t height = faceObject->imgQuality.heightAligned;
    uint32_t width = faceObject->imgQuality.widthAligned;
    uint32_t size = width * height * YUV_BYTES_NU / YUV_BYTES_DE;
#ifdef ASCEND_ACL_OPEN_VESION
    cv::Mat srcN21 =
        cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, faceObject->imgQuality.buf.deviceData.get());
#else
    std::shared_ptr<void> hostData = DeviceCopyToHost(size, faceObject->imgQuality.buf.deviceData);
    cv::Mat srcN21 = cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, hostData.get());
#endif
    cv::Mat dstBgr = cv::Mat(faceObject->imgQuality.height, faceObject->imgQuality.width, CV_8UC3);
    cv::cvtColor(srcN21, dstBgr, cv::COLOR_YUV2BGR_NV12);

    std::string fileName;
#ifdef ASCEND_ACL_OPEN_VESION
    float *landmarks = (float *)faceObject->landmarks.deviceData.get();
#else
    float *landmarks = (float *)faceObject->landmarks.hostData.get();
#endif
    fileName = "temp/channel_" + std::to_string(faceObject->frameInfo.channelId) + "_frame_idx_" +
        std::to_string(faceObject->frameInfo.frameId) + "_face_id_" + std::to_string(faceObject->trackInfo.id) +
        "_batchIdx_" + std::to_string(batchIdx) + "_score_" + std::to_string(faceObject->faceQuality.score) +
        "_raw.jpg";

    cv::imwrite(fileName, dstBgr);
    const int inputSize = 96;
    for (int i = 0; (landmarks != nullptr) && (i < numLandmark); i++) {
        int x1 = int(landmarks[i + i] * inputSize);
        int y1 = int(landmarks[i + i + 1] * inputSize);
        cv::circle(dstBgr, Point(x1, y1), radius, red);
    }
    fileName = "temp/channel_" + std::to_string(faceObject->frameInfo.channelId) + "_frame_idx_" +
        std::to_string(faceObject->frameInfo.frameId) + "_face_id_" + std::to_string(faceObject->trackInfo.id) +
        "_batchIdx_" + std::to_string(batchIdx) + ".jpg";
    cv::imwrite(fileName, dstBgr);
}

void TestCV::SaveFaceAttributeImage(std::shared_ptr<FaceObject> faceObject, int batchIdx)
{
    uint32_t height = faceObject->imgAffine.heightAligned;
    uint32_t width = faceObject->imgAffine.widthAligned;
    uint32_t size = height * width * BGR_CHANNELS;
#ifdef ASCEND_ACL_OPEN_VESION
    cv::Mat dstBgr = cv::Mat(height, width, CV_8UC3, faceObject->imgAffine.buf.deviceData.get());
#else
    std::shared_ptr<void> hostData = DeviceCopyToHost(size, faceObject->imgAffine.buf.deviceData);
    cv::Mat dstBgr = cv::Mat(height, width, CV_8UC3, hostData.get());
#endif

    std::string fileName = "./temp/channel_" + std::to_string(faceObject->frameInfo.channelId) + "_frame_idx_" +
        std::to_string(faceObject->frameInfo.frameId) + "_face_id_" + std::to_string(faceObject->trackInfo.id) +
        "_face_attribute_gender_" + std::string(faceObject->personInfo.gender) + "_age_" +
        std::to_string(faceObject->personInfo.age) + "_status_" + std::string(faceObject->personInfo.mask) +
        "_batchIdx_" + std::to_string(batchIdx) + ".jpg";

    cv::imwrite(fileName, dstBgr);
}

void TestCV::SaveDecodeImage(std::shared_ptr<FrameAiInfo> frameAiInfo) const
{
    uint32_t height = frameAiInfo->detectImg.heightAligned;
    uint32_t width = frameAiInfo->detectImg.widthAligned;
    uint32_t size = width * height * YUV_BYTES_NU / YUV_BYTES_DE;
#ifdef ASCEND_ACL_OPEN_VESION
    cv::Mat srcN21 =
        cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, frameAiInfo->detectImg.buf.deviceData.get());
#else
    std::shared_ptr<void> hostData = DeviceCopyToHost(size, frameAiInfo->detectImg.buf.deviceData);
    cv::Mat srcN21 = cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, hostData.get());
#endif
    cv::Mat dstBgr = cv::Mat(height, width, CV_8UC3);
    cv::cvtColor(srcN21, dstBgr, cv::COLOR_YUV2BGR_NV12);
    static int idx = 0;
    idx++;
    std::string fileName = "./temp/detect_img" + std::to_string(idx) + ".jpg";
    cv::imwrite(fileName, dstBgr);
}


void TestCV::SaveFaceSearchImage(std::shared_ptr<FaceObject> faceObject)
{
    std::string fileName = "./temp/channel_" + std::to_string(faceObject->frameInfo.channelId) + "_frame_idx_" +
        std::to_string(faceObject->frameInfo.frameId) + "_track_id_" + std::to_string(faceObject->trackInfo.id) +
        "_search_id_" + faceObject->personInfo.uuid + "_similarity_" +
        std::to_string(faceObject->personInfo.similarity) + ".jpg";
    cv::Mat dstBgr = cv::Mat(faceObject->imgCroped.height, faceObject->imgCroped.width, CV_8UC3,
        faceObject->imgCroped.buf.deviceData.get());
    cv::Mat dstN21 = cv::Mat(faceObject->imgCroped.height * YUV_BYTES_NU / YUV_BYTES_DE, faceObject->imgCroped.width,
        CV_8UC1, faceObject->imgCroped.buf.deviceData.get());
    cv::Mat imgBGR;
    cv::cvtColor(dstN21, dstBgr, cv::COLOR_YUV2BGR_NV12);
    cv::imwrite(fileName, dstBgr);
}


void TestCV::SaveFaceFeatureLibImage(std::shared_ptr<FaceObject> faceObject)
{
    std::string fileName = "./temp/channel_" + std::to_string(faceObject->frameInfo.channelId) + "_frame_idx_" +
        std::to_string(faceObject->frameInfo.frameId) + "_track_id_" + std::to_string(faceObject->trackInfo.id) +
        "_search_id_" + faceObject->personInfo.uuid + "_similarity_" +
        std::to_string(faceObject->personInfo.similarity) + "_xmin_" + std::to_string(int(faceObject->info.minx)) +
        "_ymin_" + std::to_string(int(faceObject->info.miny)) + "_width_" +
        std::to_string(int(faceObject->info.width)) + "_height_" + std::to_string(int(faceObject->info.height)) +
        ".jpg";
    cv::Mat srcN21 = cv::Mat(faceObject->imgOrigin.heightAligned * YUV_BYTES_NU / YUV_BYTES_DE,
        faceObject->imgOrigin.widthAligned, CV_8UC1, faceObject->imgOrigin.buf.deviceData.get());

    cv::Mat dstBgr = cv::Mat(faceObject->imgOrigin.heightAligned, faceObject->imgOrigin.widthAligned, CV_8UC3);
    cv::cvtColor(srcN21, dstBgr, cv::COLOR_YUV2BGR_NV12);
    std::string uuid = faceObject->personInfo.uuid;
    cv::rectangle(dstBgr,
        cv::Rect(int(faceObject->info.minx), int(faceObject->info.miny), int(faceObject->info.width),
            int(faceObject->info.height)),
        green, rectThickness);
    cv::putText(dstBgr, uuid, Point(int(faceObject->info.minx) + xOffset, int(faceObject->info.miny) + yOffset),
        FONT_HERSHEY_SIMPLEX, fontScale, lightGreen, thickness, lineType);

    cv::imwrite(fileName, dstBgr);
}

void TestCV::SaveLandmarkImage(std::shared_ptr<FaceObject> faceObject)
{
    int width = faceObject->imgOrigin.width;
    int height = faceObject->imgOrigin.height;
    cv::Mat srcN21 =
        cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, faceObject->imgOrigin.buf.deviceData.get());
    cv::Mat dstBgr = cv::Mat(height, width, CV_8UC3);
    cv::cvtColor(srcN21, dstBgr, cv::COLOR_YUV2BGR_NV12);
    cv::rectangle(dstBgr,
        cv::Rect(int(faceObject->info.minx), int(faceObject->info.miny), int(faceObject->info.width),
            int(faceObject->info.height)),
        green, rectThickness);
    static int idx = 0;
    idx++;
    std::string fileName = "temp/landmark_index_" + std::to_string(idx) + ".jpg";
    cv::imwrite(fileName, dstBgr);
}

void TestCV::SaveFaceImage(std::shared_ptr<FaceAttributeDataTrans> faceInfo)
{
    std::string fileName = "./temp/channel_" + std::to_string(faceInfo->channelid()) + "_frame_idx_" +
        std::to_string(faceInfo->frameid()) + "_face_id_" + std::to_string(faceInfo->trackid()) +
        "_face_attribute_gender_" + std::string(faceInfo->gender()) + "_age_" + std::to_string(faceInfo->age()) +
        "_status_" + std::string(faceInfo->mask()) + ".jpg";

    cv::Mat srcN12 = cv::Mat(faceInfo->avatar().height() * YUV_BYTES_NU / YUV_BYTES_DE, faceInfo->avatar().width(),
        CV_8UC1, (char *)faceInfo->avatar().data().c_str());
    cv::Mat dstBgr = cv::Mat(faceInfo->avatar().height(), faceInfo->avatar().width(), CV_8UC3);
    cv::cvtColor(srcN12, dstBgr, cv::COLOR_YUV2BGR_NV12);
    cv::imwrite(fileName, dstBgr);
}

void TestCV::SaveCropImage(const ImageInfo &imageInfo, const uint32_t &frameId, const uint32_t &idx)
{
    uint32_t height = imageInfo.heightAligned;
    uint32_t width = imageInfo.widthAligned;
    uint32_t size = height * width * YUV_BYTES_NU / YUV_BYTES_DE;
#ifdef ASCEND_ACL_OPEN_VESION
    cv::Mat srcN21 = cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, imageInfo.buf.deviceData.get());
#else
    std::shared_ptr<void> hostData = DeviceCopyToHost(size, imageInfo.buf.deviceData);
    cv::Mat srcN21 = cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, hostData.get());
#endif
    cv::Mat dstBgr = cv::Mat(height, width, CV_8UC3);
    cv::cvtColor(srcN21, dstBgr, cv::COLOR_YUV2BGR_NV12);
    std::string fileName = "temp/image_frame_" + std::to_string(frameId) + "_idx_" + std::to_string(idx) + "_crop.jpg";
    cv::imwrite(fileName, dstBgr);
}

void TestCV::SaveQualityImage(const ImageInfo &imageInfo, const uint32_t &frameId, const uint32_t &idx)
{
    uint32_t height = imageInfo.heightAligned;
    uint32_t width = imageInfo.widthAligned;
    uint32_t size = width * height * YUV_BYTES_NU / YUV_BYTES_DE;
#ifdef ASCEND_ACL_OPEN_VESION
    cv::Mat srcN21 = cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, imageInfo.buf.deviceData.get());
#else
    std::shared_ptr<void> hostData = DeviceCopyToHost(size, imageInfo.buf.deviceData);
    cv::Mat srcN21 = cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, hostData.get());
#endif
    cv::Mat dstBgr = cv::Mat(height, width, CV_8UC3);
    cv::cvtColor(srcN21, dstBgr, cv::COLOR_YUV2BGR_NV12);
    std::string fileName =
        "temp/image_frame_" + std::to_string(frameId) + "_idx_" + std::to_string(idx) + "_quality.jpg";
    cv::imwrite(fileName, dstBgr);
}

void TestCV::SaveEmbeddingImage(const ImageInfo &imageInfo, const uint32_t &frameId, const uint32_t &idx)
{
    uint32_t height = imageInfo.heightAligned;
    uint32_t width = imageInfo.widthAligned;
    uint32_t size = width * height * YUV_BYTES_NU / YUV_BYTES_DE;
#ifdef ASCEND_ACL_OPEN_VESION
    cv::Mat srcN21 = cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, imageInfo.buf.deviceData.get());
#else
    std::shared_ptr<void> hostData = DeviceCopyToHost(size, imageInfo.buf.deviceData);
    cv::Mat srcN21 = cv::Mat(height * YUV_BYTES_NU / YUV_BYTES_DE, width, CV_8UC1, hostData.get());
#endif
    cv::Mat dstBgr = cv::Mat(height, width, CV_8UC3);
    cv::cvtColor(srcN21, dstBgr, cv::COLOR_YUV2BGR_NV12);
    std::string fileName =
        "temp/image_frame_" + std::to_string(frameId) + "_idx_" + std::to_string(idx) + "_embedding.jpg";
    cv::imwrite(fileName, dstBgr);
}
}
