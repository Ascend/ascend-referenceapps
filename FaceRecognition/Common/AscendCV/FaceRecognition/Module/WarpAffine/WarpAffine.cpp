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

#include "WarpAffine.h"

#include <fstream>

#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "PointerDeleter/PointerDeleter.h"
#include "SimilarityTransform.h"
#include "TestCV/TestCV.h"


namespace ascendFaceRecognition {
namespace {
const uint32_t VPC_STRIDE_WIDTH = 16; // Vpc module output width need to align up to 16
const uint32_t VPC_STRIDE_HEIGHT = 2; // Vpc module output height need to align up to 2
const uint32_t YUV_BYTES_NU = 3;      // Numerator of yuv image, H x W x 3 / 2
const uint32_t YUV_BYTES_DE = 2;      // Denominator of yuv image, H x W x 3 / 2
const float HEIGHT_WEIGHT = 0.05;
const float WIDTH_WEIGHT = 1.2;
const float FACE_OBJECT_WEIGHT = 1.3;
const int KPAFTER_WEIGHT = 112;
const float KPAFTER_OFFSET = 8.0;
const int LANDMARK_PAIR_LEN = 2;
const uint32_t BGR_CHANNEL_NUM = 3;
const uint32_t IMAGE_NV12_FORMAT = 3;
const uint32_t LANDMARK_NUM = 5;
const uint32_t LANDMARK_LEN = 10;
const uint32_t AFFINE_LEN = 10;
}

WarpAffine::WarpAffine()
{
    isStop_ = false;
}

WarpAffine::~WarpAffine() {}

APP_ERROR WarpAffine::ParseConfig(ConfigParser &configParser)
{
    // load and parse config file
    LogDebug << "WarpAffine[" << instanceId_ << "]: begin to parse config values.";
    APP_ERROR ret;
    std::string itemCfgStr;

    itemCfgStr = moduleName_ + std::string(".width_output");
    ret = configParser.GetIntValue(itemCfgStr, width_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".height_output");
    ret = configParser.GetIntValue(itemCfgStr, height_);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    LogDebug << "WarpAffine[" << instanceId_ << "]: widthOutput=" << width_ << " heightOutput=" << height_ << ".";

    return ret;
}

APP_ERROR WarpAffine::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "WarpAffine[" << instanceId_ << "]: begin to init warp affine instance.";

    AssignInitArgs(initArgs);

    // initialize config params
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "WarpAffine[" << instanceId_ << "]: Fail to parse config params, ret=" << ret << "(" <<
            GetAppErrCodeInfo(ret) << ").";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR WarpAffine::DeInit(void)
{
    return APP_ERR_OK;
}

APP_ERROR WarpAffine::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "WarpAffine[" << instanceId_ << "]: begin to process.";

    std::shared_ptr<FaceObject> faceObject = std::static_pointer_cast<FaceObject>(inputData);
    if (faceObject.get() == nullptr) {
        return APP_ERR_COMM_INVALID_POINTER;
    }

    faceObjectVec_.push_back(faceObject);
    if ((int)faceObjectVec_.size() < batchSize_) {
        return APP_ERR_OK;
    }
    for (int i = 0; i < (int)faceObjectVec_.size(); i++) {
        std::shared_ptr<FaceObject> faceObject = faceObjectVec_[i];
        APP_ERROR ret = ApplyWarpAffine(faceObject);
        if (ret != APP_ERR_OK) {
            LogError << "WarpAffine[" << instanceId_ << "]: apply warpaffine error (i=" << i << "), skip warpaffine!";
            continue;
        }
#ifndef ASCEND_ACL_OPEN_VESION
        faceObject->imgCroped.buf.deviceData.reset();
#endif
        if (faceObject->frameInfo.mode == FRAME_MODE_SEARCH) {
            LogDebug << "WarpAffine[" << instanceId_ << "]:FRAME_MODE_SEARCH";
            SendToNextModule(MT_FACE_ATTRIBUTE, faceObject, faceObject->frameInfo.channelId);
        } else {
            SendToNextModule(MT_FACE_FEATURE, faceObject, faceObject->frameInfo.channelId);
        }
    }
    faceObjectVec_.clear();

    return APP_ERR_OK;
}

void WarpAffine::CalWarpImage(const std::vector<cv::Point2f> &srcPoints, const std::vector<cv::Point2f> &dstPoints,
    cv::Mat &warpImage, const cv::Mat &srcImageNv21)
{
    cv::Mat imgBGR888;
    cv::Mat warpMat = SimilarityTransform().Transform(dstPoints, srcPoints);
    cv::cvtColor(srcImageNv21, imgBGR888, cv::COLOR_YUV2BGR_NV12); // COLOR_YUV2RGB_NV12
    cv::Mat warpDst = cv::Mat::zeros(height_, width_, CV_8UC3);
    cv::warpAffine(imgBGR888, warpImage, warpMat, warpDst.size());
}

/* *
 * @brief do warpaffine
 * @[in]: face: faceobject
 * @return: HIAI_StatusT
 */

void WarpAffine::GetSrcLandmark(std::vector<cv::Point2f> &points)
{
    /* calculate the affine matrix */
    // five standard key points after warp affine ,arranged by x0,y0,x1,y1..., scale 112*112
    float kPAfter[LANDMARK_LEN] = {30.2946, 51.6963, 65.5318, 51.5014, 48.0252,
                                   71.7366, 33.5493, 92.3655, 62.7299, 92.2041};

    // map the standard key points into  the image size
    // according to insightface open source //
    // https://github.com/deepinsight/insightface/blob/master/src/align/align_facescrub.py
    for (uint32_t i = 0; i < LANDMARK_NUM; i++) {
        kPAfter[i * LANDMARK_PAIR_LEN] =
            (float)width_ / KPAFTER_WEIGHT * kPAfter[i * LANDMARK_PAIR_LEN] + KPAFTER_OFFSET;
        kPAfter[i * LANDMARK_PAIR_LEN + 1] = (float)height_ / KPAFTER_WEIGHT * kPAfter[i * LANDMARK_PAIR_LEN + 1];
    }

    for (uint32_t i = 0; i < LANDMARK_NUM; i++) {
        points.push_back(cv::Point2f(kPAfter[i * LANDMARK_PAIR_LEN], kPAfter[i * LANDMARK_PAIR_LEN + 1]));
    }
}

void WarpAffine::GetDstLandmark(std::vector<cv::Point2f> &points, std::shared_ptr<FaceObject> faceObject)
{
    float kPBefore[LANDMARK_LEN] = {};
#ifdef ASCEND_ACL_OPEN_VESION
    std::copy(faceObject->landmarks.deviceData.get(),
        faceObject->landmarks.deviceData.get() + faceObject->landmarks.dataSize, (uint8_t *)kPBefore);
#else
    std::copy(faceObject->landmarks.hostData.get(),
        faceObject->landmarks.hostData.get() + faceObject->landmarks.dataSize, (uint8_t *)kPBefore);
#endif
    for (uint32_t i = 0; i < LANDMARK_LEN / LANDMARK_PAIR_LEN; i++) {
        float x = ((HEIGHT_WEIGHT * faceObject->info.width +
            WIDTH_WEIGHT * faceObject->info.width * kPBefore[i * LANDMARK_PAIR_LEN]) /
            (FACE_OBJECT_WEIGHT * faceObject->info.width)) *
            width_;
        float y = ((HEIGHT_WEIGHT * faceObject->info.height +
            WIDTH_WEIGHT * faceObject->info.height * kPBefore[i * LANDMARK_PAIR_LEN + 1]) /
            (FACE_OBJECT_WEIGHT * faceObject->info.height)) *
            height_;
        points.push_back(cv::Point2f(x, y));
    }
}

APP_ERROR WarpAffine::GetCropImage(std::shared_ptr<FaceObject> faceObject, cv::Mat &image)
{
#ifdef ASCEND_ACL_OPEN_VESION
    cv::Mat srcImageNv21(height_ * YUV_BYTES_NU / YUV_BYTES_DE, width_, CV_8UC1,
        faceObject->imgCroped.buf.deviceData.get(), width_);
#else
    uint32_t size = height_ * YUV_BYTES_NU / YUV_BYTES_DE * width_;
    auto hostBuffer = std::shared_ptr<uint8_t>();
    hostBuffer.reset(new uint8_t[size], std::default_delete<uint8_t[]>());
    if (hostBuffer == nullptr) {
        LogError << "WarpAffine[" << instanceId_ << "]: Fail to new memory.";
        return APP_ERR_COMM_ALLOC_MEM;
    }
    std::shared_ptr<uint8_t> hostData = hostBuffer;
    APP_ERROR ret = aclrtMemcpy(hostData.get(), size, faceObject->imgCroped.buf.deviceData.get(),
        faceObject->imgCroped.buf.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != APP_ERR_OK) {
        LogError << "WarpAffine[" << instanceId_ << "]: aclrtMemcpy failed ret=" << ret;
        return ret;
    }
    cv::Mat srcImageNv21(height_ * YUV_BYTES_NU / YUV_BYTES_DE, width_, CV_8UC1, hostData.get(), width_);
    faceObject->imgCroped.buf.hostData = hostData;
#endif
    image = srcImageNv21;
    return APP_ERR_OK;
}

APP_ERROR WarpAffine::SetWarpImage(std::shared_ptr<FaceObject> faceObject, const cv::Mat &imageWarp)
{
    /* use new to allocate memeory */
    uint32_t dataSize = imageWarp.cols * imageWarp.rows * BGR_CHANNEL_NUM;
    std::shared_ptr<uint8_t> deviceData = std::make_shared<uint8_t>();
#ifdef ASCEND_ACL_OPEN_VESION
    deviceData.reset(new uint8_t[dataSize], std::default_delete<uint8_t[]>());
    std::copy((uint8_t *)imageWarp.data, (uint8_t *)imageWarp.data + dataSize, deviceData.get());
#else
    void *deviceBuffer = nullptr;
    APP_ERROR ret = aclrtMalloc(&deviceBuffer, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != APP_ERR_OK) {
        LogError << "WarpAffine[" << instanceId_ << "]: aclrtMalloc failed ret=" << ret;
        return ret;
    }
    deviceData.reset((uint8_t *)deviceBuffer, [](uint8_t *p) { aclrtFree((void *)p); });
    ret = aclrtMemcpy(deviceBuffer, dataSize, imageWarp.data, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != APP_ERR_OK) {
        LogError << "WarpAffine[" << instanceId_ << "]: aclrtMemcpy failed ret=" << ret;
        return ret;
    }
#endif
    faceObject->imgAffine.format = IMAGE_NV12_FORMAT;
    faceObject->imgAffine.width = imageWarp.cols;
    faceObject->imgAffine.height = imageWarp.rows;
    faceObject->imgAffine.widthAligned = DVPP_ALIGN_UP(imageWarp.cols, VPC_STRIDE_WIDTH);
    faceObject->imgAffine.heightAligned = DVPP_ALIGN_UP(imageWarp.rows, VPC_STRIDE_HEIGHT);
    faceObject->imgAffine.buf.deviceData = deviceData;
    faceObject->imgAffine.buf.dataSize = dataSize;
    return APP_ERR_OK;
}

APP_ERROR WarpAffine::ApplyWarpAffine(std::shared_ptr<FaceObject> faceObject)
{
    LogDebug << "WarpAffine[" << instanceId_ << "]: begin to apply warp affine.";
    std::vector<cv::Point2f> srcPoints = {};
    std::vector<cv::Point2f> destPoints = {};
    GetSrcLandmark(srcPoints);
    GetDstLandmark(destPoints, faceObject);

    cv::Mat imageN12;
    cv::Mat warpDst;
    GetCropImage(faceObject, imageN12);
    CalWarpImage(srcPoints, destPoints, warpDst, imageN12);
    APP_ERROR ret = SetWarpImage(faceObject, warpDst);
    if (ret != APP_ERR_OK) {
        LogError << "WarpAffine[" << instanceId_ << "]: SetWarpImage failed ret=" << ret;
        return ret;
    }

    return APP_ERR_OK;
}
} // namespace ascendFaceRecognition
