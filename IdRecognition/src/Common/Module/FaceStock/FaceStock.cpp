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
#include "FaceStock.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "Log/Log.h"
#include "ErrorCode/ErrorCode.h"
#include "DataType/DataType.h"
namespace ascendFaceRecognition {
const int YUV_BGR_SIZE_CONVERT_4 = 4;
const float YUV_BGR_SIZE_CONVERT_1_5 = 1.5;

APP_ERROR FaceStock::ParseConfig(ConfigParser &configParser)
{
    LogInfo << "FaceStock[" << instanceId_ << "]: FaceStock parse config.";

    std::string itemCfgStr;

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".width_stock_img");
    APP_ERROR ret = configParser.GetIntValue(itemCfgStr, widthStockImg_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".height_stock_img");
    ret = configParser.GetIntValue(itemCfgStr, heightStockImg_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    return ret;
}

APP_ERROR FaceStock::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "FaceStock[" << instanceId_ << "]: FaceStock init star.";

    AssignInitArgs(initArgs);
    // get feature lib
    faceFeatureLib_ = static_cast<FaceFeatureLib *>(initArgs.userData);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceStock[" << instanceId_ << "]:Fail to parse config params.";
        return ret;
    }

    isStop_ = false;
    LogInfo << "FaceStock[" << instanceId_ << "]: FaceStock init success.";
    return APP_ERR_OK;
}

APP_ERROR FaceStock::DeInit(void)
{
    LogInfo << "FaceStock[" << instanceId_ << "]:FaceStock deinit start.";

    StopAndDestroyQueue();

    LogInfo << "FaceStock[" << instanceId_ << "]: FaceStock deinit success.";
    return APP_ERR_OK;
}

APP_ERROR FaceStock::CalculateROI(ImageInfo &imgOrigin, FaceObject &faceObject, cv::Rect &roi) const
{
    LogDebug << "FaceStock[" << instanceId_ << "]: [FaceStock] Begin to calculate ROI.";
    float lenTmp;
    float widthTmp;
    float heightTmp;
    unsigned int widthCrop;
    unsigned int heightCrop;

    widthTmp = faceObject.info.width * YUV_BGR_SIZE_CONVERT_4 / YUV_BGR_SIZE_CONVERT_3;
    heightTmp = faceObject.info.height;
    lenTmp = (widthTmp > heightTmp) ? widthTmp : heightTmp;
    widthCrop = static_cast<unsigned int>(YUV_BGR_SIZE_CONVERT_1_5 * lenTmp);
    if (widthCrop > imgOrigin.width) {
        widthCrop = imgOrigin.width;
    }
    heightCrop = static_cast<unsigned int>(widthCrop * heightStockImg_ / widthStockImg_);
    if (heightCrop > imgOrigin.height) {
        heightCrop = imgOrigin.height;
    }
    roi.width = widthCrop;
    roi.height = heightCrop;
    roi.x = faceObject.info.minx + faceObject.info.width / YUV_BGR_SIZE_CONVERT_2 - roi.width / YUV_BGR_SIZE_CONVERT_2;
    if (roi.x < 0) {
        roi.x = 0;
    }
    roi.y =
        faceObject.info.miny + faceObject.info.height / YUV_BGR_SIZE_CONVERT_2 - roi.height / YUV_BGR_SIZE_CONVERT_2;
    if (roi.y < 0) {
        roi.y = 0;
    }
    if ((roi.x + roi.width) > (int)imgOrigin.width) {
        roi.width = imgOrigin.width - roi.x;
    }
    if ((roi.y + roi.height) > (int)imgOrigin.height) {
        roi.height = imgOrigin.height - roi.y;
    }

    return APP_ERR_OK;
}

APP_ERROR FaceStock::ValidateParm(std::shared_ptr<FrameAiInfo> frameAiInfo, unsigned int faceIdx) const
{
    if (frameAiInfo.get() == nullptr) {
        return APP_ERR_COMM_INVALID_POINTER;
    }
    if (frameAiInfo->face.size() > 1) {
        LogDebug << "FaceStock[" << instanceId_ << "]: Multiple face in the image.";
        float areaFaceMax = 0;
        float areaFaceTemp = 0;
        for (unsigned int i = 0; i < frameAiInfo->face.size(); i++) {
            areaFaceTemp = frameAiInfo->face[i].info.width * frameAiInfo->face[i].info.height;
            if (areaFaceTemp > areaFaceMax) {
                areaFaceMax = areaFaceTemp;
                faceIdx = i;
            }
        }
    } else if (frameAiInfo->face.size() == 0) {
        LogError << "FaceStock[" << instanceId_ << "]:No face in the image.";
        return APP_ERR_OK;
    }
    return APP_ERR_OK;
}

APP_ERROR FaceStock::GetFaceFeatureStatus() const
{
    return featureIsExist_;
}

void FaceStock::SetFaceFeatureStatus()
{
    featureIsExist_ = APP_ERROR_FACE_WEB_USE_SYSTEM_ERROR;
}

APP_ERROR FaceStock::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "FaceStock[" << instanceId_ << "]: Begin to process data.";
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::static_pointer_cast<FrameAiInfo>(inputData);
    ImageInfo imageInfo;
    cv::Rect roi;
    cv::Mat cropedImage;
    cv::Mat dstResize;
    unsigned int faceIdx = 0;

    // 1. validate inputs
    APP_ERROR ret = ValidateParm(frameAiInfo, faceIdx);
    if (ret != APP_ERR_OK) {
        LogError << "FaceStock[" << instanceId_ << "]:Fail to input param info.";
        return ret;
    }
    // 2. calculate roi of target face
    CalculateROI(frameAiInfo->imgOrigin, frameAiInfo->face[faceIdx], roi);

    // 3. get croped face data
    /* do nv12 to bgr888 convert */
    cv::Mat srcNV12Mat(frameAiInfo->imgOrigin.height * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2,
        frameAiInfo->imgOrigin.width, CV_8UC1, frameAiInfo->imgOrigin.buf.hostData.get());
    cv::Mat dstBGR888(frameAiInfo->imgOrigin.height, frameAiInfo->imgOrigin.width, CV_8UC3);
    cv::cvtColor(srcNV12Mat, dstBGR888, cv::COLOR_YUV2BGR_NV12);
    cropedImage = dstBGR888(roi);
    cv::resize(cropedImage, dstResize, cv::Size(widthStockImg_, heightStockImg_), 0, 0, cv::INTER_LINEAR);

    // 4. construct ImageInfo
    imageInfo.format = FACE_IMAGE_FORMAT_BGR888;
    imageInfo.width = widthStockImg_;
    imageInfo.height = heightStockImg_;
    uint32_t imageSize = widthStockImg_ * heightStockImg_ * YUV_BGR_SIZE_CONVERT_3;
    auto tmpStockImg = std::make_shared<uint8_t>(imageSize);
    imageInfo.buf.hostData.reset(new uint8_t[imageSize], [](uint8_t *p) { delete[] p; });
    std::copy(dstResize.data, dstResize.data + imageSize, imageInfo.buf.hostData.get());
    imageInfo.buf.dataSize = imageSize;

    // 5. insert feature and individual info
    featureIsExist_ = faceFeatureLib_->InsertFeatureToLib(frameAiInfo->face[faceIdx].featureVector,
        frameAiInfo->info.personInfo, imageInfo, false);
    if (!(featureIsExist_ == APP_ERR_OK || featureIsExist_ == APP_ERROR_FACE_WEB_USE_REPEAT_REG)) {
        LogError << "FaceStock[" << instanceId_ << "]:Fail to registor person info" << GetAppErrCodeInfo(ret) << ".";
        return featureIsExist_;
    }

    return APP_ERR_OK;
}
} // namespace ascendFaceRecognition
