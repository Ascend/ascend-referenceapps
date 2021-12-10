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

#include "ImagePreprocess.h"

#include <iostream>

#include "TextDetection/TextDetection.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
#include "FileManager/FileManager.h"
#include "PointerDeleter/PointerDeleter.h"
#include "CommonDataType/CommonDataType.h"
#include "Common/CommonType.h"

using namespace ascendBaseModule;

ImagePreprocess::ImagePreprocess() {}

ImagePreprocess::~ImagePreprocess() {}

APP_ERROR ImagePreprocess::ParseConfig(ConfigParser &configParser)
{
    std::string itemCfgStr = std::string("TextDetection.dynamicHWList");
    std::vector<uint32_t> dynamicHWList;
    APP_ERROR ret = configParser.GetVectorUint32Value(itemCfgStr, dynamicHWList);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImagePreprocess[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }
    const size_t dynamicListLen = 2;
    if (dynamicHWList.size() != dynamicListLen) {
        LogFatal << "ImagePreprocess[" << instanceId_ << "]: Invalid value of " << itemCfgStr \
                 << ", it need to have 2 values.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    for (auto HWTmp: dynamicHWList) {
        if ((HWTmp % VPC_WIDTH_ALIGN) != 0) {
            LogFatal << "ImagePreprocess[" << instanceId_ << "]: Invalid value of " << itemCfgStr
                     << ", it has to be a multiple of " << VPC_WIDTH_ALIGN << ".";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        if ((HWTmp % VPC_HEIGHT_ALIGN) != 0) {
            LogFatal << "ImagePreprocess[" << instanceId_ << "]: Invalid value of " << itemCfgStr
                     << ", it has to be a multiple of " << VPC_HEIGHT_ALIGN << ".";
            return APP_ERR_COMM_INVALID_PARAM;
        }
    }
    hwMin_ = std::min(dynamicHWList[0], dynamicHWList[1]);
    hwMax_ = std::max(dynamicHWList[0], dynamicHWList[1]);

    itemCfgStr = std::string("SystemConfig.debugMode");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, debugMode_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImagePreprocess[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }

    return ret;
}

APP_ERROR ImagePreprocess::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    AssignInitArgs(initArgs);
    LogDebug << "ImagePreprocess[" << instanceId_ << "]: ImagePreprocess begin to init instance " \
             << initArgs.instanceId << ".";
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImagePreprocess[" << instanceId_ << "]: Failed to parse config params: " \
                 << GetAppErrCodeInfo(ret) << ".";
        return ret;
    }

    ret = aclrtCreateStream(&dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImagePreprocess[" << instanceId_ << "]: Failed to execute aclrtCreateStream, ret=" << ret << ".";
        return ret;
    }

    dvppObjPtr_.reset(new DvppCommon(dvppStream_));
    ret = dvppObjPtr_->Init();
    if (ret != APP_ERR_OK) {
        LogFatal << "ImagePreprocess[" << instanceId_ << "]: Failed to create dvpp channel, ret = " << ret << ".";
        return ret;
    }

    LogDebug << "ImagePreprocess[" << instanceId_ << "]: ImagePreprocess init successfully.";
    return APP_ERR_OK;
}

APP_ERROR ImagePreprocess::DeInit(void)
{
    LogDebug << "ImagePreprocess[" << instanceId_ << "]: ImagePreprocess begin to deinit.";
    APP_ERROR ret = APP_ERR_OK;

    if (dvppObjPtr_ != nullptr) {
        ret = dvppObjPtr_->DeInit();
        if (ret != APP_ERR_OK) {
            LogFatal << "ImagePreprocess[" << instanceId_ << "]: Faild to execute dvpp DeInit, ret = " << ret << ".";
            return ret;
        }
        dvppObjPtr_ = nullptr;
    }

    ret = aclrtDestroyStream(dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImagePreprocess[" << instanceId_ << "]: Failed to destroy dvpp stream, ret = " << ret << ".";
        return ret;
    }
    dvppStream_ = nullptr;

    LogDebug << "ImagePreprocess[" << instanceId_ << "]: ImagePreprocess deinit successfully.";
    return ret;
}

void ImagePreprocess::FindTheBestHWSize(uint32_t &bestWidth, uint32_t &bestHight,
                                        uint32_t inWidth, uint32_t inHeight) const
{
    const int halfValue = 2;
    auto threshold = static_cast<float>(hwMax_ + hwMin_) / halfValue;
    // If the maximum value of inWidth and inHeight is less than the middle value of hwMin_ and hwMax_,
    // we select the minimum value as the model width and height without considering the ratio of inWidth and inHeight
    if (std::max(inWidth, inHeight) < threshold) {
        bestWidth = hwMin_;
        bestHight = hwMin_;
        return;
    }

    // If the minimum value of inWidth and inHeight is larger than the middle value of hwMin_ and hwMax_,
    // we select the maximum value as the model width and height without considering the ratio of inWidth and inHeight
    if (std::min(inWidth, inHeight) > threshold) {
        bestWidth = hwMax_;
        bestHight = hwMax_;
        return;
    }

    auto ratio = static_cast<float>(inWidth) / inHeight;
    const float widthGtHeight = 1.5;
    const float heightGtWidht = 0.67;
    if (ratio > widthGtHeight) {
        bestWidth = hwMax_;
        bestHight = hwMin_;
        return;
    }

    if (ratio < heightGtWidht) {
        bestWidth = hwMin_;
        bestHight = hwMax_;
        return;
    }

    bestWidth = hwMax_;
    bestHight = hwMax_;
    return;
}

APP_ERROR ImagePreprocess::Process(std::shared_ptr<void> inputData)
{
    jpegDecodeStatic_.RunTimeStatisticStart("JpegDecode_Execute_Time", instanceId_, true);
    std::shared_ptr<SendInfo> sendData = std::static_pointer_cast<SendInfo>(inputData);
    LogDebug << "ImagePreprocess[" << instanceId_ << "]: [" << sendData->imageName << "] Process start.";
    // Begin to decode jpeg image
    APP_ERROR ret = dvppObjPtr_->CombineJpegdProcess(*(sendData->imageData), PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                                     true);
    if (ret != APP_ERR_OK) {
        LogError << "ImagePreprocess[" << instanceId_ << "]: [" << sendData->imageName \
                 << "] Failed to process decode, ret = " << ret << ".";
        RELEASE_DVPP_DATA(dvppObjPtr_->GetInputImage()->data);
        return ret;
    }
    // Release input image buffer on dvpp
    RELEASE_DVPP_DATA(dvppObjPtr_->GetInputImage()->data);
    jpegDecodeStatic_.RunTimeStatisticStop();

    // Get output of decoded jpeg image
    std::shared_ptr<DvppDataInfo> decodeImg = dvppObjPtr_->GetDecodedImage();
    sendData->imageWidth = decodeImg->width;
    sendData->imageHeight = decodeImg->height;
    jpegResizeStatic_.RunTimeStatisticStart("JpegResize_Execute_Time", instanceId_);
    // Begin to resize the decoded image
    DvppDataInfo resizeInfo;
    FindTheBestHWSize(resizeInfo.width, resizeInfo.height, sendData->imageWidth, sendData->imageHeight);
    resizeInfo.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    ret = dvppObjPtr_->CombineResizeProcess(*decodeImg, resizeInfo, true, VPC_PT_FIT);
    if (ret != APP_ERR_OK) {
        LogError << "ImagePreprocess[" << instanceId_ << "]: [" << sendData->imageName \
                 << "] Failed to process resize, ret = " << ret << ".";
        RELEASE_DVPP_DATA(decodeImg->data);
        return ret;
    }
    RELEASE_DVPP_DATA(decodeImg->data);
    jpegResizeStatic_.RunTimeStatisticStop();

    std::shared_ptr<DvppDataInfo> resizeImg = dvppObjPtr_->GetResizedImage();
    sendData->resizedData.data.reset(resizeImg->data, acldvppFree);
    sendData->resizedData.lenOfByte = resizeImg->dataSize;
    sendData->resizedData.width = resizeImg->width;
    sendData->resizedData.height = resizeImg->height;

    if (debugMode_) {
        SaveResizedImage(sendData);
        LogInfo << "ImagePreprocess[" << instanceId_ << "]: [" << sendData->imageName << "] Resize WH: " \
                << sendData->resizedData.width << ", " << sendData->resizedData.height << ".";
    }
    SendToNextModule(MT_TextDetection, sendData, instanceId_);
    LogDebug << "ImagePreprocess[" << instanceId_ << "]: [" << sendData->imageName << "] Process end.";
    return APP_ERR_OK;
}

void ImagePreprocess::SaveResizedImage(std::shared_ptr<SendInfo> sendData)
{
    void *hostPtr = nullptr;
    APP_ERROR ret = aclrtMallocHost(&hostPtr, sendData->resizedData.lenOfByte);
    if (ret != APP_ERR_OK) {
        LogWarn << "ImagePreprocess[" << instanceId_ << "]: [" << sendData->imageName \
                << "]: Failed to malloc on host, ret = " << ret << ".";
        return;
    }
    std::shared_ptr<void> hostSharedPtr(hostPtr, aclrtFreeHost);
    ret = aclrtMemcpy(hostPtr, sendData->resizedData.lenOfByte, sendData->resizedData.data.get(),
                      sendData->resizedData.lenOfByte, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != APP_ERR_OK) {
        LogWarn << "ImagePreprocess[" << instanceId_ << "]: [" << sendData->imageName \
                << "]: Failed to memcpy from device to host, ret = " << ret << ".";
        return;
    }
    // Save the resized result
    std::stringstream fileName;
    fileName << "resize_image_" << sendData->imageName;
    SaveFileWithTimeStamp(hostSharedPtr, sendData->resizedData.lenOfByte, moduleName_, fileName.str(), ".yuv");
}