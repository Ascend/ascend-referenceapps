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

#include "WarpPerspective.h"

#include <iostream>

#include "TextRecognition/TextRecognition.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
#include "FileManager/FileManager.h"
#include "PointerDeleter/PointerDeleter.h"
#include "CommonDataType/CommonDataType.h"

using namespace ascendBaseModule;

namespace {
    const int KEY_POINT0_DIM = 2;
}

WarpPerspective::WarpPerspective() {}

WarpPerspective::~WarpPerspective() {}

APP_ERROR WarpPerspective::ParseConfig(ConfigParser &configParser)
{
    std::string itemCfgStr;
    APP_ERROR ret;

    itemCfgStr = std::string("TextRecognition.modelHeight");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, dstHeight_);
    if (ret != APP_ERR_OK) {
        LogFatal << "WarpPerspective[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }

    itemCfgStr = std::string("TextRecognition.ignoreWarpPerspective");
    ret = configParser.GetBoolValue(itemCfgStr, ignoreWarpPerspective_);
    if (ret != APP_ERR_OK) {
        LogFatal << "WarpPerspective[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }

    itemCfgStr = std::string("SystemConfig.debugMode");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, debugMode_);
    if (ret != APP_ERR_OK) {
        LogFatal << "WarpPerspective[" << instanceId_ << "]: Failed to get " << itemCfgStr << ", ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR WarpPerspective::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    AssignInitArgs(initArgs);
    LogDebug << "WarpPerspective[" << instanceId_ << "]: WarpPerspective begin to init instance " \
             << initArgs.instanceId << ".";
    // initialize config params
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "WarpPerspective[" << instanceId_ << "]: Failed to parse config params, ret=" << ret << "("
                 << GetAppErrCodeInfo(ret) << ").";
        return ret;
    }

    // Create dvpp stream and dvpp object for cropping the text
    ret = aclrtCreateStream(&dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "WarpPerspective[" << instanceId_ << "]: Failed to execute aclrtCreateStream, ret=" << ret << ".";
        return ret;
    }

    dvppObjPtr_.reset(new DvppCommon(dvppStream_));
    ret = dvppObjPtr_->Init();
    if (ret != APP_ERR_OK) {
        LogFatal << "WarpPerspective[" << instanceId_ << "]: Failed to create dvpp channel, ret = " << ret << ".";
        return ret;
    }

    LogDebug << "WarpPerspective[" << instanceId_ << "]: WarpPerspective init successfully.";
    return APP_ERR_OK;
}

APP_ERROR WarpPerspective::DeInit(void)
{
    LogInfo << "WarpPerspective[" << instanceId_ << "]: WarpPerspective begin to deinit.";

    if (dvppObjPtr_ != nullptr) {
        APP_ERROR ret = dvppObjPtr_->DeInit();
        if (ret != APP_ERR_OK) {
            LogFatal << "WarpPerspective[" << instanceId_ << "]: Failed to execute dvpp DeInit, ret = " << ret << ".";
            return ret;
        }
        dvppObjPtr_ = nullptr;
    }

    LogInfo << "WarpPerspective[" << instanceId_ << "]: WarpPerspective deinit successfully.";
    return APP_ERR_OK;
}

APP_ERROR WarpPerspective::Process(std::shared_ptr<void> inputData)
{
    std::shared_ptr<SendInfo> sendData = std::static_pointer_cast<SendInfo>(inputData);
    if (sendData->itemNum == 0) {
        SendToNextModule(MT_TextRecognition, sendData, instanceId_);
        return APP_ERR_OK;
    }

    warpAffineStatic_.RunTimeStatisticStart("WarpAffine_Execute_Time", instanceId_);
    APP_ERROR ret = CropTextBox(sendData);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    if (ignoreWarpPerspective_) {
        if (debugMode_ != 0) {
            std::shared_ptr<uint8_t> hostPtr = std::make_shared<uint8_t>();
            APP_ERROR ret = TransToHost(sendData, hostPtr);
            if (ret != APP_ERR_OK) {
                return ret;
            }
            // Save the cropped result
            std::stringstream fileName;
            fileName << "crop_image_" << sendData->imageName << "_" << sendData->itemInfo->itemId;
            SaveFileWithTimeStamp(hostPtr, sendData->itemInfo->cropData.lenOfByte, moduleName_, fileName.str(), ".yuv");
        }
        sendData->itemInfo->perspectiveData = sendData->itemInfo->cropData;
        SendToNextModule(MT_TextRecognition, sendData, instanceId_);
        warpAffineStatic_.RunTimeStatisticStop();
        return APP_ERR_OK;
    }
    ret = WarpPerspectiveProcess(sendData);
    if (ret != APP_ERR_OK) {
        LogError << "WarpPerspective[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                 << sendData->itemInfo->itemId << "]: Failed to process warp perspective!";
        return ret;
    }
    SendToNextModule(MT_TextRecognition, sendData, instanceId_);
    warpAffineStatic_.RunTimeStatisticStop();
    return APP_ERR_OK;
}

void WarpPerspective::InitCoordinate(std::shared_ptr<SendInfo> sendData)
{
    leftTopX_ = CONVERT_TO_EVEN(std::min(std::min(sendData->itemInfo->boxCoordinate[TEXT_BOX_X0_INDEX],
                                                  sendData->itemInfo->boxCoordinate[TEXT_BOX_X1_INDEX]),
                                         std::min(sendData->itemInfo->boxCoordinate[TEXT_BOX_X2_INDEX],
                                                  sendData->itemInfo->boxCoordinate[TEXT_BOX_X3_INDEX])));
    leftTopY_ = CONVERT_TO_EVEN(std::min(std::min(sendData->itemInfo->boxCoordinate[TEXT_BOX_Y0_INDEX],
                                                  sendData->itemInfo->boxCoordinate[TEXT_BOX_Y1_INDEX]),
                                         std::min(sendData->itemInfo->boxCoordinate[TEXT_BOX_Y2_INDEX],
                                                  sendData->itemInfo->boxCoordinate[TEXT_BOX_Y3_INDEX])));
    rightBotX_ = CONVERT_TO_ODD(std::max(std::max(sendData->itemInfo->boxCoordinate[TEXT_BOX_X0_INDEX],
                                                  sendData->itemInfo->boxCoordinate[TEXT_BOX_X1_INDEX]),
                                         std::max(sendData->itemInfo->boxCoordinate[TEXT_BOX_X2_INDEX],
                                                  sendData->itemInfo->boxCoordinate[TEXT_BOX_X3_INDEX])));
    /* In order not to lose pixel due to odd numbers, we add ODD_NUM_1 here */
    rightBotY_ = CONVERT_TO_ODD(std::max(std::max(sendData->itemInfo->boxCoordinate[TEXT_BOX_Y0_INDEX],
                                                  sendData->itemInfo->boxCoordinate[TEXT_BOX_Y1_INDEX]),
                                         std::max(sendData->itemInfo->boxCoordinate[TEXT_BOX_Y2_INDEX],
                                                  sendData->itemInfo->boxCoordinate[TEXT_BOX_Y3_INDEX])) + ODD_NUM_1);
}

APP_ERROR WarpPerspective::CropTextBox(std::shared_ptr<SendInfo> sendData)
{
    /* Initialize the coordinate according to last model output in sendData */
    InitCoordinate(sendData);
    /* Crop in device */
    DvppCropInputInfo cropInputData;
    cropInputData.dataInfo.data = sendData->resizedData.data.get();
    cropInputData.dataInfo.dataSize = sendData->resizedData.lenOfByte;
    cropInputData.dataInfo.width = sendData->resizedData.width;
    cropInputData.dataInfo.height = sendData->resizedData.height;
    cropInputData.dataInfo.widthStride = DVPP_ALIGN_UP(cropInputData.dataInfo.width, VPC_STRIDE_WIDTH);
    cropInputData.dataInfo.heightStride = DVPP_ALIGN_UP(cropInputData.dataInfo.height, VPC_STRIDE_HEIGHT);
    cropInputData.roi.left = leftTopX_;
    cropInputData.roi.up = leftTopY_;
    cropInputData.roi.right = rightBotX_;
    cropInputData.roi.down = rightBotY_;

    DvppDataInfo output;
    output.width = rightBotX_ - leftTopX_ + ODD_NUM_1;
    output.height = rightBotY_ - leftTopY_ + ODD_NUM_1;
    if (ignoreWarpPerspective_) {
        output.width = std::max(dstHeight_, dstHeight_ * (output.width / output.height));
        output.height = dstHeight_;
    }
    APP_ERROR ret = dvppObjPtr_->CombineCropProcess(cropInputData, output, true);
    if (ret != APP_ERR_OK) {
        LogError << "WarpPerspective[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                 << sendData->itemInfo->itemId << "]: Failed to process crop, ret = " << ret << ".";
        return ret;
    }

    // Get output of cropped image
    sendData->itemInfo->cropData.lenOfByte = dvppObjPtr_->GetCropedImage()->dataSize;
    sendData->itemInfo->cropData.data.reset(dvppObjPtr_->GetCropedImage()->data, acldvppFree);
    sendData->itemInfo->cropData.width = dvppObjPtr_->GetCropedImage()->widthStride;
    sendData->itemInfo->cropData.height = dvppObjPtr_->GetCropedImage()->heightStride;
    if (debugMode_ != 0) {
        LogInfo << "WarpPerspective[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                << sendData->itemInfo->itemId << "]: Cropped input: " << leftTopX_ << ", " << leftTopY_ << ", " \
                << rightBotX_ << ", " << rightBotY_ << ", Cropped WH: " << sendData->itemInfo->cropData.width \
                << ", " << sendData->itemInfo->cropData.height;
    }
    return APP_ERR_OK;
}

APP_ERROR WarpPerspective::WarpPerspectiveProcess(std::shared_ptr<SendInfo> sendData)
{
    LogDebug << "WarpPerspective[" << instanceId_ << "]: [" << sendData->imageName << "-" << sendData->itemInfo->itemId
             << "]: Process start.";
    // Copy the cropped memory from device to host
    std::shared_ptr<uint8_t> hostPtr = std::make_shared<uint8_t>();
    APP_ERROR ret = TransToHost(sendData, hostPtr);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    sendData->itemInfo->cropData.data = hostPtr;
    if (debugMode_ != 0) {
        // Save the cropped result
        std::stringstream fileName;
        fileName << "crop_image_" << sendData->imageName << "_" << sendData->itemInfo->itemId;
        SaveFileWithTimeStamp(hostPtr, sendData->itemInfo->cropData.lenOfByte, moduleName_, fileName.str(), ".yuv");
    }

    // Convert NV12 to RGB888
    cv::Mat srcNV12Mat(sendData->itemInfo->cropData.height * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2,
                       sendData->itemInfo->cropData.width, CV_8UC1,
                       sendData->itemInfo->cropData.data.get());
    cv::Mat dstRGB888(sendData->itemInfo->cropData.height,
                      sendData->itemInfo->cropData.width, CV_8UC3);
    cv::cvtColor(srcNV12Mat, dstRGB888, cv::COLOR_YUV2RGB_NV12); // COLOR_YUV2RGB_NV12
    ret = ApplyWarpPerspective(sendData, dstRGB888);
    if (ret != APP_ERR_OK) {
        LogWarn << "WarpPerspective[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                << sendData->itemInfo->itemId << "]: Failed to apply warp perspective!";
        return ret;
    }
    LogDebug << "WarpPerspective[" << instanceId_ << "]: [" << sendData->imageName << "-" << sendData->itemInfo->itemId
             << "]: Process end.";
    return APP_ERR_OK;
}

APP_ERROR WarpPerspective::ApplyWarpPerspective(std::shared_ptr<SendInfo> sendData, cv::Mat &imgRGB888)
{
    LogDebug << "WarpPerspective[" << instanceId_ << "]: Begin to apply warp perspective.";
    uint32_t srcWidth = sendData->itemInfo->cropData.width;
    uint32_t srcHeight = sendData->itemInfo->cropData.height;
    dstWidth_ = DVPP_ALIGN_UP((uint32_t)(srcWidth * dstHeight_ / (float)srcHeight), VPC_STRIDE_WIDTH);
    // four vertices after warp perspective, arranged by left top, right top, left bottom, right bottom
    uint32_t keyPointAfter[TEXT_BOX_COORDINATES_NUM] = {0, 0, 0, dstHeight_, dstWidth_, dstHeight_, dstWidth_, 0};
    auto &keyPointBefore = sendData->itemInfo->boxCoordinate;
    cv::Point2f srcPoints[TEXT_BOX_COORDINATES_NUM / KEY_POINT0_DIM];
    cv::Point2f destPoints[TEXT_BOX_COORDINATES_NUM / KEY_POINT0_DIM];
    for (size_t i = 0; i < TEXT_BOX_COORDINATES_NUM / KEY_POINT0_DIM; i++) {
        keyPointBefore[i * KEY_POINT0_DIM] -= leftTopX_;
        keyPointBefore[i * KEY_POINT0_DIM + 1] -= leftTopY_;
        srcPoints[i] = cv::Point2f(keyPointBefore[i * KEY_POINT0_DIM], keyPointBefore[i * KEY_POINT0_DIM + 1]);
        destPoints[i] = cv::Point2f(keyPointAfter[i * KEY_POINT0_DIM], keyPointAfter[i * KEY_POINT0_DIM + 1]);
    }

    // Get perspective transform matrix
    cv::Mat warpMat = cv::getPerspectiveTransform(srcPoints, destPoints);
    std::unique_ptr<uint8_t[]> warpDstData(new uint8_t[dstWidth_ * dstHeight_ * YUV_BGR_SIZE_CONVERT_3]);
    cv::Mat warpDst(dstHeight_, dstWidth_, CV_8UC3, reinterpret_cast<void **>(warpDstData.get()));
    cv::warpPerspective(imgRGB888, warpDst, warpMat, warpDst.size());

    /* Convert RGB888 to NV12 */
    cv::Mat srcRGBMat(dstHeight_, dstWidth_, CV_8UC3, warpDstData.get());
    cv::Mat dstNV12Mat(dstHeight_ * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2, dstWidth_, CV_8UC1);
    cv::cvtColor(srcRGBMat, dstNV12Mat, cv::COLOR_RGB2YUV_I420); // COLOR_RGB2YUV_I420

    // Calculate the size for the data after color convertion
    uint32_t dstSize = dstHeight_ * dstWidth_ * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2;
    sendData->itemInfo->perspectiveData.lenOfByte = dstSize;

    // Malloc dvpp memory for saving the data
    uint8_t *dvppBuf = nullptr;
    APP_ERROR ret = acldvppMalloc((void **)(&dvppBuf), sendData->itemInfo->perspectiveData.lenOfByte);
    if (ret != APP_ERR_OK) {
        LogError << "WarpPerspective[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                 << sendData->itemInfo->itemId << "]: Failed to allocate memory on dvpp, ret = " << ret << ".";
        return ret;
    }
    sendData->itemInfo->perspectiveData.data.reset(dvppBuf, acldvppFree);
    // Memcpy the output data from host to device
    ret = aclrtMemcpy(dvppBuf, sendData->itemInfo->perspectiveData.lenOfByte,
                      dstNV12Mat.data, sendData->itemInfo->perspectiveData.lenOfByte,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != APP_ERR_OK) {
        LogError << "WarpPerspective[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                 << sendData->itemInfo->itemId << "]: Failed to copy memory from host to device, ret = " << ret << ".";
        return ret;
    }

    sendData->itemInfo->perspectiveData.height = dstHeight_;
    sendData->itemInfo->perspectiveData.width = dstWidth_;
    return APP_ERR_OK;
}

APP_ERROR WarpPerspective::TransToHost(std::shared_ptr<SendInfo> sendData, std::shared_ptr<uint8_t> &hostSharedPtr)
{
    uint8_t *hostPtr = nullptr;
    APP_ERROR ret = aclrtMallocHost((void **)(&hostPtr), sendData->itemInfo->cropData.lenOfByte);
    if (ret != APP_ERR_OK) {
        LogError << "WarpPerspective[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                 << sendData->itemInfo->itemId << "]: Failed to malloc on host, ret = " << ret << ".";
        return ret;
    }
    hostSharedPtr.reset(hostPtr, aclrtFreeHost);
    ret = aclrtMemcpy((void *)hostSharedPtr.get(), sendData->itemInfo->cropData.lenOfByte,
                      sendData->itemInfo->cropData.data.get(),
                      sendData->itemInfo->cropData.lenOfByte,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != APP_ERR_OK) {
        LogError << "WarpPerspective[" << instanceId_ << "]: [" << sendData->imageName << "-" \
                 << sendData->itemInfo->itemId << "]: Failed to memcpy from device to host, ret = " \
                 << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}
