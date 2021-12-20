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
#include "ImageDecoder.h"

#include <iostream>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
#include "FileManager/FileManager.h"
#include "PointerDeleter/PointerDeleter.h"
#include "CommonDataType/CommonDataType.h"

namespace ascendFaceRecognition {
ImageDecoder::~ImageDecoder()
{
    if (!isDeInited_) {
        DeInit();
    }
}

APP_ERROR ImageDecoder::ParseConfig(ConfigParser &configParser)
{
    std::string itemCfgStr(moduleName_ + std::to_string(instanceId_) + std::string(".resizeWidth"));
    APP_ERROR ret = configParser.GetIntValue(itemCfgStr, resizeWidth_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    if ((resizeWidth_ % VPC_WIDTH_ALIGN) != 0) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: invalid value of " << itemCfgStr
                 << ", it has to be a multiple of " << VPC_WIDTH_ALIGN << ".";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".resizeHeight");
    ret = configParser.GetIntValue(itemCfgStr, resizeHeight_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    if ((resizeHeight_ % VPC_HEIGHT_ALIGN) != 0) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: invalid value of " << itemCfgStr
                 << ", it has to be a multiple of " << VPC_HEIGHT_ALIGN << ".";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    return ret;
}

APP_ERROR ImageDecoder::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "ImageDecoder[" << instanceId_ << "]: Begin to init image decoder instance " << initArgs.instanceId
            << initArgs.instanceId << ".";

    AssignInitArgs(initArgs);

    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: Fail to parse config params."
        << GetAppErrCodeInfo(ret) << ".";
        return ret;
    }

    ret = aclrtCreateStream(&dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: aclrtCreateStream failed, ret=" << ret << ".";
        return ret;
    }
#ifdef ASCEND_FACE_USE_ACL_DVPP
    pDvpp_.reset(new DvppCommon(dvppStream_));
#else
    pDvpp_.reset(new DvppCommonDevice());
#endif

    ret = pDvpp_->Init();
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: dvpp channel create failed!";
        return ret;
    }

    LogInfo << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::Init OK.";
    return APP_ERR_OK;
}

APP_ERROR ImageDecoder::DeInit(void)
{
    LogInfo << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::begin to deinit.";

    isDeInited_ = true;
    APP_ERROR ret = APP_ERR_OK;
    StopAndDestroyQueue();
    if (pDvpp_ != nullptr) {
        ret = pDvpp_->DeInit();
        if (ret != APP_ERR_OK) {
            LogFatal << "ImageDecoder[" << instanceId_ << "]: dvpp deinit failed.";
            return ret;
        }
        pDvpp_.reset();
    }
    ret = aclrtDestroyStream(dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: Fail to destroy dvppstream.";
        return ret;
    }
    dvppStream_ = nullptr;

    LogInfo << "ImageDecoder[" << instanceId_ << "]: ImageDecoder deinit success.";
    return ret;
}
#ifdef ASCEND_FACE_USE_ACL_DVPP
APP_ERROR ImageDecoder::VpcHandle(uint8_t *inputdata, uint8_t *&vpcOutBuffer)
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::begin to resize image.";

    DvppDataInfo input;
    input.width = decodedWidth_;
    input.height = decodedHeight_;
    input.widthStride = decodedWidthStride_;
    input.heightStride = decodedHeightStride_;
    input.dataSize = decodedDataSize_;
    input.data = inputdata;

    DvppDataInfo output;
    output.width = resizeWidth_;
    output.height = resizeHeight_;
    DvppCommon::GetVpcOutputStrideSize(output.width, output.height, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                       output.widthStride, output.heightStride);
    uint32_t vpcOutBufferSize = 0;
    DvppCommon::GetVpcDataSize(resizeWidth_, resizeHeight_, PIXEL_FORMAT_YUV_SEMIPLANAR_420, vpcOutBufferSize);

    APP_ERROR ret = acldvppMalloc(reinterpret_cast<void **>(&vpcOutBuffer), vpcOutBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]: acldvppMalloc failed, ret[" << ret << "] vpcOutBufferSize"
                 << vpcOutBufferSize << ".";
        return ret;
    }
    ret = aclrtMemset(reinterpret_cast<void *>(vpcOutBuffer), vpcOutBufferSize, 0, vpcOutBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]: aclrtMemset failed, ret[" << ret << "] vpcOutBufferSize"
                 << vpcOutBufferSize << ".";
        return ret;
    }
    output.data = vpcOutBuffer;
    output.dataSize = vpcOutBufferSize;

    ret = pDvpp_->VpcResize(input, output, true, VPC_PT_PADDING);
    if (ret != APP_ERR_OK) {
        // Release the output buffer when resize failed, otherwise release it after use
        RELEASE_DVPP_DATA(output.data);
    }

    return APP_ERR_OK;
}

APP_ERROR ImageDecoder::JpegDecodeHandle(uint8_t *inputData, uint32_t inputDataLen, uint8_t *&jpegDecodeBuffer,
    uint32_t jpegWidth, uint32_t jpegHeight)
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::begin to decode image data.";
    uint32_t W_Aligned = DVPP_ALIGN_UP(jpegWidth, JPEG_WIDTH_ALIGN);
    uint32_t H_Aligned = DVPP_ALIGN_UP(jpegHeight, VPC_WIDTH_ALIGN);
    uint32_t jpegDecodeBufferSize = W_Aligned * H_Aligned * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2;

    APP_ERROR ret = acldvppMalloc(reinterpret_cast<void **>(&jpegDecodeBuffer), jpegDecodeBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]:  aclrtMalloc failed, ret[" << ret
                 << "], jpegDecodeBufferSize " << jpegDecodeBufferSize << ".";
        return ret;
    }

    DvppDataInfo input;
    input.width = jpegWidth;
    input.height = jpegHeight;
    input.data = inputData;
    input.dataSize = inputDataLen;

    DvppDataInfo output;
    output.data = jpegDecodeBuffer;
    output.dataSize = jpegDecodeBufferSize;

    ret = pDvpp_->JpegDecode(input, output, true);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]:  jpeg decode failed, ret[" << ret << "].";
        acldvppFree(jpegDecodeBuffer);
        jpegDecodeBuffer = nullptr;
        return ret;
    }
    decodedWidth_ = output.width;
    decodedHeight_ = output.height;
    decodedWidthStride_ = output.widthStride;
    decodedHeightStride_ = output.heightStride;
    decodedDataSize_ = output.dataSize;
    return APP_ERR_OK;
}

#else
APP_ERROR ImageDecoder::DeviceVpcHandle(uint8_t *inputdata, uint8_t *&vpcOutBuffer, uint32_t vpcInWidth,
                                        uint32_t vpcInHeight) const
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::begin to resize image.";
    uint32_t vpcOutBufferSize = DvppCommonDevice::GetBufferSize(resizeWidth_, resizeHeight_);
    APP_ERROR ret = acldvppMalloc(reinterpret_cast<void **>(&vpcOutBuffer), vpcOutBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]: aclrtMalloc failed, ret[" << ret << "] vpcOutBufferSize"
                 << vpcOutBufferSize << ".";
        return ret;
    }
    ret = aclrtMemset(reinterpret_cast<void *>(vpcOutBuffer), vpcOutBufferSize, 0, vpcOutBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]: aclrtMemset failed, ret[" << ret << "] vpcOutBufferSize"
                 << vpcOutBufferSize << ".";
        return ret;
    }

    DvppResizeInputMsg input;
    input.outBuf = vpcOutBuffer;
    input.inW = vpcInWidth;
    input.inH = vpcInHeight;
    input.outW = resizeWidth_;
    input.outH = resizeHeight_;
    input.imgBuf = inputdata;
    ret = pDvpp_->VpcResizeWithPadding(input);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]:  VpcResizeWithPadding, ret[" << ret << "].";
        acldvppFree(vpcOutBuffer);
        vpcOutBuffer = nullptr;
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR ImageDecoder::DeviceJpegDecodeHandle(uint8_t *inputData, uint32_t inputDataLen, uint8_t *&jpegDecodeBuffer,
    uint32_t jpegWidth, uint32_t jpegHeight) const
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::begin to decode image data.";
    uint32_t widthAligned = DVPP_ALIGN_UP(jpegWidth, JPEG_WIDTH_ALIGN);
    uint32_t heightAligned = DVPP_ALIGN_UP(jpegHeight, VPC_WIDTH_ALIGN);
    uint32_t jpegDecodeBufferSize = widthAligned * heightAligned * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2;

    APP_ERROR ret = acldvppMalloc(reinterpret_cast<void **>(&jpegDecodeBuffer), jpegDecodeBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]:  aclrtMalloc failed, ret[" << ret
                 << "], jpegDecodeBufferSize " << jpegDecodeBufferSize << ".";
        return ret;
    }

    DvppJpegDecodeInputMsg input;
    input.rawBuf = inputData;
    input.rawBufByteLength = inputDataLen;
    input.decodedBuf = jpegDecodeBuffer;
    input.jpegWidth = jpegWidth;
    input.jpegHeight = jpegHeight;

    ret = pDvpp_->DvppJpegDecode(input);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]:  jpeg decode failed, ret[" << ret << "].";
        acldvppFree(jpegDecodeBuffer);
        jpegDecodeBuffer = nullptr;
        return ret;
    }

    return APP_ERR_OK;
}

#endif
APP_ERROR ImageDecoder::PrepareDecoderInput(std::shared_ptr<StreamRawData> streamData, uint8_t *&devDataPtr)
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::begin to prepare decoder input.";
    APP_ERROR ret = acldvppMalloc(reinterpret_cast<void **>(&devDataPtr),
        streamData->dataSize + DVPP_JPEG_OFFSET); // 8 byte more required by DVPP
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "] Malloc dvpp input buff failed, ret[" << ret << "].";
        return ret;
    }

    if (runMode_ == ACL_HOST) { // under ACL_HOST mode, memory from host to device
        ret = aclrtMemcpy(devDataPtr, streamData->dataSize + DVPP_JPEG_OFFSET, streamData->hostData.get(),
            streamData->dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    } else if (runMode_ == ACL_DEVICE) { // under ACL_DEVICE mode, memory from device to device
        ret = aclrtMemcpy(devDataPtr, streamData->dataSize + DVPP_JPEG_OFFSET, streamData->hostData.get(),
            streamData->dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]：copy host to device failed, ret[" << ret << "].";
        aclrtFree(devDataPtr);
        return ret;
    }
    return ret;
}

APP_ERROR ImageDecoder::PrepareHostResult(uint8_t *jpegDecodeBuffer, uint8_t *&jpegDecodeBufferHost)
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::begin to prepare host result.";

    APP_ERROR ret = aclrtMallocHost(reinterpret_cast<void **>(&jpegDecodeBufferHost),
        decodedWidthStride_ * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2 * decodedHeightStride_);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]: aclrtMallocHost failed, ret[" << ret << "].";
        return ret;
    }
    ret = aclrtMemcpy(jpegDecodeBufferHost,
        decodedWidthStride_ * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2 * decodedHeightStride_, jpegDecodeBuffer,
        decodedWidthStride_ * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2 * decodedHeightStride_,
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]: aclrtMemcpy failed, ret[" << ret << "].";
        return ret;
    }

    return ret;
}

void ImageDecoder::SendData(std::shared_ptr<StreamRawData> streamData, uint8_t *jpegDecodeBuffer, uint8_t *vpcOutBuffer)
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::begin to send result.";
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::make_shared<FrameAiInfo>();
    frameAiInfo->info = streamData->info; // frameinfo from stream puller
    frameAiInfo->imgOrigin.buf.dataSize =
        decodedWidthStride_ * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2 * decodedHeightStride_;
    frameAiInfo->trkFrameID = streamData->picID;
    frameAiInfo->imgOrigin.buf.deviceData.reset(jpegDecodeBuffer,
        [this](uint8_t *p) { AscendDeleter(p, this->aclContext_, &acldvppFree); });

    if (runMode_ == ACL_HOST) { // under ACL_HOST mode, free host memory
        uint8_t *jpegDecodeBufferHost = nullptr;
        APP_ERROR ret = PrepareHostResult(jpegDecodeBuffer, jpegDecodeBufferHost);
        if (ret != APP_ERR_OK) {
            return;
        }
        frameAiInfo->imgOrigin.buf.hostData.reset(jpegDecodeBufferHost,
            [this](uint8_t *p) { AscendDeleter(p, this->aclContext_, &aclrtFreeHost); });
    } else if (runMode_ == ACL_DEVICE) { // under ACL_DEVICE mode, hostData is the same of deviceData.
        frameAiInfo->imgOrigin.buf.hostData = frameAiInfo->imgOrigin.buf.deviceData;
    }
    frameAiInfo->imgOrigin.format = 0; // ffmpeg  decode is yuv 420P
    frameAiInfo->imgOrigin.width = decodedWidthStride_;
    frameAiInfo->imgOrigin.height = decodedHeightStride_;
    frameAiInfo->imgOrigin.widthAligned = decodedWidthStride_;
    frameAiInfo->imgOrigin.heightAligned = decodedHeightStride_;
    frameAiInfo->detectImg.buf.dataSize =
        resizeWidth_ * resizeHeight_ * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2;
    frameAiInfo->detectImg.buf.deviceData.reset(vpcOutBuffer,
        [this](uint8_t *p) { AscendDeleter(p, this->aclContext_, &acldvppFree); });
    frameAiInfo->detectImg.format = 0; // ffmpeg  decode is yuv 420P
    frameAiInfo->detectImg.width = resizeWidth_;
    frameAiInfo->detectImg.height = resizeHeight_;

    outputQueVec_[0]->Push(frameAiInfo, true);
}

APP_ERROR ImageDecoder::Process(std::shared_ptr<void> inputData)
{
    imageDecoderStatic_.RunTimeStatisticStart("ImageDecoder_Excute_Time", instanceId_);
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder:Begin to process.";
    uint8_t *devDataPtr = nullptr;
    std::shared_ptr<StreamRawData> streamData = std::static_pointer_cast<StreamRawData>(inputData);
    if (streamData.get() == nullptr) {
        LogError << "ImageDecoder[" << instanceId_ << "]: invalid input data.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    uint32_t srcWidth = streamData->info.width;
    uint32_t srcHeight = streamData->info.height;
    originalWidth_ = DVPP_ALIGN_UP(streamData->info.width, JPEG_WIDTH_ALIGN);
    streamData->info.width = originalWidth_; // due to the limitation of dvpp, force to use aligned width.
    originalHeight_ = DVPP_ALIGN_UP(streamData->info.height, JPEG_HEIGHT_ALIGN);
    streamData->info.height = originalHeight_; // due to the limitation of dvpp, force to use aligned height.
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::channelId: " << streamData->info.channelId
             << ",frameId: " << streamData->info.frameId << ",image size: aligned width " << originalWidth_
             << ", aligned height " << originalHeight_ << ".";

    APP_ERROR ret = PrepareDecoderInput(streamData, devDataPtr);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    uint8_t *jpegDecodeBuffer = nullptr;
#ifdef ASCEND_FACE_USE_ACL_DVPP
    ret = JpegDecodeHandle(devDataPtr, streamData->dataSize + DVPP_JPEG_OFFSET, jpegDecodeBuffer, srcWidth, srcHeight);
#else
    ret = DeviceJpegDecodeHandle(devDataPtr, streamData->dataSize + DVPP_JPEG_OFFSET, jpegDecodeBuffer, originalWidth_,
        originalHeight_);
#endif
    if (ret != APP_ERR_OK) {
        acldvppFree(devDataPtr);
        return ret;
    }

    uint8_t *vpcOutBuffer = nullptr;
#ifdef ASCEND_FACE_USE_ACL_DVPP
    ret = VpcHandle(jpegDecodeBuffer, vpcOutBuffer);
#else
    ret = DeviceVpcHandle(jpegDecodeBuffer, vpcOutBuffer, originalWidth_, originalHeight_);
#endif
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "] VpcHandle failed, ret[" << ret << "].";
        acldvppFree(devDataPtr);
        acldvppFree(jpegDecodeBuffer);
        return ret;
    }

    SendData(streamData, jpegDecodeBuffer, vpcOutBuffer);
    acldvppFree(devDataPtr);
    imageDecoderStatic_.RunTimeStatisticStop();
    return APP_ERR_OK;
}

double ImageDecoder::GetRunTimeAvg()
{
    return imageDecoderStatic_.GetRunTimeAvg();
}
} // namespace ascendFaceRecognition