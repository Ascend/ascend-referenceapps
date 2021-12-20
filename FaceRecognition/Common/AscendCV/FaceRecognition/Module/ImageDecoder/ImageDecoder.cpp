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
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "PointerDeleter/PointerDeleter.h"

namespace ascendFaceRecognition {
namespace {
const uint32_t JPEGD_STRIDE_WIDTH = 128; // Jpegd module output width need to align up to 128
const uint32_t JPEGD_STRIDE_HEIGHT = 16; // Jpegd module output height need to align up to 16
const int VPC_OUT_VAULE = 128;
const uint32_t YUV_BYTES_NU = 3; // Numerator of yuv image, H x W x 3 / 2
const uint32_t YUV_BYTES_DE = 2; // Denominator of yuv image, H x W x 3 / 2
const uint32_t JPEG_OFFSET = 8;  // Offset of input file for jpegd module
}

ImageDecoder::ImageDecoder()
{
    isStop_ = false;
}

ImageDecoder::~ImageDecoder() {}

APP_ERROR ImageDecoder::ParseConfig(ConfigParser &configParser)
{
    // load and parse config file
    int ret;
    std::string itemCfgStr = moduleName_ + std::string(".resizeWidth");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, resizeWidth_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    itemCfgStr = moduleName_ + std::string(".resizeHeight");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, resizeHeight_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    return ret;
}

APP_ERROR ImageDecoder::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: Begin to init image decoder instance " << initArgs.instanceId <<
        initArgs.instanceId << ".";
    // initialize member variables
    AssignInitArgs(initArgs);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret) <<
            ".";
        return ret;
    }

#ifdef ASCEND_ACL_OPEN_VESION
    pDvpp_ = std::make_shared<DvppCommonDevice>();
#else
    ret = aclrtCreateStream(&dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: aclrtCreateStream failed, ret=" << ret << ".";
        return ret;
    }
    pDvpp_ = std::make_shared<DvppCommon>(dvppStream_);
#endif
    // init dvpp
    ret = pDvpp_->Init();
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: dvpp channel create failed!";
        return ret;
    }
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::Init OK.";
    return APP_ERR_OK;
}

APP_ERROR ImageDecoder::DeInit(void)
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::begin to deinit.";
    if (pDvpp_.get() != nullptr) {
        APP_ERROR ret = pDvpp_->DeInit();
        if (ret != APP_ERR_OK) {
            LogFatal << "ImageDecoder[" << instanceId_ << "]: dvpp deinit failed.";
            return ret;
        }
    }
#ifndef ASCEND_ACL_OPEN_VESION
    APP_ERROR ret = aclrtDestroyStream(dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ImageDecoder[" << instanceId_ << "]: Fail to destroy dvppstream.";
        return ret;
    }
    dvppStream_ = nullptr;
#endif
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder deinit success.";
    return APP_ERR_OK;
}
#ifdef ASCEND_ACL_OPEN_VESION
APP_ERROR ImageDecoder::DeviceVpcHandle(std::shared_ptr<uint8_t> inputData, std::shared_ptr<uint8_t> &vpcOutBuffer)
{
    APP_ERROR ret;
    uint32_t vpcOutBufferSize = DvppCommonDevice::GetBufferSize(resizeWidth_, resizeHeight_);
    void *vpcOutData = nullptr;
    ret = acldvppMalloc(&vpcOutData, vpcOutBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]: aclrtMalloc failed, ret[" << ret << "] vpcOutBufferSize" <<
            vpcOutBufferSize << ".";
        return ret;
    }
    vpcOutBuffer.reset((uint8_t *)vpcOutData, acldvppFree);
    // vpc align
    uint32_t wAligned = DVPP_ALIGN_UP(originalWidth_, JPEGD_STRIDE_WIDTH);
    uint32_t hAligned = DVPP_ALIGN_UP(originalHeight_, JPEGD_STRIDE_HEIGHT);
    if (originalWidth_ <= resizeWidth_ && originalHeight_ <= resizeHeight_) {
        std::fill(vpcOutBuffer.get(), vpcOutBuffer.get() + vpcOutBufferSize, (uint8_t)VPC_OUT_VAULE);
        for (uint32_t h = 0; h < originalHeight_; h++) {
            std::copy(inputData.get() + h * wAligned, inputData.get() + h * wAligned + originalWidth_,
                vpcOutBuffer.get() + h * resizeWidth_);

            std::copy(inputData.get() + (h / YUV_BYTES_DE + hAligned) * wAligned,
                inputData.get() + (h / YUV_BYTES_DE + hAligned) * wAligned + originalWidth_,
                vpcOutBuffer.get() + (h / YUV_BYTES_DE + resizeHeight_) * resizeWidth_);
        }
        ret = APP_ERR_OK;
        return ret;
    } else {
        DvppResizeInputMsg input;
        input.outBuf = vpcOutBuffer.get();
        input.inW = originalWidth_;
        input.inH = originalHeight_;
        input.outW = resizeWidth_;
        input.outH = resizeHeight_;
        input.imgBuf = inputData.get();
        ret = pDvpp_->VpcResize(input);
        if (ret != APP_ERR_OK) {
            LogError << "ImageDecoder[" << instanceId_ << "]:  VpcResize, ret[" << ret << "].";
            return ret;
        }
        return ret;
    }
}

APP_ERROR ImageDecoder::DeviceJpegDecodeHandle(std::shared_ptr<uint8_t> inputData, uint32_t inputDataLen,
    std::shared_ptr<uint8_t> &jpegDecodeBuffer, uint32_t jpegWidth, uint32_t jpegHeight)
{
    // image size align
    uint32_t wAligned = DVPP_ALIGN_UP(jpegWidth, JPEGD_STRIDE_WIDTH);
    uint32_t hAligned = DVPP_ALIGN_UP(jpegHeight, JPEGD_STRIDE_HEIGHT);
    uint32_t jpegDecodeBufferSize = wAligned * hAligned * YUV_BYTES_NU / YUV_BYTES_DE;

    void *jpedDecodeData = nullptr;
    int ret = acldvppMalloc(&jpedDecodeData, jpegDecodeBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]:  aclrtMalloc failed, ret[" << ret <<
            "] jpegDecodeBufferSize " << jpegDecodeBufferSize << ".";
        return ret;
    }
    jpegDecodeBuffer.reset((uint8_t *)jpedDecodeData, acldvppFree);
    DvppJpegDecodeInputMsg input;
    input.rawBuf = inputData.get();
    input.rawBufByteLength = inputDataLen;
    input.decodedBuf = jpegDecodeBuffer.get();
    input.jpegWidth = jpegWidth;
    input.jpegHeight = jpegHeight;
    ret = pDvpp_->DvppJpegDecode(input); // jpeg image decode
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]:  jpeg decode failed, ret[" << ret << "].";
        return ret;
    }
    return APP_ERR_OK;
}
#else

APP_ERROR ImageDecoder::VpcResize(std::shared_ptr<uint8_t> inputData, std::shared_ptr<uint8_t> outputData)
{
    uint32_t vpcOutBufferSize = 0;
    DvppCommon::GetVpcDataSize(resizeWidth_, resizeHeight_, PIXEL_FORMAT_YUV_SEMIPLANAR_420, vpcOutBufferSize);
    DvppDataInfo input;
    input.width = originalWidth_;
    input.height = originalHeight_;
    DvppCommon::GetVpcInputStrideSize(input.width, input.height, PIXEL_FORMAT_YUV_SEMIPLANAR_420, input.widthStride,
        input.heightStride);
    input.dataSize = input.widthStride * input.heightStride * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2;
    input.data = inputData.get();
    DvppDataInfo output;
    output.width = resizeWidth_;
    output.height = resizeHeight_;
    DvppCommon::GetVpcOutputStrideSize(output.width, output.height, PIXEL_FORMAT_YUV_SEMIPLANAR_420, output.widthStride,
        output.heightStride);
    output.dataSize = vpcOutBufferSize;
    output.data = outputData.get();
    APP_ERROR ret = pDvpp_->VpcResize(input, output, true, VPC_PT_PADDING);
    if (ret != APP_ERR_OK) {
        // Release the output buffer when resize failed, otherwise release it after use
        LogError << "ImageDecoder[" << instanceId_ << "]: VpcResize failed, ret[" << ret << "]"
                 << ".";
        return ret;
    }
    return APP_ERR_OK;
}


APP_ERROR ImageDecoder::VpcHandle(std::shared_ptr<uint8_t> inputData, std::shared_ptr<uint8_t> &vpcOutBuffer)
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::begin to resize image.";
    uint32_t vpcOutBufferSize = 0;
    DvppCommon::GetVpcDataSize(resizeWidth_, resizeHeight_, PIXEL_FORMAT_YUV_SEMIPLANAR_420, vpcOutBufferSize);
    void *vpcOutData = nullptr;
    APP_ERROR ret = acldvppMalloc(&vpcOutData, vpcOutBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]: acldvppMalloc failed, ret[" << ret << "] vpcOutBufferSize" <<
            vpcOutBufferSize << ".";
        return ret;
    }
    vpcOutBuffer.reset((uint8_t *)vpcOutData, acldvppFree);
    ret = aclrtMemset(vpcOutData, vpcOutBufferSize, 0, vpcOutBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]: aclrtMemset failed, ret[" << ret << "] vpcOutBufferSize" <<
            vpcOutBufferSize << ".";
        return ret;
    }
    if (originalWidth_ <= resizeWidth_ && originalHeight_ <= resizeHeight_) {
        uint32_t wAligned = DVPP_ALIGN_UP(originalWidth_, JPEGD_STRIDE_WIDTH);
        uint32_t hAligned = DVPP_ALIGN_UP(originalHeight_, JPEGD_STRIDE_HEIGHT);
        for (uint32_t h = 0; h < originalHeight_; h++) {
            ret = aclrtMemcpy(vpcOutBuffer.get() + h * resizeWidth_, originalWidth_, inputData.get() + h * wAligned,
                originalWidth_, ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (ret != APP_ERR_OK) {
                LogError << "ImageDecoder[" << instanceId_ << "]: aclrtMemcpy failed, ret[" << ret << "]";
                return ret;
            }
            ret = aclrtMemcpy(vpcOutBuffer.get() + (h / YUV_BYTES_DE + resizeHeight_) * resizeWidth_, originalWidth_,
                inputData.get() + (h / YUV_BYTES_DE + hAligned) * wAligned, originalWidth_,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (ret != APP_ERR_OK) {
                LogError << "ImageDecoder[" << instanceId_ << "]: aclrtMemcpy failed, ret[" << ret << "]";
                return ret;
            }
        }
        ret = APP_ERR_OK;
        return ret;
    } else {
        ret = VpcResize(inputData, vpcOutBuffer);
        if (ret != APP_ERR_OK) {
            LogError << "ImageDecoder[" << instanceId_ << "]: VpcResize failed.ret=" << ret;
            return ret;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR ImageDecoder::JpegDecodeHandle(std::shared_ptr<uint8_t> inputData, uint32_t inputDataLen,
    std::shared_ptr<uint8_t> &jpegDecodeBuffer, uint32_t jpegWidth, uint32_t jpegHeight)
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder::begin to decode image data.";
    uint32_t wAligned = DVPP_ALIGN_UP(jpegWidth, JPEG_WIDTH_ALIGN);
    uint32_t hAligned = DVPP_ALIGN_UP(jpegHeight, JPEG_HEIGHT_ALIGN);
    uint32_t jpegDecodeBufferSize = wAligned * hAligned * YUV_BYTES_NU / YUV_BYTES_DE;
    void *jpegDecodeData = nullptr;
    APP_ERROR ret = acldvppMalloc(&jpegDecodeData, jpegDecodeBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]:  acldvppMalloc failed, ret[" << ret <<
            "], jpegDecodeBufferSize " << jpegDecodeBufferSize << ".";
        return ret;
    }
    jpegDecodeBuffer.reset((uint8_t *)jpegDecodeData, acldvppFree);
    DvppDataInfo input;
    input.width = jpegWidth;
    input.height = jpegHeight;
    input.data = inputData.get();
    input.dataSize = inputDataLen;
    DvppDataInfo output;
    output.width = jpegWidth;
    output.height = jpegHeight;
    DvppCommon::GetJpegDecodeStrideSize(output.width, output.height, output.widthStride, output.heightStride);
    output.data = jpegDecodeBuffer.get();
    output.dataSize = jpegDecodeBufferSize;
    ret = pDvpp_->JpegDecode(input, output, true);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]:  jpeg decode failed, ret[" << ret << "].";
        return ret;
    }
    return APP_ERR_OK;
}
#endif

APP_ERROR ImageDecoder::DecodeRegistFace(std::shared_ptr<RegistFaceInfoDataTrans> registFace,
    std::shared_ptr<FrameAiInfo> frameAiInfo)
{
    void *devDataPtr = nullptr;
    APP_ERROR ret = acldvppMalloc(&devDataPtr, registFace->avatar().datasize() + JPEG_OFFSET);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    std::shared_ptr<uint8_t> jpegEncodeBuffer((uint8_t *)devDataPtr, acldvppFree);
#ifdef ASCEND_ACL_OPEN_VESION
    std::copy(registFace->avatar().data().c_str(),
        registFace->avatar().data().c_str() + registFace->avatar().datasize(), (uint8_t *)devDataPtr);
#else
    ret = aclrtMemcpy(devDataPtr, registFace->avatar().datasize(), registFace->avatar().data().c_str(),
        registFace->avatar().datasize(), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]:  aclrtMemcpy, ret[" << ret << "].";
        return ret;
    }
#endif
    std::shared_ptr<uint8_t> jpegDecodeBuffer = nullptr;
#ifdef ASCEND_ACL_OPEN_VESION
    ret = DeviceJpegDecodeHandle(jpegEncodeBuffer, registFace->avatar().datasize() + JPEG_OFFSET, jpegDecodeBuffer,
        originalWidth_, originalHeight_);
#else
    ret = JpegDecodeHandle(jpegEncodeBuffer, registFace->avatar().datasize() + JPEG_OFFSET, jpegDecodeBuffer,
        originalWidth_, originalHeight_);
#endif
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]:  jpeg decode failed, ret[" << ret << "].";
        return ret;
    }
    std::shared_ptr<uint8_t> vpcOutBuffer = nullptr;
#ifdef ASCEND_ACL_OPEN_VESION
    ret = DeviceVpcHandle(jpegDecodeBuffer, vpcOutBuffer);
#else
    ret = VpcHandle(jpegDecodeBuffer, vpcOutBuffer);
#endif
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "]: VpcResize failed, ret[" << ret << "]"
                 << ".";
        return ret;
    }
    uint32_t widthAligned = DVPP_ALIGN_UP(originalWidth_, JPEGD_STRIDE_WIDTH);
    uint32_t heightAligned = DVPP_ALIGN_UP(originalHeight_, JPEGD_STRIDE_HEIGHT);
    frameAiInfo->imgOrigin.buf.dataSize = heightAligned * widthAligned * YUV_BYTES_NU / YUV_BYTES_DE;
    frameAiInfo->imgOrigin.buf.deviceData = jpegDecodeBuffer;
    frameAiInfo->detectImg.buf.dataSize = resizeWidth_ * resizeHeight_ * YUV_BYTES_NU / YUV_BYTES_DE;
    frameAiInfo->detectImg.buf.deviceData = vpcOutBuffer;

    return APP_ERR_OK;
}

APP_ERROR ImageDecoder::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "ImageDecoder[" << instanceId_ << "]: ImageDecoder:Begin to process.";
    std::shared_ptr<DataTrans> dataTrans = std::static_pointer_cast<DataTrans>(inputData);
    LogDebug << "dataTrans";
    std::shared_ptr<RegistFaceInfoDataTrans> registFace =
        std::make_shared<RegistFaceInfoDataTrans>(dataTrans->registface());
    if (registFace.get() == nullptr) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    LogDebug << "RegistFaceInfoDataTrans";
    originalWidth_ = registFace->avatar().width();
    originalHeight_ = registFace->avatar().height();
    uint32_t widthAligned = DVPP_ALIGN_UP(originalWidth_, JPEGD_STRIDE_WIDTH);
    uint32_t heightAligned = DVPP_ALIGN_UP(originalHeight_, JPEGD_STRIDE_HEIGHT);
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::make_shared<FrameAiInfo>();
    FrameInfo frameInfo = {};
    frameInfo.mode = FRAME_MODE_REG;
    frameInfo.frameId = numCount_++;
    frameInfo.channelId = instanceId_;
    PersonInfo personInfo = {};
    personInfo.uuid = registFace->uuid();
    frameInfo.personInfo = personInfo;
    frameAiInfo->info = frameInfo;
    LogDebug << "DecodeRegistFace begin";
    APP_ERROR ret = DecodeRegistFace(registFace, frameAiInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ImageDecoder[" << instanceId_ << "] DecodeRegistFace failed, ret[" << ret << "].";
        return ret;
    }
    LogDebug << "DecodeRegistFace end";
    frameAiInfo->imgOrigin.format = 0; // ffmpeg  decode is yuv 420P
    frameAiInfo->imgOrigin.width = originalWidth_;
    frameAiInfo->imgOrigin.height = originalHeight_;
    frameAiInfo->imgOrigin.widthAligned = widthAligned;
    frameAiInfo->imgOrigin.heightAligned = heightAligned;
    frameAiInfo->detectImg.format = 0; // ffmpeg  decode is yuv 420P
    frameAiInfo->detectImg.width = resizeWidth_;
    frameAiInfo->detectImg.height = resizeHeight_;
    frameAiInfo->detectImg.widthAligned = resizeWidth_;
    frameAiInfo->detectImg.heightAligned = resizeHeight_;
    if (originalWidth_ < resizeWidth_ && originalHeight_ < resizeHeight_) {
        frameAiInfo->imgOrigin = frameAiInfo->detectImg;
    }
    SendToNextModule(MT_FACE_DETECTION, frameAiInfo, 0);
    return APP_ERR_OK;
}
} // namespace ascendFaceRecognition
