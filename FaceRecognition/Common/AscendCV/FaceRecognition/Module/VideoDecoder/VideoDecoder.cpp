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

#include <iostream>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "ErrorCode/ErrorCode.h"
#include "VideoDecoder/VideoDecoder.h"
#include "Log/Log.h"
#include "FileEx/FileEx.h"

namespace ascendFaceRecognition {
namespace {
const int FRAME_SKIP_INTERVAL = 3;
const uint32_t VPC_STRIDE_WIDTH = 16; // Vpc module output width need to align up to 16
const uint32_t VPC_STRIDE_HEIGHT = 2; // Vpc module output height need to align up to 2
const uint32_t YUV_BYTES_NU = 3;      // Numerator of yuv image, H x W x 3 / 2
const uint32_t YUV_BYTES_DE = 2;      // Denominator of yuv image, H x W x 3 / 2
const int VIDEO_FRMAT_LEN = 10;
const float SEC2MS = 1000.0;
const uint32_t CROP_IMAGE_SIZE = 2;
}
VideoDecoder::VideoDecoder()
{
    isStop_ = false;
}

VideoDecoder::~VideoDecoder() {}

void VideoDecoder::VdecResultCallback(FRAME *frame, void *hiai_data)
{
    LogDebug << "Enter Video Decode Callback";
    // judge parameter
    if (hiai_data == NULL) {
        LogFatal << "input parameter of VdecResultCallback error!";
        return;
    }
    auto decH26xInfo = (DecodeH26xInfo *)hiai_data;
    VideoDecoder *decoderPtr = decH26xInfo->videoDecoder;
    decoderPtr->frameId++;

    APP_ERROR ret = aclrtSetCurrentContext(decoderPtr->aclContext_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VdecResultCallback set contect failed!";
        return;
    }

    if ((decoderPtr->frameId % decoderPtr->skipInterval_) != 0) {
        return;
    }

    FrameInfo decodeH26xInfo = decoderPtr->frameInfo;
    decodeH26xInfo.frameId = decH26xInfo->hostFrameId;
    decodeH26xInfo.mode = FRAME_MODE_SEARCH;
    // convert hbfc to yuv420
    struct timeval startTime, endTime;
    gettimeofday(&startTime, nullptr);
    std::vector<ImageInfo> cropImageList;
    // image resize
    if (APP_ERR_OK != decoderPtr->VdecImageResize(*frame, cropImageList)) {
        LogFatal << "VdecImageResize of VdecResultCallback error!";
        return;
    }

    // construct data transmission struct
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::make_shared<FrameAiInfo>();
    frameAiInfo->info = decodeH26xInfo; // frameinfo from stream puller
    frameAiInfo->imgOrigin = cropImageList[0];
    frameAiInfo->detectImg = cropImageList[1];
    frameAiInfo->embeddingCount = 0;

    decoderPtr->SendToNextModule(MT_FACE_DETECTION, frameAiInfo, frameAiInfo->info.channelId);

    gettimeofday(&endTime, nullptr);
    double costMs = (endTime.tv_sec - startTime.tv_sec) * SEC2MS + (endTime.tv_usec - startTime.tv_usec) / SEC2MS;
    LogDebug << "[Statistic] [Module] [" << decoderPtr->moduleName_ << "] [" << decoderPtr->instanceId_ <<
        "] [Resize] [" << costMs << " ms]";

    return;
}


void VideoDecoder::VdecErrorCallback(VDECERR *vdecErr)
{
    LogWarn << "vdec failed!";
}

APP_ERROR VideoDecoder::ParseConfig(ConfigParser &configParser)
{
    std::string itemCfgStr = moduleName_ + std::string(".resizeWidth");
    int ret = configParser.GetUnsignedIntValue(itemCfgStr, resizeWidth_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".resizeHeight");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, resizeHeight_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = std::string("skipInterval");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, skipInterval_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = std::string("streamWidthMax");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, streamWidthMax_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = std::string("streamHeightMax");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, streamHeightMax_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    return ret;
}

APP_ERROR VideoDecoder::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "VideoDecoder[" << instanceId_ << "]: Begin to init image decoder instance " << initArgs.instanceId <<
        initArgs.instanceId << ".";
    // init parameter
    AssignInitArgs(initArgs);

    int ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret) <<
            ".";
        return ret;
    }

    ret = aclrtCreateStream(&dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: aclrtCreateStream failed, ret=" << ret << ".";
        return ret;
    }

    /* create the vdec object */
    if (piDvppApiVdec == NULL) {
        ret = CreateVdecApi(piDvppApiVdec, 0);
        if ((ret != APP_ERR_OK) && (piDvppApiVdec == NULL)) {
            LogFatal << "VideoDecoder[" << instanceId_ << "] fail to intialize dvpp api!";
            return APP_ERR_DVPP_H26X_DECODE_FAIL;
        }
    }

    /* create the vpc object */
    if (piDvppApiVpc == NULL) {
        ret = CreateDvppApi(piDvppApiVpc);
        if ((ret != APP_ERR_OK) && (piDvppApiVpc == NULL)) {
            LogFatal << "VideoDecoder[" << instanceId_ << "] fail to intialize vpc api!";
            return APP_ERR_DVPP_VPC_FAIL;
        }
    }
    // resize align
    resizeOutWidth_ = ALIGN_UP(resizeWidth_, VPC_STRIDE_WIDTH);
    resizeOutHeight_ = ALIGN_UP(resizeHeight_, VPC_STRIDE_HEIGHT);
    resizeOutputSize_ = resizeOutWidth_ * resizeOutHeight_ * YUV_BYTES_NU / YUV_BYTES_DE;
    ret = acldvppMalloc(&resizeOutputBuffer_, resizeOutputSize_);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "] acldvppMalloc failed, ret[" << ret << "].";
        aclrtFree(resizeOutputBuffer_);
        return ret;
    }
    // stream align
    int widthAligned = ALIGN_UP(streamWidthMax_, VPC_STRIDE_HEIGHT);
    int heightAligned = ALIGN_UP(streamHeightMax_, VPC_STRIDE_HEIGHT);
    ret = acldvppMalloc(&originalOutputBuffer_, widthAligned * heightAligned * YUV_BYTES_NU / YUV_BYTES_DE);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "] acldvppMalloc failed, ret[" << ret << "].";
        aclrtFree(originalOutputBuffer_);
        return ret;
    }

    LogDebug << "VideoDecoder[" << instanceId_ << "]: VideoDecoder::Init OK.";
    return APP_ERR_OK;
}

APP_ERROR VideoDecoder::DeInit(void)
{
    LogDebug << "VideoDecoder[" << instanceId_ << "]: VideoDecoder::begin to deinit.";
    /* Destroy vdec object */
    if (piDvppApiVdec != NULL) {
        LogDebug << "VideoDecoder[" << instanceId_ << "] Destroy vdec api!";
        DestroyVdecApi(piDvppApiVdec, 0);
        piDvppApiVdec = NULL;
    }

    /* Destroy vpc object */
    if (piDvppApiVpc != NULL) {
        LogDebug << "VideoDecoder[" << instanceId_ << "] Destroy vpc api!";
        DestroyDvppApi(piDvppApiVpc);
        piDvppApiVpc = NULL;
    }

    int ret = aclrtDestroyStream(dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: Fail to destroy dvppstream.";
        return ret;
    }
    dvppStream_ = nullptr;
    acldvppFree(originalOutputBuffer_);
    acldvppFree(resizeOutputBuffer_);

    LogDebug << "VideoDecoder[" << instanceId_ << "]: VideoDecoder deinit success.";
    return APP_ERR_OK;
}

APP_ERROR VideoDecoder::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "VideoDecoder[" << instanceId_ << "]: VideoDecoder:Begin to process.";
    std::shared_ptr<DataTrans> dataTrans = std::static_pointer_cast<DataTrans>(inputData);

    std::shared_ptr<StreamDataTrans> streamData = std::make_shared<StreamDataTrans>(dataTrans->streamdata());
    if (streamData.get() == nullptr) {
        LogError << "VideoDecoder[" << instanceId_ << "]: invalid input data.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    // filter data
    originalWidth_ = streamData->info().width();
    originalHeight_ = streamData->info().height();
    if (originalWidth_ > streamWidthMax_ || originalHeight_ > streamHeightMax_) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: VideoDecoder::channelId: " <<
            streamData->info().channelid() << ",frameId: " << streamData->info().frameid() << ",image size: width " <<
            originalWidth_ << ", height " << originalHeight_ << ". Too Big";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    struct timeval startTime, endTime;
    gettimeofday(&startTime, nullptr);
    // decode video
    if (APP_ERR_OK != DecodeH26xVideo(streamData)) {
        LogError << "VideoDecoder[" << instanceId_ << "]: DecodeH26xVideo() error!";
        return APP_ERR_OK;
    }

    gettimeofday(&endTime, nullptr);
    double costMs = (endTime.tv_sec - startTime.tv_sec) * SEC2MS + (endTime.tv_usec - startTime.tv_usec) / SEC2MS;
    LogDebug << "[Statistic] [Module] [" << moduleName_ << "] [" << instanceId_ << "] [DecodeH26xVideo] [" << costMs <<
        " ms]";

    return APP_ERR_OK;
}

APP_ERROR VideoDecoder::DecodeH26xVideo(std::shared_ptr<StreamDataTrans> streamRawData)
{
    LogDebug << "VideoDecoder[" << instanceId_ << "] VDEC begin decode frame" << streamRawData->info().frameid();
    /* define parameters for DVPP Vdec */
    vdec_in_msg vdecMsg;

    /* decode the H26x video */
    // set input fromat: h264, h265 warning, please define enum
    if (streamRawData->info().format() == 2) {
        strncpy_s(vdecMsg.video_format, VIDEO_FRMAT_LEN, "h264", sizeof("h264"));
    } else {
        strncpy_s(vdecMsg.video_format, VIDEO_FRMAT_LEN, "h265", sizeof("h264"));
    }

    // set output format: yuv420sp_UV(default)=nv12, yuv420sp_VU=nv21
    strncpy_s(vdecMsg.image_format, VIDEO_FRMAT_LEN, "nv12", sizeof("h264"));

    vdecMsg.in_buffer = static_cast<char *>((char *)streamRawData->data().data());
    vdecMsg.in_buffer_size = streamRawData->datasize();

    vdecMsg.channelId = streamRawData->info().channelid();
    vdecMsg.isEOS = false;
    this->frameInfo.frameId = streamRawData->info().frameid();
    this->frameInfo.channelId = streamRawData->info().channelid();
    this->frameInfo.height = streamRawData->info().height();
    this->frameInfo.width = streamRawData->info().width();
    this->frameInfo.format = streamRawData->info().format();

    std::shared_ptr<DecodeH26xInfo> decH26xInfo = std::make_shared<DecodeH26xInfo>();
    decH26xInfo->setFrameIndex(streamRawData->info().frameid());
    decH26xInfo->hostFrameId = streamRawData->info().frameid();
    decH26xInfo->videoDecoder = this;
    vdecMsg.hiai_data_sp = decH26xInfo; // if use hiai_data_sp, set hiai_data as NULL
    vdecMsg.hiai_data = NULL;           // if use hiai_data_sp, set hiai_data as NULL

    // callback is performed after a task is processed to reduce the memory usage
    vdecMsg.call_back = VideoDecoder::VdecResultCallback;
    vdecMsg.err_report = VideoDecoder::VdecErrorCallback;

    dvppapi_ctl_msg dvppapiCtlMsg;
    dvppapiCtlMsg.in_size = sizeof(vdec_in_msg);
    dvppapiCtlMsg.in = static_cast<void *>(&vdecMsg);

    if (VdecCtl(piDvppApiVdec, DVPP_CTL_VDEC_PROC, &dvppapiCtlMsg, 0) != 0) {
        LogError << "VideoDecoder[" << instanceId_ << "]: VdecCtl of DecodeH26xVideo() error!";
        return APP_ERR_DVPP_H26X_DECODE_FAIL;
    }

    /* if the video stream end, set isEos, and call VdecCtl() to output the frame in the cacahe of Vdec */
    if (streamRawData->info().iseos() == 1) {
        vdecMsg.isEOS = true;
        if (VdecCtl(piDvppApiVdec, DVPP_CTL_VDEC_PROC, &dvppapiCtlMsg, 0) != 0) {
            LogError << "VideoDecoder[" << instanceId_ << "]: VdecCtl of DecodeH26xVideo() error!";
            return APP_ERR_DVPP_H26X_DECODE_FAIL;
        }

        LogError << "VideoDecoder[" << instanceId_ << "] Clear buffer!";
        /* construct data transmission struct */
        std::shared_ptr<FrameAiInfo> frameAiInfo = std::make_shared<FrameAiInfo>();
    }

    LogDebug << "VideoDecoder[" << instanceId_ << "] VDEC decode end";

    return APP_ERR_OK;
}

std::shared_ptr<VpcUserImageConfigure> VideoDecoder::GetHfbcInputConfigure(FRAME &frame)
{
    std::shared_ptr<VpcUserImageConfigure> imageConfigure = std::make_shared<VpcUserImageConfigure>();
    imageConfigure->bareDataAddr = nullptr;
    imageConfigure->bareDataBufferSize = 0;
    // set input frame format
    imageConfigure->isCompressData = true;
    imageConfigure->widthStride = frame.width;
    imageConfigure->heightStride = frame.height;
    std::string imageFormat(frame.image_format);
    if (imageFormat == "nv12") {
        imageConfigure->inputFormat = INPUT_YUV420_SEMI_PLANNER_UV;
    } else {
        imageConfigure->inputFormat = INPUT_YUV420_SEMI_PLANNER_VU;
    }

    // set hfbc input address
    VpcCompressDataConfigure *compressDataConfigure = &imageConfigure->compressDataConfigure;
    uintptr_t baseAddr = (uintptr_t)frame.buffer;
    compressDataConfigure->lumaHeadAddr = baseAddr + frame.offset_head_y;
    compressDataConfigure->chromaHeadAddr = baseAddr + frame.offset_head_c;
    compressDataConfigure->lumaPayloadAddr = baseAddr + frame.offset_payload_y;
    compressDataConfigure->chromaPayloadAddr = baseAddr + frame.offset_payload_c;
    compressDataConfigure->lumaHeadStride = frame.stride_head;
    compressDataConfigure->chromaHeadStride = frame.stride_head;
    compressDataConfigure->lumaPayloadStride = frame.stride_payload;
    compressDataConfigure->chromaPayloadStride = frame.stride_payload;

    imageConfigure->outputFormat = OUTPUT_YUV420SP_UV;
    imageConfigure->yuvSumEnable = false;
    imageConfigure->cmdListBufferAddr = nullptr;
    imageConfigure->cmdListBufferSize = 0;

    return imageConfigure;
}

void VideoDecoder::HandleResizeOutput(std::shared_ptr<VpcUserImageConfigure> imageConfigure,
    std::vector<ImageInfo> &cropImageList)
{
    /* Get output image */
    int index = 0;
    while (imageConfigure->roiConfigure != nullptr) {
        ImageInfo img;
        img.format = OUTPUT_YUV420SP_UV;
        img.width = imageConfigure->roiConfigure->outputConfigure.outputArea.rightOffset;
        img.height = imageConfigure->roiConfigure->outputConfigure.outputArea.downOffset;
        img.widthAligned = imageConfigure->roiConfigure->outputConfigure.widthStride;
        img.heightAligned = imageConfigure->roiConfigure->outputConfigure.heightStride;
        img.buf.dataSize = imageConfigure->roiConfigure->outputConfigure.bufferSize;
#ifndef DVPP_MALLOC_ORIGIN_IMAGE
        auto outBuf = std::shared_ptr<uint8_t>();
        outBuf.reset(new uint8_t[img.buf.dataSize], std::default_delete<uint8_t[]>());
        if (outBuf == nullptr) {
            LogError << "VideoDecoder[" << instanceId_ << "] malloc error!";
            return;
        }
        img.buf.deviceData = outBuf;
        std::copy((uint8_t *)imageConfigure->roiConfigure->outputConfigure.addr,
            (uint8_t *)imageConfigure->roiConfigure->outputConfigure.addr + img.buf.dataSize, img.buf.deviceData.get());
#endif
        /* update */
        cropImageList.push_back(img);
        imageConfigure->roiConfigure = imageConfigure->roiConfigure->next;
        index++;
    }
}
void VideoDecoder::SetVpcUserRoiConfigure(const FRAME &frame, const FrameOutputSize &outSize, uint8_t *address,
    std::shared_ptr<VpcUserRoiConfigure> roiConfig)
{
    /* ouput 0: raw image */
    VpcUserRoiInputConfigure *inputConfigure = &roiConfig->inputConfigure;
    // set map area: the entire picture
    inputConfigure->cropArea.leftOffset = 0;                       // even
    inputConfigure->cropArea.rightOffset = CHECK_ODD(frame.width); // odd
    inputConfigure->cropArea.upOffset = 0;                         // even
    inputConfigure->cropArea.downOffset = CHECK_ODD(frame.height); // odd

    // Construct output buffer
    VpcUserRoiOutputConfigure *outputConfigure = &roiConfig->outputConfigure;
    outputConfigure->widthStride = ALIGN_UP(outSize.width, VPC_STRIDE_WIDTH);    // align to 128
    outputConfigure->heightStride = ALIGN_UP(outSize.height, VPC_STRIDE_HEIGHT); // align to 16
    outputConfigure->bufferSize =
        outputConfigure->widthStride * outputConfigure->heightStride * YUV_BYTES_NU / YUV_BYTES_DE; // yuv420sp_UV
    // Construct output buffer
    outputConfigure->addr = address; // align to 128

    // set crop area:
    outputConfigure->outputArea.leftOffset = 0; // algin to 16
    outputConfigure->outputArea.rightOffset = CHECK_ODD(outSize.width);
    outputConfigure->outputArea.upOffset = 0;
    outputConfigure->outputArea.downOffset = CHECK_ODD(outSize.height);
}

APP_ERROR VideoDecoder::GetDvppMallocSharedPtr(const uint32_t &size, std::shared_ptr<uint8_t> &buffer)
{
    void *dvppBuffer = nullptr;
    APP_ERROR ret = acldvppMalloc(&dvppBuffer, size);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "] acldvppMalloc failed, ret[" << ret << "].";
        aclrtFree(dvppBuffer);
        return ret;
    }
    buffer = std::shared_ptr<uint8_t>((uint8_t *)dvppBuffer, [](uint8_t *p) { acldvppFree((void *)p); });
    return APP_ERR_OK;
}

APP_ERROR VideoDecoder::VdecImageResize(FRAME &frame, std::vector<ImageInfo> &cropImageList)
{
    /*
     * vpc, change format and resize
     * Construct VPC input parameter
     */
    std::shared_ptr<VpcUserImageConfigure> imageConfigure = GetHfbcInputConfigure(frame);

    /* ouput 0: raw image, convert format from INPUT_YUV420_SEMI_PLANNER_VU to OUTPUT_YUV420SP_UV */
    /* ouput 1: resized image for detect network input, convert format from INPUT_YUV420_SEMI_PLANNER_VU to
     * OUTPUT_YUV420SP_UV */
    std::shared_ptr<VpcUserRoiConfigure> roiConfigureOutput0 = std::make_shared<VpcUserRoiConfigure>();
    std::shared_ptr<VpcUserRoiConfigure> roiConfigureOutput1 = std::make_shared<VpcUserRoiConfigure>();
    imageConfigure->roiConfigure = roiConfigureOutput0.get();
    roiConfigureOutput0->next = roiConfigureOutput1.get();
    roiConfigureOutput1->next = nullptr;

    FrameOutputSize originalSize = { (uint32_t)frame.width, (uint32_t)frame.height };
    FrameOutputSize resizeSize = { resizeWidth_, resizeHeight_ };
    uint8_t *originalAddress = (uint8_t *)originalOutputBuffer_;
    uint8_t *resizeAddress = (uint8_t *)resizeOutputBuffer_;

#ifdef DVPP_MALLOC_ORIGIN_IMAGE
    uint32_t outWidthAlign = ALIGN_UP(frame.width, VPC_STRIDE_WIDTH);    // align to 128
    uint32_t outHeightAlign = ALIGN_UP(frame.height, VPC_STRIDE_HEIGHT); // align to 16
    uint32_t originBufferSize = outWidthAlign * outHeightAlign * YUV_BYTES_NU / YUV_BYTES_DE;
    uint32_t resizeBufferSize = resizeOutWidth_ * resizeOutHeight_ * YUV_BYTES_NU / YUV_BYTES_DE;
    std::shared_ptr<uint8_t> originalData;
    std::shared_ptr<uint8_t> resizeData;
    APP_ERROR ret = GetDvppMallocSharedPtr(originBufferSize, originalData);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "] GetDvppMallocSharedPtr failed, ret[" << ret << "].";
        return ret;
    }
    originalAddress = originalData.get();
    ret = GetDvppMallocSharedPtr(resizeBufferSize, resizeData);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "] GetDvppMallocSharedPtr failed, ret[" << ret << "].";
        return ret;
    }
    resizeAddress = resizeData.get();
#endif
    SetVpcUserRoiConfigure(frame, originalSize, originalAddress, roiConfigureOutput0);
    SetVpcUserRoiConfigure(frame, resizeSize, resizeAddress, roiConfigureOutput1);

    /* process of VPC */
    dvppapi_ctl_msg dvppApiCtlMsg;
    dvppApiCtlMsg.in = static_cast<void *>(imageConfigure.get());
    dvppApiCtlMsg.in_size = sizeof(VpcUserImageConfigure);

    if (DvppCtl(piDvppApiVpc, DVPP_CTL_VPC_PROC, &dvppApiCtlMsg) != 0) {
        LogError << "VideoDecoder[" << instanceId_ << "]：VdecImageResize() vpc of VdecImageResize() error!";
        return APP_ERR_DVPP_VPC_FAIL;
    }

    HandleResizeOutput(imageConfigure, cropImageList);
#ifdef DVPP_MALLOC_ORIGIN_IMAGE
    if (cropImageList.size() == CROP_IMAGE_SIZE) {
        cropImageList[0].buf.deviceData = originalData;
        cropImageList[1].buf.deviceData = resizeData;
    }
#endif
    return APP_ERR_OK;
}
} // namespace ascendFaceRecognition
