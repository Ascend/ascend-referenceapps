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
#include "VideoDecoderHost/VideoDecoder.h"
#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "CostStatistic/CostStatistic.h"

namespace ascendFaceRecognition {
namespace {
const int FRAME_SKIP_INTERVAL = 3;
const uint32_t VPC_STRIDE_WIDTH = 16; // Vpc module output width need to align up to 16
const uint32_t VPC_STRIDE_HEIGHT = 2; // Vpc module output height need to align up to 2
const uint32_t YUV_BYTES_NU = 3;      // Numerator of yuv image, H x W x 3 / 2
const uint32_t YUV_BYTES_DE = 2;      // Denominator of yuv image, H x W x 3 / 2
const int VIDEO_FRMAT_LEN = 10;
const uint32_t ONE_THOUSAND_MILLISECOND = 1000;
const uint32_t MAX_VIDEO_MEMORY_BLOCK_NUM = 40;
const double COST_TIME_MS_THRESHOLD = 0.;
}


static void SetImageInfo(ImageInfo &imageInfo, std::shared_ptr<uint8_t> data, const uint32_t &width,
    const uint32_t &height)
{
    uint32_t widthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH);
    uint32_t heightStride = DVPP_ALIGN_UP(height, VPC_STRIDE_HEIGHT);
    imageInfo.buf.deviceData = data;
    imageInfo.buf.dataSize = widthStride * heightStride * YUV_BYTES_NU / YUV_BYTES_DE;
    imageInfo.width = width;
    imageInfo.height = height;
    imageInfo.widthAligned = widthStride;
    imageInfo.heightAligned = heightStride;
    imageInfo.format = 1;
}

void VideoDecoder::VideoDecoderCallback(acldvppStreamDesc *input, acldvppPicDesc *output, void *userData)
{
    APP_ERROR ret = acldvppGetPicDescRetCode(output);
    std::shared_ptr<acldvppStreamDesc> inputDecs(input, acldvppDestroyStreamDesc);
    std::shared_ptr<acldvppPicDesc> outputDecs(output, acldvppDestroyPicDesc);

    DecodeH26xInfo *decodeH26xInfo = (DecodeH26xInfo *)userData;
    if (decodeH26xInfo == nullptr) {
        LogError << "VideoDecoder: user data is nullptr";
        return;
    }
    auto decodeH26xInfoPtr = std::make_shared<DecodeH26xInfo>();
    decodeH26xInfoPtr.reset(decodeH26xInfo, [](DecodeH26xInfo *p) { delete p; });
    VideoDecoder *decoderPtr = decodeH26xInfo->videoDecoder;
    if (decoderPtr == nullptr) {
        LogError << "VideoDecoder: pointer is nullptr";
        return;
    }

    if (ret != APP_ERR_OK) {
        LogError << "vdec error ret=" << ret;
        return;
    }

    if ((decodeH26xInfoPtr->vdecInfo->frameId % decoderPtr->skipInterval_) != 0) {
        return;
    }

    std::shared_ptr<FrameAiInfo> frameAiInfo = std::make_shared<FrameAiInfo>();
    frameAiInfo->info.channelId = decoderPtr->instanceId_;
    frameAiInfo->info.frameId = decodeH26xInfoPtr->vdecInfo->frameId;
    frameAiInfo->info.mode = FRAME_MODE_SEARCH;
    ImageInfo originImageInfo = {};
    SetImageInfo(originImageInfo, decodeH26xInfo->vdecInfo->imageData, decoderPtr->streamWidthMax_,
        decoderPtr->streamHeightMax_);
    frameAiInfo->imgOrigin = originImageInfo;
    decoderPtr->SendToNextModule(MT_VIDEO_RESIZE, frameAiInfo, frameAiInfo->info.channelId);
    return;
}

void *VideoDecoder::DecoderThread(void *arg)
{
    // Notice: Create context for this thread
    VideoDecoder *videoDecoder = (VideoDecoder *)arg;
    if (videoDecoder == nullptr) {
        LogError << "arg is nullptr";
        return ((void *)(-1));
    }

    aclError ret = aclrtSetCurrentContext(videoDecoder->aclContext_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set context, ret = " << ret;
        return ((void *)(-1));
    }

    LogInfo << "DecoderThread start";
    while (!videoDecoder->stopDecoderThread_) {
        LogDebug << "VideoDecoder[" << videoDecoder->instanceId_ << "] DecoderThread report ";
        (void)aclrtProcessReport(ONE_THOUSAND_MILLISECOND);
    }

    return nullptr;
}

VideoDecoder::VideoDecoder() {}

VideoDecoder::~VideoDecoder() {}

APP_ERROR VideoDecoder::ParseConfig(ConfigParser &configParser)
{
    std::string itemCfgStr = {};
    itemCfgStr = std::string("skipInterval");
    APP_ERROR ret = configParser.GetUnsignedIntValue(itemCfgStr, skipInterval_);
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
    LogDebug << "Begin to init instance " << initArgs.instanceId;

    // init parameter
    AssignInitArgs(initArgs);
    int ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret) <<
            ".";
        return ret;
    }
    // create threadId
    ret = InitVdec();
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: InitVdec failed, ret=" << ret << ".";
        return ret;
    }

    memPoolOrigin_ = MemoryPool::NewMemoryPoolResource();

    streamWidthMax_ = DVPP_ALIGN_UP(streamWidthMax_, VPC_STRIDE_WIDTH);
    streamHeightMax_ = DVPP_ALIGN_UP(streamHeightMax_, VPC_STRIDE_HEIGHT);

    uint32_t blockSize = streamWidthMax_ * streamHeightMax_ * YUV_BYTES_NU / YUV_BYTES_DE;
    ret = memPoolOrigin_->Init(blockSize, MAX_VIDEO_MEMORY_BLOCK_NUM);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    uselessData_ = memPoolOrigin_->GetMemoryBlock();
    LogWarn << "VideoDecoder[" << instanceId_ << "]: VideoDecoder::Init OK.";
    return APP_ERR_OK;
}


APP_ERROR VideoDecoder::InitVdec()
{
    int createThreadErr = pthread_create(&decoderThreadId_, nullptr, &VideoDecoder::DecoderThread, (void *)this);
    if (createThreadErr != 0) {
        LogError << "Failed to create thread, err = " << createThreadErr;
        return APP_ERR_ACL_FAILURE;
    }

    aclvdecChannelDesc *vdecChannelDesc = aclvdecCreateChannelDesc();
    if (vdecChannelDesc == nullptr) {
        LogError << "VideoDecoder[" << instanceId_ << "]: aclvdecCreateChannelDesc fail.";
        return APP_ERR_ACL_FAILURE;
    }
    vdecChannelDesc_.reset(vdecChannelDesc, aclvdecDestroyChannelDesc);

    APP_ERROR ret1 = aclvdecSetChannelDescChannelId(vdecChannelDesc, instanceId_);
    APP_ERROR ret2 = aclvdecSetChannelDescThreadId(vdecChannelDesc, decoderThreadId_);
    APP_ERROR ret3 = aclvdecSetChannelDescCallback(vdecChannelDesc, &VideoDecoder::VideoDecoderCallback);
    APP_ERROR ret4 = aclvdecSetChannelDescEnType(vdecChannelDesc, H264_MAIN_LEVEL);
    APP_ERROR ret5 = aclvdecSetChannelDescOutPicFormat(vdecChannelDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    APP_ERROR ret6 = aclvdecSetChannelDescOutMode(vdecChannelDesc, 0);
    APP_ERROR ret7 = aclvdecSetChannelDescOutPicWidth(vdecChannelDesc, streamWidthMax_);
    APP_ERROR ret8 = aclvdecSetChannelDescOutPicHeight(vdecChannelDesc, streamHeightMax_);
    APP_ERROR ret9 = aclvdecCreateChannel(vdecChannelDesc);
    if (ret1 != APP_ERR_OK || ret2 != APP_ERR_OK || ret3 != APP_ERR_OK ||
        ret4 != APP_ERR_OK || ret5 != APP_ERR_OK || ret6 != APP_ERR_OK ||
        ret7 != APP_ERR_OK || ret8 != APP_ERR_OK || ret9 != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: aclvdecSet fail" <<
            ", ret1 = " << ret1 << ", ret2 = " << ret2 << ", ret3 = " << ret3 <<
            ", ret4 = " << ret4 << ", ret5 = " << ret5 << ", ret6 = " << ret6 <<
            ", ret7 = " << ret7 << ", ret8 = " << ret8 << ", ret9 = " << ret9;
        return APP_ERR_ACL_FAILURE;
    }
    return APP_ERR_OK;
}

APP_ERROR VideoDecoder::DeInitVdec()
{
    APP_ERROR ret = aclvdecDestroyChannel(vdecChannelDesc_.get());
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: aclvdecDestroyChannel fail. ret=" << ret;
        return ret;
    }
    vdecChannelDesc_.reset();
    stopDecoderThread_ = true;
    pthread_join(decoderThreadId_, NULL);
    return APP_ERR_OK;
}

APP_ERROR VideoDecoder::DeInit(void)
{
    LogDebug << "VideoDecoder [" << instanceId_ << "] begin to deinit";

    APP_ERROR ret = DeInitVdec();
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: DeInitVdec fail. ret=" << ret;
    }
    uselessData_.reset();
    memPoolOrigin_->DeInit();
    LogDebug << "VideoDecoder [" << instanceId_ << "] deinit success.";
    return APP_ERR_OK;
}

std::shared_ptr<uint8_t> VideoDecoder::AclrtMallocAndCopy(StreamDataTrans streamData)
{
    auto startTime = CostStatistic::GetStart();
    void *streamBuffer = nullptr;
    APP_ERROR ret = aclrtMalloc(&streamBuffer, streamData.datasize(), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to malloc data with " << streamData.datasize() << " bytes, ret = " << ret << ".";
        return nullptr;
    }
    double costMs = CostStatistic::GetCostTime(startTime);
    LogDebug << "VideoDecoder[" << instanceId_ << "]: stream aclrtMalloc " << costMs << "ms";

    startTime = CostStatistic::GetStart();
    auto streamShared = std::make_shared<uint8_t>();
    streamShared.reset(static_cast<uint8_t *>(streamBuffer), aclrtFree);
    ret = aclrtMemcpy(streamShared.get(), streamData.datasize(), streamData.data().data(), streamData.datasize(),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to copy memory";
        return nullptr;
    }
    costMs = CostStatistic::GetCostTime(startTime);
    LogDebug << "VideoDecoder[" << instanceId_ << "]: stream aclrtMemcpy " << costMs << "ms";

    return streamShared;
}

APP_ERROR VideoDecoder::VideoDecode(std::shared_ptr<VdecInfo> vdecInfo)
{
    acldvppStreamDesc *dvppStreamDesc = acldvppCreateStreamDesc();
    if (dvppStreamDesc == nullptr) {
        LogError << "VideoDecoder[" << instanceId_ << "]: acldvppCreateStreamDesc";
        return APP_ERR_ACL_FAILURE;
    }
    APP_ERROR ret1 = acldvppSetStreamDescData(dvppStreamDesc, vdecInfo->streamData.get());
    APP_ERROR ret2 = acldvppSetStreamDescSize(dvppStreamDesc, vdecInfo->streamSize);
    APP_ERROR ret3 = acldvppSetStreamDescTimestamp(dvppStreamDesc, vdecInfo->frameId);
    if (ret1 != APP_ERR_OK || ret2 != APP_ERR_OK || ret3 != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: acldvppSetStreamDesc fail.";
        acldvppDestroyStreamDesc(dvppStreamDesc);
        return APP_ERR_ACL_FAILURE;
    }

    acldvppPicDesc *dvppPicDesc = acldvppCreatePicDesc();
    if (dvppPicDesc == nullptr) {
        LogError << "VideoDecoder[" << instanceId_ << "]: acldvppCreatePicDesc";
        acldvppDestroyStreamDesc(dvppStreamDesc);
        return APP_ERR_ACL_FAILURE;
    }

    APP_ERROR ret4 = acldvppSetPicDescData(dvppPicDesc, vdecInfo->imageData.get());
    APP_ERROR ret5 = acldvppSetPicDescSize(dvppPicDesc, vdecInfo->imageSize);
    if (ret4 != APP_ERR_OK || ret5 != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: acldvppSetPicDesc";
        acldvppDestroyStreamDesc(dvppStreamDesc);
        acldvppDestroyPicDesc(dvppPicDesc);
        return APP_ERR_ACL_FAILURE;
    }

    DecodeH26xInfo *decoderH26xInfo = new DecodeH26xInfo();
    decoderH26xInfo->videoDecoder = this;
    decoderH26xInfo->vdecInfo = vdecInfo;
    auto startTime = CostStatistic::GetStart();
    APP_ERROR ret6 = aclvdecSendFrame(vdecChannelDesc_.get(), dvppStreamDesc, dvppPicDesc, nullptr,
        static_cast<void *>(decoderH26xInfo));
    if (ret6 != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: aclvdecSendFrame fail. ret=" << ret6;
        delete decoderH26xInfo;
        acldvppDestroyStreamDesc(dvppStreamDesc);
        acldvppDestroyPicDesc(dvppPicDesc);
        return ret6;
    }
    auto costMs = CostStatistic::GetCostTime(startTime);
    LogDebug << "VideoDecoder[" << instanceId_ << "]: aclvdecSendFrame " << costMs << "ms";

    return APP_ERR_OK;
}

APP_ERROR VideoDecoder::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "VideoDecoder[" << instanceId_ << "]: VideoDecoder: process start.";
    std::shared_ptr<DataTrans> dataTrans = std::static_pointer_cast<DataTrans>(inputData);

    auto streamData = dataTrans->streamdata();
    // filter data
    originalWidth_ = streamData.info().width();
    originalHeight_ = streamData.info().height();
    if (originalWidth_ > streamWidthMax_ || originalHeight_ > streamHeightMax_) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: VideoDecoder::channelId: " << streamData.info().channelid() <<
            ",frameId: " << streamData.info().frameid() << ",image size: width " << originalWidth_ << ", height " <<
            originalHeight_ << ". Too Big";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    auto startTime = CostStatistic::GetStart();
    std::shared_ptr<uint8_t> streamShared = AclrtMallocAndCopy(streamData);
    double costMs = CostStatistic::GetCostTime(startTime);
    LogDebug << "VideoDecoder[" << instanceId_ << "]: VideoDecoder: AclrtMallocAndCopy " << costMs << "ms";

    std::shared_ptr<VdecInfo> vdecInfo = std::make_shared<VdecInfo>();
    vdecInfo->streamData = streamShared;
    vdecInfo->streamSize = streamData.datasize();
    startTime = CostStatistic::GetStart();
    if (frameId_ % skipInterval_ != 0) {
        vdecInfo->imageData = uselessData_;
    } else {
        vdecInfo->imageData = memPoolOrigin_->GetMemoryBlock();
    }
    costMs = CostStatistic::GetCostTime(startTime);
    LogDebug << "VideoDecoder[" << instanceId_ << "]: VideoDecoder: GetMemoryBlock " << costMs << "ms";
    vdecInfo->imageSize = streamWidthMax_ * streamHeightMax_ * YUV_BYTES_NU / YUV_BYTES_DE;
    vdecInfo->frameId = frameId_;
    frameId_++;
    startTime = CostStatistic::GetStart();
    APP_ERROR ret = VideoDecode(vdecInfo);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: VideoDecode fail. ret=" << ret;
        return ret;
    }
    costMs = CostStatistic::GetCostTime(startTime);
    LogDebug << "VideoDecoder[" << instanceId_ << "]: VideoDecoder: CombineVdecProcess " << costMs << "ms";
    LogDebug << "VideoDecoder[" << instanceId_ << "]: VideoDecoder: process end.";
    return APP_ERR_OK;
}
} // namespace ascendFaceRecognition
