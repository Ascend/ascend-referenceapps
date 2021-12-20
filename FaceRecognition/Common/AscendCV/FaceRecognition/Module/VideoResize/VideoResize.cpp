/*
 * @Author: your name
 * @Date: 2020-06-28 12:43:01
 * @LastEditTime: 2020-06-28 12:43:50
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /facerecognition/src/CtrlCPU/VideoDecoder/VideoDecoder.h
 */
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
#include "VideoResize.h"

#include <iostream>

#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "CostStatistic/CostStatistic.h"
#include "ChannelStatus/ChannelStatus.h"

namespace ascendFaceRecognition {
namespace {
const uint32_t VPC_STRIDE_WIDTH = 16; // Vpc module output width need to align up to 16
const uint32_t VPC_STRIDE_HEIGHT = 2; // Vpc module output height need to align up to 2
const uint32_t YUV_BYTES_NU = 3;      // Numerator of yuv image, H x W x 3 / 2
const uint32_t YUV_BYTES_DE = 2;      // Denominator of yuv image, H x W x 3 / 2
const uint32_t MAX_DETECT_MEMORY_BLOCK_NUM = 50;
const uint32_t MAX_ENCODER_MEMORY_BLOCK_NUM = 50;
const float ENCODE_VIDEO_RESIZE_WIDTH_SCALE = 0.2;
const float ENCODE_VIDEO_RESIZE_HEIGHT_SCALE = 0.2;
const double COST_TIME_MS_THRESHOLD = 10.;
}


static DvppDataInfo GetDvppDataInfo(std::shared_ptr<uint8_t> data, const uint32_t &width, const uint32_t &height)
{
    DvppDataInfo output = {};
    output.width = width;
    output.height = height;
    output.widthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH);
    output.heightStride = DVPP_ALIGN_UP(height, VPC_STRIDE_HEIGHT);
    output.data = data.get();
    output.dataSize = output.widthStride * output.heightStride * YUV_BYTES_NU / YUV_BYTES_DE;
    return output;
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

VideoResize::VideoResize() {}
VideoResize::~VideoResize() {}

APP_ERROR VideoResize::ParseConfig(ConfigParser &configParser)
{
    std::string itemCfgStr = {};
    itemCfgStr = std::string("VideoDecoder.resizeWidth");
    int ret = configParser.GetUnsignedIntValue(itemCfgStr, detectWidth_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoResize[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = std::string("VideoDecoder.resizeHeight");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, detectHeight_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoResize[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = std::string("streamWidthMax");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, originWidth_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoResize[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = std::string("streamHeightMax");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, originHeight_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoResize[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    return ret;
}

APP_ERROR VideoResize::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "Begin to init instance " << initArgs.instanceId;

    // init parameter
    AssignInitArgs(initArgs);
    int ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoResize[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret) << ".";
        return ret;
    }
    // init vpc dvpp common
    ret = aclrtCreateStream(&vpcDvppStream_);
    if (ret != APP_ERR_OK) {
        LogError << "VideoResize[" << instanceId_ << "]: aclrtCreateStream failed, ret=" << ret << ".";
        return ret;
    }

    vpcDvppCommon_ = std::make_shared<DvppCommon>(vpcDvppStream_);
    if (vpcDvppCommon_.get() == nullptr) {
        LogError << "create vpcDvppCommon_ Failed";
        return APP_ERR_COMM_ALLOC_MEM;
    }

    ret = vpcDvppCommon_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "vpcDvppCommon_ Init Failed";
        return ret;
    }

    memPool_ = MemoryPool::NewMemoryPoolResource();

    uint32_t blockSize = detectWidth_ * detectHeight_ * YUV_BYTES_NU / YUV_BYTES_DE;
    ret = memPool_->Init(blockSize, MAX_DETECT_MEMORY_BLOCK_NUM);
    if (ret != APP_ERR_OK) {
        LogError << "MemoryPool Init Failed";
        return ret;
    }
    LogWarn << "VideoResize[" << instanceId_ << "]: VideoResize::Init OK.";
    return APP_ERR_OK;
}

APP_ERROR VideoResize::DeInit(void)
{
    LogDebug << "VideoResize [" << instanceId_ << "] begin to deinit";

    if (vpcDvppCommon_.get() != nullptr) {
        APP_ERROR ret = vpcDvppCommon_->DeInit();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to deinitialize vpcDvppCommon, ret = " << ret;
        }
    }

    if (vpcDvppStream_) {
        APP_ERROR ret = aclrtDestroyStream(vpcDvppStream_);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy stream, ret = " << ret;
        }
        vpcDvppStream_ = nullptr;
    }

    memPool_->DeInit();
    LogDebug << "VideoResize [" << instanceId_ << "] deinit success.";
    return APP_ERR_OK;
}

APP_ERROR VideoResize::Resize(DvppDataInfo &input, DvppDataInfo &output)
{
    APP_ERROR ret = vpcDvppCommon_->VpcResize(input, output, true);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to VpcResize ret=" << ret;
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR VideoResize::DetectResize(DvppDataInfo &input, ImageInfo &imageInfo)
{
    auto startTime = CostStatistic::GetStart();
    std::shared_ptr<uint8_t> memblock = memPool_->GetMemoryBlock();
    double costMs = CostStatistic::GetCostTime(startTime);
    LogDebug << "VideoResize[" << instanceId_ << "]: VideoResize: GetMemoryBlock " << costMs << "ms";
    SetImageInfo(imageInfo, memblock, detectWidth_, detectHeight_);
    DvppDataInfo output = GetDvppDataInfo(memblock, detectWidth_, detectHeight_);
    startTime = CostStatistic::GetStart();
    APP_ERROR ret = Resize(input, output);
    costMs = CostStatistic::GetCostTime(startTime);
    LogDebug << "VideoResize[" << instanceId_ << "]: VideoResize: Resize " << costMs << "ms";

    if (ret != APP_ERR_OK) {
        LogFatal << "VideoResize[" << instanceId_ << "]:  DetectResize fail ret=" << ret;
        return ret;
    }
    return APP_ERR_OK;
}


APP_ERROR VideoResize::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "VideoResize[" << instanceId_ << "]::Begin to process data.";
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::static_pointer_cast<FrameAiInfo>(inputData);
    if (frameAiInfo.get() == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }

    auto inputInfo = GetDvppDataInfo(frameAiInfo->imgOrigin.buf.deviceData, frameAiInfo->imgOrigin.width,
        frameAiInfo->imgOrigin.height);

    ImageInfo detectImageInfo = {};
    APP_ERROR ret = DetectResize(inputInfo, detectImageInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to VideoResizeResize";
        return ret;
    }
    frameAiInfo->detectImg = detectImageInfo;
    SendToNextModule(MT_FACE_DETECTION, frameAiInfo, frameAiInfo->info.channelId);
    return APP_ERR_OK;
}
}