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

#include "FaceResize.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <cstring>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "FrameCache/FrameCache.h"
#include "CostStatistic/CostStatistic.h"

namespace ascendFaceRecognition {
namespace {
const uint32_t YUV_BYTES_NU = 3;
const uint32_t YUV_BYTES_DE = 2;
const uint32_t DVPP_WIDTH_ALIGN = 16;
const uint32_t DVPP_HEIGHT_ALIGN = 2;

const uint32_t WIDTH_INDEX = 0;
const uint32_t HEIGHT_INDEX = 1;
const uint32_t LEFT_EXPAND_INDEX = 0;
const uint32_t RIGHT_EXPAND_INDEX = 1;
const uint32_t UP_EXPAND_INDEX = 2;
const uint32_t DOWN_EXPAND_INDEX = 3;
const uint32_t EMBEDDING_INDEX = 0;
const uint32_t QUALITY_INDEX = 1;
const uint32_t CROP_INDEX = 2;
const uint32_t MAX_ROINUMS = 128;

const uint32_t VPC_EVEN_ALIGN = 2;
const uint32_t INPUT_IMAGE_NUM = 1;
const double COST_TIME_MS_THRESHOLD = 10.;
}

FaceResize::FaceResize()
{
    isStop_ = true;
    instanceId_ = -1;
}

FaceResize::~FaceResize() {}


#ifdef ASCEND_ACL_OPEN_VESION
APP_ERROR FaceResize::InitDvpp(void)
{
    LogDebug << "FaceResize[" << instanceId_ << "]: begin to init dvpp.";
    pDvpp_ = std::make_shared<DvppCommonDevice>();
    APP_ERROR ret = pDvpp_->Init();
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceResize[" << instanceId_ << "]: dvpp channel create failed!";
        return ret;
    }
    // pre allocate vpc output data buffer, different frames will reuse this
    // buffer, so memory copy is needed
    outBufferSize_ = (embeddingWidthStride_ * embeddingHeightStride_ + qualityWidthStride_ * qualityHeightStride_ +
        warpAffineWidthStride_ * warpAffineHeightStride_) *
        YUV_BYTES_NU / YUV_BYTES_DE; // yuv420sp

    for (uint32_t i = 0; i < maxFaceNumPerFrame_; i++) {
        void *outputBuffer = nullptr;
        ret = acldvppMalloc(&outputBuffer, outBufferSize_);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceResize[" << instanceId_ << "]: aclrtMalloc failed";
            return ret;
        }
        outputBufferShared_.push_back(outputBuffer);
    }
    return APP_ERR_OK;
}
#else
APP_ERROR FaceResize::InitDvpp(void)
{
    APP_ERROR ret = aclrtCreateStream(&dvppStream_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceResize[" << instanceId_ << "]: aclrtCreateStream failed, ret=" << ret;
        return ret;
    }

    dvppChannelDesc_ = acldvppCreateChannelDesc();
    if (dvppChannelDesc_ == nullptr) {
        LogError << "FaceResize[" << instanceId_ << "]: acldvppCreateChannelDesc failed";
        return APP_ERR_ACL_BAD_ALLOC;
    }

    ret = acldvppCreateChannel(dvppChannelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceResize[" << instanceId_ << "]: acldvppCreateChannel failed, ret=" << ret;
        return ret;
    }

    for (uint32_t i = 0; i < maxRoiNums_; i++) {
        acldvppRoiConfig *dvppRoiConfig = acldvppCreateRoiConfig(0, 1, 0, 1);
        if (dvppRoiConfig == nullptr) {
            LogError << "FaceResize[" << instanceId_ << "]: acldvppCreateRoiConfig failed";
            return APP_ERR_ACL_BAD_ALLOC;
        }
        dvppRoiConfigs_.push_back(
            std::shared_ptr<acldvppRoiConfig>(dvppRoiConfig, [](acldvppRoiConfig *p) { acldvppDestroyRoiConfig(p); }));
    }

    memPoolEmbedding_ = MemoryPool::NewMemoryPoolResource();
    memPoolQuality_ = MemoryPool::NewMemoryPoolResource();
    memPoolCrop_ = MemoryPool::NewMemoryPoolResource();

    uint32_t size = embeddingWidthStride_ * embeddingHeightStride_ * YUV_BYTES_NU / YUV_BYTES_DE;
    ret = memPoolEmbedding_->Init(size, MODULE_QUEUE_SIZE);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceResize[" << instanceId_ << "]: memPool Init fail ret=" << ret;
        return ret;
    }

    size = qualityWidthStride_ * qualityHeightStride_ * YUV_BYTES_NU / YUV_BYTES_DE;
    ret = memPoolQuality_->Init(size, MODULE_QUEUE_SIZE);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceResize[" << instanceId_ << "]: memPool Init fail ret=" << ret;
        return ret;
    }

    size = warpAffineWidthStride_ * warpAffineHeightStride_ * YUV_BYTES_NU / YUV_BYTES_DE;
    ret = memPoolCrop_->Init(size, MODULE_QUEUE_SIZE);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceResize[" << instanceId_ << "]: memPool Init fail ret=" << ret;
        return ret;
    }
    return APP_ERR_OK;
}
#endif

APP_ERROR FaceResize::ParseConfig(ConfigParser &configParser)
{
    LogDebug << "FaceResize[" << instanceId_ << "]: begin to parse config values.";
    std::string itemCfgStr = {};
    itemCfgStr = std::string("streamWidthMax");
    APP_ERROR ret = configParser.GetIntValue(itemCfgStr, streamWidthMax_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceResize[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = std::string("streamHeightMax");
    ret = configParser.GetIntValue(itemCfgStr, streamHeightMax_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceResize[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = std::string("max_face_num_per_frame");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, maxFaceNumPerFrame_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceResize[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    LogDebug << "FaceResize[" << instanceId_ << "]" << " maxFaceNumPerFrame_:" <<
        maxFaceNumPerFrame_;

    return ret;
}

APP_ERROR FaceResize::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "FaceResize[" << instanceId_ << "]: Begin to init MOT feature extraction Pre instance" <<
        initArgs.instanceId;

    AssignInitArgs(initArgs);

    isStop_ = false;

    // initialize config params
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceResize[" << instanceId_ << "]: Fail to parse config params, ret=" << ret << "(" <<
            GetAppErrCodeInfo(ret) << ").";
        return ret;
    }

    if (maxFaceNumPerFrame_ < 1) {
        LogFatal << "FaceResize[" << instanceId_ << "]: max_face_num_per_frame must be larger than 0";
        return APP_ERR_COMM_FAILURE;
    }

    ret = InitDvpp();
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceResize[" << instanceId_ << "]: dvpp init failed";
        return ret;
    }
    faceSizes_ = { 
        { embeddingWidthStride_, embeddingHeightStride_ },
        { qualityWidthStride_, qualityHeightStride_ },
        { warpAffineWidthStride_, warpAffineHeightStride_ }
    };

    faceExpands_ = { 
        { 0.15, 0.15, 0.3, 0.5 },
        { 0.1, 0.1, 0.1, 0.1 },
        { 0.15, 0.15, 0.15, 0.15 }
    };

    return APP_ERR_OK;
}

#ifdef ASCEND_ACL_OPEN_VESION
APP_ERROR FaceResize::DeInitDvpp()
{
    for (uint32_t i = 0; i < outputBufferShared_.size(); i++) {
        acldvppFree(outputBufferShared_[i]);
    }
    outputBufferShared_.clear();

    if (pDvpp_.get() != nullptr) {
        pDvpp_->DeInit();
    }
    return APP_ERR_OK;
}
#else
APP_ERROR FaceResize::DeInitDvpp()
{
    APP_ERROR ret = aclrtSynchronizeStream(dvppStream_);
    if (ret != APP_ERR_OK) {
        LogFatal << "Failed to synchronize stream, ret = " << ret << ".";
        return ret;
    }

    ret = aclrtDestroyStream(dvppStream_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceResize[" << instanceId_ << "]: Fail to destroy dvppstream.";
    }
    dvppStream_ = nullptr;
    ret = acldvppDestroyChannelDesc(dvppChannelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceResize[" << instanceId_ << "]: Fail to destroy dvppchannel.";
    }
    dvppBatchPicDescInput_.reset();
    dvppBatchPicDescOutput_.reset();
    dvppRoiConfigs_.clear();

    memPoolEmbedding_->DeInit();
    memPoolQuality_->DeInit();
    memPoolCrop_->DeInit();
    return ret;
}
#endif

APP_ERROR FaceResize::DeInit(void)
{
    LogDebug << "FaceResize[" << instanceId_ << "]::begin to deinit.";
    DeInitDvpp();
    faceSizes_.clear();
    faceExpands_.clear();
    return APP_ERR_OK;
}

APP_ERROR FaceResize::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "FaceResize[" << instanceId_ << "]: Begin to process data.";
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::static_pointer_cast<FrameAiInfo>(inputData);
    if (frameAiInfo.get() == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }

    if (frameAiInfo->detectResult.size() > 0) {
        FaceCropResize(frameAiInfo);
    }

    if (frameAiInfo->detectResult.size() > maxFaceNumPerFrame_) {
        LogDebug << "Chn" << frameAiInfo->info.channelId << " detect result:" << frameAiInfo->detectResult.size();
        for (uint32_t i = maxFaceNumPerFrame_; i < frameAiInfo->detectResult.size(); i++) {
            FaceObject faceObject;
            faceObject.info = frameAiInfo->detectResult[i];
            faceObject.frameInfo = frameAiInfo->info;
            frameAiInfo->face.push_back(faceObject);
        }
    }

    if (frameAiInfo->info.mode == FRAME_MODE_SEARCH) {
        // search
        FrameCache::GetInstance(frameAiInfo->info.channelId)->CacheFrame(frameAiInfo);
        if (frameAiInfo->face.size() != 0) {
            LogDebug << "[Cache] " << frameAiInfo->face.size() << " Faces Send Ch:" << frameAiInfo->info.channelId <<
                " frame:" << frameAiInfo->info.frameId;
            for (uint32_t i = 0; i < frameAiInfo->face.size(); i++) {
                std::shared_ptr<FaceObject> faceObject = std::make_shared<FaceObject>(frameAiInfo->face[i]);
                LogDebug << "FrameCache MOTEmbeddingPre Send " << faceObject->frameInfo.channelId << "_" <<
                    faceObject->frameInfo.frameId;
                SendToNextModule(MT_MOT_EMBEDDING, faceObject, frameAiInfo->info.channelId);
            }
        } else {
            // send empty face object to embedding
            LogDebug << "[Cache] Empty Faces Send Ch:" << frameAiInfo->info.channelId << " frame:" <<
                frameAiInfo->info.frameId;
            std::shared_ptr<FaceObject> faceObject = std::make_shared<FaceObject>();
            faceObject->frameInfo = frameAiInfo->info;
            SendToNextModule(MT_MOT_EMBEDDING, faceObject, frameAiInfo->info.channelId);
        }
    } else {
        // regist
        SendToNextModule(MT_FACE_LANDMARK, frameAiInfo, frameAiInfo->info.channelId);
    }
    return APP_ERR_OK;
}

#ifdef ASCEND_ACL_OPEN_VESION
void FaceResize::GetDvppCropInputMsg(const std::shared_ptr<FrameAiInfo> &frameAiInfo,
    std::vector<DvppCropInputMsg> &inputs)
{
    uint32_t minSize = std::min((uint32_t)frameAiInfo->detectResult.size(), maxFaceNumPerFrame_);
    for (uint32_t i = 0; i < minSize; i++) {
        uint32_t outputOffset = 0;
        for (uint32_t j = 0; j < faceSizes_.size(); j++) {
            DvppCropInputMsg input;
            input.inH = frameAiInfo->imgOrigin.height;
            input.inW = frameAiInfo->imgOrigin.width;
            input.imgBuf = frameAiInfo->imgOrigin.buf.deviceData.get();
            input.outW = faceSizes_[j][WIDTH_INDEX];
            input.outH = faceSizes_[j][HEIGHT_INDEX];
            input.outBuf = (uint8_t *)outputBufferShared_[i] + outputOffset;
            outputOffset += faceSizes_[j][WIDTH_INDEX] * faceSizes_[j][HEIGHT_INDEX] * YUV_BYTES_NU / YUV_BYTES_DE;
            const uint32_t minWidthIndex = 0;
            const uint32_t minHeightIndex = 0;
            const uint32_t maxWidthIndex = frameAiInfo->imgOrigin.width - 1;
            const uint32_t maxHeightIndex = frameAiInfo->imgOrigin.height - 1;
            // modify the input rect
            input.roi.left = std::max((uint32_t)(frameAiInfo->detectResult[i].minx -
                frameAiInfo->detectResult[i].width * faceExpands_[j][LEFT_EXPAND_INDEX]),
                minWidthIndex);
            input.roi.right = std::min((uint32_t)(frameAiInfo->detectResult[i].minx +
                frameAiInfo->detectResult[i].width * (1.f + faceExpands_[j][RIGHT_EXPAND_INDEX])),
                maxWidthIndex);
            input.roi.up = std::max((uint32_t)(frameAiInfo->detectResult[i].miny -
                frameAiInfo->detectResult[i].height * faceExpands_[j][UP_EXPAND_INDEX]),
                minHeightIndex);
            input.roi.down = std::min((uint32_t)(frameAiInfo->detectResult[i].miny +
                frameAiInfo->detectResult[i].height * (1.f + faceExpands_[j][DOWN_EXPAND_INDEX])),
                maxHeightIndex);
            inputs.push_back(input);
        }
    }
}
#else
APP_ERROR FaceResize::SetRoiConfigDesc(const std::shared_ptr<FrameAiInfo> &frameAiInfo)
{
    uint32_t minSize = std::min((uint32_t)frameAiInfo->detectResult.size(), maxFaceNumPerFrame_);
    for (uint32_t i = 0; i < minSize; i++) {
        // left,right,up,down
        for (uint32_t j = 0; j < faceSizes_.size(); j++) {
            const float minWidthIndex = 0.f;
            const float minHeightIndex = 0.f;
            const float maxWidthIndex = frameAiInfo->imgOrigin.width - 1;
            const float maxHeightIndex = frameAiInfo->imgOrigin.height - 1;
            // modify the input rect
            uint32_t left = std::max((frameAiInfo->detectResult[i].minx -
                frameAiInfo->detectResult[i].width * faceExpands_[j][LEFT_EXPAND_INDEX]),
                minWidthIndex);
            uint32_t right = std::min((frameAiInfo->detectResult[i].minx +
                frameAiInfo->detectResult[i].width * (1.f + faceExpands_[j][RIGHT_EXPAND_INDEX])),
                maxWidthIndex);
            uint32_t up = std::max((frameAiInfo->detectResult[i].miny -
                    frameAiInfo->detectResult[i].height * faceExpands_[j][UP_EXPAND_INDEX]),
                minHeightIndex);
            uint32_t down = std::min((frameAiInfo->detectResult[i].miny +
                frameAiInfo->detectResult[i].height * (1.f + faceExpands_[j][DOWN_EXPAND_INDEX])),
                maxHeightIndex);
            // align
            left = left / VPC_EVEN_ALIGN * VPC_EVEN_ALIGN;
            right = right / VPC_EVEN_ALIGN * VPC_EVEN_ALIGN + 1;
            up = up / VPC_EVEN_ALIGN * VPC_EVEN_ALIGN;
            down = down / VPC_EVEN_ALIGN * VPC_EVEN_ALIGN + 1;
            APP_ERROR ret = acldvppSetRoiConfig(dvppRoiConfigs_[i * faceSizes_.size() + j].get(),
                left, right, up, down);
            if (ret != APP_ERR_OK) {
                LogError << "FaceResize[" << instanceId_ << "]:acldvppSetRoiConfig failed";
                return ret;
            }
            roiNums_++;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR FaceResize::SetDvppPicDesc(std::shared_ptr<acldvppBatchPicDesc> dvppBatchPicDesc, const uint32_t &index,
    const PicDescInfo &picDescInfo) const
{
    acldvppPicDesc *dvppPicDesc = acldvppGetPicDesc(dvppBatchPicDesc.get(), index);
    if (dvppPicDesc == nullptr) {
        LogDebug << "FaceResize[" << instanceId_ << "]:acldvppGetPicDesc failed";
        return APP_ERR_ACL_FAILURE;
    }
    APP_ERROR ret = acldvppSetPicDescData(dvppPicDesc, picDescInfo.data);
    if (ret != APP_ERR_OK) {
        LogDebug << "FaceResize[" << instanceId_ << "]:acldvppSetPicDescData failed";
        return ret;
    }
    ret = acldvppSetPicDescSize(dvppPicDesc, picDescInfo.dataSize);
    if (ret != APP_ERR_OK) {
        LogDebug << "FaceResize[" << instanceId_ << "]:acldvppSetPicDescSize failed";
        return ret;
    }
    ret = acldvppSetPicDescFormat(dvppPicDesc, picDescInfo.format);
    if (ret != APP_ERR_OK) {
        LogDebug << "FaceResize[" << instanceId_ << "]:acldvppSetPicDescFormat failed";
        return ret;
    }
    ret = acldvppSetPicDescWidth(dvppPicDesc, picDescInfo.width);
    if (ret != APP_ERR_OK) {
        LogDebug << "FaceResize[" << instanceId_ << "]:acldvppSetPicDescWidth failed";
        return ret;
    }
    ret = acldvppSetPicDescHeight(dvppPicDesc, picDescInfo.height);
    if (ret != APP_ERR_OK) {
        LogDebug << "FaceResize[" << instanceId_ << "]:acldvppSetPicDescHeight failed";
        return ret;
    }
    ret = acldvppSetPicDescWidthStride(dvppPicDesc, picDescInfo.widthStride);
    if (ret != APP_ERR_OK) {
        LogDebug << "FaceResize[" << instanceId_ << "]:acldvppSetPicDescWidthStride failed";
        return ret;
    }
    ret = acldvppSetPicDescHeightStride(dvppPicDesc, picDescInfo.heightStride);
    if (ret != APP_ERR_OK) {
        LogDebug << "FaceResize[" << instanceId_ << "]:acldvppSetPicDescHeightStride failed";
        return ret;
    }
    return APP_ERR_OK;
}

static PicDescInfo GetPicDescInfo(const std::shared_ptr<FrameAiInfo> frameAiInfo)
{
    PicDescInfo picDecsInfo = {};
    picDecsInfo.width = frameAiInfo->imgOrigin.width;
    picDecsInfo.height = frameAiInfo->imgOrigin.height;
    picDecsInfo.widthStride = frameAiInfo->imgOrigin.widthAligned;
    picDecsInfo.heightStride = frameAiInfo->imgOrigin.heightAligned;
    picDecsInfo.dataSize = frameAiInfo->imgOrigin.buf.dataSize;
    picDecsInfo.data = (void *)frameAiInfo->imgOrigin.buf.deviceData.get();
    picDecsInfo.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    return picDecsInfo;
}

static PicDescInfo GetPicDescInfo(const uint32_t &width, const uint32_t &height, std::shared_ptr<uint8_t> data)
{
    PicDescInfo picDecsInfo = {};
    picDecsInfo.width = width;
    picDecsInfo.height = height;
    uint32_t widthStride = DVPP_ALIGN_UP(width, DVPP_WIDTH_ALIGN);
    uint32_t heightStride = DVPP_ALIGN_UP(height, DVPP_HEIGHT_ALIGN);
    picDecsInfo.widthStride = widthStride;
    picDecsInfo.heightStride = heightStride;
    picDecsInfo.dataSize = widthStride * heightStride * YUV_BYTES_NU / YUV_BYTES_DE;
    picDecsInfo.data = (void *)data.get();
    picDecsInfo.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    return picDecsInfo;
}

APP_ERROR FaceResize::SetBatchPicDesc(const std::shared_ptr<FrameAiInfo> &frameAiInfo)
{
    acldvppBatchPicDesc *dvppBatchPicDescInput = acldvppCreateBatchPicDesc(INPUT_IMAGE_NUM);
    if (dvppBatchPicDescInput == nullptr) {
        LogError << "FaceResize[" << instanceId_ << "]:acldvppCreateBatchPicDesc failed";
        return APP_ERR_ACL_BAD_ALLOC;
    }
    dvppBatchPicDescInput_.reset(dvppBatchPicDescInput, [](acldvppBatchPicDesc *p) { acldvppDestroyBatchPicDesc(p); });

    uint32_t minSize = std::min((uint32_t)frameAiInfo->detectResult.size(), maxFaceNumPerFrame_);
    acldvppBatchPicDesc *dvppBatchPicDescOutput =
        acldvppCreateBatchPicDesc(minSize * faceSizes_.size());
    if (dvppBatchPicDescOutput == nullptr) {
        LogError << "FaceResize[" << instanceId_ << "]:acldvppCreateBatchPicDesc failed";
        return APP_ERR_ACL_BAD_ALLOC;
    }
    dvppBatchPicDescOutput_.reset(dvppBatchPicDescOutput,
        [](acldvppBatchPicDesc *p) { acldvppDestroyBatchPicDesc(p); });

    PicDescInfo picDecsInfoInput = GetPicDescInfo(frameAiInfo);
    APP_ERROR ret = SetDvppPicDesc(dvppBatchPicDescInput_, 0, picDecsInfoInput);
    if (ret != APP_ERR_OK) {
        LogError << "FaceResize[" << instanceId_ << "]:SetDvppPicDesc failed";
        return ret;
    }

    memoryBlockVector_.clear();
    for (uint32_t i = 0; i < minSize; i++) {
        for (uint32_t j = 0; j < faceSizes_.size(); j++) {
            std::shared_ptr<uint8_t> memoryBlock = nullptr;
            if (j == EMBEDDING_INDEX)
                memoryBlock = memPoolEmbedding_->GetMemoryBlock();
            if (j == QUALITY_INDEX)
                memoryBlock = memPoolQuality_->GetMemoryBlock();
            if (j == CROP_INDEX)
                memoryBlock = memPoolCrop_->GetMemoryBlock();
            PicDescInfo picDecsInfo = GetPicDescInfo(faceSizes_[j][WIDTH_INDEX],
                faceSizes_[j][HEIGHT_INDEX], memoryBlock);
            memoryBlockVector_.push_back(memoryBlock);
            picDecsInfo.data = memoryBlock.get();
            APP_ERROR ret = SetDvppPicDesc(dvppBatchPicDescOutput_, i * faceSizes_.size() + j, picDecsInfo);
            if (ret != APP_ERR_OK) {
                LogError << "FaceResize[" << instanceId_ << "]:SetDvppPicDesc failed";
                return ret;
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR FaceResize::AclBatchCrop(std::shared_ptr<FrameAiInfo> frameAiInfo)
{
    APP_ERROR ret = SetBatchPicDesc(frameAiInfo);
    if (ret != APP_ERR_OK) {
        LogError << "FaceResize[" << instanceId_ << "]:SetBatchPicDesc failed";
        return ret;
    }

    ret = SetRoiConfigDesc(frameAiInfo);
    if (ret != APP_ERR_OK) {
        LogError << "FaceResize[" << instanceId_ << "]:SetRoiConfigDesc failed ret=" << ret;
        return ret;
    }

    acldvppRoiConfig *cropAreas[MAX_ROINUMS] = {};
    for (uint32_t i = 0; i < std::min(roiNums_, MAX_ROINUMS); i++) {
        cropAreas[i] = dvppRoiConfigs_[i].get();
    }

    ret = acldvppVpcBatchCropAsync(dvppChannelDesc_, dvppBatchPicDescInput_.get(), &roiNums_, 1,
        dvppBatchPicDescOutput_.get(), cropAreas, dvppStream_);
    if (ret != APP_ERR_OK) {
        roiNums_ = 0;
        LogError << "FaceResize[" << instanceId_ << "]:acldvppVpcBatchCropAsync failed";
        return ret;
    }
    roiNums_ = 0;

    ret = aclrtSynchronizeStream(dvppStream_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed tp synchronize stream, ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}
#endif
void FaceResize::SetImgEmbedding(FaceObject &faceObject, std::shared_ptr<uint8_t> buffer, const uint32_t &width,
    const uint32_t &height) const
{
    uint32_t widthAlign = DVPP_ALIGN_UP(width, DVPP_WIDTH_ALIGN);
    uint32_t heightAlign = DVPP_ALIGN_UP(height, DVPP_HEIGHT_ALIGN);
    faceObject.imgEmbedding.format = 1;
    faceObject.imgEmbedding.width = width;
    faceObject.imgEmbedding.height = height;
    faceObject.imgEmbedding.widthAligned = widthAlign;
    faceObject.imgEmbedding.heightAligned = heightAlign;
    faceObject.imgEmbedding.buf.deviceData = buffer;
    faceObject.imgEmbedding.buf.dataSize = heightAlign * widthAlign * YUV_BYTES_NU / YUV_BYTES_DE;
}

void FaceResize::SetImgQuality(FaceObject &faceObject, std::shared_ptr<uint8_t> buffer, const uint32_t &width,
    const uint32_t &height) const
{
    uint32_t widthAlign = DVPP_ALIGN_UP(width, DVPP_WIDTH_ALIGN);
    uint32_t heightAlign = DVPP_ALIGN_UP(height, DVPP_HEIGHT_ALIGN);
    faceObject.imgQuality.format = 1;
    faceObject.imgQuality.width = width;
    faceObject.imgQuality.height = height;
    faceObject.imgQuality.widthAligned = widthAlign;
    faceObject.imgQuality.heightAligned = heightAlign;
    faceObject.imgQuality.buf.deviceData = buffer;
    faceObject.imgQuality.buf.dataSize = heightAlign * widthAlign * YUV_BYTES_NU / YUV_BYTES_DE;
}


void FaceResize::SetImgCroped(FaceObject &faceObject, std::shared_ptr<uint8_t> buffer, const uint32_t &width,
    const uint32_t &height) const
{
    uint32_t widthAlign = DVPP_ALIGN_UP(width, DVPP_WIDTH_ALIGN);
    uint32_t heightAlign = DVPP_ALIGN_UP(height, DVPP_HEIGHT_ALIGN);
    faceObject.imgCroped.format = 1;
    faceObject.imgCroped.width = width;
    faceObject.imgCroped.height = height;
    faceObject.imgCroped.widthAligned = widthAlign;
    faceObject.imgCroped.heightAligned = heightAlign;
    faceObject.imgCroped.buf.deviceData = buffer;
    faceObject.imgCroped.buf.dataSize = heightAlign * widthAlign * YUV_BYTES_NU / YUV_BYTES_DE;
}

void FaceResize::SetImage(FaceObject &faceObject, std::shared_ptr<uint8_t> buffer, const FaceSizeInfo &faceSizeInfo,
    const uint32_t &index)
{
    if (index == EMBEDDING_INDEX) {
        SetImgEmbedding(faceObject, buffer, faceSizeInfo.width, faceSizeInfo.height);
    }
    if (index == QUALITY_INDEX) {
        SetImgQuality(faceObject, buffer, faceSizeInfo.width, faceSizeInfo.height);
    }
    if (index == CROP_INDEX) {
        SetImgCroped(faceObject, buffer, faceSizeInfo.width, faceSizeInfo.height);
    }
}


#ifdef ASCEND_ACL_OPEN_VESION
APP_ERROR FaceResize::FaceCropResize(std::shared_ptr<FrameAiInfo> frameAiInfo)
{
    LogDebug << "FaceResize[" << instanceId_ << "]: face crop and resize.";

    std::vector<DvppCropInputMsg> inputs;
    GetDvppCropInputMsg(frameAiInfo, inputs);
    LogDebug << "FaceReisze[" << instanceId_ << "]: inputs size: " << inputs.size() << "\n";
    if (inputs.size() == 0) {
        return APP_ERR_OK;
    }
    APP_ERROR ret = pDvpp_->VpcMultiCrop(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "FaceResize[" << instanceId_ << "]: VpcMultiCrop failed";
        return ret;
    }
    uint32_t minSize = std::min((uint32_t)frameAiInfo->detectResult.size(), maxFaceNumPerFrame_);
    for (uint32_t i = 0; i < minSize; i++) {
        uint32_t outputOffset = 0;
        FaceObject faceObject;
        for (uint32_t j = 0; j < faceSizes_.size(); j++) {
            uint8_t *imgResize = nullptr;
            uint32_t outputSize = faceSizes_[j][0] * faceSizes_[j][1] * YUV_BYTES_NU / YUV_BYTES_DE;
            imgResize = (uint8_t *)malloc(outputSize);
            if (imgResize == NULL) {
                LogFatal << "FaceResize[" << instanceId_ << "]: malloc failed\n";
                return APP_ERR_COMM_FAILURE;
            }
            std::copy((uint8_t *)outputBufferShared_[i] + outputOffset,
                (uint8_t *)outputBufferShared_[i] + outputOffset + outputSize, imgResize);
            std::shared_ptr<uint8_t> sharedPtr((uint8_t *)imgResize, free);

            outputOffset += outputSize;
            FaceSizeInfo faceSizeInfo = {};
            faceSizeInfo.width = faceSizes_[j][WIDTH_INDEX];
            faceSizeInfo.height = faceSizes_[j][HEIGHT_INDEX];
            SetImage(faceObject, sharedPtr, faceSizeInfo, j);
        }
        faceObject.info = frameAiInfo->detectResult[i];
        faceObject.frameInfo = frameAiInfo->info;
        if (frameAiInfo->info.mode == FRAME_MODE_REG) {
            faceObject.personInfo = frameAiInfo->info.personInfo;
        }
        frameAiInfo->face.push_back(faceObject);
    }
    frameAiInfo->imgOrigin.buf.deviceData.reset();
    return APP_ERR_OK;
}
#else
APP_ERROR FaceResize::FaceCropResize(std::shared_ptr<FrameAiInfo> frameAiInfo)
{
    LogDebug << "FaceResize[" << instanceId_ << "]: face crop and resize.";

    auto startTime = CostStatistic::GetStart();
    APP_ERROR ret = AclBatchCrop(frameAiInfo);
    if (ret != APP_ERR_OK) {
        LogError << "FaceResize[" << instanceId_ << "]: AclBatchCrop failed ret=" << ret;
        return ret;
    }
    double costMs = CostStatistic::GetCostTime(startTime);
    LogDebug << "AclBatchCrop[" << costMs << "ms]";

    uint32_t minSize = std::min((uint32_t)frameAiInfo->detectResult.size(), maxFaceNumPerFrame_);
    for (uint32_t i = 0; i < minSize; i++) {
        FaceObject faceObject;
        for (uint32_t j = 0; j < faceSizes_.size(); j++) {
            auto sharedPtr = memoryBlockVector_[i * faceSizes_.size() + j];
            FaceSizeInfo faceSizeInfo = {};
            faceSizeInfo.width = faceSizes_[j][WIDTH_INDEX];
            faceSizeInfo.height =  faceSizes_[j][HEIGHT_INDEX];
            SetImage(faceObject, sharedPtr, faceSizeInfo, j);
        }
        faceObject.info = frameAiInfo->detectResult[i];
        faceObject.frameInfo = frameAiInfo->info;
        if (frameAiInfo->info.mode == FRAME_MODE_REG) {
            faceObject.personInfo = frameAiInfo->info.personInfo;
        }
        frameAiInfo->face.push_back(faceObject);
    }
    frameAiInfo->imgOrigin.buf.deviceData.reset();
    return APP_ERR_OK;
}
#endif
} // namespace ascendFaceRecognition
