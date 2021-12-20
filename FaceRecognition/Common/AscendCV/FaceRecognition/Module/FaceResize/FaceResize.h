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

#ifndef INC_FACEDEMO_FACERESIZEMODULE_BASE_H
#define INC_FACEDEMO_FACERESIZEMODULE_BASE_H

#include <thread>
#include "ModuleBase/ModuleBase.h"
#ifdef ASCEND_ACL_OPEN_VESION
#include "DvppCommonDevice/DvppCommonDevice.h"
#else
#include "DvppCommon/DvppCommon.h"
#include "MemoryPool/MemoryPool.h"
#endif

#include "acl/acl.h"

namespace ascendFaceRecognition {
struct FaceSizeInfo {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct PicDescInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t widthStride = 0;
    uint32_t heightStride = 0;
    uint32_t dataSize = 0;
    void *data = nullptr;
    acldvppPixelFormat format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
};


class FaceResize : public ModuleBase {
public:
    FaceResize();
    ~FaceResize();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);
protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);
private:
    APP_ERROR InitDvpp(void);
    APP_ERROR DeInitDvpp(void);
    APP_ERROR FaceCropResize(std::shared_ptr<FrameAiInfo> frameAiInfo);
    APP_ERROR ParseConfig(ConfigParser &configParser);
    void SetImgEmbedding(FaceObject &faceObject, std::shared_ptr<uint8_t> buffer, const uint32_t &width,
        const uint32_t &height) const;
    void SetImgQuality(FaceObject &faceObject, std::shared_ptr<uint8_t> buffer, const uint32_t &width,
        const uint32_t &height) const;
    void SetImgCroped(FaceObject &faceObject, std::shared_ptr<uint8_t> buffer, const uint32_t &width,
        const uint32_t &height) const;
    void SetImage(FaceObject &faceObject, std::shared_ptr<uint8_t> buffer, const FaceSizeInfo &faceSizeInfo,
        const uint32_t &index);

#ifdef ASCEND_ACL_OPEN_VESION
    void GetDvppCropInputMsg(const std::shared_ptr<FrameAiInfo> &frameAiInfo, std::vector<DvppCropInputMsg> &inputs);
#else
    APP_ERROR AclBatchCrop(std::shared_ptr<FrameAiInfo> frameAiInfo);
    APP_ERROR SetRoiConfigDesc(const std::shared_ptr<FrameAiInfo> &frameAiInfo);
    APP_ERROR SetBatchPicDesc(const std::shared_ptr<FrameAiInfo> &frameAiInfo);
    APP_ERROR SetDvppPicDesc(std::shared_ptr<acldvppBatchPicDesc> dvppBatchPicDesc, const uint32_t &index,
        const PicDescInfo &picDescInfo) const;
#endif
private:
    uint32_t maxFaceNumPerFrame_ = 10;
    int streamWidthMax_ = 1920;
    int streamHeightMax_ = 1080;
    uint32_t outBufferSize_ = 0;
    uint32_t embeddingWidthStride_ = 64;
    uint32_t embeddingHeightStride_ = 96;
    uint32_t qualityWidthStride_ = 96;
    uint32_t qualityHeightStride_ = 96;
    uint32_t warpAffineWidthStride_ = 112;
    uint32_t warpAffineHeightStride_ = 112;
    std::vector<void *> outputBufferShared_ = {};
    std::vector<std::vector<uint32_t>> faceSizes_ = {};
    std::vector<std::vector<float>> faceExpands_ = {};

#ifdef ASCEND_ACL_OPEN_VESION
    std::shared_ptr<DvppCommonDevice> pDvpp_ = nullptr;
#else
    aclrtStream dvppStream_ = nullptr;
    acldvppChannelDesc *dvppChannelDesc_ = nullptr;
    std::shared_ptr<acldvppBatchPicDesc> dvppBatchPicDescInput_ = nullptr;
    std::shared_ptr<acldvppBatchPicDesc> dvppBatchPicDescOutput_ = nullptr;
    std::vector<std::shared_ptr<acldvppRoiConfig>> dvppRoiConfigs_ = {};
    std::shared_ptr<MemoryPool> memPoolEmbedding_ = {};
    std::shared_ptr<MemoryPool> memPoolQuality_ = {};
    std::shared_ptr<MemoryPool> memPoolCrop_ = {};
    std::vector<std::shared_ptr<uint8_t>> memoryBlockVector_ = {};
    uint32_t roiNums_ = 0;
    uint32_t maxRoiNums_ = 128;
    uint32_t batchSize_ = 1;
#endif
};
}
#endif
