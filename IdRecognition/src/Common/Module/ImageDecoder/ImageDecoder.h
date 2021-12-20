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

#ifndef IMAGE_DECODER_H
#define IMAGE_DECODER_H

#include "ModuleBase/ModuleBase.h"
#ifdef ASCEND_FACE_USE_ACL_DVPP
#include "DvppCommon/DvppCommon.h"
#else
#include "DvppCommonDevice/DvppCommonDevice.h"
#endif
#include "ConfigParser/ConfigParser.h"
#include "Statistic/Statistic.h"

namespace ascendFaceRecognition {
class ImageDecoder : public ModuleBase {
public:
    ImageDecoder() : resizeWidth_(0), resizeHeight_(0), originalWidth_(0), originalHeight_(0) {};
    ~ImageDecoder();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);
    double GetRunTimeAvg();

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
#ifdef ASCEND_FACE_USE_ACL_DVPP
    APP_ERROR VpcHandle(uint8_t *inputdata, uint8_t *&vpcOutBuffer);
    APP_ERROR JpegDecodeHandle(uint8_t *inputData, uint32_t inputDataLen, uint8_t *&jpegDecodeBuffer,
        uint32_t jpegWidth, uint32_t jpegHeight);
#else
    APP_ERROR DeviceVpcHandle(uint8_t *inputdata, uint8_t *&vpcOutBuffer, uint32_t vpcInWidth, uint32_t vpcInHeight)
    const;
    APP_ERROR DeviceJpegDecodeHandle(uint8_t *inputData, uint32_t inputDataLen, uint8_t *&jpegDecodeBuffer,
        uint32_t jpegWidth, uint32_t jpegHeight) const;
#endif
    
    APP_ERROR PrepareDecoderInput(std::shared_ptr<StreamRawData> streamData, uint8_t *&devDataPtr);
    APP_ERROR PrepareHostResult(uint8_t *jpegDecodeBuffer, uint8_t *&jpegDecodeBufferHost);
    void SendData(std::shared_ptr<StreamRawData> streamData, uint8_t *jpegDecodeBuffer, uint8_t *vpcOutBuffer);

private:
#ifdef ASCEND_FACE_USE_ACL_DVPP
    std::unique_ptr<DvppCommon> pDvpp_ = nullptr;
#else
    std::unique_ptr<DvppCommonDevice> pDvpp_ = nullptr;
#endif
    int resizeWidth_;
    int resizeHeight_;
    int originalWidth_;
    int originalHeight_;
    uint32_t decodedWidth_ = {};
    uint32_t decodedHeight_ = {};
    uint32_t decodedWidthStride_ = {};
    uint32_t decodedHeightStride_ = {};
    uint32_t decodedDataSize_ = {};
    aclrtStream dvppStream_ = nullptr;
    Statistic imageDecoderStatic_ = {};
};
} // namespace ascendFaceRecognition

#endif
