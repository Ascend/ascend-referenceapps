/*
 * @Author: your name
 * @Date: 2020-06-28 12:15:04
 * @LastEditTime: 2020-06-28 12:15:05
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /facerecognition/src/CtrlCPU/ImageDecoder/ImageDecoder.h
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

#ifndef INC_IMAGE_DECODER_H
#define INC_IMAGE_DECODER_H

#include "ModuleBase/ModuleBase.h"
#ifndef ASCEND_ACL_OPEN_VESION
#include "DvppCommon/DvppCommon.h"
#else
#include "DvppCommonDevice/DvppCommonDevice.h"
#endif
#include "ConfigParser/ConfigParser.h"
#include "DataTrans/DataTrans.pb.h"

namespace ascendFaceRecognition {
class ImageDecoder : public ModuleBase {
public:
    ImageDecoder();
    ~ImageDecoder();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
#ifdef ASCEND_ACL_OPEN_VESION
    APP_ERROR DeviceVpcHandle(std::shared_ptr<uint8_t> inputData, std::shared_ptr<uint8_t> &vpcOutBuffer);
    APP_ERROR DeviceJpegDecodeHandle(std::shared_ptr<uint8_t> inputData, uint32_t inputDataLen, 
        std::shared_ptr<uint8_t> &jpegDecodeBuffer, uint32_t jpegWidth, uint32_t jpegHeight);
#else
    APP_ERROR VpcHandle(std::shared_ptr<uint8_t> inputData, std::shared_ptr<uint8_t> &vpcOutBuffert);
    APP_ERROR JpegDecodeHandle(std::shared_ptr<uint8_t> inputData, uint32_t inputDataLen, 
        std::shared_ptr<uint8_t> &jpegDecodeBuffer, uint32_t jpegWidth, uint32_t jpegHeight);
    APP_ERROR VpcResize(std::shared_ptr<uint8_t> input, std::shared_ptr<uint8_t> output);
#endif
    APP_ERROR DecodeRegistFace(std::shared_ptr<RegistFaceInfoDataTrans> registFace,
        std::shared_ptr<FrameAiInfo> frameAiInfo);

#ifndef ASCEND_ACL_OPEN_VESION
    std::shared_ptr<DvppCommon> pDvpp_ = nullptr;
#else
    std::shared_ptr<DvppCommonDevice> pDvpp_ = nullptr;
#endif

private:
    uint32_t resizeWidth_ = 0;
    uint32_t resizeHeight_ = 0;
    uint32_t originalWidth_ = 0;
    uint32_t originalHeight_ = 0;
    uint32_t numCount_ = 0;
#ifndef ASCEND_ACL_OPEN_VESION
    aclrtStream dvppStream_ = nullptr;
#endif
};
} // namespace ascendFaceRecognition

#endif
