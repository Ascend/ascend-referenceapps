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

#ifndef __VIDEO_DECODER__
#define __VIDEO_DECODER__

#include "ModuleBase/ModuleBase.h"
#include "ConfigParser/ConfigParser.h"
#include "Dvpp.h"
#include "acl/ops/acl_dvpp.h"
#include "DataTrans/DataTrans.pb.h"

#define CHECK_ODD(NUM) ((((NUM) % (2)) != (0)) ? (NUM) : ((NUM) - (1)))
#define CHECK_EVEN(NUM) ((((NUM) % (2)) == (0)) ? (NUM) : ((NUM) - (1)))


namespace ascendFaceRecognition {
struct FrameDepressInfo {
    std::shared_ptr<FrameAiInfo> frameAiInfo;
    std::shared_ptr<FRAME> frame;
    std::shared_ptr<unsigned char> buffer;
};

struct FrameOutputSize {
    uint32_t width;
    uint32_t height;
};

class VideoDecoder : public ModuleBase {
public:
    VideoDecoder();
    ~VideoDecoder();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR VpcHandle(void *inputdata, void **vpcOutBuffer, uint32_t vpcInWidth, uint32_t vpcInHeight);
    static void VdecResultCallback(FRAME *frame, void *hiai_data);
    static void VdecErrorCallback(VDECERR *vdecErr);
    APP_ERROR DecodeH26xVideo(std::shared_ptr<StreamDataTrans> streamRawData);
    APP_ERROR VdecImageResize(FRAME &frame, std::vector<ImageInfo> &cropImageList);
    std::shared_ptr<VpcUserImageConfigure> GetHfbcInputConfigure(FRAME &frame);
    void HandleResizeOutput(std::shared_ptr<VpcUserImageConfigure> imageConfigure,
        std::vector<ImageInfo> &cropImageList);
    void SetVpcUserRoiConfigure(const FRAME &frame, const FrameOutputSize &outSize, uint8_t *address,
        std::shared_ptr<VpcUserRoiConfigure> roiConfig);
    APP_ERROR GetDvppMallocSharedPtr(const uint32_t &size, std::shared_ptr<uint8_t> &buffer);

private:
    FrameInfo frameInfo;
    int64_t frameId = 0;
    uint32_t streamWidthMax_ = 1920;
    uint32_t streamHeightMax_ = 1080;

    uint32_t originalWidth_ = 0;
    uint32_t originalHeight_ = 0;
    void *originalOutputBuffer_ = nullptr;

    uint32_t resizeWidth_ = 0;
    uint32_t resizeHeight_ = 0;
    uint32_t resizeOutWidth_ = 0;
    uint32_t resizeOutHeight_ = 0;
    void *resizeOutputBuffer_ = nullptr;
    int resizeOutputSize_ = 0;
    uint32_t skipInterval_ = 3;

    IDVPPAPI *piDvppApiVdec = nullptr;
    IDVPPAPI *piDvppApiVpc = nullptr;
    aclrtStream dvppStream_ = nullptr;
};

class DecodeH26xInfo : public HIAI_DATA_SP {
public:
    VideoDecoder *videoDecoder = nullptr;
    int64_t hostFrameId = 0;
    ~DecodeH26xInfo() {}
};
} // namespace ascendFaceRecognition

#endif
