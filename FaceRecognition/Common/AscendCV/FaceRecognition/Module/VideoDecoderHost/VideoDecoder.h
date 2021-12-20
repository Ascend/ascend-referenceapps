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
#include "acl/ops/acl_dvpp.h"
#include "DataTrans/DataTrans.pb.h"
#include "DvppCommon/DvppCommon.h"
#include "MemoryPool/MemoryPool.h"
#include <unistd.h>
#include <atomic>

namespace ascendFaceRecognition {
struct VdecInfo {
    std::shared_ptr<uint8_t> streamData;
    std::shared_ptr<uint8_t> imageData;
    uint32_t streamSize;
    uint32_t imageSize;
    uint64_t frameId;
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
    APP_ERROR InitVdec();
    APP_ERROR DeInitVdec();
    APP_ERROR VideoDecode(std::shared_ptr<VdecInfo> vdecInfo);
    APP_ERROR ParseConfig(ConfigParser &configParser);
    static void *DecoderThread(void *arg);
    static void VideoDecoderCallback(acldvppStreamDesc *input, acldvppPicDesc *output, void *userData);
    std::shared_ptr<uint8_t> AclrtMallocAndCopy(StreamDataTrans streamData);

private:
    FrameInfo frameInfo;
    uint64_t frameId_ = 0;
    uint32_t streamWidthMax_ = 1920;
    uint32_t streamHeightMax_ = 1080;
    uint32_t originalWidth_ = 0;
    uint32_t originalHeight_ = 0;
    uint32_t skipInterval_ = 3;
    pthread_t decoderThreadId_ = 0;
    bool stopDecoderThread_ = false;

    std::shared_ptr<aclvdecChannelDesc> vdecChannelDesc_ = nullptr;
    std::shared_ptr<MemoryPool> memPoolOrigin_ = nullptr;
    std::shared_ptr<uint8_t> uselessData_ = nullptr;
};

class DecodeH26xInfo {
public:
    VideoDecoder *videoDecoder = nullptr;
    std::shared_ptr<VdecInfo> vdecInfo = nullptr;
    ~DecodeH26xInfo() {}
    DecodeH26xInfo() {}
};
} // namespace ascendFaceRecognition

#endif