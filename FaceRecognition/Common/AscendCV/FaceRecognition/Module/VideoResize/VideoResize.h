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
#ifndef VIDEO_RESIZE_H
#define VIDEO_RESIZE_H

#include <unistd.h>

#include "ModuleBase/ModuleBase.h"
#include "ConfigParser/ConfigParser.h"
#include "DataTrans/DataTrans.pb.h"
#include "DvppCommon/DvppCommon.h"
#include "MemoryPool/MemoryPool.h"
#include "MemoryPool/MemoryPool.h"
#include "acl/ops/acl_dvpp.h"

namespace ascendFaceRecognition {
class VideoResize : public ModuleBase {
public:
    VideoResize();
    ~VideoResize();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR Resize(DvppDataInfo &input, DvppDataInfo &output);
    APP_ERROR DetectResize(DvppDataInfo &input, ImageInfo &imageInfo);

private:
    uint32_t originWidth_ = 1920;
    uint32_t originHeight_ = 1080;
    uint32_t detectWidth_ = 0;
    uint32_t detectHeight_ = 0;
    std::shared_ptr<DvppCommon> vpcDvppCommon_ = nullptr;
    aclrtStream vpcDvppStream_ = nullptr;
    std::shared_ptr<MemoryPool> memPool_ = nullptr;
};
}
#endif