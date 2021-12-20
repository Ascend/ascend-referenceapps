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

#ifndef INFEROFFLINEVIDEOBASE_STREAMPULLER_H
#define INFEROFFLINEVIDEOBASE_STREAMPULLER_H

#include "acl/acl.h"
#include "ErrorCode/ErrorCode.h"
#include "ModuleBase/ModuleBase.h"
#include "ConfigParser/ConfigParser.h"
#include "DataTrans/DataTrans.pb.h"

extern "C" {
#include "libavformat/avformat.h"
}

namespace ascendFaceRecognition {
class StreamPuller : public ModuleBase {
public:
    StreamPuller();
    ~StreamPuller();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);

    APP_ERROR StartStream();
    AVFormatContext *CreateFormatContext();
    APP_ERROR GetStreamInfo();
    void PullStreamDataLoop();

private:
    int videoStream_ = 0;
    FrameInfoTrans frameInfo_ = {};
    std::string streamName_ = {};

    // class member
    AVFormatContext *pFormatCtx_ = nullptr;
    int videoIndex_ = 0;
    int webChannelId_ = 0;
    uint32_t skipInterval_ = 0;
};
} // namespace ascendFaceRecognition

#endif // INFEROFFLINEVIDEOBASE_STREAMPULLER_H
