/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef VIDEO_READER_H
#define VIDEO_READER_H

#include <string>
#include <cstdint>
// description : use ffmpeg to split h264/h265 stream to seprate nalu
#ifdef __cplusplus
extern "C" {
#endif // #ifdef __cplusplus

#include "libavutil/imgutils.h"
#include "libavutil/samplefmt.h"
#include "libavformat/avformat.h"

#ifdef __cplusplus
}
#endif // #ifdef __cplusplus

#include "acl/ops/acl_dvpp.h"
#include "ErrorCode/ErrorCode.h"

struct PicDesc {
    std::string picName;
    int width;
    int height;
};

class VideoReader {
public:
    VideoReader(const std::string &srcFileName)
        : videoStreamIndex_(-1), formatContext_(nullptr), srcFileName_(srcFileName) {}

    bool OpenVideo();
    int32_t GetNalu(AVPacket &packet);
    void GetVideoWH(PicDesc &picDesc);
    APP_ERROR GetVideoType(acldvppStreamFormat &videoFormat);
    void CloseVideo();

private:
    int32_t videoStreamIndex_;
    AVFormatContext *formatContext_;
    std::string srcFileName_;
};

#endif // #ifndef VIDEO_READER_H
