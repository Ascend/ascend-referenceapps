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

#include "VideoReader.h"
#include "Log/Log.h"

/*
 * @description: Use FFmpeg to get the video stream
 * @return: true if success, false if failure
 */
bool VideoReader::OpenVideo()
{
    char c[PATH_MAX + 1] = { 0x00 };
    size_t count = srcFileName_.copy(c, PATH_MAX + 1);
    if (count != srcFileName_.length()) {
        LogError << "Failed to copy" << c;
    }
    // Get the absolute path of input directory
    char path[PATH_MAX + 1] = { 0x00 };
    if ((strlen(c) > PATH_MAX) || (realpath(c, path) == nullptr)) {
        LogError << "Failed to get canonicalized path.";
        return false;
    }

    // register all format and codecs
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 9, 100)
    av_register_all();
#endif
    avformat_network_init();
    // open input file, and allocate format context
    if (avformat_open_input(&formatContext_, path, nullptr, nullptr) != 0) { // Open a file and parse
        LogError << "Failed to open file " << srcFileName_;
        return false;
    }
    // retrieve stream information
    if (avformat_find_stream_info(formatContext_, nullptr) != 0) { // Find format and index
        LogError << "Failed to find stream information.";
        return false;
    }

    // find stream
    videoStreamIndex_ = av_find_best_stream(formatContext_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoStreamIndex_ < 0) {
        LogError << "Failed to find " << av_get_media_type_string(AVMEDIA_TYPE_VIDEO) << "stream in file "
                 << srcFileName_;
        return false;
    }
    return true;
}

/*
 * @description: Read video frame using FFmpeg
 * @return: 0 if success, other values if failure
 */
int32_t VideoReader::GetNalu(AVPacket &packet)
{
    return av_read_frame(formatContext_, &packet);
}

/*
 * @description: Get width and height of video
 * @param: picDesc specifies the description of picture which contains Picname, width and height
 */
void VideoReader::GetVideoWH(PicDesc &picDesc)
{
    unsigned int i;
    // Iterate all the streams of this file and catch the video stream to get its width and height
    // Normally a video file without audio has only one stream: Video
    for (i = 0; i < formatContext_->nb_streams; i++) {
        if (formatContext_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            LogInfo << "width: " << formatContext_->streams[i]->codecpar->width;
            LogInfo << "height: " << formatContext_->streams[i]->codecpar->height;
            picDesc.width = formatContext_->streams[i]->codecpar->width;
            picDesc.height = formatContext_->streams[i]->codecpar->height;
            break;
        }
    }
}

/*
 * @description: Get video format from formatContext_
 * @param: videoFormat specifies the video type(h264, h265 or others)
 * @return: 0 if success, other values if failure
 */
APP_ERROR VideoReader::GetVideoType(acldvppStreamFormat &videoFormat)
{
    AVCodecID codecId = formatContext_->streams[0]->codecpar->codec_id;
    if (codecId == AV_CODEC_ID_H264) {
        videoFormat = H264_MAIN_LEVEL;
        return APP_ERR_OK;
    }
    if (codecId == AV_CODEC_ID_H265) {
        videoFormat = H265_MAIN_LEVEL;
        return APP_ERR_OK;
    }
    LogError << "\033[0;31mError unsupported format \033[0m" << codecId;
    return APP_ERR_COMM_FAILURE;
}

/*
 * @description: Close AVformatContext_
 */
void VideoReader::CloseVideo()
{
    avformat_close_input(&formatContext_);
}