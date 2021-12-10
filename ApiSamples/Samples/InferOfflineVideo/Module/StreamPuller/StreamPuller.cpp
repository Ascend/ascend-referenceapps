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

#include "StreamPuller.h"

#include <chrono>
#include <iostream>
#include <atomic>
#include <unistd.h>
#include "Log/Log.h"
#include "VideoDecoder/VideoDecoder.h"
#include "Singleton.h"

using namespace ascendBaseModule;

namespace {
const int LOW_THRESHOLD = 128;
const int MAX_THRESHOLD = 4096;
}

StreamPuller::StreamPuller()
{
    withoutInputQueue_ = true;
}

StreamPuller::~StreamPuller() {}

APP_ERROR StreamPuller::ParseConfig(ConfigParser &configParser)
{
    LogDebug << "StreamPuller [" << instanceId_ << "]: begin to parse config values.";
    std::string itemCfgStr = std::string("stream.ch") + std::to_string(instanceId_);
    return configParser.GetStringValue(itemCfgStr, streamName_);
}

APP_ERROR StreamPuller::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "Begin to init instance " << initArgs.instanceId;

    AssignInitArgs(initArgs);

    int ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogError << "StreamPuller[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }

    isStop_ = false;
    pFormatCtx_ = nullptr;

    LogDebug << "StreamPuller [" << instanceId_ << "] Init success.";
    return APP_ERR_OK;
}

APP_ERROR StreamPuller::DeInit(void)
{
    LogDebug << "StreamPuller [" << instanceId_ << "]: Deinit start.";

    // clear th cache of the queue
    avformat_close_input(&pFormatCtx_);

    isStop_ = true;
    pFormatCtx_ = nullptr;
    LogDebug << "StreamPuller [" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

APP_ERROR StreamPuller::Process(std::shared_ptr<void> inputData)
{
    int failureNum = 0;
    while (failureNum < 1) {
        StartStream();
        failureNum++;
    }
    return APP_ERR_OK;
}

APP_ERROR StreamPuller::StartStream()
{
    avformat_network_init(); // init network
    pFormatCtx_ = CreateFormatContext(); // create context
    if (pFormatCtx_ == nullptr) {
        LogError << "pFormatCtx_ null!";
        return APP_ERR_COMM_FAILURE;
    }
    // for debug dump
    av_dump_format(pFormatCtx_, 0, streamName_.c_str(), 0);

    // get stream infomation
    APP_ERROR ret = GetStreamInfo();
    if (ret != APP_ERR_OK) {
        LogError << "Stream Info Check failed, ret = " << ret;
        return APP_ERR_COMM_FAILURE;
    }

    LogInfo << "Start the stream......";
    PullStreamDataLoop(); // Cyclic stream pull

    return APP_ERR_OK;
}

APP_ERROR StreamPuller::GetStreamInfo()
{
    if (pFormatCtx_ != nullptr) {
        videoStream_ = -1;
        channelId_ = instanceId_;
        AVCodecID codecId = pFormatCtx_->streams[0]->codecpar->codec_id;
        if (codecId == AV_CODEC_ID_H264) {
            videoFormat_ = H264_MAIN_LEVEL;
        } else if (codecId == AV_CODEC_ID_H265) {
            videoFormat_ = H265_MAIN_LEVEL;
        } else {
            LogError << "\033[0;31mError unsupported format \033[0m" << codecId;
            return APP_ERR_COMM_FAILURE;
        }

        for (unsigned int i = 0; i < pFormatCtx_->nb_streams; i++) {
            AVStream *inStream = pFormatCtx_->streams[i];
            if (inStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                videoStream_ = i;
                videoHeight_ = inStream->codecpar->height;
                videoWidth_ = inStream->codecpar->width;
                break;
            }
        }

        if (videoStream_ == -1) {
            LogError << "Didn't find a video stream!";
            return APP_ERR_COMM_FAILURE;
        }

        if (videoHeight_ < LOW_THRESHOLD || videoWidth_ < LOW_THRESHOLD ||
            videoHeight_ > MAX_THRESHOLD || videoWidth_ > MAX_THRESHOLD) {
            LogError << "Size of frame is not supported in DVPP Video Decode!";
            return APP_ERR_COMM_FAILURE;
        }
    }
    return APP_ERR_OK;
}

AVFormatContext *StreamPuller::CreateFormatContext() const
{
    // create message for stream pull
    AVFormatContext *formatContext = nullptr;
    AVDictionary *options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "3000000", 0);
    int ret = avformat_open_input(&formatContext, streamName_.c_str(), nullptr, &options);
    if (options != nullptr) {
        av_dict_free(&options);
    }
    if (ret != 0) {
        LogError << "Couldn't open input stream" << streamName_.c_str() << ", ret=" << ret;
        return nullptr;
    }
    ret = avformat_find_stream_info(formatContext, nullptr);
    if (ret != 0) {
        LogError << "Couldn't find stream information, ret = " << ret;
        return nullptr;
    }
    return formatContext;
}

void StreamPuller::PullStreamDataLoop()
{
    // Pull data cyclically
    const uint16_t delayTime = 35000;
    AVPacket pkt;
    while (1) {
        if (isStop_ || pFormatCtx_ == nullptr) {
            break;
        }
        av_init_packet(&pkt);
        int ret = av_read_frame(pFormatCtx_, &pkt);
        if (ret != 0) {
            if (ret == AVERROR_EOF) {
                LogInfo << "StreamPuller [" << instanceId_ << "]: channel StreamPuller is EOF, exit";
                std::shared_ptr<CommonData> frameData = std::make_shared<CommonData>();
                frameData->eof = true;
                SendToNextModule(MT_VideoDecoder, frameData, channelId_);
                break;
            }
            LogInfo << "StreamPuller [" << instanceId_ << "]: channel Read frame failed, continue";
            av_packet_unref(&pkt);
            continue;
        } else if (pkt.stream_index == videoStream_) {
            if (pkt.size <= 0) {
                LogError << "Invalid pkt.size: " << pkt.size;
                av_packet_unref(&pkt);
                continue;
            }

            std::shared_ptr<CommonData> commonData = std::make_shared<CommonData>();
            commonData->eof = false;
            commonData->channelId = channelId_;
            commonData->srcWidth = videoWidth_;
            commonData->srcHeight = videoHeight_;
            commonData->videoFormat = videoFormat_;
            commonData->streamData.data.reset(new uint8_t[pkt.size], std::default_delete<uint8_t[]>());
            std::copy(pkt.data, pkt.data + pkt.size, static_cast<uint8_t*>(commonData->streamData.data.get()));
            commonData->streamData.size = pkt.size;
            SendToNextModule(MT_VideoDecoder, commonData, commonData->channelId);
        }
        av_packet_unref(&pkt);
        if (streamName_.find("rtsp:") != 0) {
            usleep(delayTime); // delay 35ms
        }
    }
}
