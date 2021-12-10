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

#include <unistd.h>
#include <chrono>
#include <iostream>

#include "StreamCache/StreamCache.h"
#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "Common.h"
#include "ChannelStatus/ChannelStatus.h"

#ifdef ASCEND_ACL_OPEN_VESION
#include "HdcChannel/HdcChannel.h"
#endif


namespace ascendFaceRecognition {
namespace {
const int LOW_THRESHOLD = 128;
const int MAX_THRESHOLD = 4096;
const int FORMAT_H264 = 2;
const int FORMAT_H265 = 3;
}

StreamPuller::StreamPuller()
{
    withoutInputQueue_ = true;
}

StreamPuller::~StreamPuller() {}

APP_ERROR StreamPuller::ParseConfig(ConfigParser &configParser)
{
    // load and parse config file
    LogDebug << "StreamPuller[" << instanceId_ << "]: begin to parse config values.";
    std::string itemCfgStr = std::string("stream.ch") + std::to_string(instanceId_);
    APP_ERROR ret = configParser.GetStringValue(itemCfgStr, streamName_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = std::string("skipInterval");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, skipInterval_);
    if (ret != APP_ERR_OK) {
        LogFatal << "VideoDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    return ret;
}

APP_ERROR StreamPuller::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "StreamPuller[" << instanceId_ << "]: StreamPuller init start.";
    // initialize member variables
    AssignInitArgs(initArgs);

    int ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "StreamPuller[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }

    isStop_ = false;
    pFormatCtx_ = nullptr;
    webChannelId_ = initArgs.instanceId + 1;

    LogDebug << "StreamPuller[" << instanceId_ << "]: StreamPuller init success.";
    return APP_ERR_OK;
}

APP_ERROR StreamPuller::DeInit(void)
{
    LogWarn << "StreamPuller[" << instanceId_ << "]: StreamPuller deinit start.";
    avformat_close_input(&pFormatCtx_);

    pFormatCtx_ = nullptr;
    LogDebug << "StreamPuller[" << instanceId_ << "]: StreamPuller deinit success.";

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
    pFormatCtx_ = avformat_alloc_context();
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
        LogError << "Stream Info Check failed!";
        return APP_ERR_COMM_FAILURE;
    }

    LogDebug << "Start the stream......";
    PullStreamDataLoop(); // Cyclic stream pull

    return APP_ERR_OK;
}

APP_ERROR StreamPuller::GetStreamInfo()
{
    if (pFormatCtx_ != nullptr) {
        videoStream_ = -1;
        frameInfo_.set_frameid(0);
        frameInfo_.set_channelid(instanceId_);
        AVCodecID codecId = pFormatCtx_->streams[0]->codecpar->codec_id;
        if (codecId == AV_CODEC_ID_H264) {
            frameInfo_.set_format(FORMAT_H264);
        } else if (codecId == AV_CODEC_ID_H265) {
            frameInfo_.set_format(FORMAT_H265);
        } else {
            LogError << "\033[0;31mError unsupported format \033[0m" << codecId;
            return APP_ERR_COMM_FAILURE;
        }

        for (unsigned int i = 0; i < pFormatCtx_->nb_streams; i++) {
            AVStream *inStream = pFormatCtx_->streams[i];
            if (inStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                videoStream_ = i;
                frameInfo_.set_height(inStream->codecpar->height);
                frameInfo_.set_width(inStream->codecpar->width);
                break;
            }
        }
        if (videoStream_ == -1) {
            LogError << "Didn't find a video stream!";
            return APP_ERR_COMM_FAILURE;
        }
        if (frameInfo_.height() < LOW_THRESHOLD || frameInfo_.width() < LOW_THRESHOLD ||
            frameInfo_.height() > MAX_THRESHOLD || frameInfo_.width() > MAX_THRESHOLD) {
            LogError << "Size of frame is not supported in DVPP Video Decode!";
            return APP_ERR_COMM_FAILURE;
        }
    }
    return APP_ERR_OK;
}

AVFormatContext *StreamPuller::CreateFormatContext()
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
        LogError << "Couldn't open input stream" << streamName_.c_str() << "ret=" << ret;
        return nullptr;
    }
    ret = avformat_find_stream_info(formatContext, nullptr);
    if (ret != 0) {
        LogError << "Couldn't find stream information";
        return nullptr;
    }
    return formatContext;
}

void StreamPuller::PullStreamDataLoop()
{
    // Pull data cyclically
    AVPacket pkt;
    while (1) {
        if (isStop_ || pFormatCtx_ == nullptr) {
            break;
        }
        av_init_packet(&pkt);
        int ret = av_read_frame(pFormatCtx_, &pkt);
        if (ret != 0) {
            LogError << "[StreamPuller] channel Read frame failed, continue!";
            if (ret == AVERROR_EOF) {
                LogError << "[StreamPuller] channel StreamPuller is EOF, over!";
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        } else if (pkt.stream_index == videoStream_) {
            if (pkt.size <= 0) {
                LogError << "Invalid pkt.size: " << pkt.size;
                continue;
            }
            
            // sent to the device
            frameInfo_.set_frameid(frameInfo_.frameid() + 1);
            std::shared_ptr<DataTrans> dataTrans = std::make_shared<DataTrans>();
            StreamDataTrans *streamData = dataTrans->mutable_streamdata();
            FrameInfoTrans *frameInfo = streamData->mutable_info();
            frameInfo->CopyFrom(frameInfo_);
            streamData->set_datasize(pkt.size);
            streamData->set_data(pkt.data, pkt.size);

            LogDebug << "channelId=" << frameInfo_.channelid() << ", frameId" << frameInfo_.frameid();

#ifdef ASCEND_ACL_OPEN_VESION
            HdcChannel::GetInstance()->SendData(HDC_SEARCH_CH_START_INDEX + instanceId_, dataTrans);
#else
            SendToNextModule(MT_VIDEO_DECODER, dataTrans, instanceId_);
#endif
            if (ChannelStatus::GetInstance()->IsAlive(webChannelId_)) {
                std::shared_ptr<StreamCache> streamCache = StreamCache::GetInstance(webChannelId_);
                streamCache->CacheFrame(dataTrans);
            }
            av_packet_unref(&pkt);
        }
    }
    av_init_packet(&pkt);
    isStop_ = true;
}
} // end ascendFaceRecognition
