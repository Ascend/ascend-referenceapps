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

#include "VideoDecoder.h"
#include <iostream>

#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
#include "FileManager/FileManager.h"
#include "ModelInfer/ModelInfer.h"

using namespace ascendBaseModule;

namespace {
    const int CALLBACK_TRIGGER_TIME = 1000;
}

VideoDecoder::VideoDecoder()
{
    isStop_ = false;
}

VideoDecoder::~VideoDecoder() {}

void VideoDecoder::VideoDecoderCallBack(acldvppStreamDesc *input, acldvppPicDesc *output, void *userdata)
{
    void *dataDev = acldvppGetStreamDescData(input);
    APP_ERROR ret = (APP_ERROR)acldvppFree(dataDev);
    if (ret != APP_ERR_OK) {
        LogError << "fail to free input stream desc dataDev";
    }
    ret = (APP_ERROR)acldvppDestroyStreamDesc(input);
    if (ret != APP_ERR_OK) {
        LogError << "fail to destroy input stream desc";
    }

    auto videoDecoder = static_cast<VideoDecoder*>(userdata);
    if (videoDecoder == nullptr) {
        LogError << "VideoDecoder: user data is nullptr";
        return;
    }

    if (videoDecoder->frameId_ % videoDecoder->skipInterval_ == 0) {
        DvppDataInfo tmp;
        tmp.width = videoDecoder->streamWidth_;
        tmp.height = videoDecoder->streamHeight_;
        tmp.widthStride = DVPP_ALIGN_UP(videoDecoder->streamWidth_, VPC_STRIDE_WIDTH);
        tmp.heightStride = DVPP_ALIGN_UP(videoDecoder->streamHeight_, VPC_STRIDE_HEIGHT);
        tmp.dataSize = static_cast<uint32_t>(acldvppGetPicDescSize(output));
        tmp.data = static_cast<uint8_t*>(acldvppGetPicDescData(output));

        DvppDataInfo out;
        out.width = videoDecoder->resizeWidth_;
        out.height = videoDecoder->resizeHeight_;
        videoDecoder->vpcDvppCommon_->CombineResizeProcess(tmp, out, true, VPC_PT_FIT);

        std::shared_ptr<CommonData> toNext = std::make_shared<CommonData>();
        toNext->eof = false;
        toNext->channelId = videoDecoder->channelId_;
        toNext->srcWidth = videoDecoder->streamWidth_;
        toNext->srcHeight = videoDecoder->streamHeight_;
        toNext->frameId = videoDecoder->frameId_;
        toNext->dvppData = std::move(videoDecoder->vpcDvppCommon_->GetResizedImage());
        videoDecoder->SendToNextModule(MT_ModelInfer, toNext, toNext->channelId);
    }
    videoDecoder->frameId_++;
    acldvppFree(acldvppGetPicDescData(output));
    ret = (APP_ERROR)acldvppDestroyPicDesc(output);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to destroy pic desc";
    }
}

void *VideoDecoder::DecoderThread(void *arg)
{
    auto videoDecoder = static_cast<VideoDecoder *>(arg);
    if (videoDecoder == nullptr) {
        LogError << "arg is nullptr";
        return reinterpret_cast<void*>(-1);
    }

    aclError ret = aclrtSetCurrentContext(videoDecoder->aclContext_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set context, ret = " << ret;
        return reinterpret_cast<void*>(-1);
    }

    LogInfo << "DecoderThread start";
    while (!videoDecoder->stopDecoderThread_) {
        (void)aclrtProcessReport(CALLBACK_TRIGGER_TIME);
    }
    return nullptr;
}

VdecConfig VideoDecoder::GetVdecConfig() const
{
    VdecConfig vdecConfig;
    vdecConfig.inputWidth = streamWidth_;
    vdecConfig.inputHeight = streamHeight_;
    vdecConfig.outFormat = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    vdecConfig.channelId = instanceId_;
    vdecConfig.threadId = decoderThreadId_;
    vdecConfig.callback = &VideoDecoder::VideoDecoderCallBack;
    return vdecConfig;
}

APP_ERROR VideoDecoder::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "Begin to init instance " << initArgs.instanceId;

    AssignInitArgs(initArgs);

    int ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret) <<
            ".";
        return ret;
    }

    int createThreadErr = pthread_create(&decoderThreadId_, nullptr, &VideoDecoder::DecoderThread, (void *)this);
    if (createThreadErr != 0) {
        LogError << "Failed to create thread, err = " << createThreadErr;
        return APP_ERR_ACL_FAILURE;
    }
    LogInfo <<"thread create ID = " << decoderThreadId_;

    ret = aclrtCreateStream(&vpcDvppStream_);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: aclrtCreateStream failed, ret=" << ret << ".";
        return ret;
    }

    vpcDvppCommon_.reset(new DvppCommon(vpcDvppStream_));
    if (vpcDvppCommon_ == nullptr) {
        LogError << "create vpcDvppCommon_ Failed";
        return APP_ERR_COMM_ALLOC_MEM;
    }

    ret = vpcDvppCommon_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "vpcDvppCommon_ Init Failed";
        return ret;
    }

    LogDebug << "VideoDecoder [" << instanceId_ << "] Init success";
    return APP_ERR_OK;
}

APP_ERROR VideoDecoder::ParseConfig(ConfigParser &configParser)
{
    std::string itemCfgStr = moduleName_ + std::string(".resizeWidth");
    APP_ERROR ret = configParser.GetUnsignedIntValue(itemCfgStr, resizeWidth_);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".resizeHeight");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, resizeHeight_);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = std::string("SystemConfig.deviceId");
    ret = configParser.GetIntValue(itemCfgStr, deviceId_);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }

    itemCfgStr = std::string("skipInterval");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, skipInterval_);
    if (ret != APP_ERR_OK) {
        LogError << "VideoDecoder[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    if (skipInterval_ == 0) {
        LogError << "The value of skipInterval_ must be greater than 0";
        return APP_ERR_ACL_FAILURE;
    }

    return ret;
}

APP_ERROR VideoDecoder::CreateVdecDvppCommon(acldvppStreamFormat format)
{
    auto vdecConfig = GetVdecConfig();
    vdecConfig.inFormat = format;
    vdecDvppCommon_.reset(new DvppCommon(vdecConfig));
    if (vdecDvppCommon_ == nullptr) {
        LogError << "create vdecDvppCommon_ Failed";
        return APP_ERR_COMM_ALLOC_MEM;
    }

    APP_ERROR ret = vdecDvppCommon_->InitVdec();
    if (ret != APP_ERR_OK) {
        LogError << "vdecDvppCommon_ InitVdec Failed";
        return ret;
    }
    return ret;
}

APP_ERROR VideoDecoder::Process(std::shared_ptr<void> commonData)
{
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
    if (vdecDvppCommon_ == nullptr) {
        channelId_ = data->channelId;
        streamWidth_ = data->srcWidth;
        streamHeight_ = data->srcHeight;
        APP_ERROR ret = CreateVdecDvppCommon(data->videoFormat);
        if (ret != APP_ERR_OK) {
            LogError << "CreateVdecDvppCommon Failed";
            return ret;
        }
    }

    if (data->eof) {
        APP_ERROR ret = vdecDvppCommon_->VdecSendEosFrame();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to send eos frame, ret = " << ret;
            return ret;
        }
        SendToNextModule(MT_ModelInfer, data, data->channelId);
        return APP_ERR_OK;
    }
    std::shared_ptr<DvppDataInfo> vdecData = std::make_shared<DvppDataInfo>();
    vdecData->dataSize = data->streamData.size;
    vdecData->data = static_cast<uint8_t*>(data->streamData.data.get());

    APP_ERROR ret = vdecDvppCommon_->CombineVdecProcess(vdecData, this);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to do VdecProcess, ret = " << ret;
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR VideoDecoder::DeInit(void)
{
    LogDebug << "VideoDecoder [" << instanceId_ << "] begin to deinit";

    if (vdecDvppCommon_) {
        APP_ERROR ret = vdecDvppCommon_->DeInit();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to deinitialize vdecDvppCommon, ret = " << ret;
            return ret;
        }
    }

    stopDecoderThread_ = true;
    pthread_join(decoderThreadId_, NULL);

    if (vpcDvppCommon_) {
        APP_ERROR ret = vpcDvppCommon_->DeInit();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to deinitialize vpcDvppCommon, ret = " << ret;
            return ret;
        }
    }

    if (vpcDvppStream_) {
        APP_ERROR ret = aclrtDestroyStream(vpcDvppStream_);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy stream, ret = " << ret;
            return ret;
        }
        vpcDvppStream_ = nullptr;
    }

    LogDebug << "VideoDecoder [" << instanceId_ << "] deinit success.";
    return APP_ERR_OK;
}
