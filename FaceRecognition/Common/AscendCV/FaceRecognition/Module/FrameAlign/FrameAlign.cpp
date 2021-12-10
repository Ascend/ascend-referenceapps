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

#include <iostream>
#include "Log/Log.h"
#include "FrameAlign/FrameAlign.h"
#include "FileEx/FileEx.h"
#include "StreamCache/StreamCache.h"
#include "DeliveryChannel/DeliveryChannel.h"
#include "ChannelStatus/ChannelStatus.h"


namespace ascendFaceRecognition {
namespace {
const int CONST_INT_2 = 2;
const float SCALE = 1.2;
} // namespace

FrameAlign::FrameAlign() {}

FrameAlign::~FrameAlign() {}

APP_ERROR FrameAlign::ParseConfig(ConfigParser &configParser)
{
    // load and parse config file
    LogDebug << "FrameAlign[" << instanceId_ << "]: begin to parse config values.";
    APP_ERROR ret = APP_ERR_OK;
    std::string itemCfgStr;

    return ret;
}

APP_ERROR FrameAlign::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "FrameAlign[" << instanceId_ << "]: FrameAlign init start.";
    // initialize member variables
    AssignInitArgs(initArgs);

    int ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "FrameAlign[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }

    // get channel id
    webChannelId_ = initArgs.instanceId + 1;
    LogDebug << "FrameAlign[" << instanceId_ << "]: FrameAlign init success.";
    return APP_ERR_OK;
}

APP_ERROR FrameAlign::DeInit(void)
{
    return APP_ERR_OK;
}

APP_ERROR FrameAlign::Process(std::shared_ptr<void> inputData)
{
    // start threads and receive data from the device.
    std::shared_ptr<DataTrans> dataTrans = std::static_pointer_cast<DataTrans>(inputData);
    std::shared_ptr<FrameDetectDataTrans> frameDetectInfo =
        std::make_shared<FrameDetectDataTrans>(dataTrans->framedetectinfo());
    if (frameDetectInfo.get() == nullptr) {
        LogError << "no frame detect data";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    // get FaceImage information
    if (frameDetectInfo->faces_size() > 0) {
        LogInfo << "Recv Ch #" << frameDetectInfo->channelid() << " Frame #" << frameDetectInfo->frameid() <<
            " FaceCount " << frameDetectInfo->faces_size();
    }
    SendStreamFrame(frameDetectInfo);
    return APP_ERR_OK;
}


APP_ERROR FrameAlign::SendStreamFrame(std::shared_ptr<FrameDetectDataTrans> frameDetectInfo)
{
    if (!ChannelStatus::GetInstance()->IsAlive(webChannelId_)) {
        return APP_ERR_OK;
    }
    std::shared_ptr<StreamCache> streamCache = StreamCache::GetInstance(webChannelId_);
    int frameId = frameDetectInfo->frameid();
    std::shared_ptr<DataTrans> streamDataTrans = streamCache->GetFrame(frameId);
    if (streamDataTrans == nullptr) {
        return APP_ERR_OK;
    }

    std::vector<FaceInfoDataTrans> faceDetectInfoAllVector;
    std::vector<FaceInfoDataTrans> faceDetectInfoMatchVector;
    for (int i = 0; i < frameDetectInfo->faces_size(); i++) {
        faceDetectInfoAllVector.push_back(frameDetectInfo->faces(i));
    }
    // Compare the face in the sixth frame with lastFrameDetect
    for (int i = 0; i < frameDetectInfo->faces_size(); i++) {
        FaceInfoDataTrans faceInfo = frameDetectInfo->faces(i);
        if (lastTrackIds_.find(faceInfo.trackid()) != lastTrackIds_.end()) {
            faceDetectInfoMatchVector.push_back(faceInfo);
        }
    }

    while (lastSentFrameId_ <= frameId) {
        if (!streamCache->IsFrameExist(lastSentFrameId_)) {
            ++lastSentFrameId_;
            continue;
        }
        std::shared_ptr<DataTrans> tmpDataTrans = streamCache->GetFrame(lastSentFrameId_);
        if (lastSentFrameId_ == frameId) {
            SendOneStreamFrame(tmpDataTrans, faceDetectInfoAllVector);
        } else {
            SendOneStreamFrame(tmpDataTrans, faceDetectInfoMatchVector);
        }
        streamCache->ClearFrame(lastSentFrameId_);
        ++lastSentFrameId_;
    }
    return APP_ERR_OK;
}

APP_ERROR FrameAlign::SendOneStreamFrame(const std::shared_ptr<DataTrans> &streamDataTrans,
    const std::vector<FaceInfoDataTrans> &faceDetectInfoVector)
{
    if (streamDataTrans == nullptr) {
        LogError << "streamDataTrans is null";
        return APP_ERR_COMM_FAILURE;
    }
    // process one frame
    std::shared_ptr<VideoDataT> videoDataT = std::make_shared<VideoDataT>();
    videoDataT->channelId = webChannelId_;

    videoDataT->data = std::make_shared<VehicleData>();
    videoDataT->data->set_h264_data(streamDataTrans->streamdata().data());
    videoDataT->data->set_frame_index(streamDataTrans->streamdata().info().frameid());
    videoDataT->data->set_h264_size(streamDataTrans->streamdata().datasize());

    lastTrackIds_.clear(); // clear lastTrackIds container
    VehicleBox *box = nullptr;
    // change the image size.
    for (const auto &faceInfo : faceDetectInfoVector) {
        box = videoDataT->data->add_vehicle_box();
        box->set_x(faceInfo.rect().left() + faceInfo.rect().width() / CONST_INT_2);
        box->set_y(faceInfo.rect().top() + faceInfo.rect().height() / CONST_INT_2);
        box->set_width(faceInfo.rect().width() * SCALE);
        box->set_height(faceInfo.rect().height() * SCALE);
        lastTrackIds_.insert(faceInfo.trackid());
    }
    DeliveryChannel::GetInstance()->SendData(DELIVERY_VIDEO_STREAM_INDEX + webChannelId_, videoDataT);
    return 0;
}
} // namespace ascendFaceRecognition
