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

#include "HdcChannel.h"

#include <cstring>

#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
#include "acl/acl.h"
#include "DataTrans/DataTrans.pb.h"

namespace ascendFaceRecognition {
std::shared_ptr<HdcChannel> HdcChannel::GetInstance()
{
    static std::shared_ptr<HdcChannel> hdcChannelInst = std::make_shared<HdcChannel>();
    return hdcChannelInst;
}

HdcChannel::~HdcChannel()
{
    LogDebug << "free HdcChanel";
}

APP_ERROR HdcChannel::Init(int deviceId, HdcChannelMode hdcMode, int sendChCount, int recvChCount)
{
    deviceId_ = deviceId;
    hdcMode_ = hdcMode;
    isStop_ = false;
    // register HDC Channel
    if (hdcMode == HDC_CHANNEL_HOST) {
        RegisterHdcChannel(HDC_CHANNEL_SEND, sendChCount);
        RegisterHdcChannel(HDC_CHANNEL_RECV, recvChCount);
    } else {
        RegisterHdcChannel(HDC_CHANNEL_RECV, recvChCount);
        RegisterHdcChannel(HDC_CHANNEL_SEND, sendChCount);
    }

    channelCount_ = sendChCount + recvChCount;
    return APP_ERR_OK;
}

void HdcChannel::RegisterHdcChannel(HdcChannelDirection direction, int count)
{
    std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> dataQueue = nullptr;
    for (int i = 0; i < count; i++) {
        dataQueue = std::make_shared<BlockingQueue<std::shared_ptr<void>>>(MODULE_QUEUE_SIZE);
        RegisterHdcChWithQueue(direction, dataQueue);
        if (direction == HDC_CHANNEL_SEND) {
            sendChVec_.push_back(dataQueue); // data queue put into send Channel
        } else {
            recvChVec_.push_back(dataQueue); // data queue put into recv Channel
        }
    }
}

void HdcChannel::RegisterHdcChWithQueue(HdcChannelDirection direction,
    std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> dataQueue)
{
    ThreadData temp;
    temp.isStop = false;
    temp.hdc = nullptr;
    temp.session = nullptr;
    temp.direction = direction;
    temp.dataQueue = dataQueue;
    threadsData_.push_back(temp);
}

void *HdcChannel::ReciveDataProcess(void *args)
{
    auto data = (ThreadData *)args;

    // recv data from hdc and parse, put to out queue
    while (!data->isStop) {
        char *recvBuffer = nullptr;
        uint32_t recvBufferLength = 0;
        int ret = data->hdc->HdcFastRecv(data->session, recvBuffer, recvBufferLength);
        if (ret != APP_ERR_OK) {
            return nullptr;
        }

        LogDebug << "Recv " << recvBufferLength << " data";
        std::shared_ptr<void> receDataT = nullptr;
        std::shared_ptr<DataTrans> dataTrans = std::make_shared<DataTrans>();
        std::string dataStr((char *)recvBuffer, recvBufferLength);
        dataTrans->ParseFromString(dataStr); // parse and Converte data into Strings
        receDataT = std::static_pointer_cast<void>(dataTrans);
        if (receDataT == nullptr) {
            LogWarn << "Fail to deserialize data failed.";
            continue;
        }

        data->dataQueue->Push(std::static_pointer_cast<void>(receDataT), true);
        LogDebug << "Push streamRawData to recDataQueue";
    }

    return nullptr;
}

void *HdcChannel::SendDataProcess(void *args)
{
    auto data = (ThreadData *)args;
    std::shared_ptr<void> sendDataT = nullptr;
    // fetch data from in queue, and send to hdc
    while (!data->isStop) {
        data->dataQueue->Pop(sendDataT);

        std::string serlizeT {};
        std::shared_ptr<DataTrans> dataTrans = std::static_pointer_cast<DataTrans>(sendDataT);
        bool success = dataTrans->SerializeToString(&serlizeT); // serialize data
        if (!success) {
            LogDebug << "fast SerializeToString failed";
            continue;
        }
        // fast send data
        LogDebug << "fast send data size: " << serlizeT.length() << ".";
        int ret = data->hdc->HdcFastSendto(data->session, (char *)serlizeT.c_str(), serlizeT.length());
        if (ret != APP_ERR_OK) {
            LogError << "send data fail, session=" << data->session << ", sendBufferLength=" << serlizeT.length();
            return nullptr;
        }

        usleep(HDC_SEND_INTERVAL);
    }

    return nullptr;
}

APP_ERROR HdcChannel::SendData(int hdcChannelId, std::shared_ptr<void> data)
{
    if (hdcChannelId < 0 || hdcChannelId >= (int)sendChVec_.size()) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    auto queue = sendChVec_[hdcChannelId];
    queue->Push(data, true);
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> HdcChannel::GetRecvChannelVec(int startIndex,
    int endIndex)
{
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> tmpVec = {};
    if (startIndex < 0 || endIndex >= (int)recvChVec_.size()) {
        return tmpVec;
    }

    for (int i = startIndex; i <= endIndex; i++) {
        tmpVec.push_back(recvChVec_[i]);
    }
    return tmpVec;
}

APP_ERROR HdcChannel::Run()
{
    if (hdcMode_ == HDC_CHANNEL_DEVICE) {
        deviceId_ = 0;
    } else if (deviceId_ < 0) {
        LogFatal << "HdcChannel: invalid device id, deviceId=" << deviceId_ << ".";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    hdc_ = std::make_shared<Hdc>(deviceId_, true);

    APP_ERROR ret;
    if (hdcMode_ == HDC_CHANNEL_DEVICE) {
        ret = hdc_->HdcServerCreate(channelCount_, sessions_); // session for in and out queue
    } else {
        ret = hdc_->HdcClientCreate(channelCount_, sessions_); // session for in and out queue
    }

    if (ret != APP_ERR_OK) {
        LogError << "create HDC channel fail";
        return APP_ERR_COMM_INIT_FAIL;
    }

    LogDebug << ((hdcMode_ == HDC_CHANNEL_HOST) ? "client create sucessfully" : "server create sucessfully");

    threadsId_.resize(channelCount_); // determine new length, filling with _Ty() elements

    for (int i = 0; i < channelCount_; ++i) {
        threadsData_[i].hdc = hdc_.get();
        threadsData_[i].session = sessions_[i];
        ret = pthread_create(&threadsId_[i], nullptr,
            (threadsData_[i].direction == HDC_CHANNEL_SEND) ? &(HdcChannel::SendDataProcess) :
                                                              &(HdcChannel::ReciveDataProcess),
            &threadsData_[i]);
        if (ret != 0) {
            LogError << "Create pthread error";
            return APP_ERR_COMM_INIT_FAIL;
        }
    }

    return APP_ERR_OK;
}

APP_ERROR HdcChannel::DeInit()
{
    // stop recv data
    for (int i = 0; i < channelCount_; ++i) {
        threadsData_[i].isStop = true;
        hdc_->HdcStopRecv(sessions_[i]);
    }
    return APP_ERR_OK;
}
}
