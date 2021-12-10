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
#include "Hdc.h"

#include <chrono>
#include <cstring>
#include <algorithm>

#include "Log/Log.h"
namespace {
const int TIME_1000_MS = 1000;
const int DIGIT_0 = 0;
const int DIGIT_1 = 1;
const uint32_t SEND_TIMEOUT = 0;
const uint32_t RECV_TIMEOUT = 0;
}

Hdc::~Hdc()
{
    HdcNormalBufferMapFree();
    HdcFastBufferMapFree();
    HdcSessionMapFree();
    HdcInfoMapFree();
    HdcClientFree();
    HdcServerFree();
}

/*
 * Description: single session pair server create
 * maxFastDataBufferSize is used to set fast data buffer,default value 8M
 * maxFastCtrlBufferSize is used to set fast ctrl buffer,default value 128
 * memFlag:HDC_FLAG_MAP_VA32BIT,dvpp 4G memey;HDC_FLAG_MAP_HUGE:big page map,defual=HDC_FLAG_MAP_HUGE
 */
APP_ERROR Hdc::HdcServerCreate(HdcSession &session, uint64_t maxFastDataBufferSize, uint64_t maxFastCtrlBufferSize)
{
    if (client_ != nullptr) {
        LogError << "the client alrady exist, one hdc entity can not have both server and client";
        return APP_ERR_COMM_FAILURE;
    }
    while (true) {
        LogDebug << "start single session server, deviceId=" << deviceId_;
        // non-blocking
        int ret = drvHdcServerCreate(deviceId_, HDC_SERVICE_TYPE_USER3, &server_);
        if (ret != DRV_ERROR_NONE) {
            LogError << "create HDC single session server on device=" << deviceId_ << " fial, ret=" << ret;
            return APP_ERR_COMM_FAILURE;
        }

        while (true) {
            // Blocking here, wait to client to connect
            ret = drvHdcSessionAccept(server_, &session);
            if (ret != DRV_ERROR_NONE) {
                LogWarn << "create HDC single session fail, deviceId=" << deviceId_ << ", ret=" << ret;
                break;
            }
            ret = drvHdcSetSessionReference(session);
            if (ret != DRV_ERROR_NONE) {
                LogError << "banding session to process fail, ret=" << ret;
                break;
            }
            // default to support normal send and recv
            ret = HdcNormalBufferMalloc(session);
            if (ret != APP_ERR_OK) {
                break;
            }
            // default to support fast send, but you can set not support
            if (enableFastMode_) {
                ret = HdcFastModeServerSourcePrepare(session, maxFastDataBufferSize, maxFastCtrlBufferSize);
                if (ret != APP_ERR_OK) {
                    break;
                }
            }
            LogInfo << "create server and session successfully";
            return APP_ERR_OK;
        }
        HdcNormalBufferMapFree();
        HdcFastBufferMapFree();
        HdcSessionMapFree();
        HdcInfoMapFree();
        HdcServerFree();
    }
}

APP_ERROR Hdc::HdcNormalBufferMalloc(HdcSession &session)
{
    // default to support normal send and recv
    LogDebug << "start to malloc normal message head";
    struct drvHdcMsg *sendBuffer = nullptr;
    int ret = drvHdcAllocMsg(session, &sendBuffer, DIGIT_1);
    if (ret != DRV_ERROR_NONE) {
        LogError << "alloc send message head fail, ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    normalSendInfo_[session] = sendBuffer;
    LogDebug << "malloc normal message head succesfully";
    LogDebug << "normalSendInfo_[" << session << "]=" << normalSendInfo_[session];
    struct drvHdcMsg *recvBuffer = nullptr;
    ret = drvHdcAllocMsg(session, &recvBuffer, DIGIT_1);
    if (ret != DRV_ERROR_NONE) {
        LogError << "alloc recv message head fail, ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    normalRecvInfo_[session] = recvBuffer;
    LogDebug << "malloc normal message head succesfully";
    LogDebug << "normalRecvInfo_[" << session << "]=" << normalRecvInfo_[session];
    return APP_ERR_OK;
}

APP_ERROR Hdc::HdcFastModeServerSourcePrepare(HdcSession &session, uint64_t maxFastDataBufferSize,
    uint64_t maxFastCtrlBufferSize)
{
    int ret = HdcFastBufferMalloc(session, maxFastDataBufferSize, maxFastCtrlBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "set single session fast send buff fail";
        return APP_ERR_COMM_OUT_OF_MEM;
    }
    ret = HdcReplyBufferCapability(session);
    if (ret != APP_ERR_OK) {
        LogError << "server reply to client fail, session=" << session << ", ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    return APP_ERR_OK;
}

APP_ERROR Hdc::HdcFastModeClientSourcePrepare(HdcSession &session, uint64_t maxFastDataBufferSize,
    uint64_t maxFastCtrlBufferSize)
{
    int ret = HdcFastBufferMalloc(session, maxFastDataBufferSize, maxFastCtrlBufferSize);
    if (ret != APP_ERR_OK) {
        LogError << "set single session fast send buff fail";
        return APP_ERR_COMM_OUT_OF_MEM;
    }
    ret = HdcStartToExchangeBufferCapability(session);
    if (ret != APP_ERR_OK) {
        LogError << "server reply to client fail, session=" << session << ", ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    return APP_ERR_OK;
}

APP_ERROR Hdc::HdcFastBufferMalloc(const HdcSession &session, const uint64_t maxDataBufferSize,
    const uint64_t maxCtrlBufferSize)
{
    LogDebug << "start to malloc fast buffer";
    char *sendDataBuf = static_cast<char *>(
        drvHdcMallocEx(HDC_MEM_TYPE_TX_DATA, nullptr, DIGIT_0, maxDataBufferSize, deviceId_, DIGIT_0));
    drvHdcDmaMap(HDC_MEM_TYPE_TX_DATA, sendDataBuf, deviceId_);
    char *recvDataBuf = static_cast<char *>(
        drvHdcMallocEx(HDC_MEM_TYPE_RX_DATA, nullptr, DIGIT_0, maxDataBufferSize, deviceId_, DIGIT_0));
    drvHdcDmaMap(HDC_MEM_TYPE_RX_DATA, recvDataBuf, deviceId_);
    char *sendCtrlBuf = static_cast<char *>(
        drvHdcMallocEx(HDC_MEM_TYPE_TX_CTRL, nullptr, DIGIT_0, maxCtrlBufferSize, deviceId_, DIGIT_0));
    drvHdcDmaMap(HDC_MEM_TYPE_TX_CTRL, sendCtrlBuf, deviceId_);
    char *recvCtrlBuf = static_cast<char *>(
        drvHdcMallocEx(HDC_MEM_TYPE_RX_CTRL, nullptr, DIGIT_0, maxCtrlBufferSize, deviceId_, DIGIT_0));
    drvHdcDmaMap(HDC_MEM_TYPE_RX_CTRL, recvCtrlBuf, deviceId_);
    if (sendDataBuf == nullptr || recvDataBuf == nullptr || sendCtrlBuf == nullptr || recvCtrlBuf == nullptr) {
        LogError << "malloc hdc fast channel memery fail";
        LogError << "maxDataBufferSize=" << maxDataBufferSize << ", maxCtrlBufferSize=" << maxCtrlBufferSize;
        HdcFastBufferFree(sendDataBuf, recvDataBuf, sendCtrlBuf, recvCtrlBuf);
        return APP_ERR_COMM_OUT_OF_MEM;
    }
    // init local buffer map
    fastBufferInfo_.insert(std::make_pair(session, HdcBuffer()));
    fastBufferInfo_[session].sendDataBuf = reinterpret_cast<uint64_t>(sendDataBuf);
    fastBufferInfo_[session].recvDataBuf = reinterpret_cast<uint64_t>(recvDataBuf);
    fastBufferInfo_[session].sendCtrlBuf = reinterpret_cast<uint64_t>(sendCtrlBuf);
    fastBufferInfo_[session].recvCtrlBuf = reinterpret_cast<uint64_t>(recvCtrlBuf);
    fastBufferInfo_[session].datalen = maxDataBufferSize;
    fastBufferInfo_[session].ctrllen = maxCtrlBufferSize;
    LogDebug << std::hex << "fastBufferInfo_[" << session << "].sendDataBuf=" << fastBufferInfo_[session].sendDataBuf;
    LogDebug << std::hex << "fastBufferInfo_[" << session << "].recvDataBuf=" << fastBufferInfo_[session].recvDataBuf;
    LogDebug << std::hex << "fastBufferInfo_[" << session << "].sendCtrlBuf=" << fastBufferInfo_[session].sendCtrlBuf;
    LogDebug << std::hex << "fastBufferInfo_[" << session << "].recvCtrlBuf=" << fastBufferInfo_[session].recvCtrlBuf;
    LogDebug << "fastBufferInfo_[" << session << "].datalen=" << fastBufferInfo_[session].datalen;
    LogDebug << "fastBufferInfo_[" << session << "].ctrllen=" << fastBufferInfo_[session].ctrllen;
    // init fast send map
    fastSendInfo_.insert(std::make_pair(session, drvHdcFastSendMsg()));
    fastSendInfo_[session].srcDataAddr = (uint64_t)sendDataBuf;
    fastSendInfo_[session].dstDataAddr = DIGIT_0;
    fastSendInfo_[session].srcCtrlAddr = (uint64_t)sendCtrlBuf;
    fastSendInfo_[session].dstCtrlAddr = DIGIT_0;
    fastSendInfo_[session].dataLen = DIGIT_0;
    fastSendInfo_[session].ctrlLen = DIGIT_0;

    // init fast recv map
    fastRecvInfo_.insert(std::make_pair(session, drvHdcFastRecvMsg()));
    fastRecvInfo_[session].dataAddr = (uint64_t)recvDataBuf;
    fastRecvInfo_[session].ctrlAddr = (uint64_t)recvCtrlBuf;
    fastRecvInfo_[session].dataLen = DIGIT_0;
    fastRecvInfo_[session].ctrlLen = DIGIT_0;

    LogDebug << "malloc fast buffer successfully";
    return APP_ERR_OK;
}

void Hdc::HdcFastBufferFree(char *&sendDataBuf, char *&recvDataBuf, char *&sendCtrlBuf, char *&recvCtrlBuf)
{
    if (sendDataBuf != nullptr) {
        LogDebug << "free sendDataBuf";
        int ret = drvHdcFreeEx(HDC_MEM_TYPE_TX_DATA, sendDataBuf);
        if (ret != DRV_ERROR_NONE) {
            LogError << "free sendDataBuf fail, ret=" << ret;
        }
    }
    if (recvDataBuf != nullptr) {
        LogDebug << "free recvDataBuf";
        int ret = drvHdcFreeEx(HDC_MEM_TYPE_RX_DATA, recvDataBuf);
        if (ret != DRV_ERROR_NONE) {
            LogError << "free recvDataBuf fail, ret=" << ret;
        }
    }
    if (sendCtrlBuf != nullptr) {
        LogDebug << "free sendCtrlBuf";
        int ret = drvHdcFreeEx(HDC_MEM_TYPE_TX_CTRL, sendCtrlBuf);
        if (ret != DRV_ERROR_NONE) {
            LogError << "free sendCtrlBuf fail, ret=" << ret;
        }
    }
    if (recvCtrlBuf != nullptr) {
        LogDebug << "free recvCtrlBuf";
        int ret = drvHdcFreeEx(HDC_MEM_TYPE_RX_CTRL, recvCtrlBuf);
        if (ret != DRV_ERROR_NONE) {
            LogError << "free recvCtrlBuf fail, ret=" << ret;
        }
    }
    return;
}

// first to recv then to send
APP_ERROR Hdc::HdcReplyBufferCapability(HdcSession &session)
{
    LogDebug << "ready to reply capability to peer";
    char *message = nullptr;
    uint32_t recvMsgLength = 0;
    int ret = HdcNormalRecv(session, message, recvMsgLength);
    if (ret != APP_ERR_OK || recvMsgLength != SIZE_OF_HDC_BUFFER) {
        LogError << "get recv message from recv message head fail, ret=" << ret << ", recvMsgLength=" <<
            recvMsgLength << ", SIZE_OF_HDC_BUFFER=" << SIZE_OF_HDC_BUFFER;
        return APP_ERR_COMM_FAILURE;
    }
    // save fast send buffer information
    auto peerBuffer = reinterpret_cast<HdcBuffer *>(message);
    fastSendInfo_[session].dstDataAddr = peerBuffer->recvDataBuf;
    fastSendInfo_[session].dstCtrlAddr = peerBuffer->recvCtrlBuf;
    peerMaxDataLength_[session] = peerBuffer->datalen;
    peerMaxCtrlLength_[session] = peerBuffer->ctrllen;
    LogDebug << "recv peer buffer information: session=" << session << std::hex << ", dstDataAddr=" <<
        fastSendInfo_[session].dstDataAddr << ", dstCtrlAddr=" << fastSendInfo_[session].dstCtrlAddr << std::dec <<
        ", peerMaxDataLength=" << peerMaxDataLength_[session] << ", peerMaxCtrlLength=" << peerMaxCtrlLength_[session];

    // send local buffer capability
    message = reinterpret_cast<char *>(&(fastBufferInfo_[session]));
    ret = HdcNormalSendto(session, message, SIZE_OF_HDC_BUFFER);
    if (ret != APP_ERR_OK) {
        return APP_ERR_COMM_FAILURE;
    }
    LogDebug << "send local fast buffer information to peer successfully";

    LogDebug << "reply capability to peer successfully";
    return APP_ERR_OK;
}

// to free drvHdcMsg map
void Hdc::HdcNormalBufferMapFree()
{
    for (auto it : normalSendInfo_) {
        if (it.second != nullptr) {
            LogDebug << "free normal send header memery, normalSendInfo_[" << it.first << "]=" << it.second;
            int ret = drvHdcFreeMsg(it.second);
            if (ret != DRV_ERROR_NONE) {
                LogError << "free hdc buffer head fail, ret=" << ret;
            }
        }
        it.second = nullptr;
    }
    for (auto it : normalRecvInfo_) {
        if (it.second != nullptr) {
            LogDebug << "free normal recv header memery, normalRecvInfo_[" << it.first << "]=" << it.second;
            int ret = drvHdcFreeMsg(it.second);
            if (ret != DRV_ERROR_NONE) {
                LogError << "free hdc buffer head fail, ret=" << ret;
            }
        }
        it.second = nullptr;
    }
    return;
}

// to free drvHdcMsg map
void Hdc::HdcFastBufferMapFree()
{
    if (!enableFastMode_) {
        return;
    }
    for (auto it : fastBufferInfo_) {
        char *sendDataBuf = reinterpret_cast<char *>(it.second.sendDataBuf);
        char *recvDataBuf = reinterpret_cast<char *>(it.second.recvDataBuf);
        char *sendCtrlBuf = reinterpret_cast<char *>(it.second.sendCtrlBuf);
        char *recvCtrlBuf = reinterpret_cast<char *>(it.second.recvCtrlBuf);
        HdcFastBufferFree(sendDataBuf, recvDataBuf, sendCtrlBuf, recvCtrlBuf);
        it.second.sendDataBuf = DIGIT_0;
        it.second.recvDataBuf = DIGIT_0;
        it.second.sendCtrlBuf = DIGIT_0;
        it.second.recvCtrlBuf = DIGIT_0;
    }
    return;
}

// to free map session
void Hdc::HdcSessionMapFree()
{
    for (auto it : normalSendInfo_) {
        if (it.first != nullptr) {
            LogDebug << "close session " << it.first;
            int ret = drvHdcSessionClose(it.first);
            if (ret != DRV_ERROR_NONE) {
                LogError << "free hdc session fail, session=" << it.first << "ret=" << ret;
            }
        }
    }
    return;
}

// all info map clear
void Hdc::HdcInfoMapFree()
{
    if (!fastBufferInfo_.empty()) {
        fastBufferInfo_.clear();
        LogDebug << "clear fastBufferInfo_";
    }
    if (!fastSendInfo_.empty()) {
        fastSendInfo_.clear();
        LogDebug << "clear fastSendInfo_";
    }
    if (!fastRecvInfo_.empty()) {
        fastRecvInfo_.clear();
        LogDebug << "clear fastRecvInfo_";
    }
    if (!normalSendInfo_.empty()) {
        normalSendInfo_.clear();
        LogDebug << "clear normalSendInfo_";
    }
    if (!normalRecvInfo_.empty()) {
        normalRecvInfo_.clear();
        LogDebug << "clear normalRecvInfo_";
    }
    if (!peerMaxDataLength_.empty()) {
        peerMaxDataLength_.clear();
        LogDebug << "clear peerMaxDataLength_";
    }
    if (!peerMaxCtrlLength_.empty()) {
        peerMaxCtrlLength_.clear();
        LogDebug << "clear peerMaxCtrlLength_";
    }
    return;
}

void Hdc::HdcServerFree()
{
    if (server_ != nullptr) {
        LogDebug << "destroy server_=" << server_;
        int ret = drvHdcServerDestroy(server_);
        if (ret != DRV_ERROR_NONE) {
            LogError << "free HDC server fail, deviceId=" << deviceId_ << ", ret=" << ret;
            return;
        }
        server_ = nullptr;
    }
    return;
}

/*
 * Description: multi sessions pair
 * sessionCount?ready to create session num
 * sessions: list of sessions when the application is successful
 * maxFastDataBufferSize: fast mode, max data buffer size
 * maxFastCtrlBufferSize: fast mode, max ctrl buffer size
 * memFlag:HDC_FLAG_MAP_VA32BIT,dvpp 4G memey;HDC_FLAG_MAP_HUGE:big page map
 */
APP_ERROR Hdc::HdcServerCreate(uint32_t sessionCount, std::vector<HdcSession> &sessions, uint64_t maxFastDataBufferSize,
    uint64_t maxFastCtrlBufferSize)
{
    if (client_ != nullptr) {
        LogError << "the client alrady exist, one hdc entity can not have both server and client";
        return APP_ERR_COMM_FAILURE;
    }
    while (true) {
        LogInfo << "start multi sessions server, deviceId=" << deviceId_;
        // non-blocking
        int ret = drvHdcServerCreate(deviceId_, HDC_SERVICE_TYPE_USER3, &server_);
        if (ret != DRV_ERROR_NONE) {
            LogWarn << "create HDC multi session server on device=" << deviceId_ << " fail, ret=" << ret;
            return APP_ERR_COMM_FAILURE;
        }
        for (uint32_t i = 0; i < sessionCount; ++i) {
            HdcSession session = nullptr;
            // Blocking here, wait to client to connect
            ret = drvHdcSessionAccept(server_, &session);
            if (ret != DRV_ERROR_NONE) {
                LogWarn << "create HDC session fail, sessionCount=" << sessionCount << ", deviceId=" << deviceId_ <<
                    ", ret=" << ret;
                break;
            }
            ret = drvHdcSetSessionReference(session);
            if (ret != DRV_ERROR_NONE) {
                LogError << "banding session to process fail, ret=" << ret;
                break;
            }
            ret = HdcNormalBufferMalloc(session);
            if (ret != APP_ERR_OK) {
                break;
            }
            if (enableFastMode_) {
                ret = HdcFastModeServerSourcePrepare(session, maxFastDataBufferSize, maxFastCtrlBufferSize);
                if (ret != APP_ERR_OK) {
                    break;
                }
            }
            sessions.push_back(session);
        }
        // error process
        if (ret != DRV_ERROR_NONE) {
            HdcNormalBufferMapFree();
            HdcFastBufferMapFree();
            HdcSessionMapFree();
            HdcInfoMapFree();
            HdcServerFree();
            sessions.clear();
            continue;
        }
        LogInfo << "create server and sessions successfully";
        return APP_ERR_OK;
    }
}

/*
 * Description: single session pair server create
 * maxFastDataBufferSize is used to set fast data buffer,default value 8M
 * maxFastCtrlBufferSize is used to set fast ctrl buffer,default value 128
 * memFlag:HDC_FLAG_MAP_VA32BIT,dvpp 4G memey;HDC_FLAG_MAP_HUGE:big page map,defual=HDC_FLAG_MAP_HUGE
 */
APP_ERROR Hdc::HdcClientCreate(HdcSession &session, uint64_t maxFastDataBufferSize, uint64_t maxFastCtrlBufferSize)
{
    if (server_ != nullptr) {
        LogError << "the server alrady exist, one hdc entity can not have both server and client";
        return APP_ERR_COMM_FAILURE;
    }
    while (true) {
        LogInfo << "start single session clinet, deviceId=" << deviceId_;
        // non-blocking
        int ret = drvHdcClientCreate(&client_, DIGIT_1, HDC_SERVICE_TYPE_USER3, DIGIT_0);
        if (ret != DRV_ERROR_NONE) {
            LogError << "client create fail, ret=" << ret;
            return APP_ERR_COMM_FAILURE;
        }
        while (true) {
            // Blocking here, wait to client to connect
            ret = drvHdcSessionConnect(DIGIT_0, deviceId_, client_, &session);
            if (ret != DRV_ERROR_NONE) {
                LogWarn << "client session connect to server fail, ret=" << ret;
                std::this_thread::sleep_for(std::chrono::milliseconds(TIME_1000_MS));
                continue;
            }
            ret = drvHdcSetSessionReference(session);
            if (ret != DRV_ERROR_NONE) {
                LogError << "banding session to process fail, ret=" << ret;
                break;
            }
            // default to support normal send and recv
            ret = HdcNormalBufferMalloc(session);
            if (ret != APP_ERR_OK) {
                LogError << "alloc normal buffer head fail, ret=" << ret;
                break;
            }
            // default to support fast send, but you can set not support
            if (enableFastMode_) {
                ret = HdcFastModeClientSourcePrepare(session, maxFastDataBufferSize, maxFastCtrlBufferSize);
                if (ret != APP_ERR_OK) {
                    break;
                }
            }
            LogInfo << "create client and session successfully";
            return APP_ERR_OK;
        }
        HdcNormalBufferMapFree();
        HdcFastBufferMapFree();
        HdcSessionMapFree();
        HdcInfoMapFree();
        HdcClientFree();
    }
}

// Both server and cleient can start to exchange the buffer information
APP_ERROR Hdc::HdcStartToExchangeBufferCapability(HdcSession &session)
{
    LogDebug << "start to exchange capability to peer";
    char *message = (char *)&(fastBufferInfo_[session]);
    int ret = HdcNormalSendto(session, message, SIZE_OF_HDC_BUFFER);
    if (ret != APP_ERR_OK) {
        return APP_ERR_COMM_FAILURE;
    }
    LogDebug << "send fast buffer infor to peer successfully";

    uint32_t recvMsgLength = 0;
    message = nullptr;
    ret = HdcNormalRecv(session, message, recvMsgLength);
    if (ret != APP_ERR_OK || recvMsgLength != SIZE_OF_HDC_BUFFER) {
        LogError << "get recv message from recv message head fail, ret=" << ret << ", recvMsgLength=" <<
            recvMsgLength << ", SIZE_OF_HDC_BUFFER=" << SIZE_OF_HDC_BUFFER;
        return APP_ERR_COMM_FAILURE;
    }
    // save fast send buffer information
    auto peerBuffer = reinterpret_cast<HdcBuffer *>(message);
    fastSendInfo_[session].dstDataAddr = peerBuffer->recvDataBuf;
    fastSendInfo_[session].dstCtrlAddr = peerBuffer->recvCtrlBuf;
    peerMaxDataLength_[session] = peerBuffer->datalen;
    peerMaxCtrlLength_[session] = peerBuffer->ctrllen;
    LogDebug << "recv peer buffer information successfully: session=" << std::hex << session << ", dstDataAddr=" <<
        fastSendInfo_[session].dstDataAddr << ", dstCtrlAddr=" << fastSendInfo_[session].dstCtrlAddr << std::dec <<
        ", peerMaxDataLength_=" << peerMaxDataLength_[session] << ", peerMaxCtrlLength_=" <<
        peerMaxCtrlLength_[session];

    LogDebug << "change buffer information successfully";
    return APP_ERR_OK;
}

void Hdc::HdcClientFree()
{
    if (client_ != nullptr) {
        LogDebug << "destroy client_=" << client_;
        int ret = drvHdcClientDestroy(client_);
        if (ret != DRV_ERROR_NONE) {
            LogError << "free HDC server fail, deviceId=" << deviceId_ << ", ret=" << ret;
            return;
        }
        client_ = nullptr;
    }
    return;
}

/*
 * Description: multi sessions pair
 * sessionCount?ready to create session num
 * sessions: list of sessions when the application is successful
 * maxFastDataBufferSize: fast mode, max data buffer size
 * maxFastCtrlBufferSize: fast mode, max ctrl buffer size
 * memFlag:HDC_FLAG_MAP_VA32BIT,dvpp 4G memey;HDC_FLAG_MAP_HUGE:big page map
 */
APP_ERROR Hdc::HdcClientCreate(uint32_t sessionCount, std::vector<HdcSession> &sessions, uint64_t maxFastDataBufferSize,
    uint64_t maxFastCtrlBufferSize)
{
    if (server_ != nullptr) {
        LogError << "the server alrady exist, one hdc entity can not have both server and client";
        return APP_ERR_COMM_FAILURE;
    }
    while (true) {
        LogInfo << "start multi sessions clinet, deviceId=" << deviceId_;
        // non-blocking
        int ret = drvHdcClientCreate(&client_, sessionCount, HDC_SERVICE_TYPE_USER3, DIGIT_0);
        if (ret != DRV_ERROR_NONE) {
            LogWarn << "client create fail, ret=" << ret;
            return APP_ERR_COMM_FAILURE;
        }
        for (uint32_t i = 0; i < sessionCount; ++i) {
            HdcSession session = nullptr;
            // Blocking here, wait to client to connect
            ret = drvHdcSessionConnect(DIGIT_0, deviceId_, client_, &session);
            if (ret != DRV_ERROR_NONE) {
                LogWarn << "client session connect to server fail, ret=" << ret;
                std::this_thread::sleep_for(std::chrono::milliseconds(TIME_1000_MS));
                break;
            }
            ret = drvHdcSetSessionReference(session);
            if (ret != DRV_ERROR_NONE) {
                LogError << "banding session to process fail, ret=" << ret;
                break;
            }
            // default to support normal send and recv
            ret = HdcNormalBufferMalloc(session);
            if (ret != APP_ERR_OK) {
                break;
            }
            // default to support fast send, but you can set not support
            if (enableFastMode_) {
                ret = HdcFastModeClientSourcePrepare(session, maxFastDataBufferSize, maxFastCtrlBufferSize);
                if (ret != APP_ERR_OK) {
                    break;
                }
            }
            sessions.push_back(session);
        }
        // error process
        if (ret != DRV_ERROR_NONE) {
            HdcNormalBufferMapFree();
            HdcFastBufferMapFree();
            HdcSessionMapFree();
            HdcInfoMapFree();
            HdcClientFree();
            sessions.clear();
            continue;
        }
        LogInfo << "create client and sessions successfully";
        return APP_ERR_OK;
    }
}

// sendto without one more memery copy, first malloc buffer, then send
// blockFlag show blocking or not, default blocking(value 0), non-blocking is HDC_FLAG_NOWAIT.
APP_ERROR Hdc::HdcFastSendto(HdcSession &session, uint32_t dataBufferLength, uint32_t ctrlBufferLength)
{
    if (!enableFastMode_) {
        LogError << "sendto fail, current hdc class does not support fast mode!";
        return APP_ERR_COMM_FAILURE;
    }
    if (dataBufferLength > peerMaxDataLength_[session] || ctrlBufferLength > peerMaxCtrlLength_[session]) {
        LogError << "the data buffer length(" << dataBufferLength << ") is out of range(" <<
            peerMaxDataLength_[session] << ")"
                 << ", or the ctrl buffer length(" << ctrlBufferLength << ") is out of range(" <<
            peerMaxCtrlLength_[session] << ")";
        return APP_ERR_COMM_OUT_OF_MEM;
    }
    struct drvHdcFastSendMsg &fastSendMsg = fastSendInfo_[session];
    fastSendMsg.dataLen = (uint32_t)dataBufferLength;
    fastSendMsg.ctrlLen = (uint32_t)ctrlBufferLength;
    int ret = halHdcFastSend(session, fastSendMsg, DIGIT_0, SEND_TIMEOUT);
    if (ret != DRV_ERROR_NONE) {
        LogError << "fast send message fail, ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    return APP_ERR_OK;
}

// blockFlag show blocking or not, default blocking(value 0), non-blocking is HDC_FLAG_NOWAIT.
// here will be one more memry copy
APP_ERROR Hdc::HdcFastSendto(HdcSession &session, char *dataBuffer, uint32_t dataBufferLength, char *ctrlBuffer,
    uint32_t ctrlBufferLength)
{
    if (!enableFastMode_) {
        LogError << "send fail, current hdc object does not support fast mode!";
        return APP_ERR_COMM_FAILURE;
    }
    if (dataBufferLength > peerMaxDataLength_[session] || ctrlBufferLength > peerMaxCtrlLength_[session]) {
        LogError << "the data buffer length(" << dataBufferLength << ") is out of range(" <<
            peerMaxDataLength_[session] << ")"
                 << ", or the ctrl buffer length(" << ctrlBufferLength << ") is out of range(" <<
            peerMaxCtrlLength_[session] << ")";
        return APP_ERR_COMM_OUT_OF_MEM;
    }
    if (dataBufferLength == 0 || (ctrlBuffer != nullptr && ctrlBufferLength == 0)) {
        LogError << "input error, dataBufferLength=" << dataBufferLength << ", ctrlBufferLength=" << ctrlBufferLength;
        return APP_ERR_COMM_INVALID_PARAM;
    }
    struct drvHdcFastSendMsg &fastSendMsg = fastSendInfo_[session];
    LogDebug << std::hex << "session=" << session << ", srcDataAddr=" << fastSendMsg.srcDataAddr << ", dstDataAddr=" <<
        fastSendMsg.dstDataAddr << ", srcCtrlAddr=" << fastSendMsg.srcCtrlAddr << ", dstCtrlAddr=" <<
        fastSendMsg.dstCtrlAddr;
    LogDebug << ", dataLen=" << fastSendMsg.dataLen << ", ctrlLen=" << fastSendMsg.ctrlLen;
    char *hdcDataBuffer = reinterpret_cast<char *>(fastSendMsg.srcDataAddr);
    std::copy(dataBuffer, dataBuffer + dataBufferLength, hdcDataBuffer);
    fastSendMsg.dataLen = dataBufferLength;
    fastSendMsg.ctrlLen = 0;
    if (ctrlBuffer) {
        char *hdcCtrlBuffer = reinterpret_cast<char *>(fastSendMsg.srcCtrlAddr);
        std::copy(ctrlBuffer, ctrlBuffer + ctrlBufferLength, hdcCtrlBuffer);
        fastSendMsg.ctrlLen = ctrlBufferLength;
    }
    int ret = halHdcFastSend(session, fastSendMsg, DIGIT_0, SEND_TIMEOUT);
    if (ret != DRV_ERROR_NONE) {
        LogError << "fast send message fail, ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    LogDebug << "fast send successfully";
    return APP_ERR_OK;
}

// normal send
APP_ERROR Hdc::HdcNormalSendto(HdcSession &session, char *sendBuffer, uint32_t dataBufferLength)
{
    if (sendBuffer == nullptr) {
        LogError << "sendBuffer is null";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    struct drvHdcMsg *&sendMsgHead = normalSendInfo_[session];
    int ret = drvHdcReuseMsg(sendMsgHead);
    if (ret != DRV_ERROR_NONE) {
        LogError << "reuse sendMsgHead fail, ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    ret = drvHdcAddMsgBuffer(sendMsgHead, sendBuffer, dataBufferLength);
    if (ret != DRV_ERROR_NONE) {
        LogError << "add normal buffer information to head fail, session=" << session << std::hex << ", sendMsgHead=" <<
            sendMsgHead;
        LogError << "dataBufferLength=" << dataBufferLength << ", ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    ret = halHdcSend(session, sendMsgHead, DIGIT_0, SEND_TIMEOUT);
    if (ret != DRV_ERROR_NONE) {
        LogError << "normal send buffer information to head fail, ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    return APP_ERR_OK;
}

/*
 * Description: blocking to recv, the memery of recv cannot be used for long-term storage,
 * the return memery will be rewrited when the next package coming.
 */
APP_ERROR Hdc::HdcFastRecv(HdcSession &session, char *&dataBuffer, uint32_t &dataBufferLength)
{
    // fast recv data
    struct drvHdcFastRecvMsg &fastRecvMsg = fastRecvInfo_[session];
    LogDebug << "ready to recv, session=" << session << std::hex << ", dataAddr=" << fastRecvMsg.dataAddr;
    int ret = halHdcFastRecv(session, &fastRecvMsg, DIGIT_0, RECV_TIMEOUT);
    if (ret != DRV_ERROR_NONE) {
        if (ret == DRV_ERROR_SOCKET_CLOSE) {
            LogInfo << "hdc channel closed.";
            return APP_ERR_COMM_CONNECTION_CLOSE;
        }

        LogError << "fast recv data fail, ret=" << ret;
        dataBuffer = nullptr;
        return APP_ERR_COMM_FAILURE;
    }
    dataBuffer = reinterpret_cast<char *>(fastRecvMsg.dataAddr);
    dataBufferLength = fastRecvMsg.dataLen;
    return APP_ERR_OK;
}

/*
 * Description: blocking to recv, the memery of recv cannot be used for long-term storage,
 * the return memery will be rewrited when the next package coming.
 */
APP_ERROR Hdc::HdcFastRecv(HdcSession &session, char *&dataBuffer, uint32_t &dataBufferLength, char *&ctrlBuffer,
    int &ctrlBufferLength)
{
    // fast recv data
    struct drvHdcFastRecvMsg &fastRecvMsg = fastRecvInfo_[session];
    LogDebug << "ready to recv, session=" << session << std::hex << ", dataAddr=" << fastRecvMsg.dataAddr <<
        ", ctrlAddr=" << fastRecvMsg.ctrlAddr;
    int ret = halHdcFastRecv(session, &fastRecvMsg, DIGIT_0, RECV_TIMEOUT);
    if (ret != DRV_ERROR_NONE) {
        LogError << "fast recv data fail, ret=" << ret;
        dataBuffer = nullptr;
        ctrlBuffer = nullptr;
        return APP_ERR_COMM_FAILURE;
    }
    dataBuffer = reinterpret_cast<char *>(fastRecvMsg.dataAddr);
    dataBufferLength = fastRecvMsg.dataLen;
    ctrlBuffer = reinterpret_cast<char *>(fastRecvMsg.ctrlAddr);
    ctrlBufferLength = fastRecvMsg.ctrlLen;
    return APP_ERR_OK;
}

/*
 * Description: It is use to normal recieve message.
 * recvBuffer should be malloc already, recv message will story in it.
 * recvBufferLength for both input and output, input means the recvBuffer max leangth,
 * the output of recvBufferLength means recive message length.
 */
APP_ERROR Hdc::HdcNormalRecv(HdcSession &session, char *&recvBuffer, uint32_t &recvBufferLength)
{
    struct drvHdcMsg *&recvMsgHead = normalRecvInfo_[session];
    int ret = drvHdcReuseMsg(recvMsgHead);
    if (ret != DRV_ERROR_NONE) {
        LogError << "reuse recvMsgHead fail, ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    ret = halHdcRecv(session, recvMsgHead, recvBufferLength, DIGIT_0, &useless_, RECV_TIMEOUT);
    if (ret != DRV_ERROR_NONE) {
        LogError << "normal recv message fail, ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    int recvLength = 0;
    ret = drvHdcGetMsgBuffer(recvMsgHead, DIGIT_0, &recvBuffer, &recvLength);
    if (ret != DRV_ERROR_NONE) {
        LogError << "get recv message from recv message head fail, ret=" << ret;
        return APP_ERR_COMM_FAILURE;
    }
    recvBufferLength = recvLength;
    return APP_ERR_OK;
}

APP_ERROR Hdc::HdcStopRecv(HdcSession &session)
{
    if (session != nullptr) {
        LogDebug << "close session " << session;
        int ret = drvHdcSessionClose(session);
        if (ret != DRV_ERROR_NONE) {
            LogError << "close hdc session fail, session=" << session << "ret=" << ret;
        }
        session = nullptr;
    }
    return APP_ERR_OK;
}