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

#ifndef INC_HDC_H
#define INC_HDC_H

#include <cstdint>
#include <map>
#include <thread>
#include <vector>

#include "ascend_hal.h"

#include "ErrorCode/ErrorCode.h"

const int HDC_SEND_INTERVAL =
    5000; // for the same session, receive memery will be rewrited when the next package coming.

// to ignored the differences between 32-bit and 64-bit
struct HdcBuffer {
    uint64_t sendDataBuf;
    uint64_t recvDataBuf;
    uint64_t sendCtrlBuf;
    uint64_t recvCtrlBuf;
    uint64_t datalen;
    uint64_t ctrllen;
};
#define SIZE_OF_HDC_BUFFER sizeof(HdcBuffer)
using HdcSession = HDC_SESSION;

// a entity of HDC class is a pair of server and clients.
class Hdc {
public:
    Hdc(int deviceId, bool enableFastMode)
        : deviceId_(deviceId), server_(nullptr), client_(nullptr), enableFastMode_(enableFastMode), useless_(0) {};
    ~Hdc();

    APP_ERROR HdcServerCreate(HdcSession &session, uint64_t maxFastDataBufferSize = 8388608,
        uint64_t maxFastCtrlBufferSize = 128);
    APP_ERROR HdcServerCreate(uint32_t sessionCount, std::vector<HdcSession> &sessions,
        uint64_t maxFastDataBufferSize = 8388608, uint64_t maxFastCtrlBufferSize = 128);
    APP_ERROR HdcClientCreate(HdcSession &session, uint64_t maxFastDataBufferSize = 8388608,
        uint64_t maxFastCtrlBufferSize = 128);
    APP_ERROR HdcClientCreate(uint32_t sessionCount, std::vector<HdcSession> &sessions,
        uint64_t maxFastDataBufferSize = 8388608, uint64_t maxFastCtrlBufferSize = 128);
    APP_ERROR HdcFastSendto(HdcSession &session, uint32_t dataBufferLength, uint32_t ctrlBufferLength = 0);
    // fast send, sendto with one more memery copy, you can use one api to send you message
    APP_ERROR HdcFastSendto(HdcSession &session, char *dataBuffer, uint32_t dataBufferLength,
        char *ctrlBuffer = nullptr, uint32_t ctrlBufferLength = 0);
    // normal send, less than 512k
    APP_ERROR HdcNormalSendto(HdcSession &session, char *sendBuffer, uint32_t dataBufferLength);
    // fast recv
    APP_ERROR HdcFastRecv(HdcSession &session, char *&dataBuffer, uint32_t &dataBufferLength);
    APP_ERROR HdcFastRecv(HdcSession &session, char *&dataBuffer, uint32_t &dataBufferLength, char *&ctrlBuffer,
        int &ctrlBufferLength);
    // normal recv
    APP_ERROR HdcNormalRecv(HdcSession &session, char *&recvBuffer, uint32_t &recvBufferLength);

    APP_ERROR HdcStopRecv(HdcSession &session);

private:
    int deviceId_ = -1;
    HDC_SERVER server_ = 0;
    HDC_CLIENT client_ = 0;
    bool enableFastMode_ = false; // whether to enable the fast sending mode
    int useless_ = 0;             // for hdc internal use, no meaning
    // send and recv resource
    std::map<HdcSession, struct HdcBuffer> fastBufferInfo_ = {};       // for maloc and free
    std::map<HdcSession, struct drvHdcFastSendMsg> fastSendInfo_ = {}; // for fast send
    std::map<HdcSession, struct drvHdcFastRecvMsg> fastRecvInfo_ = {}; // for fast recv
    std::map<HdcSession, struct drvHdcMsg *> normalSendInfo_ = {};     // for normal send
    std::map<HdcSession, struct drvHdcMsg *> normalRecvInfo_ = {};     // for normal recv
    std::map<HdcSession, uint64_t> peerMaxDataLength_ = {};            // for max
    std::map<HdcSession, uint64_t> peerMaxCtrlLength_ = {};            // for normal recv

    APP_ERROR HdcNormalBufferMalloc(HdcSession &session);
    APP_ERROR HdcFastModeClientSourcePrepare(HdcSession &session, uint64_t maxFastDataBufferSize,
        uint64_t maxFastCtrlBufferSize);
    APP_ERROR HdcFastModeServerSourcePrepare(HdcSession &session, uint64_t maxFastDataBufferSize,
        uint64_t maxFastCtrlBufferSize);
    APP_ERROR HdcFastBufferMalloc(const HdcSession &session, const uint64_t maxDataBufferSize,
        const uint64_t maxCtrlBufferSize);
    void HdcFastBufferFree(char *&sendDataBuf, char *&recvDataBuf, char *&sendCtrlBuf, char *&recvCtrlBuf);
    APP_ERROR HdcReplyBufferCapability(HdcSession &session);
    void HdcNormalBufferMapFree();
    void HdcFastBufferMapFree();
    void HdcSessionMapFree();
    void HdcInfoMapFree();
    void HdcServerFree();
    APP_ERROR HdcStartToExchangeBufferCapability(HdcSession &session);
    void HdcClientFree();
};
#endif
