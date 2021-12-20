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

#include "HdcBase.h"

#include <mutex>
#include <cstdlib>
#include <securec.h>

namespace faiss {
namespace ascend {
const uint32_t TIME_OUT = 0;

HdcMsgHeader::~HdcMsgHeader() {}

HdcMsgHeader::HdcMsgHeader() : isRequest(true), service(-1), lastSeg(true), segNum(0) {}

HdcMsgHeader::HdcMsgHeader(bool isRequest, int service)
    : isRequest(isRequest), service(service), lastSeg(true), segNum(0)
{}

size_t HdcMsgHeader::SerializeToArray(uint8_t *buf, size_t bufSize) const
{
    RPC_REQUIRE_NOT_NULL(buf);
    RPC_ASSERT(bufSize >= HEADER_SIZE);

    uint32_t tmp = 0;
    tmp = tmp | ((static_cast<uint32_t>(service) & 0xFF) << SEVICE_MASK_OFFSET); // service bits, 8bits
    tmp = tmp | ((lastSeg ? 0x1 : 0x0) << LAST_SEG_MASK_OFFSET);                 // last seg bit, 1bit
    tmp = tmp | ((isRequest ? 0x1 : 0x0) << REQUEST_MASK_OFFSET);                // requeset bit, 1bit
    tmp = tmp | ((static_cast<uint32_t>(segNum) & 0xFFFF) <<
        TOTAL_SEG_NUM_MASK_OFFSET); // total segment num bits, 16bits

    *(reinterpret_cast<uint32_t *>(buf)) = tmp;

    return HEADER_SIZE;
}

size_t HdcMsgHeader::ParseFromArray(uint8_t *buf, size_t bufSize)
{
    RPC_REQUIRE_NOT_NULL(buf);
    RPC_ASSERT(bufSize >= HEADER_SIZE);

    uint32_t tmp = *(reinterpret_cast<uint32_t *>(buf));
    service = (tmp >> SEVICE_MASK_OFFSET) & 0xFF;
    lastSeg = ((tmp >> LAST_SEG_MASK_OFFSET) & 0x1) > 0;
    isRequest = ((tmp >> REQUEST_MASK_OFFSET) & 0x1) > 0;
    segNum = (tmp >> TOTAL_SEG_NUM_MASK_OFFSET) & 0xFFFF;

    return HEADER_SIZE;
}

void HdcMsgHeader::Display() const
{
    printf("isRequest: %d, isLast: %d, service: %d, segNum: %d\n", isRequest, lastSeg, service, segNum);
}


HdcSession::HdcSession(int devId, uint32_t segmentSize, uint32_t bufSize, uint32_t maxBufSize)
    : devId(devId),
      segmentSize(segmentSize),
      bufSize(bufSize),
      maxBufSize(maxBufSize),
      channelNumber(MAX_CHANNEL_NUMBER),
      session(nullptr),
      msgSend(nullptr),
      msgRecv(nullptr),
      hdcSendData(nullptr),
      hdcRecvData(nullptr)
{
}

HdcSession::~HdcSession()
{
    if (msgSend != nullptr) {
        drvHdcFreeMsg(msgSend);
        msgSend = nullptr;
    }
    if (msgRecv != nullptr) {
        drvHdcFreeMsg(msgRecv);
        msgRecv = nullptr;
    }
    if (session != nullptr) {
        drvHdcSessionClose(session);
        session = nullptr;
    }
    if (hdcSendData != nullptr) {
        delete[] hdcSendData;
        hdcSendData = nullptr;
    }
    if (hdcRecvData != nullptr) {
        delete[] hdcRecvData;
        hdcRecvData = nullptr;
    }
}

HdcRpcError HdcSession::Init(HDC_SESSION session)
{
    int ret;
    RPC_REQUIRE_NOT_NULL(session);
    this->session = session;

    msgBuf.resize(this->bufSize);

    ret = drvHdcSetSessionReference(session);
    if (ret != DRV_ERROR_NONE) {
        RPC_LOG_ERROR("drvHdcSetSessionReference failed, ret = %d\n", ret);
        return HDC_RPC_ERROR_ERROR;
    }
    ret = drvHdcAllocMsg(session, &msgSend, 1);
    if (ret != DRV_ERROR_NONE) {
        RPC_LOG_ERROR("drvHdcAllocMsg msgSend failed, ret = %d\n", ret);
        msgSend = nullptr;
        return HDC_RPC_ERROR_ERROR;
    }
    ret = drvHdcAllocMsg(session, &msgRecv, 1);
    if (ret != DRV_ERROR_NONE) {
        RPC_LOG_ERROR("drvHdcAllocMsg msgRecv failed, ret = %d\n", ret);
        msgRecv = nullptr;
        return HDC_RPC_ERROR_ERROR;
    }
    hdcSendData = new(std::nothrow) HdcSendData[channelNumber];
    if (hdcSendData == nullptr) {
            RPC_LOG_ERROR("send hdcSendData malloc failed\n");
            return HDC_RPC_ERROR_ERROR;
        }
    hdcRecvData = new(std::nothrow) HdcRecvData[channelNumber];
    if (hdcRecvData == nullptr) {
            RPC_LOG_ERROR("receive hdcRecvData malloc failed\n");
            return HDC_RPC_ERROR_ERROR;
        }
    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::SerializeAndSendMessage(int service, bool isRequest, const Message &protoMsg)
{
    HdcMsgHeader msgHeader(isRequest, service);
    int ret;
    size_t headerLen = msgHeader.Size();
    size_t serializedLen = protoMsg.ByteSizeLong();
    size_t segDataLen = this->segmentSize - headerLen;
    int segNum = (serializedLen + segDataLen - 1) / segDataLen;
    int totalLen = segNum * headerLen + serializedLen;

    // due to proto3 behavior, if all value is default, then nothing will be serialized
    // ByteSizeLong() == 0 and segNum == 0, buf we still need to send 1 empty segment
    // reference: https://github.com/protocolbuffers/protobuf/issues/6066
    if (segNum == 0) {
        msgHeader.SetSegNum(1);
        msgHeader.SetLastSegment(true);
        msgHeader.SerializeToArray(msgBuf.data(), headerLen);
        ret = drvHdcReuseMsg(msgSend);
        RPC_REQUIRE_OK(ret);
        ret = drvHdcAddMsgBuffer(msgSend, reinterpret_cast<char *>(msgBuf.data()), headerLen);
        RPC_REQUIRE_OK(ret);

        ret = halHdcSend(session, msgSend, 0, TIME_OUT);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("Send empty segment error, ret = %d\n", ret);
            return HDC_RPC_ERROR_ERROR;
        }

        return HDC_RPC_ERROR_NONE;
    }

    // resize msg buf if current buf size is not large enough
    uint32_t resizedBufSize = segNum * this->segmentSize;
    if (resizedBufSize > this->bufSize) {
        if (resizedBufSize > this->maxBufSize) {
            RPC_LOG_ERROR("Need to enlarge msg buffer to %u bytes, exceed limit %u\n", totalLen, this->maxBufSize);
            return HDC_RPC_ERROR_ERROR;
        } else {
            RPC_LOG_INFO("Enlarge msg buf from %u to %u\n", this->bufSize, resizedBufSize);
            msgBuf.resize(resizedBufSize);
            this->bufSize = resizedBufSize;
        }
    }

    bool serializeResult = protoMsg.SerializeToArray(msgBuf.data() + headerLen, this->bufSize - headerLen);
    RPC_ASSERT(serializeResult);

    size_t offset = headerLen;
    for (int i = 0; i < segNum; i++) {
        // fill header
        msgHeader.SetSegNum(segNum);
        msgHeader.SetLastSegment(i == segNum - 1);
        msgHeader.SerializeToArray(msgBuf.data() + offset - headerLen, headerLen);

        // send segment
        int dataLen = (i == segNum - 1) ? (totalLen - i * this->segmentSize) : this->segmentSize;

        ret = drvHdcReuseMsg(msgSend);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("drvHdcReuseMsg error, ret = %d\n", ret);
            return HDC_RPC_ERROR_ERROR;
        }
        ret = drvHdcAddMsgBuffer(msgSend, reinterpret_cast<char *>(msgBuf.data() + offset - headerLen), dataLen);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("drvHdcAddMsgBuffer error, ret = %d\n", ret);
            return HDC_RPC_ERROR_ERROR;
        }

        ret = halHdcSend(session, msgSend, 0, TIME_OUT);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("Send segment %d error, ret = %d\n", i, ret);
            return HDC_RPC_ERROR_ERROR;
        }
        offset += segDataLen;
    }

    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::ParseMessage(Message &protoMsg, int msgLen) const
{
    // parse protobuf
    ArrayInputStream stream(msgBuf.data(), msgLen);
    bool result = protoMsg.ParseFromZeroCopyStream(&stream);
    RPC_ASSERT_FMT(result, "msgLen = %d", msgLen);

    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::RecvAndParseResponse(Message &protoMsg)
{
    HdcRpcError ret;
    int service;
    bool isRequest = false;
    int msgLen;
    ret = RecvMessage(service, isRequest, msgLen);
    RPC_ASSERT(!isRequest);
    if (ret != HDC_RPC_ERROR_NONE) {
        return ret;
    }

    return ParseMessage(protoMsg, msgLen);
}

HdcRpcError HdcSession::RecvMessage(int &service, bool &isRequest, int &msgLen)
{
    HdcMsgHeader msgHeader;
    const size_t headerLen = msgHeader.Size();
    size_t offset = 0;
    service = -1;
    int segNum = 0;
    int totalSegNum = 0;

    do {
        int ret = drvHdcReuseMsg(msgRecv);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("drvHdcReuseMsg error, ret = %d\n", ret);
            return HDC_RPC_ERROR_ERROR;
        }

        int recvBufCount = 0;
        char *buf = nullptr;
        int bufLen = 0;
        ret = halHdcRecv(session, msgRecv, 0, 0, &recvBufCount, TIME_OUT);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("drvHdcRecv error, ret = %d\n", ret);
            return HDC_RPC_ERROR_ERROR;
        }
        ret = drvHdcGetMsgBuffer(msgRecv, 0, &buf, &bufLen);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("drvHdcGetMsgBuffer error, ret = %d\n", ret);
            return HDC_RPC_ERROR_ERROR;
        }

        msgHeader.ParseFromArray(reinterpret_cast<uint8_t *>(buf), bufLen);
        if (segNum == 0) {
            service = msgHeader.GetService();
            isRequest = msgHeader.IsRequestMsg();
            totalSegNum = msgHeader.GetSegNum();
            // resize msg buf if necessary
            uint32_t resizedBufSize = totalSegNum * this->segmentSize;
            if (resizedBufSize > this->bufSize) {
                // guaranteed by sender
                RPC_ASSERT(resizedBufSize <= this->maxBufSize);
                msgBuf.resize(resizedBufSize);
                this->bufSize = resizedBufSize;
            }
        } else {
            RPC_ASSERT(service == msgHeader.GetService());
            RPC_ASSERT(isRequest == msgHeader.IsRequestMsg());
            RPC_ASSERT(totalSegNum == msgHeader.GetSegNum());
        }
        if (!msgHeader.IsLastSegment()) {
            RPC_ASSERT_FMT(bufLen == static_cast<int>(this->segmentSize), "bufLen error: %d\n", bufLen);
        }

        auto err = memcpy_s(msgBuf.data() + offset, this->bufSize - offset, buf + headerLen, bufLen - headerLen);
        if (err != EOK) {
            RPC_LOG_ERROR("memcpy_s error, ret = %d\n", err);
            return HDC_RPC_ERROR_ERROR;
        }
        offset += bufLen - headerLen;
        segNum++;
    } while (!msgHeader.IsLastSegment());

    msgLen = offset;
    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::SendAndReceive(int service, const Message &req, Message &resp)
{
    std::lock_guard<std::mutex> guard(this->sessionLock);

    HdcRpcError ret = SerializeAndSendMessage(service, true, req);
    if (ret != HDC_RPC_ERROR_NONE) {
        return ret;
    }
    ret = RecvAndParseResponse(resp);
    if (ret != HDC_RPC_ERROR_NONE) {
        return ret;
    }

    return HDC_RPC_ERROR_NONE;
}

namespace {
const int SEND_DATA_POS = 0;
const int REMOTE_DATA_POS = 1;
const int SEND_CTRL_POS = 2;
const int REMOTE_CTRL_POS = 3;
const int SEND_DATA_SIZE_POS = 4;
const int SEND_CTRL_SIZE_POS = 5;
}

HdcRpcError HdcSession::HdcFastSendBufferMalloc(size_t dataLen, size_t channelId, size_t ctrlLen)
{
    if (hdcSendData[channelId].localSendDataBuf != nullptr || hdcSendData[channelId].localSendCtrlBuf != nullptr) {
        if (HdcFastSendBufferFree() != HDC_RPC_ERROR_NONE) {
            return HDC_RPC_ERROR_ERROR;
        }
    }

    hdcSendData[channelId].localSendDataSize = dataLen;
    hdcSendData[channelId].localSendCtrlSize = ctrlLen;

    hdcSendData[channelId].localSendDataBuf = 
        drvHdcMallocEx(HDC_MEM_TYPE_TX_DATA, NULL, 0, hdcSendData[channelId].localSendDataSize, devId, 0);
    if (hdcSendData[channelId].localSendDataBuf == nullptr) {
        RPC_LOG_ERROR("drvHdcMalloc send dataBuf failed\n");
        return HDC_RPC_ERROR_ERROR;
    }

    hdcSendData[channelId].localSendCtrlBuf = 
        drvHdcMallocEx(HDC_MEM_TYPE_TX_CTRL, NULL, 0, hdcSendData[channelId].localSendCtrlSize, devId, 0);
    if (hdcSendData[channelId].localSendCtrlBuf == nullptr) {
        RPC_LOG_ERROR("drvHdcMalloc send ctrlBuf failed\n");
        return HDC_RPC_ERROR_ERROR;
    }

    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::HdcFastSendBufferFree()
{
    for (size_t i = 0; i < channelNumber; i++) {
        int ret;
        if (hdcSendData[i].localSendDataBuf != nullptr) {
            ret = drvHdcFreeEx(HDC_MEM_TYPE_TX_DATA, hdcSendData[i].localSendDataBuf);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("free fast dataBuf fail, ret=%d\n", ret);
                return HDC_RPC_ERROR_ERROR;
            }
            hdcSendData[i].localSendDataBuf = nullptr;
        }

        if (hdcSendData[i].localSendCtrlBuf != nullptr) {
            ret = drvHdcFreeEx(HDC_MEM_TYPE_TX_CTRL, hdcSendData[i].localSendCtrlBuf);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("free fast ctrlBuf fail, ret=%d\n", ret);
                return HDC_RPC_ERROR_ERROR;
            }
            hdcSendData[i].localSendCtrlBuf = nullptr;
        }
    }
    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::HdcFastRecvBufferMalloc(size_t dataLen, size_t channelId, size_t ctrlLen)
{
    if (hdcRecvData[channelId].localRecvDataBuf != nullptr || hdcRecvData[channelId].localRecvCtrlBuf != nullptr) {
        if (HdcFastRecvBufferFree() != HDC_RPC_ERROR_NONE) {
            return HDC_RPC_ERROR_ERROR;
        }
    }

    hdcRecvData[channelId].localRecvDataSize = dataLen;
    hdcRecvData[channelId].localRecvCtrlSize = ctrlLen;

    hdcRecvData[channelId].localRecvDataBuf = 
        drvHdcMallocEx(HDC_MEM_TYPE_RX_DATA, NULL, 0, hdcRecvData[channelId].localRecvDataSize, devId, 0);
    if (hdcRecvData[channelId].localRecvDataBuf == nullptr) {
        RPC_LOG_ERROR("drvHdcMalloc recv dataBuf failed\n");
        return HDC_RPC_ERROR_ERROR;
    }

    hdcRecvData[channelId].localRecvCtrlBuf = 
        drvHdcMallocEx(HDC_MEM_TYPE_RX_CTRL, NULL, 0, hdcRecvData[channelId].localRecvCtrlSize, devId, 0);
    if (hdcRecvData[channelId].localRecvCtrlBuf == nullptr) {
        RPC_LOG_ERROR("drvHdcMalloc recv ctrlBuf failed\n");
        return HDC_RPC_ERROR_ERROR;
    }

    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::HdcFastRecvBufferFree()
{
    for (size_t i = 0; i < channelNumber; i++) {
        int ret;
        if (hdcRecvData[i].localRecvDataBuf != nullptr) {
            ret = drvHdcFreeEx(HDC_MEM_TYPE_RX_DATA, hdcRecvData[i].localRecvDataBuf);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("free fast recv dataBuf fail, ret=%d\n", ret);
                return HDC_RPC_ERROR_ERROR;
            }
            hdcRecvData[i].localRecvDataBuf = nullptr;
        }

        if (hdcRecvData[i].localRecvCtrlBuf != nullptr) {
            ret = drvHdcFreeEx(HDC_MEM_TYPE_RX_CTRL, hdcRecvData[i].localRecvCtrlBuf);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("free fast recv ctrlBuf fail, ret=%d\n", ret);
                return HDC_RPC_ERROR_ERROR;
            }
            hdcRecvData[i].localRecvCtrlBuf = nullptr;
        }
    }
    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::HdcFastSendPrepare(size_t bufferSize)
{
    for (size_t i = 0; i < channelNumber; i++) {
        if (HdcFastSendBufferMalloc(bufferSize, i) != HDC_RPC_ERROR_NONE) {
            RPC_LOG_ERROR("fast send buffer malloc failed in fast send prepare\n");
            return HDC_RPC_ERROR_ERROR;
        }

        char* buffer = new(std::nothrow) char[BUFFER_LEN];
        if (buffer == nullptr) {
            RPC_LOG_ERROR("send buffer malloc failed\n");
            return HDC_RPC_ERROR_ERROR;
        }
        *((reinterpret_cast<uint64_t*>(buffer)) + SEND_DATA_POS) = 
            reinterpret_cast<uint64_t>(hdcSendData[i].localSendDataBuf);
        *((reinterpret_cast<uint64_t*>(buffer)) + REMOTE_DATA_POS) = 0;
        *((reinterpret_cast<uint64_t*>(buffer)) + SEND_CTRL_POS) = 
            reinterpret_cast<uint64_t>(hdcSendData[i].localSendCtrlBuf);
        *((reinterpret_cast<uint64_t*>(buffer)) + REMOTE_CTRL_POS) = 0;
        *((reinterpret_cast<uint64_t*>(buffer)) + SEND_DATA_SIZE_POS) = hdcSendData[i].localSendDataSize;
        *((reinterpret_cast<uint64_t*>(buffer)) + SEND_CTRL_SIZE_POS) = hdcSendData[i].localSendCtrlSize;

        int ret;
        do {
            ret = drvHdcReuseMsg(msgSend);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("HdcFastSendPrepare reuse msg err, ret %d\n", ret);
                break;
            }

            ret = drvHdcAddMsgBuffer(msgSend, buffer, BUFFER_LEN);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("HdcFastSendPrepare add msg err, ret %d\n", ret);
                break;
            }

            ret = halHdcSend(session, msgSend, 0, 0);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("HdcFastSendPrepare send err, ret %d\n", ret);
                break;
            }

            ret = drvHdcReuseMsg(msgRecv);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("HdcFastSendPrepare reuse msg err, ret %d\n", ret);
                break;
            }

            int recvBufCount;
            ret = halHdcRecv(session, msgRecv, BUFFER_LEN, 0, &recvBufCount, 0);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("HdcFastSendPrepare recv err, ret %d\n", ret);
                break;
            }

            char* pBuf = nullptr;
            int bufLen;
            ret = drvHdcGetMsgBuffer(msgRecv, 0, &pBuf, &bufLen);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("HdcFastSendPrepare get msg buffer err, ret %d\n", ret);
                break;
            }

            hdcSendData[i].remoteRecvDataBuf = *((reinterpret_cast<uint64_t*>(pBuf)) + REMOTE_DATA_POS);
            hdcSendData[i].remoteRecvCtrlBuf = *((reinterpret_cast<uint64_t*>(pBuf)) + REMOTE_CTRL_POS);
        } while (false);

        delete [] buffer;
        if (ret != DRV_ERROR_NONE) {
            return HDC_RPC_ERROR_ERROR;
        }
    }
    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::HdcFastRecvPrepare()
{
    for (size_t i = 0; i < channelNumber; i++) {
        int ret = drvHdcReuseMsg(msgRecv);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("HdcFastRecvPrepare reuse msg err, ret %d\n", ret);
            return HDC_RPC_ERROR_ERROR;
        }

        int recvBufCount;
        ret = halHdcRecv(session, msgRecv, BUFFER_LEN, 0, &recvBufCount, 0);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("HdcFastRecvPrepare recv err, ret %d\n", ret);
            return HDC_RPC_ERROR_ERROR;
        }

        char* pBuf = nullptr;
        int bufLen;
        ret = drvHdcGetMsgBuffer(msgRecv, 0, &pBuf, &bufLen);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("HdcFastRecvPrepare get msg buffer err, ret %d\n", ret);
            return HDC_RPC_ERROR_ERROR;
        }

        hdcRecvData[i].localRecvDataSize =
            static_cast<size_t>(*(reinterpret_cast<uint64_t*>(pBuf) + SEND_DATA_SIZE_POS));
        hdcRecvData[i].localRecvCtrlSize = 
            static_cast<size_t>(*(reinterpret_cast<uint64_t*>(pBuf) + SEND_CTRL_SIZE_POS));
        if (HdcFastRecvBufferMalloc(hdcRecvData[i].localRecvDataSize, i, hdcRecvData[i].localRecvCtrlSize) != 
            HDC_RPC_ERROR_NONE) {
            return HDC_RPC_ERROR_ERROR;
        }

        char* buffer = new(std::nothrow) char[BUFFER_LEN];
        if (buffer == nullptr) {
            RPC_LOG_ERROR("send buffer malloc failed\n");
            return HDC_RPC_ERROR_ERROR;
        }
        *(reinterpret_cast<uint64_t*>(buffer) + SEND_DATA_POS) = 0;
        *(reinterpret_cast<uint64_t*>(buffer) + REMOTE_DATA_POS) = 
            reinterpret_cast<uint64_t>(hdcRecvData[i].localRecvDataBuf);
        *(reinterpret_cast<uint64_t*>(buffer) + SEND_CTRL_POS) = 0;
        *(reinterpret_cast<uint64_t*>(buffer) + REMOTE_CTRL_POS) = 
            reinterpret_cast<uint64_t>(hdcRecvData[i].localRecvCtrlBuf);
        *(reinterpret_cast<uint64_t*>(buffer) + SEND_DATA_SIZE_POS) = 1;
        *(reinterpret_cast<uint64_t*>(buffer) + SEND_CTRL_SIZE_POS) = 1;

        do {
            ret = drvHdcReuseMsg(msgSend);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("HdcFastRecvPrepare reuse send msg err, ret %d\n", ret);
                break;
            }

            ret = drvHdcAddMsgBuffer(msgSend, buffer, BUFFER_LEN);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("HdcFastRecvPrepare add msg err, ret %d\n", ret);
                break;
            }

            ret = halHdcSend(session, msgSend, 0, 0);
            if (ret != DRV_ERROR_NONE) {
                RPC_LOG_ERROR("HdcFastRecvPrepare send err, ret %d\n", ret);
                break;
            }
        } while (false);

        delete [] buffer;
        if (ret != DRV_ERROR_NONE) {
            return HDC_RPC_ERROR_ERROR;
        }
    }
    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::HdcFastSendRelease()
{
    return HdcFastSendBufferFree();
}

HdcRpcError HdcSession::HdcFastRecvRelease()
{
    return HdcFastRecvBufferFree();
}

HdcRpcError HdcSession::HdcGetFastSendAddr(void** sendDataPtr, void** sendCtrlPtr, size_t channelId)
{
    if (sendDataPtr == nullptr || sendCtrlPtr == nullptr) {
        RPC_LOG_ERROR("HdcGetFastSendAddr input arguments err\n");
        return HDC_RPC_ERROR_ERROR;
    }

    *sendDataPtr = hdcSendData[channelId].localSendDataBuf;
    *sendCtrlPtr = hdcSendData[channelId].localSendCtrlBuf;

    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::HdcFastSend(void* sendDataPtr, void* sendCtrlPtr, size_t channelId)
{
    sMsg.srcDataAddr = reinterpret_cast<uint64_t>(sendDataPtr);
    sMsg.dstDataAddr = hdcSendData[channelId].remoteRecvDataBuf;
    sMsg.srcCtrlAddr = reinterpret_cast<uint64_t>(sendCtrlPtr);
    sMsg.dstCtrlAddr = hdcSendData[channelId].remoteRecvCtrlBuf;

    sMsg.dataLen = hdcSendData[channelId].localSendDataSize;
    sMsg.ctrlLen = hdcSendData[channelId].localSendCtrlSize;
    int ret = halHdcFastSend(session, sMsg, 0, 0);
    if (ret != DRV_ERROR_NONE) {
        RPC_LOG_ERROR("halHdcFastSend err, ret %d\n", ret);
        return HDC_RPC_ERROR_ERROR;
    }

    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::HdcFastRecv(void** recvDataPtr, void** recvCtrlPtr, size_t channelId)
{
    if (recvDataPtr == nullptr || recvCtrlPtr == nullptr) {
        RPC_LOG_ERROR("HdcFastRecv input arguments err\n");
        return HDC_RPC_ERROR_ERROR;
    }

    int ret = halHdcFastRecv(session, &rMsg, 0, 0);
    if (ret != DRV_ERROR_NONE) {
        RPC_LOG_ERROR("halHdcFastRecv err, ret %d\n", ret);
        return HDC_RPC_ERROR_ERROR;
    }

    *recvDataPtr = hdcRecvData[channelId].localRecvDataBuf;
    *recvCtrlPtr = hdcRecvData[channelId].localRecvCtrlBuf;

    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::HdcSendRecvSignal()
{
    int ret;
    char* buffer = new(std::nothrow) char[BUFFER_LEN];
    if (buffer == nullptr) {
        RPC_LOG_ERROR("send buffer malloc failed\n");
        return HDC_RPC_ERROR_ERROR;
    }

    do {
        ret = drvHdcReuseMsg(msgSend);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("send recv signal, resue msg err, ret %d\n", ret);
            break;
        }

        ret = drvHdcAddMsgBuffer(msgSend, buffer, BUFFER_LEN);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("send recv signal, add msg err, ret %d\n", ret);
            break;
        }

        ret = halHdcSend(session, msgSend, 0, 0);
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("send recv signal, send err, ret %d\n", ret);
            break;
        }
    } while (false);

    delete [] buffer;
    if (ret != DRV_ERROR_NONE) {
        return HDC_RPC_ERROR_ERROR;
    }

    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcSession::HdcWaitRecvSignal()
{
    int ret = drvHdcReuseMsg(msgRecv);
    if (ret != DRV_ERROR_NONE) {
        RPC_LOG_ERROR("wait recv signal, reuse msg err, ret %d\n", ret);
        return HDC_RPC_ERROR_ERROR;
    }

    int recvBufCount;
    ret = halHdcRecv(session, msgRecv, BUFFER_LEN, 0, &recvBufCount, 0);
    if (ret != DRV_ERROR_NONE) {
        RPC_LOG_ERROR("wait recv signal, recv err, ret %d\n", ret);
        return HDC_RPC_ERROR_ERROR;
    }

    char *pBuf = nullptr;
    int bufLen;
    ret = drvHdcGetMsgBuffer(msgRecv, 0, &pBuf, &bufLen);
    if (ret != DRV_ERROR_NONE) {
        RPC_LOG_ERROR("wait recv signal, get msg buffer err, ret %d\n", ret);
        return HDC_RPC_ERROR_ERROR;
    }

    return HDC_RPC_ERROR_NONE;
}

uint32_t HdcSession::HdcGetFastSendChannel()
{
    return this->channelNumber;
}
} // namespace ascend
} // namespace faiss
