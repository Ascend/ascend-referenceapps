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

#ifndef ASCEND_HDC_BASE_H
#define ASCEND_HDC_BASE_H

#include <memory>
#include <map>
#include <mutex>
#include <utility>
#include <cstdint>

#include <google/protobuf/message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <ascend_hal.h>

#include "RpcUtils.h"

namespace faiss {
namespace ascend {
using ::google::protobuf::Message;
using ::google::protobuf::io::ArrayInputStream;

const int HDC_SERVICE_TYPE_RPC = HDC_SERVICE_TYPE_USER3;
const int MAX_RPC_SESSION_NUM = 32;

enum HdcRpcError {
    HDC_RPC_ERROR_NONE,
    HDC_RPC_ERROR_ERROR,
};

enum HdcRpcServiceType {
    // common
    RPC_CREATE_CLIENT = 1,
    RPC_DESTROY_CLIENT,
    RPC_DESTROY_INDEX,
    RPC_INDEX_SEARCH,
    RPC_INDEX_REMOVE_IDS,
    RPC_INDEX_REMOVE_RANGE_IDS,
    RPC_INDEX_RESET,
    RPC_TEST_DATA_INTEGRITY,

    // IndexInt8
    RPC_DESTROY_INDEX_INT8,
    RPC_INDEX_INT8_SEARCH,
    RPC_INDEX_INT8_SEARCH_FILTER,
    RPC_INDEX_INT8_REMOVE_IDS,
    RPC_INDEX_INT8_REMOVE_RANGE_IDS,
    RPC_INDEX_INT8_RESET,

    // IVF common
    RPC_INDEX_IVF_UPDATE_COARSE_CENT,
    RPC_INDEX_IVF_UPDATE_NPROBE,
    RPC_INDEX_IVF_GET_LIST_LENGTH,
    RPC_INDEX_IVF_GET_LIST_CODES,
    RPC_INDEX_IVF_FAST_GET_LIST_CODES,
    RPC_INDEX_RESERVE_MEM,
    RPC_INDEX_RECLAIM_MEM,

    // INT8IVF common
    RPC_INDEX_INT8_IVF_UPDATE_COARSE_CENT,
    RPC_INDEX_INT8_IVF_UPDATE_NPROBE,
    RPC_INDEX_INT8_IVF_GET_LIST_LENGTH,
    RPC_INDEX_INT8_IVF_GET_LIST_CODES,
    RPC_INDEX_INT8_RESERVE_MEM,
    RPC_INDEX_INT8_RECLAIM_MEM,

    // IVFPQ
    RPC_CREATE_INDEX_IVFPQ,
    RPC_INDEX_IVFPQ_UPDATE_PQ_CENT,
    RPC_INDEX_IVFPQ_ADD,
    
    // Flat and IVFFlat
    RPC_CREATE_INDEX_FLAT,
    RPC_INDEX_FLAT_ADD,
    RPC_INDEX_FLAT_GET_BASE,
    RPC_INDEX_FLAT_GET_BASE_SIZE,
    RPC_CREATE_INDEX_IVFFLAT,
    RPC_INDEX_IVFFLAT_ADD,

    // SQ and IVFSQ
    RPC_CREATE_INDEX_SQ,
    RPC_INDEX_SQ_ADD,
    RPC_INDEX_SQ_GET_BASE,
    RPC_INDEX_SQ_FAST_GET_BASE,
    RPC_INDEX_SQ_GET_BASE_SIZE,
    RPC_INDEX_SQ_UPDATE_TRAINED_VALUE,
    RPC_CREATE_INDEX_IVFSQ,
    RPC_INDEX_IVFSQ_ADD,

    // INT8FLAT
    RPC_CREATE_INDEX_INT8_FLAT,
    RPC_INDEX_INT8_FLAT_ADD,
    RPC_INDEX_INT8_FLAT_GET_BASE,
    RPC_INDEX_INT8_FLAT_GET_BASE_SIZE,

    // INT8IVFFLAT
    RPC_CREATE_INDEX_INT8_IVFFLAT, // 50
    RPC_INDEX_INT8_IVFFLAT_ADD, // 51

    // TRANSFORM
    RPC_CREATE_INDEX_PRETRANSFORM,
    RPC_TRANSFORM_LINEAR_UPDATE_TRAINED_VALUE,
    RPC_INDEX_PRETRANSFORM_PREPEND,
    RPC_CREATE_TRANSFORM_LINEAR,
    RPC_DESTROY_TRANSFORM,

    // NN DIM REDUCTION
    RPC_CREATE_NN_DIM_REDUCTION,
    RPC_INFER_NN_DIM_REDUCTION,
    RPC_DESTROY_NN_DIM_REDUCTION,

    RPC_SERVICE_TYPE_MAX,
};

struct HdcSendData {
    void* localSendDataBuf {nullptr};
    void* localSendCtrlBuf {nullptr};
    size_t localSendDataSize {0};
    size_t localSendCtrlSize {0};
    uint64_t remoteRecvDataBuf {0};
    uint64_t remoteRecvCtrlBuf {0};
};

struct HdcRecvData {
    void* localRecvDataBuf {nullptr};
    void* localRecvCtrlBuf {nullptr};
    size_t localRecvDataSize {0};
    size_t localRecvCtrlSize {0};
    uint64_t remoteRecvDataBuf {0};
    uint64_t remoteRecvCtrlBuf {0};
};

class HdcMsgHeader {
public:
    explicit HdcMsgHeader();
    ~HdcMsgHeader();
    explicit HdcMsgHeader(bool isRequest, int service);

    inline bool IsRequestMsg() const
    {
        return isRequest;
    }
    inline int GetService() const
    {
        return service;
    }
    inline bool IsLastSegment() const
    {
        return lastSeg;
    }
    inline void SetLastSegment(bool last)
    {
        this->lastSeg = last;
    }
    inline int GetSegNum() const
    {
        return segNum;
    }
    inline void SetSegNum(int segNum)
    {
        this->segNum = segNum;
    }
    inline size_t Size() const
    {
        return HEADER_SIZE;
    }

    size_t SerializeToArray(uint8_t *buf, size_t bufSize) const;
    size_t ParseFromArray(uint8_t *buf, size_t bufSize);
    void Display() const;

private:
    static const int HEADER_SIZE = sizeof(uint32_t);
    static const int SEVICE_MASK_OFFSET = 0;
    static const int SEVICE_MASK_LEN = 8;
    static const int LAST_SEG_MASK_OFFSET = 8;
    static const int LAST_SEG_MASK_LEN = 1;
    static const int REQUEST_MASK_OFFSET = 9;
    static const int REQUEST_MASK_LEN = 1;
    static const int TOTAL_SEG_NUM_MASK_OFFSET = 16;
    static const int TOTAL_SEG_NUM_MASK_LEN = 16;

    bool isRequest;
    int service;
    bool lastSeg;
    int segNum;
};


class HdcSession {
public:
    explicit HdcSession(int devId, uint32_t segmentSize = DEFAULT_SEGMENT_SIZE, uint32_t bufSize = DEFAULT_BUF_SIZE,
        uint32_t maxBufSize = MAX_BUF_SIZE);
    ~HdcSession();

    HdcRpcError Init(HDC_SESSION session);
    void Destroy();
    // without lock
    HdcRpcError SerializeAndSendMessage(int service, bool isRequest, const Message &protoMsg);
    HdcRpcError RecvMessage(int &service, bool &isRequest, int &msgLen);
    HdcRpcError ParseMessage(Message &protoMsg, int msgLen) const;
    HdcRpcError RecvAndParseResponse(Message &protoMsg);
    // with lock to avoid race condition when multithreads using same session to send messages
    HdcRpcError SendAndReceive(int service, const Message &req, Message &resp);

    HdcRpcError HdcFastSendBufferMalloc(size_t dataLen, size_t channelId, size_t ctrlLen = 128);
    HdcRpcError HdcFastSendBufferFree();

    HdcRpcError HdcFastRecvBufferMalloc(size_t dataLen, size_t channelId, size_t ctrlLen = 128);
    HdcRpcError HdcFastRecvBufferFree();

    HdcRpcError HdcFastSendPrepare(size_t bufferSize);
    HdcRpcError HdcFastRecvPrepare();
    HdcRpcError HdcFastSendRelease();
    HdcRpcError HdcFastRecvRelease();

    HdcRpcError HdcGetFastSendAddr(void** sendDataPtr, void** sendCtrlPtr, size_t channelId);

    HdcRpcError HdcFastSend(void* sendDataPtr, void* sendCtrlPtr, size_t channelId);
    HdcRpcError HdcFastRecv(void** recvDataPtr, void** recvCtrlPtr, size_t channelId);

    HdcRpcError HdcSendRecvSignal();
    HdcRpcError HdcWaitRecvSignal();

    uint32_t HdcGetFastSendChannel();

    std::mutex& GetSessionLock()
    {
        return sessionLock;
    }

protected:
    const int devId;

private:
    static const uint32_t DEFAULT_SEGMENT_SIZE = 0x40000; // normal send, buf size must < 512KB
    static const uint32_t DEFAULT_BUF_SIZE = 0x2000000;   // preallocated msg buffer size for HDC
    static const uint32_t MAX_BUF_SIZE = 0x8000000;       // max msg buffer size
    static const uint32_t BUFFER_LEN = 256;
    static const uint32_t MAX_CHANNEL_NUMBER = 16;        // max fast HDC channel number

    uint32_t segmentSize;
    uint32_t bufSize;
    uint32_t maxBufSize;
    uint32_t channelNumber;

    HDC_SESSION session;
    std::mutex sessionLock;
    std::vector<uint8_t> msgBuf;
    struct drvHdcMsg *msgSend;
    struct drvHdcMsg *msgRecv;
    
    struct drvHdcFastSendMsg sMsg;
    struct drvHdcFastRecvMsg rMsg;
    
    HdcSendData *hdcSendData;
    HdcRecvData *hdcRecvData;
};
} // namespace ascend
} // namespace faiss

#endif // ASCEND_HDC_BASE_H
