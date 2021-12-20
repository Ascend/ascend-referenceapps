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

#ifndef ASCEND_FAISS_RPC_HDC_H
#define ASCEND_FAISS_RPC_HDC_H

#include <thread>
#include <chrono>

#include <faiss/impl/FaissAssert.h>

#include <securec.h>

#include <faiss/common/AscendIndex.pb.h>
#include <faiss/common/RpcUtils.h>
#include <faiss/common/HdcBase.h>

namespace faiss {
namespace ascend {
class HdcClient {
public:
    static HdcClient &GetInstance();
    HdcRpcError Connect(int devId, HdcSession *&session);
    HdcRpcError DisConnect(const HdcSession *session);
private:
    explicit HdcClient(int maxSessionNum, int serviceType = HDC_SERVICE_TYPE_RPC);
    ~HdcClient();
    HdcRpcError Init();
    inline bool IsInitialized()
    { 
        return initialized;
    }

    inline void SetInitialized(bool value)
    {
        initialized = value;
    }

    const int sessionNum;
    const int serviceType;
    HDC_CLIENT client;
    bool initialized;

    std::mutex lock;
    std::map<int, std::pair<HdcSession*, int>> sessions;
};

#define CALL_RPC(ctx, cmd, request, response, errorGet)                                \
    do {                                                                               \
        HdcSession *session_ = static_cast<HdcSession *>(ctx);                         \
        RPC_REQUIRE_NOT_NULL(session_);                                                \
        HdcRpcError ret_ = session_->SendAndReceive(cmd, request, response);           \
        if (ret_ != HDC_RPC_ERROR_NONE || errorGet != CommonResponse_ErrorCode_OK) {   \
            return RPC_ERROR_ERROR;                                                    \
        }                                                                              \
    } while (false)         
}  // namespace ascend
}  // namespace faiss
#endif
