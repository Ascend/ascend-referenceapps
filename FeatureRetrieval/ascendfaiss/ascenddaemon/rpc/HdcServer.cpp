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

#include "ascenddaemon/rpc/HdcServer.h"
#include "ascenddaemon/utils/AscendAssert.h"

using namespace ::faiss::ascend;

namespace ascend {
HdcServer::HdcServer(int serviceType) : initialized(false), devId(0), serviceType(serviceType), server(nullptr) {}

HdcServer::~HdcServer()
{
    if (server != nullptr) {
        drvHdcServerDestroy(server);
        server = nullptr;
    }
}

HdcServer &HdcServer::GetInstance()
{
    static HdcServer gServer;

    if (!gServer.IsInitialized()) {
        int ret = gServer.Init();
        if (ret != HDC_RPC_ERROR_NONE) {
            ASCEND_THROW_MSG("Fatal error: hdc server init error, exit!\n");
        } else {
            gServer.SetInitialized(true);
        }
    }

    return gServer;
}

HdcRpcError HdcServer::Init()
{
    int ret = drvHdcServerCreate(devId, serviceType, &server);
    if (ret != DRV_ERROR_NONE) {
        server = nullptr;
        RPC_LOG_ERROR("drvHdcServerCreate failed, ret = %d\n", ret);
        return HDC_RPC_ERROR_ERROR;
    }
    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcServer::Accept(HdcSession *&session)
{
    HDC_SESSION sess;
    int ret = drvHdcSessionAccept(server, &sess);
    if (ret != DRV_ERROR_NONE) {
        RPC_LOG_ERROR("drvHdcSessionAccept error, ret = %d\n", ret);
        return HDC_RPC_ERROR_ERROR;
    }

    HdcSession *newSession = new (std::nothrow) HdcSession(0);
    if (newSession == nullptr) {
        return HDC_RPC_ERROR_ERROR;
    }
    ret = newSession->Init(sess);
    if (ret != HDC_RPC_ERROR_NONE) {
        RPC_LOG_ERROR("session init error, ret = %d\n", ret);
        delete newSession;
        return HDC_RPC_ERROR_ERROR;
    }

    session = newSession;
    return HDC_RPC_ERROR_NONE;
}
} // namespace ascend