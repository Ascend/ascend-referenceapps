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

#include <cstdio>

#include "common/HdcBase.h"
#include "common/RpcUtils.h"
#include "ascenddaemon/rpc/HdcServer.h"
#include "ascenddaemon/rpc/SessionHandler.h"

using namespace ::ascend;
using namespace ::faiss::ascend;

namespace {
bool g_loopEnd = false;
}

int main()
{
    aclError ret = aclInit(NULL);
    if (ret != ACL_ERROR_NONE) {
        RPC_LOG_ERROR("acl init failed");
        return 0;
    }
    RPC_LOG_INFO("acl init success");
    try {
        HdcServer &server = HdcServer::GetInstance();

        while (!g_loopEnd) {
            RPC_LOG_INFO("start listening...\n");

            HdcSession *sess = nullptr;
            ret = server.Accept(sess);
            if (ret != HDC_RPC_ERROR_NONE) {
                fprintf(stderr, "server accept error\n");
                continue;
            }
            RPC_LOG_INFO("server accept success, start handling session...\n");

            DeviceScope deviceScope;
            SessionHandler handler(sess);
            handler.Handle();
        }
    } catch (std::exception &e) {
        RPC_LOG_ERROR("catch exception: %s\n", e.what());
    }
    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        RPC_LOG_ERROR("finalize acl failed");
    }
    return 0;
}
