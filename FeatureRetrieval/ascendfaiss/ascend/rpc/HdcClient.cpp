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

#include <faiss/ascend/rpc/HdcClient.h>

namespace faiss {
namespace ascend {
HdcClient &HdcClient::GetInstance()
{
    static std::mutex initLock;
    static HdcClient gClient(MAX_RPC_SESSION_NUM);

    if (!gClient.IsInitialized()) {
        std::lock_guard<std::mutex> guard(initLock);
        // double check
        if (!gClient.IsInitialized()) {
            int ret = gClient.Init();
            if (ret != HDC_RPC_ERROR_NONE) {
                FAISS_THROW_MSG("Fatal: gClient init error!\n");
            } else {
                gClient.SetInitialized(true);
            }
        }
    }

    return gClient;
}

HdcClient::HdcClient(int maxSessionNum, int serviceType)
    : sessionNum(maxSessionNum), serviceType(serviceType), client(nullptr), initialized(false)
{
}

HdcClient::~HdcClient()
{
    for (auto &kv : sessions) {
        delete kv.second.first;
    }

    if (client != nullptr) {
        drvHdcClientDestroy(client);
        client = nullptr;
    }
}

HdcRpcError HdcClient::Init()
{
    int ret;
    if ((ret = drvHdcClientCreate(&client, sessionNum, serviceType, 0)) != DRV_ERROR_NONE) {
        RPC_LOG_ERROR("drvHdcClientCreate failed, ret = %d\n", ret);
        client = nullptr;
        return HDC_RPC_ERROR_ERROR;
    }

    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcClient::Connect(int devId, HdcSession *&session)
{
    RPC_LOG_INFO("Connect session on %d\n", devId);
    int ret;
    HDC_SESSION sess;

    std::lock_guard<std::mutex> guard(lock);

    // one device shared one session
    if (sessions.find(devId) == sessions.end()) {
        // create new session
        RPC_LOG_INFO("Create new session\n");
        RPC_ASSERT(sessions.size() < static_cast<size_t>(sessionNum));

        const int retryTimes = 5;
        const int waitTimeMs = 20000;
        for (int i = 0; i < retryTimes; i++) {
            ret = drvHdcSessionConnect(0, devId, client, &sess);
            if (ret == DRV_ERROR_NONE) {
                break;
            } else {
                RPC_LOG_ERROR("[%ld] Connect HdcServer failed, wait %d ms and retry\n",
                    std::chrono::system_clock::now().time_since_epoch().count() *
                        std::chrono::system_clock::period::num /
                            std::chrono::system_clock::period::den, waitTimeMs);
                std::this_thread::sleep_for(std::chrono::milliseconds(waitTimeMs));
                continue;
            }
        }
        if (ret != DRV_ERROR_NONE) {
            RPC_LOG_ERROR("Session connect failed after %d retries\n", retryTimes);
            return HDC_RPC_ERROR_ERROR;
        }

        HdcSession *newSession = new (std::nothrow) HdcSession(devId);
        if (newSession == nullptr) {
            return HDC_RPC_ERROR_ERROR;
        }
        ret = newSession->Init(sess);
        if (ret != DRV_ERROR_NONE) {
            delete newSession;
            return HDC_RPC_ERROR_ERROR;
        }

        session = newSession;
        sessions[devId] = {newSession, 1};
    } else {
        // add ref count
        session = sessions[devId].first;
        sessions[devId].second++;
        RPC_LOG_INFO("Inc session ref count to %d\n", sessions[devId].second);
    }

    return HDC_RPC_ERROR_NONE;
}

HdcRpcError HdcClient::DisConnect(const HdcSession *session)
{
    bool find = false;
    RPC_LOG_INFO("DisConnect session %p\n", session);

    std::lock_guard<std::mutex> guard(lock);

    for (auto &kv : sessions) {
        if (kv.second.first == session) {
            find = true;
            // decrease ref count, if down to 0, release session
            kv.second.second--;
            RPC_LOG_INFO("Session on device %d, ref count: %d\n", kv.first, kv.second.second);
            if (kv.second.second == 0) {
                RPC_LOG_INFO("Release session\n");
                delete kv.second.first;
                sessions.erase(kv.first);
            }
            break;
        }
    }

    RPC_ASSERT(find);
    return HDC_RPC_ERROR_NONE;
}
} // namespace ascend
} // namespace faiss