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

#include "common/HdcBase.h"

namespace ascend {
// can only be used by one thread at the same time
class HdcServer {
public:
    static HdcServer &GetInstance();
    faiss::ascend::HdcRpcError Accept(faiss::ascend::HdcSession *&session);

private:
    explicit HdcServer(int serviceType = faiss::ascend::HDC_SERVICE_TYPE_RPC);
    ~HdcServer();

    faiss::ascend::HdcRpcError Init();
    inline bool IsInitialized()
    {
        return initialized;
    }

    inline void SetInitialized(bool init)
    {
        this->initialized = init;
    }

    bool initialized;
    int devId;
    int serviceType;
    HDC_SERVER server;
};
} // namespace ascend