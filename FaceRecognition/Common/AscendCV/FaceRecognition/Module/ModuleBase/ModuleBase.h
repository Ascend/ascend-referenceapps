/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1 Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2 Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3 Neither the names of the copyright holders nor the names of the
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * Description: base class of application pipeline modules.
 * Date: 2020/04/16
 * History:
 * [2020-04-16]:
 */
#ifndef INC_MODULE_BASE_H
#define INC_MODULE_BASE_H

#include <thread>
#include <sys/time.h>

#include "BlockingQueue/BlockingQueue.h"
#include "ConfigParser/ConfigParser.h"
#include "DataType/DataType.h"
#include "ErrorCode/ErrorCode.h"
#include "Statistic/Statistic.h"
#include "ModelResource/ModelResource.h"
#ifdef ASCEND_FACE_USE_ACL
#include "acl/acl.h"
#endif

namespace ascendFaceRecognition {
struct ModuleInitArguments {
#ifdef ASCEND_FACE_USE_ACL
    aclrtRunMode runMode;
    aclrtContext context;
#endif
    std::string pipelineName = {};
    std::string moduleName = {};
    int instanceId = -1;
    std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue = nullptr;
    PipelineMode pipelineMode = PIPELINE_MODE_BOTTOM;
    void *userData = nullptr;
};

struct ModuleOutputInformation {
    ModuleType moduleType = MT_BOTTOM;
    ModuleConnectType connectType = MODULE_CONNECT_RANDOM;
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec = {};
    uint32_t outputQueVecSize = 0;
};

using ModuleInitArgs = ModuleInitArguments;
using ModuleOutputInfo = ModuleOutputInformation;

class ModuleBase {
public:
    ModuleBase() {};
    virtual ~ModuleBase() {};

    virtual APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs) = 0;
    virtual APP_ERROR DeInit(void) = 0;
    APP_ERROR Run(void); // create and run process thread
    APP_ERROR Stop(void);
    void SetInputVec(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue);
    void SetOutputInfo(ModuleType moduleType, ModuleConnectType connectType,
        std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec);
    void SendToNextModule(ModuleType moduleNext, std::shared_ptr<void> outputData, int channelId = 0);

public:
#ifdef ASCEND_FACE_USE_ACL
    aclrtRunMode runMode_ = {};
    aclrtContext aclContext_ = {};
#endif
    std::string moduleName_ = {};
    int instanceId_ = -1;

protected:
    void ProcessThread();
    virtual APP_ERROR Process(std::shared_ptr<void> inputData) = 0;
    void CallProcess(std::shared_ptr<void> &frameAiInfo);
    void AssignInitArgs(ModuleInitArgs &initArgs);

protected:
    std::string pipelineName_ = {};
    int32_t deviceId_ = -1;
    std::thread processThr_ = {};
    bool isStop_ = false;
    bool withoutInputQueue_ = false;
    PipelineMode pipeMode_ = PIPELINE_MODE_BOTTOM;
    std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue_ = nullptr;
    std::map<ModuleType, ModuleOutputInfo> outputQueMap_ = {};
    int outputQueVecSize_ = 0;
    ModuleConnectType connectType_ = MODULE_CONNECT_RANDOM;
    int sendCount_ = 0;
};
} // namespace ascendFaceRecognition

#endif
