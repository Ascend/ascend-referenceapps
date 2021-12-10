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
#ifndef MODULE_BASE_H
#define MODULE_BASE_H

#include <thread>

#include "BlockingQueue/BlockingQueue.h"
#include "ConfigParser/ConfigParser.h"
#include "DataType/DataType.h"
#include "ErrorCode/ErrorCode.h"
#ifdef ASCEND_FACE_USE_ACL
#include "acl/acl.h"
#endif

namespace ascendFaceRecognition {
struct ModuleInitArgs {
#ifdef ASCEND_FACE_USE_ACL
    aclrtRunMode runMode = {};
    aclrtContext context = {};
#endif
    std::string moduleName = {};
    int instanceId = 0;
    BlockingQueue<std::shared_ptr<void>> *inputQueue = nullptr;
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec = {};
    PipelineMode pipelineMode = {};
    void *userData = nullptr;
};

class ModuleBase {
public:
    ModuleBase();
    virtual ~ModuleBase();

    virtual APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs) = 0;
    virtual APP_ERROR DeInit() = 0;
    APP_ERROR Run(); // create and run process thread

protected:
    void ProcessThread();
    virtual APP_ERROR Process(std::shared_ptr<void> inputData) = 0;
    void AssignInitArgs(const ModuleInitArgs &initArgs);
    void StopAndDestroyQueue();

protected:
#ifdef ASCEND_FACE_USE_ACL
    aclrtRunMode runMode_ = {};
    aclrtContext aclContext_ = {};
#endif
    bool isInited_ = false;
    bool isDeInited_ = false;
    int instanceId_ = 0;
    int32_t deviceId_ = 0;
    bool isStop_ = false;
    bool withoutInputQueue_ = false;
    BlockingQueue<std::shared_ptr<void>> *inputQueue_ = nullptr;

    std::string moduleName_ = "";
    std::thread processThr_ = {};
    PipelineMode pipeMode_ = PIPELINE_MODE_SEARCH;
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec_ = {};
};
} // namespace ascendFaceRecognition

#endif