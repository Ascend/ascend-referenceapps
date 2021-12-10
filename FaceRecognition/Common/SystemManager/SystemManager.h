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

#ifndef INC_SYSTEM_MANAGER_H
#define INC_SYSTEM_MANAGER_H

#include "ModuleBase/ModuleBase.h"
#include "acl/acl.h"
#include "FaceFeatureLib/FaceFeatureLib.h"
#include "Log/Log.h"

namespace ascendFaceRecognition {
const std::string PIPELINE_DEFAULT = "DefaultPipeline";

// module description of pipeline
struct ModuleDesc {
    int moduleCount; // -1 channel count
    ModuleType moduleType;
    std::string moduleName;
};

struct ModuleConnectDesc {
    ModuleType moduleSend;
    ModuleType moduleRecv;
    ModuleConnectType connectType;
};

// data type of the input argument of system manager init function
struct SystemInitArguments {
    FaceFeatureLib *faceFeature = nullptr;
    void *userData = nullptr;
};

using SystemInitArgs = SystemInitArguments;

// information for one type of module
struct ModulesInformation {
    std::vector<std::shared_ptr<ModuleBase>> moduleVec = {};
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueueVec = {};
};

using ModulesInfo = ModulesInformation;

class SystemManager {
public:
    static std::shared_ptr<SystemManager> GetInstance();
    SystemManager();
    ~SystemManager();
    APP_ERROR Init(std::string &configPath, SystemInitArgs &initArgs, std::string &aclConfigPath);
    APP_ERROR DeInit(void);

    APP_ERROR RegisterModules(std::string pipelineName, ModuleDesc *moduleDesc, int moduleTypeCount, int defaultCount);
    APP_ERROR RegisterModuleConnects(std::string pipelineName, ModuleConnectDesc *connnectDesc, int moduleConnectCount);

    APP_ERROR RegisterInputVec(std::string pipelineName, ModuleType moduleType,
        std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueVec);
    APP_ERROR RegisterOutputModule(std::string pipelineName, ModuleType moduleSend, ModuleType moduleRecv,
        ModuleConnectType connectType, std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec);

    APP_ERROR RunPipeline();
    void SendDataToModule(std::string pipelineName, ModuleType moduleType, 
        std::shared_ptr<void> data, uint32_t channelId);

private:
#ifdef ASCEND_FACE_USE_ACL
    APP_ERROR InitAcl(std::string &aclConfigPath);
#endif
    APP_ERROR InitModuleInstance(std::shared_ptr<ModuleBase> moduleInstance, int instanceId, std::string pipelineName,
        std::string moduleName);

    APP_ERROR InitPipelineModule();
    APP_ERROR DeInitPipelineModule();

    static void StopModule(std::shared_ptr<ModuleBase> moduleInstance);

private:
    int32_t deviceId_ = 0;
#ifdef ASCEND_FACE_USE_ACL
    aclrtContext aclContext_ = nullptr;
    aclrtRunMode runMode_ = ACL_DEVICE;
#endif
    std::map<std::string, std::map<ModuleType, ModulesInfo>> pipelineMap_ = {};
    ConfigParser configParser_ = {};
    int moduleTypeCount_ = 0;
    int moduleConnectCount_ = 0;
    ModuleConnectDesc *connnectDesc_ = nullptr;
    PipelineMode pipeMode_ = PIPELINE_MODE_BOTTOM;
    FaceFeatureLib *faceFeature_ = nullptr;
};
} // namespace ascendFaceRecognition

#endif // INC_SYSTEM_MANAGER_H
