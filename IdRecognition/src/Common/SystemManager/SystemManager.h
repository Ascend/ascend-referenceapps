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
#ifndef SYSTEM_MANAGER_H
#define SYSTEM_MANAGER_H
#include "ModuleBase/ModuleBase.h"
#include "acl/acl.h"
#include "FaceFeatureLib/FaceFeatureLib.h"

namespace ascendFaceRecognition {
// module description of pipeline
struct ModuleDesc {
    ModuleDesc(int channelCount, ModuleType moduleType, const std::string moduleName, ModuleType nextModuleType)
        : channelCount(channelCount), moduleType(moduleType), moduleName(moduleName), nextModuleType(nextModuleType){};
    int channelCount = 0;
    ModuleType moduleType = {};
    std::string moduleName = {};
    ModuleType nextModuleType = {};
};

// data type of the input argument of system manager init function
struct SystemInitArguments {
    PipelineMode pipelineMode = {};
    int moduleTypeCount = 0;
    ModuleDesc *pipelineDesc = {};
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueVec = {};
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec = {};
    FaceFeatureLib *faceFeature = nullptr;
    void *userData = nullptr;
};
using SystemInitArgs = SystemInitArguments;

// information for one type of module
struct ModulesInformation {
    std::vector<std::shared_ptr<ModuleBase>> moduleVec;
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueueVec;
};
using ModulesInfo = ModulesInformation;

class SystemManager {
public:
    ~SystemManager();
    APP_ERROR Init(std::string &configPath, SystemInitArgs &initArgs);
    APP_ERROR DeInit(void);
    APP_ERROR RunPipeline();

    const std::map<ModuleType, ModulesInfo> &GetModulesInfoMap() const;
    const ModuleDesc *GetPipelineDesc() const;
    const std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> &GetOutputQueVec();

private:
    APP_ERROR InitDevice();
    APP_ERROR DeInitDevice() const;
    APP_ERROR InitModuleInstance(int moduleIdx, int instanceId, std::shared_ptr<ModuleBase> moduleInstance,
        BlockingQueue<std::shared_ptr<void>> &moduleInputQueue);
    APP_ERROR InitPipelineModule();
    APP_ERROR DeInitPipelineModule();

private:
    int32_t deviceId_ = 0;
    aclrtContext aclContext_ = nullptr;
    aclrtRunMode runMode_ = ACL_HOST;
    ModuleDesc *pipelineDesc_ = nullptr;
    int moduleTypeCount_ = 0;
    PipelineMode pipeMode_ = PIPELINE_MODE_SEARCH;

    ConfigParser configParser_ = {};
    std::map<ModuleType, ModulesInfo> modulesInfoMap_ = {};
    FaceFeatureLib *faceFeature_ = nullptr;
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueVec_ = {};
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec_ = {};
};
} // namespace ascendFaceRecognition

#endif // __INC_SYSTEM_MANAGER_H__
