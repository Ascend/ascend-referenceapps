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

#include "SystemManager.h"
#include "Log/Log.h"
#include "ModuleFactory/ModuleFactory.h"
#ifdef ASCEND_FACE_USE_ACL
#include "ModelResource/ModelResource.h"
#include "ResourceManager/ResourceManager.h"
#endif

namespace ascendFaceRecognition {
SystemManager::SystemManager() {}

SystemManager::~SystemManager() {}


std::shared_ptr<SystemManager> SystemManager::GetInstance()
{
    static std::shared_ptr<SystemManager> systemManager = std::make_shared<SystemManager>();
    return systemManager;
}

APP_ERROR SystemManager::Init(std::string &configPath, SystemInitArgs &initArgs, std::string &aclConfigPath)
{
    LogDebug << "SystemManager: begin to init.";

    // 0. init variables
    faceFeature_ = initArgs.faceFeature;

    // 1. load and parse config file
    APP_ERROR ret = configParser_.ParseConfig(configPath);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: cannot parse file.";
        return ret;
    }

    // 2. Init device
#ifdef ASCEND_FACE_USE_ACL
    ret = InitAcl(aclConfigPath);
    if (ret != APP_ERR_OK) {
        LogFatal << "ModuleManager: fail to init Acl.";
        return ret;
    }
#endif

    // 3. Init pipeline module
    ret = InitPipelineModule();
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to init pipeline module.";
        return ret;
    }

    return ret;
}

#ifdef ASCEND_FACE_USE_ACL
APP_ERROR SystemManager::InitAcl(std::string &aclConfigPath)
{
    LogDebug << "ModuleManager: begin to init Acl.";
    std::string itemCfgStr;

    itemCfgStr = "SystemConfig.deviceId";
    APP_ERROR ret = configParser_.GetIntValue(itemCfgStr, deviceId_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ModuleManager: fail to get device id.";
        return ret;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret != APP_ERR_OK) {
        LogFatal << "ModuleManager: fail to get run mode of device, ret=" << ret << ".";
        return ret;
    }

    if (runMode_ == ACL_DEVICE) {
        deviceId_ = 0;
    } else if (deviceId_ < 0) {
        LogFatal << "ModuleManager: invalid device id, deviceId=" << deviceId_ << ".";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    ResourceInfo resourceInfo;
    resourceInfo.aclConfigPath = aclConfigPath;
    resourceInfo.deviceIds.insert(deviceId_);
    return ResourceManager::GetInstance()->InitResource(resourceInfo);
}
#endif

APP_ERROR SystemManager::InitModuleInstance(std::shared_ptr<ModuleBase> moduleInstance, int instanceId,
    std::string pipelineName, std::string moduleName)
{
    LogDebug << "SystemManager: begin to init module instance." << moduleName;
    ModuleInitArgs initArgs;
#ifdef ASCEND_FACE_USE_ACL
    initArgs.context = ResourceManager::GetInstance()->GetContext(deviceId_);
    initArgs.runMode = runMode_;
#endif
    initArgs.pipelineName = pipelineName;
    initArgs.moduleName = moduleName;
    initArgs.instanceId = instanceId;
    initArgs.userData = faceFeature_;

    // Initialize the Init function of each module
    APP_ERROR ret = moduleInstance->Init(configParser_, initArgs);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to init module, name = " << moduleName.c_str() << ", instance id = " <<
            instanceId << ".";
        return ret;
    }
    LogDebug << "SystemManager: module " << initArgs.moduleName << "[" << instanceId << "] init success.";
    return ret;
}

APP_ERROR SystemManager::RegisterModules(std::string pipelineName, ModuleDesc *modulesDesc, int moduleTypeCount,
    int defaultCount)
{
    std::map<std::string, std::map<ModuleType, ModulesInfo>>::iterator iter = pipelineMap_.find(pipelineName);
    std::map<ModuleType, ModulesInfo> modulesInfoMap;
    if (iter != pipelineMap_.end()) {
        modulesInfoMap = iter->second;
    }

    std::shared_ptr<ModuleBase> moduleInstance = nullptr;

    // create new object of module
    // auto initialize the Init function of each module
    for (int i = 0; i < moduleTypeCount; i++) {
        ModuleDesc moduleDesc = modulesDesc[i];
        int moduleCount = (moduleDesc.moduleCount == -1) ? defaultCount : moduleDesc.moduleCount;
        ModulesInfo modulesInfo;
        for (int j = 0; j < moduleCount; j++) {
            moduleInstance = ModuleFactory::MakeModule(moduleDesc.moduleType);
            APP_ERROR ret = InitModuleInstance(moduleInstance, j, pipelineName, moduleDesc.moduleName);
            if (ret != APP_ERR_OK) {
                return ret;
            }
            modulesInfo.moduleVec.push_back(moduleInstance);
        }
        modulesInfoMap[moduleDesc.moduleType] = modulesInfo;
    }

    pipelineMap_[pipelineName] = modulesInfoMap;

    return APP_ERR_OK;
}

APP_ERROR SystemManager::RegisterModuleConnects(std::string pipelineName, ModuleConnectDesc *connnectDesc,
    int moduleConnectCount)
{
    std::map<std::string, std::map<ModuleType, ModulesInfo>>::iterator iter = pipelineMap_.find(pipelineName);
    if (iter == pipelineMap_.end()) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    std::map<ModuleType, ModulesInfo> &modulesInfoMap = iter->second;
    // add connect
    for (int i = 0; i < moduleConnectCount; i++) {
        ModuleConnectDesc connectDesc = connnectDesc[i];
        LogDebug << "Add Connect " << connectDesc.moduleSend << " " << connectDesc.moduleRecv << " type " <<
            connectDesc.connectType;
        std::map<ModuleType, ModulesInfo>::iterator iterSend, iterRecv;
        iterSend = modulesInfoMap.find(connectDesc.moduleSend);
        iterRecv = modulesInfoMap.find(connectDesc.moduleRecv);
        if (iterSend == modulesInfoMap.end() || iterRecv == modulesInfoMap.end()) {
            LogFatal << "Cann't find Module";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        ModulesInfo &moduleInfoRecv = iterRecv->second;
        // create input queue fro recv module
        if (moduleInfoRecv.inputQueueVec.size() == 0) {
            for (unsigned int j = 0; j < moduleInfoRecv.moduleVec.size(); j++) {
                moduleInfoRecv.inputQueueVec.push_back(
                    std::make_shared<BlockingQueue<std::shared_ptr<void>>>(MODULE_QUEUE_SIZE));
            }
            // register  queue
            RegisterInputVec(pipelineName, connectDesc.moduleRecv, moduleInfoRecv.inputQueueVec);
        }
        RegisterOutputModule(pipelineName, connectDesc.moduleSend, connectDesc.moduleRecv, connectDesc.connectType,
            moduleInfoRecv.inputQueueVec);
    }
    return APP_ERR_OK;
}

APP_ERROR SystemManager::RegisterInputVec(std::string pipelineName, ModuleType moduleType,
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueVec)
{
    std::map<std::string, std::map<ModuleType, ModulesInfo>>::iterator pipelineIter = pipelineMap_.find(pipelineName);
    std::map<ModuleType, ModulesInfo> modulesInfoMap;
    if (pipelineIter == pipelineMap_.end()) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    modulesInfoMap = pipelineIter->second;

    // set inputQueue
    std::map<ModuleType, ModulesInfo>::iterator iter = modulesInfoMap.find(moduleType);
    if (iter != modulesInfoMap.end()) {
        ModulesInfo moduleInfo = iter->second;
        if (moduleInfo.moduleVec.size() != inputQueVec.size()) {
            return APP_ERR_COMM_FAILURE;
        }
        for (unsigned int j = 0; j < moduleInfo.moduleVec.size(); j++) {
            std::shared_ptr<ModuleBase> moduleInstance = moduleInfo.moduleVec[j];
            moduleInstance->SetInputVec(inputQueVec[j]);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR SystemManager::RegisterOutputModule(std::string pipelineName, ModuleType moduleSend, ModuleType moduleRecv,
    ModuleConnectType connectType, std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec)
{
    std::map<std::string, std::map<ModuleType, ModulesInfo>>::iterator pipelineIter = pipelineMap_.find(pipelineName);
    std::map<ModuleType, ModulesInfo> modulesInfoMap;
    if (pipelineIter == pipelineMap_.end()) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    modulesInfoMap = pipelineIter->second;

    // set outputInfo
    std::map<ModuleType, ModulesInfo>::iterator iter = modulesInfoMap.find(moduleSend);
    if (iter != modulesInfoMap.end()) {
        ModulesInfo moduleInfo = iter->second;
        for (unsigned int j = 0; j < moduleInfo.moduleVec.size(); j++) {
            std::shared_ptr<ModuleBase> moduleInstance = moduleInfo.moduleVec[j];
            moduleInstance->SetOutputInfo(moduleRecv, connectType, outputQueVec);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR SystemManager::InitPipelineModule()
{
    return APP_ERR_OK;
}

APP_ERROR SystemManager::RunPipeline()
{
    LogInfo << "SystemManager: begin to run pipeline.";

    // start the thread of the corresponding module
    std::map<ModuleType, ModulesInfo> modulesInfoMap;
    std::shared_ptr<ModuleBase> moduleInstance;
    std::map<std::string, std::map<ModuleType, ModulesInfo>>::iterator pipelineIter;
    for (pipelineIter = pipelineMap_.begin(); pipelineIter != pipelineMap_.end(); pipelineIter++) {
        modulesInfoMap = pipelineIter->second;

        std::map<ModuleType, ModulesInfo>::iterator iter;
        for (iter = modulesInfoMap.begin(); iter != modulesInfoMap.end(); iter++) {
            ModulesInfo modulesInfo = iter->second;
            for (uint32_t i = 0; i < modulesInfo.moduleVec.size(); i++) {
                moduleInstance = modulesInfo.moduleVec[i];
                APP_ERROR ret = moduleInstance->Run();
                if (ret != APP_ERR_OK) {
                    LogFatal << "SystemManager: fail to run module ";
                    return ret;
                }
            }
        }
    }

    LogInfo << "SystemManager: run pipeline success.";
    return APP_ERR_OK;
}

void SystemManager::SendDataToModule(std::string pipelineName, ModuleType moduleType, 
    std::shared_ptr<void> data, uint32_t channelId)
{
    auto pipelineIter = pipelineMap_.find(pipelineName);
    std::map<ModuleType, ModulesInfo> modulesInfoMap;
    if (pipelineIter == pipelineMap_.end()) {
        LogFatal << "SystemManager: no pipeline :" << pipelineName;
        return;
    }

    modulesInfoMap = pipelineIter->second;
    if (modulesInfoMap.find(moduleType) == modulesInfoMap.end()) {
        LogFatal << "SystemManager: pipeline " << pipelineName << " no module type: " << moduleType;
        return;
    }

    if (modulesInfoMap[moduleType].inputQueueVec.size() <= channelId) {
        LogFatal << "SystemManager: pipeline " << pipelineName << " module type " 
            << moduleType << " input channel is out of range (" 
            << modulesInfoMap[moduleType].inputQueueVec.size() 
            << ") vs (" << channelId << ") module size " 
            << modulesInfoMap[moduleType].moduleVec.size();
        return;
    }
    modulesInfoMap[moduleType].inputQueueVec[channelId]->Push(data, true);
}

APP_ERROR SystemManager::DeInit(void)
{
    LogInfo << "SystemManager: begin to deinit system manager.";
    // 1. DeInit pipeline module
    APP_ERROR ret = DeInitPipelineModule();
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to deinit pipeline module, ret[%d]" << ret << ".";
        return ret;
    }

#ifdef ASCEND_FACE_USE_ACL
    ModelResource::GetInstance().Release();
    ResourceManager::GetInstance()->Release();
#endif

    return ret;
}

void SystemManager::StopModule(std::shared_ptr<ModuleBase> moduleInstance)
{
    LogInfo << moduleInstance->moduleName_ << "[" << moduleInstance->instanceId_ << "] stop begin";
    APP_ERROR ret = moduleInstance->Stop();
    if (ret != APP_ERR_OK) {
        LogFatal << moduleInstance->moduleName_ << " [" << moduleInstance->instanceId_ << "] deinit failed";
    } else {
        LogInfo << moduleInstance->moduleName_ << " [" << moduleInstance->instanceId_ << "] deinit success";
    }
}

APP_ERROR SystemManager::DeInitPipelineModule()
{
    LogInfo << "SystemManager: begin to deinit pipeline module.";
    std::map<ModuleType, ModulesInfo> modulesInfoMap;
    std::shared_ptr<ModuleBase> moduleInstance;
    std::map<std::string, std::map<ModuleType, ModulesInfo>>::iterator pipelineIter;
    for (pipelineIter = pipelineMap_.begin(); pipelineIter != pipelineMap_.end(); pipelineIter++) {
        modulesInfoMap = pipelineIter->second;
        std::map<ModuleType, ModulesInfo>::iterator iter;
        for (iter = modulesInfoMap.begin(); iter != modulesInfoMap.end(); iter++) {
            ModulesInfo modulesInfo = iter->second;
            for (auto &moduleInstance : modulesInfo.moduleVec) {
                StopModule(moduleInstance);
            }
        }
    }
    pipelineMap_.clear();
    LogInfo << "SystemManager: deinit pipeline success.";
    return APP_ERR_OK;
}
} // namespace ascendFaceRecognition
