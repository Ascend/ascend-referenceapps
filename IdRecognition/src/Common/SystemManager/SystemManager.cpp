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

namespace ascendFaceRecognition {
SystemManager::~SystemManager()
{
    DeInit();
}

APP_ERROR SystemManager::Init(std::string &configPath, SystemInitArgs &initArgs)
{
    LogInfo << "SystemManager: begin to init.";

    // 0. init variables
    pipeMode_ = initArgs.pipelineMode;
    moduleTypeCount_ = initArgs.moduleTypeCount;
    pipelineDesc_ = initArgs.pipelineDesc;
    faceFeature_ = initArgs.faceFeature;
    inputQueVec_ = initArgs.inputQueVec;
    outputQueVec_ = initArgs.outputQueVec;

    if (pipeMode_ == PIPELINE_MODE_REG) {
        LogDebug << "SystemManager: pipeline running at registration mode.";
    } else if (pipeMode_ == PIPELINE_MODE_SEARCH) {
        LogDebug << "SystemManager: pipeline running at search mode.";
    } else {
        LogFatal << "SystemManager: invalid pipeline mode.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    // 1. load and parse config file
    APP_ERROR ret = configParser_.ParseConfig(configPath);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: cannot parse file.";
        return ret;
    }

    // 2. Init device
    ret = InitDevice();
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to init device.";
        return ret;
    }

    // 3. Init pipeline module
    ret = InitPipelineModule();
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to init pipeline module.";
        return ret;
    }

    return ret;
}

APP_ERROR SystemManager::InitDevice()
{
    LogInfo << "SystemManager: begin to init device.";
    std::string itemCfgStr;

    itemCfgStr = "SystemConfig.deviceId";
    APP_ERROR ret = configParser_.GetIntValue(itemCfgStr, deviceId_);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to get device id.";
        return ret;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to get run mode of device, ret=" << ret << ".";
        return ret;
    }

    if (runMode_ == ACL_DEVICE) {
        deviceId_ = 0;
    } else if (deviceId_ < 0) {
        LogFatal << "SystemManager: invalid device id, deviceId=" << deviceId_ << ".";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    ret = aclrtSetDevice(deviceId_);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to open device " << deviceId_ << ", ret=" << ret << ".";
        return ret;
    }

    ret = aclrtCreateContext(&aclContext_, deviceId_);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to create context of device " << deviceId_ << ", ret=" << ret << ".";
        return ret;
    }

    return ret;
}

APP_ERROR SystemManager::InitModuleInstance(int moduleIdx, int instanceId, std::shared_ptr<ModuleBase> moduleInstance,
    BlockingQueue<std::shared_ptr<void>> &moduleInputQueue)
{
    LogInfo << "SystemManager: begin to init module instance.";
    ModuleInitArgs initArgs;

    initArgs.context = aclContext_;
    initArgs.runMode = runMode_;
    initArgs.moduleName = pipelineDesc_[moduleIdx].moduleName;
    initArgs.instanceId = instanceId;
    initArgs.pipelineMode = pipeMode_;
    initArgs.userData = faceFeature_;
    initArgs.inputQueue = &moduleInputQueue;
    if (pipelineDesc_[moduleIdx].nextModuleType != MT_BOTTOM) {
        std::map<ModuleType, ModulesInfo>::iterator it;

        // Find the element with key 'nextModuleType'
        it = modulesInfoMap_.find(pipelineDesc_[moduleIdx].nextModuleType);
        // Check if element exists in map or not
        if (it != modulesInfoMap_.end()) {
            initArgs.outputQueVec = it->second.inputQueueVec;
        } else {
            // Element with key 'hat' Not Found
            LogFatal << "SystemManager: fail to find output queue of module, name = "
                     << pipelineDesc_[moduleIdx].moduleName << ", instance id = " << instanceId << ".";
            return APP_ERR_COMM_NO_EXIST;
        }
    } else {
        initArgs.outputQueVec = outputQueVec_;
    }

    APP_ERROR ret = moduleInstance->Init(configParser_, initArgs);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to init module, name = " << pipelineDesc_[moduleIdx].moduleName
                 << ", instance id = " << instanceId << ".";
        return ret;
    }
    LogDebug << "SystemManager: module " << initArgs.moduleName << "[" << instanceId << "] init success.";
    return ret;
}

APP_ERROR SystemManager::InitPipelineModule()
{
    LogInfo << "SystemManager: begin to init pipeline modules.";
    int channelCount = 0;
    std::string itemCfgStr;

    // 1. Get channel count and set for related modules
    itemCfgStr = "SystemConfig.channelCount";
    APP_ERROR ret = configParser_.GetIntValue(itemCfgStr, channelCount);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to get channel count.";
        return ret;
    }
    if (channelCount <= 0) {
        LogFatal << "SystemManager: invalid channel count, ChannelCount=" << channelCount << ".";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    for (int i = 0; i < moduleTypeCount_; i++) {
        if (pipelineDesc_[i].channelCount == -1) {
            pipelineDesc_[i].channelCount = channelCount;
        }
    }

    // 2. Create and init module instance of pipelineDesc_ in reversed order
    /* e.g., moduleDescSearch defines the order of modules, then create module "FaceSearch", "FaceFeature",
                "WarpAffine"...
       ModuleDesc moduleDescSearch[MODULE_TYPE_COUNT_SEARCH] =
        {{-1, MT_IMAGE_DECODER, "ImageDecoder", MT_FACE_DETECTION},
            {1, MT_FACE_DETECTION, "FaceDetection", MT_FACE_LANDMARK},
            {FACE_LANDMARK_COUNT, MT_FACE_LANDMARK, "FaceLandmark", MT_WARP_AFFINE},
            {1, MT_WARP_AFFINE, "WarpAffine", MT_FACE_FEATURE},
            {FACE_FEATURE_COUNT, MT_FACE_FEATURE, "FaceFeature", MT_FACE_SEARCH},
            {-1, MT_FACE_SEARCH, "FaceSearch", MT_OTHERS}};
    */
    for (int i = (moduleTypeCount_ - 1); i >= 0; i--) {
        ModulesInfo modulesInfo;
        for (int j = 0; j < pipelineDesc_[i].channelCount; j++) {
            std::shared_ptr<ModuleBase> moduleInstance;
            std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> moduleInputQueue = nullptr;

            // 2.1 create module instance and keep in the ModulesInfo
            LogDebug << "SystemManager: begin to make modeule, name " << pipelineDesc_[i].moduleName << ", type "
                     << pipelineDesc_[i].moduleType << ", instance " << j << ".";
            moduleInstance = std::shared_ptr<ModuleBase>(ModuleFactory::MakeModule(pipelineDesc_[i].moduleType));
            modulesInfo.moduleVec.push_back(moduleInstance);

            // 2.2 create input queue of the module and keep in the ModulesInfo
            if (i != 0) {
                moduleInputQueue = std::make_shared<BlockingQueue<std::shared_ptr<void>>>(MODULE_QUEUE_SIZE);
                modulesInfo.inputQueueVec.push_back(moduleInputQueue);
            } else if (i == 0) {
                moduleInputQueue = inputQueVec_[j];
                modulesInfo.inputQueueVec.push_back(moduleInputQueue);
            }

            // 2.3 init instance
            ret = InitModuleInstance(i, j, moduleInstance, *moduleInputQueue);
            if (ret != APP_ERR_OK) {
                return ret;
            }
        }
        // 2.4 keep ModulesInfo in the modulesInfoMap_
        modulesInfoMap_[pipelineDesc_[i].moduleType] = modulesInfo;
    }

    return ret;
}

APP_ERROR SystemManager::RunPipeline()
{
    LogInfo << "SystemManager: begin to run pipeline.";
    std::shared_ptr<ModuleBase> moduleInstance;
    // run pipeline modules in reversed order
    for (int i = (moduleTypeCount_ - 1); i >= 0; i--) {
        for (int j = 0; j < pipelineDesc_[i].channelCount; j++) {
            moduleInstance = modulesInfoMap_[pipelineDesc_[i].moduleType].moduleVec[j];

            APP_ERROR ret = moduleInstance->Run();
            if (ret != APP_ERR_OK) {
                LogFatal << "SystemManager: fail to run module " << pipelineDesc_[i].moduleName << "[" << j << "].";
                return ret;
            }
            LogDebug << "SystemManager: run module " << pipelineDesc_[i].moduleName << "[" << j << "] success.";
        }
    }

    return APP_ERR_OK;
}

APP_ERROR SystemManager::DeInit(void)
{
    static bool isDone = false;
    if (isDone) {
        return APP_ERR_OK;
    } else {
        isDone = true;
    }

    LogInfo << "SystemManager: begin to deinit system manager.";

    // 0. Set context
    APP_ERROR ret = aclrtSetCurrentContext(aclContext_);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to set context, ret[%d]" << ret << ".";
        return ret;
    }

    // 1. DeInit pipeline module
    ret = DeInitPipelineModule();
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to deinit pipeline module, ret[%d]" << ret << ".";
        return ret;
    }

    // 2. DeInit device
    ret = DeInitDevice();
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to deinit device, ret[%d]" << ret << ".";
        return ret;
    }

    return ret;
}

APP_ERROR SystemManager::DeInitPipelineModule()
{
    LogInfo << "SystemManager: begin to deinit pipeline module.";
    std::shared_ptr<ModuleBase> moduleInstance;

    // deinit pipeline modules in order
    for (int i = 0; i < moduleTypeCount_; i++) {
        for (int j = 0; j < pipelineDesc_[i].channelCount; j++) {
            moduleInstance = modulesInfoMap_[pipelineDesc_[i].moduleType].moduleVec[j];

            APP_ERROR ret = moduleInstance->DeInit();
            if (ret != APP_ERR_OK) {
                LogFatal << "SystemManager: fail to deinit module " << pipelineDesc_[i].moduleName << "[" << j << "].";
                return ret;
            }
        }
    }

    return APP_ERR_OK;
}

APP_ERROR SystemManager::DeInitDevice() const
{
    LogInfo << "SystemManager: begin to deinit device " << deviceId_ << ".";

    APP_ERROR ret = aclrtDestroyContext(aclContext_);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to destroy context.";
        return ret;
    }
    ret = aclrtResetDevice(deviceId_);
    if (ret != APP_ERR_OK) {
        LogFatal << "SystemManager: fail to reset device " << deviceId_ << ".";
        return ret;
    }

    return ret;
}
const std::map<ModuleType, ModulesInfo> &SystemManager::GetModulesInfoMap() const
{
    return modulesInfoMap_;
}

const ModuleDesc *SystemManager::GetPipelineDesc() const
{
    return pipelineDesc_;
}

const std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> &SystemManager::GetOutputQueVec()
{
    return outputQueVec_;
}
} // namespace ascendFaceRecognition
