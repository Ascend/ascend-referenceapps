/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include "AclProcess/AclProcess.h"
#include "ResourceManager/ResourceManager.h"
#include "CommonDataType/CommonDataType.h"
#include "Log/Log.h"
#include "ErrorCode/ErrorCode.h"
#include "CommandParser/CommandParser.h"
#include "ConfigParser/ConfigParser.h"

namespace {
/*
 * @description Parse And check the command line
 * @param argc Num of input arguments
 * @param argv Input arguments
 * @param options Command parser
 * @return APP_ERROR error code
 * @attention this function will cause the program to exit directly when parse command line error
 */
APP_ERROR ParseAndCheckArgs(int argc, const char *argv[], CommandParser &options)
{
    // Construct the command parser
    options.AddOption("-i", "./data/test.jpg", "Optional. Specify the input image, default: ./data/test.jpg");
    options.ParseArgs(argc, argv);
    std::string fileName = options.GetStringOption("-i"); // Parse the input image file
    // check the validity of input argument
    // when the path does not exist or some other error occurred, access function return -1
    if (access(fileName.c_str(), R_OK) == -1) {
        LogError << "[ERROR] Input file " << fileName << " doesn't exist or read failed!";
        return APP_ERR_COMM_NO_EXIST;
    }

    return APP_ERR_OK;
}

/*
 * @description Get model info from config file
 * @param configData Config parser
 * @param modelInfos model info collection
 * @return APP_ERROR error code
 */
APP_ERROR ReadModelConfig(ConfigParser& configData, std::vector<ModelInfo>& modelInfos)
{
    ModelInfo modelInfo;
    std::string tmp;
    APP_ERROR ret = configData.GetStringValue("model_path", tmp); // Get the model path from config file
    if (ret != APP_ERR_OK) {
        LogError << "ModelPath in the config file is invalid.";
        return ret;
    }
    std::string resolvedPath;
    ret = GetRealPath(tmp, resolvedPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    // Check the validity of model path
    int folderExist = access(resolvedPath.c_str(), R_OK);
    if (folderExist == -1) {
        LogError << "ModelPath " << resolvedPath << " doesn't exist or read failed.";
        return APP_ERR_COMM_NO_EXIST;
    }
    modelInfo.modelPath = std::string(resolvedPath); // Set the absolute path of model file
    modelInfo.method = LOAD_FROM_FILE_WITH_MEM; // Set the ModelLoadMethod
    int tmpNum;
    ret = configData.GetIntValue("model_width", tmpNum); // Get the input width of model from config file
    if (ret != APP_ERR_OK) {
        LogError << "Model width in the config file is invalid.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    modelInfo.modelWidth = tmpNum;

    ret = configData.GetIntValue("model_height", tmpNum); // Get the input height of model from config file
    if (ret != APP_ERR_OK) {
        LogError << "Model height in the config file is invalid.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    modelInfo.modelHeight = tmpNum;
    modelInfo.modelName = "resnet50";
    modelInfos.push_back(std::move(modelInfo));
    return APP_ERR_OK;
}

/*
 * @description Get device id and single operator model path from config file
 * @param configData Config parser
 * @param resourceInfo resource info of deviceId, model info, single Operator Path, etc
 * @return APP_ERROR error code
 */
APP_ERROR ReadConfigFromFile(ConfigParser& configData, ResourceInfo& resourceInfo)
{
    std::string tmp;
    // Get the device id used by application from config file
    int deviceId;
    APP_ERROR ret = configData.GetIntValue("device_id", deviceId);
    // Convert from string to digital number
    if (ret != APP_ERR_OK) {
        LogError << "Device id is not digit, please check.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    // Check validity of device id
    if (deviceId < 0) {
        LogError << "deviceId = " << deviceId << " is less than 0, not valid, please check.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    resourceInfo.deviceIds.insert(deviceId);
    std::vector<ModelInfo> modelInfos;
    ret = ReadModelConfig(configData, modelInfos); // Read model config from the config file
    if (ret != APP_ERR_OK) {
        LogError << "Failed to get model info from config file, ret = " << ret;
        return APP_ERR_COMM_READ_FAIL;
    }
    DeviceResInfo deviceResInfo;
    deviceResInfo.modelInfos = std::move(modelInfos);
    resourceInfo.deviceResInfos[deviceId] = std::move(deviceResInfo);

    ret = configData.GetStringValue("single_op_model", tmp); // Get the path of single operator models
    if (ret != APP_ERR_OK) {
        LogError << "Single_op_model in config file is invalid.";
        return APP_ERR_COMM_READ_FAIL;
    }
    std::string resolvedPath;
    // Get the absolute path of single operator models
    ret = GetRealPath(tmp, resolvedPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    // Check the validity of single operator models
    int folderExist = access(resolvedPath.c_str(), R_OK);
    if (folderExist == -1) {
        LogError << "SingleOpPath " << resolvedPath << " doesn't exist or read failed.";
        return APP_ERR_COMM_NO_EXIST;
    }
    resourceInfo.singleOpFolderPath = std::string(resolvedPath); // Set single operator path
    return APP_ERR_OK;
}

/*
 * @description Initialize and run AclProcess module
 * @param resourceInfo resource info of deviceIds, model info, single Operator Path, etc
 * @param file the absolute path of input file
 * @return APP_ERROR error code
 */
APP_ERROR Process(ResourceInfo& resourceInfo, const std::string& file)
{
    std::shared_ptr<ResourceManager> instance = ResourceManager::GetInstance();
    int deviceId = *(resourceInfo.deviceIds.begin());
    aclrtContext context = instance->GetContext(deviceId);
    // Initialize an AclProcess module
    AclProcess aclProcess(deviceId, resourceInfo.deviceResInfos[deviceId].modelInfos[0],
        resourceInfo.singleOpFolderPath, context); // Initialize an AclProcess module
    APP_ERROR ret = aclProcess.InitResource();
    if (ret != APP_ERR_OK) {
        aclProcess.Release();
        return ret;
    }
    ret = aclProcess.Process(file);
    if (ret != APP_ERR_OK) {
        aclProcess.Release();
        return ret;
    }
    aclProcess.Release();
    return APP_ERR_OK;
}
}

int main(int argc, const char* argv[])
{
    auto startTime = std::chrono::high_resolution_clock::now();
    // Print log with the level of info and above
    AtlasAscendLog::Log::LogInfoOn();
    CommandParser options;
    // Parse and check arguments
    APP_ERROR ret = ParseAndCheckArgs(argc, argv, options);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    std::string fileName = options.GetStringOption("-i");
    // get the absolute path of input file
    std::string resolvedPath;
    ret = GetRealPath(fileName, resolvedPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    std::string path(argv[0], argv[0] + strlen(argv[0]));
    ret = ChangeDir(path.c_str()); // Change the directory to the execute file
    if (ret != APP_ERR_OK) {
        LogError << "Failed to change directory to " << path.c_str();
        return ret;
    }
    ConfigParser config;
    // Initialize the config parser module
    if (config.ParseConfig("./data/config/setup.config") != APP_ERR_OK) {
        return APP_ERR_COMM_INIT_FAIL;
    }
    ResourceInfo resourceInfo;
    // The relevant information of dump data can be configured in "acl.json",
    // it is an empty file in this sample because we don't need.
    // For details, please refer to the description of aclInit function
    resourceInfo.aclConfigPath = "./data/config/acl.json";
    // Read config data from file
    ret =  ReadConfigFromFile(config, resourceInfo);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    // instance is a singleton pointer of ResourceManager class, used for resource management
    std::shared_ptr<ResourceManager> instance = ResourceManager::GetInstance();
    // InitResource will create context for device
    ret = instance->InitResource(resourceInfo);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    std::string file(resolvedPath);
    ret = Process(resourceInfo, file);
    if (ret != APP_ERR_OK) {
        instance->Release();
        return ret;
    }
    instance->Release();
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    LogDebug << "[Main Delay] cost: " << costMs << "ms. ";
    return 0;
}
