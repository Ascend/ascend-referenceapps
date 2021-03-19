/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
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

#include "AclProcess/AclProcess.h"
#include "ResourceManager/ResourceManager.h"
#include "CommandLine/CommandLine.h"
#include "CommonDataType/CommonDataType.h"
#include "Log/Log.h"
#include "ErrorCode/ErrorCode.h"
#include "CommandParser/CommandParser.h"
#include "ConfigParser/ConfigParser.h"

namespace {
/*
 * @description: Parse input parameters of command line and check the validity of them
 * @param argc     number of input arguments
 * @param argv     input arguments pointer
 * @param options  used to store the parsed parameter pairs
 * @return: APP_ERR_OK success
 * @return: Other values failure
 * @attention: The program will exit when parameter parsed error
 */
APP_ERROR ParseAndCheckArgs(int argc, const char *argv[], CommandParser &options)
{
    // Construct the command parser
    options.AddOption("-i", "./data/test.yuv", "Optional. Specify the input image, default: ./data/test.yuv");
    options.AddOption("-width", "0", "Requested. Specify the width of input image");
    options.AddOption("-height", "0", "Requested. Specify the height of input image");
    options.AddOption("-format", "-1",
        "Requested. Specify the format of input image, {1:NV12, 2:NV21, 7:YUYV, 8:UYVY, 9:YVYU, 10:VYUY}");

    options.ParseArgs(argc, argv);
    std::string fileName = options.GetStringOption("-i");                            // Parse the input image file
    int height = options.GetIntOption("-height");                                    // Parse the height of input image
    int width = options.GetIntOption("-width");                                      // Parse the width of input image
    acldvppPixelFormat format = acldvppPixelFormat(options.GetIntOption("-format")); // Parse the format of input image
    // check the validity of input argument
    APP_ERROR ret = CheckArgs(fileName, height, width, format);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    // get the absolute path of input file
    char absPath[PATH_MAX];
    if (realpath(fileName.c_str(), absPath) == nullptr) {
        LogError << "Failed to get the real path of input file.";
        return APP_ERR_COMM_NO_EXIST;
    }
    std::string path(argv[0], argv[0] + strlen(argv[0]));
    ChangeDir(path.c_str()); // Change the directory to the execute file
    return ret;
}

/*
 * @description: Get the device id from the configuration file and set it to resourceInfo
 * @param  configData    config data obtained from the configuration file setup.config
 * @param  resourceInfo  used to store device id
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR ReadConfigFromFile(ConfigParser &configData, ResourceInfo &resourceInfo)
{
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
        LogError << "deviceId:" << deviceId << " is less than 0, not valid, please check.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    resourceInfo.deviceIds.insert(deviceId);
    return APP_ERR_OK;
}

/*
 * @description: Entry function for calling function in class AclProcess to encode image
 * @param  resourceInfo  device id obtained from the configuration file setup.config
 * @param  imageFile     absolute path of the input image
 * @param  width         width of the input image
 * @param  height        height of the input image
 * @param  format        pixel format supported by dvpp
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR Process(ResourceInfo &resourceInfo, std::string fileName, int width, int height, acldvppPixelFormat format)
{
    std::shared_ptr<ResourceManager> instance = ResourceManager::GetInstance();
    int deviceId = *(resourceInfo.deviceIds.begin());
    aclrtContext context = instance->GetContext(deviceId);
    AclProcess aclProcess(deviceId, context); // Initialize AclProcess module
    APP_ERROR ret = aclProcess.InitResource();
    if (ret != APP_ERR_OK) {
        aclProcess.Release();
        return ret;
    }

    ret = aclProcess.Process(fileName, width, height, format);
    if (ret != APP_ERR_OK) {
        aclProcess.Release();
        return ret;
    }

    aclProcess.Release();
    return APP_ERR_OK;
}
}

int main(int argc, const char *argv[])
{
    AtlasAscendLog::Log::LogInfoOn();
    CommandParser options;
    // Parse and check arguments
    APP_ERROR ret = ParseAndCheckArgs(argc, argv, options);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    std::string fileName = options.GetStringOption("-i");
    int height = options.GetIntOption("-height");
    int width = options.GetIntOption("-width");
    acldvppPixelFormat format = acldvppPixelFormat(options.GetIntOption("-format"));

    ConfigParser config;
    // Initialize the config parser module
    if (config.ParseConfig("./data/config/setup.config") != APP_ERR_OK) {
        return ret;
    }
    ResourceInfo resourceInfo;
    resourceInfo.aclConfigPath = "./data/config/acl.json";
    // Read config data from file
    ret = ReadConfigFromFile(config, resourceInfo);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    std::shared_ptr<ResourceManager> instance = ResourceManager::GetInstance();
    ret = instance->InitResource(resourceInfo);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    ret = Process(resourceInfo, fileName, width, height, format);
    if (ret != APP_ERR_OK) {
        instance->Release();
        return ret;
    }
    instance->Release();
    return 0;
}
