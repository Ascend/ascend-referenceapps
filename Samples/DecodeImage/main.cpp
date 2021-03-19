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

#include <cstdio>
#include <sys/time.h>
#include "AclProcess/AclProcess.h"
#include "ResourceManager/ResourceManager.h"
#include "CommonDataType/CommonDataType.h"
#include "Log/Log.h"
#include "ErrorCode/ErrorCode.h"
#include "CommandParser/CommandParser.h"
#include "ConfigParser/ConfigParser.h"

/*
 * @description: parse the input arguments and check validity
 * @param: argc Num of input arguments
 * @param: argv Input arguments
 * @param: options Command parser
 * @return: aclError which is error code of ACL API
 */

namespace {
APP_ERROR ParseAndCheckArgs(int argc, const char *argv[], CommandParser& options)
{
    // Construct the command parser
    options.AddOption("-i", "./data/test.jpg", "Optional. Specify the input image, default: ./data/test.jpg");
    options.ParseArgs(argc, argv);

    // Check the validity of input argument of image file
    std::string fileName = options.GetStringOption("-i");
    APP_ERROR ret = access(fileName.c_str(), R_OK);
    if (ret != APP_ERR_OK) {
        LogError << "Input file " << fileName << " doesn't exist or not find '-i', ret = " << ret;
        return APP_ERR_COMM_NO_EXIST;
    }
    return ret;
}

/*
* @description: read config file
* @param: configData specify the config file
* @param: resourceInfo is used to save the device ID of the specified chip
* @param: resizeWidth is used to save the specified scaling width parameter.
* @param: resizeHeight is used to save the specified scaling height parameter.
* @return: aclError which is error code of ACL API
*/
APP_ERROR ReadConfigFromFile(ConfigParser& configData, ResourceInfo& resourceInfo, uint32_t& resizeWidth,
    uint32_t& resizeHeight)
{
    // Obtain the ID and check its validity
    int deviceId = 0;
    APP_ERROR ret = configData.GetIntValue("device_id", deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "deviceId is not digit, please check, ret = " << ret;
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (deviceId < 0) {
        LogError << "deviceId < 0, not valid, please check, ret = " << ret;
        return APP_ERR_COMM_INVALID_PARAM;
    }
    resourceInfo.deviceIds.insert(deviceId);

    // Obtains the width for scaling
    int tmpNum = 0;
    ret = configData.GetIntValue("resize_width", tmpNum);
    if (ret != APP_ERR_OK) {
        LogError << "resize_width in the config file is invalid, ret = " << ret;
        return APP_ERR_COMM_INVALID_PARAM;
    }
    resizeWidth = tmpNum;

    // Obtains the height for scaling
    ret = configData.GetIntValue("resize_height", tmpNum);
    if (ret != APP_ERR_OK) {
        LogError << "resize_height in the config file is invalid, ret = " << ret;
        return APP_ERR_COMM_INVALID_PARAM;
    }
    resizeHeight = tmpNum;
    return APP_ERR_OK;
}
}

/*
 * @description: program entry
 * @param: argc specify the number of argv
 * @param: argv contains the path and command parameters of the program
 * @return: program exit status
 */
int main(int argc, const char* argv[])
{
    AtlasAscendLog::Log::LogInfoOn();
    // Parse the parameters of the main function
    CommandParser options;
    APP_ERROR ret = ParseAndCheckArgs(argc, argv, options);
    if (ret != APP_ERR_OK) {
        return APP_ERR_COMM_FAILURE;
    }

    // Change the directory to the execute file
    std::string exePath(argv[0], argv[0] + strlen(argv[0]));
    ChangeDir(exePath.c_str());

    // Initialize the config parser module
    ConfigParser config;
    if (config.ParseConfig("./data/config/setup.config") != APP_ERR_OK) {
        return APP_ERR_COMM_INIT_FAIL;
    }

    // Read config data from config file
    ResourceInfo resourceInfo;
    resourceInfo.aclConfigPath = "./data/config/acl.json";
    uint32_t resizeWidth, resizeHeight;
    ret =  ReadConfigFromFile(config, resourceInfo, resizeWidth, resizeHeight);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    std::shared_ptr<ResourceManager> instance = ResourceManager::GetInstance();
    ret = instance->InitResource(resourceInfo);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    // Initialize an AclProcess module
    int deviceId = *(resourceInfo.deviceIds.begin());
    aclrtContext context = instance->GetContext(deviceId);
    AclProcess aclProcess(resizeWidth, resizeHeight, context);
    ret = aclProcess.InitResource();
    if (ret != APP_ERR_OK) {
        instance->Release();
        return ret;
    }

    ret = aclProcess.Process(options.GetStringOption("-i"));
    if (ret != APP_ERR_OK) {
        LogInfo << "Failed to execute the aclprocess, ret = " << ret;
    }

    ret = aclProcess.Release();
    if (ret != APP_ERR_OK) {
        LogInfo << "Failed to execute the aclRelease, ret = " << ret;
    }
    instance->Release();
    return 0;
}
