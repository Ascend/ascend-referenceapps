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

#include <iostream>
#include "Log/Log.h"
#include "ConfigParser/ConfigParser.h"
#include "CommandParser/CommandParser.h"
#include "AclProcess.h"
#include "VdecUtils.h"
using namespace std;


/*
 * @description: Check the validity of input argument
 * @param: filename specifies the file path
 * @return: APP_ERR_OK if success, other values if failure
 */
static APP_ERROR CheckArgs(const std::string &fileName)
{
    int folderExist = access(fileName.c_str(), R_OK);
    if (folderExist == -1) {
        LogError << "Input file " << fileName << " doesn't exist or read failed!";
        return APP_ERR_COMM_NO_EXIST;
    }
    return APP_ERR_OK;
}

/*
 * @description: Parse input parameters of command line and check the validity of them
 * @param: argc specifies the num of input arguments
 * @param: argv specifies the input arguments
 * @return: APP_ERR_OK if success, other values if failure
 */
static APP_ERROR ParseAndCheckArgs(int argc, const char *argv[], CommandParser &options)
{
    // Construct the command parser
    options.AddOption("-i", "./data/test.h264", "Optional. Specify the input image, default: ./data/test.h264");
    options.ParseArgs(argc, argv);
    std::string fileName = options.GetStringOption("-i"); // Parse the input image file
    // check the validity of input argument
    APP_ERROR ret = CheckArgs(fileName);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    return ret;
}

int main(int argc, const char *argv[])
{
    AtlasAscendLog::Log::LogInfoOn();
    CommandParser options;
    // Parse and check arguments
    APP_ERROR ret = ParseAndCheckArgs(argc, argv, options);
    if (ret != APP_ERR_OK) {
        return APP_ERR_COMM_FAILURE;
    }
    std::string fileName = options.GetStringOption("-i");

    AclProcess processSample;
    processSample.SetInputFilePath(fileName);
    ret = processSample.InitResource();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to initialize resource, ret = " << ret;
        return APP_ERR_ACL_FAILURE;
    }

    ret = processSample.DoVdecProcess();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to do vdec process, ret = " << ret;
        return APP_ERR_ACL_FAILURE;
    }

    LogInfo << "Execute sample successfully";
    return APP_ERR_OK;
}