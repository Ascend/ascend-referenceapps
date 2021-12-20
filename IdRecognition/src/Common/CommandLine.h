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

#ifndef COMMAND_LINE_H
#define COMMAND_LINE_H
#include "CommandParser/CommandParser.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
#include "Statistic/Statistic.h"

// command parameters of device process
typedef struct CommandParameters {
    std::string aclConfig;
    std::string regConfig;
    std::string searchConfig;
    int runMode;
    int debugLevel;
} CmdParams;

APP_ERROR ParseACommandLine(int argc, const char *argv[], CmdParams &cmdParams)
{
    LogDebug << "Begin to parse and check command line.";
    APP_ERROR ret = APP_ERR_OK;
    CommandParser option;

    option.AddOption("-sample_command", "", "./AclIDRcgHost -debug_level 0 -run_mode 0");
    option.AddOption("-acl_setup", "./config/aclInit.config", "the config file using for AscendCL init.");
    option.AddOption("-setup_reg", "./config/facedemoReg.config",
        "the config file using for face registration pipeline.");
    option.AddOption("-setup_search", "./config/facedemoSearch.config",
        "the config file using for face search pipeline.");
    option.AddOption("-run_mode", "0", "0-run face search pipeline only, 1-run face registration pipeline only.");
    option.AddOption("-debug_level", "1", "debug level:0-debug, 1-info, 2-warn, 3-error, 4-fatal, 5-off.");
    option.AddOption("-stats", "false", "to enable the statistic, false-disabled statistics, true-enable statistics");

    option.ParseArgs(argc, argv);
    cmdParams.aclConfig = option.GetStringOption("-acl_setup");
    cmdParams.regConfig = option.GetStringOption("-setup_reg");
    cmdParams.searchConfig = option.GetStringOption("-setup_search");
    cmdParams.runMode = option.GetIntOption("-run_mode");
    if (cmdParams.runMode < 0 || cmdParams.runMode > 1) {
        LogError << "runMode is incorrect, runMode=" << cmdParams.runMode;
        return APP_ERR_COMM_FAILURE;
    }
    cmdParams.debugLevel = option.GetIntOption("-debug_level");
    if (cmdParams.debugLevel < AtlasAscendLog::LOG_LEVEL_DEBUG
        || cmdParams.debugLevel > AtlasAscendLog::LOG_LEVEL_NONE) {
        LogError << "debugLevel is incorrect, debugLevel=" << cmdParams.debugLevel;
        return APP_ERR_COMM_FAILURE;
    }

    Statistic::SetStatisticEnable(option.GetBoolOption("-stats"));
    return ret;
}

void SetLogLevel(int debugLevel)
{
    switch (debugLevel) {
        case AtlasAscendLog::LOG_LEVEL_DEBUG:
            AtlasAscendLog::Log::LogDebugOn();
            break;
        case AtlasAscendLog::LOG_LEVEL_INFO:
            AtlasAscendLog::Log::LogInfoOn();
            break;
        case AtlasAscendLog::LOG_LEVEL_WARN:
            AtlasAscendLog::Log::LogWarnOn();
            break;
        case AtlasAscendLog::LOG_LEVEL_ERROR:
            AtlasAscendLog::Log::LogErrorOn();
            break;
        case AtlasAscendLog::LOG_LEVEL_FATAL:
            AtlasAscendLog::Log::LogFatalOn();
            break;
        case AtlasAscendLog::LOG_LEVEL_NONE:
            AtlasAscendLog::Log::LogAllOff();
            break;
        default:
            break;
    }
}

#endif // __INC_COMMAND_LINE_H__
