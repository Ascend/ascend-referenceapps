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

#ifndef COMMAND_LINE_H
#define COMMAND_LINE_H

#include <fstream>
#include <iostream>
#include <string>
#include "CommonDataType/CommonDataType.h"
#include "Log/Log.h"
#include "ErrorCode/ErrorCode.h"
#include "FileManager/FileManager.h"

#define CHECK_HEIGHT_WIDTH_SIZE(height, width)         \
                                ((height) < MIN_SIZE   \
                                || (width) < MIN_SIZE  \
                                || (height) > MAX_SIZE \
                                || (width) > MAX_SIZE)

#define CHECK_FORMAT(format)                                                                                 \
                    (((format) > PIXEL_FORMAT_YVU_SEMIPLANAR_420 && (format) < PIXEL_FORMAT_YUYV_PACKED_422) \
                    || (format) > PIXEL_FORMAT_VYUY_PACKED_422                                               \
                    || (format) < PIXEL_FORMAT_YUV_SEMIPLANAR_420)

#define IS_YUV420_HEIGHT_WIDTH_ODD(height, width, format)                                \
                                    ((!((width) % MOD2 == 0) || !((height) % MOD2 == 0)) \
                                    && (format) <= PIXEL_FORMAT_YVU_SEMIPLANAR_420       \
                                    && (format) >= PIXEL_FORMAT_YUV_SEMIPLANAR_420)

#define IS_YUV422_WIDTH_ODD(width, format)                              \
                            ((format) >= PIXEL_FORMAT_YUYV_PACKED_422   \
                            && (format) <= PIXEL_FORMAT_VYUY_PACKED_422 \
                            && !((width) % MOD2 == 0))

static const uint32_t MIN_SIZE = 32;
static const uint32_t MAX_SIZE = 8192;
const int MOD2 = 2;

/*
 * @description: check the validity of input argument
 * @param fileName, name of input file
 * @param height   height of input image
 * @param width    width of input image
 * @param format   format of input image
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
static APP_ERROR CheckArgs(const std::string &fileName, const unsigned int height, const unsigned int width,
                           const unsigned int format)
{
    int folderExist = access(fileName.c_str(), R_OK);
    if (folderExist == -1) {
        LogError << "Input file " << fileName << " doesn't exist or read failed.";
        return APP_ERR_COMM_NO_EXIST;
    }
    if (CHECK_HEIGHT_WIDTH_SIZE(height, width)) {
        LogError << "The height or width is not valid or given, please check the help message.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (CHECK_FORMAT(format)) {
        LogError << "The format is not valid or given, please check the help message.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (IS_YUV420_HEIGHT_WIDTH_ODD(height, width, format)) {
        LogError << "The height or width cannot be an odd number if the format is YUV420sp.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (IS_YUV422_WIDTH_ODD(width, format)) {
        LogError << "The width cannot be an odd number if the format is YUV422.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    return APP_ERR_OK;
}

#endif
