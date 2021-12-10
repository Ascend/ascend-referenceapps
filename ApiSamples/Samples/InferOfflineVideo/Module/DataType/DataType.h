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

#ifndef INC_DATA_TYPE_H
#define INC_DATA_TYPE_H

#include "CommonDataType/CommonDataType.h"
#include "DvppCommon/DvppCommon.h"

struct CommonData {
    bool eof = {};
    uint32_t channelId = {};
    uint64_t frameId = {};
    uint32_t srcWidth = {};
    uint32_t srcHeight = {};
    acldvppStreamFormat videoFormat = {};
    StreamData streamData = {};
    uint32_t resizeWidth = {};
    uint32_t resizeHeight = {};
    uint32_t modelWidth = {};
    uint32_t modelHeight = {};
    uint32_t modelType = {};
    std::shared_ptr<DvppDataInfo> dvppData = {};
    std::vector<RawData> inferOutput = {};
};

#endif