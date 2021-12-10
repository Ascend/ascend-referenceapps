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
#ifndef COST_STATISTIC_H
#define COST_STATISTIC_H

#include <map>
#include <mutex>
#include <unistd.h>
#include <string>
#include "Log/Log.h"
#include <queue>
#include <stack>
#include <utility>
#include <stdio.h>
#include <sys/time.h>


namespace ascendFaceRecognition {
namespace {
const double ONE_SCECOND = 1000.;
}


struct ProcessTimeStamp {
    std::string processName = "";
    timeval startTime;
    timeval endTime;
};

class CostStatistic {
public:
    static timeval GetStart()
    {
        timeval startTime;
        gettimeofday(&startTime, nullptr);
        return startTime;
    }

    static timeval GetEnd()
    {
        timeval endTime;
        gettimeofday(&endTime, nullptr);
        return endTime;
    }

    static double GetCostTime(timeval startTime)
    {
        timeval endTime;
        gettimeofday(&endTime, nullptr);
        double costMs =
            (endTime.tv_sec - startTime.tv_sec) * ONE_SCECOND + (endTime.tv_usec - startTime.tv_usec) / ONE_SCECOND;
        return costMs;
    }

    CostStatistic() {}
    ~CostStatistic() {}
};
} // namespace ascendFaceRecognition
#endif