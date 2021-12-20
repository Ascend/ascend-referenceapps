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

#ifndef INC_FACEDEMO_MOTCONNECTIONMODULE_BASE_H
#define INC_FACEDEMO_MOTCONNECTIONMODULE_BASE_H

#include <thread>
#include <list>
#include <utility>
#include "ModuleBase/ModuleBase.h"
#include "acl/acl.h"
#include "KalmanTracker.h"
#include "Hungarian.h"
#include "DataType/DataType.h"

namespace ascendFaceRecognition {
struct TraceLet {
    FaceTraceInfo info = {};
    int32_t lostAge = 0;
    KalmanTracker kalman;
    std::list<std::pair<DataBuffer, float>> shortFeatureQueue;
    DetectInfo detectInfo = {};
};

enum MethodFlag {
    EMBEDDING = 0,
    IOU,
    MIXED,
    NEW_MIXED
};

class MOTConnection : public ModuleBase {
public:
    MOTConnection();
    ~MOTConnection();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    double trackThreshold_ = 0;
    double kIOU_ = 1.0;
    int32_t method_ = 1;
    int32_t lostThreshold_ = 0;
    uint32_t maxNumberFeature_ = 0;
    int32_t generatedId_ = 0;
    std::vector<TraceLet> traceList_ = {};

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);

    void FilterLowThreshold(const HungarianHandle &hungarianHandleObj, const std::vector<std::vector<int>> &disMatrix,
        std::vector<cv::Point> &matchedTracedDetected, std::vector<bool> &detectFaceFlagVec);

    void UpdateUnmatchedTraceLet(const std::shared_ptr<FrameAiInfo> &frameAiInfo);

    void UpdateMatchedTraceLet(const std::vector<cv::Point> &matchedTracedDetected,
        std::shared_ptr<FrameAiInfo> &frameAiInfo);

    void AddNewDetectedFace(std::vector<FaceObject *> &unmatchedFaceObjectQueue);

    void UpdateTraceLetAndFrame(const std::vector<cv::Point> &matchedTracedDetected,
        std::shared_ptr<FrameAiInfo> &frameAiInfo, std::vector<FaceObject *> &unmatchedFaceObjectQueue);

    void TrackObjectPredict();
    void TrackObjectUpdate(const std::shared_ptr<FrameAiInfo> &frameAiInfo,
        std::vector<cv::Point> &matchedTracedDetected, std::vector<FaceObject *> &unmatchedFaceObjectQueue);
};
} // namespace ascendFaceRecognition
#endif
