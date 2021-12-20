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

#include "MOTConnection.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "Hungarian.h"

#include "Common.h"
#include "QualityEvaluation/FaceBlockingMap.h"
#include "DataTrans/DataTrans.pb.h"
#include "TestCV/TestCV.h"

#ifdef ASCEND_ACL_OPEN_VESION
#include "HdcChannel/HdcChannel.h"
#endif

namespace ascendFaceRecognition {
namespace {
// convert double to int
const int FLOAT_TO_INT = 1000;
const int MULTIPLE = 2;
const double SIMILARITY_THRESHOLD = 0.66;
const int MULTIPLE_IOU = 6;
const float NORM_EPS = 1e-10;
const double TIME_COUNTS = 1000.0;
const double COST_TIME_MS_THRESHOLD = 10.;
const float WIDTH_RATE_THRESH = 1.f;
const float HEIGHT_RATE_THRESH = 1.f;
const float X_DIST_RATE_THRESH = 1.3f;
const float Y_DIST_RATE_THRESH = 1.f;
} // namespace

MOTConnection::MOTConnection()
{
    isStop_ = true;
    instanceId_ = -1;
}

MOTConnection::~MOTConnection() {}

APP_ERROR MOTConnection::ParseConfig(ConfigParser &configParser)
{
    LogDebug << "MOTConnection[" << instanceId_ << "]: begin to parse config values.";
    std::string itemCfgStr = moduleName_ + std::string(".trackThreshold");
    APP_ERROR ret = configParser.GetDoubleValue(itemCfgStr, trackThreshold_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".lostThreshold");
    ret = configParser.GetIntValue(itemCfgStr, lostThreshold_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".method");
    ret = configParser.GetIntValue(itemCfgStr, method_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".kIOU");
    ret = configParser.GetDoubleValue(itemCfgStr, kIOU_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".maxNumberFeature");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, maxNumberFeature_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    LogDebug << "MOTConnection[" << instanceId_ << "]" << " trackThreshold_:" <<
        trackThreshold_;

    return ret;
}

APP_ERROR MOTConnection::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "MOTConnection[" << instanceId_ << "]: Begin to init MOT Connection instance" << initArgs.instanceId;
    AssignInitArgs(initArgs);
    isStop_ = false;
    // initialize config params
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "MOTConnection[" << instanceId_ << "]: Fail to parse config params, ret=" << ret << "(" <<
            GetAppErrCodeInfo(ret) << ").";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR MOTConnection::DeInit(void)
{    
    traceList_.clear();
    return APP_ERR_OK;
}

float CalDot(const DataBuffer &histFeature1, const DataBuffer &histFeature2)
{
    if (histFeature1.dataSize != histFeature2.dataSize) {
        LogError << "MOTConnection featureLen is not equal";
        return APP_ERR_ACL_FAILURE;
    }
#ifdef ASCEND_ACL_OPEN_VESION
    const float *feature1 = (float *)histFeature1.deviceData.get();
#else
    const float *feature1 = (float *)histFeature1.hostData.get();
#endif
    if (feature1 == nullptr) {
        LogError << "MOTConnection histFeature1  is ilegal!!";
        return APP_ERR_ACL_FAILURE;
    }
#ifdef ASCEND_ACL_OPEN_VESION
    const float *feature2 = (float *)histFeature2.deviceData.get();
#else
    const float *feature2 = (float *)histFeature2.hostData.get();
#endif
    if (feature2 == nullptr) {
        LogError << "MOTConnection histFeature2  is ilegal!!";
        return APP_ERR_ACL_FAILURE;
    }
    float dotSum = 0.f;
    uint32_t featureLen = histFeature1.dataSize / sizeof(float);
    for (uint32_t i = 0; i < featureLen; i++) {
        dotSum += feature1[i] * feature2[i];
    }
    LogDebug << "MOTConnection dotSum: " << dotSum << "\n";
    return dotSum;
}

float CalshortFeatureCosine(const TraceLet &traceLet, const FaceObject &faceObject)
{
    float max = 0.f;
    uint32_t maxIdx = traceLet.shortFeatureQueue.size();
    uint32_t idx = 0;
#ifdef ASCEND_ACL_OPEN_VESION
    if (traceLet.shortFeatureQueue.size() == 0 || faceObject.embedding.deviceData.get() == nullptr) {
        return 0.f;
    }
#else
    if (traceLet.shortFeatureQueue.size() == 0 || faceObject.embedding.hostData.get() == nullptr) {
        return 0.f;
    }
#endif
    for (auto &shortFeature : traceLet.shortFeatureQueue) {
        float tmp = CalDot(shortFeature.first, faceObject.embedding);
        if (shortFeature.second > NORM_EPS) {
            tmp /= shortFeature.second;
        } else {
            tmp = 0.f;
        }
        if (tmp > max) {
            max = tmp;
            maxIdx = idx;
        }
        idx++;
    }
    // normalization
    if (maxIdx < traceLet.shortFeatureQueue.size()) {
        if (faceObject.embeddingNorm > NORM_EPS) {
            max /= faceObject.embeddingNorm;
        } else {
            max = 0.f;
        }
    }
    LogDebug << "MOTConnection CalCosine: " << max << "\n";
    return max;
}

float CalIOU(DetectInfo detect1, DetectInfo detect2)
{
    cv::Rect_<float> bbox1(detect1.minx, detect1.miny, detect1.height, detect1.width);
    cv::Rect_<float> bbox2(detect2.minx, detect2.miny, detect2.height, detect2.width);
    float intersectionArea = (bbox1 & bbox2).area();
    float unionArea = bbox1.area() + bbox2.area() - intersectionArea;
    if (unionArea < DBL_EPSILON) {
        return 0.f;
    }
    return (intersectionArea / unionArea);
}


float CalDistSimilarity(DetectInfo detect1, DetectInfo detect2)
{
    float cx1 = detect1.minx + detect1.width / 2.f;
    float cy1 = detect1.miny + detect1.height / 2.f;
    float cx2 = detect2.minx + detect2.width / 2.f;
    float cy2 = detect2.miny + detect2.height / 2.f;
    float xDistance = cx1 - cx2;
    float yDistance = cy1 - cy2;
    float minWidth = std::min(detect1.width, detect2.width);
    float minHeight = std::min(detect1.height, detect2.height);
    float value = (1.f - xDistance / minWidth) * (1.f - yDistance / minHeight);
    return value;
}

float CalSimilarity(const TraceLet &traceLet, const FaceObject &faceObject, const int &method, const double &kIOU)
{
    switch (method) {
        case EMBEDDING:
            return CalshortFeatureCosine(traceLet, faceObject);
        case IOU:
            return CalIOU(traceLet.detectInfo, faceObject.info);
        case MIXED: {
            float iou = CalIOU(traceLet.detectInfo, faceObject.info);
            float cosine = CalshortFeatureCosine(traceLet, faceObject);
            return kIOU * iou + (1.0 - kIOU) * cosine;
        }
        case NEW_MIXED: {
            float widthRate = abs(faceObject.info.width - traceLet.detectInfo.width) / traceLet.detectInfo.width;
            float heightRate = abs(faceObject.info.height - traceLet.detectInfo.height) / traceLet.detectInfo.height;
            float xDistanceRate = abs(faceObject.info.minx - traceLet.detectInfo.minx +
                faceObject.info.width / MULTIPLE - traceLet.detectInfo.width / MULTIPLE) /
                traceLet.detectInfo.width;
            float yDistanceRate = abs(faceObject.info.miny - traceLet.detectInfo.miny +
                faceObject.info.height / MULTIPLE - traceLet.detectInfo.height / MULTIPLE) /
                traceLet.detectInfo.height;
            if (widthRate > WIDTH_RATE_THRESH || heightRate > HEIGHT_RATE_THRESH) {
                return 0.f;
            }
            if (xDistanceRate > X_DIST_RATE_THRESH || yDistanceRate > Y_DIST_RATE_THRESH) {
                return 0.f;
            }

            float distSim = CalDistSimilarity(traceLet.detectInfo, faceObject.info);
            float cosine = CalshortFeatureCosine(traceLet, faceObject);
            float value = (cosine > SIMILARITY_THRESHOLD) ? std::max(distSim * cosine * MULTIPLE_IOU, 1.f) : distSim;
            return value;
        }
        default:
            LogError << "pleease input suitable method to calculate the Similarity Matrix!!!";
            return 0.0;
    }
}

void MOTConnection::FilterLowThreshold(const HungarianHandle &hungarianHandleObj,
    const std::vector<std::vector<int>> &disMatrix, std::vector<cv::Point> &matchedTracedDetected,
    std::vector<bool> &detectFaceFlagVec)
{
    for (unsigned int i = 0; i < traceList_.size(); ++i) {
        if ((hungarianHandleObj.resX[i] != -1) &&
            (disMatrix[i][hungarianHandleObj.resX[i]] >= (trackThreshold_ * FLOAT_TO_INT))) {
            matchedTracedDetected.push_back(cv::Point(i, hungarianHandleObj.resX[i]));
            detectFaceFlagVec[hungarianHandleObj.resX[i]] = true;
        } else {
            traceList_[i].info.flag = LOST_FACE;
        }
    }
}

void MOTConnection::UpdateUnmatchedTraceLet(const std::shared_ptr<FrameAiInfo> &frameAiInfo)
{
    for (auto itr = traceList_.begin(); itr != traceList_.end();) {
        if ((*itr).info.flag != LOST_FACE) {
            ++itr;
            continue;
        }

        (*itr).lostAge++;
        (*itr).kalman.Update((*itr).detectInfo);

        if ((*itr).lostAge < lostThreshold_) {
            continue;
        }

        int32_t id = itr->info.id;
        auto faceObject = FaceBlockingMap::GetInstance(frameAiInfo->info.channelId)->Get(id);
        if (faceObject.get() != nullptr) {
            faceObject->trackInfo.flag = LOST_FACE;
        }
        itr = traceList_.erase(itr);
    }
}

void MOTConnection::UpdateMatchedTraceLet(const std::vector<cv::Point> &matchedTracedDetected,
    std::shared_ptr<FrameAiInfo> &frameAiInfo)
{
    for (unsigned int i = 0; i < matchedTracedDetected.size(); ++i) {
        int traceIndex = matchedTracedDetected[i].x;
        int detectIndex = matchedTracedDetected[i].y;
        LogDebug << "[frame id = " << frameAiInfo->info.frameId << "] matched traceindex =" << traceIndex <<
            ",matched detectindex = " << detectIndex << "";
        if (traceList_[traceIndex].info.survivalTime > MULTIPLE) {
            traceList_[traceIndex].info.flag = TRACkED_FACE;
        }
        traceList_[traceIndex].info.survivalTime++;
        traceList_[traceIndex].info.detectedTime++;
        traceList_[traceIndex].lostAge = 0;
        traceList_[traceIndex].detectInfo = frameAiInfo->face[detectIndex].info;
        traceList_[traceIndex].kalman.Update(frameAiInfo->face[detectIndex].info);
        if (traceList_[traceIndex].shortFeatureQueue.size() >= maxNumberFeature_) {
            traceList_[traceIndex].shortFeatureQueue.pop_front();
        }

        // MOT embedding module may be disable
#ifdef ASCEND_ACL_OPEN_VESION
        if (frameAiInfo->face[detectIndex].embedding.deviceData.get() != nullptr &&
            frameAiInfo->face[detectIndex].embedding.dataSize > 0) {
            DataBuffer dataBuffer;
            dataBuffer.dataSize = frameAiInfo->face[detectIndex].embedding.dataSize;
            dataBuffer.deviceData = frameAiInfo->face[detectIndex].embedding.deviceData;
            auto shortFeature = std::make_pair(dataBuffer, frameAiInfo->face[detectIndex].embeddingNorm);
            traceList_[traceIndex].shortFeatureQueue.push_back(shortFeature);
        }
#else
        if (frameAiInfo->face[detectIndex].embedding.hostData.get() != nullptr &&
            frameAiInfo->face[detectIndex].embedding.dataSize > 0) {
            DataBuffer dataBuffer;
            dataBuffer.dataSize = frameAiInfo->face[detectIndex].embedding.dataSize;
            dataBuffer.hostData = frameAiInfo->face[detectIndex].embedding.hostData;
            auto shortFeature = std::make_pair(dataBuffer, frameAiInfo->face[detectIndex].embeddingNorm);
            traceList_[traceIndex].shortFeatureQueue.push_back(shortFeature);
        }
#endif
        // update frame
        frameAiInfo->face[detectIndex].trackInfo = traceList_[traceIndex].info;
    }
}
void MOTConnection::AddNewDetectedFace(std::vector<FaceObject *> &unmatchedFaceObjectQueue)
{
    using Time = std::chrono::high_resolution_clock;
    for (auto &faceObject : unmatchedFaceObjectQueue) {
        // add new detected into traceList
        TraceLet traceLet;
        generatedId_++;
        traceLet.info.id = generatedId_;
        traceLet.info.survivalTime = 1;
        traceLet.info.detectedTime = 1;
        traceLet.lostAge = 0;
        traceLet.info.flag = NEW_FACE;
        traceLet.detectInfo = faceObject->info;
        traceLet.info.createTime = Time::now();
        traceLet.kalman.CvKalmanInit(faceObject->info);
        // MOT embedding module may be disable
#ifdef ASCEND_ACL_OPEN_VESION
        if (faceObject->embedding.deviceData.get() != nullptr && faceObject->embedding.dataSize > 0) {
            LogDebug << "Face Embedding Size " << faceObject->embedding.dataSize;
            DataBuffer dataBuffer;
            dataBuffer.dataSize = faceObject->embedding.dataSize;
            dataBuffer.deviceData = faceObject->embedding.deviceData;
            auto shortFeature = std::make_pair(dataBuffer, faceObject->embeddingNorm);
            traceLet.shortFeatureQueue.push_back(shortFeature);
        }
#else
        if (faceObject->embedding.hostData.get() != nullptr && faceObject->embedding.dataSize > 0) {
            LogDebug << "Face Embedding Size " << faceObject->embedding.dataSize;
            DataBuffer dataBuffer;
            dataBuffer.dataSize = faceObject->embedding.dataSize;
            dataBuffer.hostData = faceObject->embedding.hostData;
            auto shortFeature = std::make_pair(dataBuffer, faceObject->embeddingNorm);
            traceLet.shortFeatureQueue.push_back(shortFeature);
        }
#endif
        traceList_.push_back(traceLet);

        // update frame
        faceObject->trackInfo = traceLet.info;
    }
}

void MOTConnection::UpdateTraceLetAndFrame(const std::vector<cv::Point> &matchedTracedDetected,
    std::shared_ptr<FrameAiInfo> &frameAiInfo, std::vector<FaceObject *> &unmatchedFaceObjectQueue)
{
    UpdateMatchedTraceLet(matchedTracedDetected, frameAiInfo);
    AddNewDetectedFace(unmatchedFaceObjectQueue);
    UpdateUnmatchedTraceLet(frameAiInfo);
}


void MOTConnection::TrackObjectPredict()
{
    // every traceLet should do kalman predict
    for (auto &traceLet : traceList_) {
        traceLet.detectInfo = traceLet.kalman.Predict();
    }
}
void MOTConnection::TrackObjectUpdate(const std::shared_ptr<FrameAiInfo> &frameAiInfo,
    std::vector<cv::Point> &matchedTracedDetected, std::vector<FaceObject *> &unmatchedFaceObjectQueue)
{
    if (frameAiInfo->face.size() > 0) {
        LogDebug << "[frame id = " << frameAiInfo->info.frameId << "], trace size =" << traceList_.size() <<
            "detect size = " << frameAiInfo->face.size() << "";
        // init face matched flag
        std::vector<bool> detectFaceFlagVec;
        for (unsigned int i = 0; i < frameAiInfo->face.size(); ++i) {
            detectFaceFlagVec.push_back(false);
        }
        // calculate the associated matrix
        std::vector<std::vector<int>> disMatrix;
        disMatrix.clear();
        disMatrix.resize(traceList_.size(), std::vector<int>(frameAiInfo->face.size(), 0));
        for (unsigned int j = 0; j < frameAiInfo->face.size(); ++j) {
            for (unsigned int i = 0; i < traceList_.size(); ++i) {
                float sim = CalSimilarity(traceList_[i], frameAiInfo->face[j], method_, kIOU_);
                disMatrix[i][j] = (int)(sim * FLOAT_TO_INT);
                LogDebug << "frame id = " << frameAiInfo->info.frameId << ", disMatrix[" << i << "][" << j << "] = " <<
                    disMatrix[i][j] << "";
            }
        }

        // solve the assignment problem using hungarian
        HungarianHandle hungarianHandleObj = {};
        HungarianHandleInit(hungarianHandleObj, traceList_.size(), frameAiInfo->face.size());
        HungarianSolve(hungarianHandleObj, disMatrix, traceList_.size(), frameAiInfo->face.size());
        // filter out matched but with low distance
        FilterLowThreshold(hungarianHandleObj, disMatrix, matchedTracedDetected, detectFaceFlagVec);
        LogDebug << "matchedTracedDetected = " << matchedTracedDetected.size() << "";
        // fill unmatchedFaceObjectQueue
        for (unsigned int i = 0; i < detectFaceFlagVec.size(); ++i) {
            if (detectFaceFlagVec[i] == false) {
                unmatchedFaceObjectQueue.push_back(&frameAiInfo->face[i]);
            }
        }
    }
}


static void SetDataTrans(std::shared_ptr<DataTrans> dataTrans, const FaceObject &faceObject)
{
    FrameDetectDataTrans *dectectInfo = dataTrans->mutable_framedetectinfo();
    FaceInfoDataTrans *faceInfo = dectectInfo->add_faces();
    faceInfo->set_trackid(faceObject.trackInfo.id);
    RectDataTrans *rect = faceInfo->mutable_rect();
    rect->set_left(faceObject.info.minx);
    rect->set_top(faceObject.info.miny);
    rect->set_width(faceObject.info.width);
    rect->set_height(faceObject.info.height);
}

APP_ERROR MOTConnection::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "MOTConnection[" << instanceId_ << "]: Begin to process data.";
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::static_pointer_cast<FrameAiInfo>(inputData);
    LogDebug << "FrameCache MOTConnection Recv " << frameAiInfo->info.channelId << "_" << frameAiInfo->info.frameId;
    std::vector<FaceObject *> unmatchedFaceObjectQueue;
    std::vector<cv::Point> matchedTracedDetected;
    if (frameAiInfo.get() == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }

    if (traceList_.size() > 0) {
        // every traceLet should do kalman predict
        TrackObjectPredict();
        TrackObjectUpdate(frameAiInfo, matchedTracedDetected, unmatchedFaceObjectQueue);
    } else {
        // traceList is empty, all the face detected in the new frame are unmatched.
        if (frameAiInfo->face.size() > 0) {
            for (unsigned int i = 0; i < frameAiInfo->face.size(); ++i) {
                unmatchedFaceObjectQueue.push_back(&frameAiInfo->face[i]);
            }
        }
    }
    LogDebug << "[frame id = " << frameAiInfo->info.frameId << "], matched detect size = " <<
        matchedTracedDetected.size() << ", unmatched detect size = " << unmatchedFaceObjectQueue.size() << "";

    // update all the tracelet in the tracelist per frame
    UpdateTraceLetAndFrame(matchedTracedDetected, frameAiInfo, unmatchedFaceObjectQueue);
    std::shared_ptr<DataTrans> dataTrans = std::make_shared<DataTrans>();
    FrameDetectDataTrans *dectectInfo = dataTrans->mutable_framedetectinfo();
    dectectInfo->set_channelid(frameAiInfo->info.channelId);
    dectectInfo->set_frameid(frameAiInfo->info.frameId);
    for (uint32_t i = 0; i < frameAiInfo->face.size(); i++) {
        auto flag = frameAiInfo->face[i].trackInfo.flag;
        if (flag == LOST_FACE || flag == NEW_FACE) {
            continue;
        }
        SetDataTrans(dataTrans, frameAiInfo->face[i]);
        if (frameAiInfo->face[i].imgQuality.buf.deviceData.get() != nullptr) {
            std::shared_ptr<FaceObject> face = std::make_shared<FaceObject>();
            *face = frameAiInfo->face[i];
            SendToNextModule(MT_QUALITY_EVALUATION, face, frameAiInfo->info.channelId);
        }
    }
#ifdef ASCEND_ACL_OPEN_VESION
    HdcChannel::GetInstance()->SendData(HDC_FRAME_ALIGN_CH_START_INDEX + frameAiInfo->info.channelId, dataTrans);
#else
    SendToNextModule(MT_FRAME_ALIGN, dataTrans, instanceId_);
#endif
    return APP_ERR_OK;
}
} // namespace ascendFaceRecognition
