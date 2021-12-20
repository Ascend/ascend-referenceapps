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

#ifndef INC_TEST_CV_H
#define INC_TEST_CV_H

#include <map>
#include <memory>
#include <vector>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "DataType/DataType.h"
#include "Log/Log.h"
#include "DataTrans/DataTrans.pb.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

namespace ascendFaceRecognition {
class TestCV {
public:
    TestCV();
    ~TestCV();
    static TestCV &GetInstance();
    void AddRectIntoImage(ImageInfo &image, std::vector<DetectInfo> &detectResult);
    void AddTrackIntoImage(FrameAiInfo &aiInfo);
    void AddQEIntoImage(FrameAiInfo &aiInfo);
    void AddKPIntoImage(void *buf, std::vector<Coordinate2D> &keyPoints);
    void WrapAffineImage(void *srcBuf, void *waBuf, const std::vector<cv::Point2f> &landmarks);
    void WrapAffineImage(void *srcBuf, void *waBuf, const float *landmarks = nullptr);
    void WrapAffineImage(const cv::Mat &warpImage);
    void WAAddKPIntoImage(void *buf, std::vector<Coordinate2D> &keyPoints);
    void SaveDetectResult(std::shared_ptr<FrameAiInfo> frameAiInfo);
    void SaveTrackImage(std::shared_ptr<FrameAiInfo> frameAiInfo);
    void SaveQEImage(std::shared_ptr<FaceObject> faceObject, int batchIdx = 0);
    void SaveFaceAttributeImage(std::shared_ptr<FaceObject> faceObject, int batchIdx = 0);
    void SaveDecodeImage(std::shared_ptr<FrameAiInfo> frameAiInfo) const;
    void SaveFaceSearchImage(std::shared_ptr<FaceObject> faceObject);
    void SaveFaceFeatureLibImage(std::shared_ptr<FaceObject> faceObject);
    void SaveLandmarkImage(std::shared_ptr<FaceObject> faceObject);
    void SaveFaceImage(std::shared_ptr<FaceAttributeDataTrans> faceInfo);
    void SaveCropImage(const ImageInfo &imageInfo, const uint32_t &frameId, const uint32_t &idx);
    void SaveEmbeddingImage(const ImageInfo &imageInfo, const uint32_t &frameId, const uint32_t &idx);
    void SaveQualityImage(const ImageInfo &imageInfo, const uint32_t &frameId, const uint32_t &idx);
#ifndef ASCEND_ACL_OPEN_VESION
    std::shared_ptr<void> DeviceCopyToHost(const uint32_t size, std::shared_ptr<uint8_t> data) const;
#endif
private:
    const int grayEmptyValue = 0;
    const int grayDeepValue = 255;
    const int grayLightValue = 23;
    const cv::Scalar green = cv::Scalar(grayEmptyValue, grayDeepValue, grayEmptyValue);
    const cv::Scalar blue = cv::Scalar(grayDeepValue, grayLightValue, grayEmptyValue);
    const cv::Scalar red = cv::Scalar(grayEmptyValue, grayEmptyValue, grayDeepValue);
    const cv::Scalar lightGreen = cv::Scalar(grayLightValue, grayDeepValue, grayEmptyValue);
    const float fontScale = 1.0;
    const int thickness = 4;
    const int lineType = 8;
    const int xOffset = 10;
    const int yOffset = 10;
    const int radius = 3;
    const int rectThickness = 2;
    const int numLandmark = 5;
};
}
#endif
