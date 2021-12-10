/*
 * @Author: your name
 * @Date: 2020-06-28 12:20:17
 * @LastEditTime: 2020-06-28 12:47:27
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /facerecognition/src/CtrlCPU/WarpAffine/WarpAffine.h
 */
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

#ifndef INC_WARP_AFFINE_H
#define INC_WARP_AFFINE_H

#include "ModuleBase/ModuleBase.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


namespace ascendFaceRecognition {
class WarpAffine : public ModuleBase {
public:
    WarpAffine();
    ~WarpAffine();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR InitDvpp(void);
    APP_ERROR ParseConfig(ConfigParser &configParser);
    void GetSrcLandmark(std::vector<cv::Point2f> &points);
    void GetDstLandmark(std::vector<cv::Point2f> &points, std::shared_ptr<FaceObject> faceObject);
    APP_ERROR GetCropImage(std::shared_ptr<FaceObject> faceObject, cv::Mat &image);
    APP_ERROR SetWarpImage(std::shared_ptr<FaceObject> faceObject, const cv::Mat &imageWarp);
    APP_ERROR ApplyWarpAffine(std::shared_ptr<FaceObject> faceObject);

    void CalWarpImage(const std::vector<cv::Point2f> &srcPoints, const std::vector<cv::Point2f> &dstPoints,
        cv::Mat &warpImage, const cv::Mat &srcImageNv21);

private:
    int width_ = 0;
    int height_ = 0;
    int maxFaceNumPerFrame_ = 20;
    const int batchSize_ = 1;

    int streamWidthMax_ = 1920;
    int streamHeightMax_ = 1080;
    std::vector<std::shared_ptr<FaceObject>> faceObjectVec_ = {};
};
} // namespace ascendFaceRecognition

#endif
