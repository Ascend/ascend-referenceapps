/*
 * @Author: your name
 * @Date: 2020-06-28 12:36:17
 * @LastEditTime: 2020-06-28 12:38:21
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /facerecognition/src/CtrlCPU/WarpAffine/SimilarityTransform.h
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

#ifndef INC_SIMILARITY_TRANSFORM_H
#define INC_SIMILARITY_TRANSFORM_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace ascendFaceRecognition {
class SimilarityTransform {
public:
    SimilarityTransform();
    ~SimilarityTransform();
    cv::Mat Transform(const std::vector<cv::Point2f> &srcPoint, const std::vector<cv::Point2f> &dstPoint) const;

private:
    cv::Point2f GetMean(const std::vector<cv::Point2f> &srcPoint) const;
    double GetSumVars(const cv::Mat &array) const;
};
}

#endif
