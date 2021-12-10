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
#include "SimilarityTransform.h"

#include <cmath>

#include "Log/Log.h"

namespace ascendFaceRecognition {
namespace {
const int CONST_INT_2 = 2;
const int CONST_INT_3 = 3;
const double SMALL_EPSINON = 1e-6;
}

SimilarityTransform::SimilarityTransform() {}
SimilarityTransform::~SimilarityTransform() {}

cv::Point2f SimilarityTransform::GetMean(const std::vector<cv::Point2f> &srcPoint) const
{
    float sumX = 0;
    float sumY = 0;
    for (auto point : srcPoint) {
        sumX += point.x;
        sumY += point.y;
    }

    return cv::Point2f(sumX / srcPoint.size(), sumY / srcPoint.size());
}

double SimilarityTransform::GetSumVars(const cv::Mat &array) const
{
    uint32_t rows = array.rows;
    double meanX = 0;
    double meanY = 0;
    for (uint32_t i = 0; i < rows; i++) {
        meanX += array.at<double>(i, 0);
        meanY += array.at<double>(i, 1);
    }
    if (rows > 0) {
        meanX /= double(rows);
        meanY /= double(rows);
    }
    double sumXX = 0;
    double sumYY = 0;
    for (uint32_t i = 0; i < rows; i++) {
        sumXX += (array.at<double>(i, 0) - meanX) * (array.at<double>(i, 0) - meanX);
        sumYY += (array.at<double>(i, 1) - meanY) * (array.at<double>(i, 1) - meanY);
    }
    return (rows != 0) ? (sumXX / rows + sumYY / rows) : 0;
}

cv::Mat SimilarityTransform::Transform(const std::vector<cv::Point2f> &src, const std::vector<cv::Point2f> &dst) const
{
    int dim = 2;
    int nPoint = src.size();

    cv::Point2f srcMean = GetMean(src);
    cv::Point2f dstMean = GetMean(dst);

    cv::Mat srcDemean(nPoint, dim, CV_64F);
    cv::Mat dstDemean(nPoint, dim, CV_64F);
    for (int i = 0; i < nPoint; i++) {
        srcDemean.at<double>(i, 0) = src[i].x - srcMean.x;
        srcDemean.at<double>(i, 1) = src[i].y - srcMean.y;
        dstDemean.at<double>(i, 0) = dst[i].x - dstMean.x;
        dstDemean.at<double>(i, 1) = dst[i].y - dstMean.y;
    }

    cv::Mat a = (dstDemean.t() * srcDemean) / nPoint;
    cv::Mat d = cv::Mat::ones(dim, 1, CV_64F);
    if (cv::determinant(a) < 0) {
        d.at<double>(1, 0) = -1;
    }

    cv::Mat s;
    cv::Mat u;
    cv::Mat v;
    cv::SVDecomp(a, s, u, v);

    LogDebug << "S:" << s << "";
    LogDebug << "U:" << u << "";
    LogDebug << "V:" << v << "";

    cv::Mat t = cv::Mat::eye(dim + 1, dim + 1, CV_64F);

    cv::Mat tmpT = u * (cv::Mat::diag(d) * v);

    cv::Mat subT = t(cv::Rect(0, 0, dim, dim));
    tmpT.copyTo(subT);

    double var = GetSumVars(srcDemean);
    double varInv = 0.;
    if (var > SMALL_EPSINON || var < -SMALL_EPSINON) {
        varInv = 1.0 / var;
    }

    cv::Mat tmptd = s.t() * d;
    cv::Mat scale = varInv * tmptd;

    cv::Mat srcMeanMat(dim, 1, CV_64F);
    srcMeanMat.at<double>(0, 0) = srcMean.x;
    srcMeanMat.at<double>(1, 0) = srcMean.y;

    cv::Mat dstMeanMat(dim, 1, CV_64F);
    dstMeanMat.at<double>(0, 0) = dstMean.x;
    dstMeanMat.at<double>(1, 0) = dstMean.y;

    cv::Mat offset = dstMeanMat - scale.at<double>(0, 0) * (subT * srcMeanMat);
    cv::Mat subOffset = t(cv::Rect(dim, 0, 1, dim));
    offset.copyTo(subOffset);

    cv::Mat subOffsetT = t(cv::Rect(0, 0, dim, dim));
    subOffsetT = subOffsetT * scale.at<double>(0, 0);

    return t(cv::Rect(0, 0, CONST_INT_3, CONST_INT_2));
}
}
