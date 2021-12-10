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

#include "CenterFace.h"
#include <algorithm>
#include <cmath>
#include <string>

#include <opencv2/opencv.hpp>

using namespace Centerface;

namespace {
float InterRectArea(const cv::Rect &a, const cv::Rect &b)
{
    cv::Point leftTop = cv::Point(MAX(a.x, b.x), MAX(a.y, b.y));
    cv::Point rightBottom = cv::Point(MIN(a.br().x, b.br().x), MIN(a.br().y, b.br().y));
    cv::Point diff = rightBottom - leftTop;
    return (MAX(diff.x + 1, 0) * MAX(diff.y + 1, 0));
}

int ComputeIOU(const cv::Rect &rect1, const cv::Rect &rect2, float &iou, const std::string &type = "UNION")
{
    float interArea = InterRectArea(rect1, rect2);
    if (type == "UNION") {
        iou = interArea / (rect1.area() + rect2.area() - interArea);
    } else {
        iou = interArea / MIN(rect1.area(), rect2.area());
    }

    return 0;
}

int NMS(const std::vector<DetectResult> &faces, std::vector<DetectResult> &result, const float &thresHold,
    const std::string &type = "UNION")
{
    int ret = 0;
    result.clear();
    if (faces.size() == 0) {
        ret = -1;
        return ret;
    }

    std::vector<size_t> idx(faces.size());
    for (unsigned i = 0; i < idx.size(); i++) {
        idx[i] = i;
    }

    while (idx.size() > 0) {
        int goodIdx = idx[0];
        result.push_back(faces[goodIdx]);
        std::vector<size_t> tmp = idx;
        idx.clear();
        for (unsigned i = 1; i < tmp.size(); i++) {
            int tmpIdx = tmp[i];
            float iou = 0.0f;
            cv::Rect rectGood(faces[goodIdx].bbox.x, faces[goodIdx].bbox.y, faces[goodIdx].bbox.w,
                faces[goodIdx].bbox.h);
            cv::Rect rectTmp(faces[tmpIdx].bbox.x, faces[tmpIdx].bbox.y, faces[tmpIdx].bbox.w, faces[tmpIdx].bbox.h);
            ComputeIOU(rectGood, rectTmp, iou, type);
            if (iou <= thresHold)
                idx.push_back(tmpIdx);
        }
    }
    return ret;
}

int PostProcess(std::vector<DetectResult> &faceTmp, std::vector<ascendFaceRecognition::DetectInfo> &detectResult)
{
    std::sort(faceTmp.begin(), faceTmp.end(),
        [](const DetectResult &a, const DetectResult &b) { return a.score > b.score; });

    std::vector<DetectResult> faceNMS;
    int ret = NMS(faceTmp, faceNMS, POST_NMS_THRESHOLD);
    if (ret < 0) {
        return ret;
    }

    for (size_t j = 0; j < faceNMS.size(); ++j) {
        detectResult.push_back({});
        detectResult[j].classId = 1;
        detectResult[j].confidence = faceNMS[j].score;
        detectResult[j].minx = faceNMS[j].bbox.x;
        detectResult[j].miny = faceNMS[j].bbox.y;
        detectResult[j].width = faceNMS[j].bbox.w;
        detectResult[j].height = faceNMS[j].bbox.h;

        for (int k = 0; k < ascendFaceRecognition::LANDMARK_NUM; ++k) {
            detectResult[j].landmarks[k] = faceNMS[j].keyPoints[k];
        }
    }

    return ret;
}
}

int CenterFace::Process(const std::vector<void *> &outTensorAddrs, const ImageInfoPostProcess &imageInfo,
    std::vector<ascendFaceRecognition::DetectInfo> &detectResult)
{
    const int outputWidth = imageInfo.widthModelInput / RESIZE_FACTOR;
    const int outputHeight = imageInfo.heightModelInput / RESIZE_FACTOR;
    const float *heatmapHost = static_cast<float *>(outTensorAddrs[TENSOR_INDEX_0]);
    const float *scaleHost = static_cast<float *>(outTensorAddrs[TENSOR_INDEX_1]);
    const float *offsetHost = static_cast<float *>(outTensorAddrs[TENSOR_INDEX_2]);
    const float *landmarkHost = static_cast<float *>(outTensorAddrs[TENSOR_INDEX_3]);
    const float scaleX = imageInfo.widthOrigin / imageInfo.widthModelInput;
    const float scaleY = imageInfo.heightOrigin / imageInfo.heightModelInput;
    const int channelStep = outputWidth * outputHeight;

    std::vector<DetectResult> faceTmp;
    for (int h = 0; h < outputHeight; ++h) {
        for (int w = 0; w < outputWidth; ++w) {
            int index = h * outputWidth + w;
            float score = heatmapHost[index];
            if (score < SIMILARITY_THRESHOLD) {
                continue;
            }
            fastMath_.Init();
            float s0 = RESIZE_FACTOR * fastMath_.Fexp(scaleHost[index]);
            float s1 = RESIZE_FACTOR * fastMath_.Fexp(scaleHost[index + channelStep]);
            float o0 = offsetHost[index];
            float o1 = offsetHost[index + channelStep];
            float ymin = std::max(0.0, RESIZE_FACTOR * (h + o0 + ROUND_HALF) - ROUND_HALF * s0);
            float xmin = std::max(0.0, RESIZE_FACTOR * (w + o1 + ROUND_HALF) - ROUND_HALF * s1);
            float ymax = std::min(ymin + s0, imageInfo.heightModelInput);
            float xmax = std::min(xmin + s1, imageInfo.widthModelInput);

            DetectResult faceInfo;
            faceInfo.score = score;
            faceInfo.bbox.x = scaleX * xmin;
            faceInfo.bbox.y = scaleY * ymin;
            faceInfo.bbox.w = scaleX * (xmax - xmin);
            faceInfo.bbox.h = scaleY * (ymax - ymin);
            for (int num = 0; num < KEY_POINTER_IDX_RANGE; ++num) {
                faceInfo.keyPoints[KEY_POINTER_IDX_FACTOR * num] =
                    scaleX * (s1 * landmarkHost[(KEY_POINTER_IDX_FACTOR * num + 1) * channelStep + index] + xmin);
                faceInfo.keyPoints[KEY_POINTER_IDX_FACTOR * num + 1] =
                    scaleY * (s0 * landmarkHost[(KEY_POINTER_IDX_FACTOR * num + 0) * channelStep + index] + ymin);
            }
            faceTmp.push_back(faceInfo);
        }
    }

    if (PostProcess(faceTmp, detectResult) < 0) {
        return -1;
    }

    return detectResult.size();
}