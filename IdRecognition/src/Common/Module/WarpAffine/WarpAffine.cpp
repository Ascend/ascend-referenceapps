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
#include <fstream>

#include "FileManager/FileManager.h"
#include "Log/Log.h"
#include "WarpAffine.h"

namespace ascendFaceRecognition {
const int KEY_POINT_OFFSET = 8;
const int STD_IMAGE_SIZE = 112;
const int KEY_POINT_COUNT = 5;

const int KEY_POINT0_DIM = 2;
const int KEY_POINT1_DIM = 3;
const int KEY_POINT2_DIM = 4;

const int KP_INDEX_0 = 0;
const int KP_INDEX_1 = 1;
const int KP_INDEX_2 = 2;
const int KP_INDEX_3 = 3;
const int KP_INDEX_4 = 4;
const int KP_INDEX_5 = 5;

WarpAffine::~WarpAffine()
{
    if (!isDeInited_) {
        DeInit();
    }
}

APP_ERROR WarpAffine::ParseConfig(ConfigParser &configParser)
{
    LogInfo << "WarpAffine[" << instanceId_ << "]: begin to parse config values.";
    std::string itemCfgStr;

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".width_output");
    APP_ERROR ret = configParser.GetIntValue(itemCfgStr, widthOutput_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".height_output");
    ret = configParser.GetIntValue(itemCfgStr, heightOutput_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    LogDebug << "WarpAffine[" << instanceId_ << "]: widthOutput=" << widthOutput_ << " heightOutput=" << heightOutput_
             << ".";

    return ret;
}

APP_ERROR WarpAffine::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "WarpAffine[" << instanceId_ << "]: begin to init warp affine instance.";

    AssignInitArgs(initArgs);

    // initialize config params
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "WarpAffine[" << instanceId_ << "]: Fail to parse config params, ret=" << ret << "("
                 << GetAppErrCodeInfo(ret) << ").";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR WarpAffine::DeInit(void)
{
    LogInfo << "WarpAffine[" << instanceId_ << "]: deinit start.";
    isDeInited_ = true;
    StopAndDestroyQueue();

    return APP_ERR_OK;
}

APP_ERROR WarpAffine::Process(std::shared_ptr<void> inputData)
{
    warpAffineStatic_.RunTimeStatisticStart("WarpAffine_Excute_Time", instanceId_);
    LogDebug << "WarpAffine[" << instanceId_ << "]: begin to process.";

    std::shared_ptr<FrameAiInfo> frameAiInfo = std::static_pointer_cast<FrameAiInfo>(inputData);
    if (frameAiInfo.get() == nullptr) {
        return APP_ERR_COMM_INVALID_POINTER;
    }

    /* do nv12 to bgr888 convert */
    cv::Mat srcNV12Mat(frameAiInfo->imgOrigin.height * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2,
        frameAiInfo->imgOrigin.width, CV_8UC1, frameAiInfo->imgOrigin.buf.hostData.get());
    cv::Mat dstBGR888(frameAiInfo->imgOrigin.height, frameAiInfo->imgOrigin.width, CV_8UC3);
    waColorCvtStatic_.RunTimeStatisticStart("WarAffineCvtColor_Excute_Time", instanceId_);
    cv::cvtColor(srcNV12Mat, dstBGR888, cv::COLOR_YUV2BGR_NV12); // COLOR_YUV2RGB_NV12
    waColorCvtStatic_.RunTimeStatisticStop();

    waPostStatic_.RunTimeStatisticStart("WarAffinePost_Excute_Time", instanceId_);
    for (int i = 0; i < (int)frameAiInfo->face.size(); i++) {
        APP_ERROR ret = ApplyWarpAffine((frameAiInfo->face[i]), dstBGR888);
        if (ret != APP_ERR_OK) {
            LogError << "WarpAffine[" << instanceId_ << "]: apply warpaffine error (i=" << i << "), skip warpaffine!";
            continue;
        }
    }

    outputQueVec_[instanceId_]->Push(frameAiInfo, true);

    waPostStatic_.RunTimeStatisticStop();
    warpAffineStatic_.RunTimeStatisticStop();

    return APP_ERR_OK;
}

APP_ERROR WarpAffine::KeyPointConversion(std::shared_ptr<KeyPointInfo> keyPointInfo, float &deno0)
{
    LogDebug << "WarpAffine[" << instanceId_ << "]: begin to convert key points.";
    int flag = 0;
    for (uint32_t idxKp0 = 0; idxKp0 <= KEY_POINT0_DIM; idxKp0++) {
        for (uint32_t idxKp1 = idxKp0 + 1; idxKp1 <= KEY_POINT1_DIM; idxKp1++) {
            for (uint32_t idxKp2 = idxKp1 + 1; idxKp2 <= KEY_POINT2_DIM; idxKp2++) {
                // chose 3 couple of points to cal
                keyPointInfo->kPBefore[KP_INDEX_0] = keyPointInfo->keyPointBefore[idxKp0 * KEY_POINT0_DIM];
                keyPointInfo->kPBefore[KP_INDEX_1] = keyPointInfo->keyPointBefore[idxKp0 * KEY_POINT0_DIM + 1];
                keyPointInfo->kPBefore[KP_INDEX_2] = keyPointInfo->keyPointBefore[idxKp1 * KEY_POINT0_DIM];
                keyPointInfo->kPBefore[KP_INDEX_3] = keyPointInfo->keyPointBefore[idxKp1 * KEY_POINT0_DIM + 1];
                keyPointInfo->kPBefore[KP_INDEX_4] = keyPointInfo->keyPointBefore[idxKp2 * KEY_POINT0_DIM];
                keyPointInfo->kPBefore[KP_INDEX_5] = keyPointInfo->keyPointBefore[idxKp2 * KEY_POINT0_DIM + 1];

                keyPointInfo->kPAfter[KP_INDEX_0] = keyPointInfo->keyPointAfter[idxKp0 * KEY_POINT0_DIM];
                keyPointInfo->kPAfter[KP_INDEX_1] = keyPointInfo->keyPointAfter[idxKp0 * KEY_POINT0_DIM + 1];
                keyPointInfo->kPAfter[KP_INDEX_2] = keyPointInfo->keyPointAfter[idxKp1 * KEY_POINT0_DIM];
                keyPointInfo->kPAfter[KP_INDEX_3] = keyPointInfo->keyPointAfter[idxKp1 * KEY_POINT0_DIM + 1];
                keyPointInfo->kPAfter[KP_INDEX_4] = keyPointInfo->keyPointAfter[idxKp2 * KEY_POINT0_DIM];
                keyPointInfo->kPAfter[KP_INDEX_5] = keyPointInfo->keyPointAfter[idxKp2 * KEY_POINT0_DIM + 1];
                // cal the denominator which shared by the first row of affineMatrix_
                deno0 = (keyPointInfo->kPBefore[KP_INDEX_4] - keyPointInfo->kPBefore[KP_INDEX_0]) *
                    (keyPointInfo->kPBefore[KP_INDEX_3] - keyPointInfo->kPBefore[KP_INDEX_1]) -
                    (keyPointInfo->kPBefore[KP_INDEX_2] - keyPointInfo->kPBefore[KP_INDEX_0]) *
                    (keyPointInfo->kPBefore[KP_INDEX_5] - keyPointInfo->kPBefore[KP_INDEX_1]);
                if (deno0 != 0) {
                    flag = 1;
                    break;
                }

                LogDebug << "WarpAffine[" << instanceId_ << "]: a0=" << keyPointInfo->kPBefore[KP_INDEX_0]
                         << ",a1=" << keyPointInfo->kPBefore[KP_INDEX_1] << ",a2=" << keyPointInfo->kPBefore[KP_INDEX_2]
                         << ",a3=" << keyPointInfo->kPBefore[KP_INDEX_3] << ",a4=" << keyPointInfo->kPBefore[KP_INDEX_4]
                         << ",a5=" << keyPointInfo->kPBefore[KP_INDEX_5];
            }
        }
    }
    if (flag == 0) {
        LogError << "WarpAffine[" << instanceId_ << "]: can not find 3 couple of key points which are not in al line";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    return APP_ERR_OK;
}

APP_ERROR WarpAffine::CalAffineMatrix(float *keyPointBefore, int keyPointBeforeSize, float *keyPointAfter,
    int keyPointAfterSize, int affineMatrixSize)
{
    LogDebug << "WarpAffine[" << instanceId_ << "]: begin to calculate affine matrix.";
    int kPBefore[AFFINE_LEN];
    int kPAfter[AFFINE_LEN];
    float deno0;

    if (keyPointBeforeSize != keyPointAfterSize) {
        LogError << "WarpAffine[" << instanceId_ << "]: the size of keypoint must be the same";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    std::shared_ptr<KeyPointInfo> keyPointInfo = std::make_shared<KeyPointInfo>();
    keyPointInfo->keyPointBefore = keyPointBefore;
    keyPointInfo->keyPointBeforeSize = keyPointBeforeSize;
    keyPointInfo->keyPointAfter = keyPointAfter;
    keyPointInfo->keyPointAfterSize = keyPointAfterSize;
    keyPointInfo->kPBefore = kPBefore;
    keyPointInfo->kPAfter = kPAfter;
    APP_ERROR ret = KeyPointConversion(keyPointInfo, deno0);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    // cal the denominator which shared by the second row of affineMatrix_
    float deno1 = -deno0;
    // cal the first row of affineMatrix_
    affineMatrix_[KP_INDEX_0] =
        ((kPAfter[KP_INDEX_4] - kPAfter[KP_INDEX_0]) * (kPBefore[KP_INDEX_3] - kPBefore[KP_INDEX_1]) -
        (kPAfter[KP_INDEX_2] - kPAfter[KP_INDEX_0]) * (kPBefore[KP_INDEX_5] - kPBefore[KP_INDEX_1])) *
        1.0 / deno0;
    affineMatrix_[KP_INDEX_1] =
        ((kPAfter[KP_INDEX_4] - kPAfter[KP_INDEX_0]) * (kPBefore[KP_INDEX_2] - kPBefore[KP_INDEX_0]) -
        (kPAfter[KP_INDEX_2] - kPAfter[KP_INDEX_0]) * (kPBefore[KP_INDEX_4] - kPBefore[KP_INDEX_0])) *
        1.0 / deno1;
    affineMatrix_[KP_INDEX_2] = kPAfter[KP_INDEX_0] - affineMatrix_[KP_INDEX_0] * kPBefore[KP_INDEX_0] -
        affineMatrix_[KP_INDEX_1] * kPBefore[KP_INDEX_1];

    // cal the second row of affineMatrix_
    affineMatrix_[KP_INDEX_3] =
        ((kPAfter[KP_INDEX_3] - kPAfter[KP_INDEX_1]) * (kPBefore[KP_INDEX_3] - kPBefore[KP_INDEX_1]) -
        (kPAfter[KP_INDEX_3] - kPAfter[KP_INDEX_1]) * (kPBefore[KP_INDEX_3] - kPBefore[KP_INDEX_1])) *
        1.0 / deno0;
    affineMatrix_[KP_INDEX_4] =
        ((kPAfter[KP_INDEX_3] - kPAfter[KP_INDEX_1]) * (kPBefore[KP_INDEX_2] - kPBefore[KP_INDEX_0]) -
        (kPAfter[KP_INDEX_3] - kPAfter[KP_INDEX_1]) * (kPBefore[KP_INDEX_4] - kPBefore[KP_INDEX_0])) *
        1.0 / deno1;
    affineMatrix_[KP_INDEX_3] = kPAfter[KP_INDEX_1] - affineMatrix_[KP_INDEX_3] * kPBefore[KP_INDEX_0] -
        affineMatrix_[KP_INDEX_4] * kPBefore[KP_INDEX_1];
    return APP_ERR_OK;
}

/* *
 * @brief do warpaffine
 * @[in]: face: faceobject
 * @return: HIAI_StatusT
 */
APP_ERROR WarpAffine::ApplyWarpAffine(FaceObject &face, cv::Mat &imgBGR888)
{
    LogDebug << "WarpAffine[" << instanceId_ << "]: begin to apply warp affine.";
    // five standard key points after warp affine ,arranged by x0,y0,x1,y1..., scale 112*112
    float kPAfter[LANDMARK_NUM] = {
        30.2946, 51.6963, 65.5318, 51.5014, 48.0252,
        71.7366, 33.5493, 92.3655, 62.7299, 92.2041
        };

    // map the standard key points into  the image size
    // according to insightface open source //
    // https://github.com/deepinsight/insightface/blob/master/src/align/align_facescrub.py
    for (int i = 0; i < LANDMARK_NUM / KEY_POINT0_DIM; i++) {
        kPAfter[i * KEY_POINT0_DIM] =
            (float)widthOutput_ / STD_IMAGE_SIZE * kPAfter[i * KEY_POINT0_DIM] + KEY_POINT_OFFSET;
        kPAfter[i * KEY_POINT0_DIM + 1] = (float)heightOutput_ / STD_IMAGE_SIZE * kPAfter[i * KEY_POINT0_DIM + 1];
    }

    float *kPBefore = (float *)face.landmarks.hostData.get();
    if (CalAffineMatrix(kPBefore, LANDMARK_NUM, kPAfter, LANDMARK_NUM, AFFINE_LEN) != APP_ERR_OK) {
        // when affineMatrix_ cannot be calculated, do nothing
        LogWarn << "WarpAffine[" << instanceId_ << "]: fail to generate affineMatrix_.";
        face.imgAffine.buf.hostData.reset();
        face.imgAffine.buf.dataSize = 0;
        face.landmarks.hostData.reset();
        face.landmarks.dataSize = 0;
        return APP_ERR_COMM_FAILURE;
    }

    cv::Point2f srcPoints[KEY_POINT_COUNT];
    cv::Point2f destPoints[KEY_POINT_COUNT];
    for (int i = 0; i < KEY_POINT_COUNT; i++) {
        srcPoints[i] = cv::Point2f(kPBefore[i * KEY_POINT0_DIM], kPBefore[i * KEY_POINT0_DIM + 1]);
        destPoints[i] = cv::Point2f(kPAfter[i * KEY_POINT0_DIM], kPAfter[i * KEY_POINT0_DIM + 1]);
    }

    cv::Mat warpMat = cv::getAffineTransform(srcPoints, destPoints);
    auto warpDstData = std::make_shared<uint8_t>();
    warpDstData.reset(new uint8_t[widthOutput_ * heightOutput_ * YUV_BGR_SIZE_CONVERT_3], 
        std::default_delete<uint8_t[]>());
    cv::Mat warpDst(heightOutput_, widthOutput_, CV_8UC3, reinterpret_cast<void **>(warpDstData.get()));
    cv::warpAffine(imgBGR888, warpDst, warpMat, warpDst.size());
    face.imgAffine.buf.hostData = warpDstData;
    face.imgAffine.buf.dataSize = heightOutput_ * widthOutput_ * YUV_BGR_SIZE_CONVERT_3;
    face.imgAffine.height = heightOutput_;
    face.imgAffine.width = widthOutput_;

    return APP_ERR_OK;
}

double WarpAffine::GetRunTimeAvg()
{
    return warpAffineStatic_.GetRunTimeAvg();
}
} // namespace ascendFaceRecognition
