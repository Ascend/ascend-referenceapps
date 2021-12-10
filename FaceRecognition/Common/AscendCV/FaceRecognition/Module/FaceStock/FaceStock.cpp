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

#include "FaceStock.h"
#include "Log/Log.h"
#include "ErrorCode/ErrorCode.h"
#include "Common.h"
#include "DataTrans/DataTrans.pb.h"

#ifdef ASCEND_ACL_OPEN_VESION
#include "HdcChannel/HdcChannel.h"
#endif

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

// namespace ascendFaceRecognition
namespace ascendFaceRecognition {
namespace {
const int CONST_INT_2 = 2;
const int CONST_INT_3 = 3;
const int CONST_INT_4 = 4;
const float WIDTH_CROP_SCALA = 1.5;
}

FaceStock::FaceStock() {}


APP_ERROR FaceStock::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "FaceStock[" << instanceId_ << "]: FaceStock init star.";
    AssignInitArgs(initArgs);
    // get feature lib
    faceFeatureLib_ = (FaceFeatureLib *)initArgs.userData;
    isStop_ = false;
    LogDebug << "FaceStock[" << instanceId_ << "]: FaceStock init success.";
    return APP_ERR_OK;
}

APP_ERROR FaceStock::DeInit(void)
{
    return APP_ERR_OK;
}


APP_ERROR FaceStock::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "FaceStock[" << instanceId_ << "]: Begin to process data.";
    int ret;
    std::shared_ptr<FaceObject> faceObject = std::static_pointer_cast<FaceObject>(inputData);
    ImageInfo imageInfo;
    cv::Rect roi;
    cv::Mat cropedImage;
    cv::Mat dstResize;
    // validate inputs
    if (faceObject.get() == nullptr) {
        return APP_ERR_COMM_INVALID_POINTER;
    }
    LogDebug << "FaceStock[" << instanceId_ << "]: InsertFeatureToLib.";
    // insert feature and individual info
    ret = faceFeatureLib_->InsertFeatureToLib(faceObject, true);
    if (ret != APP_ERR_OK) {
        LogError << "FaceStock[" << instanceId_ << "]:Fail to registor person info" << GetAppErrCodeInfo(ret);
        return ret;
    }
    // get croped face data
    LogDebug << "FaceStock[" << instanceId_ << "]: InsertFeatureToLib end";
    std::shared_ptr<DataTrans> dataTrans = std::make_shared<DataTrans>();
    RegistResult *regResult = dataTrans->mutable_regresult();
    regResult->set_errcode(0);
    regResult->set_msg("success");
#ifdef ASCEND_ACL_OPEN_VESION
    HdcChannel::GetInstance()->SendData(HDC_REGIST_RESULT_CH_INDEX, dataTrans);
#else
    SendToNextModule(MT_REG_RESULT_HANDLER, dataTrans, 0);
#endif

    return APP_ERR_OK;
}
}
