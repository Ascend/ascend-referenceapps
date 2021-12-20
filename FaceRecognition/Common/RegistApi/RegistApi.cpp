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

#include "RegistApi.h"
#include "Log/Log.h"
#include "Common.h"
#ifdef ASCEND_ACL_OPEN_VESION
#include "HdcChannel/HdcChannel.h"
#else
#include "SystemManager/SystemManager.h"
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <cstdio>


namespace ascendFaceRecognition {
static int REGISTER_FAIL = 1;

std::shared_ptr<RegistApi> RegistApi::GetInstance()
{
    static std::shared_ptr<RegistApi> registApiInst = std::make_shared<RegistApi>();
    return registApiInst;
}

RegistApi::RegistApi() {}

RegistApi::~RegistApi() {}

void RegistApi::RegistFace(std::string &name, const std::string &imgStr, CallbackFaceRegisterFunc callback)
{
    callback_ = callback;
    std::shared_ptr<DataTrans> dataTrans = std::make_shared<DataTrans>();
    APP_ERROR ret = RegistTheFace(name, imgStr, dataTrans);
    if (ret != APP_ERR_OK) {
        callback(REGISTER_FAIL, "Regist Failed");
        LogError << "Regist The Face Failed";
    }

#ifdef ASCEND_ACL_OPEN_VESION
    HdcChannel::GetInstance()->SendData(HDC_REGIST_CH_INDEX, dataTrans);
#else
    SystemManager::GetInstance()->SendDataToModule("Regist", MT_IMAGE_DECODER, dataTrans, 0);
#endif
}

void RegistApi::HandleRegResult(std::shared_ptr<RegistResult> regResult)
{
    LogDebug << "RegResultHandler Recv ErrCode: " << regResult->errcode() << " msg: " << regResult->msg();
    if (callback_ != NULL) {
        callback_(regResult->errcode(), regResult->msg());
    }
    callback_ = NULL;
}


APP_ERROR RegistApi::RegistTheFace(std::string name, const std::string &imgStr,
    std::shared_ptr<DataTrans> dataTrans) const
{
    // judge image format
    if (!IsJpeg(imgStr)) {
        LogWarn << "The image is not jpeg/jpg format.";
        return APP_ERR_COMM_FAILURE;
    }

    std::vector<unsigned char> ImVec(imgStr.begin(), imgStr.end());
    cv::Mat img = cv::imdecode(ImVec, CV_LOAD_IMAGE_COLOR);
    if (img.data == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }
    // register face
    RegistFaceInfoDataTrans *registFaceInfo = dataTrans->mutable_registface();
    registFaceInfo->set_uuid(name);
    PictureDataTrans *avatar = registFaceInfo->mutable_avatar();
    avatar->set_width(img.cols);
    avatar->set_height(img.rows);
    avatar->set_datasize(imgStr.length());
    avatar->set_data(imgStr.c_str(), imgStr.length());
    LogDebug << "width:" << img.cols << ", height:" << img.rows << ", bufferSize:" << imgStr.length();
    return APP_ERR_OK;
}

bool RegistApi::IsJpeg(const std::string &imgStr) const
{
    if (imgStr.length() < JPEG_HEADER_BYTE_LEN) {
        return false;
    }
    const char *buff = imgStr.c_str();
    return ((uint8_t)buff[0] == JPEG_FIRST_BYTE) && ((uint8_t)buff[1] == JPEG_SECOND_BYTE);
}
}
