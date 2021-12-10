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
#include "FaceSearch.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "Common.h"
#include "Log/Log.h"
#include "ErrorCode/ErrorCode.h"
#include "DataTrans/DataTrans.pb.h"
#ifdef ASCEND_ACL_OPEN_VESION
#include "HdcChannel/HdcChannel.h"
#endif


namespace ascendFaceRecognition {
FaceSearch::FaceSearch() {}

FaceSearch::~FaceSearch() {}

APP_ERROR FaceSearch::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "FaceSearch[" << instanceId_ << "]: FaceSearch init start.";

    AssignInitArgs(initArgs);
    // get feature lib
    faceFeatureLib_ = (FaceFeatureLib *)initArgs.userData;

    isStop_ = false;
    LogDebug << "FaceSearch[" << instanceId_ << "]: FaceSearch init success.";
    return APP_ERR_OK;
}

APP_ERROR FaceSearch::DeInit(void)
{
    return APP_ERR_OK;
}

APP_ERROR FaceSearch::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "FaceSearch[" << instanceId_ << "]: Begin to process data.";
    std::shared_ptr<FaceObject> faceObject = std::static_pointer_cast<FaceObject>(inputData);
    if (faceObject.get() == nullptr) {
        return APP_ERR_COMM_INVALID_POINTER;
    }

    LogDebug << "Send Host ChannelID:" << faceObject->frameInfo.channelId << " FrameID:" <<
        faceObject->frameInfo.frameId << " Gender:" << faceObject->personInfo.gender;

    int ret = faceFeatureLib_->GetPersonInfoByFeature(faceObject);
    if (ret != APP_ERR_OK) {
        LogError << "FaceSearch[" << instanceId_ << "]: Fail to get person info.";
        return ret;
    }
    // get face message
    std::shared_ptr<DataTrans> dataTrans = std::make_shared<DataTrans>();
    FaceAttributeDataTrans *faceInfo = dataTrans->mutable_faceinfo();
    faceInfo->set_channelid(faceObject->frameInfo.channelId);
    faceInfo->set_frameid(faceObject->frameInfo.frameId);
    faceInfo->set_trackid(faceObject->trackInfo.id);
    faceInfo->set_name(faceObject->personInfo.uuid);
    faceInfo->set_gender(faceObject->personInfo.gender);
    faceInfo->set_age(faceObject->personInfo.age);
    faceInfo->set_mask(faceObject->personInfo.mask);
    faceInfo->set_similarity(faceObject->personInfo.similarity);
    PictureDataTrans *avatar = faceInfo->mutable_avatar();
    avatar->set_width(faceObject->imgCroped.width);
    avatar->set_height(faceObject->imgCroped.height);
    avatar->set_datasize(faceObject->imgCroped.buf.dataSize);
#ifdef ASCEND_ACL_OPEN_VESION
    avatar->set_data(faceObject->imgCroped.buf.deviceData.get(), faceObject->imgCroped.buf.dataSize);
    HdcChannel::GetInstance()->SendData(HDC_FACE_DETAIL_CH_INDEX, dataTrans);
#else
    avatar->set_data(faceObject->imgCroped.buf.hostData.get(), faceObject->imgCroped.buf.dataSize);
    SendToNextModule(MT_FACE_DETAIL_INFO, dataTrans, 0);
#endif
    return APP_ERR_OK;
}
} // End ascendFaceRecognition
