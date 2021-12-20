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
#include "FaceDetailInfo/FaceDetailInfo.h"
#include "Log/Log.h"
#include "FileEx/FileEx.h"
#include "DataTrans/DataTrans.pb.h"
#include "ChannelStatus/ChannelStatus.h"
#include "DeliveryChannel/DeliveryChannel.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>


namespace ascendFaceRecognition {
FaceDetailInfo::FaceDetailInfo() {}

FaceDetailInfo::~FaceDetailInfo() {}

APP_ERROR FaceDetailInfo::ParseConfig(ConfigParser &configParser)
{
    // load and parse config file
    LogDebug << "FaceDetailInfo[" << instanceId_ << "]: begin to parse config values.";
    std::string itemCfgStr = moduleName_ + std::string(".similarity_threshold");
    APP_ERROR ret = configParser.GetDoubleValue(itemCfgStr, similarityThreshold_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceFeature[" << instanceId_ << "]: Fail to get config variable named " << itemCfgStr << ".";
        return ret;
    }
    return ret;
}

APP_ERROR FaceDetailInfo::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "FaceDetailInfo[" << instanceId_ << "]: FaceDetailInfo init start.";
    // initialize member variables
    AssignInitArgs(initArgs);

    int ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceDetailInfo[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }

    LogDebug << "FaceDetailInfo[" << instanceId_ << "]: FaceDetailInfo init success.";
    return APP_ERR_OK;
}

APP_ERROR FaceDetailInfo::DeInit(void)
{  
    return APP_ERR_OK;
}

APP_ERROR FaceDetailInfo::Process(std::shared_ptr<void> inputData)
{
    // start threads from ModuleBase
    // get data from device
    std::shared_ptr<DataTrans> dataTrans = std::static_pointer_cast<DataTrans>(inputData);

    std::shared_ptr<FaceAttributeDataTrans> faceInfo = std::make_shared<FaceAttributeDataTrans>(dataTrans->faceinfo());
    if (faceInfo == nullptr) {
        LogError << "FaceDetailInfo[" << instanceId_ << "]: invalid input data.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    LogDebug << "Recv Ch #" << faceInfo->channelid() << " FrameId #" << faceInfo->frameid() <<
        " Name " << faceInfo->name() << " Gender " << faceInfo->gender() << " Age " << faceInfo->age() <<
        " Mask:" << faceInfo->mask() << " Similar:" << faceInfo->similarity();

    uint32_t webChnannelId = faceInfo->channelid() + 1;
    if (!ChannelStatus::GetInstance()->IsAlive((int)webChnannelId)) {
        return APP_ERR_OK;
    }
    std::vector<unsigned char> buff;
    MatToByteArray(faceInfo, buff);
    // deserializer of buff
    std::shared_ptr<PictureDataT> pictureDataPtr = std::make_shared<PictureDataT>();
    std::string buffStr((char *)buff.data(), buff.size());
    pictureDataPtr->channelId = faceInfo->channelid() + 1;
    pictureDataPtr->data = buffStr;
    AttrT cameraAttr("camera", "camera" + std::to_string(faceInfo->channelid() + 1));
    AttrT genderAttr("sex", faceInfo->gender());
    AttrT nameAttr("name", "---");
    AttrT ageAttr("age", std::to_string(faceInfo->age()));
    if (faceInfo->similarity() > similarityThreshold_) {
        LogDebug << "FaceDetailInfo[" << instanceId_ << "] similarity: " << faceInfo->similarity() << "";
        nameAttr.value = faceInfo->name();
    }
    // save parameters
    pictureDataPtr->attrList.push_back(nameAttr);
    pictureDataPtr->attrList.push_back(cameraAttr);
    pictureDataPtr->attrList.push_back(genderAttr);
    pictureDataPtr->attrList.push_back(ageAttr);
    DeliveryChannel::GetInstance()->SendData(DELIVERY_PICTURE_INDEX, pictureDataPtr);
    return APP_ERR_OK;
}

int FaceDetailInfo::MatToByteArray(std::shared_ptr<FaceAttributeDataTrans> faceInfo, std::vector<unsigned char> &buff)
{
    const int BGR_CHANNEL = 3;
    const int YUV_CHANNEL = 2;
    cv::Mat srcN12 = cv::Mat(faceInfo->avatar().height() * BGR_CHANNEL / YUV_CHANNEL, faceInfo->avatar().width(),
        CV_8UC1, (char *)faceInfo->avatar().data().c_str());
    cv::Mat dstBgr = cv::Mat(faceInfo->avatar().height(), faceInfo->avatar().width(), CV_8UC3);
    cv::cvtColor(srcN12, dstBgr, cv::COLOR_YUV2BGR_NV12);
    const int compressRate = 95;
    std::vector<int> param;
    param.push_back(cv::IMWRITE_JPEG_QUALITY);
    param.push_back(compressRate); // default(95) 0-100
    cv::imencode(".jpg", dstBgr, buff, param);
    return 0;
}
} // End ascendFaceRecognition
