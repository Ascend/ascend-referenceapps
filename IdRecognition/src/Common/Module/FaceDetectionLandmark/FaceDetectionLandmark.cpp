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
#include "FaceDetectionLandmark.h"

#include <memory>

#include "ConfigParser/ConfigParser.h"
#include "Log/Log.h"
#include "ModuleBase/ModuleBase.h"
#include "FileManager/FileManager.h"

#ifdef ASCEND_FACE_USE_ACL_DVPP
#include "DvppCommon/DvppCommon.h"
#else
#include "DvppCommonDevice/DvppCommonDevice.h"
#endif

using namespace Centerface;

#define DVPP_ALIGN_UP(x, align) ((((x) + ((align)-1)) / (align)) * (align))

namespace ascendFaceRecognition {
FaceDetectionLandmark::FaceDetectionLandmark()
{
}
FaceDetectionLandmark::~FaceDetectionLandmark()
{
    if (!isDeInited_) {
        DeInit();
    }
}

APP_ERROR FaceDetectionLandmark::DvppInit()
{
    LogInfo << "FaceDetectionLandmark[" << instanceId_ << "]::Begin to init dvpp.";

    outWidthStride_ = DVPP_ALIGN_UP(width_, VPC_WIDTH_ALIGN);
    outHeightStride_ = DVPP_ALIGN_UP(height_, VPC_HEIGHT_ALIGN);
    vpcOutBufferSize_ =
        batchSize_ * outWidthStride_ * outHeightStride_ * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2; // yuv420sp
    APP_ERROR ret = aclrtMalloc((void **)&vpcOutBuffer_, vpcOutBufferSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != APP_ERR_OK) {
        LogFatal << "aclrtMalloc failed, ret=" << ret << ", size=" << vpcOutBufferSize_ << ".";
        return ret;
    }
    LogDebug << "batchSize_=" << batchSize_ << ",vpcOutBufferSize_=" << vpcOutBufferSize_
             << ",outWidthStride_=" << outWidthStride_ << ",outHeightStride_=" << outHeightStride_ << ".";
    return APP_ERR_OK;
}

APP_ERROR FaceDetectionLandmark::ModelInit()
{
    LogInfo << "FaceDetectionLandmark[" << instanceId_ << "]::Begin to init model.";
    APP_ERROR ret = modelInfer_.Init(modelPath_);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::init model failed";
        return ret;
    }
    ret = GetNumInputs();
    if (ret != APP_ERR_OK) {
        LogError << "GetNumInputs fail";
        return APP_ERR_COMM_FAILURE;
    }
    ret = GetNumOutputs();
    if (ret != APP_ERR_OK) {
        LogError << "GetNumOutputs fail";
        return APP_ERR_COMM_FAILURE;
    }
    return APP_ERR_OK;
}

void FaceDetectionLandmark::GetSizePaddingResize(float resizeWidth, float resizeHeight,
                                                 float &oriWidth, float &oriHeight) const
{
    if (resizeHeight == 0 || resizeWidth == 0) {
        LogError << "resizeHeight or resizeWidth is 0.";
        return;
    }
    float resizeRatio = oriWidth / resizeWidth;
    if (resizeRatio < (oriHeight / resizeHeight)) {
        resizeRatio = oriHeight / resizeHeight;
    }

    oriWidth = static_cast<float>(resizeWidth) * resizeRatio;
    oriHeight = static_cast<float>(resizeHeight) * resizeRatio;
}

APP_ERROR FaceDetectionLandmark::GetNumInputs()
{
    aclmdlDesc *modelDesc = modelInfer_.GetModelDesc();
    uint32_t inputTensorSize = aclmdlGetNumInputs(modelDesc);
    if (inputTensorSize != 1) { // verify model input size
        LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::inputtensorsize invaild " << inputTensorSize << ".";
        return APP_ERR_COMM_FAILURE;
    }
    outWidthStride_ = DVPP_ALIGN_UP(width_, VPC_WIDTH_ALIGN);
    outHeightStride_ = DVPP_ALIGN_UP(height_, VPC_HEIGHT_ALIGN);
    vpcOutBufferSize_ =
        batchSize_ * outWidthStride_ * outHeightStride_ * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2; // yuv420sp
    uint32_t inputSize = aclmdlGetInputSizeByIndex(modelDesc, 0);
    if (inputSize != vpcOutBufferSize_) {
        LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::input size is invaild " << inputSize
                 << ",vpcOutBufferSize_=" << vpcOutBufferSize_ << ".";
        return APP_ERR_COMM_FAILURE;
    }
    return APP_ERR_OK;
}

APP_ERROR FaceDetectionLandmark::GetNumOutputs()
{
    aclmdlDesc *modelDesc = modelInfer_.GetModelDesc();
    size_t outputTensorSize = aclmdlGetNumOutputs(modelDesc);
    size_t singleSize = 0;
    APP_ERROR ret = APP_ERR_OK;
    LogDebug << "FaceDetectionLandmark[" << instanceId_ << "]::outputTensorSize=" << outputTensorSize << ".";
    for (size_t i = 0; i < outputTensorSize; i++) {
        size_t tensorSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
        LogDebug << "FaceDetectionLandmark[" << instanceId_ << "]::tensorSize=" << tensorSize << ", i=" << i << ".";
        void *outputBuf = nullptr;
        ret = aclrtMalloc(&outputBuf, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::aclrtMalloc failed, ret=" << ret << ".";
            if (outputBuf != nullptr) {
                aclrtFree(outputBuf);
                outputBuf = nullptr;
            }
            return ret;
        }
        singleSize += tensorSize / batchSize_;
        outputBufs_.push_back(outputBuf);
        outputSizes_.push_back(tensorSize);
    }

    if (runMode_ == ACL_HOST) { // under ACL_HOST mode, host memory is used
        ret = aclrtMallocHost(&outputBufsHost_, singleSize);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::malloc outputBufsHost_ failed, ret=" << ret
                     << ".";
            return ret;
        }
    }

    return APP_ERR_OK;
}

APP_ERROR FaceDetectionLandmark::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "FaceDetectionLandmark[" << instanceId_ << "]::Begin to init instance.";

    // 1. initialize member variables
    AssignInitArgs(initArgs);

    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::Fail to parse config params, ret=" << ret << "("
                 << GetAppErrCodeInfo(ret) << ").";
        return ret;
    }

    // 2. initialize dvpp acltodo??? it has been moved to decoder
    ret = DvppInit();
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::DvppInit failed.";
        return ret;
    }

    // 3. initialize inference model
    ret = ModelInit();
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::ModelInit failed.";
        return ret;
    }

    // 4. create yolo postprocess instance
    centerface_ = std::make_shared<Centerface::CenterFace>();
    if (centerface_ == nullptr) {
        LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::new centerface failed";
        return APP_ERR_COMM_INIT_FAIL;
    }
    isStop_ = false;

    return APP_ERR_OK;
}

APP_ERROR FaceDetectionLandmark::DeInit(void)
{
    LogInfo << "FaceDetectionLandmark[" << instanceId_ << "]::Begin to deinit instance.";

    isDeInited_ = true;
    StopAndDestroyQueue();

    // model deinit
    APP_ERROR ret = modelInfer_.DeInit();
    if (ret != APP_ERR_OK) {
        return ret;
    }

    for (uint32_t i = 0; i < outputBufs_.size(); i++) {
        ret = aclrtFree(outputBufs_[i]);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::aclrtFree failed.";
            return ret;
        }
        outputBufs_[i] = nullptr;
    }
    if (runMode_ == ACL_HOST) { // under ACL_HOST mode, free host memory
        if (outputBufsHost_ != nullptr) {
            ret = aclrtFreeHost(outputBufsHost_);
            if (ret != APP_ERR_OK) {
                LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::aclrtFreeHost failed.";
                return ret;
            }
        }
    }
    outputBufsHost_ = nullptr;

    if (vpcOutBuffer_ != nullptr) {
        ret = aclrtFree(vpcOutBuffer_);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::aclrtFree failed.";
            return ret;
        }
        vpcOutBuffer_ = nullptr;
    }

    return APP_ERR_OK;
}

inline std::vector<int> SortIndexVector(std::vector<float> &v)
{
    std::vector<int> idx;
    idx.resize(v.size());
    for (uint32_t i = 0; i < v.size(); i++) {
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(), [&v](int i, int j) { return v[i] > v[j]; });
    return idx;
}

APP_ERROR FaceDetectionLandmark::FaceSelection(std::vector<DetectInfo> &detectResult) const
{
    std::vector<DetectInfo> detectResultCopy(detectResult);
    std::vector<float> metric;

    for (auto i = detectResult.begin(); i != detectResult.end(); i++) {
        float area = ((*i).width) * ((*i).height);
        metric.push_back(area * ((*i).confidence));
    }
    auto idx = SortIndexVector(metric); // sort by descend order and return the index
    detectResult.clear();
    for (auto i = idx.begin(); (i != idx.begin() + maxFaceNumPerFrame_) && (i != idx.end()); i++) {
        detectResult.push_back(detectResultCopy[*i]);
    }
    return APP_ERR_OK;
}

APP_ERROR FaceDetectionLandmark::PostProcessDetection(void)
{
    APP_ERROR ret = APP_ERR_OK;
    size_t singleSize = 0;
    for (uint32_t j = 0; j < outputBufs_.size(); j++) {
        singleSize += outputSizes_[j] / batchSize_;
    }
    for (uint32_t j = 0; j < batchSize_; j++) {
        std::shared_ptr<FrameAiInfo> aiInfo = inputArgQueue_[j];
        int32_t idx = 0;
        std::vector<void *> outTensorAddrs;
        for (uint32_t i = 0; i < outputBufs_.size(); i++) {
            int32_t offset = j * outputSizes_[i] / batchSize_;
            void *resultTensor = (void *)((uint8_t *)outputBufs_[i] + offset);

            // under ACL_device mode, memory copy is not required
            if (runMode_ == ACL_DEVICE) {
                outTensorAddrs.push_back(resultTensor);
                continue;
            }

            // under ACL_HOST mode, copy result from device to host
            void *tmpBuf = (void *)((uint8_t *)outputBufsHost_ + idx);
            ret = aclrtMemcpy(tmpBuf, outputSizes_[i] / batchSize_, resultTensor, outputSizes_[i] / batchSize_,
                ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != APP_ERR_OK) {
                LogFatal << "FaceDetectionLandmark[" << instanceId_
                         << "]::memcopy for outputBufsHost_ failed, ret=" << ret << ".";
                return ret;
            }
            outTensorAddrs.push_back(tmpBuf);
            idx += outputSizes_[i] / batchSize_;
        }
        LogDebug << "FaceDetectionLandmark[" << instanceId_ << "]::width=" << aiInfo->imgOrigin.width
                 << ",height=" << aiInfo->imgOrigin.height << ",singleSize=" << singleSize << ".";

        ret = ProcessCenterfaceResult(aiInfo, singleSize, outTensorAddrs);
        if (ret != APP_ERR_OK) {
            return ret;
        }
        if (aiInfo->face.size() > 1 && pipeMode_ == PIPELINE_MODE_REG) {
            isMultiFace_ = APP_ERROR_FACE_WEB_USE_MUL_FACE;
            LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]::multiple faces, name = "
            << aiInfo->info.personInfo.name << " face count = " << aiInfo->face.size();
            continue;
        }
        outputQueVec_[instanceId_]->Push(aiInfo, true);
    }
    return APP_ERR_OK;
}

APP_ERROR FaceDetectionLandmark::ProcessCenterfaceResult(std::shared_ptr<FrameAiInfo> frameAiInfo, size_t singleSize,
    std::vector<void *> &outTensorAddrs)
{
    std::vector<DetectInfo> boxinfo;
    ImageInfoPostProcess imageInfo = { 0 };
    imageInfo.widthModelInput = width_;
    imageInfo.heightModelInput = height_;
    imageInfo.widthOrigin = frameAiInfo->imgOrigin.width;
    imageInfo.heightOrigin = frameAiInfo->imgOrigin.height;

    GetSizePaddingResize(width_, height_, imageInfo.widthOrigin, imageInfo.heightOrigin);

    int numDetected = centerface_->Process(outTensorAddrs, imageInfo, boxinfo);
    if (numDetected < 0) {
        LogError << "FaceDetectionLandmark[" << instanceId_ << "]::PostProcess failed, numDetected=" << numDetected
                 << ".";
        if (pipeMode_ == PIPELINE_MODE_REG) {
            isMultiFace_ = APP_ERROR_FACE_WEB_USE_NO_FACE;
            LogInfo << "FaceDetectionLandmark[" << instanceId_ << "]::no face detected, the face registration is not "
                "complete, name = " << frameAiInfo->info.personInfo.name << ".";
        } else if (pipeMode_ == PIPELINE_MODE_SEARCH) {
            LogInfo << "FaceDetectionLandmark[" << instanceId_ << "]::no face detected.";
        }
        return APP_ERR_COMM_FAILURE;
    }
    LogDebug << "FaceDetectionLandmark[" << instanceId_ << "]::boxinfoNum = " << boxinfo.size() << ".";

    APP_ERROR ret = ConstructDetectResult(frameAiInfo, boxinfo);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    if (numDetected > 0) {
        FaceSelection(frameAiInfo->detectResult); // select the top k faces from one frame
    }

    return APP_ERR_OK;
}

APP_ERROR FaceDetectionLandmark::ConstructDetectResult(std::shared_ptr<FrameAiInfo> frameAiInfo,
    std::vector<DetectInfo> &boxinfo) const
{
    for (uint32_t i = 0; i < boxinfo.size(); i++) {
        DetectInfo tmp;
        tmp.classId = boxinfo[i].classId;
        tmp.confidence = boxinfo[i].confidence;
        tmp.minx = boxinfo[i].minx;
        tmp.miny = boxinfo[i].miny;
        tmp.height = boxinfo[i].height;
        tmp.width = boxinfo[i].width;

        const int idxFactor = 2;
        const int landmarkSize = LANDMARK_NUM;
        for (int j = 0; j < landmarkSize / idxFactor; ++j) {
            tmp.landmarks[j * idxFactor] = boxinfo[i].landmarks[j * idxFactor];
            tmp.landmarks[j * idxFactor + 1] = boxinfo[i].landmarks[j * idxFactor + 1];
            LogDebug << "landmark location: point " << j << " (" << tmp.landmarks[j * idxFactor] << ", "
                     << tmp.landmarks[j * idxFactor + 1] << ")";
        }

        const int landmarkByteSize = landmarkSize * sizeof(float);
        std::shared_ptr<uint8_t> landmarkBuffer = std::make_shared<uint8_t>();
        landmarkBuffer.reset(new uint8_t[landmarkByteSize], std::default_delete<uint8_t[]>());
        std::copy((uint8_t *)tmp.landmarks, (uint8_t *)tmp.landmarks + landmarkByteSize, landmarkBuffer.get());

        FaceObject faceObject;
        faceObject.info = tmp;
        faceObject.landmarks.hostData = landmarkBuffer;
        faceObject.landmarks.dataSize = landmarkByteSize;
        frameAiInfo->face.push_back(faceObject);

        LogDebug << "FaceDetectionLandmark[" << instanceId_ << "]::classId=" << tmp.classId << ", minx=" << tmp.minx
                 << ", miny=" << tmp.miny << ", height = " << tmp.height << ", width=" << tmp.width << ".";
        frameAiInfo->detectResult.push_back(tmp);
    }
    return APP_ERR_OK;
}

APP_ERROR FaceDetectionLandmark::Process(std::shared_ptr<void> inputData)
{
    faceDecLMStatic_.RunTimeStatisticStart("FaceDetectionLandmark_Excute_Time", instanceId_);
    centerfacePreStatic_.RunTimeStatisticStart("CenterfacePre_Excute_Time", instanceId_);

    LogDebug << "FaceDetectionLandmark[" << instanceId_ << "]::Begin to process data.";
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::static_pointer_cast<FrameAiInfo>(inputData);
    if (frameAiInfo.get() == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }

    uint32_t outBufferSize =
        outWidthStride_ * outHeightStride_ * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2; // yuv420sp
    APP_ERROR ret = aclrtMemcpy(vpcOutBuffer_ + vpcBufferOffset_, outBufferSize,
        frameAiInfo->detectImg.buf.deviceData.get(), outBufferSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != APP_ERR_OK) {
        LogError << "FaceDetectionLandmark[" << instanceId_ << "]::aclrtMemcpy failed, retDevice[" << ret << "].";
        return APP_ERR_OK;
    }
    vpcBufferOffset_ += outBufferSize;
    inputArgQueue_.push_back(frameAiInfo);

    if (inputArgQueue_.size() < batchSize_) { // get more data if batch not complete
        return APP_ERR_OK;
    }
    vpcBufferOffset_ = 0;
    std::vector<void *> inputBufs;
    inputBufs.push_back(vpcOutBuffer_);
    std::vector<size_t> inputSizes;
    inputSizes.push_back(vpcOutBufferSize_);
    centerfacePreStatic_.RunTimeStatisticStop();

    // model Inference
    centerfaceStatic_.RunTimeStatisticStart("CenterFace_model_excute_time", instanceId_);
    ret = modelInfer_.ModelInference(inputBufs, inputSizes, outputBufs_, outputSizes_);
    centerfaceStatic_.RunTimeStatisticStop();
    if (ret != APP_ERR_OK) {
        LogError << "FaceDetectionLandmark[" << instanceId_ << "]::Fail to execute model.";
        return ret;
    }
    centerfacePostStatic_.RunTimeStatisticStart("CenterfacePost_Excute_Time", instanceId_);
    PostProcessDetection();
    centerfacePostStatic_.RunTimeStatisticStop();
    inputArgQueue_.clear();

    faceDecLMStatic_.RunTimeStatisticStop();

    return ret;
}

APP_ERROR FaceDetectionLandmark::ParseConfig(ConfigParser &configParser)
{
    LogInfo << "FaceDetectionLandmark[" << instanceId_ << "]::Begin to parse config file.";
    std::string itemCfgStr;

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".batch_size");
    APP_ERROR ret = configParser.GetUnsignedIntValue(itemCfgStr, batchSize_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".channel_count");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, channelCount_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".width");
    ret = configParser.GetIntValue(itemCfgStr, width_);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    if ((width_ % VPC_WIDTH_ALIGN) != 0) {
        LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]: invalid value of " << itemCfgStr
                 << ", it has to be a multiple of " << VPC_WIDTH_ALIGN << ".";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".height");
    ret = configParser.GetIntValue(itemCfgStr, height_);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    if ((height_ % VPC_HEIGHT_ALIGN) != 0) {
        LogFatal << "FaceDetectionLandmark[" << instanceId_ << "]: invalid value of " << itemCfgStr
                 << ", it has to be a multiple of " << VPC_HEIGHT_ALIGN << ".";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".model_path");
    ret = configParser.GetStringValue(itemCfgStr, modelPath_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::to_string(instanceId_) + std::string(".max_face_num_per_frame");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, maxFaceNumPerFrame_);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    LogDebug << "FaceDetectionLandmark[" << instanceId_ << "]: batchSize_:" << batchSize_ << " width_:" << width_
             << " height_:" << height_ << " modelPath_:" << modelPath_.c_str()
             << " maxFaceNumPerFrame_:" << maxFaceNumPerFrame_ << ".";

    return ret;
}

double FaceDetectionLandmark::GetRunTimeAvg()
{
    return faceDecLMStatic_.GetRunTimeAvg();
}

APP_ERROR FaceDetectionLandmark::GetFaceStatus() const
{
    return isMultiFace_;
}

void FaceDetectionLandmark::SetFaceStatus()
{
    isMultiFace_ = APP_ERR_OK;
}
} // namespace ascendFaceRecognition