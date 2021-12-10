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

#include "FaceDetection.h"
#include <memory>
#include "CostStatistic/CostStatistic.h"
#include "ConfigParser/ConfigParser.h"
#include "Log/Log.h"
#include "ModuleBase/ModuleBase.h"


#define DVPP_ALIGN_UP(x, align) ((((x) + ((align)-1)) / (align)) * (align))

namespace ascendFaceRecognition {
namespace {
const int YUV_BGR_SIZE_CONVERT_3 = 3;
const int YUV_BGR_SIZE_CONVERT_2 = 2;
const int DVPP_WIDTH_ALIGN = 16;
const int DVPP_HEIGHT_ALIGN = 2;
const float SEC2MS = 1000.0;
const int CPU_OUTPUT_LAYER_NUN = 3;
const int AICORE_OUTPUT_LAYER_NUN = 2;
const int CPU_INPUT_LAYER_NUN = 1;
const int AICORE_INPUT_LAYER_NUN = 2;
const int AICORE_OUTPUT_DETECT_RESULT_CHANNAL = 6;
const int AICORE_OUTPUT_XMIN_CHANNAL_INDEX = 0;
const int AICORE_OUTPUT_YMIN_CHANNAL_INDEX = 1;
const int AICORE_OUTPUT_XMAX_CHANNAL_INDEX = 2;
const int AICORE_OUTPUT_YMAX_CHANNAL_INDEX = 3;
const int AICORE_OUTPUT_CONFIDENCE_CHANNAL_INDEX = 4;
const int AICORE_OUTPUT_CLASS_ID_CHANNAL_INDEX = 5;
const double COST_TIME_MS_THRESHOLD = 10.;
}

FaceDetection::FaceDetection()
{
    isStop_ = true;
    instanceId_ = -1;
}

FaceDetection::~FaceDetection() {}

APP_ERROR FaceDetection::ModelInit()
{
    LogDebug << "FaceDetection[" << instanceId_ << "]::Begin to init model.";
    APP_ERROR ret;
    // init model
    modelInfer_ = ModelResource::GetInstance().GetModelProcess(modelPath_);
    if (modelInfer_ == nullptr) {
        LogFatal << "FaceDetection[" << instanceId_ << "]::init model failed";
        return APP_ERR_COMM_FAILURE;
    }

    // verify model input size
    uint32_t inputTensorSize = modelInfer_->GetModelNumInputs();
    for (size_t i = 0; i < inputTensorSize; i++) {
        size_t inputSize = modelInfer_->GetModelInputSizeByIndex(i);
        LogDebug << "FaceDetection[" << instanceId_ << "]::input tensorSize=" << inputSize << ", i=" << i << "";
        void *inputBuf = nullptr;
        ret = aclrtMalloc(&inputBuf, inputSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceDetection[" << instanceId_ << "]::aclrtMalloc failed, ret=" << ret << ".";
            return ret;
        }
        inputBufs_.push_back(inputBuf);
        inputSizes_.push_back(inputSize);
    }

    // init output resource
    size_t outputTensorSize = modelInfer_->GetModelNumOutputs();
    LogDebug << "FaceDetection[" << instanceId_ << "]::outputTensorSize=" << outputTensorSize << "";
    for (size_t i = 0; i < outputTensorSize; i++) {
        size_t outputSize = modelInfer_->GetModelOutputSizeByIndex(i);
        LogDebug << "FaceDetection[" << instanceId_ << "]::output tensorSize=" << outputSize << ", i=" << i << "";
        void *outputBuf = nullptr;
        ret = aclrtMalloc(&outputBuf, outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != APP_ERR_OK) {
            LogFatal << "FaceDetection[" << instanceId_ << "]::aclrtMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputBufs_.push_back(outputBuf);
        outputSizes_.push_back(outputSize);

        std::shared_ptr<void> outputBufHost = std::make_shared<uint8_t>();
        outputBufHost.reset(new uint8_t[outputSize], std::default_delete<uint8_t[]>());
        if (outputBufHost == nullptr) {
            LogFatal << "FaceDetection[" << instanceId_ << "] new failed, ret=" << ret << ".";
            return APP_ERR_COMM_ALLOC_MEM;
        }
        outputBufsHost_.push_back(outputBufHost);
    }

    return APP_ERR_OK;
}

APP_ERROR FaceDetection::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogDebug << "FaceDetection[" << instanceId_ << "]::Begin to init instance.";

    // initialize member variables
    AssignInitArgs(initArgs);

    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceDetection[" << instanceId_ << "]::Fail to parse config params, ret=" << ret << "(" <<
            GetAppErrCodeInfo(ret) << ").";
        return ret;
    }

    // initialize inference model
    ret = ModelInit();
    if (ret != APP_ERR_OK) {
        LogFatal << "FaceDetection[" << instanceId_ << "]::ModelInit failed.";
        return ret;
    }

    // create yolo postprocess instance
    std::vector<uint32_t> anchors = { 54, 71, 77, 102, 122, 162, 207, 268, 15, 19, 21, 26,
        28, 36, 38, 49,  3,   4,   5,   6,   7,  9,  10, 13 };
    const uint32_t scaleAnchorNumber = 4;
    const uint32_t classNumber = 1;
    YoloParameter parameter = {};
    parameter.scaleAnchorNumber = scaleAnchorNumber;
    parameter.classNumber = classNumber;
    parameter.netWidth = width_;
    parameter.netHeight = height_;
    parameter.batch = batchSize_;

    yolov3 = std::make_shared<Yolov3Batch>(anchors, parameter);

    isStop_ = false;
    return APP_ERR_OK;
}

APP_ERROR FaceDetection::DeInit(void)
{
    LogDebug << "FaceDetection[" << instanceId_ << "]::Begin to deinit instance.";

    // model deinit
    for (uint32_t i = 0; i < outputBufs_.size(); i++) {
        aclrtFree(outputBufs_[i]);
        outputBufs_[i] = nullptr;
    }

    for (uint32_t i = 0; i < inputBufs_.size(); i++) {
        aclrtFree(inputBufs_[i]);
        inputBufs_[i] = nullptr;
    }

    return APP_ERR_OK;
}

inline std::vector<int> SortIndexVector(std::vector<float> &v)
{
    std::vector<int> idx(v.size());
    for (uint32_t i = 0; i < v.size(); i++) {
        idx[i] = i;
    }
    // sort and return index
    std::sort(idx.begin(), idx.end(), [&v](int i, int j) { return v[i] > v[j]; });
    return idx;
}

APP_ERROR FaceDetection::FaceSelection(std::vector<DetectInfo> &detectResult) const
{
    std::vector<DetectInfo> detectResultCopy(detectResult);
    std::vector<float> metric;

    for (auto i = detectResult.begin(); i != detectResult.end(); i++) {
        float area = ((*i).width) * ((*i).height);
        metric.push_back(area * ((*i).confidence));
    }

    auto idx = SortIndexVector(metric); // sort by descend order and return the index
    detectResult.clear();
    const uint32_t widthThreshold = 10;
    const uint32_t heightThreshold = 7;
    for (auto i = idx.begin(); (i != idx.begin() + maxFaceNumPerFrame_) && (i != idx.end()); i++) {
        DetectInfo detectInfo = detectResultCopy[*i];
        if (detectInfo.width >= widthThreshold && detectInfo.height >= heightThreshold) {
            detectResult.push_back(detectInfo);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR FaceDetection::PreProcess(std::shared_ptr<FrameAiInfo> frameAiInfo)
{
    if (inputBufs_.size() != CPU_INPUT_LAYER_NUN && inputBufs_.size() != AICORE_INPUT_LAYER_NUN) {
        LogError << "FaceDetection[" << instanceId_ << "]::input buffer size(" << inputBufs_.size() <<
            ") is wrong, must be greater than 2 or 1";
        return APP_ERR_COMM_ALLOC_MEM;
    }
    if (inputArgQueue_.size() >= batchSize_) {
        LogError << "FaceDetection[" << instanceId_ << "]::batch size(" << inputArgQueue_.size() <<
            ") is wrong, must be less than " << batchSize_ << "";
        inputArgQueue_.clear();
        return APP_ERR_COMM_OUT_OF_RANGE;
    }

    uint32_t batchLen = inputSizes_[0] / batchSize_;
    uint32_t batchOffset = inputArgQueue_.size() * batchLen;
#ifndef ASCEND_ACL_OPEN_VESION

    APP_ERROR ret = aclrtMemcpy((uint8_t *)inputBufs_[0] + batchOffset, batchLen,
        frameAiInfo->detectImg.buf.deviceData.get(), batchLen, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != APP_ERR_OK) {
        LogError << "FaceDetection[" << instanceId_ << "]::aclrtMemcpy failed, retDevice[" << ret << "]";
        return APP_ERR_OK;
    }
#else
    uint8_t *srcData = frameAiInfo->detectImg.buf.deviceData.get();
    std::copy(srcData, srcData + batchLen, (uint8_t *)inputBufs_[0] + batchOffset);
#endif
    if (inputBufs_.size() == AICORE_INPUT_LAYER_NUN) {
        batchLen = inputSizes_[1] / batchSize_;
        batchOffset = inputArgQueue_.size() * batchLen;
        std::vector<float> imgInfo = { float(frameAiInfo->detectImg.height), float(frameAiInfo->detectImg.width),
                                       float(frameAiInfo->imgOrigin.height), float(frameAiInfo->imgOrigin.width) };
        for (size_t i = 0; i < imgInfo.size(); i++) {
            float imgSize = imgInfo[i];
#ifndef ASCEND_ACL_OPEN_VESION
            APP_ERROR ret = aclrtMemcpy((uint8_t *)inputBufs_[1] + batchOffset + i * sizeof(float), sizeof(float),
                (void *)&imgSize, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != APP_ERR_OK) {
                LogError << "FaceDetection[" << instanceId_ << "]::aclrtMemcpy failed, retDevice[" << ret << "]";
                return ret;
            }
#else
            std::copy((uint8_t *)&imgSize, (uint8_t *)&imgSize + sizeof(float),
                (uint8_t *)inputBufs_[1] + batchOffset + i * sizeof(float));
#endif
        }
    }
    return APP_ERR_OK;
}

APP_ERROR FaceDetection::PostProcessCPU(const std::vector<void *> &tensors) const
{
    if (tensors.size() != CPU_OUTPUT_LAYER_NUN) {
        LogFatal << "FaceDetection[" << instanceId_ << "]::PostProcessCPU output layers number is invalid";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    std::vector<std::vector<YoloBox>> batchBoxes;
    std::vector<uint32_t> imgWidths;
    std::vector<uint32_t> imgHeights;
    for (uint32_t i = 0; i < batchSize_; i++) {
        std::shared_ptr<FrameAiInfo> frameAiInfo = inputArgQueue_[i];
        uint32_t imgHeight = frameAiInfo->imgOrigin.height;
        uint32_t imgWidth = frameAiInfo->imgOrigin.width;
        if (imgHeight < height_ && imgWidth < width_) {
            imgWidths.push_back(width_);
            imgHeights.push_back(height_);
        } else {
            imgWidths.push_back(imgWidth);
            imgHeights.push_back(imgHeight);
        }
    }
    yolov3->PostProcess(tensors, imgWidths, imgHeights, batchBoxes);
    // get face coordinate, height and width
    for (uint32_t j = 0; j < batchSize_; j++) {
        std::shared_ptr<FrameAiInfo> aiInfo = inputArgQueue_[j];
        std::vector<YoloBox> bboxes = batchBoxes[j];
        for (uint32_t i = 0; i < bboxes.size(); i++) {
            DetectInfo tmp;
            tmp.classId = bboxes[i].classId;
            tmp.confidence = bboxes[i].score;
            tmp.minx = bboxes[i].x;
            tmp.miny = bboxes[i].y;
            tmp.height = bboxes[i].h;
            tmp.width = bboxes[i].w;
            LogDebug << "FaceDetection[" << instanceId_ << "]::classId=" << tmp.classId << ", minx=" << tmp.minx <<
                ", miny=" << tmp.miny << ", height = " << tmp.height << ", width=" << tmp.width << "";
            aiInfo->detectResult.push_back(tmp);
        }

        if (bboxes.size() > 0) {
            FaceSelection(aiInfo->detectResult); // select the top k faces from one frame
        }
    }
    return APP_ERR_OK;
}
void FaceDetection::GetResultAiCore(DetectInfo &detectInfo, const uint8_t *resultAddress, const uint32_t &boxIndex,
    const uint32_t &boxSize) const
{
    for (size_t k = 0; k < AICORE_OUTPUT_DETECT_RESULT_CHANNAL; k++) {
        float value = *(reinterpret_cast<const float *>(resultAddress + (k * boxSize + boxIndex) * sizeof(float)));
        switch (k) {
            case AICORE_OUTPUT_XMIN_CHANNAL_INDEX:
                detectInfo.minx = value;
                break;
            case AICORE_OUTPUT_YMIN_CHANNAL_INDEX:
                detectInfo.miny = value;
                break;
            case AICORE_OUTPUT_XMAX_CHANNAL_INDEX:
                detectInfo.width = value - detectInfo.minx;
                break;
            case AICORE_OUTPUT_YMAX_CHANNAL_INDEX:
                detectInfo.height = value - detectInfo.miny;
                break;
            case AICORE_OUTPUT_CONFIDENCE_CHANNAL_INDEX:
                detectInfo.confidence = value;
                break;
            case AICORE_OUTPUT_CLASS_ID_CHANNAL_INDEX:
                detectInfo.classId = (int32_t)value;
                break;
            default:
                break;
        }
    }
}

APP_ERROR FaceDetection::PostProcessAiCore(const std::vector<void *> &tensors) const
{
    std::vector<uint32_t> batchBoxSizes;
    uint32_t batchOffset;
    if (tensors.size() != AICORE_OUTPUT_LAYER_NUN) {
        LogFatal << "FaceDetection[" << instanceId_ << "]::PostProcessAiCore output layers number is invalid";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    for (size_t i = 0; i < batchSize_; i++) {
        batchOffset = i * outputSizes_[1] / batchSize_;
        uint32_t size = *(uint32_t *)((uint8_t *)tensors[1] + batchOffset);
        batchBoxSizes.push_back(size);
    }

    for (size_t i = 0; i < batchSize_; i++) {
        batchOffset = i * outputSizes_[0] / batchSize_;
        for (size_t j = 0; j < batchBoxSizes[i]; j++) {
            uint8_t *address = (uint8_t *)tensors[0] + batchOffset;
            DetectInfo detectInfo;
            GetResultAiCore(detectInfo, address, j, batchBoxSizes[i]);
            detectInfo.minx = std::min(std::max(detectInfo.minx, 0.f),
                std::max(float(inputArgQueue_[i]->imgOrigin.width) - 1.f, 0.f));
            detectInfo.miny = std::min(std::max(detectInfo.miny, 0.f),
                std::max(float(inputArgQueue_[i]->imgOrigin.height) - 1.f, 0.f));
            detectInfo.width = std::min(std::max(detectInfo.width, 0.f),
                std::max(float(inputArgQueue_[i]->imgOrigin.width) - detectInfo.minx - 1.f, 0.f));
            detectInfo.height = std::min(std::max(detectInfo.height, 0.f),
                std::max(float(inputArgQueue_[i]->imgOrigin.height) - detectInfo.miny - 1.f, 0.f));
            inputArgQueue_[i]->detectResult.push_back(detectInfo);
        }
        FaceSelection(inputArgQueue_[i]->detectResult); // select the top k faces from one frame
    }
    return APP_ERR_OK;
}

APP_ERROR FaceDetection::PostProcess(void) const
{
    std::vector<void *> tensors;
#ifdef ASCEND_ACL_OPEN_VESION
    for (auto tensor : outputBufs_) {
        tensors.push_back(tensor);
    }
#else
    for (uint32_t i = 0; i < outputBufs_.size(); i++) {
        APP_ERROR ret = aclrtMemcpy(outputBufsHost_[i].get(), outputSizes_[i], outputBufs_[i], outputSizes_[i],
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != APP_ERR_OK) {
            LogError << "FaceDetection[" << instanceId_ << "]::PostProcess aclrtMemcpy Failed.";
            return ret;
        }
        tensors.push_back(outputBufsHost_[i].get());
    }
#endif
    if (outputBufs_.size() == CPU_OUTPUT_LAYER_NUN) {
        // cpu post process
        return PostProcessCPU(tensors);
    } else {
        // aicore post process
        if (outputBufs_.size() == AICORE_OUTPUT_LAYER_NUN)
            return PostProcessAiCore(tensors);
    }
    // if outputBufs_ size is nether CPU's output size nor AICORE's output size, it will return error
    return APP_ERR_COMM_INVALID_PARAM;
}

APP_ERROR FaceDetection::Process(std::shared_ptr<void> inputData)
{
    LogDebug << "FaceDetection[" << instanceId_ << "]::Begin to process data.";
    std::shared_ptr<FrameAiInfo> frameAiInfo = std::static_pointer_cast<FrameAiInfo>(inputData);
    if (frameAiInfo.get() == nullptr) {
        return APP_ERR_COMM_FAILURE;
    }
    auto startTime = CostStatistic::GetStart();
    APP_ERROR ret = PreProcess(frameAiInfo);
    double costMs = CostStatistic::GetCostTime(startTime);
    if (ret != APP_ERR_OK) {
        LogError << "FaceDetection[" << instanceId_ << "]::Fail to prepare data.";
        return ret;
    }
    LogDebug << "FaceDetection[" << instanceId_ << "]::PreProcess " << costMs << "ms";

    frameAiInfo->detectImg.buf.deviceData.reset();
    inputArgQueue_.push_back(frameAiInfo);
    // get more data if batch not complete
    if (inputArgQueue_.size() < batchSize_) {
        return APP_ERR_OK;
    }
    // model Inference
    startTime = CostStatistic::GetStart();
    ret = modelInfer_->ModelInference(inputBufs_, inputSizes_, outputBufs_, outputSizes_);
    if (ret != APP_ERR_OK) {
        LogError << "FaceDetection[" << instanceId_ << "]::Fail to execute model.";
        return ret;
    }
    costMs = CostStatistic::GetCostTime(startTime);
    LogDebug << "FaceDetection[" << instanceId_ << "]::ModelInference " << costMs << "ms";

    startTime = CostStatistic::GetStart();
    PostProcess();
    costMs = CostStatistic::GetCostTime(startTime);
    LogDebug << "FaceDetection[" << instanceId_ << "]::PostProcess " << costMs << "ms";

    for (size_t i = 0; i < inputArgQueue_.size(); i++) {
        inputArgQueue_[i]->detectImg.buf.deviceData.reset();
        SendToNextModule(MT_FACE_RESIZE, inputArgQueue_[i], inputArgQueue_[i]->info.channelId);
    }
    inputArgQueue_.clear();
    return ret;
}

APP_ERROR FaceDetection::ParseConfig(ConfigParser &configParser)
{
    LogDebug << "FaceDetection[" << instanceId_ << "]::Begin to parse config file.";
    std::string itemCfgStr = moduleName_ + std::string(".channel_count");
    APP_ERROR ret = configParser.GetUnsignedIntValue(itemCfgStr, channelCount_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".width");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, width_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + std::string(".height");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, height_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + "." + pipelineName_ + std::string(".batch_size");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, batchSize_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = moduleName_ + "." + pipelineName_ + std::string(".model_path");
    ret = configParser.GetStringValue(itemCfgStr, modelPath_);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    itemCfgStr = std::string("max_face_num_per_frame");
    ret = configParser.GetUnsignedIntValue(itemCfgStr, maxFaceNumPerFrame_);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    LogDebug << "FaceDetection[" << instanceId_ << "]" << " batchSize_:" << batchSize_ <<
        " width_:" << width_ << " height_:" << height_ << " modelPath_:" << modelPath_.c_str() <<
        " maxFaceNumPerFrame_:" << maxFaceNumPerFrame_;

    return ret;
}
} // namespace ascendFaceRecognition
