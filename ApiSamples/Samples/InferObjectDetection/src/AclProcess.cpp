/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "AclProcess.h"
#ifdef _WIN32
    #define access _access
#else
    #include <unistd.h>
#endif

/*
 * @description: Implementation of constructor for class AclProcess with parameter list
 * @attention: context is passed in as a parameter after being created in ResourceManager::InitResource
 */
AclProcess::AclProcess(int deviceId, ModelInfo modelInfo, std::string opModelPath, aclrtContext context)
    : deviceId_(deviceId), modelInfo_(modelInfo), context_(context), stream_(nullptr), modelProcess_(nullptr),
      dvppCommon_(nullptr), labelMap_(std::map<int, std::string>())
{
}

/*
 * @description: Release stream resource which is created in AclProcess::InitResource and other shared pointer
 * @attention: context will be released in ResourceManager::Release
 */
void AclProcess::Release()
{
    APP_ERROR ret;
    // Synchronize stream and release Dvpp channel
    dvppCommon_->DeInit();
    // Release stream
    if (stream_ != nullptr) {
        ret = aclrtSynchronizeStream(stream_);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to synchronize stream before destroy, ret = " << ret;
        }
        ret = aclrtDestroyStream(stream_);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy stream, ret = " << ret;
        }
        stream_ = nullptr;
    }
    LogInfo << "Destroy stream successfully";
    // Destroy resources of modelProcess_
    modelProcess_->DeInit();
    // Release Dvpp buffer
    dvppCommon_->ReleaseDvppBuffer();
}

/*
 * @description Load the labels from the file
 * @param labelPath coco label path
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR AclProcess::LoadLabels(const std::string &labelPath)
{
    std::ifstream in;
    in.open(labelPath, std::ios_base::in); // Open label file
    std::string s;
    // Check label file validity
    if (in.fail()) {
        LogError << "Failed to open label file: " << labelPath;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // Skip the first line(source information) of file
    std::getline(in, s);
    // Construct label map
    for (int i = 0; i < CLASS_NUM; ++i) {
        std::getline(in, s);
        labelMap_.insert(std::pair<int, std::string>(i, s));
    }
    in.close();
    return APP_ERR_OK;
}

/*
 * @description: Initialize DVPP and model related resources
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR AclProcess::InitModule()
{
    // Create Dvpp JpegD object
    if (dvppCommon_ == nullptr) {
        dvppCommon_.reset(new DvppCommon(stream_));
        APP_ERROR retDvppCommon = dvppCommon_->Init();
        if (retDvppCommon != APP_ERR_OK) {
            LogError << "Failed to initialize dvppCommon, ret = " << retDvppCommon;
            return retDvppCommon;
        }
    }
    LogInfo << "Initialize dvppCommon_ successfully";
    // Create model inference object
    if (modelProcess_ == nullptr) {
        modelProcess_.reset(new ModelProcess(deviceId_, modelInfo_.modelName));
    }

    APP_ERROR ret;
#ifdef USE_DCMI_INTERFACE
    // Initialize ModelProcess module
    ret = modelProcess_->Init(modelInfo_.modelPath, modelInfo_.isEncrypted, modelInfo_.cardId, deviceId_);
#else
    // Initialize ModelProcess module
    ret = modelProcess_->Init(modelInfo_.modelPath);
#endif
    if (ret != APP_ERR_OK) {
        LogError << "Failed to initialize ModelProcess_, ret = " << ret << ".";
        return ret;
    }
    LogInfo << "Initialize ModelProcess_ successfully";
    ret = LoadLabels(LABEL_PATH); // Load labels from file
    if (ret != APP_ERR_OK) {
        LogError << "Failed to load labels, ret = " << ret << ".";
        return APP_ERR_COMM_READ_FAIL;
    }
    LogInfo << "Loaded label successfully.";
    return APP_ERR_OK;
}

/*
 * @description: Initialize all resources, including stream, DVPP and model related resources
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR AclProcess::InitResource()
{
    APP_ERROR ret = aclrtSetCurrentContext(context_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set current context, ret = " << ret;
        return ret;
    }
    LogInfo << "Create context successfully";
    ret = aclrtCreateStream(&stream_); // Create stream for application
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create stream, ret = " << ret;
        return ret;
    }
    LogInfo << "Create stream successfully";
    // Initialize dvpp module
    if (InitModule() != APP_ERR_OK) {
        return APP_ERR_COMM_INIT_FAIL;
    }

    return APP_ERR_OK;
}

/*
 * @description: Image preprocess, decoded and resize the input images to meet the input requirements of the model
 * @param imagefile  absolute path of the input image
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR AclProcess::Preprocess(const std::string& imageFile)
{
    // Calcute the time cost of DVPP preprocess
    RawData imageInfo;
    APP_ERROR ret = ReadFile(imageFile, imageInfo); // Read image data from input image file
    if (ret != APP_ERR_OK) {
        LogError << "Failed to read image file, ret = " << ret;
        return ret;
    }
    ret = dvppCommon_->CombineJpegdProcess(imageInfo, PIXEL_FORMAT_YUV_SEMIPLANAR_420, true);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to decode image, ret = " << ret;
        return ret;
    }
    std::shared_ptr<DvppDataInfo> decodeOutData = dvppCommon_->GetDecodedImage();
    DvppDataInfo resizeOutputData;
    resizeOutputData.width = modelInfo_.modelWidth;
    resizeOutputData.height = modelInfo_.modelHeight;
    resizeOutputData.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    // Resize the width and height with the same ratio
    ret = dvppCommon_->CombineResizeProcess(*decodeOutData, resizeOutputData, true, VPC_PT_FIT);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to resize image, ret = " << ret;
        return ret;
    }
    return APP_ERR_OK;
}

/*
 * @description: Get the result data from dvpp resize and do the model inference.
 *               According to the difference between Caffe model and TF model,
 *               Caffe model need one more parameter imageInfo as input
 * @param outputBuffers  Model infer result data in device
 * @param outputSizes  Model infer data size in device
 * @param modelType  Infer model type, 0 is Caffe, 1 is TF
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR AclProcess::ModelInfer(std::vector<void *> &outputBuffers, std::vector<size_t> &outputSizes, int modelType)
{
    // Get output of resize module
    std::shared_ptr<DvppDataInfo> resizeOutData = dvppCommon_->GetResizedImage();
    std::vector<void *> inputBuffers { resizeOutData->data };
    std::vector<size_t> inputSizes { resizeOutData->dataSize };
    void* yoloImgInfo = nullptr;
    if (modelType == YOLOV3_CAFFE) {
        // We need to get width and weight of model and original picture
        std::shared_ptr<DvppDataInfo> inputData = dvppCommon_->GetInputImage();
        float imgInfo[IMAGE_INFO_ARRAY_SIZE] = {0};
        imgInfo[MODEL_HEIGHT_INDEX] = modelInfo_.modelHeight;
        imgInfo[MODEL_WIDTH_INDEX] = modelInfo_.modelWidth;
        imgInfo[IMAGE_HEIGHT_INDEX] = inputData->height; // It should be the height of the original picture
        imgInfo[IMAGE_WIDTH_INDEX] = inputData->width; // It should be the width of the original picture
        uint32_t imgInfoInputSize =  sizeof(float) * IMAGE_INFO_ARRAY_SIZE;
        APP_ERROR ret = aclrtMalloc(&yoloImgInfo, imgInfoInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            LogError << "Failed to malloc normal memory for input buffer on device, ret = " << ret;
            return ret;
        }
        ret = aclrtMemcpy((uint8_t *)yoloImgInfo, imgInfoInputSize, imgInfo, imgInfoInputSize,
            ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            LogError << "Failed to copy input buffer of model from host to on device, ret = " << ret;
            aclrtFree(yoloImgInfo);
            return ret;
        }
        inputBuffers.push_back(yoloImgInfo);
        inputSizes.push_back(sizeof(float) * IMAGE_INFO_ARRAY_SIZE);
    }
    APP_ERROR ret = modelProcess_->ModelInference(inputBuffers, inputSizes, outputBuffers, outputSizes);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to execute ModelInference, ret = " << ret;
    }
    if (yoloImgInfo != nullptr) {
        aclrtFree(yoloImgInfo);
    }
    return ret;
}

/*
 * @description: Write inference result to result file
 * @param objInfo  The array of information about all detected objects
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR AclProcess::WriteResult(const std::vector<ObjDetectInfo> objInfos)
{
    std::string resultPathName = "result";
    uint32_t objNum = objInfos.size();
    // Create result directory when it does not exist
    if (access(resultPathName.c_str(), 0) != 0) {
#ifdef _WIN32
        int ret = _mkdir(resultPathName.c_str());
#else
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR); // for linux
#endif
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName << ", ret = " << ret;
            return ret;
        }
    }
    // Result file name use the time stamp as a suffix
    std::string timeString;
    GetCurTimeString(timeString);

    // Create result file under result directory
    resultPathName = resultPathName + "/result_" + timeString + ".txt";
#ifndef _WIN32
    SetFileDefaultUmask();
#endif
    std::ofstream tfile(resultPathName);
    // Check result file validity
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << resultPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    tfile << "Object detected number is " << objNum << std::endl;
    // Write inference result into file
    for (uint32_t i = 0; i < objNum; i++) {
        tfile << "#Obj" << i << ", " << "box(" << objInfos[i].leftTopX << ", " << objInfos[i].leftTopY << ", "
              << objInfos[i].rightBotX << ", " << objInfos[i].rightBotY << ") "
              << " confidence: " << objInfos[i].confidence << " lable: " << labelMap_[objInfos[i].classId] << std::endl;
    }
    tfile.close();
    return APP_ERR_OK;
}

/*
 * @description: Get Caffe model output and copy to host
 * @param outputBuffers  Caffe model infer result(bbox, classId, confidence) in device
 * @param outputSizes  Caffe model infer data size in device
 * @param objInfos  Vector to save infer result
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR AclProcess::GetModelOutputCaffe(std::vector<void *> outputBuffers, std::vector<size_t> outputSizes,
    std::vector<ObjDetectInfo> &objInfos)
{
    std::vector<std::shared_ptr<void>> hostPtr;
    for (size_t j = 0; j < outputSizes.size(); j++) {
        void *hostPtrBuffer = nullptr;
        APP_ERROR ret = (APP_ERROR)aclrtMallocHost(&hostPtrBuffer, outputSizes[j]);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to malloc output buffer of model on host, ret = " << ret;
            return ret;
        }
        std::shared_ptr<void> hostPtrBufferManager(hostPtrBuffer, aclrtFreeHost);
        ret = (APP_ERROR)aclrtMemcpy(hostPtrBuffer, outputSizes[j], outputBuffers[j],
                                     outputSizes[j], ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to copy output buffer of model from device to host, ret = " << ret;
            return ret;
        }
        hostPtr.push_back(hostPtrBufferManager);
    }
    uint32_t objNum = ((uint32_t *)(hostPtr[1].get()))[0];
    for (uint32_t k = 0; k < objNum; k++) {
        int pos = 0;
        ObjDetectInfo objInfo = {};
        objInfo.leftTopX = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfo.leftTopY = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfo.rightBotX = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfo.rightBotY = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfo.confidence = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfo.classId = ((float *)hostPtr[0].get())[objNum * (pos++) + k];
        objInfos.push_back(objInfo);
    }
    return APP_ERR_OK;
}

/*
 * @description: Get Tensorflow model feature output, copy to host and computer the final result by Yolo layer
 * @param outputBuffers  Caffe model infer feature result in device
 * @param outputSizes  Caffe model infer data size in device
 * @param objInfos  Vector to save final result
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR AclProcess::GetModelOutputTF(std::vector<void *> outputBuffers, std::vector<size_t> outputSizes,
    std::vector<ObjDetectInfo> &objInfos)
{
    int outputLen = outputSizes.size();
    std::vector<std::shared_ptr<void>> singleResult;
    for (int j = 0; j < outputLen; j++) {
        void *hostPtrBuffer = nullptr;
        APP_ERROR ret = (APP_ERROR)aclrtMallocHost(&hostPtrBuffer, outputSizes[j]);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to malloc host memory for hostPtr";
            return ret;
        }
        std::shared_ptr<void> hostPtrBufferManager(hostPtrBuffer, aclrtFreeHost);
        APP_ERROR retmem = (APP_ERROR)aclrtMemcpy(hostPtrBuffer, outputSizes[j], outputBuffers[j], outputSizes[j],
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (retmem != APP_ERR_OK) {
            LogError << "Failed to memcpy device to host";
            return retmem;
        }
        singleResult.push_back(hostPtrBufferManager);
    }
    std::shared_ptr<DvppDataInfo> inputData = dvppCommon_->GetInputImage();
    YoloImageInfo imgInfo;
    imgInfo.modelWidth = modelInfo_.modelWidth;
    imgInfo.modelHeight = modelInfo_.modelHeight;
    imgInfo.imgWidth = inputData->width;
    imgInfo.imgHeight = inputData->height;
    Yolov3DetectionOutput(singleResult, objInfos, imgInfo);
    return APP_ERR_OK;
}

/*
 * @description: Post-processing of YoloV3 model output
 * @param outputBuffers  Caffe model infer feature result in device
 * @param outputSizes  Caffe model infer data size in device
 * @param modelType  Infer model type, 0 is Caffe, 1 is TF
 * @return: APP_ERR_OK success
 * @return: Other values failure
 * @attention: the analysis of the output memory needs to refer to "operator Specifications" released by HiSilicon
 */
APP_ERROR AclProcess::YoloV3PostProcess(std::vector<void *> outputBuffers, std::vector<size_t> outputSizes,
    int modelType)
{
    const size_t outputLen = outputSizes.size();
    if (outputLen <= 0) {
        LogError << "Failed to get model output data";
        return APP_ERR_INFER_GET_OUTPUT_FAIL;
    }
    LogInfo << "The number of output buffers of yolov3 model is " << outputLen;

    std::vector<ObjDetectInfo> objInfos;
    APP_ERROR ret;
    if (modelType == YOLOV3_CAFFE) {
        ret = GetModelOutputCaffe(outputBuffers, outputSizes, objInfos);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to get Caffe model output, ret = " << ret;
            return ret;
        }
    } else {
        ret = GetModelOutputTF(outputBuffers, outputSizes, objInfos);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to get TF model output, ret = " << ret;
            return ret;
        }
    }
    for (uint32_t k = 0; k < objInfos.size(); k++) {
        LogInfo << "#Obj" << k << ", " << "box(" << objInfos[k].leftTopX << ", " << objInfos[k].leftTopY << ", "
                << objInfos[k].rightBotX << ", " << objInfos[k].rightBotY << ")  "
                << " confidence: " << objInfos[k].confidence << " label: " << labelMap_[objInfos[k].classId];
    }
    ret = WriteResult(objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to write result file, ret = " << ret;
        return ret;
    }
    return APP_ERR_OK;
}

/*
 * @description: Entry function that connects pre-processing, inference and post-processing processes in series
 * @param imagefile  absolute path of the input image
 * @param modelType  Infer model type, 0 is Caffe, 1 is TF
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR AclProcess::Process(const std::string& imageFile, int modelType)
{
    APP_ERROR ret = Preprocess(imageFile);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    std::vector<void *> outputBuffers;
    std::vector<size_t> outputSizes;
    // Get model description
    aclmdlDesc *modelDesc = modelProcess_->GetModelDesc();
    size_t outputSize = aclmdlGetNumOutputs(modelDesc);
    for (size_t i = 0; i < outputSize; i++) {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
        void *outputBuffer = nullptr;
        APP_ERROR ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to malloc buffer, ret = " << ret << ".";
            return ret;
        }
        outputBuffers.push_back(outputBuffer);
        outputSizes.push_back(bufferSize);
    }
    ret = ModelInfer(outputBuffers, outputSizes, modelType);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    ret = YoloV3PostProcess(outputBuffers, outputSizes, modelType); // post inference
    if (ret != APP_ERR_OK) {
        return ret;
    }
    for (size_t i = 0; i < outputBuffers.size(); i++) {
        if (outputBuffers[i] != nullptr) {
            aclrtFree(outputBuffers[i]);
            outputBuffers[i] = nullptr;
        }
    }
    return APP_ERR_OK;
}
