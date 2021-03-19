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
#include <chrono>
#include <thread>

namespace {
// resnet class num
const int CLASS_TYPE_NUM = 1000;
// path of label file
const std::string LABEL_PATH = "./data/models/resnet/imagenet1000_clsidx_to_labels.txt";
};

/*
 * @description Implementation of constructor for class AclProcess with parameter list
 * @attention context is passed in as a parameter after being created in ResourceManager::InitResource
 */
AclProcess::AclProcess(int deviceId, ModelInfo modelInfo, std::string opModelPath, aclrtContext context)
    : deviceId_(deviceId), modelInfo_(modelInfo), opModelPath_(opModelPath), context_(context),
      stream_(nullptr), modelProcess_(nullptr), dvppCommon_(nullptr), argMaxOp_(nullptr),
      castOp_(nullptr), labelMap_(std::map<int, std::string>())
{
}

/*
 * @description Release all the resource
 * @attention context will be released in ResourceManager::Release
 */
void AclProcess::Release()
{
    APP_ERROR ret;
    // Synchronize stream and release Dvpp channel
    dvppCommon_->DeInit();
    // Release stream
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy the stream, ret = " << ret << ".";
        }
        stream_ = nullptr;
    }
    // Destroy resources of modelProcess_
    modelProcess_->DeInit();

    // Release Dvpp buffer
    dvppCommon_->ReleaseDvppBuffer();
}

/*
 * @description Initialize the modules used by this sample
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::InitModule()
{
    // Create Dvpp common object
    if (dvppCommon_ == nullptr) {
        dvppCommon_.reset(new DvppCommon(stream_));
        APP_ERROR retDvppCommon = dvppCommon_->Init();
        if (retDvppCommon != APP_ERR_OK) {
            LogError << "Failed to initialize dvppCommon, ret = " << retDvppCommon;
            return retDvppCommon;
        }
    }
    // Create model inference object
    if (modelProcess_ == nullptr) {
        modelProcess_.reset(new ModelProcess(deviceId_, modelInfo_.modelName));
    }
    // Initialize ModelProcess module
    APP_ERROR ret = modelProcess_->Init(modelInfo_.modelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to initialize the model process module, ret = " << ret << ".";
        return ret;
    }
    LogInfo << "Initialized the model process module successfully.";
    // Create Cast operator
    if (castOp_ == nullptr) {
        castOp_.reset(new SingleOpProcess(stream_));
    }
    LogInfo << "Initialized the cast operator successfully.";
    // Create ArgMax operator
    if (argMaxOp_ == nullptr) {
        argMaxOp_.reset(new SingleOpProcess(stream_));
    }
    LogInfo << "Initialized the argMax operator successfully.";
    ret = LoadLabels(LABEL_PATH); // Load labels from file
    if (ret != APP_ERR_OK) {
        LogError << "Failed to load labels, ret = " << ret << ".";
        return APP_ERR_COMM_READ_FAIL;
    }
    LogInfo << "Loaded label successfully.";
    return APP_ERR_OK;
}

/*
 * @description Create resource for this sample
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::InitResource()
{
    APP_ERROR ret = aclrtSetCurrentContext(context_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set the acl context, ret = " << ret << ".";
        return ret;
    }
    LogInfo << "Created the acl context successfully.";
    ret = aclrtCreateStream(&stream_); // Create stream for application
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create the acl stream, ret = " << ret << ".";
        return ret;
    }
    LogInfo << "Created the acl stream successfully.";
    // Initialize dvpp module
    if (InitModule() != APP_ERR_OK) {
        return APP_ERR_COMM_INIT_FAIL;
    }
    // Initialize Cast operator module
    if (InitOpCastResource() != APP_ERR_OK) {
        return APP_ERR_COMM_INIT_FAIL;
    }
    // Initialize ArgMax operator module
    if (InitOpArgMaxResource() != APP_ERR_OK) {
        return APP_ERR_COMM_INIT_FAIL;
    }
    return APP_ERR_OK;
}

/*
 * @description Initialize the resource for Cast operator
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::InitOpCastResource() const
{
    std::string opTypeName("Cast");
    castOp_->SetTypeName(opTypeName); // Set type name of operator
    int inputTensorNum = 1;
    std::shared_ptr<aclopAttr> opAttr(aclopCreateAttr(), aclopDestroyAttr); // Create attribute for operator
    OpAttr attr;
    attr.name = std::string("dst_type");
    attr.type = INT_TYPE;
    attr.numInt = 1;
    APP_ERROR ret = castOp_->SetOpAttr(opAttr, attr); // Set attribute for operator
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set attribute of the cast operator, ret = " << ret << ".";
        return ret;
    }
    // Operator input tensor info
    std::vector<Tensor> tensors = { {ACL_FLOAT, 1, {CLASS_TYPE_NUM}, ACL_FORMAT_ND}, };
    castOp_->SetInputTensorNum(inputTensorNum); // Set input tensor number of operator
    ret = castOp_->SetInputTensor(tensors); // Set input tensor of operator
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set input of the cast operator, ret = " << ret << ".";
        return ret;
    }
    int outputTensorNum = 1;
    // Operator output tensor info
    std::vector<Tensor> outTensors = { {ACL_FLOAT16, 1, {CLASS_TYPE_NUM}, ACL_FORMAT_ND}, };
    castOp_->SetOutputTensorNum(outputTensorNum); // Set output tensor number of operator
    ret = castOp_->SetOutputTensor(outTensors); // Set output tensor of operator
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set output of the cast operator, ret = " << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/*
 * @description Initialize the resource for ArgMax operator
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::InitOpArgMaxResource() const
{
    int inputTensorNum = 1;
    std::string opTypeName = "ArgMaxD";
    argMaxOp_->SetTypeName(opTypeName); // Set type name of operator
    std::shared_ptr<aclopAttr> opAttr(aclopCreateAttr(), aclopDestroyAttr); // Create attribute for operator
    OpAttr attr;
    attr.name = std::string("dimension");
    attr.type = INT_TYPE;
    attr.numInt = 0;
    APP_ERROR ret = argMaxOp_->SetOpAttr(opAttr, attr); // Set attribute for operator
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set attribute of the argMax operator, ret = " << ret << ".";
        return ret;
    }
    // Operator input tensor info
    std::vector<Tensor> tensors = { {ACL_FLOAT16, 1, {1000}, ACL_FORMAT_ND}, };
    argMaxOp_->SetInputTensorNum(inputTensorNum); // Set input tensor number
    ret = argMaxOp_->SetInputTensor(tensors); // Set input tensor
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set input of the argMax operator, ret = " << ret << ".";
        return ret;
    }
    int argMaxOutputTensorNum = 1;
    // Operator output tensor info
    std::vector<Tensor> outTensors = { {ACL_INT32, 1, {1}, ACL_FORMAT_ND} };
    argMaxOp_->SetOutputTensorNum(argMaxOutputTensorNum); // Set output tensor number
    ret = argMaxOp_->SetOutputTensor(outTensors); // Set output tensor
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set output of the argMax operator, ret = " << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/*
 * @description Load the labels from the file
 * @param labelPath classification label path
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::LoadLabels(const std::string& labelPath)
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
    for (int i = 0; i < CLASS_TYPE_NUM; ++i) {
        std::getline(in, s);
        labelMap_.insert(std::pair<int, std::string>(i, s));
    }
    in.close();
    return APP_ERR_OK;
}

/*
 * @description Write result index and class name into file
 * @param index result index of classification label
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::WriteResult(int index)
{
    std::string resultPathName = "result";
    // Create result directory when it does not exist
    if (access(resultPathName.c_str(), 0) != 0) {
#ifdef _WIN32
        int ret = _mkdir(resultPathName.c_str());
#else
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR); // for linux
#endif
        if (ret != 0) {
            LogError << "Failed to create the result directory: " << resultPathName << ", ret = " << ret;
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
    LogInfo << "inference output index: " << index;
    tfile << "inference output index: " <<  index << std::endl; // Write label index into file
    LogInfo << "classname: " << labelMap_[index];
    tfile << "classname: " <<  labelMap_[index]  << std::endl; // Write label name into file
    tfile.close();
    return APP_ERR_OK;
}

/*
 * @description Preprocess the input image
 * @param imageFile input image path
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::Preprocess(const std::string& imageFile) const
{
    RawData imageInfo;
    APP_ERROR ret = ReadFile(imageFile, imageInfo); // Read image data from input image file
    if (ret != APP_ERR_OK) {
        LogError << "Failed to read file, ret = " << ret << ".";
        return ret;
    }
    // Run process of jpegD
    ret = dvppCommon_->CombineJpegdProcess(imageInfo, PIXEL_FORMAT_YUV_SEMIPLANAR_420, true);
    if (ret !=  APP_ERR_OK) {
        LogError  << "Failed to execute image decoded of preprocess module, ret = " << ret << ".";
        return ret;
    }
    // Get output of decode jpeg image
    std::shared_ptr<DvppDataInfo> decodeOutData = dvppCommon_->GetDecodedImage();
    // Run resize application function
    DvppDataInfo resizeOutData;
    resizeOutData.height = modelInfo_.modelHeight;
    resizeOutData.width = modelInfo_.modelWidth;
    resizeOutData.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    ret = dvppCommon_->CombineResizeProcess(*decodeOutData, resizeOutData, true, VPC_PT_FILL);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to execute image resized of preprocess module, ret = " << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/*
 * @description Inference of model
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::ModelInfer(std::vector<RawData> &modelOutput)
{
    // Get output of resize module
    std::shared_ptr<DvppDataInfo> resizeOutData = dvppCommon_->GetResizedImage();
    std::vector<void *> inputBuffers { resizeOutData->data };
    std::vector<size_t> inputSizes { resizeOutData->dataSize };

    std::vector<void *> outputBuffers;
    std::vector<size_t> outputSizes;
    // Get model description
    aclmdlDesc *modelDesc = modelProcess_->GetModelDesc();
    size_t outputSize = aclmdlGetNumOutputs(modelDesc);
    for (size_t i = 0; i < outputSize; i++) {
        // Get output size according to index
        size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
        // Malloc buffer for output data
        void *outputBuffer = nullptr;
        APP_ERROR ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to malloc buffer, ret = " << ret << ".";
            return ret;
        }
        outputBuffers.push_back(outputBuffer);
        outputSizes.push_back(bufferSize);
    }
    // Execute classification model
    APP_ERROR ret = modelProcess_->ModelInference(inputBuffers, inputSizes, outputBuffers, outputSizes);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to execute the classification model, ret = " << ret << ".";
        return ret;
    }

    for (size_t i = 0; i < outputBuffers.size(); i++) {
        RawData rawDevData = RawData();
        rawDevData.data.reset(outputBuffers[i], aclrtFree);
        rawDevData.lenOfByte = outputSizes[i];
        modelOutput.push_back(std::move(rawDevData));
    }

    return APP_ERR_OK;
}

/*
 * @description Inference of Cast operator
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::CastOpInfer(std::vector<RawData> modelOutput)
{
    // Get output of resnet model inference
    if (modelOutput.empty()) {
        LogError << "Failed to get output data of classification model.";
        return APP_ERR_INFER_GET_OUTPUT_FAIL;
    }
    // Construct input data for Cast operator
    std::vector<std::shared_ptr<void>> inputDataBuf { modelOutput[0].data };
    std::vector<size_t> inputBufSize { modelOutput[0].lenOfByte };
    castOp_->SetInputDataBuffer(inputDataBuf, inputBufSize); // Set input data of cast operator
    // Execute cast operator
    APP_ERROR ret = castOp_->RunSingleOp(true);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to execute the cast Operator, ret = " << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/*
 * @description Inference of ArgMax operator
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::ArgMaxOpInfer()
{
    // Get output of Cast operator
    std::vector<std::shared_ptr<void>> castOpOutput = castOp_->GetOutputData();
    std::vector<size_t> outDataSize = castOp_->GetOutputDataSize();
    std::shared_ptr<void> castOpOutputDataPtr = castOpOutput[0];
    // Construct input data for ArgMax operator
    std::vector<std::shared_ptr<void>> inputDataBuf { castOpOutputDataPtr };
    std::vector<size_t> inputBufSize { outDataSize };
    argMaxOp_->SetInputDataBuffer(inputDataBuf, inputBufSize); // Set input data for ArgMax operator
    // Execute argMax operator
    APP_ERROR ret = argMaxOp_->RunSingleOp(true);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to execute the argMax Operator, ret = " << ret << ".";
        return ret;
    }
    // Synchronize the stream
    ret = aclrtSynchronizeStream(stream_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to synchronize the stream, ret = " << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/*
 * @description Process classification
 *
 * @par Function
 * 1.Dvpp module preprocess
 * 2.Execute classification model
 * 3.Execute single operator
 * 4.Write result
 *
 * @param imageFile input file path
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::Process(const std::string& imageFile)
{
    // The following code provides a method to calculate the time cost of preprocess module
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = Preprocess(imageFile);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    // Infer Classification
    std::vector<RawData> modelOutput;
    ret = ModelInfer(modelOutput);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    // If classification model does not include the agrMax operator,
    // you need the Cast and AgrMax operator to process the output of the model
    ret = CastOpInfer(modelOutput); // Cast operator inference
    if (ret != APP_ERR_OK) {
        return ret;
    }
    // ArgMax operator inference
    ret = ArgMaxOpInfer();
    if (ret != APP_ERR_OK) {
        return ret;
    }
    // Post process the inference result
    ret = PostProcess();
    if (ret != APP_ERR_OK) {
        return ret;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    LogInfo << "[Process Delay] cost: " << costMs << "ms.";
    return APP_ERR_OK;
}

/*
 * @description PostProcess the result
 * @return APP_ERROR error code
 */
APP_ERROR AclProcess::PostProcess()
{
    // Get output of ArgMax operator
    std::vector<std::shared_ptr<void>> argMaxOpOutput =  argMaxOp_->GetOutputData();
    std::vector<size_t> outDataSize = argMaxOp_->GetOutputDataSize();
    void* argMaxOpOutputData = argMaxOpOutput[0].get();
    // Malloc host memory for the inference output
    void *resHostBuf = nullptr;
    APP_ERROR ret = aclrtMallocHost(&resHostBuf, outDataSize[0]);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to print the result, malloc host failed, ret = " << ret << ".";
        return ret;
    }
    // If you use aclrtMallocHost to allocate memory, you need to release the memory through aclrtFreeHost
    std::shared_ptr<void> outBuf(resHostBuf, aclrtFreeHost);
    // Memcpy the output data from device to host
    ret = aclrtMemcpy(outBuf.get(), outDataSize[0], argMaxOpOutputData,
        outDataSize[0], ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to print result, memcpy device to host failed, ret = " << ret << ".";
        return ret;
    }
    // Get the index of output label
    auto *index = static_cast<int32_t *>(resHostBuf);
    ret = WriteResult(*index); // Write result into result.txt
    if (ret != APP_ERR_OK) {
        LogError << "Failed to write result file, ret = " << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}
