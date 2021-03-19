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

#ifndef ACLMANAGER_H
#define ACLMANAGER_H

#include <map>
#include <iostream>
#include "acl/acl.h"
#include "CommonDataType/CommonDataType.h"
#include "Log/Log.h"
#include "ErrorCode/ErrorCode.h"
#include "FileManager/FileManager.h"
#include "ModelProcess/ModelProcess.h"
#include "DvppCommon/DvppCommon.h"
#include "ResourceManager/ResourceManager.h"
#include "Yolov3Post.h"

const int YOLOV3_CAFFE = 0;
const int YOLOV3_TF = 1;
// path of label file
const std::string LABEL_PATH = "./data/models/yolov3/coco.names";

// Definition of input image info array index
enum {
    MODEL_HEIGHT_INDEX = 0,
    MODEL_WIDTH_INDEX,
    IMAGE_HEIGHT_INDEX,
    IMAGE_WIDTH_INDEX,
    IMAGE_INFO_ARRAY_SIZE // size of input image info array in yolo
};

class AclProcess {
public:
    AclProcess(int deviceId, ModelInfo modelInfo, std::string opModelPath, aclrtContext context);
    ~AclProcess() {};
    // Release all the resource
    void Release();
    // Create resource for this sample
    APP_ERROR InitResource();
    // Main process of image detection
    APP_ERROR Process(const std::string& imageFile, int modelType);

private:
    // Load the labels from the file
    APP_ERROR LoadLabels(const std::string &labelPath);
    // Initialize the modules used by this sample
    APP_ERROR InitModule();
    // Preprocess the input image
    APP_ERROR Preprocess(const std::string& imageFile);
    // Inference of model
    APP_ERROR ModelInfer(std::vector<void *> &outputBuffers, std::vector<size_t> &outputSizes, int modelType);
    // Transpose NHWC to NCHW
    void TransPose(std::shared_ptr<void> &dst, std::shared_ptr<void> &src, int len);
    // Get Caffe model inference result data
    APP_ERROR GetModelOutputCaffe(std::vector<void *> outputBuffers, std::vector<size_t> outputSizes,
        std::vector<ObjDetectInfo> &objInfos);
    // Get Tensorflow model inference result data
    APP_ERROR GetModelOutputTF(std::vector<void *> outputBuffers, std::vector<size_t> outputSizes,
        std::vector<ObjDetectInfo> &objInfos);
    // Post-process for inference output data
    APP_ERROR YoloV3PostProcess(std::vector<void *> outputBuffers, std::vector<size_t> outputSizes, int modelType);
    // Write result to file
    APP_ERROR WriteResult(const std::vector<ObjDetectInfo> objInfo);

    int32_t deviceId_; // device id used
    ModelInfo modelInfo_; // info of input model
    aclrtContext context_;
    aclrtStream stream_;
    std::unique_ptr<ModelProcess> modelProcess_; // model inference object
    std::unique_ptr<DvppCommon> dvppCommon_; // dvpp object
    std::map<int, std::string> labelMap_; // labels info
};

#endif
