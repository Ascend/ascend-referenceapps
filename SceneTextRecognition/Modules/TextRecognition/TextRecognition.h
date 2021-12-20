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

#ifndef INC_TEXT_RECOGNITION_H
#define INC_TEXT_RECOGNITION_H

#include <chrono>

#include "DvppCommon/DvppCommon.h"
#include "ConfigParser/ConfigParser.h"
#include "Framework/ModuleManager/ModuleManager.h"
#include "Statistic/Statistic.h"
#include "Framework/ModelProcess/ModelProcess.h"
#include "Common/CommonType.h"

enum ModelType {
    CHINESE_OCR = 0,
    CRNN = 1
};

class TextRecognition : public ascendBaseModule::ModuleBase {
public:
    TextRecognition();
    ~TextRecognition();
    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfigCommon(ConfigParser &configParser);
    APP_ERROR ParseConfigModel(ConfigParser &configParser);
    APP_ERROR Preprocess(std::shared_ptr<SendInfo> sendData);
    void SaveResizedImage(std::shared_ptr<SendInfo> sendData);
    APP_ERROR PrepareModelInputBuffer(std::shared_ptr<void> &ptrBufferManager, size_t singleBatchSize,
                                      uint32_t itemNum);
    APP_ERROR PrepareModelOutputBuffer(std::vector<void *> &outputBuffers, std::vector<size_t> &outputSizes);
    void PostProcessChineseOCR(const std::vector<std::shared_ptr<void>> featLayerData,
                               const std::vector<size_t> outputSizes, size_t imageIndex,
                               uint32_t charactersNum, uint32_t& cnt);
    void RecognizeOutputChineseOCR(const std::vector<std::shared_ptr<void>> featLayerData,
                                   const std::vector<size_t> outputSizes, uint32_t itemNum);
    void RecognizeOutputCRNN(const std::vector<std::shared_ptr<void>> featLayerData,
                             const std::vector<size_t> outputSizes, uint32_t itemNum);
    APP_ERROR RecognizePostProcess(const std::vector<void *> &outputBuffers, const std::vector<size_t> &outputSizes,
                                   uint32_t itemNum);
    APP_ERROR LoadKeysUTF8File(const std::string& fileName, std::vector<std::string>& keysVector);
    void CreateThread();
    void WatchThread();
    APP_ERROR BatchModelInference(uint32_t itemNum);
    APP_ERROR CheckModelInputInfo(void);
    APP_ERROR ModelProcessInit(void);

    uint32_t deviceId_ = 0;
    std::string modelName_ = "";
    std::string modelPath_ = "";
    uint32_t modelType_ = 0;
    uint32_t modelWidth_ = 0;
    uint32_t modelHeight_ = 0;
    uint32_t batchSize_ = 1;
    double timeoutInterval_ = 0.0;
    std::chrono::high_resolution_clock::time_point startTime_ = {};
    std::unique_ptr<ModelProcess> modelProcess_ = nullptr;
    std::unique_ptr<DvppCommon> dvppObjPtr_ = nullptr;
    aclrtStream dvppStream_ = nullptr;
    uint32_t debugMode_ = false;
    Statistic textRecognitionStatic_ = {};
    std::vector<std::string> keysVec_ = {};
    std::string keysFilePath_ = "";
    int keysNum_ = 0;
    std::unique_ptr<std::thread> thread_ = nullptr;
    std::mutex mtx_ = {};
    std::vector<std::shared_ptr<SendInfo>> sendInfoVec_ = {};
};

MODULE_REGIST(TextRecognition)
#endif
