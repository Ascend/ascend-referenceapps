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

#ifndef JPEG_READER_H
#define JPEG_READER_H

#include <atomic>
#include <mutex>
#include <thread>

#include "ModuleBase/ModuleBase.h"
#include "Statistic/Statistic.h"

namespace ascendFaceRecognition {
class JpegReader : public ModuleBase {
public:
    JpegReader();
    ~JpegReader();
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);
    uint32_t GetFormat() const;
    uint32_t GetInstanceId() const;
    uint32_t GetIsDisplay() const;
    uint32_t GetFrameNum() const;
    void SetFrameNum();
    std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> GetOutputQueVec() const;

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR GetPersonData(std::vector<std::string>::iterator it);
    APP_ERROR GetImageData(std::string &filePath, std::shared_ptr<StreamRawData> &output);
    APP_ERROR ScanFolder(std::string folderPath);
    APP_ERROR GetConfigureData(ConfigParser &configParser, std::string &imageFilePath, int &personCount);
    void ShowEndMessage() const;
    // stream info
    uint32_t format_ = 0;
    uint32_t isDisplay_ = 0;
    std::atomic<int> stop_ = { 0 };
    uint64_t frameNum_ = 0;
    std::string jpegFolderPath_ = "";
    std::string regConfigPath_ = "";
    std::vector<std::string> fileNameSet_ = {};
    Statistic jpegReaderStatic_ = {};
};
} // namespace ascendFaceRecognition

#endif
