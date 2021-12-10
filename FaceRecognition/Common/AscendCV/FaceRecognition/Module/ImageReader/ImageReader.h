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

#ifndef INC_IMAGE_READER_H
#define INC_IMAGE_READER_H

#include "ModuleBase/ModuleBase.h"
#include "DataTrans/DataTrans.pb.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <atomic>
#include <mutex>
#include <thread>

namespace ascendFaceRecognition {
class ImageReader : public ModuleBase {
public:
    ImageReader();
    ~ImageReader() {};
    APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs);
    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR ParseConfig(ConfigParser &configParser);
    APP_ERROR GetImageData(std::string &filePath, std::string uuid);
    APP_ERROR ScanFolder(std::string folderPath);

    // stream info
    uint32_t format_ = 0;
    uint32_t isDisplay_ = 0;
    std::atomic<int> stop_ = { 0 };
    uint64_t frameNum_ = 0;
    std::string jpegFolderPath_ = {};
    std::vector<std::string> fileNameSet_ = {};
};
} // namespace ascendFaceRecognition

#endif
