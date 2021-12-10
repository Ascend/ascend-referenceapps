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
#include <sys/time.h>

/*
 * @description: Implementation of constructor for class AclProcess with parameter list
 * @attention: context is passed in as a parameter after being created in ResourceManager::InitResource
 */
AclProcess::AclProcess(int deviceId, aclrtContext context)
    : deviceId_(deviceId), context_(context),
      stream_(nullptr), dvppCommon_(nullptr)
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
        ret = (APP_ERROR)aclrtDestroyStream(stream_);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy the stream, ret = " << ret << ".";
        }
        stream_ = nullptr;
    }
    // Release Dvpp buffer
    dvppCommon_->ReleaseDvppBuffer();
    // Release objects resource after destroy stream
    dvppCommon_.reset();
}

/*
 * @description: Create stream resource and DvppJpegeProcess object pointer
 * @return: APP_ERR_OK success
 * @return: Other values failure
 * @attention: context will be released in ResourceManager::Release
 */
APP_ERROR AclProcess::InitResource()
{
    APP_ERROR ret = (APP_ERROR)aclrtSetCurrentContext(context_); // Set current context
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set the acl context, ret = " << ret << ".";
        return ret;
    }
    LogInfo << "Created the context successfully.";
    ret = (APP_ERROR)aclrtCreateStream(&stream_); // Create stream for application
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create the acl stream, ret = " << ret << ".";
        return ret;
    }
    LogInfo << "Created the stream successfully.";

    if (dvppCommon_ == nullptr) {
        dvppCommon_ = std::make_shared<DvppCommon>(stream_);
        ret = dvppCommon_->Init();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to initialize dvppCommon, ret = " << ret << ".";
            return ret;
        }
    }

    return APP_ERR_OK;
}

/*
 * @description: Encode image with the specified format and write result
 * @param imageFile  absolute path of the input image
 * @param width      width of the input image
 * @param height     height of the input image
 * @param format     pixel format supported by dvpp
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR AclProcess::Process(std::string imageFile, int width, int height, acldvppPixelFormat format) const
{
    struct timeval begin = {0};
    struct timeval end = {0};
    gettimeofday(&begin, nullptr);

    bool withSynchronize = true;
    RawData imageInfo;
    APP_ERROR ret = ReadFile(imageFile, imageInfo); // Read image data from input image file
    if (ret != APP_ERR_OK) {
        LogError << "Failed to read file, ret = " << ret << ".";
        return ret;
    }

    ret = dvppCommon_->CombineJpegeProcess(imageInfo, width, height, format, withSynchronize);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to process file, ret = " << ret << ".";
        return ret;
    }
    // Get output of encode jpeg iamge
    std::shared_ptr<DvppDataInfo> encodeOutData = dvppCommon_->GetEncodedImage();
    // Get output file name
    std::string fileFullName = GetName(imageFile);
    std::set<char> delims { '.' };
    std::vector<std::string> namePath = SplitPath(fileFullName, delims);
    std::string fileName = namePath[0];
    void* resHostBuf = nullptr;
    // Malloc host memory to save the output data
    ret = (APP_ERROR)aclrtMallocHost(&resHostBuf, encodeOutData->dataSize);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to malloc memory on host, ret = " << ret << ".";
        return ret;
    }
    std::shared_ptr<void> outBuf(resHostBuf, aclrtFreeHost);
    ret = (APP_ERROR)aclrtMemcpy(outBuf.get(), encodeOutData->dataSize, encodeOutData->data,
        encodeOutData->dataSize, ACL_MEMCPY_DEVICE_TO_HOST); // Copy the output data from device to host
    if (ret != APP_ERR_OK) {
        LogError << "Failed to copy memory from device to host, ret = " << ret << ".";
        return ret;
    }
    ret = WriteResult(fileName, outBuf, encodeOutData->dataSize);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to write result file, ret = " << ret << ".";
        return ret;
    }

    gettimeofday(&end, nullptr);
    // Calculate the time cost of encode jpeg
    const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
    const double fps = 1 * SEC2MS / costMs;
    LogInfo << "[dvpp Delay] cost: " << costMs << "ms\tfps: " << fps;
    return APP_ERR_OK;
}

/*
 * @description: Write result of encoded image
 * @param fileName           result file name
 * @param outBuf             output data buffer address pointer in host
 * @param encodeOutDataSize  size of encoded output data
 * @return: APP_ERR_OK success
 * @return: Other values failure
 */
APP_ERROR AclProcess::WriteResult(std::string& fileName, const std::shared_ptr<void>& outBuf,
    const uint32_t& encodeOutDataSize) const
{
    std::string resultPathName = "result";
    // Create result directory when it does not exist
    if (access(resultPathName.c_str(), 0) != 0) {
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR); // for linux
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName << ", ret = " << ret << ".";
            return ret;
        }
    }
    std::string outputFileName = resultPathName + "/" + fileName + ".jpg";
    SetFileDefaultUmask();
    FILE *outFileFp = fopen(outputFileName.c_str(), "wb+"); // Open the output file
    if (outFileFp == nullptr) {
        LogError << "Failed to open file " << outputFileName << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    fwrite(outBuf.get(), 1, encodeOutDataSize, outFileFp); // Write the output file
    fflush(outFileFp);
    fclose(outFileFp);
    return APP_ERR_OK;
}