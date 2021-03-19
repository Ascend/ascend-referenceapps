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
#include <thread>

/*
 * @description: Constructor
 * @param: resizeWidth specifies the resized width
 * @param: resizeHeight specifies the resized hegiht
 * @param: stream is used to maintain the execution order of operations
 * @param: context is used to manage the life cycle of objects
 * @param: dvppCommon is a class for decoding and resizing
 */
AclProcess::AclProcess(uint32_t resizeWidth, uint32_t resizeHeight, aclrtContext context,
                       aclrtStream stream, std::shared_ptr<DvppCommon> dvppCommon)
    : resizeWidth_(resizeWidth), resizeHeight_(resizeHeight), context_(context),
      stream_(stream), dvppCommon_(dvppCommon)
{
}

/*
 * @description: Release AclProcess resources
 * @return: aclError which is error code of ACL API
 */
APP_ERROR AclProcess::Release()
{
    // Release objects resource
    APP_ERROR ret = dvppCommon_->DeInit();
    dvppCommon_->ReleaseDvppBuffer();

    if (ret != APP_ERR_OK) {
        LogError << "Failed to deinitialize dvppCommon_, ret = " << ret;
        return ret;
    }
    LogInfo << "dvppCommon_ object deinitialized successfully";
    dvppCommon_.reset();

    // Release stream
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy stream, ret = " << ret;
            stream_ = nullptr;
            return ret;
        }
        stream_ = nullptr;
    }
    LogInfo << "The stream is destroyed successfully";
    return APP_ERR_OK;
}

/*
 * @description: Initialize DvppCommon object
 * @return: aclError which is error code of ACL API
 */
APP_ERROR AclProcess::InitModule()
{
    // Create Dvpp JpegD object
    dvppCommon_ = std::make_shared<DvppCommon>(stream_);
    if (dvppCommon_ == nullptr) {
        LogError << "Failed to create dvppCommon_ object";
        return APP_ERR_COMM_INIT_FAIL;
    }
    LogInfo << "DvppCommon object created successfully";
    APP_ERROR ret = dvppCommon_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to initialize dvppCommon_ object, ret = " << ret;
        return ret;
    }
    LogInfo << "DvppCommon object initialized successfully";
    return APP_ERR_OK;
}

/*
 * @description: Initialize AclProcess resources
 * @return: aclError which is error code of ACL API
 */
APP_ERROR AclProcess::InitResource()
{
    APP_ERROR ret = aclrtSetCurrentContext(context_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to get ACL context, ret = " << ret;
        return ret;
    }
    LogInfo << "The context is created successfully";
    ret = aclrtCreateStream(&stream_); // Create stream for application
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create ACL stream, ret = " << ret;
        return ret;
    }
    LogInfo << "The stream is created successfully";
    // Initialize dvpp module
    if (InitModule() != APP_ERR_OK) {
        return APP_ERR_COMM_INIT_FAIL;
    }
    return APP_ERR_OK;
}

/*
 * @description: Read image files, and perform decoding and scaling
 * @param: imageFile specifies the image path to be processed
 * @return: aclError which is error code of ACL API
 */
APP_ERROR AclProcess::Preprocess(std::string imageFile) const
{
    // test the image file valid
    if (imageFile.empty()) {
        LogError << "Failed to get image";
        return APP_ERR_ACL_INVALID_FILE;
    }

    RawData imageInfo;
    // Read image data from input image file
    APP_ERROR ret = ReadFile(imageFile, imageInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to read image on " << imageFile << ", ret = " << ret << ".";
        return ret;
    }
    ret = dvppCommon_->CombineJpegdProcess(imageInfo, PIXEL_FORMAT_YUV_SEMIPLANAR_420, true);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to process decode, ret = " << ret << ".";
        return ret;
    }
    // Get output of decoded jpeg image
    std::shared_ptr<DvppDataInfo> decodeOutData = dvppCommon_->GetDecodedImage();

    if (decodeOutData == nullptr) {
        LogError << "Decode output buffer is null.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    DvppDataInfo resizeOut;
    resizeOut.width = resizeWidth_;
    resizeOut.height = resizeHeight_;
    resizeOut.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    // Run resize application function
    ret = dvppCommon_->CombineResizeProcess(*(decodeOutData.get()), resizeOut, true);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to process resize, ret = " << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/*
 * @description: Decode and scale the picture, and write the result to a file
 * @param: imageFile specifies the image path to be processed
 * @return: aclError which is error code of ACL API
 */
APP_ERROR AclProcess::Process(std::string imageFile) const
{
    struct timeval begin = {0};
    struct timeval end = {0};
    gettimeofday(&begin, nullptr);
    // deal with image
    APP_ERROR ret = Preprocess(imageFile);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to preprocess, ret = " << ret;
        return ret;
    }
    gettimeofday(&end, nullptr);
    // Calculate the time cost of preprocess
    const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + \
        (end.tv_usec - begin.tv_usec) / SEC2MS;
    const double fps = 1 * SEC2MS / costMs;
    LogInfo << "[dvpp Delay] cost: " << costMs << "ms\tfps: " << fps;
    // Get output of resize module
    std::shared_ptr<DvppDataInfo> resizeOutData = dvppCommon_->GetResizedImage();
    if (resizeOutData->dataSize == 0) {
        LogError << "resizeOutData return NULL";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    // Malloc host memory for the inference output
    void *resHostBuf = nullptr;
    ret = aclrtMallocHost(&resHostBuf, resizeOutData->dataSize);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to allocate memory from host ret = " << ret;
        return ret;
    }
    std::shared_ptr<void> outBuf(resHostBuf, aclrtFreeHost);
    // Memcpy the output data from device to host
    ret = aclrtMemcpy(outBuf.get(), resizeOutData->dataSize, resizeOutData->data,
                      resizeOutData->dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to copy memory from device to host, ret = " << ret;
        return ret;
    }
    // write resize result
    ret = WriteResult(resizeOutData->dataSize, outBuf);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to write result, ret = " << ret;
        return ret;
    }
    return APP_ERR_OK;
}

/*
 * @description: Write result image to file
 * @param: resultSize specifies the size of the result image
 * @param: outBuf specifies the memory on the host to save the result image
 * @return: aclError which is error code of ACL API
 */
APP_ERROR AclProcess::WriteResult(uint32_t resultSize, std::shared_ptr<void> outBuf) const
{
    std::string resultPathName = "result";
    // Create result directory when it does not exist
    if (access(resultPathName.c_str(), 0) != 0) {
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR); // for linux
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName << ", ret = " << ret;
            return ret;
        }
    }
    // Result file name use the time stamp as a suffix
    std::string timeString;
    GetCurTimeString(timeString);
    // Create result file under result directory
    resultPathName = resultPathName + "/result_" + timeString + ".yuv";
    SetFileDefaultUmask();
    FILE *fp = fopen(resultPathName.c_str(), "wb");
    if (fp == nullptr) {
        LogError << "Failed to open file";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    uint32_t result = fwrite(outBuf.get(), 1, resultSize, fp);
    if (result != resultSize) {
        LogError << "Failed to write file";
        return APP_ERR_COMM_WRITE_FAIL;
    }
    LogInfo << "Write result to file successfully";
    uint32_t ff = fflush(fp);
    if (ff != 0) {
        LogError << "Failed to fflush file";
        return APP_ERR_COMM_DESTORY_FAIL;
    }
    uint32_t fc = fclose(fp);
    if (fc != 0) {
        LogError << "Failed to fclose file";
        return APP_ERR_COMM_DESTORY_FAIL;
    }
    return APP_ERR_OK;
}