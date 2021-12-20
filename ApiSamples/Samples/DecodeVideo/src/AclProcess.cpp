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
using namespace std;

namespace {
    bool g_runFlag = true;
    /*
     * @description: Callback function to write YUV result and free resources
     * @param: input specifies the input stream description
     * @param: output specifies the output pic description after Vdec
     * @param: userdata specifies User-defined data
     */
    void Callback(acldvppStreamDesc *input, acldvppPicDesc *output, void *userdata)
    {
        // free input dataDev
        void *dataDev = acldvppGetStreamDescData(input);
        APP_ERROR ret = acldvppFree(dataDev);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to free input stream desc dataDev";
        }
        // destroy stream desc
        ret = acldvppDestroyStreamDesc(input);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy input stream desc";
        }

        // free output vdecOutBufferDev
        std::unique_ptr<void, decltype(acldvppFree)*> vdecOutBufferDev(acldvppGetPicDescData(output), acldvppFree);
        uint32_t size = acldvppGetPicDescSize(output);
        static int count = 1;
        // write to file
        std::string fileNameSave = "result/image" + std::to_string(count);
        if (!VdecUtils::WriteToFile(fileNameSave.c_str(), vdecOutBufferDev, size)) {
            LogError << "Failed to write file";
        }

        // destroy pic desc
        ret = acldvppDestroyPicDesc(output);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy pic desc";
        }

        LogInfo << "Succeed to callback " << count;
        count++;
    }

    /*
     * @description: Thread running function, waiting for trigger callback processing
     */
    void *ThreadFunc(void *arg)
    {
        aclrtContext context = static_cast<aclrtContext>(arg);
        APP_ERROR ret = aclrtSetCurrentContext(context);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to set context, ret = " << ret;
            return reinterpret_cast<void*>(-1);
        }

        LogInfo << "thread start ";
        while (g_runFlag) {
            // Notice: timeout 1000ms
            aclrtProcessReport(1000);
        }

        return nullptr;
    }

    VdecConfig DvppInit(const PicDesc &picDesc, pthread_t threadId_, acldvppStreamFormat videoFormat)
    {
        VdecConfig vdecConfig;
        vdecConfig.inputWidth = picDesc.width;
        vdecConfig.inputHeight = picDesc.height;
        vdecConfig.outFormat = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        vdecConfig.channelId = 0;
        vdecConfig.threadId = threadId_;
        vdecConfig.callback = Callback;
        vdecConfig.inFormat = videoFormat;
        return vdecConfig;
    }

    /*
     * @description: Get each frame of video and use Dvpp to decode it
     * @param: threadId_ specifies the thread Id
     * @param: videoReader is used to help to read frames in the video stream
     * @param: processVdec specifies DvppVdecProcess
     * @param: stream_ specifies the current stream
     * @return: APP_ERR_OK if success, other values if failure
     */
    APP_ERROR SplitVdec(pthread_t threadId_, VideoReader videoReader, DvppCommon &processVdec, aclrtStream stream_)
    {
        int32_t count = 0;
        AVPacket packet = {};
        (void)av_init_packet(&packet);
        while (videoReader.GetNalu(packet) == 0) {
            std::shared_ptr<DvppDataInfo> vdecData_ = std::make_shared<DvppDataInfo>();
            vdecData_->dataSize = packet.size;
            void *inHostBuff = nullptr;
            APP_ERROR ret = (APP_ERROR)aclrtMallocHost(&inHostBuff, packet.size); // Malloc memory for host buffer
            if (ret != APP_ERR_OK) {
                LogError << "Failed to malloc device buffer, size is " << packet.size << ", ret = " << ret;
                return ret;
            }
            std::shared_ptr<void> inHostBuffManager(inHostBuff, aclrtFreeHost);
            vdecData_->data = (uint8_t *)inHostBuff; // Set input host buffer
            ret = (APP_ERROR)aclrtMemcpy(inHostBuff, packet.size, packet.data, packet.size, ACL_MEMCPY_HOST_TO_HOST);
            if (ret != APP_ERR_OK) {
                LogError << "Failed to memcpy acl, ret=" << ret;
                return ret;
            }

            ret = processVdec.CombineVdecProcess(vdecData_, NULL); // Decode Video
            if (ret != APP_ERR_OK) {
                LogError << "Failed to do VdecProcess, ret = " << ret;
                (void)aclrtUnSubscribeReport(static_cast<uint64_t>(threadId_), stream_);
                return ret;
            }
            ++count;
            (void)av_packet_unref(&packet);
            LogInfo << "Send aclVdec Frame successfully, count= " << count;
        }
        return APP_ERR_OK;
    }
}

AclProcess::AclProcess()
{
}

void AclProcess::SetInputFilePath(const std::string &inputFilePath)
{
    filePath_ = inputFilePath;
}

AclProcess::~AclProcess()
{
    outFolder_ = nullptr;
    DestroyResource();
}

/*
 * @description: Initialize the Acl process, set device Id, create context and stream
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR AclProcess::InitResource()
{
    // ACL init
    const char *aclConfigPath = "./data/config/acl.json";
    APP_ERROR ret = aclInit(aclConfigPath);
    if (ret != APP_ERR_OK) {
        LogError << "ACL init failed";
        return APP_ERR_ACL_FAILURE;
    }
    LogInfo << "ACL init successfully";

    ConfigParser config;
    // Initialize the config parser module
    if (config.ParseConfig("./data/config/setup.config") != APP_ERR_OK) {
        return APP_ERR_COMM_INIT_FAIL;
    }

    ret = config.GetIntValue("device_id", deviceId_);
    // Convert from string to digital number
    if (ret != APP_ERR_OK) {
        LogError << "DeviceId is not digit, please check, ret = " << ret;
        return APP_ERR_COMM_INVALID_PARAM;
    }
    // Check validity of device id
    if (deviceId_ < 0) {
        LogError << "DeviceId:" << deviceId_ << " is less than 0, not valid, please check";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    // open device
    ret = aclrtSetDevice(deviceId_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set device " << deviceId_ << ", ret = " << ret;
        return APP_ERR_ACL_FAILURE;
    }
    LogInfo << "Set device successfully " << deviceId_;

    // create context (set current)
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create context for device " << deviceId_ << ", ret = " << ret;
        return APP_ERR_ACL_FAILURE;
    }
    LogInfo << "Create context successfully";

    // create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create stream, ret = " << ret;
        return APP_ERR_ACL_FAILURE;
    }
    LogInfo << "Create stream successfully";

    return APP_ERR_OK;
}

/*
 * @description: Vdec process for decoding video
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR AclProcess::DoVdecProcess()
{
    // create threadId
    int createThreadErr = pthread_create(&threadId_, nullptr, ThreadFunc, context_);
    if (createThreadErr != 0) {
        LogError << "Failed to create thread, err = " << createThreadErr;
        return APP_ERR_ACL_FAILURE;
    }
    LogInfo << "Create thread successfully, threadId = " << threadId_;
    (void)aclrtSubscribeReport(static_cast<uint64_t>(threadId_), stream_);

    APP_ERROR ret = VdecUtils::CheckFolder(outFolder_);
    if (ret != APP_ERR_OK) {
        LogError << "Mkdir out folder error.";
        (void)aclrtUnSubscribeReport(static_cast<uint64_t>(threadId_), stream_);
        return APP_ERR_ACL_FAILURE;
    }

    std::string inFileName(filePath_.c_str());
    VideoReader videoReader(inFileName);
    videoReader.OpenVideo();

    videoReader.GetVideoWH(picDesc_);

    acldvppStreamFormat videoFormat;
    ret = videoReader.GetVideoType(videoFormat);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to get video format";
        videoReader.CloseVideo();
        (void)aclrtUnSubscribeReport(static_cast<uint64_t>(threadId_), stream_);
        return APP_ERR_ACL_FAILURE;
    }

    // dvpp init
    VdecConfig vdecConfig = DvppInit(picDesc_, threadId_, videoFormat);

    DvppCommon processVdec(vdecConfig);
    processVdec.InitVdec();

    // use ffmpeg to split h264
    ret = SplitVdec(threadId_, videoReader, processVdec, stream_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to split video frame";
        videoReader.CloseVideo();
        processVdec.DeInit();
        (void)aclrtUnSubscribeReport(static_cast<uint64_t>(threadId_), stream_);
        return ret;
    }

    // free resource
    videoReader.CloseVideo();
    processVdec.DeInit();
    (void)aclrtUnSubscribeReport(static_cast<uint64_t>(threadId_), stream_);

    return APP_ERR_OK;
}

/*
 * @description: Destroy thread, Stream, Context and reset device
 */
void AclProcess::DestroyResource()
{
    APP_ERROR ret;
    // destroy thread
    g_runFlag = false;
    void *res = nullptr;
    LogInfo << "Destroy thread start.";
    int joinThreadErr = pthread_join(threadId_, &res);
    if (joinThreadErr != 0) {
        LogError << "Failed to join thread, threadId = " << threadId_ << "err = " << joinThreadErr;
    } else {
        if ((uint64_t)res != 0) {
            LogError << "Failed to run thread. ret is " << (uint64_t)res;
        }
    }
    LogInfo << "Destroy thread success.";

    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy stream, ret = " << ret;
        }
        stream_ = nullptr;
    }
    LogInfo << "End to destroy stream";

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy context, ret = " << ret;
        }
        context_ = nullptr;
    }
    LogInfo << "End to destroy context";

    ret = aclrtResetDevice(deviceId_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to reset device, ret = " << ret;
    }
    LogInfo << "End to reset device is " << deviceId_;
}
