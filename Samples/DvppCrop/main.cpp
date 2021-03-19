/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
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

#include <string>
#include <sstream>
#include <ResourceManager/ResourceManager.h>
#include <ConfigParser/ConfigParser.h>
#include "DvppCommon/DvppCommon.h"
#include "CommandParser/CommandParser.h"

using CropOption = struct CropOption {
    /* input image to crop */
    std::string inputImage = {};
    uint32_t inputWidth = {};
    uint32_t inputHeight = {};
    /* position to crop */
    Rect cropArea = {};
    /* path to save cropped image */
    std::string outputImage = {};
    /* input image pixel format */
    acldvppPixelFormat pixelFormat = {};
};

namespace {
/* For ACL framework initialization, can be empty. */
const std::string ACL_CONFIG = "./data/config/acl.json";
const std::string DEFAULT_INPUT = "./data/test.yuv";
const std::string DEFAULT_OUTPUT = "./data/cropped.yuv";
std::unique_ptr<DvppCommon> g_cropProcessObj = nullptr;
}

/* *
 * Allocate Dvpp buffer.
 *
 * Notice that the Dvpp buffer can not be replaced by normal device buffer
 * if we need to use this buffer for vdec, vpc, jpeg, etc.
 * @param size bytes of buffer to allocate
 * @return shared pointer to buffer
 */
static std::shared_ptr<void> GetDevDvppBuffer(const uint32_t &size)
{
    if (size == 0) {
        return nullptr;
    }

    void *buffer = nullptr;
    APP_ERROR errRet = acldvppMalloc(&buffer, size);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to malloc " << size << " bytes dvpp buffer, ret = " << errRet << ".";
        return nullptr;
    }

    return std::shared_ptr<void>(buffer, acldvppFree);
}

/* *
 * Allocate host buffer.
 *
 * Noticed that this buffer is managed by ACL, which is different from the buffer allocated by pure malloc(3).
 *
 * @param size bytes of buffer to allocate
 * @return shared pointer to buffer
 */
static std::shared_ptr<void> GetHostBuffer(const uint32_t &size)
{
    if (size == 0) {
        return nullptr;
    }

    void *buffer = nullptr;
    APP_ERROR errRet = aclrtMallocHost(&buffer, size);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to malloc " << size << " bytes host buffer, ret = " << errRet << ".";
        return nullptr;
    }

    return std::shared_ptr<void>(buffer, aclrtFreeHost);
}

/* *
 * Read image from file, then copy to device Dvpp memory.
 *
 * @param hostImage path to image
 * @param dataDev shared pointer to device memory
 * @param dataSize data size of image
 * @return
 */
static APP_ERROR CopyHostImageToDev(const std::string &hostImage, std::shared_ptr<void> &dataDev, uint32_t &dataSize)
{
    /* get the absolute path of input file */
    char resolvedPath[PATH_MAX] = {0};
    if (realpath(hostImage.c_str(), resolvedPath) == nullptr) {
        LogError << "Failed to resolve the real path of " << hostImage << ".";
        return APP_ERR_COMM_NO_EXIST;
    }

    /* Open file to read */
    FILE *fp = fopen(resolvedPath, "rb");
    if (fp == nullptr) {
        LogError << "Failed to open file " << std::string(resolvedPath);
        return APP_ERR_ACL_INVALID_FILE;
    }

    /* Get input image size */
    fseek(fp, 0, SEEK_END);
    uint32_t fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fileSize == 0) {
        LogError << "File: " << std::string(resolvedPath) << " size is 0.";
        fclose(fp);
        return APP_ERR_ACL_INVALID_FILE;
    }

    /* Get host buffer for image */
    std::shared_ptr<void> hostBuf = GetHostBuffer(fileSize);
    if (hostBuf == nullptr) {
        LogError << "Failed to get host buffer for image of " << std::string(resolvedPath) << ".";
        fclose(fp);
        return APP_ERR_COMM_ALLOC_MEM;
    }

    /* Read image data to host buffer */
    size_t retRd = fread(hostBuf.get(), 1, fileSize, fp);
    if (retRd == 0) {
        LogError << "Failed to read image data to host buffer, image: " << std::string(resolvedPath) << ".";
        fclose(fp);
        return APP_ERR_COMM_READ_FAIL;
    }
    fclose(fp);

    /* Get image buffer in device, the buffer muse be dvpp buffer */
    std::shared_ptr<void> imageBufDev = GetDevDvppBuffer(fileSize);
    if (imageBufDev == nullptr) {
        LogError << "Failed to get image from device dvpp buffer.";
        return APP_ERR_COMM_ALLOC_MEM;
    }

    /* Copy host image data to device dvpp memory */
    APP_ERROR errRet = aclrtMemcpy(imageBufDev.get(), fileSize, hostBuf.get(), fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to copy data to device, errRet = " << errRet << ".";
        return APP_ERR_COMM_INNER;
    }
    dataDev = imageBufDev;
    dataSize = fileSize;
    return APP_ERR_OK;
}

/* *
 * Copy image from device
 *
 * @param devImage shared pointer to device memory, referencing an image
 * @param bufHost shared pointer to host buffer
 * @param bufSize size of buffer of image
 * @return
 */
static APP_ERROR CopyDevImageToHost(std::shared_ptr<DvppDataInfo> devImage, std::shared_ptr<void> bufHost,
    const uint32_t bufSize)
{
    if ((devImage == nullptr) || (bufHost == nullptr) || (bufSize == 0)) {
        return APP_ERR_ACL_INVALID_PARAM;
    }

    if (devImage->dataSize > bufSize) {
        return APP_ERR_ACL_INVALID_PARAM;
    }

    APP_ERROR errRet =
        aclrtMemcpy(bufHost.get(), bufSize, devImage->data, devImage->dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to copy data to host, ret = " << errRet << ".";
        return APP_ERR_COMM_INNER;
    }
    return APP_ERR_OK;
}

/* *
 * Save cropped image to host
 *
 * @param path for image
 * @param imageData shared pointer to image buffer
 * @param imageSize size of image
 * @return APP_ERR_OK if save success. Error code otherwise
 */
static APP_ERROR SaveCropImage(const std::string &path, const std::shared_ptr<void> imageData,
    const uint32_t &imageSize)
{
    if ((imageData == nullptr) || (imageSize == 0)) {
        return APP_ERR_ACL_INVALID_PARAM;
    }
    SetFileDefaultUmask();
    FILE *fp = fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        LogError << "Failed to create file " << path << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }

    size_t retWrt = fwrite(imageData.get(), 1, imageSize, fp);
    if (retWrt == 0) {
        LogError << "Failed to write file " << path << ".";
        fclose(fp);
        return APP_ERR_COMM_WRITE_FAIL;
    }

    fclose(fp);
    return APP_ERR_OK;
}

/* Init system, such as ACL runtime, choose device id, create context, create stream */
/* *
 * Init the crop system.
 * 1. Init ACL runtime
 * 2. Choose device id
 * 3. Create ACL Context
 * 4. Create stream
 *
 * @param devID id of device to use
 * @param context ACL context to init
 * @param stream ACL stream to init
 * @return APP_ERR_OK if save success. Error code otherwise
 */
static APP_ERROR CropSystemInit(const uint32_t &devID, aclrtContext &context, aclrtStream &stream)
{
    /* Init ACL environment */
    APP_ERROR errRet = aclInit(ACL_CONFIG.c_str());
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to initialize acl, ret = " << errRet << ".";
        return errRet;
    }

    errRet = aclrtSetDevice(devID);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to set device " << devID << " , ret = " << errRet << ".";
        return errRet;
    }

    /* Create context */
    errRet = aclrtCreateContext(&context, devID);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to create context, ret = " << errRet << ".";
        (void)aclrtResetDevice(devID);
        return errRet;
    }

    /* Create stream to process */
    errRet = aclrtCreateStream(&stream);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to create stream, ret = " << errRet << ".";
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(devID);
        return errRet;
    }

    return APP_ERR_OK;
}

/* *
 * Release the crop system.
 *
 * @param devID id of device to use
 * @param context ACL context to destroy
 * @param stream ACL stream to destroy
 * @return APP_ERR_OK if save success. Error code otherwise
 */
static APP_ERROR CropSystemDestroy(const uint32_t &devID, const aclrtContext &context, const aclrtStream &stream)
{
    if (g_cropProcessObj != nullptr) {
        g_cropProcessObj->DeInit();
        g_cropProcessObj->ReleaseDvppBuffer();
        g_cropProcessObj = nullptr;
    }

    APP_ERROR errRet = aclrtDestroyStream(stream);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to destory stream, ret = " << errRet << ".";
        return errRet;
    }

    errRet = aclrtDestroyContext(context);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to destory context, ret = " << errRet << ".";
        return errRet;
    }

    errRet = aclrtResetDevice(devID);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to reset device, ret = " << errRet << ".";
        return errRet;
    }

    return APP_ERR_OK;
}

/* *
 * Parse the command line option of resolution.
 *
 * Expected format: <int>,<int>
 * Example: 100,100
 *
 * @param resStr input string from command line
 * @param width reference to width
 * @param height reference to height
 * @return APP_ERR_OK if save success. Error code otherwise
 */
static APP_ERROR ParseResolutionOption(const std::string &resStr, uint32_t &width, uint32_t &height)
{
    size_t delimPos = resStr.find_first_of(',');
    if (std::string::npos == delimPos) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    std::string widthStr = resStr.substr(0, delimPos);
    std::string heightStr = resStr.substr(delimPos + 1);

    std::stringstream widthSS(widthStr);
    widthSS >> width;
    std::stringstream heightSS(heightStr);
    heightSS >> height;

    return APP_ERR_OK;
}

/* *
 * Parse the command line option of crop area.
 *
 * Expected format: <int>,<int>,<int>,<int>
 * Example: 100,100,100,100
 *
 * @param resStr input string from command line
 * @param rect reference to the Rect struct, filling in the data of crop area of user input
 * @return APP_ERR_OK if save success. Error code otherwise
 */
static APP_ERROR ParseCropAreaOption(const std::string &resStr, Rect &rect)
{
    size_t delimPos = resStr.find_first_of(',');
    if (std::string::npos == delimPos) {
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::string leftPosStr = resStr.substr(0, delimPos);
    std::stringstream leftPosSS(leftPosStr);
    leftPosSS >> rect.x;

    std::string strRemain = resStr.substr(delimPos + 1);
    delimPos = strRemain.find_first_of(',');
    if (std::string::npos == delimPos) {
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::string topPosStr = strRemain.substr(0, delimPos);
    std::stringstream topPosSS(topPosStr);
    topPosSS >> rect.y;

    strRemain = strRemain.substr(delimPos + 1);
    delimPos = strRemain.find_first_of(',');
    if (std::string::npos == delimPos) {
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::string widthPosStr = strRemain.substr(0, delimPos);
    std::stringstream widthPosSS(widthPosStr);
    widthPosSS >> rect.width;

    std::string heightPosStr = strRemain.substr(delimPos + 1);
    std::stringstream heightPosSS(heightPosStr);
    heightPosSS >> rect.height;

    return APP_ERR_OK;
}

/* *
 * Define the command line option of main function.
 *
 * @param cmdOpt reference of CommandParser, declared in main function, extends from AscendBase.
 */
static void DefineCommandOption(CommandParser &cmdOpt)
{
    cmdOpt.AddOption("--input", DEFAULT_INPUT, "Specify the input image");
    cmdOpt.AddOption("--output", DEFAULT_OUTPUT, "Set the path to save the cropped image");
    cmdOpt.AddOption("--resolution", "1920,1080",
        "Set the resolution of input image, such as: 1920,1080; 1920 is width, 1080 is height");
    cmdOpt.AddOption("--crop", "0,0,16,2",
        "Set crop area, such as 0,0,512,416; 0 is left side begin to crop, 0 is top side begin to crop, 512 is width, "
        "416 is height");
    cmdOpt.AddOption("--format", "1", "Set pixelformat: 0:Gray 1:NV12 2:NV21");
    return;
}

/* *
 * Filling crop option from command line input
 *
 * @param cmdOpt command parser from main function
 * @param cropOpt target crop option to filling in
 * @return APP_ERR_OK if save success. Error code otherwise
 */
static APP_ERROR GetCropOption(CommandParser cmdOpt, CropOption &cropOpt)
{
    std::string inputImage = cmdOpt.GetStringOption("--input");
    std::string outputImage = cmdOpt.GetStringOption("--output");
    std::string resolution = cmdOpt.GetStringOption("--resolution");
    std::string cropAre = cmdOpt.GetStringOption("--crop");
    uint32_t pixelFormat = cmdOpt.GetIntOption("--format");

    uint32_t inputWidth = 0;
    uint32_t inputHeight = 0;
    APP_ERROR errRet = ParseResolutionOption(resolution, inputWidth, inputHeight);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to parse resolution parameters, ret = " << errRet << ".";
        return errRet;
    }

    Rect rect;
    errRet = ParseCropAreaOption(cropAre, rect);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to parse crop area parameters, ret = " << errRet << ".";
        return errRet;
    }

    cropOpt.inputWidth = inputWidth;
    cropOpt.inputHeight = inputHeight;
    cropOpt.inputImage = inputImage;
    cropOpt.cropArea = rect;
    cropOpt.outputImage = outputImage;
    cropOpt.pixelFormat = (acldvppPixelFormat)pixelFormat;

    return APP_ERR_OK;
}

/* *
 * @brief Read config from file
 *
 * @param configData Config parser
 * @param resourceInfo Resource info
 * @return APP_ERROR
 */
APP_ERROR ReadConfigFromFile(ConfigParser &configData, ResourceInfo &resourceInfo)
{
    // Get the device id used by application from config file
    int deviceId;
    APP_ERROR ret = configData.GetIntValue("device_id", deviceId);
    // Convert from string to digital number
    if (ret != APP_ERR_OK) {
        LogError << "Failed to get device with id " << deviceId << "which seems not an integer.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    // Check validity of device id
    if (deviceId < 0) {
        LogError << "Failed to get device with id " << deviceId << " which is less than 0.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    resourceInfo.deviceIds.insert(deviceId);
    return APP_ERR_OK;
}

/* *
 * Initialize config
 * which can -5 lines for "super big function" main
 * and +7 lines of comment
 *
 * @param config instance of ConfigParser
 * @param resourceInfo instance of ResourceInfo
 * @return APP_ERR_OK if config init ok, error code otherwise.
 */
APP_ERROR InitConfig(ConfigParser &config, ResourceInfo &resourceInfo)
{
    if (config.ParseConfig("./data/config/setup.config") != APP_ERR_OK) {
        return APP_ERR_COMM_INIT_FAIL;
    }
    resourceInfo.aclConfigPath = ACL_CONFIG;
    return ReadConfigFromFile(config, resourceInfo);
}

/* *
 * Fill in input data with the command line parameters of cropping
 *
 * @param inputData  DvppCropInputInfo structure
 * @param dataDev    shared pointer to buffer alloced in device
 * @param dataSize   size of data
 * @param cropOpt    crop information
 * @return APP_ERR_OK
 */
void FillCropInputData(DvppCropInputInfo &inputData, std::shared_ptr<void> dataDev, uint32_t dataSize,
    CropOption &cropOpt)
{
    inputData.dataInfo.data = static_cast<uint8_t *>(dataDev.get());
    inputData.dataInfo.dataSize = dataSize;
    inputData.dataInfo.width = cropOpt.inputWidth;
    inputData.dataInfo.height = cropOpt.inputHeight;
    inputData.dataInfo.widthStride = DVPP_ALIGN_UP(cropOpt.inputWidth, VPC_STRIDE_WIDTH);
    inputData.dataInfo.heightStride = DVPP_ALIGN_UP(cropOpt.inputHeight, VPC_STRIDE_HEIGHT);
    inputData.dataInfo.format = cropOpt.pixelFormat;
    inputData.roi.left = cropOpt.cropArea.x;
    inputData.roi.up = cropOpt.cropArea.y;
    inputData.roi.right = cropOpt.cropArea.x + cropOpt.cropArea.width - ODD_NUM_1;
    inputData.roi.down = cropOpt.cropArea.y + cropOpt.cropArea.height - ODD_NUM_1;
}

/* *
 * Main routine of crop process.
 * 1. Copy host image to device
 * 2. Fill in parameters of crop process
 * 3. Crop in device
 * 4. Get pointer of memory in device
 *
 * @param cropOpt Crop option, parsed
 * @param devID device id
 * @param cropImageHost pointer to host image
 * @param cropImageSize size of host image
 * @return APP_ERR_OK if config init ok, error code otherwise.
 */
APP_ERROR CropMainRoutine(CropOption cropOpt, aclrtStream stream, std::shared_ptr<void> &cropImageHost,
    uint32_t &cropImageSize)
{
    APP_ERROR errRet;

    g_cropProcessObj.reset(new DvppCommon(stream));
    if (g_cropProcessObj == nullptr) {
        return APP_ERR_COMM_ALLOC_MEM;
    }

    errRet = g_cropProcessObj->Init();
    if (errRet != APP_ERR_OK) {
        return errRet;
    }

    /* Copy host image to device */
    std::shared_ptr<void> dataDev = nullptr;
    uint32_t dataSize = 0;
    errRet = CopyHostImageToDev(cropOpt.inputImage, dataDev, dataSize);
    if (errRet != APP_ERR_OK) {
        return errRet;
    }

    /* Crop in device */
    DvppCropInputInfo cropInputData;
    FillCropInputData(cropInputData, dataDev, dataSize, cropOpt);

    DvppDataInfo output;
    output.width = cropOpt.cropArea.width;
    output.height = cropOpt.cropArea.height;
    errRet = g_cropProcessObj->CombineCropProcess(cropInputData, output, true);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to crop image, ret = " << errRet << ".";
        return errRet;
    }

    /* Get the pointer of device memory for cropped data, we cannot access the data until we copy it to host */
    std::shared_ptr<DvppDataInfo> cropImage = g_cropProcessObj->GetCropedImage();
    cropImageSize = cropImage->dataSize;
    if (cropImageSize == 0) {
        LogError << "Failed to crop Image, cropped image size is 0.";
        return APP_ERR_DVPP_CROP_FAIL;
    }

    /* Get host buffer and copy cropped data from device to host */
    cropImageHost = GetHostBuffer(cropImageSize);
    if (cropImageHost == nullptr) {
        LogError << "Failed to get image buffer for cropped image.";
        return APP_ERR_DVPP_CROP_FAIL;
    }

    /* Copy device image back to host */
    errRet = CopyDevImageToHost(cropImage, cropImageHost, cropImageSize);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to copy cropped image from device to host.";
        return errRet;
    }

    return APP_ERR_OK;
}

int main(int argc, const char *argv[])
{
    AtlasAscendLog::Log::LogInfoOn();

    CommandParser cmdOpt;
    /* Set the command line options */
    DefineCommandOption(cmdOpt);
    cmdOpt.ParseArgs(argc, argv);

    /* Get the user configuration for cropping */
    CropOption cropOpt;
    APP_ERROR errRet = GetCropOption(cmdOpt, cropOpt);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to get crop parameters, ret " << errRet << ".";
        return errRet;
    }

    /* Initialize the config parser module */
    ConfigParser config;
    ResourceInfo resourceInfo;
    errRet = InitConfig(config, resourceInfo);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to init config from file, ret = " << errRet << ".";
    }

    /* Init system resource */
    uint32_t devID = *(resourceInfo.deviceIds.begin());
    aclrtContext context;
    aclrtStream stream;
    errRet = CropSystemInit(devID, context, stream);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to init system resource for cropping, ret = " << errRet << ".";
        return errRet;
    }

    /* The next two will be assigned in the main routine */
    std::shared_ptr<void> cropImageHost;
    uint32_t cropImageSize;
    errRet = CropMainRoutine(cropOpt, stream, cropImageHost, cropImageSize);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to crop, ret = " << errRet << ".";
        CropSystemDestroy(devID, context, stream);
        return errRet;
    }

    /* Save cropped image to output file path set by user */
    errRet = SaveCropImage(cropOpt.outputImage, cropImageHost, cropImageSize);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to save crop Image to " << cropOpt.outputImage << ".";
        CropSystemDestroy(devID, context, stream);
        return errRet;
    }

    /* Destroy crop system, recycle resources. */
    errRet = CropSystemDestroy(devID, context, stream);
    if (errRet != APP_ERR_OK) {
        LogError << "Failed to destroy crop system resource, ret = "
                 << ".";
        return errRet;
    }
    LogInfo << "Crop image from " << cropOpt.inputImage << " successfully, save in " << cropOpt.outputImage << ".";
    return 0;
}
