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

#include "VdecUtils.h"
#include "FileManager/FileManager.h"


/*
 * @description: Save YUV file into disk
 * @param: fileName specifies the output file name
 * @param: dataDev specifies the data in device
 * @param: dataSize specifies the size of file data
 * @return: true if success, false if failure
 */
bool VdecUtils::WriteToFile(const std::string &fileName, const std::unique_ptr<void, decltype(acldvppFree)*>& dataDev,
    uint32_t dataSize)
{
    std::unique_ptr<char[]> dataHost(new char[dataSize]);
    if (dataHost == nullptr) {
        LogError << "malloc host data buffer failed. dataSize= " << dataSize << "\n";
        return false;
    }

    // copy output to host memory
    auto aclRet = aclrtMemcpy(dataHost.get(), dataSize, dataDev.get(), dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        LogError << "acl memcpy data to host failed, dataSize= " << dataSize << "ret= " << aclRet << "\n";
        return false;
    }
    SetFileDefaultUmask();
    FILE *outFileFp = fopen(fileName.c_str(), "wb+");
    if (outFileFp == nullptr) {
        LogError << "fopen out file failed " << fileName;
        return false;
    }

    bool ret = true;
    size_t writeRet = fwrite(dataHost.get(), 1, dataSize, outFileFp);
    if (writeRet != dataSize) {
        LogError << "need write " << dataSize << "bytes to " << fileName << "but only write " << writeRet;
        ret = false;
    }
    fflush(outFileFp);
    fclose(outFileFp);
    return ret;
}

/*
 * @description: Check if the directory exists, create it if it does not exist
 * @param: foldName specifies the output directory
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR VdecUtils::CheckFolder(const std::string &foldName)
{
    LogInfo << "start check result fold:" << foldName;
    if (access(foldName.c_str(), 0) == -1) {
        int flag = mkdir(foldName.c_str(),  S_IRUSR | S_IWUSR | S_IXUSR);
        if (flag == 0) {
            LogInfo << "make successfully.";
        } else {
            LogInfo << "make errorly.";
            return APP_ERR_ACL_FAILURE;
        }
    }
    LogInfo << "check result success, fold exist";
    return APP_ERR_OK;
}
