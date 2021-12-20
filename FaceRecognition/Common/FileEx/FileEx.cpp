/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1 Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2 Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3 Neither the names of the copyright holders nor the names of the
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * Description:
 * Date: 2020/04/26
 * History:
 * [2020-04-26]:
 */

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

#include "FileEx/FileEx.h"
#include "Log/Log.h"

std::mutex FileEx::fileMutex_;
namespace {
const int BUFFER_SIZE = 2048;
}

void FileEx::SaveBinaryFileApp(const std::string &fileName, const char *stream, const int streamLength)
{
    LogDebug << "saving binary file by app: fileName=" << fileName << ", streamLength=" << streamLength;
    std::lock_guard<std::mutex> locker(fileMutex_);
    std::ofstream outfile(fileName, std::ios::app | std::ofstream::binary);
    outfile.write(stream, streamLength);
    outfile.close();
    return;
}

void FileEx::SaveBinaryFileApp(const std::string &fileName, uint8_t *stream, const int streamLength)
{
    char *stream_s = reinterpret_cast<char *>(stream);
    FileEx::SaveBinaryFileApp(fileName, stream_s, streamLength);
    return;
}

void FileEx::SaveBinaryFileOverwrite(const std::string &fileName, const char *stream, const int streamLength)
{
    LogDebug << "saving binary file by over write: fileName=" << fileName << ", streamLength=" << streamLength;
    mode_t new_umask = 0077; // 0077 is for S_IRUSR | S_IWUSR
    mode_t old_umask = umask(new_umask);
    std::lock_guard<std::mutex> locker(fileMutex_);
    std::ofstream outfile(fileName, std::ios::out | std::ofstream::binary);
    outfile.write(stream, streamLength);
    outfile.close();
    umask(old_umask);
    return;
}

void FileEx::SaveBinaryFileOverwrite(const std::string &fileName, uint8_t *stream, const int streamLength)
{
    char *stream_s = reinterpret_cast<char *>(stream);
    SaveBinaryFileOverwrite(fileName, stream_s, streamLength);
    return;
}

void FileEx::SaveFloatFileOverwrite(const std::string &fileName, const float *data, const int dataLength)
{
    LogDebug << "saving float data file by over write: fileName=" << fileName << ", dataLength=" << dataLength;
    std::lock_guard<std::mutex> locker(fileMutex_);
    mode_t new_umask = 0077; // 0077 is for S_IRUSR | S_IWUSR
    mode_t old_umask = umask(new_umask);
    std::ofstream outfile(fileName, std::ios::out);
    for (int i = 0; i < dataLength; i++) {
        outfile << data[i] << std::endl;
    }
    outfile.close();
    umask(old_umask);
    return;
}

APP_ERROR FileEx::ReadBinaryFile(const std::string &fileName, std::shared_ptr<uint8_t> &buffShared, int &buffLength)
{
    // read the file content.
    std::ifstream inFile(fileName, std::ios::in | std::ios::binary);
    if (!inFile) {
        LogError << "FaceFeatureLib: read file " << fileName << " fail.";
        return APP_ERR_COMM_READ_FAIL;
    }

    // get length of file:
    inFile.seekg(0, inFile.end);
    buffLength = inFile.tellg();
    inFile.seekg(0, inFile.beg);

    auto tempShared = std::make_shared<uint8_t>();
    tempShared.reset(new uint8_t[buffLength], std::default_delete<uint8_t[]>());

    inFile.read((char *)tempShared.get(), buffLength);
    inFile.close();
    buffShared = tempShared;

    LogDebug << "read file: fileName=" << fileName << ", size=" << buffLength << ".";

    return APP_ERR_OK;
}

APP_ERROR FileEx::ReadBinaryFile(const std::string &fileName, std::shared_ptr<uint8_t> &buffShared,
    uint32_t &buffLength)
{
    int dataLen = 0;
    APP_ERROR ret = ReadBinaryFile(fileName, buffShared, dataLen);
    buffLength = static_cast<uint32_t>(dataLen);

    return ret;
}

// only for file to file, not dir
void FileEx::CopyFile(const std::string &sourFile, const std::string &destFile)
{
    std::ifstream in(sourFile, std::ios::binary);
    if (!in) {
        LogError << "sourFile is not exsit, sourFile=" << sourFile;
        return;
    }
    std::ofstream out(destFile, std::ios::binary);
    if (!out) {
        LogError << "save destFile fail, destFile=" << destFile;
        in.close();
        return;
    }
    char flush[BUFFER_SIZE];
    while (!in.eof()) {
        in.read(flush, BUFFER_SIZE);
        out.write(flush, in.gcount());
    }
    out.close();
    in.close();
    return;
}

void FileEx::MakeDirRecursion(const std::string &file)
{
    FileEx::MakeDirRecursion_(file);
    if (access(file.c_str(), 0) != 0) {
        int result = mkdir(file.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO); // for linux
        if (result < 0) {
            LogError << "mkdir logs file " << file << " fail.";
            return;
        }
    }
    return;
}

// file contains file path and file
void FileEx::MakeDirRecursion_(const std::string &file)
{
    size_t pos = file.rfind("/"); // for linux
    std::string filePath = file.substr(0, pos);
    if (access(filePath.c_str(), 0) != 0) {
        FileEx::MakeDirRecursion_(filePath);
        int result = mkdir(filePath.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO); // for linux
        if (result < 0) {
            LogError << "mkdir logs file " << filePath << " fail.";
            return;
        }
    }
    return;
}

void FileEx::SaveFileApp(const std::string &fileName, std::string context)
{
    std::lock_guard<std::mutex> locker(fileMutex_);
    std::ofstream fs(fileName, std::ios::app);
    if (!fs) {
        LogError << "open file " << fileName << " fail";
        return;
    }
    fs << context;
    fs.close();
}
