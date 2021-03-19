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

#include "FileManager.h"
#include <ctime>
#include <string>

namespace {
    const int BUFFER_SIZE = 2048;
#ifndef _WIN32
    const mode_t DEFAULT_FILE_PERMISSION = 0077;
#endif
}

#ifndef _WIN32
mode_t SetFileDefaultUmask()
{
    return umask(DEFAULT_FILE_PERMISSION);
}

mode_t SetFileUmask(mode_t newUmask)
{
    return umask(newUmask);
}
#endif

/**
 * Check whether the file exists.
 *
 * @param filePath the file path we want to check
 * @return APP_ERR_OK if file exists, error code otherwise
 */
APP_ERROR ExistFile(const std::string &filePath)
{
    std::string resolvedPath;
    APP_ERROR ret = GetRealPath(filePath, resolvedPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
#ifndef _WIN32
    struct stat fileStat = {0};
    if (stat(resolvedPath.c_str(), &fileStat) == 0 && S_ISREG(fileStat.st_mode)) {
#else
    DWORD fileStat = GetFileAttributes((LPCSTR)resolvedPath.c_str());
    if ((fileStat == FILE_ATTRIBUTE_ARCHIVE) || (fileStat == FILE_ATTRIBUTE_NORMAL) ||
        (fileStat == FILE_ATTRIBUTE_DIRECTORY)) {
#endif
        return APP_ERR_OK;
    }
    return APP_ERR_COMM_FAILURE;
}

/**
 * Split the input path string with delimiter
 *
 * @param str path string
 * @param delimiters a set of delimiter
 * @return the vector of splitted path
 */
std::vector<std::string> SplitPath(const std::string &str, const std::set<char> delimiters)
{
    std::vector<std::string> result;
    std::string temp = str;
    auto start = temp.begin();
    auto it = temp.begin();
    for (; it != temp.end(); ++it) {
        if (delimiters.find(*it) != delimiters.end()) {
            if (it != start) {
                result.emplace_back(start, it);
            } else {
                result.emplace_back("");
            }
            start = it + 1;
        }
    }
    result.emplace_back(start, it);
    return result;
}

/**
 * Create a directory
 *
 * @param dirPath the directory we want to create
 * @return APP_ERR_OK if create success, error code otherwise
 */
APP_ERROR CreateDir(const std::string &dirPath)
{
#ifndef _WIN32
    SetFileDefaultUmask();
    if (dirPath.length() > PATH_MAX) {
        LogError << dirPath << "is larger than " << std::to_string(PATH_MAX) << ".";
        return APP_ERR_COMM_NO_EXIST;
    }
#else
    if (dirPath.length() > MAX_PATH) {
        LogError << dirPath << "is larger than " << std::to_string(MAX_PATH) << ".";
    }
    return APP_ERR_COMM_NO_EXIST;
#endif

    // Check the write authority of directory, if not exist, create it
    int dirExist = access(dirPath.c_str(), W_OK);
    if (-1 == dirExist) {
#ifdef _WIN32
        if (_mkdir(dirPath.c_str()) == -1) {
#else
        if (mkdir(dirPath.c_str(), S_IRUSR | S_IWUSR | S_IXUSR) == -1) {
#endif
            return APP_ERR_COMM_NO_EXIST;
        }
    }
    return APP_ERR_OK;
}

/**
 * Create directory recursively
 *
 * @param file target directory to create
 */
void CreateDirRecursively(const std::string &file)
{
    CreateDirRecursivelyByFile(file);
    if (access(file.c_str(), 0) != 0) {
#ifndef _WIN32
        int result = mkdir(file.c_str(), S_IRUSR | S_IWUSR | S_IXUSR); // for linux
#else
        int result = _mkdir(file.c_str());
#endif
        if (result < 0) {
            LogError << "mkdir logs file " << file << " fail.";
            return;
        }
    }
}

/**
 * Create directory recursively by file
 *
 * @param file target file to create
 */
void CreateDirRecursivelyByFile(const std::string &file)
{
    size_t pos = file.rfind('/'); // for linux
    std::string filePath = file.substr(0, pos);
    if (access(filePath.c_str(), 0) != 0) {
        CreateDirRecursivelyByFile(filePath);
#ifndef _WIN32
        int result = mkdir(filePath.c_str(), S_IRUSR | S_IWUSR | S_IXUSR); // for linux
#else
        int result = _mkdir(filePath.c_str());
#endif
        if (result < 0) {
            LogError << "mkdir logs file " << filePath << " fail.";
            return;
        }
    }
}

/**
 * Read a file, store it into the RawData structure
 *
 * @param filePath file to read to
 * @param fileData RawData structure to store in
 * @return APP_ERR_OK if create success, error code otherwise
 */
APP_ERROR ReadFile(const std::string &filePath, RawData &fileData)
{
    std::string resolvedPath;
    APP_ERROR ret = GetRealPath(filePath, resolvedPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    // Open file with reading mode
    FILE *fp = fopen(resolvedPath.c_str(), "rb");
    if (fp == nullptr) {
        LogError << "Failed to open file";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // Get the length of input file
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    // If file not empty, read it into FileInfo and return it
    if (fileSize > 0) {
        fileData.lenOfByte = fileSize;
        fileData.data = std::make_shared<uint8_t>();
        fileData.data.reset(new uint8_t[fileSize], std::default_delete<uint8_t[]>());
        size_t readRet = fread(fileData.data.get(), 1, fileSize, fp);
        if (readRet <= 0) {
            fclose(fp);
            return APP_ERR_COMM_READ_FAIL;
        }
        fclose(fp);
        return APP_ERR_OK;
    }
    fclose(fp);
    return APP_ERR_COMM_FAILURE;
}

/**
 * Read a binary file, store the data into a uint8_t array
 *
 * @param fileName the file for reading
 * @param buffShared a shared pointer to a uint8_t array for storing file
 * @param buffLength the length of the array
 * @return APP_ERR_OK if create success, error code otherwise
 */
APP_ERROR ReadBinaryFile(const std::string &fileName, std::shared_ptr<uint8_t> &buffShared, int &buffLength)
{
    // read binary file
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

/**
 * Read a file with specified offset
 * Only used in Jpegd
 *
 * @param fileName the file for reading
 * @param fileData RawData structure to store in
 * @param offset offset for file
 * @return APP_ERR_OK if create success, error code otherwise
 */
APP_ERROR ReadFileWithOffset(const std::string &fileName, RawData &fileData, const uint32_t offset)
{
    std::string resolvedPath;
    APP_ERROR ret = GetRealPath(fileName, resolvedPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    // Open file with reading mode
    FILE *fp = fopen(resolvedPath.c_str(), "rb");
    if (fp == nullptr) {
        LogError << "Failed to open file";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // Get the length of input file
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    // If file not empty, read it into FileInfo and return it
    if (fileSize > 0) {
        fileData.lenOfByte = fileSize;
        fileData.data = std::make_shared<uint8_t>();
        fileData.data.reset(new uint8_t[fileSize + offset], std::default_delete<uint8_t[]>());
        size_t readRet = fread(fileData.data.get(), 1, fileSize, fp);
        if (readRet <= 0) {
            fclose(fp);
            return APP_ERR_COMM_READ_FAIL;
        }
        fclose(fp);
        return APP_ERR_OK;
    }
    fclose(fp);
    return APP_ERR_COMM_FAILURE;
}

APP_ERROR GetRealPath(const std::string &srcPath, std::string &resolvedPath)
{
    // Get the absolute path of input file
#ifndef _WIN32
    char path[PATH_MAX + 1] = {0};
    if ((strlen(srcPath.c_str()) > PATH_MAX) || (realpath(srcPath.c_str(), path) == nullptr)) {
#else
    #pragma comment(lib, "Shlwapi.lib")
    char path[MAX_PATH + 1] = {0};
    if ((strlen(srcPath.c_str()) > MAX_PATH) || (_fullpath(path, srcPath.c_str(), MAX_PATH) == nullptr)) {
#endif
        LogError << "Failed to get canonicalize path";
        return APP_ERR_COMM_NO_EXIST;
    }
    resolvedPath = path;
    return APP_ERR_OK;
}

/**
 * Get the extension name of input file
 *
 * @param filePath the file for reading extension name
 * @return extension name of the file
 */
std::string GetExtension(const std::string &filePath)
{
    std::set<char> delims { '.' };
    std::vector<std::string> path = SplitPath(filePath, delims);
    return path[path.size() - 1];
}

/**
 * Get file canonical name
 *
 * @param filePath absolute path of the target file
 * @return filename of the file
 */
std::string GetName(const std::string &filePath)
{
    std::set<char> delims { '/' };
    std::vector<std::string> path = SplitPath(filePath, delims);
    return (path.size() < 1) ? "" : path[path.size() - 1];
}

/**
 * Get the Parent of input file
 *
 * @param filePath file for looking for parent
 * @return parent of the file
 */
std::string GetParent(const std::string &filePath)
{
    std::set<char> delims { '/' };
    std::vector<std::string> path = SplitPath(filePath, delims);
    return (path.size() < TWO) ? "" : path[path.size() - TWO];
}

/**
 * Change the current directory
 *
 * @param dir target directory to change to
 * @return APP_ERR_OK if create success, error code otherwise
 */
APP_ERROR ChangeDir(const std::string &dir)
{
    std::string resolvedPath;
    APP_ERROR ret = GetRealPath(dir, resolvedPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }
#ifndef _WIN32
    char path[PATH_MAX + 1] = {0};
    resolvedPath.copy(path, resolvedPath.length());
    char *dName = dirname(path);
    if (dName == nullptr) {
        return APP_ERR_COMM_NO_EXIST;
    }
#else
    char path[MAX_PATH + 1] = {0};
    resolvedPath.copy(path, resolvedPath.length());
    if (!PathRemoveFileSpecA(path)) {
        return APP_ERR_COMM_NO_EXIST;
    }
    char *dName = path;
#endif
    if (chdir(dName) != 0) {
        return APP_ERR_COMM_NO_EXIST;
    }
    return APP_ERR_OK;
}

/**
 * Append stream to file
 *
 * @param fileName to append to
 * @param stream content of string
 * @param streamLength length of string
 */
void SaveFileAppend(const std::string &fileName, const std::string &stream, const int streamLength)
{
    LogDebug << "saving binary file by app: fileName=" << fileName << ", streamLength=" << streamLength;
    std::ofstream outfile(fileName, std::ios::app | std::ofstream::binary);
    outfile.write(stream.c_str(), streamLength);
    outfile.close();
}

/**
 * Overwrite a file with stream
 *
 * @param fileName to overwrite to
 * @param stream content of string
 * @param streamLength length of string
 */
void SaveFileOverwrite(const std::string &fileName, const std::string &stream, const int streamLength)
{
    LogDebug << "Saving binary file by over write: fileName=" << fileName << ", streamLength=" << streamLength;
    std::ofstream outfile(fileName, std::ios::out | std::ofstream::binary);
    outfile.write(stream.c_str(), streamLength);
    outfile.close();
}

/**
 * Copy file
 *
 * @param srcFile from source
 * @param destFile to destination
 */
void CopyFile(const std::string &srcFile, const std::string &destFile)
{
    std::ifstream in(srcFile, std::ios::binary);
    if (!in) {
        LogError << "Failed to get source file, it may be not exists. srcFile=" << srcFile;
        return;
    }
    std::ofstream out(destFile, std::ios::binary);
    if (!out) {
        LogError << "Failed to save destination file. destFile=" << destFile;
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
}

/**
 * Save file with timestamp under specified folder
 *
 * @param dataBuffer  buffer of file
 * @param bufferSize   buffer size
 * @param folderName   specified folder will be created if it not existed
 * @param fileName     file name without suffix, the finally name will append time stamp to it
 * @param fileSuffix   suffix name of file
 */
APP_ERROR SaveFileWithTimeStamp(std::shared_ptr<void> dataBuffer, uint32_t bufferSize, std::string folderName,
                                std::string fileName, std::string fileSuffix)
{
#ifndef _WIN32
    SetFileDefaultUmask();
#endif
    APP_ERROR ret;
    if (folderName.length() != 0) {
        ret = CreateDir(folderName);
        if (ret != APP_ERR_OK) {
            return ret;
        }
    }

    // Result file name use the time stamp as a suffix
    std::string timeString;
    GetCurTimeString(timeString);

    // Create file under folderName directory
    std::stringstream resultPathName;
    if (folderName.length() == 0) {
        resultPathName << "./" << fileName << "_" << timeString << fileSuffix;
    } else {
        resultPathName << folderName << "/" << fileName << "_" << timeString << fileSuffix;
    }
    std::string resolvedPath;
    ret = GetRealPath(resultPathName.str(), resolvedPath);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    FILE *fp = fopen(resolvedPath.c_str(), "wb");
    if (fp == nullptr) {
        LogError << "Failed to open file";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    uint32_t result = fwrite(dataBuffer.get(), 1, bufferSize, fp);
    if (result != bufferSize) {
        LogError << "Failed to write file";
        fclose(fp);
        return APP_ERR_COMM_WRITE_FAIL;
    }
    LogInfo << "Write result to file successfully";
    uint32_t ff = fflush(fp);
    if (ff != 0) {
        LogError << "Failed to fflush file";
        fclose(fp);
        return APP_ERR_COMM_DESTORY_FAIL;
    }
    uint32_t fc = fclose(fp);
    if (fc != 0) {
        LogError << "Failed to fclose file";
        return APP_ERR_COMM_DESTORY_FAIL;
    }
    return APP_ERR_OK;
}

/**
 * Convert the current time to the format "%Y%m%d%H%M%S"
 *
 * @param timeString buffer to save the time string with format "%Y%m%d%H%M%S"
 */
void GetCurTimeString(std::string &timeString)
{
    // Result file name use the time stamp as a suffix
    const int timeZoneDiff = 28800; // 8 hour time difference
    const int timeStringSize = 32;
    char timeStr[timeStringSize] = {0};
    time_t tmValue = time(nullptr) + timeZoneDiff;
    struct tm tmStruct = {0};
#ifdef _WIN32
    if (0 == gmtime_s(&tmStruct, &tmValue)) {
#else
    if (nullptr != gmtime_r(&tmValue, &tmStruct)) {
#endif
        strftime(timeStr, sizeof(timeStr), "%Y%m%d%H%M%S", &tmStruct);
    }
    timeString = timeStr;
    return;
}