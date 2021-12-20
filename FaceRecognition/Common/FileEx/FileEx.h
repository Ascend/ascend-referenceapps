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
 * Description: Ascend File Process base on C++ file process
 * Date: 2020/04/26
 * History:
 * [2020-04-26]:
 */

#ifndef FILE_EX_H
#define FILE_EX_H

#include <memory>
#include <mutex>

#include "ErrorCode/ErrorCode.h"

class FileEx {
public:
    FileEx() {};
    ~FileEx() {};
    static void SaveBinaryFileApp(const std::string &fileName, const char *stream, const int streamLength);
    static void SaveBinaryFileApp(const std::string &fileName, uint8_t *stream, const int streamLength);
    static void SaveBinaryFileOverwrite(const std::string &fileName, const char *stream, const int streamLength);
    static void SaveBinaryFileOverwrite(const std::string &fileName, uint8_t *stream, const int streamLength);
    static void SaveFloatFileOverwrite(const std::string &fileName, const float *data, const int dataLength);

    static APP_ERROR ReadBinaryFile(const std::string &fileName, std::shared_ptr<uint8_t> &buffShared, int &buffLength);
    static APP_ERROR ReadBinaryFile(const std::string &fileName, std::shared_ptr<uint8_t> &buffShared,
        uint32_t &buffLength);

    static void CopyFile(const std::string &sourFile, const std::string &destFile);
    static void MakeDirRecursion(const std::string &file);

    static void SaveFileApp(const std::string &fileName, std::string context);

private:
    static std::mutex fileMutex_;

    static void MakeDirRecursion_(const std::string &file);
};

#endif
