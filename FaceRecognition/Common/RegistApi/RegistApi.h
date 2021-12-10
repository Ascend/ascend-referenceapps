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

#ifndef INC_REGIST_API_H
#define INC_REGIST_API_H

#include <cstdio>
#include <functional>
#include "ErrorCode/ErrorCode.h"
#include "DataTrans/DataTrans.pb.h"

using CallbackFaceRegisterFunc = std::function<void(const int&, const std::string&)>;

namespace ascendFaceRecognition {
const uint32_t JPEG_HEADER_BYTE_LEN = 2;
const uint8_t JPEG_FIRST_BYTE = 0xff;
const uint8_t JPEG_SECOND_BYTE = 0xd8;

class RegistApi {
public:
    static std::shared_ptr<RegistApi> GetInstance();
    RegistApi();
    ~RegistApi();
    void RegistFace(std::string &name, const std::string &imgStr, CallbackFaceRegisterFunc callback);
    void HandleRegResult(std::shared_ptr<RegistResult> regResult);
    APP_ERROR RegistTheFace(std::string name, const std::string &imgStr, std::shared_ptr<DataTrans> dataTrans) const;

private:
    bool IsJpeg(const std::string &imgStr) const;

private:
    CallbackFaceRegisterFunc callback_ = nullptr;
};
}
#endif
