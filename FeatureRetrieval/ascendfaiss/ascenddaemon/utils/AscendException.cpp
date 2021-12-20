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

#include <ascenddaemon/utils/AscendException.h>
#include <sstream>
#include <securec.h>

namespace ascend {
const int MSG_LEN = 1024;
AscendException::AscendException(const std::string& m)
    : msg(m)
{
}

AscendException::AscendException(const std::string& m,
                                 const char* funcName,
                                 const char* file,
                                 int line)
{
    const int scaleSize = 2;
    int size = MSG_LEN;
    msg.resize(size, '\0');
    while (snprintf_s(&msg[0], msg.size(), msg.size(), "Error in %s at %s:%d: %s",
        funcName, file, line, m.c_str()) < 0) {
        size = size * scaleSize;
        msg.resize(size, '\0');
    }
}

const char* AscendException::what() const noexcept
{
    return msg.c_str();
}
}  // namespace ascend