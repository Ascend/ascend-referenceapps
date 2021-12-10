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

#ifndef ASCEND_EXCEPTION_INCLUDED
#define ASCEND_EXCEPTION_INCLUDED

#include <exception>
#include <string>

namespace ascend {
// Base class for Ascend exceptions
class AscendException : public std::exception {
public:
    explicit AscendException(const std::string& msg);

    AscendException(const std::string& msg,
                    const char* funcName,
                    const char* file,
                    int line);

    // from std::exception
    const char* what() const noexcept override;

    std::string msg;
};
}  // namespace ascend

#endif
