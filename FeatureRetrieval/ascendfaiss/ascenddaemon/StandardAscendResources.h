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

#ifndef ASCEND_STANDARDASCENDRESOURCES_INCLUDED
#define ASCEND_STANDARDASCENDRESOURCES_INCLUDED

#include <ascenddaemon/utils/AscendStackMemory.h>
#include <ascenddaemon/utils/AscendUtils.h>
#include <vector>
#include <memory>
#include <limits>

namespace ascend {
class StandardAscendResources {
public:
    StandardAscendResources(std::string modelPath = "modelpath");

    ~StandardAscendResources();

    /// Disable allocation of temporary memory; all temporary memory
    /// requests will call aclrtMalloc / aclrtFree at the point of use
    void noTempMemory();

    /// Specify that we wish to use a certain fixed size of memory on as
    /// temporary memory. This is the upper bound for the Ascend Device
    /// memory that we will reserve.
    /// To avoid any temporary memory allocation, pass 0.
    void setTempMemory(size_t size);

    void setDefaultTempMemory();

    /// Enable or disable the warning about not having enough temporary memory
    /// when aclrtMalloc gets called
    void setAscendMallocWarning(bool flag);

    size_t getResourceSize() const
    {
        return tempMemSize;
    }

public:
    /// Initialize resources
    void initialize();

    aclrtStream getDefaultStream();

    void syncDefaultStream();

    std::vector<aclrtStream> getAlternateStreams();

    AscendMemory& getMemoryManager();

private:
    /// Have Ascend resources been initialized yet?
    bool isInitialized() const;

    size_t getDefaultTempMemSize(size_t requested);

private:
    /// Our default stream that work is ordered on
    aclrtStream defaultStream;

    /// Other streams we can use
    std::vector<aclrtStream> alternateStreams;

    /// Another option is to use a specified amount of memory
    size_t tempMemSize;
};
}  // namespace ascend

#endif  // ASCEND_STANDARDASCENDRESOURCES_INCLUDED
