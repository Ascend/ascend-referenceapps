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

#include <ascenddaemon/utils/MemorySpace.h>
#include <ascenddaemon/utils/AscendAssert.h>
#include "acl/acl.h"

namespace ascend {
void AllocMemorySpaceV(MemorySpace space, void **p, size_t size)
{
    switch (space) {
        case MemorySpace::DEVICE: {
            aclError err = aclrtMalloc(p, size, ACL_MEM_MALLOC_NORMAL_ONLY);

            ASCEND_THROW_IF_NOT_FMT(err == ACL_ERROR_NONE,
                                    "failed to aclrtMalloc %zu bytes (error %d)",
                                    size, (int)err);
            break;
        }
        case MemorySpace::DEVICE_HUGEPAGE: {
            aclError err = aclrtMalloc(p, size, ACL_MEM_MALLOC_HUGE_FIRST);

            ASCEND_THROW_IF_NOT(err == ACL_ERROR_NONE);
            break;
        }
        default:
            ASCEND_THROW_FMT("unknown MemorySpace %d", (int)space);
            break;
    }
}

void FreeMemorySpace(MemorySpace space, void *p)
{
    switch (space) {
        case MemorySpace::DEVICE:
        case MemorySpace::DEVICE_HUGEPAGE: {
            aclError err = aclrtFree(p);

            ASCEND_THROW_IF_NOT_FMT(err == ACL_ERROR_NONE,
                                    "Failed to aclrtFree pointer %p (error %d)",
                                    p, (int)err);
            break;
        }
        default:
            ASCEND_THROW_FMT("unknown MemorySpace %d", (int)space);
            break;
    }
}
}  // namespace ascend