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

#ifndef ASCEND_UTILS_INCLUDED
#define ASCEND_UTILS_INCLUDED

#include <future>

namespace faiss { 
namespace ascend {
#define CALL_PARALLEL_FUNCTOR(devices, threadPool, functor)                 \
    do {                                                                    \
        if (devices > 1) {                                                  \
            std::vector<std::future<void>> functorRets;                     \
            for (size_t i = 0; i < devices; i++) {                             \
                functorRets.emplace_back(threadPool->Enqueue(functor, i));  \
            }                                                               \
                                                                            \
            try {                                                           \
                for (auto & ret : functorRets) {                            \
                    ret.get();                                              \
                }                                                           \
            } catch (std::exception & e) {                                  \
                FAISS_THROW_FMT("%s.", e.what());                           \
            }                                                               \
        } else {                                                            \
            functor(0);                                                     \
        }                                                                   \
    } while (false)

#define CALL_PARALLEL_FUNCTOR_INDEXMAP(map, threadPool, functor)            \
    do {                                                                    \
        std::vector<std::future<void>> functorRets;                         \
        for (auto & index : (map)) {                                        \
            functorRets.emplace_back(                                       \
                threadPool->Enqueue(functor, index.first, index.second));   \
        }                                                                   \
                                                                            \
        try {                                                               \
            for (auto & ret : functorRets) {                                \
                ret.get();                                                  \
            }                                                               \
        } catch (std::exception & e) {                                      \
            FAISS_THROW_FMT("%s.", e.what());                               \
        }                                                                   \
    } while (false)
} // ascend
} // faiss
#endif
