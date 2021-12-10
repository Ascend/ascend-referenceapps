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

#ifndef ASCEND_CLONER_OPTIONS_INCLUDED
#define ASCEND_CLONER_OPTIONS_INCLUDED

namespace faiss {
namespace ascend {
struct AscendClonerOptions {
    AscendClonerOptions();

    long reserveVecs;
    bool verbose;
    int resourceSize;
};
}  // namespace ascend
}  // namespace faiss

#endif  // ASCEND_CLONER_OPTIONS_INCLUDED
