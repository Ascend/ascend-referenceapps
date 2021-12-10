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

#ifndef INC_FACE_BLOCKING_MAP_H
#define INC_FACE_BLOCKING_MAP_H

#include <map>
#include <set>
#include <thread>
#include "DataType/DataType.h"
#include "ModuleBase/ModuleBase.h"

namespace ascendFaceRecognition {
class FaceBlockingMap {
public:
    static std::shared_ptr<FaceBlockingMap> &GetInstance(const int32_t &channelId);
    void Insert(const int32_t &id, std::shared_ptr<FaceObject> faceObject);
    std::shared_ptr<FaceObject> Get(const int32_t &id);
    std::shared_ptr<FaceObject> Pop(const int32_t &id);
    void Clear(const int32_t &id);
    std::vector<int32_t> Keys();
    size_t Size() const;
    FaceBlockingMap();
    ~FaceBlockingMap();

private:
    std::mutex mtx_ = {};
    std::map<int32_t, std::shared_ptr<FaceObject>> faceMap_ = {};
    std::set<int32_t> keys_ = {};
};
}
#endif