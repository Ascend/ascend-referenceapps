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

#include "ModuleFactory.h"
#include "FaceFeature/FaceFeature.h"
#include "FaceDetectionLandmark/FaceDetectionLandmark.h"
#include "FaceSearch/FaceSearch.h"
#include "FaceStock/FaceStock.h"
#include "ImageDecoder/ImageDecoder.h"
#include "Log/Log.h"
#include "WarpAffine/WarpAffine.h"

#ifdef HOST_CPU_SOLUTION
#include "JpegReader/JpegReader.h"
#else
#ifdef  CTRL_CPU_SOLUTION
#else
#include "JpegReader/JpegReader.h"
#endif
#endif

namespace ascendFaceRecognition {
std::shared_ptr<ModuleBase> ModuleFactory::MakeModule(ModuleType moduleType)
{
    LogInfo << "ModuleFactory: begin to make module.";

    if (moduleType < 0 || moduleType >= MT_BOTTOM) {
        LogFatal << "ModuleFactory: invalid module type.";
        return NULL;
    }

    switch (moduleType) {
#ifdef HOST_CPU_SOLUTION
        case MT_JPEG_READER:
            return std::make_shared<JpegReader>();
#else
#ifdef  CTRL_CPU_SOLUTION
#else
        case MT_JPEG_READER:
            return std::make_shared<JpegReader>();
#endif
#endif
        case MT_IMAGE_DECODER:
            return std::make_shared<ImageDecoder>();
        case MT_FACE_DETECTION_LANDMARK:
            return std::make_shared<FaceDetectionLandmark>();
        case MT_WARP_AFFINE:
            return std::make_shared<WarpAffine>();
        case MT_FACE_FEATURE:
            return std::make_shared<FaceFeature>();
        case MT_FACE_STOCK:
            return std::make_shared<FaceStock>();
        case MT_FACE_SEARCH:
            return std::make_shared<FaceSearch>();
        default:
            LogError << "ModuleFactory: invalid module type " << moduleType << ".";
            return NULL;
    }
}
} // namespace ascendFaceRecognition
