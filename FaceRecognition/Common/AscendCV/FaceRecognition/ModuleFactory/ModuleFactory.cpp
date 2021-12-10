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
#include "Log/Log.h"

#ifdef ASCEND_ACL_OPEN_VESION
#ifdef ACL_CTRL_CPU_MODE
#include "FaceDetection/FaceDetection.h"
#include "FaceFeature/FaceFeature.h"
#include "MOTEmbedding/MOTEmbedding.h"
#include "MOTConnection/MOTConnection.h"
#include "QualityEvaluation/QualityEvaluation.h"
#include "FaceAttribute/FaceAttribute.h"
#include "FaceLandmark/FaceLandmark.h"
#include "FaceSearch/FaceSearch.h"
#include "FaceStock/FaceStock.h"
#include "ImageDecoder/ImageDecoder.h"
#include "VideoDecoder/VideoDecoder.h"
#include "WarpAffine/WarpAffine.h"
#include "FaceResize/FaceResize.h"
#else // ACL_CTRL_CPU_MODE
#include "StreamPuller/StreamPuller.h"
#include "FrameAlign/FrameAlign.h"
#include "FaceDetailInfo/FaceDetailInfo.h"
#include "ImageReader/ImageReader.h"
#include "RegResultHandler/RegResultHandler.h"
#endif

#else
#include "StreamPuller/StreamPuller.h"
#include "VideoDecoderHost/VideoDecoder.h"
#include "FaceDetection/FaceDetection.h"
#include "FaceResize/FaceResize.h"
#include "MOTEmbedding/MOTEmbedding.h"
#include "MOTConnection/MOTConnection.h"
#include "QualityEvaluation/QualityEvaluation.h"
#include "WarpAffine/WarpAffine.h"
#include "FaceAttribute/FaceAttribute.h"
#include "FaceFeature/FaceFeature.h"
#include "FaceSearch/FaceSearch.h"
#include "ImageReader/ImageReader.h"
#include "ImageDecoder/ImageDecoder.h"
#include "FaceLandmark/FaceLandmark.h"
#include "FaceStock/FaceStock.h"
#include "FrameAlign/FrameAlign.h"
#include "FaceDetailInfo/FaceDetailInfo.h"
#include "RegResultHandler/RegResultHandler.h"
#include "VideoResize/VideoResize.h"
#endif // ASCEND_ACL_OPEN_VESION


namespace ascendFaceRecognition {
#ifdef ASCEND_ACL_OPEN_VESION
#ifdef ACL_CTRL_CPU_MODE
std::shared_ptr<ModuleBase> ModuleFactory::MakeModule(ModuleType moduleType)
{
    LogDebug << "ModuleFactory: begin to make module.";
    if (moduleType == MT_IMAGE_DECODER) {
        return std::make_shared<ImageDecoder>();
    }
    if (moduleType == MT_VIDEO_DECODER) {
        return std::make_shared<VideoDecoder>();
    }
    if (moduleType == MT_FACE_DETECTION) {
        return std::make_shared<FaceDetection>();
    }
    if (moduleType == MT_MOT_EMBEDDING) {
        return std::make_shared<MOTEmbedding>();
    }
    if (moduleType == MT_MOT_CONNECTION) {
        return std::make_shared<MOTConnection>();
    }
    if (moduleType == MT_FACE_LANDMARK) {
        return std::make_shared<FaceLandmark>();
    }
    if (moduleType == MT_WARP_AFFINE) {
        return std::make_shared<WarpAffine>();
    }
    if (moduleType == MT_FACE_FEATURE) {
        return std::make_shared<FaceFeature>();
    }
    if (moduleType == MT_FACE_STOCK) {
        return std::make_shared<FaceStock>();
    }
    if (moduleType == MT_FACE_SEARCH) {
        return std::make_shared<FaceSearch>();
    }
    if (moduleType == MT_QUALITY_EVALUATION) {
        return std::make_shared<QualityEvaluation>();
    }
    if (moduleType == MT_FACE_ATTRIBUTE) {
        return std::make_shared<FaceAttribute>();
    }
    if (moduleType == MT_FACE_RESIZE) {
        return std::make_shared<FaceResize>();
    }
    LogError << "ModuleFactory: invalid module type " << moduleType << ".";
    return NULL;
}
#else
std::shared_ptr<ModuleBase> ModuleFactory::MakeModule(ModuleType moduleType)
{
    LogDebug << "ModuleFactory: begin to make module.";
    if (moduleType == MT_STREAM_PULLER) {
        return std::make_shared<StreamPuller>();
    }
    if (moduleType == MT_FRAME_ALIGN) {
        return std::make_shared<FrameAlign>();
    }
    if (moduleType == MT_FACE_DETAIL_INFO) {
        return std::make_shared<FaceDetailInfo>();
    }
    if (moduleType == MT_IMAGE_READER) {
        return std::make_shared<ImageReader>();
    }
    if (moduleType == MT_REG_RESULT_HANDLER) {
        return std::make_shared<RegResultHandler>();
    }
    LogError << "ModuleFactory: invalid module type " << moduleType << ".";
    return nullptr;
}
#endif
#else
std::shared_ptr<ModuleBase> MakeSubModulePartOne(ModuleType moduleType)
{
    if (moduleType == MT_STREAM_PULLER) {
        return std::make_shared<StreamPuller>();
    }
    if (moduleType == MT_VIDEO_DECODER) {
        return std::make_shared<VideoDecoder>();
    }
    if (moduleType == MT_FACE_DETECTION) {
        return std::make_shared<FaceDetection>();
    }
    if (moduleType == MT_FACE_RESIZE) {
        return std::make_shared<FaceResize>();
    }
    if (moduleType == MT_MOT_EMBEDDING) {
        return std::make_shared<MOTEmbedding>();
    }
    if (moduleType == MT_MOT_CONNECTION) {
        return std::make_shared<MOTConnection>();
    }
    if (moduleType == MT_QUALITY_EVALUATION) {
        return std::make_shared<QualityEvaluation>();
    }
    if (moduleType == MT_WARP_AFFINE) {
        return std::make_shared<WarpAffine>();
    }
    if (moduleType == MT_FACE_ATTRIBUTE) {
        return std::make_shared<FaceAttribute>();
    }
    if (moduleType == MT_FACE_FEATURE) {
        return std::make_shared<FaceFeature>();
    }
    if (moduleType == MT_FACE_SEARCH) {
        return std::make_shared<FaceSearch>();
    }
    return nullptr;
}


std::shared_ptr<ModuleBase> MakeSubModulePartTwo(ModuleType moduleType)
{
    if (moduleType == MT_IMAGE_READER) {
        return std::make_shared<ImageReader>();
    }
    if (moduleType == MT_IMAGE_DECODER) {
        return std::make_shared<ImageDecoder>();
    }
    if (moduleType == MT_FACE_LANDMARK) {
        return std::make_shared<FaceLandmark>();
    }
    if (moduleType == MT_FACE_STOCK) {
        return std::make_shared<FaceStock>();
    }
    if (moduleType == MT_REG_RESULT_HANDLER) {
        return std::make_shared<RegResultHandler>();
    }
    if (moduleType == MT_FRAME_ALIGN) {
        return std::make_shared<FrameAlign>();
    }
    if (moduleType == MT_FACE_DETAIL_INFO) {
        return std::make_shared<FaceDetailInfo>();
    }
    if (moduleType == MT_VIDEO_RESIZE) {
        return std::make_shared<VideoResize>();
    }
    return nullptr;
}

std::shared_ptr<ModuleBase> ModuleFactory::MakeModule(ModuleType moduleType)
{
    LogDebug << "ModuleFactory: begin to make module.";
    auto module = MakeSubModulePartOne(moduleType);
    if (module.get() != nullptr) {
        return module;
    }
    module = MakeSubModulePartTwo(moduleType);
    if (module.get() != nullptr) {
        return module;
    }
    LogError << "ModuleFactory: invalid module type " << moduleType << ".";
    return nullptr;
}
#endif
} // namespace ascendFaceRecognition
