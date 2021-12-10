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

#ifndef INC_BATCH_YOLOV3_H
#define INC_BATCH_YOLOV3_H

#include <vector>
#include <cmath>
#include "Log/Log.h"

namespace {
const uint32_t MAX_NUM_OBJECT = 200;
const uint32_t NET_CHANNEL_COUNT = 3;
const uint32_t ANCHOR_NUMBER = 5;
const float CONFIDENCE_THRESHOLD = 0.4;
const float NMS_THRESHOLD = 0.45;

// grid stride by level type
const uint32_t GRID_STRIDE_LEVEL_0 = 64;
const uint32_t GRID_STRIDE_LEVEL_1 = 32;
const uint32_t GRID_STRIDE_LEVEL_2 = 16;

const uint32_t LEVEL_TYPE_0 = 0;
const uint32_t LEVEL_TYPE_1 = 1;
const uint32_t LEVEL_TYPE_2 = 2;

const float SMALL_EPSINON = 1e-6;

const uint32_t WIDTH_OFFSET = 2;
const uint32_t HEIGHT_OFFSET = 3;
const uint32_t SCORE_OFFSET = 4;
const float HALF_OF_LENGHT_SCALE = 0.5f;
}

struct YoloBox {
    float x = 0.f;
    float y = 0.f;
    float w = 0.f;
    float h = 0.f;
    float score = 0.f;
    uint32_t classId = 0;
};

struct YoloParameter {
    uint32_t scaleAnchorNumber = 0;
    uint32_t classNumber = 0;
    uint32_t netWidth = 0;
    uint32_t netHeight = 0;
    uint32_t batch = 0;
};

static inline bool CompareBox(const YoloBox &box1, const YoloBox &box2)
{
    return (box1.score > box2.score);
}

class Yolov3Batch {
public:
    explicit Yolov3Batch(const std::vector<uint32_t> &anchors, const YoloParameter &parameter)
    {
        batchSize_ = parameter.batch;
        netChannel_ = NET_CHANNEL_COUNT;
        netHeight_ = parameter.netHeight;
        netWidth_ = parameter.netWidth;
        classNumber_ = parameter.classNumber;
        anchorChannel_ = ANCHOR_NUMBER + parameter.classNumber;
        anchorNumber_ = parameter.scaleAnchorNumber;
        confidenceThresh_ = CONFIDENCE_THRESHOLD;
        nmsThresh_ = NMS_THRESHOLD;
        anchors_ = anchors;
        inputSize_ = batchSize_ * netHeight_ * netWidth_ * netChannel_;
    };

    ~Yolov3Batch() {};

    void PostProcess(const std::vector<void *> &outputs, const uint32_t &imgWidth, const uint32_t &imgHeight,
        std::vector<std::vector<YoloBox>> &detectBBoxes) const
    {
        float *tensor[NET_CHANNEL_COUNT];
        for (uint32_t i = 0; i < NET_CHANNEL_COUNT; ++i) {
            tensor[i] = (float *)outputs[i];
        }

        std::vector<std::vector<YoloBox>> bboxes;
        bboxes.clear();
        bboxes.resize(batchSize_);
        detectBBoxes.clear();
        detectBBoxes.resize(batchSize_);

        for (uint32_t batchIdx = 0; batchIdx < batchSize_; batchIdx++) {
            for (uint32_t i = 0; i < NET_CHANNEL_COUNT; ++i) {
                GetScaleResult(tensor[i], i, bboxes, batchIdx);
            }
            uint32_t count = 0;
            std::vector<int> masks;
            sort(bboxes[batchIdx].begin(), bboxes[batchIdx].end(), CompareBox);
            ApplyNMS(bboxes[batchIdx], masks, nmsThresh_, MAX_NUM_OBJECT);

            for (uint32_t i = 0; i < bboxes[batchIdx].size() && (count < MAX_NUM_OBJECT); i++) {
                if (masks[i] == 0) {
                    YoloBox box;
                    box.x = std::min(std::max(bboxes[batchIdx][i].x * float(imgWidth), 0.f), float(imgWidth - 1));
                    box.y = std::min(std::max(bboxes[batchIdx][i].y * float(imgHeight), 0.f), float(imgHeight - 1));
                    float width = std::max(float(imgWidth) - box.x - 1.f, 0.f);
                    float height = std::max(float(imgHeight) - box.y - 1.f, 0.f);
                    box.w = std::min(std::max(bboxes[batchIdx][i].w * float(imgWidth), 0.f), width);
                    box.h = std::min(std::max(bboxes[batchIdx][i].h * float(imgHeight), 0.f), height);
                    box.score = bboxes[batchIdx][i].score;
                    box.classId = 0;
                    detectBBoxes[batchIdx].push_back(box);
                    count++;
                }
            }
        }
    }

    void PostProcess(const std::vector<void *> &outputs, const std::vector<uint32_t> &imgWidths,
        const std::vector<uint32_t> &imgHeights, std::vector<std::vector<YoloBox>> &detectBBoxes) const
    {
        float *tensor[NET_CHANNEL_COUNT];
        for (uint32_t i = 0; i < NET_CHANNEL_COUNT; ++i) {
            tensor[i] = (float *)outputs[i];
        }

        std::vector<std::vector<YoloBox>> bboxes;
        bboxes.clear();
        bboxes.resize(batchSize_);
        detectBBoxes.clear();
        detectBBoxes.resize(batchSize_);

        for (uint32_t batchIdx = 0; batchIdx < batchSize_; batchIdx++) {
            for (uint32_t i = 0; i < NET_CHANNEL_COUNT; ++i) {
                GetScaleResult(tensor[i], i, bboxes, batchIdx);
            }
            uint32_t count = 0;
            std::vector<int> masks;
            sort(bboxes[batchIdx].begin(), bboxes[batchIdx].end(), CompareBox);
            ApplyNMS(bboxes[batchIdx], masks, nmsThresh_, MAX_NUM_OBJECT);

            uint32_t imgWidth = imgWidths[batchIdx];
            uint32_t imgHeight = imgHeights[batchIdx];
            for (uint32_t i = 0; i < bboxes[batchIdx].size() && (count < MAX_NUM_OBJECT); i++) {
                if (masks[i] == 0) {
                    YoloBox box;
                    box.x = std::min(std::max(bboxes[batchIdx][i].x * float(imgWidth), 0.f), float(imgWidth - 1));
                    box.y = std::min(std::max(bboxes[batchIdx][i].y * float(imgHeight), 0.f), float(imgHeight - 1));

                    float width = std::max(float(imgWidth) - box.x - 1.f, 0.f);
                    float height = std::max(float(imgHeight) - box.y - 1.f, 0.f);

                    box.w = std::min(std::max(bboxes[batchIdx][i].w * float(imgWidth), 0.f), width);
                    box.h = std::min(std::max(bboxes[batchIdx][i].h * float(imgHeight), 0.f), height);

                    box.score = bboxes[batchIdx][i].score;
                    box.classId = 0;
                    detectBBoxes[batchIdx].push_back(box);
                    count++;
                }
            }
        }
    }

private:
    uint32_t netChannel_ = 3;
    uint32_t netHeight_ = 448;
    uint32_t netWidth_ = 768;
    uint32_t batchSize_ = 1;

    uint32_t anchorChannel_ = 6;
    uint32_t anchorNumber_ = 4;
    uint32_t classNumber_ = 1;

    float confidenceThresh_ = 0.3;
    float nmsThresh_ = 0.45;
    uint32_t inputSize_ = 0;
    std::vector<uint32_t> anchors_ = {};

    float Overlap(const YoloBox &bbox1, const YoloBox &bbox2) const
    {
        float left = std::max(bbox1.x, bbox2.x);
        float right = std::min(bbox1.x + bbox1.w, bbox2.x + bbox2.w);
        float top = std::max(bbox1.y, bbox2.y);
        float down = std::min(bbox1.y + bbox1.h, bbox2.y + bbox2.h);
        if (left >= right or top >= down) {
            return 0;
        } else {
            float interArea = (right - left) * (down - top);
            float unionArea = bbox1.w * bbox1.h + bbox2.w * bbox2.h - interArea;
            return interArea / (unionArea + SMALL_EPSINON);
        }
    }

    void ApplyNMS(const std::vector<YoloBox> &boxes, std::vector<int> &masks, float nms, uint32_t maxNums) const
    {
        masks.clear();
        masks.reserve(boxes.size());
        masks.assign(boxes.size(), 0);
        for (uint32_t i = 0, num = 0; i < boxes.size() && num < (uint32_t)maxNums; ++i) {
            if (masks[i] == 1) {
                continue;
            }
            ++num;
            for (uint32_t j = i + 1; j < boxes.size(); ++j) {
                if (masks[j] == 1) {
                    continue;
                }

                float iou = Overlap(boxes[i], boxes[j]);
                if (iou >= nms) {
                    masks[j] = 1;
                }
            }
        }
    }

    uint32_t GetGridWidth(uint32_t level) const
    {
        uint32_t width;
        switch (level) {
            case LEVEL_TYPE_0:
                width = netWidth_ / GRID_STRIDE_LEVEL_0;
                break;
            case LEVEL_TYPE_1:
                width = netWidth_ / GRID_STRIDE_LEVEL_1;
                break;
            case LEVEL_TYPE_2:
                width = netWidth_ / GRID_STRIDE_LEVEL_2;
                break;
            default:
                width = netWidth_ / GRID_STRIDE_LEVEL_0;
                break;
        }
        return width;
    }

    uint32_t GetGridHeight(uint32_t level) const
    {
        uint32_t height;
        switch (level) {
            case LEVEL_TYPE_0:
                height = netHeight_ / GRID_STRIDE_LEVEL_0;
                break;
            case LEVEL_TYPE_1:
                height = netHeight_ / GRID_STRIDE_LEVEL_1;
                break;
            case LEVEL_TYPE_2:
                height = netHeight_ / GRID_STRIDE_LEVEL_2;
                break;
            default:
                height = netHeight_ / GRID_STRIDE_LEVEL_0;
                break;
        }
        return height;
    }

    inline float Sigmoid(float x) const
    {
        return 1.0 / (1.0 + exp(-x));
    }

    float GetMaxVal(const std::vector<float> &vals, const uint32_t &size, uint32_t &maxValueIndex) const
    {
        float maxTmp = vals[0];
        maxValueIndex = 0;
        for (uint32_t i = 1; i < size; ++i) {
            if (vals[i] > maxTmp) {
                maxTmp = vals[i];
                maxValueIndex = i;
            }
        }
        return maxTmp;
    }

    void GetBatchIdxResult(const float *inputAddress, const uint32_t &scale, const uint32_t &batchIdx,
        std::vector<std::vector<YoloBox>> &scaleBox) const
    {
        float cx, cy, boxWidth, boxHeight;
        float objScore;
        uint32_t gridHeight = GetGridHeight(scale);
        uint32_t gridWidth = GetGridWidth(scale);
        uint32_t gridStride = gridWidth * gridHeight;
        for (uint32_t stride = 0; stride < gridStride; stride++) {
            if (gridWidth <= 0) {
                return;
            }
            uint32_t h = stride / gridWidth;
            uint32_t w = stride % gridWidth;
            for (uint32_t anchorIdx = 0; anchorIdx < anchorNumber_; anchorIdx++) {
                uint32_t offset = gridStride * (anchorIdx * anchorChannel_ + SCORE_OFFSET) + h * gridWidth + w;
                float value = *(inputAddress + offset);
                objScore = value;

                if (objScore < confidenceThresh_) {
                    continue;
                }
                uint32_t maxIdx = 0;
                value = *(inputAddress + gridStride * (anchorIdx * anchorChannel_ + 0) + h * gridWidth + w);
                cx = (static_cast<float>(w) + value) / static_cast<float>(gridWidth);

                value = *(inputAddress + gridStride * (anchorIdx * anchorChannel_ + 1) + h * gridWidth + w);
                cy = (static_cast<float>(h) + value) / static_cast<float>(gridHeight);

                value = *(inputAddress + gridStride * (anchorIdx * anchorChannel_ + WIDTH_OFFSET) + h * gridWidth + w);
                boxWidth = 
                    static_cast<float>(anchors_[WIDTH_OFFSET * anchorNumber_ * scale + anchorIdx * WIDTH_OFFSET]) *
                    value / (1 - value) / static_cast<float>(netWidth_);

                value = *(inputAddress + gridStride * (anchorIdx * anchorChannel_ + HEIGHT_OFFSET) + h * gridWidth + w);
                boxHeight =
                    static_cast<float>(anchors_[WIDTH_OFFSET * anchorNumber_ * scale + anchorIdx * WIDTH_OFFSET + 1]) *
                    value / (1 - value) / static_cast<float>(netHeight_);

                YoloBox tmpBox = {};
                tmpBox.x = cx - boxWidth * HALF_OF_LENGHT_SCALE;
                tmpBox.y = cy - boxHeight * HALF_OF_LENGHT_SCALE;
                tmpBox.w = boxWidth;
                tmpBox.h = boxHeight;
                tmpBox.score = objScore;
                tmpBox.classId = maxIdx;
                scaleBox[batchIdx].push_back(tmpBox);
                if (scaleBox[batchIdx].size() > MAX_NUM_OBJECT) {
                    return;
                }
            }
        }
        return;
    }

    void GetScaleResult(const float *tensor, const uint32_t &scale, std::vector<std::vector<YoloBox>> &scaleBox,
        const uint32_t &batchIdx) const
    {
        if (tensor == nullptr) {
            return;
        }

        uint32_t gridHeight = GetGridHeight(scale);
        uint32_t gridWidth = GetGridWidth(scale);
        uint32_t tensorChannel = anchorNumber_ * anchorChannel_;
        uint32_t batchStride = gridHeight * gridWidth * tensorChannel;
        uint32_t batchOffset = batchIdx * batchStride;
        const float *inputAddress = tensor + batchOffset;
        GetBatchIdxResult(inputAddress, scale, batchIdx, scaleBox);
    }
};

#endif
