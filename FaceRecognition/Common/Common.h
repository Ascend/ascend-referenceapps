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

#ifndef __INC_COMMON_H__
#define __INC_COMMON_H__

/*
 * 队列规划
 * Device限制session 不超过 64个
 * Device接收队列， channelCount个视频数据接收队列，保序， 1个注册图片队列
 * Device发送队列， channelCount个框数据发送 MOTC返回，1个人脸检索属性信息队列.Search返回， 1 个注册结果
 * Host相反
 */
const int HDC_REGIST_CH_INDEX = 0;
const int HDC_SEARCH_CH_START_INDEX = 1;

const int HDC_REGIST_RESULT_CH_INDEX = 0;
const int HDC_FACE_DETAIL_CH_INDEX = 1;
const int HDC_FRAME_ALIGN_CH_START_INDEX = 2;


const int HDC_HOST_SEND_CH_COUNT_BASE = 1;
const int HDC_HOST_RECV_CH_COUNT_BASE = 2;

// Host相反
const int HDC_DEVIEC_SEND_CH_COUNT_BASE = HDC_HOST_RECV_CH_COUNT_BASE;
const int HDC_DEVIEC_RECV_CH_COUNT_BASE = HDC_HOST_SEND_CH_COUNT_BASE;

#endif