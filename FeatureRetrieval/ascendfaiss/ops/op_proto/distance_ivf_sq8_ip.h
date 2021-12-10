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

#ifndef GE_OP_DISTANCE_IVF_SQ8_IP_H
#define GE_OP_DISTANCE_IVF_SQ8_IP_H

#include "graph/operator_reg.h"

namespace ge {
    REG_OP(DistanceIVFSQ8IP)
        .INPUT(x0, TensorType({ DT_FLOAT16 })) /* "First operand." */
        .INPUT(x1, TensorType({ DT_UINT8 }))   /* "Second operand." */
        .INPUT(x2, TensorType({ DT_FLOAT16 }))   /* "Third operand." */
        .INPUT(x3, TensorType({ DT_FLOAT16 })) /* "Fourth operand." */
        .INPUT(x4, TensorType({ DT_UINT32 })) /* "Fifth operand." */
        /* "Result, has same element type as three inputs" */
        .OUTPUT(y0, TensorType({ DT_FLOAT16 }))
        .OUTPUT(y1, TensorType({ DT_FLOAT16 }))
        .OUTPUT(y2, TensorType({ DT_UINT16 }))
        .OP_END_FACTORY_REG(DistanceIVFSQ8IP)
} // namespace ge
#endif // GE_OP_DISTANCE_IVF_SQ8_IP_H