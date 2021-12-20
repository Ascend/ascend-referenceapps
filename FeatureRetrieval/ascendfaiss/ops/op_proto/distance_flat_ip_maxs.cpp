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

#include "distance_flat_ip_maxs.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(DistanceFlatIPMaxs, DistanceFlatIPMaxsVerify)
{
    DataType inputTypeX0 = op.GetInputDesc("x0").GetDataType();
    DataType inputTypeX1 = op.GetInputDesc("x1").GetDataType();
    if ((inputTypeX0 != inputTypeX1)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistanceFlatIPMaxsInferShape)
{
    DataType inputDtype = op.GetInputDesc("x0").GetDataType();
    Format inputFormat = op.GetInputDesc("x0").GetFormat();

    Shape x0_shape = op.GetInputDesc("x0").GetShape();
    Shape x1_shape = op.GetInputDesc("x1").GetShape();
    TensorDesc OutputDesc0 = op.GetOutputDesc("y0");
    TensorDesc OutputDesc1 = op.GetOutputDesc("y1");
    TensorDesc OutputDesc2 = op.GetOutputDesc("y2");

    std::vector<int64_t> dimsX0 = x0_shape.GetDims();
    std::vector<int64_t> dimsX1 = x1_shape.GetDims();

    std::vector<int64_t> dimY;
    dimY.push_back(dimsX0[0]);
    dimY.push_back(dimsX1[0] * 16);

    ge::Shape outputShape0 = ge::Shape(dimY);

    OutputDesc0.SetShape(outputShape0);
    OutputDesc0.SetDataType(inputDtype);
    OutputDesc0.SetFormat(inputFormat);
    op.UpdateOutputDesc("y0", OutputDesc0);

    std::vector<int64_t> dim_max_y;
    dim_max_y.push_back(dimsX0[0]);
    dim_max_y.push_back((dimsX1[0] * 16 + 63) / 64 * 2);

    ge::Shape outputShape1 = ge::Shape(dim_max_y);
    
    OutputDesc1.SetShape(outputShape1);
    OutputDesc1.SetDataType(inputDtype);
    OutputDesc1.SetFormat(inputFormat);
    op.UpdateOutputDesc("y1", OutputDesc1);

    std::vector<int64_t> dimVec2 { 32 };
    ge::Shape outputShape2 = ge::Shape(dimVec2);

    OutputDesc2.SetShape(outputShape2);
    OutputDesc2.SetDataType(DT_UINT16);
    OutputDesc2.SetFormat(inputFormat);
    op.UpdateOutputDesc("y2", OutputDesc2);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DistanceFlatIPMaxs, DistanceFlatIPMaxsInferShape);

// Registered verify function
VERIFY_FUNC_REG(DistanceFlatIPMaxs, DistanceFlatIPMaxsVerify);
} // namespace ge