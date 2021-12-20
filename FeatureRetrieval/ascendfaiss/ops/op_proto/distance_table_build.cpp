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

#include "distance_table_build.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(DistanceTableBuild, DistanceTableBuildVerify)
{
    DataType inputTypeX0 = op.GetInputDesc("x0").GetDataType();
    DataType inputTypeX1 = op.GetInputDesc("x1").GetDataType();
    DataType inputTypeX3 = op.GetInputDesc("x3").GetDataType();
    if ((inputTypeX0 != inputTypeX1) || (inputTypeX0 != inputTypeX3) || (inputTypeX1 != inputTypeX3)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistanceTableBuildInferShape)
{
    DataType inputDtype = op.GetInputDesc("x0").GetDataType();
    Format inputFormat = op.GetInputDesc("x0").GetFormat();

    Shape x0Shape = op.GetInputDesc("x0").GetShape();
    Shape x1Shape = op.GetInputDesc("x1").GetShape();
    Shape x2Shape = op.GetInputDesc("x2").GetShape();
    TensorDesc OutputDesc0 = op.GetOutputDesc("y0");
    TensorDesc OutputDesc1 = op.GetOutputDesc("y1");

    std::vector<int64_t> dimsX0 = x0Shape.GetDims();
    std::vector<int64_t> dimsX1 = x1Shape.GetDims();
    std::vector<int64_t> dimsX2 = x2Shape.GetDims();

    int64_t dims_m = dimsX1[1] * dimsX1[0] / dimsX0[1];

    std::vector<int64_t> dimY;
    dimY.push_back(dimsX0[0]);
    dimY.push_back(dimsX2[1]);
    dimY.push_back(dimsX1[0]);
    dimY.push_back(dims_m);

    ge::Shape outputShape = ge::Shape(dimY);

    OutputDesc0.SetShape(outputShape);
    OutputDesc0.SetDataType(inputDtype);
    OutputDesc0.SetFormat(inputFormat);
    op.UpdateOutputDesc("y0", OutputDesc0);

    std::vector<int64_t> dimVec1 { 16 };
    ge::Shape outputShape1 = ge::Shape(dimVec1);

    OutputDesc1.SetShape(outputShape1);
    OutputDesc1.SetDataType(DT_UINT16);
    OutputDesc1.SetFormat(inputFormat);
    op.UpdateOutputDesc("y1", OutputDesc1);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DistanceTableBuild, DistanceTableBuildInferShape);

// Registered verify function
VERIFY_FUNC_REG(DistanceTableBuild, DistanceTableBuildVerify);
} // namespace ge