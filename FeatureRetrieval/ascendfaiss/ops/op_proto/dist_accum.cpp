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

#include "dist_accum.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {
bool InferShapeAndTypeOneInOneOut(Operator &op, const string &inputName, const string &outputName1,
    const string &outputName2)
{
    TensorDesc vOutputDesc1 = op.GetOutputDesc(outputName1);
    TensorDesc vOutputDesc2 = op.GetOutputDesc(outputName2);

    DataType inputDType = op.GetInputDesc(inputName).GetDataType();
    Format inputFormat = op.GetInputDesc(inputName).GetFormat();
    ge::Shape shapeX = op.GetInputDesc(inputName).GetShape();
    std::vector<int64_t> dimsX = shapeX.GetDims();

    // 设置输出的shape维度
    std::vector<int64_t> dimVec { dimsX[0] };
    ge::Shape outputShape = ge::Shape(dimVec);

    vOutputDesc1.SetShape(outputShape);
    vOutputDesc1.SetDataType(inputDType);
    vOutputDesc1.SetFormat(inputFormat);
    op.UpdateOutputDesc(outputName1, vOutputDesc1);

    std::vector<int64_t> dimVec1 { 16 };
    ge::Shape outputShape1 = ge::Shape(dimVec1);

    vOutputDesc2.SetShape(outputShape1);
    vOutputDesc2.SetDataType(DT_UINT16);
    vOutputDesc2.SetFormat(inputFormat);
    op.UpdateOutputDesc(outputName2, vOutputDesc2);

    return true;
}

IMPLEMT_VERIFIER(DistAccum, DistAccumVerify)
{
    // there is only one input, so return SUCCESS
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistAccumInferShape)
{
    if (InferShapeAndTypeOneInOneOut(op, "x", "y", "z")) {
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DistAccum, DistAccumInferShape);

// Registered verify function
VERIFY_FUNC_REG(DistAccum, DistAccumVerify);
} // namespace ge