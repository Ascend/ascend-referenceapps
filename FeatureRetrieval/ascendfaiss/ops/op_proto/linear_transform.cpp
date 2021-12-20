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

#include "linear_transform.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(LinearTransform, LinearTransformVerify)
{
    DataType input_type_x0 = op.GetInputDesc("x0").GetDataType();
    DataType input_type_x1 = op.GetInputDesc("x1").GetDataType();
    if (input_type_x0 != input_type_x1) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(LinearTransformInferShape)
{
    DataType input_dtype = op.GetInputDesc("x0").GetDataType();
    Format input_format = op.GetInputDesc("x0").GetFormat();

    Shape x0_shape = op.GetInputDesc("x0").GetShape();
    Shape x2_shape = op.GetInputDesc("x2").GetShape();
    TensorDesc OutputDesc = op.GetOutputDesc("y");

    std::vector<int64_t> dims_x0 = x0_shape.GetDims();
    std::vector<int64_t> dims_x2 = x2_shape.GetDims();

    std::vector<int64_t> dim_y;
    dim_y.push_back(dims_x0[0]);
    dim_y.push_back(dims_x2[0]);

    ge::Shape outputShape = ge::Shape(dim_y);

    OutputDesc.SetShape(outputShape);
    OutputDesc.SetDataType(input_dtype);
    OutputDesc.SetFormat(input_format);
    op.UpdateOutputDesc("y", OutputDesc);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(LinearTransform, LinearTransformInferShape);

// Registered verify function
VERIFY_FUNC_REG(LinearTransform, LinearTransformVerify);
} // namespace ge