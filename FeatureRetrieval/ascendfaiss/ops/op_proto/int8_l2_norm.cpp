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

#include "int8_l2_norm.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(Int8L2Norm, Int8L2NormVerify)
    {
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(Int8L2NormInferShape)
    {
        DataType x1_dtype = op.GetInputDesc("x1").GetDataType();
        Format x1_format = op.GetInputDesc("x1").GetFormat();

        Shape x0_shape = op.GetInputDesc("x0").GetShape();
        TensorDesc outputDesc0 = op.GetOutputDesc("y0");

        std::vector<int64_t> dims_x0 = x0_shape.GetDims();

        std::vector<int64_t> dim_y0;
        dim_y0.push_back(dims_x0[0]);

        ge::Shape outputShape0 = ge::Shape(dim_y0);

        outputDesc0.SetShape(outputShape0);
        outputDesc0.SetDataType(x1_dtype);
        outputDesc0.SetFormat(x1_format);
        op.UpdateOutputDesc("y0", outputDesc0);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(Int8L2Norm, Int8L2NormInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(Int8L2Norm, Int8L2NormVerify);
} // namespace ge