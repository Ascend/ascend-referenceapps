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

#include "distance_ivf_int8_cos_l1.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(DistanceIVFInt8CosL1, DistanceIVFInt8CosL1Verify)
    {
        DataType input_type_x0 = op.GetInputDesc("x0").GetDataType();
        DataType input_type_x1 = op.GetInputDesc("x1").GetDataType();

        if ((input_type_x0 != input_type_x1)) {
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(DistanceIVFInt8CosL1InferShape)
    {
        DataType x2_dtype = op.GetInputDesc("x2").GetDataType();
        Format x2_format = op.GetInputDesc("x2").GetFormat();

        Shape x0_shape = op.GetInputDesc("x0").GetShape();
        Shape x3_shape = op.GetInputDesc("x3").GetShape();
        TensorDesc outputDesc0 = op.GetOutputDesc("y0");
        TensorDesc outputDesc1 = op.GetOutputDesc("y1");

        std::vector<int64_t> dims_x0 = x0_shape.GetDims();
        std::vector<int64_t> dims_x3 = x3_shape.GetDims();

        std::vector<int64_t> dim_y0;
        dim_y0.push_back(dims_x0[0]);
        dim_y0.push_back(dims_x3[0]);

        ge::Shape outputShape0 = ge::Shape(dim_y0);

        outputDesc0.SetShape(outputShape0);
        outputDesc0.SetDataType(x2_dtype);
        outputDesc0.SetFormat(x2_format);
        op.UpdateOutputDesc("y0", outputDesc0);

        std::vector<int64_t> dimVec1 { 32 };
        ge::Shape outputShape1 = ge::Shape(dimVec1);

        outputDesc1.SetShape(outputShape1);
        outputDesc1.SetDataType(DT_UINT16);
        outputDesc1.SetFormat(x2_format);
        op.UpdateOutputDesc("y1", outputDesc1);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(DistanceIVFInt8CosL1, DistanceIVFInt8CosL1InferShape);

    // Registered verify function
    VERIFY_FUNC_REG(DistanceIVFInt8CosL1, DistanceIVFInt8CosL1Verify);
} // namespace ge