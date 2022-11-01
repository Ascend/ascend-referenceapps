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

#include "distance_int8_cos_maxs.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(DistanceInt8CosMaxs, DistanceInt8CosMaxsVerify)
    {
        DataType input_type_x0 = op.GetInputDesc("x0").GetDataType();
        DataType input_type_x1 = op.GetInputDesc("x1").GetDataType();
        DataType input_type_x2 = op.GetInputDesc("x2").GetDataType();

        if ((input_type_x0 != input_type_x2) || (input_type_x1 != DT_UINT8)) {
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(DistanceInt8CosMaxsInferShape)
    {
        DataType x3_dtype = op.GetInputDesc("x3").GetDataType();
        Format x3_format = op.GetInputDesc("x3").GetFormat();

        Shape x0_shape = op.GetInputDesc("x0").GetShape();
        Shape x4_shape = op.GetInputDesc("x4").GetShape();
        TensorDesc outputDesc0 = op.GetOutputDesc("y0");
        TensorDesc outputDesc1 = op.GetOutputDesc("y1");
        TensorDesc outputDesc2 = op.GetOutputDesc("y2");

        std::vector<int64_t> dims_x0 = x0_shape.GetDims();
        std::vector<int64_t> dims_x4 = x4_shape.GetDims();

        std::vector<int64_t> dim_y0;
        dim_y0.push_back(dims_x0[0]);
        dim_y0.push_back(dims_x4[0]);

        ge::Shape outputShape0 = ge::Shape(dim_y0);

        outputDesc0.SetShape(outputShape0);
        outputDesc0.SetDataType(x3_dtype);
        outputDesc0.SetFormat(x3_format);
        op.UpdateOutputDesc("y0", outputDesc0);

        std::vector<int64_t> dim_y1;
        dim_y1.push_back(dims_x0[0]);
        dim_y1.push_back((dims_x4[0] + 63) / 64 * 2);

        ge::Shape outputShape1 = ge::Shape(dim_y1);

        outputDesc1.SetShape(outputShape1);
        outputDesc1.SetDataType(x3_dtype);
        outputDesc1.SetFormat(x3_format);
        op.UpdateOutputDesc("y1", outputDesc1);

        std::vector<int64_t> dimVec2 { 16, 16 };
        ge::Shape outputShape2 = ge::Shape(dimVec2);

        outputDesc2.SetShape(outputShape2);
        outputDesc2.SetDataType(DT_UINT16);
        outputDesc2.SetFormat(x3_format);
        op.UpdateOutputDesc("y2", outputDesc2);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(DistanceInt8CosMaxs, DistanceInt8CosMaxsInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(DistanceInt8CosMaxs, DistanceInt8CosMaxsVerify);
} // namespace ge
