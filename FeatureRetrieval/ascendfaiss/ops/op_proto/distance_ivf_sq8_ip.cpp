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

#include "distance_ivf_sq8_ip.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(DistanceIVFSQ8IP, DistanceIVFSQ8IPVerify)
    {
        DataType input_type_x2 = op.GetInputDesc("x2").GetDataType();
        DataType input_type_x3 = op.GetInputDesc("x3").GetDataType();

        if ((input_type_x2 != input_type_x3)) {
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(DistanceIVFSQ8IPInferShape)
    {
        DataType x0_dtype = op.GetInputDesc("x0").GetDataType();
        Format x0_format = op.GetInputDesc("x0").GetFormat();

        Shape x0_shape = op.GetInputDesc("x0").GetShape();
        Shape x1_shape = op.GetInputDesc("x1").GetShape();
        TensorDesc outputDesc0 = op.GetOutputDesc("y0");
        TensorDesc outputDesc1 = op.GetOutputDesc("y1");
        TensorDesc outputDesc2 = op.GetOutputDesc("y2");

        std::vector<int64_t> dims_x0 = x0_shape.GetDims();
        std::vector<int64_t> dims_x1 = x1_shape.GetDims();

        std::vector<int64_t> dim_y0;
        dim_y0.push_back(dims_x0[0]);
        dim_y0.push_back(dims_x1[0] * 16);

        ge::Shape outputShape0 = ge::Shape(dim_y0);

        outputDesc0.SetShape(outputShape0);
        outputDesc0.SetDataType(x0_dtype);
        outputDesc0.SetFormat(x0_format);
        op.UpdateOutputDesc("y0", outputDesc0);

        int max_batch = 32;
        if (dims_x0[1] > 128) {
            max_batch = 16;
        }

        std::vector<int64_t> dim_max_y;
        dim_max_y.push_back(dims_x0[0]);
        dim_max_y.push_back((dims_x1[0] * 16 + max_batch - 1) / max_batch * 2);

        ge::Shape outputShape1 = ge::Shape(dim_max_y);

        outputDesc1.SetShape(outputShape1);
        outputDesc1.SetDataType(x0_dtype);
        outputDesc1.SetFormat(x0_format);
        op.UpdateOutputDesc("y1", outputDesc1);

        std::vector<int64_t> dimVec2 { 32 };
        ge::Shape outputShape2 = ge::Shape(dimVec2);

        outputDesc2.SetShape(outputShape2);
        outputDesc2.SetDataType(DT_UINT16);
        outputDesc2.SetFormat(x0_format);
        op.UpdateOutputDesc("y2", outputDesc2);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(DistanceIVFSQ8IP, DistanceIVFSQ8IPInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(DistanceIVFSQ8IP, DistanceIVFSQ8IPVerify);
} // namespace ge