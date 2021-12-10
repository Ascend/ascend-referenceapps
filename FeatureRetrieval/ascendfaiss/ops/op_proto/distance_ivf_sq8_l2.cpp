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

#include "distance_ivf_sq8_l2.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(DistanceIVFSQ8L2, DistanceIVFSQ8L2Verify)
    {
        DataType input_type_x0 = op.GetInputDesc("x0").GetDataType();
        DataType input_type_x3 = op.GetInputDesc("x3").GetDataType();

        if ((input_type_x0 != input_type_x3)) {
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(DistanceIVFSQ8L2InferShape)
    {
        DataType input_dtype = op.GetInputDesc("x0").GetDataType();
        Format input_format = op.GetInputDesc("x0").GetFormat();

        Shape x0_shape = op.GetInputDesc("x0").GetShape();
        Shape x2_shape = op.GetInputDesc("x2").GetShape();
        TensorDesc OutputDesc0 = op.GetOutputDesc("y0");
        TensorDesc OutputDesc1 = op.GetOutputDesc("y1");
        TensorDesc OutputDesc2 = op.GetOutputDesc("y2");

        std::vector<int64_t> dims_x0 = x0_shape.GetDims();
        std::vector<int64_t> dims_x2 = x2_shape.GetDims();

        std::vector<int64_t> dim_y;
        dim_y.push_back(dims_x0[0]);
        dim_y.push_back(dims_x2[0]);

        ge::Shape outputShape = ge::Shape(dim_y);

        OutputDesc0.SetShape(outputShape);
        OutputDesc0.SetDataType(input_dtype);
        OutputDesc0.SetFormat(input_format);
        op.UpdateOutputDesc("y0", OutputDesc0);

        int min_batch = 32;
        if (dims_x0[1] > 128) {
            min_batch = 16;
        }

        std::vector<int64_t> dim_min_y;
        dim_min_y.push_back(dims_x0[0]);
        dim_min_y.push_back((dims_x2[0] + min_batch - 1) / min_batch * 2);

        ge::Shape outputShape1 = ge::Shape(dim_min_y);
        
        OutputDesc1.SetShape(outputShape1);
        OutputDesc1.SetDataType(input_dtype);
        OutputDesc1.SetFormat(input_format);
        op.UpdateOutputDesc("y1", OutputDesc1);

        std::vector<int64_t> dimVec2 { 32 };
        ge::Shape outputShape2 = ge::Shape(dimVec2);

        OutputDesc2.SetShape(outputShape2);
        OutputDesc2.SetDataType(DT_UINT16);
        OutputDesc2.SetFormat(input_format);
        op.UpdateOutputDesc("y2", OutputDesc2);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(DistanceIVFSQ8L2, DistanceIVFSQ8L2InferShape);

    // Registered verify function
    VERIFY_FUNC_REG(DistanceIVFSQ8L2, DistanceIVFSQ8L2Verify);
} // namespace ge