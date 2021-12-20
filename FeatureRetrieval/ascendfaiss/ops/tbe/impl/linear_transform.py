#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from te import tik


class LinearTransform():
    def __init__(self,
                 input_queries,
                 input_weights,
                 input_bias,
                 output_queries,
                 kernel_name="linear_transform"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_weights = input_weights.get("shape")
        self.dtype_weights = input_weights.get("dtype")
        self.shape_bias = input_bias.get("shape")
        self.dtype_bias = input_bias.get("dtype")
        self.shape_output_queries = output_queries.get("shape")
        self.dtype_output_queries = output_queries.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.M, self.K = self.shape_queries
        self.N = self.shape_bias[0]

        self.fp32_vector_mask_max = 64
        # check parameter
        self.check_parameter()

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        if self.K % 16 != 0:
            raise RuntimeError("feature dim must be a multiple of 16")
        if self.N % 16 != 0:
            raise RuntimeError("code num must be a multiple of 16")

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_dprofile = tik.Dprofile("v100", "mini")
        self.tik_instance = tik.Tik(tik_dprofile)

        self.queries_num_each_loop = 32

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        # creat input tensor: input_queries_gm, input_weights_gm
        # and input_bias_gm
        # and output tensor: output_queries_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_weights_gm = self.tik_instance.Tensor(
            self.dtype_weights,
            self.shape_weights,
            name="input_weights_gm",
            scope=tik.scope_gm)
        self.input_bias_gm = self.tik_instance.Tensor(
            self.dtype_bias,
            self.shape_bias,
            name="input_bias_gm",
            scope=tik.scope_gm)
        self.output_queries_gm = self.tik_instance.Tensor(
            self.dtype_output_queries,
            self.shape_output_queries,
            name="output_queries_gm",
            scope=tik.scope_gm)

    def linear_transform_each_loop(self, move_offset, move_num):
        queries_align = (move_num + 15) // 16 * 16
        queries_l1 = self.tik_instance.Tensor(
            "float16", (self.K // 16, queries_align, 16),
            name="queries_l1",
            scope=tik.scope_cbuf)
        weights_l1 = self.tik_instance.Tensor("float16",
                                              (self.K // 16, self.N, 16),
                                              name="weights_l1",
                                              scope=tik.scope_cbuf)
        bias_ub = self.tik_instance.Tensor("float32", (self.N, ),
                                           name='bias_l1',
                                           scope=tik.scope_ubuf)

        # move queries from out to L1
        with self.tik_instance.for_range(0, move_num) as i:
            self.tik_instance.data_move(
                queries_l1[0, i, 0], self.input_queries_gm[move_offset + i, 0],
                0, self.K // 16, 1, 0, queries_align - 1)

        # move weights from out to L1
        self.tik_instance.data_move(weights_l1, self.input_weights_gm, 0, 1,
                                    self.K * self.N // 16, 0, 0)
        mat_l0c = self.tik_instance.Tensor("float32",
                                           (self.N // 16, queries_align, 16),
                                           name='mat_l0c',
                                           scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(mat_l0c, queries_l1, weights_l1,
                                 queries_align, self.K, self.N)

        # move bias from out to UB
        self.tik_instance.data_move(bias_ub, self.input_bias_gm, 0, 1,
                                    self.N // 8, 0, 0)

        mat_ub = self.tik_instance.Tensor("float32",
                                          (self.N // 16, queries_align, 16),
                                          name="queries_square_ub",
                                          scope=tik.scope_ubuf)
        self.tik_instance.data_move(mat_ub, mat_l0c, 0, 1,
                                    self.N * queries_align // 256, 0, 0)

        add_ub = self.tik_instance.Tensor("float32",
                                          (move_num, self.N),
                                          name="add_ub",
                                          scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, move_num) as i:
            self.tik_instance.vadd(16, add_ub[i, 0], mat_ub[0, i, 0], bias_ub,
                                   self.N // 16, 1, 1, 1, 2, queries_align * 2,
                                   2)
        dst_ub = self.tik_instance.Tensor("float16",
                                          (move_num, self.N),
                                          name="dst_ub",
                                          scope=tik.scope_ubuf)

        vconv_repeat_time = (move_num * self.N) // self.fp32_vector_mask_max
        vconv_offset = 0
        if vconv_repeat_time > 0:
            self.tik_instance.vconv(self.fp32_vector_mask_max, "none",
                                    dst_ub[vconv_offset],
                                    add_ub[vconv_offset],
                                    vconv_repeat_time, 1, 1, 4, 8)
            vconv_offset += vconv_repeat_time * self.fp32_vector_mask_max

        vconv_last_num = (move_num * self.N) % self.fp32_vector_mask_max
        if vconv_last_num > 0:
            self.tik_instance.vconv(vconv_last_num, "none",
                                    dst_ub[vconv_offset],
                                    add_ub[vconv_offset], 1, 1, 1, 4,
                                    8)

        self.tik_instance.data_move(self.output_queries_gm[move_offset,
                                                           0], dst_ub, 0, 1,
                                    move_num * self.N // 16, 0, 0)

    def linear_transform_compute(self):
        """
        the compute process
        """
        queries_loop_time = self.M // self.queries_num_each_loop
        if queries_loop_time > 0:
            thread_num_need = 2 if queries_loop_time > 1 else 1
            with self.tik_instance.for_range(
                    0, queries_loop_time,
                    thread_num=thread_num_need) as loop_queries:
                self.linear_transform_each_loop(
                    loop_queries * self.queries_num_each_loop,
                    self.queries_num_each_loop)

        queries_last_num = self.M % self.queries_num_each_loop
        if queries_last_num > 0:
            self.linear_transform_each_loop(
                queries_loop_time * self.queries_num_each_loop,
                queries_last_num)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.linear_transform_compute()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_weights_gm,
                                       self.input_bias_gm
                                   ],
                                   outputs=[self.output_queries_gm])

        return self.tik_instance


def linear_transform(input_queries,
                     input_weights,
                     input_bias,
                     output_queries,
                     kernel_name="linear_transform"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_weights : dict
        shape and dtype of weights
    input_bias : dict
        shape and dtype of bias
    output_queries : dict
        shape and dtype of distances, should be same dtype as input_queries
    kernel_name : str
        kernel name, default value is "distance_compute"

    Returns
    -------
    None
    """
    linear_transform = LinearTransform(input_queries, input_weights,
                                       input_bias, output_queries, kernel_name)
    tik_instance = linear_transform.get_tik_instance()
    return tik_instance
