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


class DistanceComputeSQ8Min64:
    def __init__(self,
                 input_queries,
                 input_code,
                 input_precomputed,
                 input_diff,
                 input_min,
                 input_actual_code_num,
                 output_distances,
                 output_mins,
                 output_flag,
                 kernel_name="distance_compute_sq8_min64"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_code = input_code.get("shape")
        self.dtype_code = input_code.get("dtype")
        self.shape_precomputed = input_precomputed.get("shape")
        self.dtype_precomputed = input_precomputed.get("dtype")
        self.shape_diff = input_diff.get("shape")
        self.dtype_diff = input_diff.get("dtype")
        self.shape_min = input_min.get("shape")
        self.dtype_min = input_min.get("dtype")
        self.shape_actual_code_num = input_actual_code_num.get("shape")
        self.dtype_actual_code_num = input_actual_code_num.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_output_mins = output_mins.get("shape")
        self.dtype_output_mins = output_mins.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.code_num, = self.shape_precomputed

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.fp32_mask = 64
        self.fp16_mask = 128
        self.min_batch = 64

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        if self.dim % 16 != 0:
            raise RuntimeError("feature dim must be a multiple of 16")
        if self.code_num % 16 != 0:
            raise RuntimeError("code num must be a multiple of 16")

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_dprofile = tik.Dprofile("v100", "mini")
        self.tik_instance = tik.Tik(tik_dprofile)

        self.aicore_use = 2
        self.queries_num_each_loop = min(self.queries_num, 48)
        self.code_num_each_loop = min(48 // 16 // ((self.queries_num_each_loop + 15) // 16) * 1024,
                                      (1 + 256 // (self.dim + 128)) * 1024)

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("float32",
                                              name="coeff",
                                              init_value=-2 / 255)

        # creat input tensor: input_queries_gm, input_code_gm,
        # input_precomputed_gm, input_diff_gm, input_min_gm,
        # input_actual_code_num_gm
        # and creat output tensor: output_distances_gm
        # and output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_code_gm = self.tik_instance.Tensor(
            self.dtype_code,
            self.shape_code,
            name="input_code_gm",
            scope=tik.scope_gm)
        self.input_precomputed_gm = self.tik_instance.Tensor(
            self.dtype_precomputed,
            self.shape_precomputed,
            name="input_precomputed_gm",
            scope=tik.scope_gm)
        self.input_diff_gm = self.tik_instance.Tensor(
            self.dtype_diff,
            self.shape_diff,
            name="input_diff_gm",
            scope=tik.scope_gm)
        self.input_min_gm = self.tik_instance.Tensor(
            self.dtype_min,
            self.shape_min,
            name="input_min_gm",
            scope=tik.scope_gm)
        self.input_actual_code_num_gm = self.tik_instance.Tensor(
            self.dtype_actual_code_num,
            self.shape_actual_code_num,
            name="input_actual_code_num_gm",
            scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(
            self.dtype_distances,
            self.shape_distances,
            name="output_distances_gm",
            scope=tik.scope_gm)
        self.output_mins_gm = self.tik_instance.Tensor(
            self.dtype_output_mins,
            self.shape_output_mins,
            name="output_mins_gm",
            scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(
            self.dtype_flag,
            self.shape_flag,
            name="output_flag_gm",
            scope=tik.scope_gm)

    def cal_num_each_core(self):
        """
        calculate actual code num of each core
        """
        # move actual code num from out to UB
        actual_code_num_ub = self.tik_instance.Tensor(
            "uint32", (8,), name="actual_code_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_code_num_ub,
                                    self.input_actual_code_num_gm, 0, 1, 1, 0,
                                    0)
        actual_code_num = self.tik_instance.Scalar(dtype="uint32",
                                                   name="actual_code_num",
                                                   init_value=0)
        actual_code_num.set_as(actual_code_num_ub[0])

        self.code_num_each_core = \
            (actual_code_num // self.aicore_use + self.min_batch * 8) // self.min_batch // 16 * self.min_batch * 16
        self.code_num_last_core = actual_code_num - (self.aicore_use - 1) * self.code_num_each_core

    def _muls(self, dst, src, scalar, compute_num, mask):
        # process 256B data per repeat for vmuls
        repeat = compute_num // mask
        offset = 0
        if repeat > 0:
            self.tik_instance.vmuls(mask, dst[0], src[0], scalar,
                                    repeat, 1, 1, 8, 8)
            offset += repeat * mask

        remain = compute_num % mask
        if remain > 0:
            self.tik_instance.vmuls(remain, dst[offset], src[offset], scalar,
                                    1, 1, 1, 8, 8)

    def _add(self, dst, src0, src1, compute_num, mask):
        # process 256B data per repeat for vadd
        repeat = compute_num // mask
        offset = 0
        if repeat > 0:
            self.tik_instance.vadd(mask, dst[offset], src0[offset],
                                   src1[offset], repeat, 1, 1, 1, 8, 8, 8)
            offset += repeat * mask

        remain = compute_num % mask
        if remain > 0:
            self.tik_instance.vadd(remain, dst[offset], src0[offset],
                                   src1[offset], 1, 1, 1, 1, 8, 8, 8)

    def distance_compute_each_loop(self, aicore_move_offset, aicore_code_num,
                                   move_offset, move_num):
        queries_align = (move_num + 15) // 16 * 16
        add_scalar_list = [
            self.tik_instance.Scalar(dtype="float32") for i in range(move_num)
        ]
        data_xd_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, queries_align, 16),
                                              name="data_xd_l1", scope=tik.scope_cbuf)

        with self.tik_instance.new_stmt_scope():
            queries_ub = self.tik_instance.Tensor("float16",
                                                  (move_num, self.dim),
                                                  name="queries_ub", scope=tik.scope_ubuf)
            diff_ub = self.tik_instance.Tensor("float16",
                                               (self.dim,),
                                               name="diff_ub", scope=tik.scope_ubuf)
            min_ub = self.tik_instance.Tensor("float16",
                                              (self.dim,),
                                              name="min_ub", scope=tik.scope_ubuf)

            # move diff adn min from out to UB
            self.tik_instance.data_move(diff_ub, self.input_diff_gm, 0, 1, self.dim // 16, 0, 0)
            self.tik_instance.data_move(min_ub, self.input_min_gm, 0, 1, self.dim // 16, 0, 0)
            diff2_ub = self.tik_instance.Tensor("float16",
                                                (self.dim,),
                                                name="diff2_ub", scope=tik.scope_ubuf)
            # cal 0.5 * d
            self._muls(diff2_ub, diff_ub, 0.5, self.dim, self.fp16_mask)
            # cal 255 * m
            self._muls(min_ub, min_ub, 255.0, self.dim, self.fp16_mask)
            # cal 0.5 * d + 255 * m
            self._add(diff2_ub, diff2_ub, min_ub, self.dim, self.fp16_mask)
            dm_transfer_ub = self.tik_instance.Tensor("float16",
                                                      (self.dim // 16, 16, 16),
                                                      name="dm_transfer_ub", scope=tik.scope_ubuf)
            # use data_move function to make 0.5 * d + 255 * m align 16
            self.tik_instance.data_move(dm_transfer_ub, diff2_ub, 0, self.dim // 16, 1, 0, 15)
            # move d+m from UB to L1
            data_dm_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, 16, 16),
                                                  name="data_dm_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(data_dm_l1, dm_transfer_ub, 0, 1, self.dim, 0, 0)

            # move x from out to UB
            self.tik_instance.data_move(queries_ub,
                                        self.input_queries_gm[move_offset, 0],
                                        0, 1, move_num * self.dim // 16, 0, 0)
            queries_transfer_ub = self.tik_instance.Tensor("float16",
                                                           (self.dim // 16, queries_align, 16),
                                                           name="queries_transfer_ub", scope=tik.scope_ubuf)
            # use data_move function to make x align 16
            with self.tik_instance.for_range(0, move_num) as i:
                self.tik_instance.data_move(queries_transfer_ub[0, i, 0],
                                            queries_ub[i, 0],
                                            0, self.dim // 16, 1, 0, queries_align - 1)

            if self.dim <= 128:
                self.tik_instance.vmul(self.dim, queries_ub, queries_ub,
                                       diff_ub, move_num, 1, 1, 1,
                                       self.dim // 16, self.dim // 16, 0)
            else:
                with self.tik_instance.for_range(0, move_num) as i:
                    repeat = self.dim // self.fp16_mask
                    offset = 0
                    if repeat > 0:
                        self.tik_instance.vmul(self.fp16_mask,
                                               queries_ub[i, offset],
                                               queries_ub[i, offset],
                                               diff_ub[offset], repeat, 1, 1,
                                               1, 8, 8, 8)
                        offset += repeat * self.fp16_mask

                    remain = self.dim % self.fp16_mask
                    if remain > 0:
                        self.tik_instance.vmul(remain, queries_ub[i, offset],
                                               queries_ub[i, offset],
                                               diff_ub[offset], 1, 1, 1, 1, 8,
                                               8, 8)

            # use data_move function to make x*d align 16
            xd_transfer_ub = self.tik_instance.Tensor("float16",
                                                      (self.dim // 16, queries_align, 16),
                                                      name="xd_transfer_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, move_num) as i:
                self.tik_instance.data_move(xd_transfer_ub[0, i, 0],
                                            queries_ub[i, 0], 0, self.dim // 16,
                                            1, 0, queries_align - 1)

            # move x from ub to L1
            queries_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, queries_align, 16),
                                                  name="queries_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(queries_l1, queries_transfer_ub, 0, 1,
                                        queries_align * self.dim // 16, 0, 0)
            queries_square_l0c = self.tik_instance.Tensor("float32",
                                                          (queries_align // 16, queries_align, 16),
                                                          name="queries_square_l0c", scope=tik.scope_cbuf_out)
            # cal x*x
            self.tik_instance.matmul(queries_square_l0c, queries_l1,
                                     queries_l1, queries_align, self.dim,
                                     queries_align)
            # use data_move function to make x*x align 16
            queries_square_ub = self.tik_instance.Tensor("float32",
                                                         (queries_align // 16, queries_align, 16),
                                                         name="queries_square_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(queries_square_ub, queries_square_l0c,
                                        0, 1,
                                        queries_align * queries_align // 256,
                                        0, 0)

            # move x*d from UB to L1
            self.tik_instance.data_move(data_xd_l1, xd_transfer_ub, 0, 1,
                                        queries_align * self.dim // 16, 0, 0)

            data_xdm_l0c = self.tik_instance.Tensor("float32",
                                                    (queries_align // 16, 16, 16),
                                                    name="data_xdm_l0c", scope=tik.scope_cbuf_out)
            # cal x * (0.5 * d + 255 * m)
            self.tik_instance.matmul(data_xdm_l0c, data_dm_l1, queries_l1, 16,
                                     self.dim, queries_align)

            queries_l2_ub = self.tik_instance.Tensor("float32",
                                                     (queries_align,),
                                                     name="queries_l2_ub", scope=tik.scope_ubuf)
            for i in range(move_num):
                mask = 2 ** (i % 16)
                # get sum
                self.tik_instance.vcadd([0, mask],
                                        queries_l2_ub[i],
                                        queries_square_ub[i // 16, i, 0],
                                        1, 1, 1, 8)

            data_xdm_ub = self.tik_instance.Tensor("float32",
                                                   (queries_align // 16, 16, 16),
                                                   name="data_xdm_ub", scope=tik.scope_ubuf)
            # move data from l0 to ub
            self.tik_instance.data_move(data_xdm_ub, data_xdm_l0c, 0, 1, queries_align // 16, 0, 0)
            data_xdm_align_ub = self.tik_instance.Tensor("float32",
                                                         (queries_align,),
                                                         name="data_xdm_align_ub", scope=tik.scope_ubuf)
            # 2 / 255 * x * (0.5 * d + 255 * m)
            self.tik_instance.vmuls(16, data_xdm_align_ub, data_xdm_ub,
                                    self.coeff, queries_align // 16, 1, 1, 2, 32)
            # get sum for x*x - 2/255*(0.5*d + 255_*m)
            self._add(queries_l2_ub, queries_l2_ub, data_xdm_align_ub, queries_align, self.fp16_mask)

            # set as scalar for next cal
            for i in range(move_num):
                add_scalar_list[i].set_as(queries_l2_ub[i])

        # compute xy using cube
        code_loop_time = aicore_code_num // self.code_num_each_loop
        with self.tik_instance.if_scope(code_loop_time > 0):
            with self.tik_instance.for_range(0, code_loop_time) as loop_code:
                self.cube_compute_each_loop(data_xd_l1, add_scalar_list,
                                            aicore_move_offset, loop_code * self.code_num_each_loop,
                                            self.code_num_each_loop, move_offset, move_num, 0)

        code_last_num = aicore_code_num % self.code_num_each_loop
        with self.tik_instance.if_scope(code_last_num > 0):
            self.cube_compute_each_loop(data_xd_l1, add_scalar_list,
                                        aicore_move_offset, code_loop_time * self.code_num_each_loop, code_last_num,
                                        move_offset, move_num, 1)

    def cube_compute_each_loop(self, data_xd_l1, add_scalar_list,
                               aicore_move_offset, code_move_offset,
                               code_move_num, queries_move_offset,
                               queries_move_num, flag):
        queries_align = (queries_move_num + 15) // 16 * 16
        inner_product_l0c = self.tik_instance.Tensor("float32",
                                                     (self.code_num_each_loop // 16, queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        code_l2_ub_fp16 = self.tik_instance.Tensor("float32",
                                                   (self.code_num_each_loop,),
                                                   name="code_l2_ub_fp16", scope=tik.scope_ubuf)
        code_move_num_ub = (code_move_num + 15) // 16 * 16
        with self.tik_instance.new_stmt_scope():
            loop_times = max((self.dim + 255) // 256 * 8, 8)
            code_num_ub_each_loop = self.code_num_each_loop // loop_times
            if flag == 0:
                with self.tik_instance.for_range(0, loop_times, thread_num=2) as loop_ub:
                    # move y from out to UB
                    code_ub = self.tik_instance.Tensor("uint8",
                                                       (self.dim // 16, code_num_ub_each_loop // 16, 16, 16),
                                                       name="code_ub", scope=tik.scope_ubuf)
                    code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                            (self.dim // 16, code_num_ub_each_loop // 16, 16, 16),
                                                            name="code_ub_fp16", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, self.dim // 16) as i:
                        self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                                    self.input_code_gm[(aicore_move_offset + code_move_offset
                                                                        + loop_ub * code_num_ub_each_loop) // 16,
                                                                       i, 0, 0],
                                                    0, code_num_ub_each_loop // 16, 8, self.dim // 2 - 8, 0)

                    # y do conv from uint8 to fp16
                    vconv_loop = (code_num_ub_each_loop * self.dim) // (self.fp16_mask * 255)
                    vconv_offset = 0
                    if vconv_loop > 0:
                        for conv_index in range(vconv_loop):
                            vconv_offset = conv_index * self.fp16_mask * 255
                            self.tik_instance.vconv(self.fp16_mask, "none",
                                                    code_ub_fp16[vconv_offset],
                                                    code_ub[vconv_offset],
                                                    255, 1, 1, 8, 4)
                        vconv_offset += self.fp16_mask * 255

                    vconv_repeat_time = (code_num_ub_each_loop * self.dim) % (self.fp16_mask * 255) // self.fp16_mask
                    if vconv_repeat_time > 0:
                        self.tik_instance.vconv(self.fp16_mask, "none",
                                                code_ub_fp16[vconv_offset],
                                                code_ub[vconv_offset],
                                                vconv_repeat_time, 1, 1, 8, 4)

                    # move y from ub to L1
                    code_l1 = self.tik_instance.Tensor("float16",
                                                       (self.dim // 16, code_num_ub_each_loop, 16),
                                                       name="code_l1", scope=tik.scope_cbuf)
                    self.tik_instance.data_move(code_l1, code_ub_fp16, 0, self.dim // 16, code_num_ub_each_loop, 0, 0)

                    # cal (x*d)*y
                    self.tik_instance.matmul(inner_product_l0c[(loop_ub * code_num_ub_each_loop * queries_align):],
                                             data_xd_l1, code_l1,
                                             queries_align, self.dim, code_num_ub_each_loop)

                # move v*v to ub
                self.tik_instance.data_move(code_l2_ub_fp16,
                                            self.input_precomputed_gm[aicore_move_offset + code_move_offset],
                                            0, 1, self.code_num_each_loop // 8, 0, 0)
            else:
                code_loop = code_move_num_ub // code_num_ub_each_loop
                with self.tik_instance.for_range(0, code_loop, thread_num=2) as loop_ub:
                    # move y from out to UB
                    code_ub = self.tik_instance.Tensor("uint8",
                                                       (self.dim // 16, code_num_ub_each_loop // 16, 16, 16),
                                                       name="code_ub", scope=tik.scope_ubuf)
                    code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                            (self.dim // 16, code_num_ub_each_loop // 16, 16, 16),
                                                            name="code_ub_fp16", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, self.dim // 16) as i:
                        self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                                    self.input_code_gm[(aicore_move_offset + code_move_offset +
                                                                        loop_ub * code_num_ub_each_loop) // 16,
                                                                       i, 0, 0],
                                                    0, code_num_ub_each_loop // 16, 8, self.dim // 2 - 8, 0)

                    # y do conv from uint8 to fp16
                    vconv_loop = (code_num_ub_each_loop * self.dim) // (self.fp16_mask * 255)
                    vconv_offset = 0
                    if vconv_loop > 0:
                        for conv_index in range(vconv_loop):
                            vconv_offset = conv_index * self.fp16_mask * 255
                            self.tik_instance.vconv(self.fp16_mask, "none",
                                                    code_ub_fp16[vconv_offset],
                                                    code_ub[vconv_offset], 255,
                                                    1, 1, 8, 4)
                        vconv_offset += self.fp16_mask * 255

                    vconv_repeat_time = (code_num_ub_each_loop * self.dim) % (self.fp16_mask * 255) // self.fp16_mask
                    if vconv_repeat_time > 0:
                        self.tik_instance.vconv(self.fp16_mask, "none",
                                                code_ub_fp16[vconv_offset],
                                                code_ub[vconv_offset],
                                                vconv_repeat_time, 1, 1, 8, 4)

                    # move y from ub to L1
                    code_l1 = self.tik_instance.Tensor("float16",
                                                       (self.dim // 16, code_num_ub_each_loop, 16),
                                                       name="code_l1", scope=tik.scope_cbuf)
                    self.tik_instance.data_move(code_l1, code_ub_fp16,
                                                0, self.dim // 16, code_num_ub_each_loop, 0, 0)

                    self.tik_instance.matmul(inner_product_l0c[(loop_ub * code_num_ub_each_loop * queries_align):],
                                             data_xd_l1, code_l1,
                                             queries_align, self.dim, code_num_ub_each_loop)

                code_last = code_move_num_ub % code_num_ub_each_loop
                with self.tik_instance.if_scope(code_last > 0):
                    # move y from out to UB
                    code_ub = self.tik_instance.Tensor("uint8",
                                                       (self.dim // 16, code_num_ub_each_loop // 16, 16, 16),
                                                       name="code_ub", scope=tik.scope_ubuf)
                    code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                            (self.dim // 16, code_num_ub_each_loop // 16, 16, 16),
                                                            name="code_ub_fp16", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, self.dim // 16) as i:
                        self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                                    self.input_code_gm[(aicore_move_offset + code_move_offset
                                                                        + code_loop * code_num_ub_each_loop) // 16,
                                                                       i, 0, 0],
                                                    0, code_last // 16, 8, self.dim // 2 - 8, 0)

                    with self.tik_instance.for_range(0, self.dim // 16) as i:
                        vconv_repeat_time = (code_last * 16) // self.fp16_mask
                        self.tik_instance.vconv(self.fp16_mask, "none",
                                                code_ub_fp16[i, 0, 0, 0],
                                                code_ub[i, 0, 0, 0],
                                                vconv_repeat_time, 1, 1, 8, 4)

                    # move y from ub to L1
                    code_l1 = self.tik_instance.Tensor("float16",
                                                       (self.dim // 16, code_num_ub_each_loop, 16),
                                                       name="code_l1", scope=tik.scope_cbuf)
                    self.tik_instance.data_move(code_l1, code_ub_fp16,
                                                0, self.dim // 16, code_last,
                                                (code_num_ub_each_loop - code_last),
                                                (code_num_ub_each_loop - code_last))
                    self.tik_instance.matmul(inner_product_l0c[(code_loop * code_num_ub_each_loop * queries_align):],
                                             data_xd_l1, code_l1,
                                             queries_align, self.dim, code_num_ub_each_loop)

                # mov code l2 from out to UB
                self.tik_instance.data_move(code_l2_ub_fp16,
                                            self.input_precomputed_gm[aicore_move_offset + code_move_offset],
                                            0, 1, code_move_num_ub // 8, 0, 0)

        code_out_num = self.code_num_each_loop // 2
        if flag == 0:
            with self.tik_instance.for_range(0, 2, thread_num=1) as i:
                add_ub = self.tik_instance.Tensor("float32",
                                                  (queries_move_num, code_out_num),
                                                  name="add_ub", scope=tik.scope_ubuf)
                repeat = code_out_num // self.fp32_mask
                for loop_queries in range(0, queries_move_num):
                    # cal x*x - 2/255*x*(0.5*d + 255*m)+v*v
                    self.tik_instance.vadds(self.fp32_mask,
                                            add_ub[loop_queries, 0],
                                            code_l2_ub_fp16[code_out_num * i], add_scalar_list[loop_queries],
                                            repeat, 1, 1, 8, 8)
                # mov xy from L0-C to UB
                inner_product_ub = self.tik_instance.Tensor("float32",
                                                            (code_out_num // 16, queries_align, 16),
                                                            name="inner_product_ub", scope=tik.scope_ubuf)
                # move (x*d)*y from l0 to ub
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c[i * code_out_num // 16, 0, 0],
                                            0, 1, code_out_num * queries_align // 256, 0, 0)

                thread_num_need = 2 if queries_move_num > 1 else 1
                with self.tik_instance.for_range(0, queries_move_num, thread_num=thread_num_need) as loop_queries:
                    # cal x*x -2/255*x*(0.5*d+255*m)+v*v-2/255*(x*d)*y
                    self.tik_instance.vaxpy(16,
                                            add_ub[loop_queries, 0],
                                            inner_product_ub[0, loop_queries, 0],
                                            self.coeff,
                                            code_out_num // 16, 1, 1, 2, queries_align * 2)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (queries_move_num, code_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)
                vconv_times = 2 if queries_move_num > 1 else 1
                query_num_each_loop = queries_move_num // vconv_times
                vconv_repeat_times = query_num_each_loop * code_out_num // self.fp32_mask

                with self.tik_instance.for_range(0, vconv_times, thread_num=vconv_times) as vconv_loop:
                    self.tik_instance.vconv(self.fp32_mask, "none",
                                            dst_ub[query_num_each_loop * vconv_loop, 0],
                                            add_ub[query_num_each_loop * vconv_loop, 0],
                                            vconv_repeat_times, 1, 1, 4, 8)

                self.tik_instance.data_move(self.output_distances_gm[queries_move_offset,
                                                                     aicore_move_offset + code_move_offset
                                                                     + i * code_out_num],
                                            dst_ub,
                                            0, queries_move_num, code_out_num // 16, 0,
                                            (self.code_num - code_out_num) // 16)

                min_size = code_out_num // self.min_batch * 2
                dst_min_ub = self.tik_instance.Tensor("float16",
                                                      (queries_move_num, min_size),
                                                      name="dst_min_ub", scope=tik.scope_ubuf)

                vcmin_loops = 2 if queries_move_num > 1 else 1
                vcmin_repeat_times = query_num_each_loop * code_out_num // self.min_batch
                with self.tik_instance.for_range(0, vcmin_loops, thread_num=vcmin_loops) as vcmin_loop:
                    self.tik_instance.vcmin(self.min_batch,
                                            dst_min_ub[vcmin_loop * query_num_each_loop, 0],
                                            dst_ub[vcmin_loop * query_num_each_loop, 0],
                                            vcmin_repeat_times, 1, 1, self.min_batch // 16)
                self.tik_instance.data_move(self.output_mins_gm[queries_move_offset,
                                                                (aicore_move_offset + code_move_offset
                                                                 + i * code_out_num) // self.min_batch * 2],
                                            dst_min_ub,
                                            0, queries_move_num, (code_out_num // self.min_batch) // 8, 0,
                                            (self.code_num - code_out_num) // self.min_batch // 8)
        else:
            code_loops = (code_move_num_ub + code_out_num - 1) // code_out_num
            with self.tik_instance.for_range(0, code_loops, thread_num=1) as i:
                add_ub = self.tik_instance.Tensor("float32",
                                                  (queries_move_num, code_out_num),
                                                  name="add_ub", scope=tik.scope_ubuf)
                repeat = code_out_num // self.fp32_mask
                for loop_queries in range(0, queries_move_num):
                    # cal x*x - 2/255*x*(0.5*d + 255*m)+v*v
                    self.tik_instance.vadds(self.fp32_mask,
                                            add_ub[loop_queries, 0],
                                            code_l2_ub_fp16[code_out_num * i], add_scalar_list[loop_queries],
                                            repeat, 1, 1, 8, 8)
                # mov xy from L0-C to UB
                inner_product_ub = self.tik_instance.Tensor("float32",
                                                            (code_out_num // 16, queries_align, 16),
                                                            name="inner_product_ub", scope=tik.scope_ubuf)
                # move (x*d)*y from l0 to ub
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c[i * code_out_num // 16, 0, 0],
                                            0, 1, code_out_num * queries_align // 256, 0, 0)

                thread_num_need = 2 if queries_move_num > 1 else 1
                with self.tik_instance.for_range(0, queries_move_num, thread_num=thread_num_need) as loop_queries:
                    # cal 2/255*(x*d)
                    self.tik_instance.vaxpy(16,
                                            add_ub[loop_queries, 0],
                                            inner_product_ub[0, loop_queries, 0],
                                            self.coeff, code_out_num // 16, 1, 1, 2, queries_align * 2)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (queries_move_num, code_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)
                vconv_times = 2 if queries_move_num > 1 else 1
                query_num_each_loop = queries_move_num // vconv_times
                vconv_repeat_time = query_num_each_loop * code_out_num // self.fp32_mask
                with self.tik_instance.for_range(0, vconv_times, thread_num=vconv_times) as vconv_loop:
                    self.tik_instance.vconv(self.fp32_mask, "none",
                                            dst_ub[query_num_each_loop * vconv_loop, 0],
                                            add_ub[query_num_each_loop * vconv_loop, 0],
                                            vconv_repeat_time, 1, 1, 4, 8)
                self.tik_instance.data_move(self.output_distances_gm[queries_move_offset,
                                                                     aicore_move_offset + code_move_offset
                                                                     + i * code_out_num],
                                            dst_ub,
                                            0, queries_move_num, code_out_num // 16, 0,
                                            (self.code_num - code_out_num) // 16)

                min_size = code_out_num // self.min_batch * 2
                dst_min_ub = self.tik_instance.Tensor("float16",
                                                      (queries_move_num, min_size),
                                                      name="dst_min_ub", scope=tik.scope_ubuf)
                with self.tik_instance.if_scope(i == code_loops - 1):
                    code_last_num = code_move_num - i * code_out_num
                    thread_num = 2 if queries_move_num > 1 else 1
                    with self.tik_instance.for_range(0, queries_move_num, thread_num=thread_num) as query:
                        vcmin_repeat_times = code_last_num // self.min_batch
                        offset = 0
                        with self.tik_instance.if_scope(vcmin_repeat_times > 0):
                            self.tik_instance.vcmin(self.min_batch,
                                                    dst_min_ub[query, 0],
                                                    dst_ub[query, 0],
                                                    vcmin_repeat_times, 1, 1, self.min_batch // 16)
                            offset += vcmin_repeat_times * self.min_batch
                        vcmin_remain = code_last_num % self.min_batch
                        with self.tik_instance.if_scope(vcmin_remain > 0):
                            self.tik_instance.vcmin(vcmin_remain,
                                                    dst_min_ub[query, vcmin_repeat_times * 2],
                                                    dst_ub[query, offset],
                                                    1, 1, 1, 1)
                        self.tik_instance.data_move(self.output_mins_gm[queries_move_offset,
                                                                        (aicore_move_offset + code_move_offset
                                                                         + i * code_out_num) // self.min_batch * 2],
                                                    dst_min_ub,
                                                    0, queries_move_num, code_out_num // self.min_batch // 8, 0,
                                                    (self.code_num - code_out_num) // self.min_batch // 8)
                with self.tik_instance.else_scope():
                    vcmin_loops = 2 if queries_move_num > 1 else 1
                    vcmin_repeat_times = query_num_each_loop * code_out_num // self.min_batch
                    with self.tik_instance.for_range(0, vcmin_loops, thread_num=vcmin_loops) as vcmin_loop:
                        self.tik_instance.vcmin(self.min_batch,
                                                dst_min_ub[vcmin_loop * query_num_each_loop, 0],
                                                dst_ub[vcmin_loop * query_num_each_loop, 0],
                                                vcmin_repeat_times, 1, 1, self.min_batch // 16)

                    self.tik_instance.data_move(self.output_mins_gm[queries_move_offset,
                                                                    (aicore_move_offset + code_move_offset
                                                                     + i * code_out_num) // self.min_batch * 2],
                                                dst_min_ub,
                                                0, queries_move_num, (code_out_num // self.min_batch) // 8, 0,
                                                (self.code_num - code_out_num) // self.min_batch // 8)

    def compute_and_min(self):
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            with self.tik_instance.new_stmt_scope():
                self.cal_num_each_core()
                # compute coarse centroids num and move offset every core
                aicore_code_num = self.tik_instance.Scalar(dtype="uint32",
                                                           name="aicore_code_num",
                                                           init_value=0)
                with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                    aicore_code_num.set_as(self.code_num_each_core)
                with self.tik_instance.else_scope():
                    aicore_code_num.set_as(self.code_num_last_core)

                queries_loop_time = self.queries_num // self.queries_num_each_loop
                if queries_loop_time > 0:
                    with self.tik_instance.for_range(0, queries_loop_time) as loop_queries:
                        self.distance_compute_each_loop(block_index * self.code_num_each_core, aicore_code_num,
                                                        loop_queries * self.queries_num_each_loop,
                                                        self.queries_num_each_loop)

                queries_last_num = self.queries_num % self.queries_num_each_loop
                if queries_last_num > 0:
                    self.distance_compute_each_loop(block_index * self.code_num_each_core, aicore_code_num,
                                                    queries_loop_time * self.queries_num_each_loop, queries_last_num)

                one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
                flag_ub = self.tik_instance.Tensor("uint16", (16,), name="flag_ub", scope=tik.scope_ubuf)
                flag_ub[0].set_as(one)

                self.tik_instance.data_move(self.output_flag_gm[block_index * 16], flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.compute_and_min()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[
                self.input_queries_gm, self.input_code_gm,
                self.input_precomputed_gm, self.input_diff_gm,
                self.input_min_gm, self.input_actual_code_num_gm
            ],
            outputs=[self.output_distances_gm, self.output_mins_gm, self.output_flag_gm])

        return self.tik_instance


def distance_compute_sq8_min64(input_queries,
                               input_code,
                               input_precomputed,
                               input_diff,
                               input_min,
                               input_actual_code_num,
                               output_distances,
                               output_mins,
                               output_flag,
                               kernel_name="distance_compute_sq8_min64"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_code : dict
        shape and dtype of coding vector
    input_precomputed : dict
        shape and dtype of precomputed L2 distance of coding vector
    input_diff: dict
        shape and dtype of coding coefficient
    input_min: dict
        shape and dtype of coding offset
    input_actual_code_num: dict
        shape and dtype of actual code num,
        shape must be (8,) and dtype must be uint32
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_mins : dict
        shape and dtype of mins
    output_flag: dict
        shape and dtype of output flag,
        shape must be (32,) and dtype must be uint16
    kernel_name : str
        kernel name, default value is "distance_compute"

    Returns
    -------
    None
    """
    distance_compute_sq8_min64 = DistanceComputeSQ8Min64(input_queries, input_code,
                                                         input_precomputed, input_diff,
                                                         input_min, input_actual_code_num,
                                                         output_distances, output_mins, output_flag,
                                                         kernel_name)
    tik_instance = distance_compute_sq8_min64.get_tik_instance()
    return tik_instance
