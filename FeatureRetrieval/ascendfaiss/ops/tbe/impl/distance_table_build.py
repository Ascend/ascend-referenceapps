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


class DistanceTableBuild():
    def __init__(self,
                 input_queries,
                 input_pq_centroids,
                 input_sequence_number,
                 input_coarse_centroids,
                 output_distance_table,
                 output_flag,
                 kernel_name="distance_table_build"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_pq_centroids = input_pq_centroids.get("shape")
        self.dtype_pq_centroids = input_pq_centroids.get("dtype")
        self.shape_sequence_number = input_sequence_number.get("shape")
        self.dtype_sequence_number = input_sequence_number.get("dtype")
        self.shape_coarse_centroids = input_coarse_centroids.get("shape")
        self.dtype_coarse_centroids = input_coarse_centroids.get("dtype")
        self.shape_distance_table = output_distance_table.get("shape")
        self.dtype_distance_table = output_distance_table.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype").lower()
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.sub_quantizers_num, pq_dim = self.shape_pq_centroids
        self.sub_dim = self.dim // self.sub_quantizers_num
        self.pq_centroids_num = pq_dim // self.sub_dim
        _, self.closest_num = self.shape_sequence_number

        # check parameter
        if self.closest_num % 8 != 0:
            raise RuntimeError("the closest num of IVF centroids to \
                                the query vector must be a multiple of 8")
        if self.dim % 16 != 0 or self.pq_centroids_num % 16 != 0:
            raise RuntimeError("feature dim and pq centroids num must be a \
                                multiple of 16")
        if self.sub_dim > 128:
            raise RuntimeError("sub_dim must be not greater than 128")
        if self.sub_dim != 4 and self.sub_dim != 8 and self.sub_dim % 16 != 0:
            raise RuntimeError("sub_dim must be 4 or 8 or a multiple of 16")
        if self.shape_flag[0] != 16:
            raise RuntimeError("output flag num must be 16")

        # set max vector mask
        self.vector_mask_max = 128

        self.aicore_use = 2

        # The target machine is defined by the Dprofile function,
        # and the TIK DSL container is constructed by the Tik function.
        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))

        self.zero = self.tik_instance.Scalar('float16')
        self.zero.set_as(0)

        # creat input tensor: input_queries_gm, input_pq_centroids_gm,
        # input_sequence_number_gm and input_coarse_centroids_gm
        # output tensor: output_distance_table_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_pq_centroids_gm = self.tik_instance.Tensor(
            self.dtype_pq_centroids,
            self.shape_pq_centroids,
            name="input_pq_centroids_gm",
            scope=tik.scope_gm)
        self.input_sequence_number_gm = self.tik_instance.Tensor(
            self.dtype_sequence_number,
            self.shape_sequence_number,
            name="input_sequence_number_gm",
            scope=tik.scope_gm)
        self.input_coarse_centroids_gm = self.tik_instance.Tensor(
            self.dtype_coarse_centroids,
            self.shape_coarse_centroids,
            name="input_coarse_centroids_gm",
            scope=tik.scope_gm)
        self.output_distance_table_gm = self.tik_instance.Tensor(
            self.dtype_distance_table,
            self.shape_distance_table,
            name="output_distance_table_gm",
            scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(self.dtype_flag,
                                                       self.shape_flag,
                                                       name="output_flag_gm",
                                                       scope=tik.scope_gm)

    # compute vadds mask of vector count
    def compute_mask(self, num, mode, tag):
        if num >= 64:
            num -= 64

        if mode == 0:
            if tag == 0:
                base = 15
            elif tag == 1:
                base = 240
            elif tag == 2:
                base = 3840
            elif tag == 3:
                base = 61440
        elif mode == 1:
            if tag == 0:
                base = 255
            elif tag == 1:
                base = 65280

        mask = 0
        for i in range(num // 16):
            mask += base << (i * 16)
        return mask

    # compute distance table build which sub dim is 4
    def distance_table_build_compute_subdim_4(self, loop_queries, sequence_ub,
                                              aicore_move_offset,
                                              aicore_sequence_num):
        queries_pad_ub = self.tik_instance.Tensor("float16", (self.dim * 4, ),
                                                  name="queries_pad_ub",
                                                  scope=tik.scope_ubuf)
        dup_repeat_time = (self.dim * 4) // self.vector_mask_max
        dup_offset = 0
        if dup_repeat_time > 0:
            self.tik_instance.vector_dup(self.vector_mask_max,
                                         queries_pad_ub[0], self.zero,
                                         dup_repeat_time, 1, 8)
            dup_offset += dup_repeat_time * self.vector_mask_max

        dup_last_num = (self.dim * 4) % self.vector_mask_max
        if dup_last_num > 0:
            self.tik_instance.vector_dup(dup_last_num,
                                         queries_pad_ub[dup_offset], self.zero,
                                         1, 1, 8)

        # move queries from out to UB
        queries_ub = self.tik_instance.Tensor("float16", (self.dim, ),
                                              name="queries_ub",
                                              scope=tik.scope_ubuf)
        self.tik_instance.data_move(queries_ub[0],
                                    self.input_queries_gm[loop_queries, 0], 0,
                                    1, self.dim // 16, 0, 0)

        # compute diastance per list
        thread_num_need = 1
        with self.tik_instance.if_scope(aicore_sequence_num > 1):
            thread_num_need = 2
        with self.tik_instance.for_range(
                0, aicore_sequence_num,
                thread_num=thread_num_need) as loop_sequence:
            coarse_centroids_ub = self.tik_instance.Tensor(
                "float16", (self.dim, ),
                name="coarse_centroids_ub",
                scope=tik.scope_ubuf)
            residual_ub = self.tik_instance.Tensor("float16", (self.dim, ),
                                                   name="residual_ub",
                                                   scope=tik.scope_ubuf)
            sequence_id = self.tik_instance.Scalar('int32')
            sequence_id.set_as(sequence_ub[loop_queries,
                                           aicore_move_offset + loop_sequence])

            # x-yc
            self.tik_instance.data_move(
                coarse_centroids_ub[0],
                self.input_coarse_centroids_gm[sequence_id,
                                               0], 0, 1, self.dim // 16, 0, 0)

            vsub_repeat_time = self.dim // self.vector_mask_max
            vsub_offset = 0
            if vsub_repeat_time > 0:
                self.tik_instance.vsub(self.vector_mask_max, residual_ub[0],
                                       queries_ub[0], coarse_centroids_ub[0],
                                       vsub_repeat_time, 1, 1, 1, 8, 8, 8)
                vsub_offset += vsub_repeat_time * self.vector_mask_max

            vsub_last_num = self.dim % self.vector_mask_max
            if vsub_last_num > 0:
                self.tik_instance.vsub(vsub_last_num, residual_ub[vsub_offset],
                                       queries_ub[vsub_offset],
                                       coarse_centroids_ub[vsub_offset], 1, 1,
                                       1, 1, 8, 8, 8)

            # expand x-yc per sub num
            vadds_repeat_time = self.dim // self.vector_mask_max
            vadds_offset = 0
            if vadds_repeat_time > 0:
                self.tik_instance.vadds([4222189076152335, 4222189076152335],
                                        queries_pad_ub[0], residual_ub[0], 0,
                                        vadds_repeat_time, 4, 1, 32, 8)
                self.tik_instance.vadds([67555025218437360, 67555025218437360],
                                        queries_pad_ub[16], residual_ub[0], 0,
                                        vadds_repeat_time, 4, 1, 32, 8)
                self.tik_instance.vadds(
                    [1080880403494997760, 1080880403494997760],
                    queries_pad_ub[32], residual_ub[0], 0, vadds_repeat_time,
                    4, 1, 32, 8)
                self.tik_instance.vadds(
                    [17294086455919964160, 17294086455919964160],
                    queries_pad_ub[48], residual_ub[0], 0, vadds_repeat_time,
                    4, 1, 32, 8)
                vadds_offset += vadds_repeat_time * self.vector_mask_max

            vadds_last_num = self.dim % self.vector_mask_max
            if vadds_last_num > 0:
                mask0 = self.compute_mask(vadds_last_num, 0, 0)
                mask1 = self.compute_mask(vadds_last_num, 0, 1)
                mask2 = self.compute_mask(vadds_last_num, 0, 2)
                mask3 = self.compute_mask(vadds_last_num, 0, 3)
                if vadds_last_num < 64:
                    vadds_mask0 = [0, mask0]
                    vadds_mask1 = [0, mask1]
                    vadds_mask2 = [0, mask2]
                    vadds_mask3 = [0, mask3]
                else:
                    vadds_mask0 = [mask0, 4222189076152335]
                    vadds_mask1 = [mask1, 67555025218437360]
                    vadds_mask2 = [mask2, 1080880403494997760]
                    vadds_mask3 = [mask3, 17294086455919964160]

                self.tik_instance.vadds(vadds_mask0,
                                        queries_pad_ub[vadds_offset * 4],
                                        residual_ub[vadds_offset], 0, 1, 4, 1,
                                        32, 8)
                self.tik_instance.vadds(vadds_mask1,
                                        queries_pad_ub[vadds_offset * 4 + 16],
                                        residual_ub[vadds_offset], 0, 1, 4, 1,
                                        32, 8)
                self.tik_instance.vadds(vadds_mask2,
                                        queries_pad_ub[vadds_offset * 4 + 32],
                                        residual_ub[vadds_offset], 0, 1, 4, 1,
                                        32, 8)
                self.tik_instance.vadds(vadds_mask3,
                                        queries_pad_ub[vadds_offset * 4 + 48],
                                        residual_ub[vadds_offset], 0, 1, 4, 1,
                                        32, 8)

            mul_pad_ub = self.tik_instance.Tensor("float16",
                                                  (self.pq_centroids_num, 16),
                                                  name="mul_pad_ub",
                                                  scope=tik.scope_ubuf)
            dup_repeat_time = (self.pq_centroids_num * 16) // \
                self.vector_mask_max
            if dup_repeat_time > 0:
                self.tik_instance.vector_dup(self.vector_mask_max,
                                             mul_pad_ub[0], self.zero,
                                             dup_repeat_time, 1, 8)

            # compute diastance per quantizers
            with self.tik_instance.for_range(0,
                                             self.sub_quantizers_num,
                                             thread_num=2) as loop_quantizers:
                data1 = self.tik_instance.Scalar(dtype="float16")
                data2 = self.tik_instance.Scalar(dtype="float16")
                data3 = self.tik_instance.Scalar(dtype="float16")
                data4 = self.tik_instance.Scalar(dtype="float16")

                pq_centroids_ub = self.tik_instance.Tensor(
                    "float16", (self.pq_centroids_num, self.sub_dim),
                    name="pq_centroids_ub",
                    scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    pq_centroids_ub[0],
                    self.input_pq_centroids_gm[loop_quantizers, 0], 0, 1,
                    self.pq_centroids_num * self.sub_dim // 16, 0, 0)

                sub_residual_ub = self.tik_instance.Tensor(
                    "float16", (16, ),
                    name="sub_residual_ub",
                    scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    sub_residual_ub[0], queries_pad_ub[loop_quantizers * 16],
                    0, 1, 1, 0, 0)

                # copy sub residual per sub num
                with self.tik_instance.if_scope(loop_quantizers % 4 == 0):
                    data1.set_as(sub_residual_ub[0])
                    data2.set_as(sub_residual_ub[1])
                    data3.set_as(sub_residual_ub[2])
                    data4.set_as(sub_residual_ub[3])
                    sub_residual_ub[4].set_as(data1)
                    sub_residual_ub[5].set_as(data2)
                    sub_residual_ub[6].set_as(data3)
                    sub_residual_ub[7].set_as(data4)
                    sub_residual_ub[8].set_as(data1)
                    sub_residual_ub[9].set_as(data2)
                    sub_residual_ub[10].set_as(data3)
                    sub_residual_ub[11].set_as(data4)
                    sub_residual_ub[12].set_as(data1)
                    sub_residual_ub[13].set_as(data2)
                    sub_residual_ub[14].set_as(data3)
                    sub_residual_ub[15].set_as(data4)
                with self.tik_instance.if_scope(loop_quantizers % 4 == 1):
                    data1.set_as(sub_residual_ub[4])
                    data2.set_as(sub_residual_ub[5])
                    data3.set_as(sub_residual_ub[6])
                    data4.set_as(sub_residual_ub[7])
                    sub_residual_ub[0].set_as(data1)
                    sub_residual_ub[1].set_as(data2)
                    sub_residual_ub[2].set_as(data3)
                    sub_residual_ub[3].set_as(data4)
                    sub_residual_ub[8].set_as(data1)
                    sub_residual_ub[9].set_as(data2)
                    sub_residual_ub[10].set_as(data3)
                    sub_residual_ub[11].set_as(data4)
                    sub_residual_ub[12].set_as(data1)
                    sub_residual_ub[13].set_as(data2)
                    sub_residual_ub[14].set_as(data3)
                    sub_residual_ub[15].set_as(data4)
                with self.tik_instance.if_scope(loop_quantizers % 4 == 2):
                    data1.set_as(sub_residual_ub[8])
                    data2.set_as(sub_residual_ub[9])
                    data3.set_as(sub_residual_ub[10])
                    data4.set_as(sub_residual_ub[11])
                    sub_residual_ub[0].set_as(data1)
                    sub_residual_ub[1].set_as(data2)
                    sub_residual_ub[2].set_as(data3)
                    sub_residual_ub[3].set_as(data4)
                    sub_residual_ub[4].set_as(data1)
                    sub_residual_ub[5].set_as(data2)
                    sub_residual_ub[6].set_as(data3)
                    sub_residual_ub[7].set_as(data4)
                    sub_residual_ub[12].set_as(data1)
                    sub_residual_ub[13].set_as(data2)
                    sub_residual_ub[14].set_as(data3)
                    sub_residual_ub[15].set_as(data4)
                with self.tik_instance.else_scope():
                    data1.set_as(sub_residual_ub[12])
                    data2.set_as(sub_residual_ub[13])
                    data3.set_as(sub_residual_ub[14])
                    data4.set_as(sub_residual_ub[15])
                    sub_residual_ub[0].set_as(data1)
                    sub_residual_ub[1].set_as(data2)
                    sub_residual_ub[2].set_as(data3)
                    sub_residual_ub[3].set_as(data4)
                    sub_residual_ub[4].set_as(data1)
                    sub_residual_ub[5].set_as(data2)
                    sub_residual_ub[6].set_as(data3)
                    sub_residual_ub[7].set_as(data4)
                    sub_residual_ub[8].set_as(data1)
                    sub_residual_ub[9].set_as(data2)
                    sub_residual_ub[10].set_as(data3)
                    sub_residual_ub[11].set_as(data4)

                mul_ub = self.tik_instance.Tensor(
                    "float16", (self.pq_centroids_num, self.sub_dim),
                    name="mul_ub",
                    scope=tik.scope_ubuf)
                dst_ub = self.tik_instance.Tensor("float16",
                                                  (self.pq_centroids_num, ),
                                                  name="dst_ub",
                                                  scope=tik.scope_ubuf)

                # sub residual do sub with pq centroids
                vsub_loop = (self.pq_centroids_num * self.sub_dim) // \
                    (16 * 255)
                vsub_offset = self.tik_instance.Scalar(dtype="int32")
                vsub_offset.set_as(0)
                if vsub_loop > 0:
                    with self.tik_instance.for_range(0, vsub_loop):
                        self.tik_instance.vsub(
                            16, mul_ub[vsub_offset // self.sub_dim,
                                       vsub_offset % self.sub_dim],
                            sub_residual_ub[0],
                            pq_centroids_ub[vsub_offset // self.sub_dim,
                                            vsub_offset % self.sub_dim], 255,
                            1, 1, 1, 1, 0, 1)
                        vsub_offset.set_as(vsub_offset + 255 * 16)

                vsub_repeat_time = (self.pq_centroids_num * self.sub_dim) % \
                    (16 * 255) // 16
                if vsub_repeat_time > 0:
                    self.tik_instance.vsub(
                        16, mul_ub[vsub_offset // self.sub_dim,
                                   vsub_offset % self.sub_dim],
                        sub_residual_ub[0],
                        pq_centroids_ub[vsub_offset // self.sub_dim,
                                        vsub_offset % self.sub_dim],
                        vsub_repeat_time, 1, 1, 1, 1, 0, 1)

                # do mul
                vmul_repeat_time = (self.pq_centroids_num * self.sub_dim) // \
                    self.vector_mask_max
                if vmul_repeat_time > 0:
                    self.tik_instance.vmul(self.vector_mask_max, mul_ub[0],
                                           mul_ub[0], mul_ub[0],
                                           vmul_repeat_time, 1, 1, 1, 8, 8, 8)

                vadds_repeat_time = (self.pq_centroids_num * self.sub_dim) // \
                    self.vector_mask_max
                if vadds_repeat_time > 0:
                    self.tik_instance.vadds(
                        [4222189076152335, 4222189076152335], mul_pad_ub[0, 0],
                        mul_ub[0], 0, vadds_repeat_time, 4, 1, 32, 8)
                    self.tik_instance.vadds(
                        [67555025218437360, 67555025218437360], mul_pad_ub[1,
                                                                           0],
                        mul_ub[0], 0, vadds_repeat_time, 4, 1, 32, 8)
                    self.tik_instance.vadds(
                        [1080880403494997760, 1080880403494997760],
                        mul_pad_ub[2, 0], mul_ub[0], 0, vadds_repeat_time, 4,
                        1, 32, 8)
                    self.tik_instance.vadds(
                        [17294086455919964160, 17294086455919964160],
                        mul_pad_ub[3, 0], mul_ub[0], 0, vadds_repeat_time, 4,
                        1, 32, 8)

                # do vcgadd
                vcgadd_repeat_time = (self.pq_centroids_num * 16) // \
                    self.vector_mask_max
                if vcgadd_repeat_time > 0:
                    self.tik_instance.vcgadd(self.vector_mask_max, dst_ub[0],
                                             mul_pad_ub[0], vcgadd_repeat_time,
                                             1, 1, 8)

                self.tik_instance.data_move(
                    self.output_distance_table_gm[loop_queries,
                                                  aicore_move_offset +
                                                  loop_sequence,
                                                  loop_quantizers, 0],
                    dst_ub[0], 0, 1, self.pq_centroids_num // 16, 0, 0)

    # compute distance table build which sub dim is 8
    def distance_table_build_compute_subdim_8(self, loop_queries, sequence_ub,
                                              aicore_move_offset,
                                              aicore_sequence_num):
        queries_pad_ub = self.tik_instance.Tensor("float16", (self.dim * 2, ),
                                                  name="queries_pad_ub",
                                                  scope=tik.scope_ubuf)
        dup_repeat_time = (self.dim * 2) // self.vector_mask_max
        dup_offset = 0
        if dup_repeat_time > 0:
            self.tik_instance.vector_dup(self.vector_mask_max,
                                         queries_pad_ub[0], self.zero,
                                         dup_repeat_time, 1, 8)
            dup_offset += dup_repeat_time * self.vector_mask_max

        dup_last_num = (self.dim * 2) % self.vector_mask_max
        if dup_last_num > 0:
            self.tik_instance.vector_dup(dup_last_num,
                                         queries_pad_ub[dup_offset], self.zero,
                                         1, 1, 8)

        # move queries from out to UB
        queries_ub = self.tik_instance.Tensor("float16", (self.dim, ),
                                              name="queries_ub",
                                              scope=tik.scope_ubuf)
        self.tik_instance.data_move(queries_ub[0],
                                    self.input_queries_gm[loop_queries, 0], 0,
                                    1, self.dim // 16, 0, 0)

        # compute diastance per list
        thread_num_need = 1
        with self.tik_instance.if_scope(aicore_sequence_num > 1):
            thread_num_need = 2
        with self.tik_instance.for_range(
                0, aicore_sequence_num,
                thread_num=thread_num_need) as loop_sequence:
            coarse_centroids_ub = self.tik_instance.Tensor(
                "float16", (self.dim, ),
                name="coarse_centroids_ub",
                scope=tik.scope_ubuf)
            residual_ub = self.tik_instance.Tensor("float16", (self.dim, ),
                                                   name="residual_ub",
                                                   scope=tik.scope_ubuf)
            sequence_id = self.tik_instance.Scalar('int32')
            sequence_id.set_as(sequence_ub[loop_queries,
                                           aicore_move_offset + loop_sequence])

            # x-yc
            self.tik_instance.data_move(
                coarse_centroids_ub[0],
                self.input_coarse_centroids_gm[sequence_id,
                                               0], 0, 1, self.dim // 16, 0, 0)

            vsub_repeat_time = self.dim // self.vector_mask_max
            vsub_offset = 0
            if vsub_repeat_time > 0:
                self.tik_instance.vsub(self.vector_mask_max, residual_ub[0],
                                       queries_ub[0], coarse_centroids_ub[0],
                                       vsub_repeat_time, 1, 1, 1, 8, 8, 8)
                vsub_offset += vsub_repeat_time * self.vector_mask_max

            vsub_last_num = self.dim % self.vector_mask_max
            if vsub_last_num > 0:
                self.tik_instance.vsub(vsub_last_num, residual_ub[vsub_offset],
                                       queries_ub[vsub_offset],
                                       coarse_centroids_ub[vsub_offset], 1, 1,
                                       1, 1, 8, 8, 8)

            # expand x-yc per sub num
            vadds_repeat_time = self.dim // self.vector_mask_max
            vadds_offset = 0
            if vadds_repeat_time > 0:
                self.tik_instance.vadds([71777214294589695, 71777214294589695],
                                        queries_pad_ub[0], residual_ub[0], 0,
                                        vadds_repeat_time, 2, 1, 16, 8)
                self.tik_instance.vadds(
                    [18374966859414961920, 18374966859414961920],
                    queries_pad_ub[16], residual_ub[0], 0, vadds_repeat_time,
                    2, 1, 16, 8)
                vadds_offset += vadds_repeat_time * self.vector_mask_max

            vadds_last_num = self.dim % self.vector_mask_max
            if vadds_last_num > 0:
                mask0 = self.compute_mask(vadds_last_num, 1, 0)
                mask1 = self.compute_mask(vadds_last_num, 1, 1)
                if vadds_last_num < 64:
                    vadds_mask0 = [0, mask0]
                    vadds_mask1 = [0, mask1]
                else:
                    vadds_mask0 = [mask0, 71777214294589695]
                    vadds_mask1 = [mask1, 18374966859414961920]

                self.tik_instance.vadds(vadds_mask0,
                                        queries_pad_ub[vadds_offset * 2],
                                        residual_ub[vadds_offset], 0, 1, 2, 1,
                                        16, 8)
                self.tik_instance.vadds(vadds_mask1,
                                        queries_pad_ub[vadds_offset * 2 + 16],
                                        residual_ub[vadds_offset], 0, 1, 2, 1,
                                        16, 8)

            mul_pad_ub = self.tik_instance.Tensor("float16",
                                                  (self.pq_centroids_num, 16),
                                                  name="mul_pad_ub",
                                                  scope=tik.scope_ubuf)
            dup_repeat_time = (self.pq_centroids_num * 16) // \
                self.vector_mask_max
            if dup_repeat_time > 0:
                self.tik_instance.vector_dup(self.vector_mask_max,
                                             mul_pad_ub[0], self.zero,
                                             dup_repeat_time, 1, 8)

            # compute diastance per quantizers
            with self.tik_instance.for_range(0,
                                             self.sub_quantizers_num,
                                             thread_num=2) as loop_quantizers:
                data1 = self.tik_instance.Scalar(dtype="float16")
                data2 = self.tik_instance.Scalar(dtype="float16")
                data3 = self.tik_instance.Scalar(dtype="float16")
                data4 = self.tik_instance.Scalar(dtype="float16")
                data5 = self.tik_instance.Scalar(dtype="float16")
                data6 = self.tik_instance.Scalar(dtype="float16")
                data7 = self.tik_instance.Scalar(dtype="float16")
                data8 = self.tik_instance.Scalar(dtype="float16")

                pq_centroids_ub = self.tik_instance.Tensor(
                    "float16", (self.pq_centroids_num, self.sub_dim),
                    name="pq_centroids_ub",
                    scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    pq_centroids_ub[0],
                    self.input_pq_centroids_gm[loop_quantizers, 0], 0, 1,
                    self.pq_centroids_num * self.sub_dim // 16, 0, 0)

                sub_residual_ub = self.tik_instance.Tensor(
                    "float16", (16, ),
                    name="sub_residual_ub",
                    scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    sub_residual_ub[0], queries_pad_ub[loop_quantizers * 16],
                    0, 1, 1, 0, 0)

                # copy sub residual per sub num
                with self.tik_instance.if_scope(loop_quantizers % 2 == 0):
                    data1.set_as(sub_residual_ub[0])
                    data2.set_as(sub_residual_ub[1])
                    data3.set_as(sub_residual_ub[2])
                    data4.set_as(sub_residual_ub[3])
                    data5.set_as(sub_residual_ub[4])
                    data6.set_as(sub_residual_ub[5])
                    data7.set_as(sub_residual_ub[6])
                    data8.set_as(sub_residual_ub[7])
                    sub_residual_ub[8].set_as(data1)
                    sub_residual_ub[9].set_as(data2)
                    sub_residual_ub[10].set_as(data3)
                    sub_residual_ub[11].set_as(data4)
                    sub_residual_ub[12].set_as(data5)
                    sub_residual_ub[13].set_as(data6)
                    sub_residual_ub[14].set_as(data7)
                    sub_residual_ub[15].set_as(data8)
                with self.tik_instance.else_scope():
                    data1.set_as(sub_residual_ub[8])
                    data2.set_as(sub_residual_ub[9])
                    data3.set_as(sub_residual_ub[10])
                    data4.set_as(sub_residual_ub[11])
                    data5.set_as(sub_residual_ub[12])
                    data6.set_as(sub_residual_ub[13])
                    data7.set_as(sub_residual_ub[14])
                    data8.set_as(sub_residual_ub[15])
                    sub_residual_ub[0].set_as(data1)
                    sub_residual_ub[1].set_as(data2)
                    sub_residual_ub[2].set_as(data3)
                    sub_residual_ub[3].set_as(data4)
                    sub_residual_ub[4].set_as(data5)
                    sub_residual_ub[5].set_as(data6)
                    sub_residual_ub[6].set_as(data7)
                    sub_residual_ub[7].set_as(data8)

                mul_ub = self.tik_instance.Tensor(
                    "float16", (self.pq_centroids_num, self.sub_dim),
                    name="mul_ub",
                    scope=tik.scope_ubuf)
                dst_ub = self.tik_instance.Tensor("float16",
                                                  (self.pq_centroids_num, ),
                                                  name="dst_ub",
                                                  scope=tik.scope_ubuf)

                # sub residual do sub with pq centroids
                vsub_loop = (self.pq_centroids_num * self.sub_dim) // \
                    (16 * 255)
                vsub_offset = self.tik_instance.Scalar(dtype="int32")
                vsub_offset.set_as(0)
                if vsub_loop > 0:
                    with self.tik_instance.for_range(0, vsub_loop):
                        self.tik_instance.vsub(
                            16, mul_ub[vsub_offset // self.sub_dim,
                                       vsub_offset % self.sub_dim],
                            sub_residual_ub[0],
                            pq_centroids_ub[vsub_offset // self.sub_dim,
                                            vsub_offset % self.sub_dim], 255,
                            1, 1, 1, 1, 0, 1)
                        vsub_offset.set_as(vsub_offset + 255 * 16)

                vsub_repeat_time = (self.pq_centroids_num * self.sub_dim) % \
                    (16 * 255) // 16
                if vsub_repeat_time > 0:
                    self.tik_instance.vsub(
                        16, mul_ub[vsub_offset // self.sub_dim,
                                   vsub_offset % self.sub_dim],
                        sub_residual_ub[0],
                        pq_centroids_ub[vsub_offset // self.sub_dim,
                                        vsub_offset % self.sub_dim],
                        vsub_repeat_time, 1, 1, 1, 1, 0, 1)

                # do mul
                vmul_repeat_time = (self.pq_centroids_num * self.sub_dim) // \
                    self.vector_mask_max
                if vmul_repeat_time > 0:
                    self.tik_instance.vmul(self.vector_mask_max, mul_ub[0],
                                           mul_ub[0], mul_ub[0],
                                           vmul_repeat_time, 1, 1, 1, 8, 8, 8)

                vadds_repeat_time = (self.pq_centroids_num * self.sub_dim) // \
                    self.vector_mask_max
                if vadds_repeat_time > 0:
                    self.tik_instance.vadds(
                        [71777214294589695, 71777214294589695], mul_pad_ub[0,
                                                                           0],
                        mul_ub[0], 0, vadds_repeat_time, 2, 1, 16, 8)
                    self.tik_instance.vadds(
                        [18374966859414961920, 18374966859414961920],
                        mul_pad_ub[1, 0], mul_ub[0], 0, vadds_repeat_time, 2,
                        1, 16, 8)

                # do vcgadd
                vcgadd_repeat_time = (self.pq_centroids_num * 16) // \
                    self.vector_mask_max
                if vcgadd_repeat_time > 0:
                    self.tik_instance.vcgadd(self.vector_mask_max, dst_ub[0],
                                             mul_pad_ub[0], vcgadd_repeat_time,
                                             1, 1, 8)

                self.tik_instance.data_move(
                    self.output_distance_table_gm[loop_queries,
                                                  aicore_move_offset +
                                                  loop_sequence,
                                                  loop_quantizers, 0],
                    dst_ub[0], 0, 1, self.pq_centroids_num // 16, 0, 0)

    # compute distance table build with other sub dim
    def distance_table_build_compute(self, loop_queries, sequence_ub,
                                     aicore_move_offset, aicore_sequence_num):
        # move queries from out to UB
        queries_ub = self.tik_instance.Tensor("float16", (self.dim, ),
                                              name="queries_ub",
                                              scope=tik.scope_ubuf)
        self.tik_instance.data_move(queries_ub[0],
                                    self.input_queries_gm[loop_queries, 0], 0,
                                    1, self.dim // 16, 0, 0)

        # compute diastance per list
        thread_num_need = 1
        with self.tik_instance.if_scope(aicore_sequence_num > 1):
            thread_num_need = 2
        with self.tik_instance.for_range(
                0, aicore_sequence_num,
                thread_num=thread_num_need) as loop_sequence:
            coarse_centroids_ub = self.tik_instance.Tensor(
                "float16", (self.dim, ),
                name="coarse_centroids_ub",
                scope=tik.scope_ubuf)
            residual_ub = self.tik_instance.Tensor("float16", (self.dim, ),
                                                   name="residual_ub",
                                                   scope=tik.scope_ubuf)
            sequence_id = self.tik_instance.Scalar('int32')
            sequence_id.set_as(sequence_ub[loop_queries,
                                           aicore_move_offset + loop_sequence])

            self.tik_instance.data_move(
                coarse_centroids_ub[0],
                self.input_coarse_centroids_gm[sequence_id,
                                               0], 0, 1, self.dim // 16, 0, 0)

            # x-yc
            vsub_repeat_time = self.dim // self.vector_mask_max
            vsub_offset = 0
            if vsub_repeat_time > 0:
                self.tik_instance.vsub(self.vector_mask_max, residual_ub[0],
                                       queries_ub[0], coarse_centroids_ub[0],
                                       vsub_repeat_time, 1, 1, 1, 8, 8, 8)
                vsub_offset += vsub_repeat_time * self.vector_mask_max

            vsub_last_num = self.dim % self.vector_mask_max
            if vsub_last_num > 0:
                self.tik_instance.vsub(vsub_last_num, residual_ub[vsub_offset],
                                       queries_ub[vsub_offset],
                                       coarse_centroids_ub[vsub_offset], 1, 1,
                                       1, 1, 8, 8, 8)

            # compute diastance per quantizers
            with self.tik_instance.for_range(0,
                                             self.sub_quantizers_num,
                                             thread_num=2) as loop_quantizers:
                pq_centroids_ub = self.tik_instance.Tensor(
                    "float16", (self.pq_centroids_num, self.sub_dim),
                    name="pq_centroids_ub",
                    scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    pq_centroids_ub[0],
                    self.input_pq_centroids_gm[loop_quantizers, 0], 0, 1,
                    self.pq_centroids_num * self.sub_dim // 16, 0, 0)

                sub_residual_ub = self.tik_instance.Tensor(
                    "float16", (self.sub_dim, ),
                    name="sub_residual_ub",
                    scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    sub_residual_ub[0],
                    residual_ub[loop_quantizers * self.sub_dim], 0, 1,
                    self.sub_dim // 16, 0, 0)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (self.pq_centroids_num, ),
                                                  name="dst_ub",
                                                  scope=tik.scope_ubuf)

                # do vsub
                vsub_loop = self.pq_centroids_num // 255
                vsub_offset = self.tik_instance.Scalar(dtype="int32")
                vsub_offset.set_as(0)
                if vsub_loop > 0:
                    with self.tik_instance.for_range(0, vsub_loop):
                        self.tik_instance.vsub(self.sub_dim,
                                               pq_centroids_ub[vsub_offset, 0],
                                               sub_residual_ub[0],
                                               pq_centroids_ub[vsub_offset,
                                                               0], 255, 1, 1,
                                               1, self.sub_dim // 16, 0,
                                               self.sub_dim // 16)
                        vsub_offset.set_as(vsub_offset + 255)

                vsub_repeat_time = self.pq_centroids_num % 255
                if vsub_repeat_time > 0:
                    self.tik_instance.vsub(
                        self.sub_dim, pq_centroids_ub[vsub_offset,
                                                      0], sub_residual_ub[0],
                        pq_centroids_ub[vsub_offset, 0], vsub_repeat_time, 1,
                        1, 1, self.sub_dim // 16, 0, self.sub_dim // 16)

                # do vmul
                vmul_loop = (self.pq_centroids_num * self.sub_dim) // \
                    (self.vector_mask_max * 255)
                vmul_offset = self.tik_instance.Scalar(dtype="int32")
                vmul_offset.set_as(0)
                if vmul_loop > 0:
                    with self.tik_instance.for_range(0,
                                                     vmul_loop) as mul_index:
                        vmul_offset.set_as(mul_index * self.vector_mask_max *
                                           255)
                        self.tik_instance.vmul(
                            self.vector_mask_max,
                            pq_centroids_ub[vmul_offset // self.sub_dim,
                                            vmul_offset % self.sub_dim],
                            pq_centroids_ub[vmul_offset // self.sub_dim,
                                            vmul_offset % self.sub_dim],
                            pq_centroids_ub[vmul_offset // self.sub_dim,
                                            vmul_offset % self.sub_dim], 255,
                            1, 1, 1, 8, 8, 8)
                    vmul_offset.set_as(vmul_offset +
                                       self.vector_mask_max * 255)

                vmul_repeat_time = (self.pq_centroids_num * self.sub_dim) % \
                    (self.vector_mask_max * 255) // self.vector_mask_max
                if vmul_repeat_time > 0:
                    self.tik_instance.vmul(
                        self.vector_mask_max,
                        pq_centroids_ub[vmul_offset // self.sub_dim,
                                        vmul_offset % self.sub_dim],
                        pq_centroids_ub[vmul_offset // self.sub_dim,
                                        vmul_offset % self.sub_dim],
                        pq_centroids_ub[vmul_offset // self.sub_dim,
                                        vmul_offset % self.sub_dim],
                        vmul_repeat_time, 1, 1, 1, 8, 8, 8)

                # do vcadd
                vcadd_loop = self.pq_centroids_num // 255
                vcadd_offset = self.tik_instance.Scalar(dtype="int32")
                vcadd_offset.set_as(0)
                if vcadd_loop > 0:
                    with self.tik_instance.for_range(0, vcadd_loop):
                        self.tik_instance.vcadd(
                            self.sub_dim, dst_ub[vcadd_offset],
                            pq_centroids_ub[vcadd_offset,
                                            0], 255, 1, 1, self.sub_dim // 16)
                        vcadd_offset.set_as(vcadd_offset + 255)

                vcadd_repeat_time = self.pq_centroids_num % 255
                if vcadd_repeat_time > 0:
                    self.tik_instance.vcadd(self.sub_dim, dst_ub[vcadd_offset],
                                            pq_centroids_ub[vcadd_offset, 0],
                                            vcadd_repeat_time, 1, 1,
                                            self.sub_dim // 16)

                self.tik_instance.data_move(
                    self.output_distance_table_gm[loop_queries,
                                                  aicore_move_offset +
                                                  loop_sequence,
                                                  loop_quantizers, 0],
                    dst_ub[0], 0, 1, self.pq_centroids_num // 16, 0, 0)

    def forward(self):
        aicore_move_offset = self.tik_instance.Scalar(dtype="int32")
        aicore_move_offset.set_as(0)
        with self.tik_instance.for_range(
                0, self.aicore_use, block_num=self.aicore_use) as block_index:
            # move sequence number from out to UB
            sequence_ub = self.tik_instance.Tensor("int32",
                                                   self.shape_sequence_number,
                                                   name="sequence_ub",
                                                   scope=tik.scope_ubuf)
            self.tik_instance.data_move(
                sequence_ub[0], self.input_sequence_number_gm[0], 0, 1,
                self.queries_num * self.closest_num // 8, 0, 0)

            # compute diastance per queries
            with self.tik_instance.for_range(0,
                                             self.queries_num) as loop_queries:
                # compute number of closest IVF centroids to the query vector
                num = self.tik_instance.Scalar('int32')
                num.set_as(0)
                with self.tik_instance.for_range(0,
                                                 self.closest_num) as loop_i:
                    with self.tik_instance.if_scope(
                            sequence_ub[loop_queries, loop_i] > -1):
                        num.set_as(num + 1)

                sequence_num_each_core = num // self.aicore_use
                sequence_num_last_core = num - (self.aicore_use - 1) * \
                    sequence_num_each_core
                aicore_move_offset.set_as(block_index * sequence_num_each_core)
                with self.tik_instance.if_scope(
                        block_index != self.aicore_use - 1):
                    aicore_sequence_num = sequence_num_each_core
                with self.tik_instance.else_scope():
                    aicore_sequence_num = sequence_num_last_core

                if self.sub_dim == 4:
                    self.distance_table_build_compute_subdim_4(
                        loop_queries, sequence_ub, aicore_move_offset,
                        aicore_sequence_num)
                elif self.sub_dim == 8:
                    self.distance_table_build_compute_subdim_8(
                        loop_queries, sequence_ub, aicore_move_offset,
                        aicore_sequence_num)
                else:
                    self.distance_table_build_compute(loop_queries,
                                                      sequence_ub,
                                                      aicore_move_offset,
                                                      aicore_sequence_num)

        one = self.tik_instance.Scalar("uint16", "one", 1)
        flag_ub = self.tik_instance.Tensor("uint16",
                                           self.shape_flag,
                                           name="flag_ub",
                                           scope=tik.scope_ubuf)
        self.tik_instance.data_move(flag_ub, self.output_flag_gm, 0, 1, 1, 0,
                                    0)
        flag_ub[0].set_as(one)
        self.tik_instance.data_move(self.output_flag_gm, flag_ub, 0, 1, 1, 0,
                                    0)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[
                self.input_queries_gm, self.input_pq_centroids_gm,
                self.input_sequence_number_gm, self.input_coarse_centroids_gm
            ],
            outputs=[self.output_distance_table_gm, self.output_flag_gm])

        return self.tik_instance


def distance_table_build(input_queries,
                         input_pq_centroids,
                         input_sequence_number,
                         input_coarse_centroids,
                         output_distance_table,
                         output_flag,
                         kernel_name="distance_table_build"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_pq_coarse_centroids : dict
        shape and dtype of pq coarse centroids
    input_sequence_number : dict
        shape and dtype of sequence number of the np closest
        IVF centroids to the query vector
    input_coarse_centroids : dict
        shape and dtype of coarse centroids
    output_distance_table : dict
        shape and dtype of distance table, should be same
        dtype as input_queries
    output_flag : dict
        shape and dtype of flag
    kernel_name : str
        kernel name, default value is "distance_table_build"

    Returns
    -------
    None
    """
    distance_table_build_instance = DistanceTableBuild(
        input_queries, input_pq_centroids, input_sequence_number,
        input_coarse_centroids, output_distance_table, output_flag,
        kernel_name)
    tik_instance = distance_table_build_instance.forward()
    return tik_instance
