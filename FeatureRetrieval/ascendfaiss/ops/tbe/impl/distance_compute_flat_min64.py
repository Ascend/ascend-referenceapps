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


class DistanceComputeFlatMin64():
    def __init__(self,
                 input_queries,
                 input_centroids,
                 input_precomputed,
                 input_actual_num,
                 output_distances,
                 output_mins,
                 output_flag,
                 kernel_name="distance_compute_flat_min64"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_centroids = input_centroids.get("shape")
        self.dtype_centroids = input_centroids.get("dtype")
        self.shape_precomputed = input_precomputed.get("shape")
        self.dtype_precomputed = input_precomputed.get("dtype")
        self.shape_actual_num = input_actual_num.get("shape")
        self.dtype_actual_num = input_actual_num.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_mins = output_mins.get("shape")
        self.dtype_mins = output_mins.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.centroids_num, = self.shape_precomputed

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.fp32_vector_mask_max = 64
        self.fp16_vector_mask_max = 128
        self.min_batch = 64

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        if self.dim % 16 != 0:
            raise RuntimeError("feature dim must be a multiple of 16")
        if self.centroids_num % 16 != 0:
            raise RuntimeError("centroids num must be a multiple of 16")

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_dprofile = tik.Dprofile("v100", "mini")
        self.tik_instance = tik.Tik(tik_dprofile)

        self.aicore_use = 2
        self.queries_num_each_loop = min(48, self.queries_num)
        self.centroids_num_each_loop = min(48 // 16 // ((self.queries_num_each_loop + 15) // 16) * 1024,
                                           (1 + 256 // (self.dim + 128)) * 1024)

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("float32", name="coeff", init_value=-2)

        # creat input tensor: input_queries_gm, input_centroids_gm
        # and input_precomputed_gm
        # and output tensor: output_distances_gm, output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_centroids_gm = self.tik_instance.Tensor(
            self.dtype_centroids,
            self.shape_centroids,
            name="input_centroids_gm",
            scope=tik.scope_gm)
        self.input_precomputed_gm = self.tik_instance.Tensor(
            self.dtype_precomputed,
            self.shape_precomputed,
            name="input_precomputed_gm",
            scope=tik.scope_gm)
        self.input_actual_num_gm = self.tik_instance.Tensor(
            self.dtype_actual_num,
            self.shape_actual_num,
            name="input_actual_num_gm",
            scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(
            self.dtype_distances,
            self.shape_distances,
            name="output_distances_gm",
            scope=tik.scope_gm)
        self.output_mins_gm = self.tik_instance.Tensor(
            self.dtype_mins,
            self.shape_mins,
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
        actual_num_ub = self.tik_instance.Tensor(
            "uint32", (8,), name="actual_code_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_num_ub, self.input_actual_num_gm, 0, 1, 1, 0, 0)
        actual_num = self.tik_instance.Scalar(dtype="uint32", name="actual_code_num", init_value=0)
        actual_num.set_as(actual_num_ub[0])

        self.centroids_num_each_core = \
            (actual_num // self.aicore_use + 8 * self.min_batch) // self.min_batch // 16 * self.min_batch * 16
        self.centroids_num_last_core = actual_num - (self.aicore_use - 1) * self.centroids_num_each_core

        self.code_num_each_core = \
            (actual_num // self.aicore_use // 16 * 16) // self.min_batch // 16 * self.min_batch * 16
        self.code_num_last_core = actual_num - (self.aicore_use - 1) * self.code_num_each_core

    def distance_compute_each_loop(self, aicore_move_offset, aicore_centroids_num, move_offset, move_num):
        queries_align = (move_num + 15) // 16 * 16
        add_scalar_list = [
            self.tik_instance.Scalar(dtype="float32") for i in range(move_num)
        ]
        queries_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, queries_align, 16),
                                              name="queries_l1", scope=tik.scope_cbuf)

        with self.tik_instance.new_stmt_scope():
            for i in range(move_num):
                self.tik_instance.data_move(queries_l1[0, i, 0],
                                            self.input_queries_gm[move_offset + i, 0],
                                            0, self.dim // 16, 1, 0, queries_align - 1)

            queries_square_l0c = self.tik_instance.Tensor("float32",
                                                          (queries_align // 16, queries_align, 16),
                                                          name="queries_square_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(queries_square_l0c,
                                     queries_l1, queries_l1,
                                     queries_align, self.dim, queries_align)
            queries_square_ub = self.tik_instance.Tensor("float32",
                                                         (queries_align // 16, queries_align, 16),
                                                         name="queries_square_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(queries_square_ub,
                                        queries_square_l0c,
                                        0, 1, queries_align * queries_align // 256, 0, 0)

            queries_l2_ub = self.tik_instance.Tensor("float32",
                                                     (queries_align,),
                                                     name="queries_l2_ub", scope=tik.scope_ubuf)
            for i in range(move_num):
                mask = 2 ** (i % 16)
                self.tik_instance.vcadd([0, mask],
                                        queries_l2_ub[i],
                                        queries_square_ub[i // 16, i, 0],
                                        1, 1, 1, 8)
            for i in range(move_num):
                add_scalar_list[i].set_as(queries_l2_ub[i])

        # compute xy using cube
        centroids_loop_time = aicore_centroids_num // self.centroids_num_each_loop
        with self.tik_instance.if_scope(centroids_loop_time > 0):
            with self.tik_instance.for_range(0, centroids_loop_time) as loop_centroids:
                self.cube_compute_each_loop(queries_l1, add_scalar_list, aicore_move_offset,
                                            loop_centroids * self.centroids_num_each_loop,
                                            self.centroids_num_each_loop, move_offset, move_num, 0)

        centroids_last_num = aicore_centroids_num % self.centroids_num_each_loop
        with self.tik_instance.if_scope(centroids_last_num > 0):
            self.cube_compute_each_loop(queries_l1, add_scalar_list, aicore_move_offset,
                                        centroids_loop_time * self.centroids_num_each_loop,
                                        centroids_last_num, move_offset, move_num, 1)

    def cube_compute_each_loop(self, queries_l1, add_scalar_list,
                               aicore_move_offset,
                               centroids_move_offset,
                               centroids_move_num, queries_move_offset,
                               queries_move_num, flag):
        queries_align = (queries_move_num + 15) // 16 * 16
        centroids_move_num_ub = (centroids_move_num + 15) // 16 * 16
        centroids_l2_ub = self.tik_instance.Tensor("float32",
                                                   (self.centroids_num_each_loop,),
                                                   name="centroids_l2_ub", scope=tik.scope_ubuf)

        centroids_l2_ub_fp16 = self.tik_instance.Tensor("float16",
                                                        (self.centroids_num_each_loop,),
                                                        name="centroids_l2_ub_fp16",
                                                        scope=tik.scope_ubuf)
        # mov centroids l2 from out to UB
        self.tik_instance.data_move(centroids_l2_ub_fp16,
                                    self.input_precomputed_gm[aicore_move_offset + centroids_move_offset],
                                    0, 1, centroids_move_num_ub // 16, 0, 0)

        # centroids_l2 do conv from fp16 to fp32
        if flag == 0:
            vconv_repeat_time = self.centroids_num_each_loop // self.fp32_vector_mask_max
            if vconv_repeat_time > 0:
                self.tik_instance.vconv(self.fp32_vector_mask_max, "none",
                                        centroids_l2_ub[0],
                                        centroids_l2_ub_fp16[0],
                                        vconv_repeat_time, 1, 1, 8, 4)
        else:
            vconv_repeat_time = centroids_move_num // self.fp32_vector_mask_max
            vconv_offset = 0
            with self.tik_instance.if_scope(vconv_repeat_time > 0):
                self.tik_instance.vconv(self.fp32_vector_mask_max, "none",
                                        centroids_l2_ub[vconv_offset],
                                        centroids_l2_ub_fp16[vconv_offset],
                                        vconv_repeat_time, 1, 1, 8, 4)
                vconv_offset += vconv_repeat_time * self.fp32_vector_mask_max

            vconv_last_num = centroids_move_num % self.fp32_vector_mask_max
            with self.tik_instance.if_scope(vconv_last_num > 0):
                self.tik_instance.vconv(vconv_last_num, "none",
                                        centroids_l2_ub[vconv_offset],
                                        centroids_l2_ub_fp16[vconv_offset],
                                        1, 1, 1, 8, 4)

        inner_product_l0c = self.tik_instance.Tensor("float32",
                                                     (self.centroids_num_each_loop // 16, queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        with self.tik_instance.new_stmt_scope():
            loop_times = max(int(self.dim / 256), 1)
            centroids_num_ub_each_loop = self.centroids_num_each_loop // loop_times
            loop_times = (centroids_move_num_ub + centroids_num_ub_each_loop - 1) // centroids_num_ub_each_loop
            centroids_l1 = self.tik_instance.Tensor("float16",
                                                    (self.dim // 16, centroids_num_ub_each_loop, 16),
                                                    name="centroids_l1", scope=tik.scope_cbuf)
            with self.tik_instance.for_range(0, loop_times) as loop_ub:
                for i in range(self.dim // 16):
                    self.tik_instance.data_move(
                        centroids_l1[i, 0, 0],
                        self.input_centroids_gm[(aicore_move_offset + centroids_move_offset
                                                 + loop_ub * centroids_num_ub_each_loop) // 16, i, 0, 0],
                        0, centroids_num_ub_each_loop // 16, 16, self.dim - 16, 0)
                self.tik_instance.matmul(inner_product_l0c[(loop_ub * centroids_num_ub_each_loop * queries_align):],
                                         queries_l1, centroids_l1,
                                         queries_align, self.dim, centroids_num_ub_each_loop)

        centroids_out_num = self.centroids_num_each_loop // 2
        if flag == 0:
            with self.tik_instance.for_range(0, 2, thread_num=1) as i:
                add_ub = self.tik_instance.Tensor("float32",
                                                  (queries_move_num, centroids_out_num),
                                                  name="add_ub", scope=tik.scope_ubuf)
                repeat = centroids_out_num // self.fp32_vector_mask_max
                for loop_queries in range(0, queries_move_num):
                    # cal x*x + y*y
                    self.tik_instance.vadds(self.fp32_vector_mask_max,
                                            add_ub[loop_queries, 0],
                                            centroids_l2_ub[centroids_out_num * i], add_scalar_list[loop_queries],
                                            repeat, 1, 1, 8, 8)

                inner_product_ub = self.tik_instance.Tensor("float32",
                                                            (centroids_out_num // 16, queries_align, 16),
                                                            name="inner_product_ub", scope=tik.scope_ubuf)
                # mov xy from L0-C to UB
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c[i * centroids_out_num // 16, 0, 0],
                                            0, 1, centroids_out_num * queries_align // 256, 0, 0)

                thread_num_vaxpy = 2 if queries_move_num > 1 else 1
                with self.tik_instance.for_range(0, queries_move_num, thread_num=thread_num_vaxpy) as loop_query:
                    # cal x*x + y*y - 2*x*y
                    self.tik_instance.vaxpy(16,
                                            add_ub[loop_query, 0],
                                            inner_product_ub[0, loop_query, 0],
                                            self.coeff,
                                            centroids_out_num // 16, 1, 1, 2, queries_align * 2)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (queries_move_num, centroids_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)

                vconv_times = 2 if queries_move_num > 1 else 1
                query_num_each_loop = queries_move_num // vconv_times
                vconv_repeat_times = query_num_each_loop * centroids_out_num // self.fp32_vector_mask_max
                with self.tik_instance.for_range(0, vconv_times, thread_num=vconv_times) as vconv_loop:
                    self.tik_instance.vconv(self.fp32_vector_mask_max, "none",
                                            dst_ub[query_num_each_loop * vconv_loop, 0],
                                            add_ub[query_num_each_loop * vconv_loop, 0],
                                            vconv_repeat_times, 1, 1, 4, 8)

                self.tik_instance.data_move(self.output_distances_gm[queries_move_offset,
                                                                     aicore_move_offset + centroids_move_offset
                                                                     + i * centroids_out_num],
                                            dst_ub,
                                            0, queries_move_num, centroids_out_num // 16, 0,
                                            (self.centroids_num - centroids_out_num) // 16)

                min_size = centroids_out_num // self.min_batch * 2
                dst_min_ub = self.tik_instance.Tensor("float16",
                                                      (queries_move_num, min_size),
                                                      name="dst_min_ub", scope=tik.scope_ubuf)
                vcmin_loops = 2 if queries_move_num > 1 else 1
                vcmin_repeat_times = query_num_each_loop * centroids_out_num // self.min_batch
                with self.tik_instance.for_range(0, vcmin_loops, thread_num=vcmin_loops) as vcmin_loop:
                    self.tik_instance.vcmin(self.min_batch,
                                            dst_min_ub[vcmin_loop * query_num_each_loop, 0],
                                            dst_ub[vcmin_loop * query_num_each_loop, 0],
                                            vcmin_repeat_times, 1, 1, self.min_batch // 16)
                self.tik_instance.data_move(self.output_mins_gm[queries_move_offset,
                                                                (aicore_move_offset + centroids_move_offset
                                                                 + i * centroids_out_num) // self.min_batch * 2],
                                            dst_min_ub,
                                            0, queries_move_num, centroids_out_num // self.min_batch // 8, 0,
                                            (self.centroids_num - centroids_out_num) // self.min_batch // 8)
        else:
            centroids_loop_times = (centroids_move_num_ub + centroids_out_num - 1) // centroids_out_num
            with self.tik_instance.for_range(0, centroids_loop_times, thread_num=1) as i:
                add_ub = self.tik_instance.Tensor("float32",
                                                  (queries_move_num, centroids_out_num),
                                                  name="add_ub", scope=tik.scope_ubuf)
                repeat = centroids_out_num // self.fp32_vector_mask_max
                for loop_queries in range(0, queries_move_num):
                    # cal x*x + y*y
                    self.tik_instance.vadds(self.fp32_vector_mask_max,
                                            add_ub[loop_queries, 0],
                                            centroids_l2_ub[centroids_out_num * i], add_scalar_list[loop_queries],
                                            repeat, 1, 1, 8, 8)

                inner_product_ub = self.tik_instance.Tensor("float32",
                                                            (centroids_out_num // 16, queries_align, 16),
                                                            name="inner_product_ub", scope=tik.scope_ubuf)
                # mov xy from L0-C to UB
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c[i * centroids_out_num // 16, 0, 0],
                                            0, 1, centroids_out_num * queries_align // 256, 0, 0)

                thread_num_vaxpy = 2 if queries_move_num > 1 else 1
                with self.tik_instance.for_range(0, queries_move_num, thread_num=thread_num_vaxpy) as loop_query:
                    # cal x*x + y*y - 2*x*y
                    self.tik_instance.vaxpy(16,
                                            add_ub[loop_query, 0],
                                            inner_product_ub[0, loop_query, 0],
                                            self.coeff,
                                            centroids_out_num // 16, 1, 1, 2, queries_align * 2)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (queries_move_num, centroids_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)

                vconv_times = 2 if queries_move_num > 1 else 1
                query_num_each_loop = queries_move_num // vconv_times
                vconv_repeat_time = query_num_each_loop * centroids_out_num // self.fp32_vector_mask_max
                with self.tik_instance.for_range(0, vconv_times, thread_num=vconv_times) as vconv_loop:
                    self.tik_instance.vconv(self.fp32_vector_mask_max, "none",
                                            dst_ub[query_num_each_loop * vconv_loop, 0],
                                            add_ub[query_num_each_loop * vconv_loop, 0],
                                            vconv_repeat_time, 1, 1, 4, 8)
                self.tik_instance.data_move(self.output_distances_gm[queries_move_offset,
                                                                     aicore_move_offset + centroids_move_offset
                                                                     + i * centroids_out_num],
                                            dst_ub,
                                            0, queries_move_num, centroids_out_num // 16, 0,
                                            (self.centroids_num - centroids_out_num) // 16)

                min_size = centroids_out_num // self.min_batch * 2
                dst_min_ub = self.tik_instance.Tensor("float16",
                                                      (queries_move_num, min_size),
                                                      name="dst_min_ub", scope=tik.scope_ubuf)
                with self.tik_instance.if_scope(i == centroids_loop_times - 1):
                    centroids_last_num = centroids_move_num - i * centroids_out_num
                    thread_num = 2 if queries_move_num > 1 else 1
                    with self.tik_instance.for_range(0, queries_move_num, thread_num=thread_num) as query:
                        vcmin_repeat_times = centroids_last_num // self.min_batch
                        offset = 0
                        with self.tik_instance.if_scope(vcmin_repeat_times > 0):
                            self.tik_instance.vcmin(self.min_batch,
                                                    dst_min_ub[query, 0],
                                                    dst_ub[query, 0],
                                                    vcmin_repeat_times, 1, 1, self.min_batch // 16)
                            offset += vcmin_repeat_times * self.min_batch
                        vcmin_remain = centroids_last_num % self.min_batch
                        with self.tik_instance.if_scope(vcmin_remain > 0):
                            self.tik_instance.vcmin(vcmin_remain,
                                                    dst_min_ub[query, vcmin_repeat_times * 2],
                                                    dst_ub[query, offset],
                                                    1, 1, 1, 1)
                    self.tik_instance.data_move(self.output_mins_gm[queries_move_offset,
                                                                    (aicore_move_offset + centroids_move_offset
                                                                     + i * centroids_out_num) // self.min_batch * 2],
                                                dst_min_ub,
                                                0, queries_move_num, centroids_out_num // self.min_batch // 8, 0,
                                                (self.centroids_num - centroids_out_num) // self.min_batch // 8)
                with self.tik_instance.else_scope():
                    vcmin_loops = 2 if queries_move_num > 1 else 1
                    vcmin_repeat_times = query_num_each_loop * centroids_out_num // self.min_batch
                    with self.tik_instance.for_range(0, vcmin_loops, thread_num=vcmin_loops) as vcmin_loop:
                        self.tik_instance.vcmin(self.min_batch,
                                                dst_min_ub[vcmin_loop * query_num_each_loop, 0],
                                                dst_ub[vcmin_loop * query_num_each_loop, 0],
                                                vcmin_repeat_times, 1, 1, self.min_batch // 16)

                    self.tik_instance.data_move(self.output_mins_gm[queries_move_offset,
                                                                    (aicore_move_offset + centroids_move_offset
                                                                     + i * centroids_out_num) // self.min_batch * 2],
                                                dst_min_ub,
                                                0, queries_move_num, centroids_out_num // self.min_batch // 8, 0,
                                                (self.centroids_num - centroids_out_num) // self.min_batch // 8)

    def distance_compute_flat_min64(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            # compute coarse centroids num and move offest every core
            aicore_centroids_num = self.tik_instance.Scalar(dtype="uint32", name="aicore_code_num", init_value=0)
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_centroids_num.set_as(self.centroids_num_each_core)
            with self.tik_instance.else_scope():
                aicore_centroids_num.set_as(self.centroids_num_last_core)

            queries_loop_time = self.queries_num // self.queries_num_each_loop
            if queries_loop_time > 0:
                with self.tik_instance.for_range(0, queries_loop_time) as loop_queries:
                    self.distance_compute_each_loop(block_index * self.centroids_num_each_core,
                                                    aicore_centroids_num,
                                                    loop_queries * self.queries_num_each_loop,
                                                    self.queries_num_each_loop)
            queries_last_num = self.queries_num % self.queries_num_each_loop
            if queries_last_num > 0:
                self.distance_compute_each_loop(block_index * self.centroids_num_each_core,
                                                aicore_centroids_num,
                                                queries_loop_time * self.queries_num_each_loop,
                                                queries_last_num)

            one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16",
                                               (16,),
                                               name="flag_ub", scope=tik.scope_ubuf)

            flag_ub[0].set_as(one)
            self.tik_instance.data_move(self.output_flag_gm[block_index * 16],
                                        flag_ub,
                                        0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_compute_flat_min64()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_centroids_gm,
                                       self.input_precomputed_gm,
                                       self.input_actual_num_gm
                                   ],
                                   outputs=[self.output_distances_gm, self.output_mins_gm, self.output_flag_gm])

        return self.tik_instance


def distance_compute_flat_min64(input_queries,
                                input_centroids,
                                input_precomputed,
                                input_actual_num,
                                output_distances,
                                output_mins,
                                output_flag,
                                kernel_name="distance_compute_flat_min64"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_centroids : dict
        shape and dtype of coarse centroids
    input_precomputed : dict
        shape and dtype of precomputed L2 distance of coarse centroids
    input_actual_num: dict
        shape and dtype of actual code num,
        shape must be (8,) and dtype must be uint32
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_mins : dict
        shape and dtype of mins
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_compute_flat_min64"

    Returns
    -------
    None
    """
    distance_compute_flat_min64 = DistanceComputeFlatMin64(input_queries, input_centroids, input_precomputed,
                                                           input_actual_num, output_distances, output_mins, output_flag,
                                                           kernel_name)
    tik_instance = distance_compute_flat_min64.get_tik_instance()
    return tik_instance
