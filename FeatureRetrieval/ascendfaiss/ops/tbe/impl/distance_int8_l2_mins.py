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


class DistanceInt8L2Mins:
    def __init__(self,
                 input_queries,
                 input_mask,
                 input_centroids,
                 input_precomputed,
                 input_actual_num,
                 output_dist,
                 output_min_dist,
                 output_flag,
                 kernel_name="distance_int8_l2_mins"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_mask = input_mask.get("shape")
        self.dtype_mask = input_mask.get("dtype")
        self.shape_centroids = input_centroids.get("shape")
        self.dtype_centroids = input_centroids.get("dtype")
        self.shape_precomputed = input_precomputed.get("shape")
        self.dtype_precomputed = input_precomputed.get("dtype")
        self.shape_actual_num = input_actual_num.get("shape")
        self.dtype_actual_num = input_actual_num.get("dtype")
        self.shape_dist = output_dist.get("shape")
        self.dtype_dist = output_dist.get("dtype")
        self.shape_min_dist = output_min_dist.get("shape")
        self.dtype_min_dist = output_min_dist.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.centroids_num, = self.shape_precomputed

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.int32_mask = 64
        self.min_mask = 64
        self.fp16_mask = 128
        # scale changed with dim
        self.scale = 0.01 / min(self.dim // 64, max(self.dim // 128 + 1, 4))

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

        self.aicore_use = self.shape_actual_num[0]
        self.queries_num_each_loop = min(48, self.queries_num)
        self.centroids_num_each_loop = min((48 // self.queries_num_each_loop) * 512, 1024)

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("int32", name="coeff", init_value=-2)
        self.query_l2 = self.tik_instance.Scalar("int32", name="query_l2", init_value=0)

        # Here L2 distance applied, we use 60000 as minimal distance
        self.default_scalar = self.tik_instance.Scalar("float16",
                                                       name="default_scalar",
                                                       init_value=65500)
        
        # creat input tensor: input_queries_gm, input_centroids_gm
        # and input_precomputed_gm
        # and output tensor: output_dist_gm, output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(self.dtype_queries, self.shape_queries,
                                                         name="input_queries_gm", scope=tik.scope_gm)
        self.input_mask_gm = self.tik_instance.Tensor(self.dtype_mask, self.shape_mask,
                                                         name="input_mask_gm", scope=tik.scope_gm)
        self.input_centroids_gm = self.tik_instance.Tensor(self.dtype_centroids, self.shape_centroids,
                                                           name="input_centroids_gm", scope=tik.scope_gm)
        self.input_precomputed_gm = self.tik_instance.Tensor(self.dtype_precomputed, self.shape_precomputed,
                                                             name="input_precomputed_gm", scope=tik.scope_gm)
        self.input_actual_num_gm = self.tik_instance.Tensor(self.dtype_actual_num, self.shape_actual_num,
                                                            name="input_actual_num_gm", scope=tik.scope_gm)
        self.output_dist_gm = self.tik_instance.Tensor(self.dtype_dist, self.shape_dist,
                                                       name="output_dist_gm", scope=tik.scope_gm)
        self.output_min_dist_gm = self.tik_instance.Tensor(self.dtype_min_dist, self.shape_min_dist,
                                                           name="output_min_dist_gm", scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(self.dtype_flag, self.shape_flag,
                                                       name="output_flag_gm", scope=tik.scope_gm)

    def cal_num_each_core(self):
        """
        calculate actual code num of each core
        """
        # move actual code num from out to UB
        actual_num_ub = self.tik_instance.Tensor("uint32", (8,), name="actual_code_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_num_ub, self.input_actual_num_gm, 0, 1, 1, 0, 0)
        actual_num = self.tik_instance.Scalar(dtype="uint32", name="actual_code_num", init_value=0)
        actual_num.set_as(actual_num_ub[0])

        self.mask_offset = self.tik_instance.Scalar(dtype="uint32",
                                                    name="mask_offset",
                                                    init_value=0)
        self.mask_offset.set_as(actual_num_ub[1])
        
        self.mask_len = self.tik_instance.Scalar(dtype="uint32",
                                                 name="mask_len",
                                                 init_value=0)
        self.mask_len.set_as(actual_num_ub[2])
        
        self.use_mask = self.tik_instance.Scalar(dtype="uint32",
                                                 name="use_mask",
                                                 init_value=0)
        self.use_mask.set_as(actual_num_ub[3])
        
        if self.aicore_use == 2:
            self.centroids_num_each_core = \
                (actual_num // self.aicore_use + self.min_mask * 8) // self.min_mask // 16 * self.min_mask * 16
        else:
            self.centroids_num_each_core = actual_num // self.aicore_use // self.min_mask // 16 * self.min_mask * 16
        
        self.centroids_num_last_core = actual_num - (self.aicore_use - 1) * self.centroids_num_each_core        

    def distance_int8_l2_each_loop(self, aicore_move_offset, aicore_centroids_num, move_offset, move_num):
        queries_align = (move_num + 15) // 16 * 16

        queries_l1 = self.tik_instance.Tensor("int8", (self.dim // 32, queries_align, 32),
                                              name="queries_l1", scope=tik.scope_cbuf)

        queries_l2_ub = self.tik_instance.Tensor("int32", (move_num,), name="queries_l2_ub", scope=tik.scope_ubuf)

        with self.tik_instance.new_stmt_scope():
            for i in range(move_num):
                self.tik_instance.data_move(queries_l1[0, i, 0], self.input_queries_gm[move_offset + i, 0],
                                            0, self.dim // 32, 1, 0, queries_align - 1)

            queries_square_l0c = self.tik_instance.Tensor("int32", (queries_align // 16, queries_align, 16),
                                                          name="queries_square_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(queries_square_l0c, queries_l1, queries_l1,
                                     queries_align, self.dim, queries_align)
            queries_square_ub = self.tik_instance.Tensor("int32", (queries_align // 16, queries_align, 16),
                                                         name="queries_square_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(queries_square_ub, queries_square_l0c, 0, 1,
                                        queries_align * queries_align // 256, 0, 0)

            repeat = move_num // self.int32_mask
            offset = 0
            if repeat > 0:
                self.tik_instance.vec_dup(self.int32_mask, queries_l2_ub[offset], 0, repeat, 8)
                offset += repeat * self.int32_mask

            remain = move_num % self.int32_mask
            if remain > 0:
                self.tik_instance.vec_dup(remain, queries_l2_ub[offset], 0, 1, 8)

            for i in range(move_num):
                mask = 2 ** (i % 16)
                self.tik_instance.vadd([0, mask],
                                       queries_l2_ub[i // 16 * 16],
                                       queries_l2_ub[i // 16 * 16],
                                       queries_square_ub[i // 16, i, 0],
                                       1, 1, 1, 1, 8, 8, 8)

        # compute xy using cube
        centroids_loop_time = aicore_centroids_num // self.centroids_num_each_loop
        with self.tik_instance.if_scope(centroids_loop_time > 0):
            with self.tik_instance.for_range(0, centroids_loop_time) as loop_centroids:
                self.cube_compute_each_loop(queries_l1, queries_l2_ub, aicore_move_offset,
                                            loop_centroids * self.centroids_num_each_loop,
                                            self.centroids_num_each_loop,
                                            move_offset, move_num, 0)

        centroids_last_num = aicore_centroids_num % self.centroids_num_each_loop
        with self.tik_instance.if_scope(centroids_last_num > 0):
            self.cube_compute_each_loop(queries_l1, queries_l2_ub, aicore_move_offset,
                                        centroids_loop_time * self.centroids_num_each_loop,
                                        centroids_last_num, move_offset, move_num, 1)

    def cube_compute_each_loop(self, queries_l1, queries_l2_ub,
                               aicore_move_offset,
                               centroids_move_offset,
                               centroids_move_num, queries_move_offset,
                               queries_move_num, flag):
        queries_align = (queries_move_num + 15) // 16 * 16
        centroids_align_16 = (centroids_move_num + 15) // 16 * 16
        centroids_l1 = self.tik_instance.Tensor("int8", (self.dim // 32, self.centroids_num_each_loop, 32),
                                                name="centroids_l1", scope=tik.scope_cbuf)
        for i in range(self.dim // 32):
            self.tik_instance.data_move(centroids_l1[i, 0, 0],
                                        self.input_centroids_gm[(aicore_move_offset + centroids_move_offset) // 16,
                                                                i, 0, 0],
                                        0, centroids_align_16 // 16, 16, self.dim // 2 - 16, 0)

        inner_product_l0c = self.tik_instance.Tensor("int32", (self.centroids_num_each_loop // 16, queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(inner_product_l0c, queries_l1, centroids_l1,
                                 queries_align, self.dim, self.centroids_num_each_loop)

        # mov centroids l2 from out to UB
        centroids_l2_ub = self.tik_instance.Tensor("int32", (self.centroids_num_each_loop,),
                                                   name="centroids_l2_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(centroids_l2_ub,
                                    self.input_precomputed_gm[aicore_move_offset + centroids_move_offset],
                                    0, 1, self.centroids_num_each_loop // 8, 0, 0)

        # x^2 + y^2 - 2xy
        add_ub = self.tik_instance.Tensor("int32", (queries_move_num, self.centroids_num_each_loop),
                                          name="add_ub", scope=tik.scope_ubuf)
        coeff_ub = self.tik_instance.Tensor("int32", (16,), tik.scope_ubuf, "coeff_ub")
        self.tik_instance.vec_dup(16, coeff_ub, self.coeff, 1, 8)

        query_l2_ub = self.tik_instance.Tensor("int32", (self.int32_mask,),
                                               name="query_l2_ub", scope=tik.scope_ubuf)

        with self.tik_instance.new_stmt_scope():
            # mov xy from L0-C to UB
            inner_product_ub = self.tik_instance.Tensor("int32",
                                                        (self.centroids_num_each_loop // 16, queries_align, 16),
                                                        name="inner_product_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(inner_product_ub, inner_product_l0c, 0, 1,
                                        centroids_align_16 * queries_align // 256, 0, 0)

            for loop_index in range(0, queries_move_num):
                self.query_l2.set_as(queries_l2_ub[loop_index])
                self.tik_instance.vec_dup(self.int32_mask, query_l2_ub, self.query_l2, 1, self.int32_mask // 8)

                # -2xy
                self.tik_instance.vmul(16, add_ub[loop_index, 0], inner_product_ub[0, loop_index, 0], coeff_ub,
                                       centroids_align_16 // 16, 1, 1, 1, 2, queries_align * 2, 0)

                # x^2
                self.tik_instance.vadd(16, add_ub[loop_index, 0], add_ub[loop_index, 0], query_l2_ub,
                                       centroids_align_16 // 16, 1, 1, 1, 16 // 8,
                                       16 // 8, 0)

                # y^2
                self.tik_instance.vadd(16, add_ub[loop_index, 0], add_ub[loop_index, 0], centroids_l2_ub,
                                       centroids_align_16 // 16, 1, 1, 1, 16 // 8,
                                       16 // 8, 2)

        dst_ub = self.tik_instance.Tensor("float16", (queries_move_num, self.centroids_num_each_loop),
                                          name="add_ub", scope=tik.scope_ubuf)
        self._conv(dst_ub, add_ub, queries_move_num * self.centroids_num_each_loop, self.int32_mask, self.scale)
    
        # Filter dst_ub
        with self.tik_instance.if_scope(self.use_mask > 0):
            min_val_ub = self.tik_instance.Tensor("float16", (128,), name="min_val_ub", scope=tik.scope_ubuf)
            self.tik_instance.vec_dup(self.fp16_mask, min_val_ub, self.default_scalar, 1, 8)
            
            # malloc memory on chip
            sel_ub = self.tik_instance.Tensor("uint8", (queries_move_num, (self.centroids_num_each_loop + 7) // 8),
                                              name="sel_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, queries_move_num) as j:
                # move data from input_mask_gm to sel_ub
                self.tik_instance.data_move(sel_ub[j, 0],
                                            self.input_mask_gm[(j + queries_move_offset) * self.mask_len + 
                                            (self.mask_offset + aicore_move_offset +
                                            centroids_move_offset) // 8],
                                            0, 1, (self.centroids_num_each_loop + 255) // 256, 8, 8)
            
            # cal the loop need execute the selection process
            vsel_loop = self.centroids_num_each_loop // self.fp16_mask
            if vsel_loop > 0:
                for vloop in range(vsel_loop):
                    # sel_ub can not use repeat times > 1, use for + offset
                    voffset = vloop * self.fp16_mask
                    # select value in dst_ub according to sel_ub
                    self.tik_instance.vec_sel(self.fp16_mask, 0, dst_ub[j, voffset],
                                              sel_ub[j, voffset // 8], dst_ub[j, voffset],
                                              min_val_ub, 1, 8, 8, 0)
                    
                    # handle tail in case of self.centroids_num_each_loop % self.fp16_mask != 0
                    vsel_last = self.centroids_num_each_loop % self.fp16_mask
                    if vsel_last > 0:
                        vsel_offset = vsel_loop * self.fp16_mask
                        self.tik_instance.vec_sel(vsel_last, 0, dst_ub[j, vsel_offset], sel_ub[j, vsel_offset // 8],
                        dst_ub[j, vsel_offset], min_val_ub, 1, 8, 8, 0)
        
        self.tik_instance.data_move(self.output_dist_gm[queries_move_offset,
                                                        aicore_move_offset + centroids_move_offset],
                                    dst_ub, 0, queries_move_num, self.centroids_num_each_loop // 16, 0,
                                    (self.centroids_num - self.centroids_num_each_loop) // 16)

        min_ub = self.tik_instance.Tensor("float16", (queries_move_num, self.centroids_num_each_loop // 32),
                                          name="min_ub", scope=tik.scope_ubuf)
        if flag == 0:
            for loop_index in range(0, queries_move_num):
                self._min(min_ub[loop_index, 0], dst_ub[loop_index, 0], centroids_move_num, self.min_mask)
                self.tik_instance.data_move(
                    self.output_min_dist_gm[queries_move_offset + loop_index, (
                            aicore_move_offset + centroids_move_offset) // self.min_mask * 2],
                    min_ub[loop_index, 0], 0, 1, (centroids_move_num + self.min_mask - 1) // self.min_mask // 8, 0, 0)
        else:
            for loop_index in range(0, queries_move_num):
                repeat_times = centroids_move_num // self.min_mask
                offset = 0
                with self.tik_instance.if_scope(repeat_times > 0):
                    self.tik_instance.vcmin(self.min_mask,
                                            min_ub[loop_index, 0], dst_ub[loop_index, 0],
                                            repeat_times, 1, 1, self.min_mask // 16)
                    offset += repeat_times * self.min_mask
                remain = centroids_move_num % self.min_mask
                with self.tik_instance.if_scope(remain > 0):
                    self.tik_instance.vcmin(remain,
                                            min_ub[loop_index, offset], dst_ub[loop_index, offset],
                                            1, 1, 1, self.min_mask // 16)
                self.tik_instance.data_move(
                    self.output_min_dist_gm[queries_move_offset + loop_index, (
                            aicore_move_offset + centroids_move_offset) // self.min_mask * 2],
                    min_ub[loop_index, 0], 0, 1,
                    ((centroids_move_num + self.min_mask - 1) // self.min_mask + 7) // 8, 0, 0)

    def _conv(self, dst, src, vconv_num, max_mask, scale):
        # process 256B data per repeat for vconv
        loop_times = vconv_num // max_mask // 255
        for _ in range(0, loop_times):
            self.tik_instance.vconv(max_mask, 'none', dst, src, 255, 1, 1, 4, 8, deqscale=scale)

        offset = 255 * max_mask * loop_times
        repeat_times = (vconv_num - offset + max_mask - 1) // max_mask
        if repeat_times > 0:
            self.tik_instance.vconv(max_mask, 'none', dst[offset], src[offset],
                                    repeat_times, 1, 1, 4, 8, deqscale=scale)

    def _mins(self, dst, src, num, mask):
        # process 256B data per repeat for vcmin
        repeat_times = num // mask
        offset = 0
        with self.tik_instance.if_scope(repeat_times > 0):
            self.tik_instance.vcmin(mask, dst, src, repeat_times, 1, 1, mask // 16)
            offset += repeat_times * mask

        remain = num % mask
        with self.tik_instance.if_scope(remain > 0):
            self.tik_instance.vcmin(remain, dst[0, offset], src[0, offset], 1, 1, 1, mask // 16)

    def _min(self, dst, src, num, mask):
        # process 256B data per repeat for vcmin
        repeat_times = num // mask
        offset = 0
        if repeat_times > 0:
            self.tik_instance.vcmin(mask, dst, src, repeat_times, 1, 1, mask // 16)
            offset += repeat_times * mask

        remain = num % mask
        if remain > 0:
            self.tik_instance.vcmin(remain, dst[0, offset], src[0, offset], 1, 1, 1, mask // 16)

    def distance_int8_l2_mins(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            aicore_centroids_num = self.tik_instance.Scalar(dtype="uint32", name="aicore_code_num", init_value=0)
            # compute  centroids num and move offset every core
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_centroids_num.set_as(self.centroids_num_each_core)
            with self.tik_instance.else_scope():
                aicore_centroids_num.set_as(self.centroids_num_last_core)

            queries_loop_time = self.queries_num // self.queries_num_each_loop
            if queries_loop_time > 0:
                with self.tik_instance.for_range(0, queries_loop_time) as loop_queries:
                    self.distance_int8_l2_each_loop(
                        block_index * self.centroids_num_each_core,
                        aicore_centroids_num,
                        loop_queries * self.queries_num_each_loop,
                        self.queries_num_each_loop)

            queries_last_num = self.queries_num % self.queries_num_each_loop
            if queries_last_num > 0:
                self.distance_int8_l2_each_loop(
                    block_index * self.centroids_num_each_core, aicore_centroids_num,
                    queries_loop_time * self.queries_num_each_loop, queries_last_num)

            one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16", (16,), name="flag_ub", scope=tik.scope_ubuf)

            flag_ub[0].set_as(one)
            self.tik_instance.data_move(self.output_flag_gm[block_index, 0], flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_int8_l2_mins()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_mask_gm,
                                       self.input_centroids_gm,
                                       self.input_precomputed_gm,
                                       self.input_actual_num_gm
                                   ],
                                   outputs=[self.output_dist_gm,
                                            self.output_min_dist_gm,
                                            self.output_flag_gm])

        return self.tik_instance


def distance_int8_l2_mins(input_queries, input_mask, input_centroids,
                          input_precomputed, input_actual_num,
                          output_dist, output_min_dist, output_flag,
                          kernel_name="distance_int8_l2"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_centroids : dict
        shape and dtype of  centroids
    input_precomputed : dict
        shape and dtype of precomputed L2 distance of  centroids
    input_actual_num: dict
        shape and dtype of actual code num,
        shape must be (8,) and dtype must be uint32
    output_dist : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_min_dist : dict
        shape and dtype of mins
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_int8_l2"

    Returns
    -------
    None
    """
    distance_int8_l2_mins_ = DistanceInt8L2Mins(input_queries,
                                                input_mask,
                                                input_centroids,
                                                input_precomputed,
                                                input_actual_num,
                                                output_dist,
                                                output_min_dist,
                                                output_flag,
                                                kernel_name)
    tik_instance_ = distance_int8_l2_mins_.get_tik_instance()
    return tik_instance_
