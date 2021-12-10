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

import time
import ascendfaiss
import datasets
from common import create_ascend_index_ivfsq
import ascendargs as args

start_time = time.time()
base, query, train, ground_truth = datasets.load_sift1M(args.data_home,
                                                        scale=args.need_scale,
                                                        normalize=args.need_norm)

# vector dims
d_in = base.shape[1]
dim = d_in // 2

# L1 IVFList size
nlist = 8192

# set ClusteringParameters
cp = ascendfaiss.ClusteringParameters()
cp.niter = 16
cp.max_points_per_centroid = 256

# create ivfsq index
devices = args.devices
print("[%.3f] Init AscendIndexIVFSQ, dim = %d, metric_tpye = %s, byResidual = %s, devices = %s"
      % (time.time() - start_time, dim, args.metric_name, str(args.by_residual), str(devices)))
ascend_index_ivfsq = create_ascend_index_ivfsq(devices, dim, nlist,
                                               args.metric_type, args.by_residual, cp)
ascend_index_ivfsq.verbose = True

print("[%.3f] Init AscendIndexPreTransform, dim_in = %d, dim_out = %d"
      % (time.time() - start_time, d_in, dim))
ascend_index_pretransform = ascendfaiss.AscendIndexPreTransform(
    ascend_index_ivfsq)
ascend_index_pretransform.prependPCA(d_in, dim, 0, True)

# train
ascend_index_pretransform.train(train)

# database size
nb = base.shape[0]
ascend_index_pretransform.reserveMemory(nb)
ascend_index_pretransform.add(base)

# search topk results
k = 1000

for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    ascendfaiss.AscendParameterSpace().set_index_parameter(
        ascend_index_pretransform, 'nprobe', i)
    t, r = datasets.evaluate(ascend_index_pretransform, query, ground_truth, k)
    print("@%3d qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f, r@1000: %.4f" %
          (i, 1000.0 / t, r[1], r[10], r[100], r[1000]))
