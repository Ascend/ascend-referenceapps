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
import ascendargs as args

start_time = time.time()
base, query, train, ground_truth = datasets.load_sift1M(args.data_home,
                                                        scale=args.need_scale,
                                                        normalize=args.need_norm)

dim = base.shape[1]    # vector dims
nlist = 2048         # L1 IVFList size

# set running devices
devices = args.devices
dev = ascendfaiss.IntVector()
for d in devices:
    dev.push_back(d)
print("[%.3f] Init AscendIndexIVFFlat, dim = %d, metric_tpye = %s, devices = %s"
      % (time.time() - start_time, dim, args.metric_name, str(devices)))
config = ascendfaiss.AscendIndexIVFFlatConfig(dev)
# create ivfsq index
ascend_index_ivfflat = ascendfaiss.AscendIndexIVFFlat(
    dim, nlist, args.metric_type, config)
ascend_index_ivfflat.verbose = True

# train
ascend_index_ivfflat.train(train)

# add database
nb = base.shape[0]   # database size
ascend_index_ivfflat.reserveMemory(nb)
ascend_index_ivfflat.add(base)

# search topk results
k = 1000
for i in [1, 2, 4, 8, 16, 32, 64]:
    ascendfaiss.AscendParameterSpace().set_index_parameter(
        ascend_index_ivfflat, 'nprobe', i)
    t, r = datasets.evaluate(ascend_index_ivfflat, query, ground_truth, k)
    print("@%3d qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f, r@1000: %.4f" %
          (i, 1000.0 / t, r[1], r[10], r[100], r[1000]))
