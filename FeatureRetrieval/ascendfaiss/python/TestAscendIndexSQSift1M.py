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

# set running devices
devices = args.devices
dev = ascendfaiss.IntVector()
for d in devices:
    dev.push_back(d)
config = ascendfaiss.AscendIndexSQConfig(dev)

# create sq index
dim = base.shape[1]    # vector dims
print("[%.3f] Init AscendIndexSQ, dim = %d, metric_tpye = %s, devices = %s"
      % (time.time() - start_time, dim, args.metric_name, str(devices)))
ascend_index_sq = ascendfaiss.AscendIndexSQ(dim, ascendfaiss.ScalarQuantizer.QT_8bit,
                                            args.metric_type, config)
ascend_index_sq.verbose = True

# train
ascend_index_sq.train(train)

# add database
ascend_index_sq.add(base)

# search topk results
k = 1000

t, r = datasets.evaluate(ascend_index_sq, query, ground_truth, k)
print("qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f, r@1000: %.4f" %
      (1000.0 / t, r[1], r[10], r[100], r[1000]))
