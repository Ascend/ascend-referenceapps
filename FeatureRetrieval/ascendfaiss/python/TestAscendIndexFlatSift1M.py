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

devices = args.devices
dev = ascendfaiss.IntVector()
for d in devices:
    dev.push_back(d)
config = ascendfaiss.AscendIndexFlatConfig(dev)

dim = base.shape[1]
print("[%.3f] Init AscendIndexFlat, dim = %d, metric_tpye = %s, devices = %s"
      % (time.time() - start_time, dim, args.metric_name, str(devices)))
index = ascendfaiss.AscendIndexFlat(dim, args.metric_type, config)

print("[%.3f] Start building database, size = %d" % (
      time.time() - start_time, base.shape[0]))
index.add(base)

K = 100
print("[%.3f] Start searching, query num = %d, topk = %d" % (
      time.time() - start_time, query.shape[0], K))

t, r = datasets.evaluate(index, query, ground_truth, K)
print("qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f" % (
      1000.0 / t, r[1], r[10], r[100]))
