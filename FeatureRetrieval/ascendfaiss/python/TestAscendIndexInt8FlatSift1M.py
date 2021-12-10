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
import faiss
import datasets
import common as utils
import ascendargs as args

start_time = time.time()
base, query, train, ground_truth = datasets.load_sift1M(args.data_home, scale=args.need_scale)

dim = train.shape[1]
sq = faiss.ScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit)
sq.train(train)

base_int8 = utils.data_to_int8(sq, base)
query_int8 = utils.data_to_int8(sq, query)
train_int8 = utils.data_to_int8(sq, train)

devices = args.devices
dev = ascendfaiss.IntVector()
for d in devices:
    dev.push_back(d)
config = ascendfaiss.AscendIndexInt8FlatConfig(dev)
metric_name = "COS" if args.metric_name == "IP" else "L2"
print("[%.3f] Init AscendIndexInt8Flat, dim = %d, metric_tpye = %s, devices = %s"
      % (time.time() - start_time, dim, metric_name, str(devices)))
ascend_index = ascendfaiss.AscendIndexInt8Flat(dim, args.metric_type, config)

# add database
nb = base_int8.shape[0]
ascend_index.add(base_int8)

# search topk results
k = 1000
t, r = datasets.evaluate(ascend_index, query_int8, ground_truth, k)
print("qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f, r@1000: %.4f" %
      (1000.0 / t, r[1], r[10], r[100], r[1000]))
