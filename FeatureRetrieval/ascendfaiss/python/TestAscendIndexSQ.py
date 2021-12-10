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

import numpy as np
import ascendfaiss

d = 128      # vector dims
nb = 100000  # databse size
nq = 1       # query size
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = xb[:nq, :]

# set running devices
dev = ascendfaiss.IntVector()
dev.push_back(0)
config = ascendfaiss.AscendIndexSQConfig(dev)

# create sq index
ascend_index_sq = ascendfaiss.AscendIndexSQ(
    d, ascendfaiss.ScalarQuantizer.QT_8bit,
    ascendfaiss.METRIC_L2, config)
ascend_index_sq.verbose = True

# train
ascend_index_sq.train(xb)

# add database
ascend_index_sq.add(xb)

# search topk results
k = 10
display = 10
distance, labels = ascend_index_sq.search(xq, k)
print("Top %d of first %d queries:" % (k, display))
print(labels[:display])
assert labels[0][0] == 0

# remove vector
ids_remove = ascendfaiss.IDSelectorRange(0, 1)
ids_remove_batch = labels[0][:int(k / 2)].copy()

print("Remove top1")
num_removed = ascend_index_sq.remove_ids(ids_remove)
print("Removed num:", num_removed)
assert num_removed == 1
distance, labels = ascend_index_sq.search(xq, k)
print("Top %d of first %d queries after remove top1:" % (k, display))
print(labels[:display])

print("Remove top5", ids_remove_batch)
num_removed = ascend_index_sq.remove_ids(ids_remove_batch)
print("Removed num:", num_removed)
assert num_removed == int(k / 2) - 1
distance, labels = ascend_index_sq.search(xq, k)
print("Top %d of first %d queries after delete top5:" % (k, display))
print(labels[:display])
