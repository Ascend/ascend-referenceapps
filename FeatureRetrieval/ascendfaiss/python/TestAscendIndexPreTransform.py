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
import faiss
import ascendfaiss

d = 128      # vector dims
d_in = 512

nb = 200000  # databse size
nq = 10000       # query size
np.random.seed(1234)
xb = np.random.random((nb, d_in)).astype('float32')
xq = xb[:nq, :]

nlist = 16384  # L1 IVFList size

# set running devices
dev = ascendfaiss.IntVector()
for i in range(0, 4):
    dev.push_back(i)

config = ascendfaiss.AscendIndexIVFSQConfig(dev)
# create ivfsq index
ascend_index_ivfsq = ascendfaiss.AscendIndexIVFSQ(
    d, nlist, ascendfaiss.ScalarQuantizer.QT_8bit,
    ascendfaiss.METRIC_L2, True, config)
ascend_index_ivfsq.verbose = True

ascend_index_pretransform = ascendfaiss.AscendIndexPreTransform(ascend_index_ivfsq)
ascend_index_pretransform.prependPCA(d_in, d, 0, True)

# train
ascend_index_pretransform.train(xb)

cpu_index_pretransform = ascendfaiss.index_ascend_to_cpu(ascend_index_pretransform)
faiss.write_index(cpu_index_pretransform, "ptInterfaceTest.afterTrain.faiss")

# cpu to ascend
cpu_index_pretransform = faiss.read_index("ptInterfaceTest.afterTrain.faiss")
ascend_index_pretransform = ascendfaiss.index_cpu_to_ascend(dev, cpu_index_pretransform)
ascend_index_pretransform.verbose = True

# add database
ascend_index_pretransform.reserveMemory(nb)
ascend_index_pretransform.add(xb)
ascend_index_pretransform.reclaimMemory()

cpu_index_pretransform = ascendfaiss.index_ascend_to_cpu(ascend_index_pretransform)
faiss.write_index(cpu_index_pretransform, "ptInterfaceTest.afterAdd.faiss")

# search topk results
k = 10
display = 10
distance, labels = ascend_index_pretransform.search(xq, k)
print("Top %d of first %d queries:" % (k, display))
print(labels[:display])
assert labels[0][0] == 0

# remove vector
ids_remove = ascendfaiss.IDSelectorRange(0, 1)
ids_remove_batch = labels[0][:int(k / 2)].copy()

print("Remove top1")
num_removed = ascend_index_pretransform.remove_ids(ids_remove)
print("Removed num:", num_removed)
assert num_removed == 1
distance, labels = ascend_index_pretransform.search(xq, k)
print("Top %d of first %d queries after remove top1:" % (k, display))
print(labels[:display])

print("Remove top5", ids_remove_batch)
num_removed = ascend_index_pretransform.remove_ids(ids_remove_batch)
print("Removed num:", num_removed)
assert num_removed == int(k / 2) - 1
distance, labels = ascend_index_pretransform.search(xq, k)
print("Top %d of first %d queries after delete top5:" % (k, display))
print(labels[:display])

# reset
ascend_index_pretransform.reset()

# cpu to ascend
cpu_index_pretransform = faiss.read_index("ptInterfaceTest.afterAdd.faiss")
ascend_index_pretransform = ascendfaiss.index_cpu_to_ascend(dev, cpu_index_pretransform)
ascend_index_pretransform.verbose = True
distance, labels = ascend_index_pretransform.search(xq, k)
print("Top %d of first %d queries after CPUtoAscend:" % (k, display))
print(labels[:display])
