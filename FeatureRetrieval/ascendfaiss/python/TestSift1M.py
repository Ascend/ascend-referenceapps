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
import argparse

import ascendfaiss
import datasets

start_time = time.time()

# parse arguments
parser = argparse.ArgumentParser(description='Test SIFT1M with AscendIVFPQ')
parser.add_argument('--nlist', '-L', type=int, required=True)
parser.add_argument('--subquantizer', '-M', type=int, required=True)
args = vars(parser.parse_args())

base, query, train, ground_truth = datasets.load_sift1M('./sift1M', scale=True)

D = base.shape[1]
K = 100

# index parameters
M = args['subquantizer']
NLIST = args['nlist']
CODE_SIZE = 8
MAX_NPROBE = 512

DEVICE = [0]
dev = ascendfaiss.IntVector()
for d in DEVICE:
    dev.push_back(d)
config = ascendfaiss.AscendIndexIVFPQConfig(dev)

print("[%.3f] Init AscendIndexIVFPQ, M = %d, KSUB = %d, NLIST = %d"
      % (time.time() - start_time, M, 2**CODE_SIZE, NLIST))
index = ascendfaiss.AscendIndexIVFPQ(D, NLIST, M, CODE_SIZE,
                                     ascendfaiss.METRIC_L2, config)

print("[%.3f] Start training, size = %d"
      % (time.time() - start_time, train.shape[0]))
index.train(train)

print("[%.3f] Start building database, size = %d"
      % (time.time() - start_time, base.shape[0]))
index.add(base)

print("[%.3f] Start searching, query num = %d, topk = %d"
      % (time.time() - start_time, query.shape[0], K))
nprobe = 1
while nprobe <= MAX_NPROBE:
    index.setNumProbes(nprobe)
    search_start = time.time()
    print("[%.3f] Query with nprobe %4d"
          % (search_start - start_time, nprobe), end=', ')
    t, r = datasets.evaluate(index, query, ground_truth, K)
    print("qps: %.3f, r@1: %.4f, r@10: %.4f, r@100: %.4f"
          % (1000.0 / t, r[1], r[10], r[100]))
    nprobe *= 2
