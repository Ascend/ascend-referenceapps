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

import sys
import time
import numpy as np


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def load_sift1M(path='./sift1M', scale=False, normalize=False):
    """
    scale_factor: 1.0 / 128.0  # scale is pre-processing to avoid fp16 overflow
    """
    start_time = time.time()
    print("Loading sift1M from %s..." % path, end='', file=sys.stderr)
    xt = fvecs_read(path + "/sift_learn.fvecs")
    xb = fvecs_read(path + "/sift_base.fvecs")
    xq = fvecs_read(path + "/sift_query.fvecs")
    gt = ivecs_read(path + "/sift_groundtruth.ivecs")
    print("done", file=sys.stderr)

    if scale:
        scale_factor = 1.0 / 128.0
        print("xt, xb, xq scaled by %f" % scale_factor)
        xt = xt * scale_factor
        xb = xb * scale_factor
        xq = xq * scale_factor

    if normalize:
        print("normalize features")
        xt_norm = np.linalg.norm(xt, ord=2, axis=1, keepdims=True)
        xt = xt / xt_norm
        xb_norm = np.linalg.norm(xb, ord=2, axis=1, keepdims=True)
        xb = xb / xb_norm
        xq_norm = np.linalg.norm(xq, ord=2, axis=1, keepdims=True)
        xq = xq / xq_norm

    print("[%.3f] Reading data from SIFT1M" % (time.time() - start_time))
    return xb, xq, xt, gt


def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    # noqa: E741
    _, indices = index.search(xq, k)
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (indices[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls
