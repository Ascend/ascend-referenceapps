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
import numpy as np
import faiss
import ascendfaiss


def log_wrapper(func):
    def wrapper():
        print('start testing %s ...' % func.__name__)
        func()
        print('--------' * 10)

    return wrapper


def create_ascend_index_ivfsq(devices, d, nlist, metric, by_residual=True, cluster_parameter=None):
    # set running devices
    dev = ascendfaiss.IntVector()
    for i in devices:
        dev.push_back(i)

    # set config(device config and clustering config)
    config = ascendfaiss.AscendIndexIVFSQConfig(dev)
    if cluster_parameter is not None:
        config.cp = cluster_parameter

    # create ivfsq index
    ascend_index_ivfsq = ascendfaiss.AscendIndexIVFSQ(d, nlist, ascendfaiss.ScalarQuantizer.QT_8bit,
                                                      metric, by_residual, config)
    return ascend_index_ivfsq


def data_to_int8(sq, data):
    codes = sq.compute_codes(data)
    codes_int8 = codes.astype("int32") - 128
    return codes_int8.astype("int8")


def train_kmeans(x, ncentroids):
    niter = 10
    verbose = True
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(x)
    return kmeans.centroids


def random_feature(dim, feature_count, seed=7777777, normalize=True):
    print("generate: dim %d, feature num: %d, seed: %d" % (dim, feature_count, seed))
    np.random.seed(seed)
    ts = time.time()
    xb = np.random.random((feature_count, dim)).astype('float32')
    if normalize:
        xb_norm = np.linalg.norm(xb, ord=2, axis=1, keepdims=True)
        xb = xb / xb_norm
        print("normalize features")
    te = time.time()
    print("generate used: %.4f s" % (te - ts))
    return xb
