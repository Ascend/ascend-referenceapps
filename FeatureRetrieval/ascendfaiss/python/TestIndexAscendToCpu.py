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

from common import log_wrapper


def test(ascend_index, xt, xb, xq, k, display):
    ascend_index.train(xt)
    ascend_index.add(xb)

    distance, labels = ascend_index.search(xq, k)
    print("Ascend Top %d of first %d queries:" % (k, display))
    print(labels[:display])
    print(distance[:display])

    cpu_index = ascendfaiss.index_ascend_to_cpu(ascend_index)
    distance, cpu_labels = cpu_index.search(xq, k)
    print("CPU Top %d of first %d queries:" % (k, display))
    print(labels[:display])
    print(distance[:display])

    assert labels[0][0] == cpu_labels[0][0]


@log_wrapper
def test_index():
    d = 512        # dims
    nlist = 16384  # L1 ivf list number
    m = 64         # PQ M
    nb = 50000     # database size
    nq = 1         # query size
    k = 10         # topk
    display = 10   # show results number

    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    xq = xb[:nq, :]

    dev_pq = ascendfaiss.IntVector()
    dev_pq.push_back(0)

    config_pq = ascendfaiss.AscendIndexIVFPQConfig(dev_pq)
    ascend_index_ivfpq = ascendfaiss.AscendIndexIVFPQ(
        d, nlist, m, 8, ascendfaiss.METRIC_L2, config_pq)

    test(ascend_index_ivfpq, xb, xb, xq, k, display)

    dev_sq = ascendfaiss.IntVector()
    dev_sq.push_back(1)

    config_sq = ascendfaiss.AscendIndexIVFSQConfig(dev_sq)
    ascend_index_ivfsq = ascendfaiss.AscendIndexIVFSQ(
        d, nlist, ascendfaiss.ScalarQuantizer.QT_8bit,
        ascendfaiss.METRIC_L2, True, config_sq)

    test(ascend_index_ivfsq, xb, xb, xq, k, display)


if __name__ == '__main__':
    test_index()
