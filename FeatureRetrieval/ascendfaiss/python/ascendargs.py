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

import os
import ast
import argparse
import ascendfaiss

parser = argparse.ArgumentParser(description='Test FeatureRetrieval Sift1M recall')
parser.add_argument("--data-home", default="/data", help="sift1M home", type=str)
parser.add_argument("--devices", default=None, help="device list, split by ,", type=str)
parser.add_argument("-m", "--metric-type", default="L2", help="metric type", type=str)
parser.add_argument("--normalized", default=None, help="if need nomalized", type=ast.literal_eval)
parser.add_argument("--scaled", default=True, help="if need scaled", type=ast.literal_eval)
parser.add_argument("--by-residual", default=True, help="for ivf index", type=ast.literal_eval)
args = parser.parse_args()

devices = args.devices
metric_name = args.metric_type
metric_type = ascendfaiss.METRIC_INNER_PRODUCT if metric_name == "IP" else ascendfaiss.METRIC_L2
need_norm = metric_type == ascendfaiss.METRIC_INNER_PRODUCT
if args.normalized is not None:
    need_norm = args.normalized
need_scale = args.scaled
by_residual = args.by_residual
data_home = os.path.join(args.data_home, "sift1M")

devices = [0]
if args.devices is not None:
    devices = args.devices.split(",")
    while '' in devices:
        devices.remove('')
    devices = [int(elem) for elem in devices]
if len(devices) == 0:
    print("wrong arguments, device list is empty")
    exit(1)
