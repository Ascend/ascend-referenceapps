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
import argparse
import traceback
import common as utils

# number of code per time
code_num = 65536


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate distance_compute_sq8 operator model')

    parser.add_argument("-d",
                        dest='dim',
                        default=128,
                        type=int,
                        help="Feature dimension")
    parser.add_argument("-c",
                        dest='coarse_centroid_num',
                        default=16384,
                        type=int,
                        help="Number of coarse centroid")

    return parser.parse_args()


def generate_distance_sq8_ip_json(query_num, dim, centroid_num, file_path):
    max_batch = 32
    if dim > 128:
        max_batch = 16

    # write distance_sq8_ip json
    dist_sq8_obj = [{
        "op":
            "DistanceIVFSQ8IP",
        "input_desc": [{
            "format": "ND",
            "shape": [query_num, dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [centroid_num // 16, dim // 16, 16, 16],
            "type": "uint8"
        }, {
            "format": "ND",
            "shape": [dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [8],
            "type": "uint32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [query_num, centroid_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [query_num, (code_num + max_batch - 1) // max_batch * 2],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [32],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(dist_sq8_obj, file_path)


def generate_distance_sq8_l2_json(query_num, dim, centroid_num, file_path):
    min_batch = 32
    if dim > 128:
        min_batch = 16

    # write distance_sq8_l2 json
    distance_sq8_obj = [{
        "op":
            "DistanceIVFSQ8L2",
        "input_desc": [{
            "format": "ND",
            "shape": [query_num, dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [centroid_num // 16, dim // 16, 16, 16],
            "type": "uint8"
        }, {
            "format": "ND",
            "shape": [centroid_num],
            "type": "float32"
        }, {
            "format": "ND",
            "shape": [dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [8],
            "type": "uint32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [query_num, centroid_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [query_num, (centroid_num + min_batch - 1) // min_batch * 2],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [32],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(distance_sq8_obj, file_path)


def generate_ivf_sq8_offline_model(dim, coarse_centroid_num, work_dir='.'):
    config_path = utils.get_config_path(work_dir)

    flat_op_name = "distance_compute_flat_l2_op{}"
    search_page_sizes = (64, 32, 16, 8, 4, 2, 1)

    try:
        for page_size in search_page_sizes:
            flat_op_name_ = flat_op_name.format(page_size)
            file_path_ = os.path.join(config_path, '%s.json' % flat_op_name_)
            utils.generate_dist_compute_json(page_size, dim, coarse_centroid_num, file_path_)
            utils.atc_model(flat_op_name_)

        file_path_ = os.path.join(config_path, 'distance_ivf_sq8_ip_op.json')
        generate_distance_sq8_ip_json(1, dim, code_num, file_path_)
        utils.atc_model('distance_ivf_sq8_ip_op')

        file_path_ = os.path.join(config_path, 'distance_ivf_sq8_l2_op.json')
        generate_distance_sq8_l2_json(1, dim, code_num, file_path_)
        utils.atc_model('distance_ivf_sq8_l2_op')
    except OSError:
        traceback.print_exc()
        print("make sure set right ASCEND_HOME environment variable")
    else:
        traceback.print_exc()


if __name__ == '__main__':
    utils.set_env()

    args = arg_parse()
    dim = args.dim
    coarse_centroid_num = args.coarse_centroid_num

    generate_ivf_sq8_offline_model(dim, coarse_centroid_num)
