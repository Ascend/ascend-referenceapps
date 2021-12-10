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

# number of code per time in distance_compute_flat_min64
code_num = 16384 * 16


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate aicore operator model')

    parser.add_argument("-d",
                        dest='dim',
                        default=512,
                        type=int,
                        help="Feature dimension")

    return parser.parse_args()


def generate_distance_flat_l2_mins_json(query_num, dim, code_num, file_path):
    # write dist_compute_flat_mins json
    dist_compute_flat_mins_obj = [{
        "op":
            "DistanceComputeFlatMin64",
        "input_desc": [{
            "format": "ND",
            "shape": [query_num, dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [code_num // 16, dim // 16, 16, 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [code_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [8],
            "type": "uint32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [query_num, code_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [query_num, (code_num + 63) // 64 * 2],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [32],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(dist_compute_flat_mins_obj, file_path)


def generate_distance_flat_ip_maxs_json(queries_num, dim, code_num, file_path):
    # write dist_compute_flat_mins json
    dist_flat_ip_maxs_obj = [{
        "op":
            "DistanceFlatIPMaxs",
        "input_desc": [{
            "format": "ND",
            "shape": [queries_num, dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [code_num // 16, dim // 16, 16, 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [8],
            "type": "uint32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [queries_num, code_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [queries_num, (code_num + 63) // 64 * 2],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [32],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(dist_flat_ip_maxs_obj, file_path)


def generate_flat_offline_model(dim, code_num, work_dir='.'):
    config_path = utils.get_config_path(work_dir)

    flat_l2_op_name = "distance_flat_l2_maxs_op{}"
    flat_ip_op_name = "distance_flat_ip_maxs_op{}"
    search_page_sizes = (96, 64, 48, 42, 36, 32, 30, 24, 18,
                         16, 12, 8, 6, 4, 2, 1)

    try:
        for page_size in search_page_sizes:
            op_name_ = flat_l2_op_name.format(page_size)
            file_path_ = os.path.join(config_path, '%s.json' % op_name_)
            generate_distance_flat_l2_mins_json(page_size, dim, code_num, file_path_)
            utils.atc_model(op_name_)

            op_name_ = flat_ip_op_name.format(page_size)
            file_path_ = os.path.join(config_path, '%s.json' % op_name_)
            generate_distance_flat_ip_maxs_json(page_size, dim, code_num, file_path_)
            utils.atc_model(op_name_)
    except OSError:
        traceback.print_exc()
        print("make sure set right ASCEND_HOME and ASCEND_VERSION "
              "environment variable")
    else:
        traceback.print_exc()


if __name__ == '__main__':
    utils.set_env()

    args = arg_parse()
    dim = args.dim

    generate_flat_offline_model(dim, code_num)
