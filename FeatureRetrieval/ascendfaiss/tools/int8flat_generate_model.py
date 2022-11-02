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

# number of code per time in int8_generate_model
code_num = 16384 * 16
norm_code_num = 16384


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate aicore operator model')

    parser.add_argument("--cores",
                        dest='core_num',
                        default=2,
                        type=int,
                        help="Core number")
    
    parser.add_argument("-d",
                        dest='dim',
                        default=512,
                        type=int,
                        help="Feature dimension")

    parser.add_argument("-c",
                        dest='code_size',
                        default=16384,
                        type=int,
                        help="Code size for compute norm")

    return parser.parse_args()


def generate_distance_int8_cos_maxs_json(core_num, query_num, dim, file_path):
    # write dist_int8_cos_maxs json
    int8_cos_maxs_obj = [{
        "op":
            "DistanceInt8CosMaxs",
        "input_desc": [{
            "format": "ND",
            "shape": [query_num, dim],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [query_num, (code_num + 7) // 8],
            "type": "uint8"
        }, {
            "format": "ND",
            "shape": [code_num // 16, dim // 32, 16, 32],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [(query_num + 15) // 16 * 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [code_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_num, 8],
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
            "shape": [16, 16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(int8_cos_maxs_obj, file_path)


def generate_distance_int8_l2_mins_json(core_num, query_num, dim, file_path):
    # write dist_int8_l2_mins json
    int8_l2_mins_obj = [{
        "op":
            "DistanceInt8L2Mins",
        "input_desc": [{
            "format": "ND",
            "shape": [query_num, dim],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [query_num, (code_num + 7) // 8],
            "type": "uint8"
        }, {
            "format": "ND",
            "shape": [code_num // 16, dim // 32, 16, 32],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [code_num],
            "type": "int32"
        }, {
            "format": "ND",
            "shape": [core_num, 8],
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
            "shape": [16, 16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(int8_l2_mins_obj, file_path)


def generate_int8_offline_model(core_num, dim, work_dir='.'):
    config_path = utils.get_config_path(work_dir)

    int8_l2_mins_op_name = "int8_flat_l2_mins_op{}"
    int8_cos_maxs_op_name = "int8_flat_cos_maxs_op{}"
    search_page_sizes = (48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1)

    try:
        for page_size in search_page_sizes:
            op_name_ = int8_l2_mins_op_name.format(page_size)
            file_path_ = os.path.join(config_path, '%s.json' % op_name_)
            generate_distance_int8_l2_mins_json(core_num, page_size, dim, file_path_)
            utils.atc_model(op_name_)

            op_name_ = int8_cos_maxs_op_name.format(page_size)
            file_path_ = os.path.join(config_path, '%s.json' % op_name_)
            generate_distance_int8_cos_maxs_json(core_num, page_size, dim, file_path_)
            utils.atc_model(op_name_)

        op_name_ = "int8_l2_norm_d{}".format(dim)
        file_path_ = os.path.join(config_path, '%s.json' % op_name_)
        utils.get_int8_l2_norm_json(norm_code_num, dim, file_path_)
        utils.atc_model(op_name_)
    except OSError:
        traceback.print_exc()
        print(
            "make sure set right ASCEND_HOME and ASCEND_VERSION environment "
            "variable")
    else:
        traceback.print_exc()


if __name__ == '__main__':
    utils.set_env()

    args = arg_parse()
    core_num = args.core_num
    dim = args.dim

    generate_int8_offline_model(core_num, dim)
