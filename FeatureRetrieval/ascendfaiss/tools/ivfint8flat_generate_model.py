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

# number of code per time in int8_l2_generate_model
code_num = 65536
norm_code_num = 16384


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

    parser.add_argument("-c",
                        dest='coarse_code_num',
                        default=65536,
                        type=int,
                        help="coarse code num")

    return parser.parse_args()


def generate_distance_int8_l2_l1_json(query_num, dim, code_num, file_path):
    # write dist_compute_int8_l2_l1 json
    int8_l2_l1_obj = [{
        "op":
            "DistanceIVFInt8L2L1",
        "input_desc": [{
            "format": "ND",
            "shape": [query_num, dim],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [code_num // 16, dim // 32, 16, 32],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [code_num],
            "type": "int32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [query_num, code_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [32],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(int8_l2_l1_obj, file_path)


def generate_distance_int8_l2_json(query_num, dim, code_num, file_path):
    # write dist_compute_int8_l2 json
    int8_l2_obj = [{
        "op":
            "DistanceIVFInt8L2",
        "input_desc": [{
            "format": "ND",
            "shape": [query_num, dim],
            "type": "int8"
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
            "shape": [8],
            "type": "uint32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [query_num, code_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [query_num, (code_num + 31) // 32 * 2],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [32],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(int8_l2_obj, file_path)


def generate_distance_int8_cos_l1_json(query_num, dim, centroid_num, file_path):
    # write dist_compute_flat_min64 json
    int8_cos_l1_obj = [{
        "op":
            "DistanceIVFInt8CosL1",
        "input_desc": [{
            "format": "ND",
            "shape": [query_num, dim],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [centroid_num // 16, dim // 32, 16, 32],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [(query_num + 15) // 16 * 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [centroid_num],
            "type": "float16"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [query_num, centroid_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [32],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(int8_cos_l1_obj, file_path)


def generate_distance_int8_cos_json(query_num, dim, centroid_num, file_path):
    # write dist_compute_flat_min64 json
    int8_cos_obj = [{
        "op":
            "DistanceIVFInt8Cos",
        "input_desc": [{
            "format": "ND",
            "shape": [query_num, dim],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [centroid_num // 16, dim // 32, 16, 32],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [(query_num + 15) // 16 * 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [centroid_num],
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
            "shape": [query_num, (centroid_num + 31) // 32 * 2],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [32],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(int8_cos_obj, file_path)


def generate_ivf_int8_flat_offline_model(dim, coarse_code_num, work_dir='.'):
    config_path = utils.get_config_path(work_dir)

    int8_l2_op_name = "int8flat_l2_distance_op{}"
    int8_cos_op_name = "int8flat_cos_distance_op{}"
    search_page_sizes = (64, 32, 16, 8, 4, 2, 1)

    try:
        for page_size in search_page_sizes:
            op_name_ = int8_l2_op_name.format(page_size)
            file_path_ = os.path.join(config_path, '%s.json' % op_name_)
            generate_distance_int8_l2_l1_json(page_size, dim, coarse_code_num, file_path_)
            utils.atc_model(op_name_)

            op_name_ = int8_cos_op_name.format(page_size)
            file_path_ = os.path.join(config_path, '%s.json' % op_name_)
            generate_distance_int8_cos_l1_json(page_size, dim, coarse_code_num, file_path_)
            utils.atc_model(op_name_)

        file_path_ = os.path.join(config_path, 'ivf_int8flat_l2_distance_op1.json')
        generate_distance_int8_l2_json(1, dim, code_num, file_path_)
        utils.atc_model("ivf_int8flat_l2_distance_op1")

        file_path_ = os.path.join(config_path, 'ivf_int8flat_cos_distance_op1.json')
        generate_distance_int8_cos_json(1, dim, code_num, file_path_)
        utils.atc_model("ivf_int8flat_cos_distance_op1")

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
    dim = args.dim
    coarse_code_num = args.coarse_code_num

    generate_ivf_int8_flat_offline_model(dim, coarse_code_num)
