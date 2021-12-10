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


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate linear_transform operator model')

    parser.add_argument("-i",
                        dest='input_dim',
                        default=512,
                        type=int,
                        help="Input dimension")
    parser.add_argument("-o",
                        dest='output_dim',
                        default=128,
                        type=int,
                        help="Output dimension")

    return parser.parse_args()


def generate_linear_transform_json(queries_num, input_dim,
                                   output_dim, file_path):
    # write linear_transform json
    linear_transform_obj = [{
        "op":
            "LinearTransform",
        "input_desc": [{
            "format": "ND",
            "shape": [queries_num, input_dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [input_dim // 16, output_dim, 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [output_dim],
            "type": "float32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [queries_num, output_dim],
            "type": "float16"
        }]
    }]

    utils.generate_op_config(linear_transform_obj, file_path)


def generate_pcar_offline_model(input_dim, output_dim, work_dir='.'):
    config_path = utils.get_config_path(work_dir)

    op_name = "linear_transform_op{}"
    search_page_sizes = (256, 128, 64, 32, 16, 8, 4, 2, 1)

    try:
        for page_size in search_page_sizes:
            op_name_ = op_name.format(page_size)
            file_path_ = os.path.join(config_path, '%s.json' % op_name_)
            generate_linear_transform_json(page_size, input_dim, output_dim, file_path_)
            utils.atc_model(op_name_)
    except OSError:
        traceback.print_exc()
        print("make sure set right ASCEND_HOME environment variable")
    else:
        traceback.print_exc()


if __name__ == '__main__':
    utils.set_env()

    args = arg_parse()
    input_dim = args.input_dim
    output_dim = args.output_dim

    generate_pcar_offline_model(input_dim, output_dim)
