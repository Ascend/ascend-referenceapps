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

code_num = 2048


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
                        dest='coarse_centroid_num',
                        default=2048,
                        type=int,
                        help="Number of coarse centroid")

    return parser.parse_args()


def generate_ivf_flat_offline_model(dim, centroid_num, work_dir='.'):
    config_path = utils.get_config_path(work_dir)

    flat_op_name = "distance_compute_op{}"
    search_page_sizes = (64, 32, 16, 8, 4, 2, 1)

    try:
        for page_size in search_page_sizes:
            flat_op_name_ = flat_op_name.format(page_size)
            file_path_ = os.path.join(config_path, '%s.json' % flat_op_name_)
            utils.generate_dist_compute_json(page_size, dim, centroid_num, file_path_)
            utils.atc_model(flat_op_name_)

        file_path_ = os.path.join(config_path, 'distance_compute_ivfflat_op.json')
        utils.generate_dist_compute_json(1, dim, code_num, file_path_)
        utils.atc_model('distance_compute_ivfflat_op')
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
    centroid_num = args.coarse_centroid_num

    generate_ivf_flat_offline_model(dim, centroid_num)
