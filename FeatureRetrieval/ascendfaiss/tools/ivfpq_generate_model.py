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

# number of distance accumulate per time
distance_accumulate_num = 2048
nprobe_slice = 8


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
                        default=16384,
                        type=int,
                        help="Number of coarse centroid")
    parser.add_argument("-s",
                        dest='sub_quantizers_num',
                        default=64,
                        type=int,
                        help="Number of sub-quantizers per vector")
    parser.add_argument("-q",
                        dest='pq_centroids_num',
                        default=256,
                        type=int,
                        help="Number of pq centroids")

    return parser.parse_args()


def generate_distance_table_build_json(dim, coarse_centroid_num, sub_quantizer_num,
                                       pq_centroid_num, file_path):
    # write distance_table_build json
    distance_table_build_obj = [{
        "op":
            "DistanceTableBuild",
        "input_desc": [{
            "format": "ND",
            "shape": [1, dim],
            "type": "float16"
        }, {
            "format":
                "ND",
            "shape":
                [sub_quantizer_num,
                 pq_centroid_num * dim // sub_quantizer_num],
            "type":
                "float16"
        }, {
            "format": "ND",
            "shape": [1, nprobe_slice],
            "type": "int32"
        }, {
            "format": "ND",
            "shape": [coarse_centroid_num, dim],
            "type": "float16"
        }],
        "output_desc": [{
            "format":
                "ND",
            "shape": [1, nprobe_slice, sub_quantizer_num, pq_centroid_num],
            "type":
                "float16"
        }, {
            "format": "ND",
            "shape": [16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(distance_table_build_obj, file_path)


def generate_distance_accumulate_json(sub_quantizer_num, file_path):
    # write distance_accumulate json
    distance_accumulate_obj = [{
        "op":
            "DistAccum",
        "input_desc": [{
            "format":
                "ND",
            "shape": [distance_accumulate_num, (sub_quantizer_num + 15) // 16 * 16],
            "type":
                "float16"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [distance_accumulate_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(distance_accumulate_obj, file_path)


def generate_ivf_pq_offline_model(dim, coarse_centroid_num, sub_quantizer_num,
                                  pq_centroid_num, work_dir='.'):
    config_path = utils.get_config_path(work_dir)

    coarse_centroid_num = (coarse_centroid_num + 15) // 16 * 16

    try:
        page_sizes = (1, 2, 4, 8, 16, 32)
        flat_op_name = "distance_compute_op{}"
        for page_size in page_sizes:
            flat_op_name_ = flat_op_name.format(page_size)
            file_path_ = os.path.join(config_path, '%s.json' % flat_op_name_)
            utils.generate_dist_compute_json(page_size, dim, coarse_centroid_num, file_path_)
            utils.atc_model(flat_op_name_)

        file_path_ = os.path.join(config_path, 'distance_table_build_op.json')
        generate_distance_table_build_json(dim, coarse_centroid_num, sub_quantizer_num,
                                           pq_centroid_num, file_path_)
        utils.atc_model('distance_table_build_op')

        file_path_ = os.path.join(config_path, 'distance_accumulate_op.json')
        generate_distance_accumulate_json(sub_quantizer_num, file_path_)
        utils.atc_model('distance_accumulate_op')
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
    sub_quantizer_num = args.sub_quantizers_num
    pq_centroid_num = args.pq_centroids_num

    generate_ivf_pq_offline_model(dim, coarse_centroid_num,
                                  sub_quantizer_num, pq_centroid_num)
