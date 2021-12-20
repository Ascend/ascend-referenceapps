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
import json
import subprocess


def set_env():
    """
    Set environment variable
    """

    python_bin_path = "/usr/local/python3.7.5/bin"
    if not os.path.exists(python_bin_path):
        print("Please set python_bin_path in PATH environment.")
        print("Usage: export PATH=${python_bin_path}:$PATH")
        exit(1)

    if os.environ.get('ASCEND_HOME') is None:
        os.environ['ASCEND_HOME'] = '/usr/local/Ascend'

    if os.environ.get('ASCEND_VERSION') is None:
        os.environ['ASCEND_VERSION'] = 'ascend-toolkit/latest'

    ascend_toolkit_path = os.path.join(os.environ['ASCEND_HOME'], os.environ['ASCEND_VERSION'])
    if not os.path.exists(os.path.join(ascend_toolkit_path, "atc/bin/atc")):
        print("Please set right ASCEND_HOME, now ASCEND_HOME=%s." % os.environ['ASCEND_HOME'])
        print("Please set right ASCEND_VERSION, now ASCEND_VERSION=%s." % os.environ['ASCEND_VERSION'])
        print("Usage: export ASCEND_HOME=${driver/ascend-toolkit_install_path}")
        print("       export ASCEND_VERSION=ascend-toolkit/latest")
        exit(1)

    os.environ['PATH'] = '/usr/local/python3.7.5/bin:' + \
        ascend_toolkit_path + '/atc/ccec_compiler/bin:' + \
        ascend_toolkit_path + '/atc/bin:' + \
        os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = \
        ascend_toolkit_path + '/atc/lib64:' + \
        os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['PYTHONPATH'] = \
        ascend_toolkit_path + '/atc/python/site-packages:' + \
        os.environ.get('PYTHONPATH', '')
    os.environ['ASCEND_OPP_PATH'] = ascend_toolkit_path + '/opp'


def atc_model(json_file):
    # generate aicore operator model
    ascend_toolkit_path = os.path.join(os.environ['ASCEND_HOME'], os.environ['ASCEND_VERSION'])
    atc_path = os.path.join(ascend_toolkit_path, 'atc/bin/atc')
    return_code = subprocess.call([
        atc_path, '--singleop=./config/%s.json' % json_file,
        '--soc_version=Ascend310', '--output=op_models'
    ], shell=False)

    if return_code != 0:
        print("ERROR: atc generate %s model failed!" % json_file)
        exit(1)


def generate_op_config(dist_compute_obj, file_path):
    obj_str = json.dumps(dist_compute_obj)
    with open(file_path, 'w') as file_object:
        file_object.write(obj_str)


def generate_dist_compute_json(queries_num, dim, code_num, file_path):
    dist_compute_obj = [{
        "op":
            "DistanceComputeFlat",
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
            "shape": [code_num],
            "type": "float16"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [queries_num, code_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [32],
            "type": "uint16"
        }]
    }]

    generate_op_config(dist_compute_obj, file_path)


def get_int8_l2_norm_json(vectors_num, dim, file_path):
    # write int8_l2_norm json
    l2_norm_obj = [{
        "op":
            "Int8L2Norm",
        "input_desc": [{
            "format": "ND",
            "shape": [vectors_num, dim],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [256, 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [8],
            "type": "uint32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [vectors_num],
            "type": "float16"
        }]
    }]

    generate_op_config(l2_norm_obj, file_path)


def generate_work_dir(work_dir='.'):
    # generate directory of config
    config_path = os.path.join(work_dir, 'config')
    if not os.path.exists(config_path):
        os.makedirs(config_path)

    # generate directory of model
    op_model_path = os.path.join(work_dir, 'op_models')
    if not os.path.exists(op_model_path):
        os.makedirs(op_model_path)


def get_config_path(work_dir=''):
    generate_work_dir(work_dir)
    return os.path.join(work_dir, 'config')
