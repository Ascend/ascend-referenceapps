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
import time
import argparse
import subprocess
from xml.dom.minidom import parse


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate linear_transform operator model')

    parser.add_argument("-m",
                        dest='mode',
                        default='ALL',
                        type=str,
                        help="Generate mode, include combination of PCAR,\
            Flat, IVFPQ, SQ8, IVFSQ8 and ALL")

    return parser.parse_args()


def get_field_value(node, field):
    value = node.getElementsByTagName(field)[0]
    return int(value.childNodes[0].data)


def get_attribute_value(node, attribute):
    return int(node.getAttribute(attribute))


def run_command(model_id, arguments):
    print("generate model of ID %d" % model_id)
    return_code_ = subprocess.call(arguments, shell=False)
    print("")
    if return_code_ != 0:
        print("ERROR: generate model of ID %d failed" % model_id)
        exit(1)


def run_pcar(parameter):
    input_dim_ = get_field_value(parameter, "input_dim")
    output_dim_ = get_field_value(parameter, "output_dim")
    model_id_ = get_attribute_value(parameter, "ID")

    arguments_ = ['python', 'pcar_generate_model.py',
                  '-i', '%d' % input_dim_,
                  '-o', '%d' % output_dim_]

    run_command(model_id_, arguments_)


def run_flat(parameter):
    dim_ = get_field_value(parameter, "dim")
    model_id_ = get_attribute_value(parameter, "ID")

    arguments_ = [
        'python', 'flat_generate_model.py',
        '-d', '%d' % dim_
    ]
    run_command(model_id_, arguments_)


def run_ivf_flat(parameter):
    dim_ = get_field_value(parameter, "dim")
    coarse_centroid_num_ = get_field_value(parameter, "nlist")
    model_id_ = get_attribute_value(parameter, "ID")

    arguments_ = [
        'python', 'ivfflat_generate_model.py',
        '-d', '%d' % dim_,
        '-c', '%d' % coarse_centroid_num_
    ]
    run_command(model_id_, arguments_)


def run_ivf_pq(parameter):
    dim_ = get_field_value(parameter, "dim")
    coarse_centroid_num_ = get_field_value(parameter, "nlist")
    sub_quantizer_num_ = get_field_value(parameter, "pq")
    model_id_ = get_attribute_value(parameter, "ID")

    arguments_ = [
        'python', 'ivfpq_generate_model.py',
        '-d', '%d' % dim_,
        '-c', '%d' % coarse_centroid_num_,
        '-s', '%d' % sub_quantizer_num_
    ]
    run_command(model_id_, arguments_)


def run_sq8(parameter):
    dim_ = get_field_value(parameter, "dim")
    model_id_ = get_attribute_value(parameter, "ID")

    arguments_ = [
        'python', 'sq8_generate_model.py',
        '-d', '%d' % dim_
    ]
    run_command(model_id_, arguments_)


def run_ivf_sq8(parameter):
    dim_ = get_field_value(parameter, "dim")
    coarse_centroid_num_ = get_field_value(parameter, "nlist")
    model_id_ = get_attribute_value(parameter, "ID")

    arguments_ = [
        'python', 'ivfsq8_generate_model.py',
        '-d', '%d' % dim_,
        '-c', '%d' % coarse_centroid_num_
    ]
    run_command(model_id_, arguments_)


def run_int8_flat(parameter):
    dim_ = get_field_value(parameter, "dim")
    model_id_ = get_attribute_value(parameter, "ID")

    arguments_ = [
        'python', 'int8flat_generate_model.py',
        '-d', '%d' % dim_
    ]
    run_command(model_id_, arguments_)


def run_int8_ivf_flat(parameter):
    dim_ = get_field_value(parameter, "dim")
    coarse_centroid_num_ = get_field_value(parameter, "nlist")
    model_id_ = get_attribute_value(parameter, "ID")

    arguments_ = [
        'python', 'ivfint8flat_generate_model.py',
        '-d', '%d' % dim_,
        '-c', '%d' % coarse_centroid_num_
    ]
    run_command(model_id_, arguments_)


def generate_offline_model(parameter, para_mode):
    if para_mode == 'PCAR':
        run_pcar(parameter)
    elif para_mode == 'Flat':
        run_flat(parameter)
    elif para_mode == 'IVFPQ':
        run_ivf_pq(parameter)
    elif para_mode == 'SQ8':
        run_sq8(parameter)
    elif para_mode == 'IVFSQ8':
        run_ivf_sq8(parameter)
    elif para_mode == 'IVFFlat':
        run_ivf_flat(parameter)
    elif para_mode == 'INT8':
        run_int8_flat(parameter)
    elif para_mode == 'IVFINT8':
        run_int8_ivf_flat(parameter)


if __name__ == '__main__':
    start_time = time.time()
    args = arg_parse()
    mode = args.mode

    modes = mode.split(',')
    for mode_ in modes:
        if mode_ not in ['PCAR', 'Flat', 'IVFPQ', 'SQ8', 'IVFSQ8', 'IVFFlat', 'INT8', 'IVFINT8', 'ALL']:
            raise RuntimeError("generate mode only support PCAR, Flat, \
                    IVFPQ, SQ8, IVFSQ8, IVFFlat, INT8, IVFINT8 and ALL!")

    config_name = './para_table.xml'
    domTree = parse(config_name)
    rootNode = domTree.documentElement

    parameters = rootNode.getElementsByTagName("parameter")
    for parameter_ in parameters:
        if parameter_.hasAttribute("ID"):
            mode_ = parameter_.getElementsByTagName("mode")[0]
            para_mode_ = str(mode_.childNodes[0].data)

            if mode == 'ALL' or para_mode_ in modes:
                generate_offline_model(parameter_, para_mode_)
            else:
                continue
    print("generate model time duration: %.3f seconds" % (time.time() - start_time))
