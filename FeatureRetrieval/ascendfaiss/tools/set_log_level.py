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
import subprocess
from common import set_env


def arg_parse():
    parser = argparse.ArgumentParser(
        description='generate aicore operator model')

    parser.add_argument("-n",
                        dest='dev_num',
                        default=4,
                        type=int,
                        help="Number of miniD device")

    return parser.parse_args()


def set_ascend_env():
    """
    Set environment variable
    """
    set_env()

    ascend_toolkit_path = os.path.join(os.environ['ASCEND_HOME'],
                                       os.environ['ASCEND_VERSION'])
    os.environ['LD_LIBRARY_PATH'] = \
        ascend_toolkit_path + '/toolkit/lib64:' + \
        os.environ['LD_LIBRARY_PATH']
    os.environ['PATH'] = \
        ascend_toolkit_path + '/toolkit/tools/ide_daemon/bin:' + \
        os.environ['PATH']


if __name__ == "__main__":
    set_ascend_env()
    args = arg_parse()

    device_num = args.dev_num

    for i in range(device_num):
        print("device %d set log..." % i)
        list_ = [
            'adc', '--host', '127.0.0.1:22118', '--device', '%d' % i,
            '--log', 'SetLogLevel(0)[error]'
        ]
        return_code = subprocess.call(list_, shell=False)
        if return_code != 0:
            print("ERROR: cmd %s failed!" % list_)
            break

        list_ = [
            'adc', '--host', '127.0.0.1:22118', '--device', '%d' % i,
            '--log', 'SetLogLevel(2)[disable]'
        ]
        return_code = subprocess.call(list_, shell=False)
        if return_code != 0:
            print("ERROR: cmd %s failed!" % list_)
            break
