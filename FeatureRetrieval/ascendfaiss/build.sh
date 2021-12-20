#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# generate configure file
./autogen.sh
result=$?
if [ ${result} -ne 0 ]; then
    echo "[ERROR] autogen faild!"
    exit 1
fi

# set python interpreter, make sure installing Numpy in advance
if [ ! $PYTHON ]; then
    export PYTHON=/usr/bin/python3.7.5
fi

# configure
./configure
result=$?
if [ ${result} -ne 0 ]; then
    echo "[ERROR] check environment failed!"
    exit 1
fi

# run make
export PKG_CONFIG_PATH=/usr/local/protobuf/lib/pkgconfig
echo "PKG_CONFIG_PATH: ${PKG_CONFIG_PATH}"
make clean && make -j
result=$?
if [ ${result} -ne 0 ]; then
    echo "[ERROR] make failed!"
    exit 1
fi

# run make install
make install
result=$?
if [ ${result} -ne 0 ];then
    echo "[ERROR] make install faild!"
    exit 1
fi

# run make python interface
make -C python
result=$?
if [ ${result} -ne 0 ];then
    echo "[ERROR] make install python interface faild!"
    exit 1
fi

# run make python interface install
make -C python install
result=$?
if [ ${result} -ne 0 ];then
    echo "[ERROR] make python interface install faild!"
    exit 1
fi

# registering custom tbe operator
cd ops && bash build.sh
result=$?
if [ ${result} -ne 0 ];then
    echo "[ERROR] build and install tbe operators faild!"
    exit 1
fi

# convert custom tbe operator to om files
if [ "$1" != "--nogeneratemodel" ];then
    cd ../tools
    rm -rf op_models config kernel_meta ../modelpath/*
    python run_generate_model.py
    result=$?
    if [ ${result} -ne 0 ];then
        echo "[ERROR] generate om models faild!"
        exit 1
    fi
    mv op_models/* ../modelpath
    cd ..
fi
