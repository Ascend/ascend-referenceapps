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

set -e

path_cur=$(cd $(dirname $0); pwd)
build_type="Release"

function prepare_path() {
    if [ -d "$1" ]; then
        rm -rf $1
    fi
    mkdir -p  $1
    cd  $1
}

function check_mode() {
    enable_web_server=false
    enable_ctrl_cpu=false
    if [ ${param_num} -gt 0 ]; then
        for param in ${params};
        do
            if [ "${param}" = "web" ]; then
                enable_web_server=true
                break
            fi
            if [ "${param}" = "ctrl" ]; then
                enable_ctrl_cpu=true
                break
            fi
        done
    fi
}

param_num=$#
params=$@
check_mode

echo ${path_cur}
path_build=$path_cur/build
prepare_path $path_build
if [ "${enable_web_server}" = true ]; then
    CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=$build_type -DENABLE_WEB_SERVER=ON ../
else
    if  [ "${enable_ctrl_cpu}" = true ]; then
        CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=$build_type -DENABLE_CTRL_CPU=ON ../
    else
        CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=$build_type ../
    fi
fi
sudo make -j
if [ $? -ne 0 ]; then
    echo "Build Failed"
    exit -1
fi
cd ..

if [ $? -ne 0 ]; then
    echo "build error"
    exit -1
fi
exit 0


