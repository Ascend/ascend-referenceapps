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
    [ -n "$1" ] && rm -rf $1
    mkdir -p $1
    cd $1
}

function build_a300() {
    if [ ! "${ARCH_PATTERN}" ]; then
        # set ARCH_PATTERN to acllib when it was not specified by user
        export ARCH_PATTERN=acllib
        echo "ARCH_PATTERN is set to the default value: ${ARCH_PATTERN}"
    else
        echo "ARCH_PATTERN is set to ${ARCH_PATTERN} by user, reset it to ${ARCH_PATTERN}/acllib"
        export ARCH_PATTERN=${ARCH_PATTERN}/acllib
    fi

    local path_build=$path_cur/build
    prepare_path $path_build

    # check dcmi_get_ai_model_info whether exist
    local dcmi_header="/usr/local/dcmi/dcmi_interface_api.h"
    local interface_num=$(cat "${dcmi_header}" |grep "dcmi_get_ai_model_info" 2>/dev/null |wc -l)
    if [ ${interface_num} -ne 0 ]; then
        export USE_DCMI_INTERFACE=1
    fi
    cmake -DCMAKE_BUILD_TYPE=$build_type ..
    make -j
    local ret=$?
    cd ..
    return ${ret}
}

# build program which can run on A500
function build_a500() {
    # need to set ARCH_PATTERN to aarch64
    version_path="${ASCEND_HOME}/${ASCEND_VERSION}"
    arch_pattern_path=$(ls ${version_path} |grep "^arm64-linux")
    if [ $? -ne 0 ] || [ ! -d "${version_path}/${arch_pattern_path}" ]; then
        echo "Thers is no aarch64 acllib under ${version_path}, please install the Ascend-Toolkit package with aarch64 pattern."
        return 1
    fi
    export ARCH_PATTERN="${arch_pattern_path}/acllib"
    echo "export ARCH_PATTERN=${arch_pattern_path}/acllib"
    local path_build=$path_cur/build
    prepare_path $path_build

    # check dcmi_get_ai_model_info whether exist
    local dcmi_header="/usr/local/dcmi/dcmi_interface_api.h"
    local interface_num=$(cat "${dcmi_header}" |grep "dcmi_get_ai_model_info" 2>/dev/null |wc -l)
    if [ ${interface_num} -ne 0 ]; then
        export USE_DCMI_INTERFACE=1
    fi
    cmake -DCMAKE_BUILD_TYPE=$build_type -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ ..
    make -j
    local ret=$?
    cd ..
    return ${ret}
}

# set ASCEND_VERSION to ascend-toolkit/latest when it was not specified by user
if [ ! "${ASCEND_VERSION}" ]; then
    export ASCEND_VERSION=ascend-toolkit/latest
    echo "Set ASCEND_VERSION to the default value: ${ASCEND_VERSION}"
else
    echo "ASCEND_VERSION is set to ${ASCEND_VERSION} by user"
fi
# build with different according to the parameter, default is A300
if [ "$1" == "A500" ]; then
    build_a500
else
    build_a300
fi

# copy config file and model into dist
if [ ! -d dist ]; then
    echo "Build failed, dist directory does not exist."
    exit 1
fi
rm -rf ./dist/data
mkdir ./dist/data
cp -r ./data/config ./dist/data/

mkdir -p ./dist/data/models/yolov3
cp ./data/models/yolov3/coco.names ./dist/data/models/yolov3

om_num=$(ls ./data/models/yolov3/*.om 2>/dev/null |wc -l)
if [ ${om_num} -eq 0 ]; then
    echo "[Warning] No .om file in ./data/models/yolov3, please convert the model to .om file first."
else
    cp ./data/models/yolov3/*.om ./dist/data/models/yolov3
fi
exit 0