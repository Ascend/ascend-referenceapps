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

if [ "$ASCEND_HOME" ]; then
    ASCEND_HOME="$ASCEND_HOME"
else
    ASCEND_HOME=/usr/local/Ascend
fi

if [ "$ASCEND_VERSION" ]; then
    ASCEND_VERSION="$ASCEND_VERSION"
else
    ASCEND_VERSION=ascend-toolkit/latest
fi

if [ ! -d "${ASCEND_HOME}/${ASCEND_VERSION}/atc/include" ]; then
    echo "Please set right ASCEND_HOME, now ASCEND_HOME=${ASCEND_HOME}"
    echo "Please set right ASCEND_VERSION, now ASCEND_VERSION=${ASCEND_VERSION}"
    echo "Usage: export ASCEND_HOME=\${driver/ascend-toolkit_install_path}"
    echo "       export ASCEND_VERSION=ascend-toolkit/latest"
    exit 1
fi

echo "ASCEND_TOOLKIT_PATH: ${ASCEND_HOME}/${ASCEND_VERSION}"

export ASCEND_TENSOR_COMPLIER_INCLUDE="${ASCEND_HOME}/${ASCEND_VERSION}/atc/include"
export ASCEND_OPP_PATH="${ASCEND_HOME}/${ASCEND_VERSION}/opp"
export SYSTEM_INFO=centos_aarch64
export PROJECT_PATH="$(pwd)"

mkdir -p  $PROJECT_PATH/build_out
cd $PROJECT_PATH/build_out
cmake .. -DCMAKE_CXX_COMPILER=g++
make

if [ $? -ne 0 ]; then
    echo "[ERROR] build operator faild!"
    exit 1
fi

./custom_opp_centos_aarch64.run

if [ $? -ne 0 ]; then
    echo "[ERROR] deploy operator faild!"
    exit 1
fi
