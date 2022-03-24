#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
# limitations under the License.mitations under the License.
 
set -e 
current_folder="$( cd "$(dirname "$0")" ;pwd -P )"

export ASCEND_HOME=/usr/local/Ascend
export DRIVER_HOME=/usr/local/Ascend
export ASCEND_VERSION=ascend-toolkit/latest
export ARCH_PATTERN=arm64-linux
export LD_LIBRARY_PATH=${ASCEND_HOME}/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=/usr/local/opencv/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/opencv/lib:$LD_LIBRARY_PATH

export FFMPEG_PATH=/usr/local/ffmpeg

SAMPLE_FOLDER=(
    ../ApiSamples/Samples/
)

err_flag=0
for sample in ${SAMPLE_FOLDER[@]};do
    cd ${current_folder}/${sample}
    bash build.sh || {
        echo -e "Failed to build ${sample}"
		err_flag=1
    }
done

if [ ${err_flag} -eq 1 ]; then
    echo build_failed
	exit 1
fi

echo build_ok
exit 0
