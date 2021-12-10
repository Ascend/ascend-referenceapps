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

current_folder="$( cd "$(dirname "$0")" ; pwd -P )"

SAMPLE_FOLDER=(
  InferClassification
  DecodeVideo
  EncodeJpeg
  DecodeImage
  CompileDemo
  DvppCrop
  InferObjectDetection
  InferOfflineVideo
)

#compile the sample
err_flag=0
for sample_ptr in ${SAMPLE_FOLDER[@]};do
    cd ${current_folder}/${sample_ptr}
    echo -e building "\033[0;31m$PWD\033[0m"
    bash build.sh || {
      echo -e "Failed to build ${sample_ptr}"
      err_flag=1
    }
    rm -rf build  
done

if [ ${err_flag} -eq 1 ]; then
    exit 1
fi
exit 0