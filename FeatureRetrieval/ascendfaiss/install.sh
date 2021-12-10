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

project_path="$(pwd)" # project path
ascend_path=/usr/local/Ascend # ascend path
acllib_minios_path=/usr/local/AscendMiniOSRun # minios path

driver_src_path=$1

# check packages
if [ ! -d ${ascend_path} ];then
    echo "Please install Ascend package in ${ascend_path}"
    exit 1
fi

if [ ! -d ${acllib_minios_path} ];then
    echo "Please install acllib minios.aarch64 in /usr/local/AscendMiniOSRun or modify acllib_minios_path in install.sh."
    exit 1
fi

if ([ -d "${project_path}/modelpath/" ]&&[ -f "${project_path}/ascendfaissdaemon" ]);then
    omfiles=$(ls modelpath/*.om 2> /dev/null)
    if [ "$omfiles" == "" ];then
        echo "[ERROR] modelpath is empty, please generate om model by scripts in tools directory first!"
        exit 1
    fi
else
    echo "[ERROR] modelpath or ascendfaissdaemon is not exist, please make it first!"
    exit 1
fi

if [ ! -d ${driver_src_path}/driver ];then
    echo "Please make sure Ascend310-driver-<version>-minios.aarch64-src.tar.gz exists and unzipped!"
    exit 1
fi

if [ ! -f ${driver_src_path}/driver/source/vendor/hisi/tools/signtool/image_pack/image_pack.py ];then
    echo "Please make sure driver tool image_pack.py exists!"
    exit 1
fi

# creat file system production directory
if [ ! -d "${project_path}/filesys_modify/" ];then
    mkdir ${project_path}/filesys_modify
else
    rm -rf ${project_path}/filesys_modify/*
fi

# add write permission
chattr -R -i ${ascend_path}/driver/device

# unzip the file system
if [ ! -f "${ascend_path}/driver/device/davinci_minibak.cpio.gz" ];then
    cp ${ascend_path}/driver/device/davinci_mini.cpio.gz ${ascend_path}/driver/device/davinci_minibak.cpio.gz
fi
cp ${ascend_path}/driver/device/davinci_minibak.cpio.gz ${project_path}/filesys_modify/davinci_mini.cpio.gz
cd ${project_path}/filesys_modify
dd if=davinci_mini.cpio.gz of=raw-davinci_mini.cpio.gz skip=16 bs=1024
gunzip raw-davinci_mini.cpio.gz
mkdir tempdir
cd tempdir
cpio -idm < ../raw-davinci_mini.cpio

if [ $? -ne 0 ];then
    echo "[ERROR] unzip the file system faild!"
    exit 1
fi

# add related files
system_path=${project_path}/filesys_modify/tempdir

if [ -d "${system_path}/usr/local/Ascend" ];then
    rm -rf ${system_path}/usr/local/Ascend
fi
mkdir ${system_path}/usr/local/Ascend
cp -axr ${acllib_minios_path}/acllib ${system_path}/usr/local/Ascend
chown -R 1000:1000 ${system_path}/usr/local/Ascend
if [ -d "${system_path}/home/HwHiAiUser/modelpath/" ];then
    rm -rf ${system_path}/home/HwHiAiUser/modelpath/
fi
if [ -f "${system_path}/home/HwHiAiUser/ascendfaissdaemon" ];then
    rm -f ${system_path}/home/HwHiAiUser/ascendfaissdaemon
fi

cp -rf ${project_path}/modelpath/ ${project_path}/ascendfaissdaemon ${system_path}/home/HwHiAiUser/
chown -R 1000:1000 ${system_path}/home/HwHiAiUser/modelpath/ ${system_path}/home/HwHiAiUser/ascendfaissdaemon

# configure environment variables and add startup script
sed -i '$a\export LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64' ${system_path}/home/HwHiAiUser/.bashrc 
#sed -i '685c #ssh_switch' ${system_path}/etc/rc.d/init.d/rcS
sed -i '/exit 0/i su - ${username} -c "./ascendfaissdaemon &"' ${system_path}/etc/rc.d/init.d/rcS

if [ $? -ne 0 ];then
    echo "[ERROR] configure environment variables and add startup script faild!"
    exit 1
fi

# compressed file system
cd ${project_path}/filesys_modify
rm davinci_mini.cpio.gz raw-davinci_mini.cpio
cd ${system_path}
find . | cpio -o -H newc | gzip > ../raw-davinci_mini.cpio.gz
cd ${project_path}/filesys_modify
python ${driver_src_path}/driver/source/vendor/hisi/tools/signtool/image_pack/image_pack.py -raw_img raw-davinci_mini.cpio.gz -out_img davinci_mini.cpio.gz -version 1.1.1.1.1 -platform hi1910
cp -r ${project_path}/filesys_modify/davinci_mini.cpio.gz /usr/local/Ascend/driver/device/

if [ $? -ne 0 ];then
    echo "[ERROR] compressed file system faild!"
    exit 1
fi

# cancel write permission
chattr -R +i ${ascend_path}/driver/device

rm -rf ${project_path}/filesys_modify

echo "Ascend faiss ctrlcpu install success! Reboot needed for installation to take effect!"