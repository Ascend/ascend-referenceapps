中文|[英文](README.md)
# DecodeVideo

## 介绍

本开发样例演示 `DecodeVideo` 程序，使用芯片进行视频文件解码。

该Sample的处理流程为:

```
ReadFile > VdecDecode > WriteResult
```

## 支持的产品

Atlas 800 (Model 3000), Atlas 800 (Model 3010), Atlas 300 (Model 3010)

## 支持的ACL版本

1.73.5.1.B050, 1.73.5.2.B050, 1.75.T11.0.B116, 20.1.0, 20.2.0

查询ACL版本号的方法是，在Atlas产品环境下，运行以下命令：
```bash
npu-smi info
```


## 依赖条件

代码依赖：

版本包中各个Sample都依赖ascendbase目录

编译时如果不是整包拷贝，请确保ascendbase和DecodeVideo目录都拷贝到了编译环境的同一路径下，否则会编译失败；如果是整包拷贝，不需要关注。

支持输入H264、H265视频文件，视频解码并保存解码后YUV图片。

设置环境变量：
*  `ASCEND_HOME`      Ascend安装的路径，一般为 `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  指定Sample程序运行时依赖的动态库查找路径

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=${ASCEND_HOME}/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

#### FFmpeg 4.2

源码下载地址：https://github.com/FFmpeg/FFmpeg/releases

FFmpeg源码编译和安装配置的方法，可以参考Ascend开发者论坛：https://bbs.huaweicloud.com/forum/thread-25834-1-1.html

如需交叉编译时，则进入到ffmpeg源码包目录下，执行以下命令完成交叉编译，请在--prefix选项自行指定安装路径

```bash
export PATH=${PATH}:${ASCEND_HOME}/ascend-toolkit/latest/toolkit/toolchain/hcc/bin
./configure --prefix=/your/specify/path --target-os=linux --arch=aarch64 --enable-cross-compile --cross-prefix=aarch64-target-linux-gnu- --enable-shared --disable-doc --disable-vaapi --disable-libxcb --disable-libxcb-shm --disable-libxcb-xfixes --disable-libxcb-shape --disable-asm
make -j
make install -j
```

配置FFmpeg4.2库，设置FFmpeg环境变量，如FFmpeg安装路径为/usr/local/ffmpeg。如使用交叉编译，则指定为交叉编译时ffmpeg安装路径
```bash
export FFMPEG_PATH=/usr/local/ffmpeg
export LD_LIBRARY_PATH=$FFMPEG_PATH/lib:$LD_LIBRARY_PATH
```

## 配置

需要在 `data/config/setup.config` 配置芯片id

```bash
#chip config
device_id = 0 #use the device to run the program
```

## 编译

```bash
bash build.sh
```

## 运行
查看帮助文档
```bash
cd dist
./main -h

------------------------------help information------------------------------
-h                            help                          show helps
-help                         help                          show helps
-i                            ./data/test.h264             Optional. Specify the input video, default: ./data/test.h264

```

对指定h264视频文件进行解码，如指定视频为test.h264
```bash
cd dist
./main -i ./data/test.h264
```

## 约束

支持输入视频格式：h264, h265

h264 分辨率范围: 最大4096 x 4096, 最小128 x 128.


## 结果

生成解码后的图片，写入到dist/result文件夹中。

