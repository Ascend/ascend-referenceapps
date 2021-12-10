EN|[CN](README.zh.md)
# DecodeVideo

## Introduction

This sample demonstrates the DecodeVideo program, describing the way of video decoding using the chip.

Process Framework

```
ReadFile > VdecDecode > WriteResult
```

## Supported Products

Atlas 800 (Model 3000), Atlas 800 (Model 3010), Atlas 300 (Model 3010), Atlas 300I (Model 6000)

## Supported ACL Version

1.73.5.1.B050, 1.73.5.2.B050, 1.75.T11.0.B116, 20.1.0, 20.2.0

Run the following command to check the version in the environment where the Atlas product is installed:
```bash
npu-smi info
```

## Dependency

Code dependency:

Each sample in the version package depends on the ascendbase directory.

If the whole package is not copied, ensure that the ascendbase and DecodeVideo directories are copied to the same directory in the compilation environment. Otherwise, the compilation will fail. If the whole package is copied, ignore it.

The input file is h264 or h265. Decode the input file and save result as YUV image files.

Set the environment variable:
*  `ASCEND_HOME`      Ascend installation path, which is generally `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  Specifies the dynamic library search path on which the sample program depends

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

#### FFmpeg 4.2

Source download address: https://github.com/FFmpeg/FFmpeg/releases

To compile and install ffmpeg with source code, you can refer to Ascend developers BBS: https://bbs.huaweicloud.com/blogs/140860

If cross compilation is required, go to the directory of the ffmpeg source code package and run the following command to complete cross compilation. Specify the installation path in the --prefix option.

```bash
export PATH=${PATH}:${ASCEND_HOME}/ascend-toolkit/latest/toolkit/toolchain/hcc/bin
./configure --prefix=/your/specify/path --target-os=linux --arch=aarch64 --enable-cross-compile --cross-prefix=aarch64-target-linux-gnu- --enable-shared --disable-doc --disable-vaapi --disable-libxcb --disable-libxcb-shm --disable-libxcb-xfixes --disable-libxcb-shape --disable-asm
make -j
make install -j
```

Config FFMPEG4.2 library, set FFMPEG environment variable, for example FFMPEG install path is "/usr/local/ffmpeg". If cross compilation is used, set this parameter to the ffmpeg installation path during cross compilation.
```bash
export FFMPEG_PATH=/usr/local/ffmpeg
export LD_LIBRARY_PATH=$FFMPEG_PATH/lib:$LD_LIBRARY_PATH
```

## Configuration

Configure the device_id in `data/config/setup.config`

```bash
#chip config
device_id = 0 #use the device to run the program
```

## Compilation

```bash
bash build.sh
```

## Execution

```bash
cd dist
./main -h

------------------------------help information------------------------------
-h                            help                          show helps
-help                         help                          show helps
-i                            ./data/test.h264             Optional. Specify the input image, default: ./data/test.h264

```

Decode the specified h264 video file
```bash
cd dist
./main -i ./data/test.h264
```

## Constraint

Support input format: h264, h265

h264, h265 resolutions: maximum 4096 x 4096, minimum 128 x 128.


## Result

Generate decoded YUV pictures. The corresponding result files are generated in the folder "dist/result".

