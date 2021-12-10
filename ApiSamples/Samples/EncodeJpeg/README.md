EN|[CN](README.zh.md)
# EncodeJpeg

## Introduction

This sample demonstrates how the EncodeJpeg program to use dvpp to Encode a YUV file into a Jpeg image.

Process Framework
```bash
ReadYUV -> EncodeJpeg -> SaveFile
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

If the whole package is not copied, ensure that the ascendbase and EncodeJpeg directories are copied to the same directory in the compilation environment. Otherwise, the compilation will fail. If the whole package is copied, ignore it.

Set the environment variable:
*  `ASCEND_HOME`      Ascend installation path, which is generally `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  Specifies the dynamic library search path on which the sample program depends

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

## Configuration

configure the device_id in `data/config/setup.config`

set device id
```bash
#chip config
device_id = 0 #use the device to run the program
```

## Compilation
```bash
bash build.sh
```

## Execution
view help document
```bash
cd dist
./main -h

------------------------------help information------------------------------
-h                            help                          show helps
-help                         help                          show helps
-i                            ./data/test.yuv               Optional. Specify the input image, default: ./data/test.yuv
-width                        0                             Requested. Specify the width of input image
-height                       0                             Requested. Specify the height of input image
-format                       -1                            Requested. Specify the format of input image, {1:NV12, 2:NV21, 7:YUYV, 8:UYVY, 9:YVYU, 10:VYUY}
```

Encode a YUV file into a Jpeg image and save the image.
```bash
cd dist
./main -i ./data/images/x4096.yuv -width 4096 -height 4096 -format 1
```

## Constraint
```
YUV resolutions:
maximum 4096 x 4096, minimum 32 x 32

YUV format:
YUV420SP: NV12, NV21
YUV422: YUYV, UYVY, YVYU, VYUY
```

## Result
```bash
Encoded data is output to dist/result/fileName.jpg
fileName indicates the name of the input image. such as test.yuv -> test.jpg
```
