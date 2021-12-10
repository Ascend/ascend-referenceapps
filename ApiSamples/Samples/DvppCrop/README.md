EN|[CN](README.zh.md)
# DvppCrop

## Introduction

This sample demonstrates the `DvppCrop` program, showing the way to use VPC module for image cropping. Only YUV420 format is supported. Cropped image will be saved to a YUV format file.

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

If the whole package is not copied, ensure that the ascendbase and DvppCrop directories are copied to the same directory in the compilation environment. Otherwise, the compilation will fail. If the whole package is copied, ignore it.

Set the environment variable:
*  `ASCEND_HOME`      Ascend installation path, which is generally `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  Specifies the dynamic library search path on which the sample program depends

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

## Compilation

```bash
bash build.sh
```

## Execution

View the help
```bash
cd dist
./main -h

------------------------------help information------------------------------
--crop                        0,0,16,2                      Set crop area, such as 0,0,512,416; 0 is left side begin to crop, 0 is top side begin to crop, 512 is width, 416 is height
--format                      1                             Set pixelformat: 0:Gray 1:NV12 2:NV21
--input                       ./data/test.yuv               Specify the input image
--output                      ./data/cropped.yuv             Set the path to save the cropped image
--resolution                  1920,1080                     Set the resolution of input image, such as: 1920,1080; 1920 is width, 1080 is height
-h                            help                          show helps
-help                         help                          show helps
```
Infer the specified YUV images, for example crop a part from an input image, the input image name is ./data/test.yuv, the format is NV12, and the size is 1920x1080, start coordinate of the target image is 0,0, resolution to crop is 512x416 and save cropped image to ./data/cropped.yuv, then example command line could be as following:
```bash
cd dist
./main --input ./data/test.yuv --output ./data/cropped.yuv --resolution 1920,1080 --crop 0,0,512,416 --format 1
```

## Constraint

```
Resolutions range: maximum 4096 x 4096, minimum 32 x 6.
```

## Result

```
The cropped image is saved to the path specified by --output. Use the YUV viewing software to check the saved file to confirm whether the result is correct.
```
