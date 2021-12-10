EN|[CN](README.zh.md)
# DecodeImage

## Introduction

This sample demonstrates how the DecodeImage program to use the chip.

Process Framework:
```
ReadJpeg -> JpegDecode -> ImageResize -> WriteResult
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

If the whole package is not copied, ensure that the ascendbase and DecodeImage directories are copied to the same directory in the compilation environment. Otherwise, the compilation will fail. If the whole package is copied, ignore it.

Set the environment variable:
*  `ASCEND_HOME`      Ascend installation path, which is generally `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  Specifies the dynamic library search path on which the sample program depends

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

## Configuration

configure the device_id, resize_width and resize_height in `data/config/setup.config`

set device id
```bash
#chip config
device_id = 0 #use the device to run the program
```
set resize_width and resize_height
```bash
#resize width and height
resize_width = 224
resize_height = 224
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
-i                            ./data/test.jpg              Optional. Specify the input image, default: ./data/test.jpg
```

Classify the Jpeg images
```bash
cd dist
./main -i ./data/test.jpg
```

## Constraint
```
Only supports Jpeg format
Jpeg resolutions: maximum 4096 x 4096, minimum 32 x 32.
```

## Result
```
Decoded data is output to dist/result/result_xxx.yuv, where xxx is the timestamp
```
