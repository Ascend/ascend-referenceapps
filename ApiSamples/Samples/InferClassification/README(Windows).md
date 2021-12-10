EN|[CN](README(Windows).zh.md)
# InferClassification

## Introduction

This sample demonstrates how the InferClassification program to use the chip to perform the 'resnet50' target classification.
Process Framework

```
ReadJpeg > JpegDecode > ImageResize > ObjectClassification > Cast_Op > ArgMax_Op > WriteResult
```
## Supported Products

HP-200-2

## Supported ACL Version

1.76.T10.0.B100, 1.76.T10.0.B110

## Dependency

The input model is Resnet50. Please refer to [model transformation instructions](data/models/README.md) to transform the model (The model only can be converted on Linux currently).

The single op is Cast and ArgMax. Please refer to [model transformation instructions](data/models/README.md) to transform the single op model (The model only can be converted on Linux currently).

Code dependency:

Each sample in the version package depends on the ascendbase directory.

If the whole package is not copied, ensure that the ascendbase and InferClassification directories are copied to the same directory in the compilation environment. Otherwise, the compilation will fail. If the whole package is copied, ignore it.

## Configuration

Configure the device_id, model_path and single_op_model in `data\config\setup.config`

Configure device_id
```bash
#chip config
device_id = 0 #use the device to run the program
```

Configure model_path
```bash
#resnet model path
model_path = .\data\models\resnet\resnet50_aipp.om
```

Configure model input format
```bash
#resnet model input width and height
model_width = 224
model_height = 224
```
Configure single op model path
```bash
#single op model path
single_op_model = .\data\models\single_op
```

## Compilation

a. Log in the development environment as the administrator.
b. Open the cmd window as the administrator, switch to the ApiSamples\src\Samples\InferClassification, run the following command to complete the compilation, main.exe will be generated in dist\release directory
```bash
.\build.bat
```

If you want to run with the compilation result on another environment, copy the dist directory

## Execution

For help
a. Log in the development environment as the administrator.
b. Open the cmd window as the administrator, switch to the directory where the dist folder is located, run the following command

```bash
cd dist\release
.\main.exe -h

------------------------------help information------------------------------
-h                            help                          show helps
-help                         help                          show helps
-i                            .\data\test.jpg              Optional. Specify the input image, default: .\data\test.jpg

```

Classify the Jpeg images
a. Log in the development environment as the administrator.
b. Open the cmd window as the administrator, switch to the directory where the dist folder is located, run the following command
```bash
cd dist\release
.\main.exe -i .\data\images\x4096.jpg
```

## Constraint
Support input format: Jpeg

Resolutions: maximum 4096 x 4096, minimum 32 x 32.

## Result

The classification result is generated in the out folder result\result.txt.

```bash
inference output index: 248
classname:  248: 'Eskimo dog, husky'
```
