EN|[CN](README(Windows).zh.md)
# InferObjectDetection

## Introduction

This sample demonstrates how the InferObjectDetection program to use the chip to perform the 'YoloV3' object detection.

Process Framework:

```
ReadJpeg > JpegDecode > ImageResize > ObjectDetection > YoloV3PostProcess > WriteResult
```

## Supported Products

HP-200-2

## Supported ACL Version

1.76.T10.0.B100, 1.76.T10.0.B110

Run the following command to check the version in the environment where the Atlas product is installed:
```bash
npu-smi info
```

## Dependency

The input model is YoloV3. Please refer to [model transformation instructions](data/models/README.md) to transform the model(The model only can be converted on Linux currently).


Code dependency:

This sample depends on the ascendbase and Common directory.

If the whole package is not copied, ensure that the ascendbase, Common and InferObjectDetection directories are copied to the same directory in the compilation environment. Otherwise, the compilation will fail. If the whole package is copied, ignore it.

## Configuration

Configure the device_id, model_path and model input format in `data\config\setup.config`

Configure device_id
```bash
#chip config
device_id = 0 #use the device to run the program
```
Configure model_path
```bash
#yolov3 model path
model_path = .\data\models\yolov3\yolov3.om
```
Configure model input format
```bash
#yolov3 model input width and height
model_width = 416
model_height = 416
```


## Compilation

a. Log in the development environment as the administrator.

b. Open the cmd window as the administrator, switch to the ApiSamples\src\Samples\InferObjectDetection, run the following command to complete the compilation

```bash
.\build.bat
```

After the compilation is complete, main.exe will be generated in dist\release directory.

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
-i                            .\data\test.jpg               Optional. Specify the input image, default: .\data\test.jpg
-t                            0                             Model type. 0: YoloV3 Caffe, 1: YoloV3 Tensorflow
```

Object detection for the Jpeg images and Caffe model

a. Log in the development environment as the administrator.

b. Open the cmd window as the administrator, switch to the directory where the dist folder is located, run the following command

```bash
cd dist\release
.\main.exe -i .\data\images\x4096.jpg -t 0
```

## Constraint

Support input format: Jpeg

Resolutions: maximum 4096 x 4096, minimum 32 x 32.


## Result

Print the result on the terminal: the number of objects, the position, confidence and label, in the meantime, save the result to result\result_xxx.txt(xxx is timestamp):
```bash
[Info ] Object Nums is 2
[Info ] #0, bobox(18, 0, 636, 499)   confidence: 1 label: 0
[Info ] #1, bobox(142, 153, 502, 499)   confidence: 0.895996 label: 16
```

## Model replacement
Please refer to [model transformation instructions](data/models/README.md) to modify the new model's configuration.
