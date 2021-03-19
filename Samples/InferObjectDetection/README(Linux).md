EN|[CN](README(Linux).zh.md)
# InferObjectDetection

## Introduction

This sample demonstrates how the InferObjectDetection program to use the chip to perform the 'YoloV3' object detection.

Process Framework:

```
ReadJpeg > JpegDecode > ImageResize > ObjectDetection > YoloV3PostProcess > WriteResult
```

## Supported Products

Atlas 800 (Model 3000), Atlas 800 (Model 3010), Atlas 300 (Model 3010), Atlas 500 (Model 3010), Atlas 300I (Model 6000)

## Supported ACL Version

1.73.5.1.B050, 1.73.5.2.B050, 1.75.T11.0.B116, 20.1.0, 20.2.0

Run the following command to check the version in the environment where the Atlas product is installed:
```bash
npu-smi info
```

## Dependency

The input model is YoloV3. Please refer to [model transformation instructions](data/models/README.md) to transform the model.


Code dependency:

This sample depends on the ascendbase and Common directory.

If the whole package is not copied, ensure that the ascendbase, Common and InferObjectDetection directories are copied to the same directory in the compilation environment. Otherwise, the compilation will fail. If the whole package is copied, ignore it.

Set the environment variable:
*  `ASCEND_HOME`      Ascend installation path, which is generally `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  Specifies the dynamic library search path on which the sample program depends

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
#If the model needs to be decrypted, run the following command
export LD_LIBRARY_PATH=/usr/local/dcmi:$LD_LIBRARY_PATH
```


## Configuration

Configure the device_id, model_path and model input format in `data/config/setup.config`

Configure device_id
```bash
#chip config
device_id = 0 #use the device to run the program
```
Configure model_path
```bash
#yolov3 model path
model_path = ./data/models/yolo/yolov3.om
```
Configure model input format
```bash
#yolov3 model input width and height
model_width = 416
model_height = 416
```
Configure model encrypt information, this configuration is valid only on A500 now
```bash
model_encrypted = false #Declares whether the model is encrypted
#NPU ID
card_id = 0 #Only need to be set when the model is encrypted
```


## Compilation

Compile Atlas 800 (Model 3000), Atlas 800 (Model 3010), Atlas 300 (Model 3010) programs
```bash
bash build.sh
```

Compile Atlas 500 (Model 3010) program
Run the following command on the ARM server
```bash
bash build.sh
```

Run the following command on the X86 server
```bash
bash build.sh A500
```

If you want to run with the compilation result on another environment, copy the dist directory

## Execution


For help
```bash
cd dist
./main -h

------------------------------help information------------------------------
-h                            help                          show helps
-help                         help                          show helps
-i                            ./data/test.jpg               Optional. Specify the input image, default: ./data/test.jpg
-t                            0                             Model type. 0: YoloV3 Caffe, 1: YoloV3 Tensorflow
```

Object detection for the Jpeg images and Caffe model
```bash
cd dist
./main -i ./data/images/x4096.jpg -t 0
```

## Constraint

Support input format: Jpeg

Resolutions: maximum 4096 x 4096, minimum 32 x 32.


## Result

Print the result on the terminal: the number of objects, the position, confidence and label, in the meantime, save the result to result/result_xxx.txt(xxx is timestamp):
```bash
[Info ] Object Nums is 2
[Info ] #0, bobox(18, 0, 636, 499)   confidence: 1 label: 0
[Info ] #1, bobox(142, 153, 502, 499)   confidence: 0.895996 label: 16
```

## Model replacement
Please refer to [model transformation instructions](data/models/README.md) to modify the new model's configuration.
