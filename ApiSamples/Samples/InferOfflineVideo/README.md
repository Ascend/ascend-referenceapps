EN|[CN](README.zh.md)
# InferOfflineVideo

## Introduction

This sample demonstrates how the InferOfflineVideo program to use the chip to perform the 'YoloV3' object detection.

Process Framework:

```
StreamPuller > VideoDecoder > ObjectDetection > PostProcess > WriteResult
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

If the whole package is not copied, ensure that the ascendbase, Common and InferOfflineVideo directories are copied to the same directory in the compilation environment. Otherwise, the compilation will fail. If the whole package is copied, ignore it.

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

## configuration

Configure the device_id, model_path and model input format in `data/config/setup.config`

Configure device_id
```bash
#chip config
device_id = 0 #use the device to run the program
```

Configure stream number and stream path

Configure stream number
```bash
#Stream number
stream_num = 4
```
Configure stream path, support rtsp video stream or local video file
Configure rtsp video stream path, the format is as follows:
```bash
stream.ch0 = rtsp://xx.xx.xx.xx:xx/xxx.264
stream.ch1 = rtsp://xx.xx.xx.xx:xx/xxx.264
stream.ch2 = rtsp://xx.xx.xx.xx:xx/xxx.264
stream.ch3 = rtsp://xx.xx.xx.xx:xx/xxx.264
```
Configure local video file path, the format is as follows:
```bash
stream.ch0 = ./videos/0_720P_25.264
stream.ch1 = ./videos/1_720P_25.264
stream.ch2 = ./videos/2_720P_25.264
stream.ch3 = ./videos/3_720P_25.264
```

Configure resize width and height
```bash
VideoDecoder.resizeWidth = 416    # must be equal to ModelInfer.modelWidth
VideoDecoder.resizeHeight = 416   # must be equal to ModelInfer.modelHeight
```

Configure model input width and height, model name, model type and model_path
```bash
#yolov3 model input width and height
ModelInfer.modelWidth = 416
ModelInfer.modelHeight = 416
ModelInfer.modelName = YoloV3
ModelInfer.modelType = 0 # 0: YoloV3 Caffe, 1: YoloV3 Tensorflow
ModelInfer.modelPath = ./data/models/yolov3/yolov3_416.om
```

Configure skipping interval
```bash
skipInterval = 3 # One frame is selected for inference every <skipInterval> frames
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

If you want to run with the compilation result on another environment, copy the dist directory and the ffmpeg dynamic libraries.

## Execution


For help
```bash
cd dist
./main -h

------------------------------help information------------------------------
-acl_setup                    ./data/config/acl.json        the config file using for AscendCL init.
-debug_level                  1                             debug level:0-debug, 1-info, 2-warn, 3-error, 4-fatal, 5-off.
-h                            help                          show helps
-help                         help                          show helps
-setup                        ./data/config/setup.config    the config file using for face recognition pipeline
```

Object detection for the video stream
```bash
cd dist
./main
```

## Constraint

Support input format: h264, h265

h264, h265 resolutions: maximum 4096 x 4096, minimum 128 x 128.



## Result
The video frame inference result is stored in result/result_x.txt (x indicates the channel ID).
If the size of result_x.txt exceeds 50 MB, the file is backed up as result_x.bak. If result_x.bak exists, the existing result_x.bak file will be deleted and then backed up.
Each result_x.txt file stores the timestamp, channel ID, frame ID, number of detected objects, coordinate frame, confidence, and label of each image frame. The format is as follows:
```bash
[Date:20000101000000 Channel:0 Frame:0] Object detected number is 7
#Obj0, box(315, 417.75, 536, 544)  confidence: 0.99707  lable: 17
#Obj1, box(87.5, 412.75, 351.5, 529.5)  confidence: 0.992188  lable: 17
#Obj2, box(1050, 441.5, 1279, 606.5)  confidence: 0.982422  lable: 17
#Obj3, box(535, 409.25, 698, 534)  confidence: 0.975586  lable: 17
#Obj4, box(957.5, 479.5, 1146, 598)  confidence: 0.905762  lable: 17
#Obj5, box(720, 445.75, 1000.5, 591.5)  confidence: 0.869141  lable: 17
#Obj6, box(0, 459.75, 84, 523)  confidence: 0.855469  lable: 17
```
