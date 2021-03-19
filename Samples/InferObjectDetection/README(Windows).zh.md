中文|[英文](README(Windows).md)
# InferObjectDetection

## 介绍

本开发样例演示 `InferObjectDetection` 程序，使用芯片进行 `YoloV3` 目标识别。

该Sample的处理流程为：

```
ReadJpeg > JpegDecode > ImageResize > ObjectDetection > YoloV3PostProcess > WriteResult
```

## 支持的产品

HP-200-2

## 支持的ACL版本

1.76.T10.0.B100, 1.76.T10.0.B110

查询ACL版本号的方法是，在Atlas产品环境下，运行以下命令：
```bash
npu-smi info
```

## 依赖条件

支持YoloV3目标检测模型，示例模型请参考[模型转换说明](data/models/README.zh.md)获取并转换（目前仅支持在Linux系统上转换模型）

代码依赖：

本Sample依赖ascendbase和Common目录

编译时如果不是整包拷贝，请确保ascendbase，Common和InferObjectDetection目录都拷贝到了编译环境的同一路径下，否则会编译失败；如果是整包拷贝，不需要关注。

## 配置

需要在 `data\config\setup.config` 配置芯片id，模型路径，模型输入格式等

修改芯片id
```bash
#chip config
device_id = 0 #use the device to run the program
```
修改模型路径
```bash
#yolov3 model path
model_path = .\data\models\yolo\yolov3.om
```
修改模型输入格式
```bash
#yolov3 model input width and height
model_width = 416
model_height = 416
```

## 编译

a. 以管理员帐号登录开发环境。

b. 以管理员身份执行cmd窗口，切换到ApiSamples\src\Samples\InferObjectDetection，执行如下命令完成编译

```bash
.\build.bat
```

编译完成后，会在dist\release目录下生成可执行文件main.exe

如果需要将编译结果拷贝到其它环境上运行，拷贝dist目录即可

## 运行

查看帮助文档

a. 以管理员帐号登录运行环境。

b. 以管理员身份执行cmd窗口，切换到dist文件夹所在目录，执行如下命令

```bash
cd dist\release
.\main.exe -h

------------------------------help information------------------------------
-h                            help                          show helps
-help                         help                          show helps
-i                            .\data\test.jpg               Optional. Specify the input image, default: .\data\test.jpg
-t                            0                             Model type. 0: YoloV3 Caffe, 1: YoloV3 Tensorflow
```

对指定jpeg图片进行推理，如指定图片为x4096.jpg，模型为Caffe转换时

a. 以管理员帐号登录运行环境。

b. 以管理员身份执行cmd窗口，切换到dist文件夹所在目录，执行如下命令

```bash
cd dist\release
.\main.exe -i .\data\images\x4096.jpg -t 0
```

## 约束

支持输入图片格式：Jpeg

分辨率范围: 最大4096 x 4096, 最小32 x 32.

## 结果

在终端打印识别结果，输出模型检测目标数目，每个目标的坐标框，置信度以及标签，同时把结果保存至result\result_xxx.txt中(xxx为运行时间戳)：
```bash
[Info ] Object Nums is 2
[Info ] #0, bobox(18, 0, 636, 499)   confidence: 1 label: 0
[Info ] #1, bobox(142, 153, 502, 499)   confidence: 0.895996 label: 16
```

## 模型更换

请参考[模型转换说明](data/models/README.zh.md)修改相关配置信息
