中文|[英文](README(Linux).md)
# InferObjectDetection

## 介绍

本开发样例演示 `InferObjectDetection` 程序，使用芯片进行 `YoloV3` 目标识别。

该Sample的处理流程为：

```
ReadJpeg > JpegDecode > ImageResize > ObjectDetection > YoloV3PostProcess > WriteResult
```

## 支持的产品

Atlas 800 (Model 3000), Atlas 800 (Model 3010), Atlas 300 (Model 3010), Atlas 500 (Model 3010), Atlas 300I (Model 6000)

## 支持的ACL版本

1.73.5.1.B050, 1.73.5.2.B050, 1.75.T11.0.B116, 20.1.0, 20.2.0

查询ACL版本号的方法是，在Atlas产品环境下，运行以下命令：
```bash
npu-smi info
```

## 依赖条件

支持YoloV3目标检测模型，示例模型请参考[模型转换说明](data/models/README.zh.md)获取并转换

代码依赖：

本Sample依赖ascendbase和Common目录

编译时如果不是整包拷贝，请确保ascendbase，Common和InferObjectDetection目录都拷贝到了编译环境的同一路径下，否则会编译失败；如果是整包拷贝，不需要关注。

设置环境变量：
*  `ASCEND_HOME`      Ascend安装的路径，一般为 `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  指定Sample程序运行时依赖的动态库查找路径

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=${ASCEND_HOME}/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
#如果使用的模型需要解密，还需执行如下命令
export LD_LIBRARY_PATH=/usr/local/dcmi:$LD_LIBRARY_PATH
```

## 配置

需要在 `data/config/setup.config` 配置芯片id，模型路径，模型输入格式等

修改芯片id
```bash
#chip config
device_id = 0 #use the device to run the program
```
修改模型路径
```bash
#yolov3 model path
model_path = ./data/models/yolov3/yolov3.om
```
修改模型输入格式
```bash
#yolov3 model input width and height
model_width = 416
model_height = 416
```
配置模型加密相关信息，目前该配置只在A500运行时有效
```bash
model_encrypted = false #Declares whether the model is encrypted
#NPU ID
card_id = 0 #Only need to be set when the model is encrypted
```

## 编译

编译Atlas 800 (Model 3000)，Atlas 800 (Model 3010)，Atlas 300 (Model 3010)程序
```bash
bash build.sh
```

编译Atlas 500 (Model 3010)程序
在ARM服务器上编译A500程序时，执行如下命令
```bash
bash build.sh
```

在X86服务器上交叉编译A500程序时，执行如下命令
```bash
bash build.sh A500
```

如果需要将编译结果拷贝到其它环境上运行，拷贝dist目录即可

## 运行

查看帮助文档
```bash
cd dist
./main -h

------------------------------help information------------------------------
-h                            help                          show helps
-help                         help                          show helps
-i                            ./data/test.jpg               Optional. Specify the input image, default: ./data/test.jpg
-t                            0                             Model type. 0: YoloV3 Caffe, 1: YoloV3 Tensorflow
```

对指定jpeg图片进行推理，如指定图片为x4096.jpg，模型为Caffe转换时
```bash
cd dist
./main -i ./data/images/x4096.jpg -t 0
```

## 约束

支持输入图片格式：Jpeg

分辨率范围: 最大4096 x 4096, 最小32 x 32.

## 结果

在终端打印识别结果，输出模型检测目标数目，每个目标的坐标框，置信度以及标签，同时把结果保存至result/result_xxx.txt中(xxx为运行时间戳)：
```bash
[Info ] Object Nums is 2
[Info ] #0, bobox(18, 0, 636, 499)   confidence: 1 label: 0
[Info ] #1, bobox(142, 153, 502, 499)   confidence: 0.895996 label: 16
```

## 模型更换

请参考[模型转换说明](data/models/README.zh.md)修改相关配置信息
