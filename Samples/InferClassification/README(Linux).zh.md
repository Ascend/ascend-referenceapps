中文|[英文](README(Linux).md)
# InferClassification

## 介绍

本开发样例演示 `InferClassification` 程序，使用芯片进行 `resnet50` 目标分类。

该Sample的处理流程为:

```
ReadJpeg > JpegDecode > ImageResize > ObjectClassification > Cast_Op > ArgMax_Op > WriteResult
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

支持单输入的Resnet50的目标分类模型，示例模型请参考[模型转换说明](data/models/README.zh.md)获取并转换

单算子模型为Cast与ArgMax，示例模型请参考[模型转换说明](data/models/README.zh.md)获取并转换

代码依赖：

版本包中各个Sample都依赖ascendbase目录

编译时如果不是整包拷贝，请确保ascendbase和InferClassification目录都拷贝到了编译环境的同一路径下，否则会编译失败；如果是整包拷贝，不需要关注。

设置环境变量：
*  `ASCEND_HOME`      Ascend安装的路径，一般为 `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  指定Sample程序运行时依赖的动态库查找路径

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=${ASCEND_HOME}/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

## 配置

需要在 `data/config/setup.config` 配置芯片id，模型路径，模型输入格式，单算子模型路径等

修改芯片id
```bash
#chip config
device_id = 0 #use the device to run the program
```
修改模型路径
```bash
#resnet model path
model_path = ./data/models/resnet/resnet50_aipp.om
```
修改模型输入格式
```bash
#resnet model input width and height
model_width = 224
model_height = 224
```
修改单算子模型路径
```bash
#single op model path
single_op_model = ./data/models/single_op
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

如果要将编译结果拷贝到其它环境上运行，拷贝dist目录即可

## 运行

查看帮助文档
```bash
cd dist
./main -h

------------------------------help information------------------------------
-h                            help                          show helps
-help                         help                          show helps
-i                            ./data/test.jpg              Optional. Specify the input image, default: ./data/test.jpg

```

对指定jpeg图片进行推理，如指定图片为x4096.jpg
```bash
cd dist
./main -i ./data/images/x4096.jpg
```

## 约束

支持输入图片格式：Jpeg

分辨率范围: 最大4096 x 4096, 最小32 x 32.

## 结果

打印识别结果类型，并写入到result/result.txt中。如下图所示：
```bash
inference output index: 248
classname:  248: 'Eskimo dog, husky'
```

