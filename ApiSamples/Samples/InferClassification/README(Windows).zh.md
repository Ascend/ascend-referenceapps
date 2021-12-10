中文|[英文](README(Windows).md)
# InferClassification

## 介绍

本开发样例演示 `InferClassification` 程序，使用芯片进行 `resnet50` 目标分类。

该Sample的处理流程为:

```
ReadJpeg > JpegDecode > ImageResize > ObjectClassification > Cast_Op > ArgMax_Op > WriteResult
```

## 支持的产品

HP-200-2

## 支持的ACL版本

1.76.T10.0.B100，1.76.T10.0.B110

## 依赖条件

支持单输入的Resnet50的目标分类模型，示例模型请参考[模型转换说明](data/models/README.zh.md)获取并转换（目前只支持在Linux环境转换）

单算子模型为Cast与ArgMax，示例模型请参考[模型转换说明](data/models/README.zh.md)获取并转换（目前只支持在Linux环境转换）

代码依赖：

版本包中各个Sample都依赖ascendbase目录

编译时如果不是整包拷贝，请确保ascendbase和InferClassification目录都拷贝到了编译环境的同一路径下，否则会编译失败；如果是整包拷贝，不需要关注。

## 配置

需要在 `data\config\setup.config` 配置芯片id，模型路径，模型输入格式，单算子模型路径等

修改芯片id
```bash
#chip config
device_id = 0 #use the device to run the program
```
修改模型路径
```bash
#resnet model path
model_path = .\data\models\resnet\resnet50_aipp.om
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
single_op_model = .\data\models\single_op
```

## 编译

a. 以管理员帐号登录开发环境。
b. 以管理员身份执行cmd窗口, 切换到ApiSamples\src\Samples\InferClassification，执行.\build.bat 完成编译，在dist\release目录下生成可执行文件main.exe
```bash
.\build.bat
```

如果要将编译结果拷贝到其它环境上运行，拷贝dist目录即可

## 运行

查看帮助文档
a. 以管理员帐号登录运行环境。
b. 以管理员身份执行cmd窗口，切换到dist文件夹所在目录，执行如下命令。
```bash
cd dist\release
.\main.exe -h

------------------------------help information------------------------------
-h                            help                          show helps
-help                         help                          show helps
-i                            .\data\test.jpg              Optional. Specify the input image, default: .\data\test.jpg

```

对指定jpeg图片进行推理，如指定图片为x4096.jpg
a. 以管理员帐号登录运行环境。
b. 以管理员身份执行cmd窗口，切换到dist文件夹所在目录，执行如下命令。
```bash
cd dist\release
.\main.exe -i .\data\images\x4096.jpg
```

## 约束

支持输入图片格式：Jpeg

分辨率范围: 最大4096 x 4096, 最小32 x 32.

## 结果

打印识别结果类型，并写入到result\result.txt中。如下图所示：
```bash
inference output index: 248
classname:  248: 'Eskimo dog, husky'
```

