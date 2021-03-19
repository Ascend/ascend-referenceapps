中文|[英文](README.md)
# EncodeJpeg

## 介绍

本开发样例演示 `EncodeJpeg` 程序，使用芯片dvpp进行Jpeg编码。

该Sample处理流程为：
```bash
ReadYUV -> EncodeJpeg -> SaveFile
```

## 支持的产品

Atlas 800 (Model 3000), Atlas 800 (Model 3010), Atlas 300 (Model 3010)

## 支持的ACL版本

1.73.5.1.B050, 1.73.5.2.B050, 1.75.T11.0.B116, 20.1.0, 20.2.0

查询ACL版本号的方法是，在Atlas产品环境下，运行以下命令：
```bash
npu-smi info
```

## 依赖条件

代码依赖：

版本包中各个Sample都依赖ascendbase目录

编译时如果不是整包拷贝，请确保ascendbase和EncodeJpeg目录都拷贝到了编译环境的同一路径下，否则会编译失败；如果是整包拷贝，不需要关注。

设置环境变量：
*  `ASCEND_HOME`      Ascend安装的路径，一般为 `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  指定Sample程序运行时依赖的动态库查找路径

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=${ASCEND_HOME}/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

## 配置

需要在 `data/config/setup.config` 配置芯片id

配置芯片id
```bash
#chip config
device_id = 0 #use the device to run the program
```

## 编译
```bash
bash build.sh
```

## 运行
查看帮助文档
```bash
cd dist
./main -h

------------------------------help information------------------------------
-h                            help                          show helps
-help                         help                          show helps
-i                            ./data/test.yuv               Optional. Specify the input image, default: ./data/test.yuv
-width                        0                             Requested. Specify the width of input image
-height                       0                             Requested. Specify the height of input image
-format                       -1                            Requested. Specify the format of input image, {1:NV12, 2:NV21, 7:YUYV, 8:UYVY, 9:YVYU, 10:VYUY}
```

对指定YUV图片进行Jpeg编码
```bash
cd dist
./main -i ./data/images/x4096.yuv -width 4096 -height 4096 -format 1
```

## 约束
```bash
YUV图片分辨率范围
最大8192 x 8192, 最小32 x 32

YUV图片格式
YUV420SP: NV12, NV21
YUV422: YUYV, UYVY, YVYU, VYUY
```

## 结果
```bash
将编码后的数据输出到dist/result/fileName.jpg
fileName代表输入图片的文件名. 例如test.yuv -> test.jpg
```
