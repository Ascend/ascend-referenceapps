中文|[英文](README.md)
# DvppCrop

## 介绍

本开发样例演示 `DvppCrop` 程序，示例如何使用 Dvpp 的 VPC 模块进行抠图，支持从一张输入 YUV420 格式的图片中扣取其中一部分，保存到 YUV 格式的文件。

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

编译时如果不是整包拷贝，请确保ascendbase和DvppCrop目录都拷贝到了编译环境的同一路径下，否则会编译失败；如果是整包拷贝，不需要关注。

设置环境变量：
*  `ASCEND_HOME`      Ascend安装的路径，一般为 `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  指定Sample程序运行时依赖的动态库查找路径

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=${ASCEND_HOME}/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
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
--crop                        0,0,16,2                      Set crop area, such as 0,0,512,416; 0 is left side begin to crop, 0 is top side begin to crop, 512 is width, 416 is height
--format                      1                             Set pixelformat: 0:Gray 1:NV12 2:NV21
--input                       ./data/test.yuv               Specify the input image
--output                      ./data/cropped.yuv             Set the path to save the cropped image
--resolution                  1920,1080                     Set the resolution of input image, such as: 1920,1080; 1920 is width, 1080 is height
-h                            help                          show helps
-help                         help                          show helps
```
对指定 YUV 图片进行推理，如从一张源图片扣取一部分，源图片文件名为 `./data/test.yuv`，文件格式为 NV12，源图片大小为 1920x1080，目标图片起始坐标为 0,0，分辨率为 512x416，目标图片保存到 `./data/cropped.yuv`，对应的运行示例如下：
```bash
cd dist
./main --input ./data/test.yuv --output ./data/cropped.yuv --resolution 1920,1080 --crop 0,0,512,416 --format 1
```

## 约束

```
源图片和目标图片分辨率范围: 最大 4096 x 4096, 最小 32 x 6.
```

## 结果

```
扣取的图片保存到 `--output` 指定的路径中。使用 YUV 查看软件查看保存的文件确认结果是否正确。
```
