[TOC]

# SceneRecognition

# 1 简介

## 1.1 背景介绍

本参考设计的目的主要包括以下几个方面：

1.为金融、安防、互联网等客户提供基于Atlas做OCR的参考样例，验证可行性；

2.作为客户利用ACL开发OCR应用的编程样例（下称Demo），开放源代码给客户参考，降低客户准入门槛；

3.提供新架构下系统优化案例，打造超高性能原型系统，为客户提供选择Atlas的理由。

本Demo模型选择是面向直排文本，不考虑弯曲文本的情况，并选择JPEG作为输入图像格式，实现识别输入图片中的文字功能。本文档提供对OCR Demo实现方案的说明。

## 1.2 支持的产品

本系统采用Atlas300-3010作为实验验证的硬件平台，具体产品实物图和硬件参数请参见《Atlas 300 AI加速卡 用户指南（型号 3010）》。由于采用的硬件平台为含有Atlas 300的Atlas 800 AI服务器，而服务器一般需要通过网络访问，因此需要通过笔记本或PC等客户端访问服务器。

### 支持的产品形态

Atlas 800 (Model 3000), Atlas 800 (Model 3010), Atlas 300 (Model 3010), Atlas 300I (Model 6000)

### 支持的ACL版本

1.75.T11.0.B116, 1.75.T15.0.B150, 20.1.0, 20.2.0

查询ACL版本号的方法是，在Atlas产品环境下，运行以下命令：
```bash
npu-smi info
```

## 1.3 软件方案介绍

软件方案包含OCR的三个必要环节：文本检测、图像校正、字符识别。主要包含文本检测和文本识别两个模型。
为了实现高性能，采用了流水线处理方案，流水线各模块具体功能描述请参考 表1.1 系统方案各子模块功能描述。

表1.1 系统方案各子模块功能：

| 序号 | 子系统            | 功能描述                                                     |
| ---- | ----------------- | ------------------------------------------------------------ |
| 1    | 读取图片          | 从用户指定的文件夹或者文件读取图片（仅支持JPEG）             |
| 2    | 图片预处理        | 为了降低传输带宽，一般需要将视频或图像进行编码压缩后通过网络传输，后面的处理需要针对RGB格式数据，所以都存在解码的需求，为了系统整体性能，一般通过硬件（DVPP）进行解码，并使用DVPP将解码后的图片缩放至文本检测模型输入要求的分辨率。 |
| 3    | 文本检测          | 文本检测是本系统的核心模块，本系统选用的是开源模型AdvancedEAST，可检测出图片中文本所在区域的4个顶点坐标，一张图片一般都会识别出多个文本框，将每个文本框分别进行透视变换和文本识别。 |
| 4    | 透视变换          | 考虑到输入图片拍摄角度的问题，图片中的文字可能存在形变问题，需要将检测到的文本框抠出来再进行透视变换，获得对齐的文字图像，然后使用DVPP将对齐后的文字图像缩放至文本识别模型输入要求的分辨率。  |
| 5    | 文本识别          | 文本识别是本系统的核心模块，本系统选用的是开源模型chinese-ocr，可检测出图片中的文字，并输出对应的文本信息。 |
| 6    | 结果处理          | 将同一张图片的多个文本框识别结果进行合并，并保存到结果文件，显示图片名称，图片id，以及文本框id，文本框坐标以及文本框中的文字信息等等。 |

### 代码主要目录介绍

本Demo工程名为SceneTextRecognition，根目录下src为源码目录，dist为目录运行，现将dist与src的子目录介绍如下：
```
.
├── AscendBase
│   └── src
│       └── Base
│           ├── BlockingQueue
│           ├── CommandParser
│           ├── CommonDataType
│           ├── ConfigParser
│           ├── DvppCommon
│           ├── ErrorCode
│           ├── FileManager
│           ├── Framework
│           ├── Log
│           ├── PointerDeleter
│           ├── ResourceManager
│           └──Statistic
├── Common
│   ├── CommandLine.h
│   ├── CommonType.h
├── Data
│   ├── Config                   // 配置文件
│   └── Models
│           ├── TextDetection    // 文本检测模型
│           ├── TextRecognition  // 文本模型
├── Modules
│   ├── DetectPost               // 文本检测后处理模块
│   ├── ImagePreprocess          // 图片预处理模块
│   ├── ImageReader              // 图片读取模块
│   ├── ResultProcess            // 结果合并处理模块
│   ├── TextDetection            // 文本检测模型推理模块
│   ├── TextRecognition          // 文本识别模型推理模块
│   └── WarpPerspective          // 图片透视变换模块
├── build.sh
└── CMakeLists.txt
```


# 2 环境搭建

### 2.1 第三方软件依赖说明

**表2-1** 第三方软件依赖说明

| 依赖软件      | 版本   | 下载地址                                                                    | 说明                                         |
| ------------- | ------ | --------------------------------------------------------------------------- | -------------------------------------------- |
| opencv        | 4.2.0  | [Link](https://github.com/opencv/opencv/releases)                           | OpenCV的基本组件，用于图像的基本处理         |

在200RC上执行安装时，如出现编译报错内存不足，请去除编译时的-j选项

### 2.2 opencv-4.2.0安装

- **步骤 1**   以**root**用户登录服务器操作后台。

- **步骤 2**   将"opencv-4.2.0.tar.gz"代码包下载至任意目录，如“/home/HwHiAiUser/opencv-4.2.0”，运行如下命令解压。

  ```
  tar -xzvf opencv-4.2.0.tar.gz
  ```

- **步骤 3**  执行如下命令，进入解压目录，创建构建与编译目录，并进入。

  如果编译环境为x86，执行如下命令编译opencv
  ```
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local/opencv \
  -DBUILD_TESTS=OFF -DWITH_WEBP=OFF -DWITH_LAPACK=OFF -DBUILD_opencv_world=ON ..
  make -j
  make install
  ```

  如果编译环境为arm，执行如下命令编译opencv
  ```
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_NEON=OFF -DCMAKE_INSTALL_PREFIX=/usr/local/opencv \
  -DCMAKE_CXX_FLAGS="-march=armv8-a" \
  -DBUILD_TESTS=OFF -DWITH_WEBP=OFF -DWITH_LAPACK=OFF -DBUILD_opencv_world=ON ..
  make -j
  make install
  ```

  根据提示确认是否安装成功。

上述第三方软件默认安装到/usr/local/下面，全部安装完成后，请设置环境变量

```bash
export LD_LIBRARY_PATH=/usr/local/opencv/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/opencv/lib:$LD_LIBRARY_PATH
```

# 3 编译运行

## 3.1 设置环境变量

在编译运行Demo前，需设置环境变量：

*  `ASCEND_HOME`      Ascend安装的路径，一般为 `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  指定程序运行时所依赖的动态库查找路径

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

## 3.2 编译

- **步骤 1**   以**root**用户登录服务器操作后台，并设置环境变量。

- **步骤 2**   将Demo代码下载至任意目录，如“/home/HwHiAiUser/SceneTextRecognition”。

- **步骤 3**   执行如下命令，构建代码。

  ```
   cd /home/HwHiAiUser/SceneTextRecognition/src
   bash build.sh
  ```

  *提示：编译完成后会生成可执行文件“ocr”，存放在“home/HwHiAiUser/SceneTextRecognition/dist/”目录下。*

## 3.3 模型转换

  文本检测使用的是开源模型AdvancedEAST，示例模型请参考[模型转换说明](Data/Models/TextDetection/README.zh.md)获取并转换

  文本识别使用的是开源模型chineseocr，示例模型请参考[模型转换说明](Data/Models/TextRecognition/README.zh.md)获取并转换


## 3.4 配置

  运行前需要在 `Data/Config/setup.config` 配置以下信息

  配置程序运行的deviceId，调试模式以及循环运行次数
  debugMode表示调试模式，开启后会保存中间过程图片和日志，主要是便于问题定位，默认设置为关闭
  **注意**：如果输入图片中包含敏感信息，使用完后请按照当地法律法规要求自行处理，防止信息泄露。
  runTimes表示循环读取文件的次数，为0表示长稳测试，需要使用ctrl + c结束长稳测试
  ```bash
  SystemConfig.deviceId = 0
  SystemConfig.debugMode = 0         // 0: debug off  1: debug on
  SystemConfig.runTimes = 1          // 0: long time test  other: cycle run times
  ```

  配置待处理图片的路径，可以指定文件夹或单个文件
  ```bash
  # configuration for ImageReader, the input path can be an path of a file or a folder
  ImageReader.inputPath = ./image
  ```

  配置文本检测模型输入要求的宽和高，模型名称以及转换好的om模型

  文本识别模型输入宽高比支持1:1,1:2,2:1等比例，以提升推理精度，dynamicHWList中设置的是模型支持的最小和最大宽高值，两两组合形成4个档位，必须跟模型转换时设置的参数一致
  ```bash
  # configuration for TextDetection
  # 2 values of width and height, we will get 4 types here: 416,416;416,832;832,416;832,832
  TextDetection.dynamicHWList = 416,832
  TextDetection.modelName = AdvancedEAST
  TextDetection.modelPath = ./Data/Models/TextDetection/advancedeast.om
  ```

  配置文本识别模型模型名称**modelName**，模型类型**modelType**，输入要求的宽**modelWidth**和高**modelHeight**以及转换好的om模型**modelPath**

  **ignoreWarpPerspective**表示是否跳过透视变换操作，针对于直排文本的识别场景可选择不做透视变换

  为了提升推理效率，我们设置了模型多batch推理参数**batchSize**，表示模型单次可推理的图片数量。超时机制参数**timeoutInterval**表示在该间隔内图片数量未达到**batchSize**，也会进行推理

  **keysFilePath**是字符字典文件所在路径，用于根据模型推理得到的索引来查找对应的字符

  ```bash
  # configuration for TextRecognition Chinese_OCR
  TextRecognition.modelName = Chinese_OCR
  TextRecognition.modelType = 0       // 0: Chinese_OCR  1: CRNN
  TextRecognition.modelHeight = 32
  TextRecognition.modelWidth = 320
  TextRecognition.batchSize = 8
  TextRecognition.timeoutInterval = 200  // milli-second
  TextRecognition.modelPath = ./Data/Models/TextRecognition/chineseocr_32_320_8_batch.om
  TextRecognition.keysFilePath = ./Data/Models/TextRecognition/chineseocr_keys.txt
  TextRecognition.ignoreWarpPerspective = true
  ```

  ```bash
  # configuration for TextRecognition CRNN
  TextRecognition.modelName = CRNN
  TextRecognition.modelType = 1       // 0: Chinese_OCR  1: CRNN
  TextRecognition.modelHeight = 32
  TextRecognition.modelWidth = 320
  TextRecognition.batchSize = 8
  TextRecognition.timeoutInterval = 200  // milli-second
  TextRecognition.modelPath = ./Data/Models/TextRecognition/crnn_chinese_32_320_8_batch.om
  TextRecognition.keysFilePath = ./Data/Models/TextRecognition/crnn_cn_keys.txt
  TextRecognition.ignoreWarpPerspective = true
  ```

  配置图片中检测到的文字结果保存路径，会在程序运行当前路径创建指定名称的文件夹
  enableCallback为是否使能处理识别结果的回调函数
  ```bash
  # configuration for ResultProcess
  ResultProcess.savePath = ./result
  ResultProcess.enableCallback = 1 # whether to enable callback functions
  ```

## 3.5 运行

### 输入图像约束

仅支持JPG格式。

### 运行程序

- **步骤 1** 将待进行OCR识别处理的图片，如”xxx.jpg“，上传至”/home/HwHiAiUser/SceneTextRecognition/dist/image“目录。

- **步骤 2** 执行如下命令，启动OCR程序。

  ```
  ./ocr
  ```

  根据屏幕日志确认是否执行成功。

  识别结果存放在“/home/HwHiAiUser/SceneTextRecognition/dist/result”目录下。

*提示：输入./ocr -h可查看该命令所有信息。运行可使用的参数如表3-2 运行可使用的参数说明所示。*

**表3-1** 运行可使用的参数说明

| 选项           | 意义                                                                        | 默认值                         |
| -------------- | --------------------------------------------------------------------------- | ------------------------------ |
| --log_level    | 调试级别，取值为，0：debug；1：info；2：warn；3：error；4：fatal；5：off 。 | 1                              |
| --stats        | 性能统计，取值为，true：开；false：关                                       | false                          |

### 结果展示

  OCR识别结果保存在配置文件中指定路径的x_xx_xxx.txt中（x 为图片名称； xx 为图片id;  xxx 为时间戳）
  每个x_xx_xxx.txt中保存了每个图片名称和ID，图片中检测到的文本框ID，文本框四个顶点的坐标位置以及文本内容，格式如下：
  ```bash
  Image: test_0
  Item0[(138, 19), (138, 46), (346, 46), (346, 19)]: 故人西辞黄鹤楼
  Item1[(137, 60), (136, 87), (346, 87), (346, 61)]: 烟花三月下扬州
  Item2[(137, 102), (136, 129), (345, 128), (345, 103)]: 孤帆远影碧空尽
  Item3[(137, 143), (136, 171), (342, 171), (343, 144)]: 惟见长江天际流
  Item4[(137, 185), (137, 212), (344, 211), (344, 185)]: 朝辞白帝彩云间
  Item5[(137, 226), (137, 255), (345, 254), (346, 229)]: 千里江陵一日还
  Item6[(137, 269), (136, 295), (346, 296), (346, 270)]: 两岸猿声啼不住
  Item7[(137, 309), (137, 337), (345, 337), (345, 313)]: 轻舟已过万重山
  Item8[(138, 351), (138, 378), (347, 380), (347, 352)]: 天门中断楚江开
  Item9[(138, 393), (137, 420), (347, 419), (347, 394)]: 碧水东流至此回
  ```
  **注意**：如果输入图片中包含敏感信息，使用完后请按照当地法律法规要求自行处理，防止信息泄露。
  
# 4 动态库依赖说明

Demo动态库依赖可参见代码中“src”目录的“CMakeLists.txt”文件，见文件”link_libraries“和“target_link_libraries”参数处。

**表4-1** 动态库依赖说明

| 依赖软件           | 说明                                     |
| ------------------ | ---------------------------------------- |
| libascendcl.so     | ACL框架接口，具体介绍可参见ACL接口文档。 |
| libacl_dvpp.so     | ACL框架接口，具体介绍可参见ACL接口文档。 |
| libpthread.so      | C++的线程库。                            |
| libopencv_world.so | OpenCV的基本组件，用于图像的基本处理。   |
