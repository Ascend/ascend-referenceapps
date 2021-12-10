[TOC]

# 1 简介

## 1.1 背景介绍

根据系统输入源不同（视频流和图片流），人脸识别系统主要有检测识别一体化和检测识别分离两种方案。检测识别一体化方案集人脸检测与人脸识别为一体，从视频输入到结果输出形成完整链条，系统整体性能较高。检测识别分离方案分为人脸抓拍子系统与人脸识别子系统，子系统间通过抓拍的图像数据进行交互。子系统简要介绍如下：
1.  人脸抓拍子系统负责在视频流中检测人脸，并抓拍输出符合人脸识别要求的人脸图像。
1.  人脸识别子系统在人脸抓拍的图像中精确定位人脸位置，将人脸对齐以提高识别精度，提取人脸描述特征向量并在人脸底库中进行检索。

本文档提供对人脸识别子系统实现方案的说明，本实现方案主要目的是打通ACL编程方案，为使用ACL进行目标检测、目标分类提供代码样例，下称Demo。

## 1.2 硬件方案介绍

本系统采用Atlas 300 AI加速卡 （型号 3000）作为实验验证的硬件平台。具体产品实物图和硬件参数请参见《Atlas 300 AI加速卡 用户指南（型号 3000）》。由于采用的硬件平台为含有Atlas 300的Atlas 800 AI服务器 （型号3000），而服务器一般需要通过网络访问，因此需要通过笔记本或PC等客户端访问服务器，而且展示界面一般在客户端。

### 支持的产品形态

Atlas 800 (Model 3000)

## 1.3 软件方案介绍
### 软件约束

**表1-1** 软件约束说明

| 环境要求   | 说明              |
| ---------- | ----------------- |
| 服务器架构 | aarch64 (arm)     |
| 操作系统   | CentOS 7.6 完整版 |
| 驱动版本   | 1.75.T15.0.B150  20.1.0   |

### 主要功能介绍

软件方案进一步将人脸识别子系统划分为注册建库和检索查询两个子系统，子系统功能具体描述请参见表1-2 系统方案各子系统功能描述。注册建库和检索查询的算法流程非常相似，主要由图像解码、图像预处理、人脸检测、人脸对齐等功能模块组成。模块功能具体介绍请参见表1-3 人脸识别子系统主要功能模块介绍。实际应用中为了保证人脸识别的效果，注册建库与检索查询的算法模型需要一致。本方案选择“CenterFace”模型作为人脸检测和关键点检测模型，模型来源“https://github.com/Star-Clouds/CenterFace/tree/master/models/onnx”中的“centerface.onnx”模型文件，选择“Resnet-18”模型作为人脸描述向量提取模型，模型来源“https://gitee.com/HuaweiAtlas/FaceRecognition/tree/master/depository/models/resnet18”中的“resnet18.pb”模型文件。

模型需要用ATC转换工具转换成om模型，具体转换步骤请参见《ATC工具使用指导》。本方案已提供转换后的om模型，模型存放在目录“/idrecognition/dist/model”。

**表1-2** 系统方案各子系统功能描述

| 子系统   | 功能描述                                                      |
| -------- | ------------------------------------------------------------ |
| 注册建库 | 注册建库流程主要负责将人脸图像底库的数据集导入，在人脸图像数据集的每张<上检测最大人脸、将人脸对齐、并提取人脸特征描述向量，最后将人脸特征描述向量和人脸图像索引建立人脸特征数据库。 |
| 检索查询 | 检索查询流程主要负责接收抓拍图像传上来的图像，在抓拍图像中检测人脸，选取合适的人脸做识别（一般人脸抓拍图像中都有一个主目标，做视频的演示demo也可以不选择主目标），将选取的人脸图像做人脸对齐、人脸描述特征向量提取、人脸特征向量比对，相似度最高且相似度达到阈值要求的检索结果。 |

**表1-3** 人脸识别子系统主要功能模块介绍

| 模块名称             | 功能描述                                                      |
| -------------------- | ------------------------------------------------------------ |
| 视频图像解码         | 通过硬件（DVPP）对编码压缩后的视频或图像进行解码，得到RGB格式数据。 |
| 图像预处理           | 进行深度神经网络的图像推理前的图像缩放、抠图等操作。如根据模型获得的坐标与信息，获取ROI人脸小图用于人脸对齐和特征提取。 |
| 人脸检测和关键点定位 | 给定人脸图像，使用深度学习模型，检测出图像中的人脸，并定位出人脸面部的关键区域位置，包括眼睛、鼻子、嘴角等关键点。 |
| 人脸对齐             | 依据图像和人脸模板将ROI人脸图像通过旋转获得人脸的正面对齐图像，并根据特征提取模块的输入大小调整ROI图像。 |
| 人脸特征提取         | 在对齐的ROI人脸图像上运用深度学习CNN神经网络提取人脸的特征值，并对特征值进行L2归一化。 |
| 人脸特征比对         | 通过深度学习神经网络获得的人脸特征向量和注册的人脸特征库中的特征向量做比对。一般采用欧式距离或余弦距离计算，根据实际网络推理效果设定特定阈值。 |
| 人脸特征库           | 对采集到的人脸图片提取人脸特征，将特征和图像的UUID进行映射，可以通过特征向量索引到对应的人脸信息。 |
| 人脸信息库           | 将实时抓拍图像或人脸注册底库获得的检测人脸信息保存在人脸信息数据库中，用于识别结果展示等用途。 |
| 结果展示             | 实时显示检测到的目标人物的信息，如姓名、性别等。             |

### 方案架构设计

该方案主要是针对Atlas300 AI加速卡设计的，将各个模块在Host CPU和Device(Atlas300加速卡芯片)之间进行划分，得到如下图所示的方架构图，该架构图中用不同颜色表示不同的计算单元，黄色表示在Host CPU计算、粉红色表示利用Device端的Dvpp进行计算、绿色表示利用Device的AICore进行计算。使用不同颜色的线段表示数据搬运是否需要跨越PCIE接口，其中黑色线段表示不需要跨域PCIE的内存数据搬移，红色线段表示需要跨域PCIE接口的数据传输。

![arch](arch.jpg)

### 代码主要目录介绍

本Demo工程名为IDRecognition，根目录下src为源码目录，dist为目录运行，现将dist与src的子目录介绍如下：
```
IDRecognition								// Demo根目录
├── build
├── opensource
├── dist									// Demo编译输出与运行目录
│   ├── config								// 存放所有运行时需要的配置文件，包括注册与检索流程
│   │   └── reg								// 存储需要注册的图像信息
│   ├── featureLib							// 人脸图像信息底库
│   ├── lib									// Demo运行时依赖库
│   ├── logs								// 运行时日志存放目录
│   ├── model								// 人脸识别模型存放目录
│   ├── pic									// 待注册与检测人脸图像的存放目录
│   │   ├── reg								// 待注册人脸图像存放目录
│   │   │   ├── reg0
│   │   │   └── reg1
│   │   └── search							// 待检索人脸图像存放目录
│   │       ├── search0
│   │       └── search1
│   ├── result								// 检索结果存放目录
│   │   └── ChannelId0
│   │       └── FrameId0
│   ├── temp								// 存储运行过程中的中间结果
│   └── tools								// 结果展示脚本存放目录
└── src										// Demo源码目录
    ├── AscendBase							// 基础组件库
    ├── Common	
    │   ├── DataType						// 数据结构
    │   ├── FaceFeatureLib					// 人脸特征库模块
    │   ├── Module
    │   │   ├── FaceDetectionLandmark		// 人脸检测和关键点提取模块
    │   │   ├── FaceFeature					// 人脸特征提取模块
    │   │   ├── FaceSearch					// 人脸检索模块
    │   │   ├── FaceStock					// 人脸特征信息入库模块
    │   │   ├── ImageDecoder				// 图像信息解码模块
    │   │   ├── JpegReader					// 图像信息读取模块
    │   │   ├── ModuleBase					// 各业务模块的基类
    │   │   ├── ModuleFactory				// 模块工厂，创建各模块实例
    │   │   └── WarpAffine					// 人脸矫正模块
    │   └── SystemManager					// 系统管理模块
    └── HostCPU								// 方案调度
```


# 2 环境搭建

### 2.1 第三方软件依赖说明

**表2-1** 第三方软件依赖说明

| 软件名称 | 版本  | 下载地址                                                     | 功能说明                     |
| -------- | ----- | ------------------------------------------------------------ | ---------------------------- |
| python   | 3.7.5 | https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz     | 用于运行结果展示脚本。       |
| opencv   | 4.2.0 | https://github.com/opencv/opencv/releases/tag/4.2.0          | 用于人脸对齐时进行仿射变换。 |

在200RC上执行安装时，如出现编译报错内存不足，请去除编译时的-j选项

### 2.2 python-3.7.5安装

- **步骤 1**   使用任意远程登陆终端，以**root**用户登录服务器操作后台。

- **步骤 2**   将"Python-3.7.5.tgz”代码包下载至任意目录，如“/home/HwHiAiUser/python3.7.5”，运行如下命令解压。

  ```
  tar -xzvf Python-3.7.5.tgz
  ```

- **步骤 3**  执行如下命令，进入解压目录，并配置python-3.7.5安装路径为"/usr/local/python3.7.5"。

  ```
  cd Python-3.7.5
  ./configure --prefix=/usr/local/python3.7.5 --enable-shared
  ```

  *提示：--enable-shared参数用于编译出libpython3.7m.so.1.0动态库。*

- **步骤 4**  执行如下命令，以64线程进行编译。

  ```
  make -j64
  ```

- **步骤 5**  执行如下命令，以64线程进行安装。

  ```
  make install -j64
  ```

- **步骤 6**  查询/usr/lib64或/usr/lib下是否有libpython3.7m.so.1.0，若有则跳过此步骤或将原有文件备份后执行如下命令：

  将编译后的如下文件复制到/usr/lib64。

  ```
  sudo cp /usr/local/python3.7.5/lib/libpython3.7m.so.1.0 /usr/lib64
  ```

  如果环境上没有/usr/lib64，则复制到/usr/lib目录：

  ```
  sudo cp /usr/local/python3.7.5/lib/libpython3.7m.so.1.0 /usr/lib
  ```

  *提示：libpython3.7m.so.1.0文件所在路径请根据实际情况进行替换。*

- **步骤 7**  执行如下命令设置软链接：

  ```
  sudo ln -s /usr/local/python3.7.5/bin/python3 /usr/bin/python3.7
  sudo ln -s /usr/local/python3.7.5/bin/pip3 /usr/bin/pip3.7
  sudo ln -s /usr/local/python3.7.5/bin/python3 /usr/bin/python3.7.5
  sudo ln -s /usr/local/python3.7.5/bin/pip3 /usr/bin/pip3.7.5
  ```

  执行上述软链接时如果提示链接已经存在，则可以先执行如下命令删除原有链接然后重新执行。

  ```
  sudo rm -rf  /usr/bin/python3.7.5
  sudo rm -rf  /usr/bin/pip3.7.5
  sudo rm -rf  /usr/bin/python3.7
  sudo rm -rf  /usr/bin/pip3.7
  ```

- **步骤 8**  安装完成之后，执行如下命令查看安装版本，如果返回相关版本信息，则说明安装成功。

  ```
  python3.7.5 --version
  pip3.7.5  --version
  ```
  

### 2.3 opencv-4.2.0安装

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
*  `DRIVER_HOME`      可选，driver安装路径，默认和$ASCEND_HOME一致，不一致时请设置
*  `ASCEND_VERSION`   acllib 的版本号，用于区分不同的版本，参考$ASCEND_HOME下两级目录，一般为 `ascend-toolkit/*version*`
*  `ARCH_PATTERN`     acllib 适用的 CPU 架构，查看$ASCEND_HOME/$ASCEND_VERSION文件夹，可取值为 `x86_64-linux` 或 `arm64-linux`等

```
export ASCEND_HOME=/usr/local/Ascend
export DRIVER_HOME=/usr/local/Ascend
export ASCEND_VERSION=ascend-toolkit/latest
export ARCH_PATTERN=x86_64-linux
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

## 3.2 编译

- **步骤 1**   以**root**用户登录服务器操作后台，并设置环境变量。

- **步骤 2**   将静态目标识别代码下载至任意目录，如“/home/HwHiAiUser/IDRecognition”，然后进入到IDRecognition目录下，新建build目录。

- **步骤 3**   执行如下命令，构建代码。

  ```
   cd /home/HwHiAiUser/IDRecognition/build
   cmake ..
  ```

- **步骤 4**   运行如下命令，以64线程编译代码。

  ```
  make -j64
  ```

  *提示：编译完成后会生成可执行文件“AclIDRcgHostCPU”，存放在“home/HwHiAiUser/IDRecognition/dist/”目录下。*

## 3.3 运行

### 设置运行Device ID

Demo运行时分为人脸注册与人脸检索两个流程，将目标人脸注册到底库后，再进行检索。由于一张Atlas300加速卡有4个Device芯片，注册与检索需要在同一Device运行，因此执行以下步骤设置Device ID。 

- **步骤 1**  执行以下命令，进入“facedemoReg.config”和“facedemoSearch.config”配置文件所在目录。

  ```
  cd /home/HwHiAiUser/IDRecognition/dist/config
  ```

- **步骤 2**  打开配置文件设置Device ID，以facedemoReg.config为例。

  ```shell
  vi facedemoReg.config
  ```

  用vi编辑器打开配置文件后，输入”**i**“，进入编辑模式，将“SystemConfig.deviceId”参数的值修改为用于运行注册流程的ID号（0、1、2、3为可选值），ID号的选择请不要超过设备装载的芯片ID号限制。按"Esc"退出编辑模式，输入“**:wq**“命令保存修改并退出vi编辑器。以相同操作修改“facedemoSearch.config”文件的Device ID，使检索与注册的Device ID相同。

### 输入图像约束

仅支持JPG格式。

### 运行人脸注册

- **步骤 1** 将待注册的人脸照，如”xxx.jpg“，上传至”/home/HwHiAiUser/IDRecognition/dist/pic/reg/reg0“目录。

- **步骤 2** 执行如下命令，进入人脸注册信息配置目录，配置注册信息。

  ```
  cd /home/HwHiAiUser/IDRecognition/dist/config/reg
  vi faceReg0.regcfg
  ```

  用vi编辑器打开配置文件后，输入”**i**“，进入编辑模式，若注册人脸数为1，则修改首行PersonCount值为1，以此类推。空一行，配置人脸图像路径、姓名、性别、年龄等信息，示例如下。配置完成后按"Esc"退出编辑模式，输入“**:wq**“命令保存修改并退出vi编辑器。

  ```
  PersonCount = 1
  
  Person0.ImagePath = ./pic/reg/reg0/xxx.jpg
  Person0.Name = xxx
  Person0.Gender = male
  Person0.Age = 50
  ```

- **步骤 3**  执行如下命令，进入可执行文件”AclIDRcgHostCPU“所在目录，并运行注册流程。

  ```
  cd /home/HwHiAiUser/IDRecognition/dist
  ./AclIDRcgHostCPU -run_mode 1
  ```

  若窗口出现以下提示，表示人脸注册成功。

  ```
  IdRecognition demo finish registration successfully on channel 0
  ```

  注册生成的特征信息存放在“/home/HwHiAiUser/IDRecognition/dist/featureLib”目录下。

- **注意**  注册生成的特征信息包含人脸特征等敏感数据，使用完后请按照当地法律法规要求自行处理，防止信息泄露。

### 运行人脸检索

- **步骤 1** 将待检索的人脸照上传至”/home/HwHiAiUser/IDRecognition/dist/pic/search/search0/“目录。

- **步骤 2** 执行如下命令，进入可执行文件”AclIDRcgHostCPU“所在目录，并运行检索流程。

  ```
  cd /home/HwHiAiUser/IDRecognition/dist
  ./AclIDRcgHostCPU -run_mode 0
  ```

  若窗口出现以下提示，表示人脸检索成功。

  ```
  IdRecognition demo finish search successfully on channel 0
  ```

  检索结果存放在“/home/HwHiAiUser/IDRecognition/dist/result”目录下。

- **注意**  检索结果包含人脸信息等敏感数据，使用完后请按照当地法律法规要求自行处理，防止信息泄露。

*提示：输入./AclIDRcgHostCPU -h可查看该命令所有信息。运行可使用的参数如表3-2 运行可使用的参数说明所示。*

**表3-2** 运行可使用的参数说明

| 选项          | 意义                                                         | 默认值                         |
| ------------- | ------------------------------------------------------------ | ------------------------------ |
| -acl_setup    | AscendCL初始化配置文件。                                     | ./config/aclInit.config        |
| -setup_reg    | 注册流程pipeline配置文件。                                   | ./config/facedemoReg.config    |
| -setup_search | 搜索流程pipeline配置文件。                                   | ./config/facedemoSearch.config |
| -run_mode     | 运行模式，0表示运行搜索流程，1表示注册流程。                 | 0                              |
| -debug_level  | 调试级别，取值为，0：debug；1：info；2：warn；3：error；4：fatal；5：off | 3                              |
| -stats        | 性能统计，取值为，true：开；false：关                        | false                          |

### 结果展示

- **步骤 1** 执行如下命令，进入结果展示脚本”IdRecognitionResultShow.py“所在目录。

  ```
  cd /home/HwHiAiUser/IDRecognition/dist/tools
  ```

- **步骤 2** 执行如下命令，进入python3.7.5命令模式，并导入cv2模块后退出。

  ```
  python3.7.5
  import cv2
  exit()
  ```

- **步骤 3** 执行如下命令，展示检索结果。

  ```
  python3.7.5 IdRecognitionResultShow.py
  ```

# 4 动态库依赖说明

Demo动态库依赖可参见静态人脸识别代码中“IDRecognition/src/HostCPU”目录的“CMakeLists.txt”文件，见文件”link_libraries“和“target_link_libraries”参数处。

**表4-1** 动态库依赖说明

| 依赖软件           | 说明                                     |
| ------------------ | ---------------------------------------- |
| libascendcl.so     | ACL框架接口，具体介绍可参见ACL接口文档。 |
| libascend_hal.so   | ACL框架接口，具体介绍可参见ACL接口文档。 |
| libacl_dvpp.so     | ACL框架接口，具体介绍可参见ACL接口文档。 |
| libc_sec.so        | ACL框架接口，具体介绍可参见ACL接口文档。 |
| libpthread.so      | C++的线程库。                            |
| libopencv_world.so | OpenCV的基本组件，用于图像的基本处理。   |


