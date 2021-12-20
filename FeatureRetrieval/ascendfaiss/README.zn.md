[TOC]

# 1 特征检索FeatureRetrieval介绍

## 1.1 应用场景

在人脸和人体识别等应用场景中，通常使用卷积神经网络提取人脸/人体的深度特征，得到半结构化的特征向量，针对特征向量的相似性计算可以实现特定目标的检索。比如基于图片库提取特征向量形成底库后，可以将目标图片使用相同的神经网络提取深度特征生成待查询的向量，并将其在底库中进行比对，若相似度很高则认为在底库中检索到了匹配的目标。

考虑到检索精度、计算和存储的开销，特征向量的长度多为512维（以float32居多）。以底库量级划分，检索库类型分为小库（精确搜索）和大库（非精确搜索）。小库规模通常在30万~100万条的量级，而大库规模可达到千万甚至亿等级别。

业界使用较多的相似性检索框架是[Faiss](https://github.com/facebookresearch/faiss )，特征检索FeatureRetrieval大库检索主要是基于Faiss特征检索框架在Ascend平台实现IVFPQ、IVFSQ、IVFINT8等算法，包括在Ascend平台进行加速的TBE算子；小库检索主要是实现了Flat、SQ、INT8等暴力检索算法。特征检索FeatureRetrieval采用C++语言结合TBE算子开发，并提供可选的Python接口，支持Linux ARM和X86_64平台。

## 1.2 主要功能介绍

大小库检索主要提供特征向量查询、底库增加、底库删除等功能，具体功能介绍请参见表1-1。

表1-1 主要功能介绍

| 功能名称          | 功能介绍                                                     |
| ----------------- | ------------------------------------------------------------ |
| 特征向量查询功能  | 用户输入待查询的特征向量后，大小库检索将待查询的特征向量，与底库中的向量进行距离计算并选取相似度最高的Top K个结果返回，从而完成一次特征向量的检索。 |
| 建库/底库添加功能 | 建库是实现用户数以百万、千万级特征底库添加到Ascend平台。底库添加是在现有的底库基础上添加新的特征数据到Ascend平台。 |
| 底库删除功能      | 根据指定的索引，删除底库中该索引对应的特征向量，支持多条删除功能。 |
| 底库保存功能      | 实现用户底库特征和索引保存在本地的功能，业务恢复时可以不用进行训练和建库操作。 |
| 底库恢复功能      | 实现把已经保存的特征库和索引恢复到Ascend平台的功能。         |

# 2 安装指导

## 2.1 环境依赖

FeatureRetrieval安装需要依赖**开放态的ACLlib**、[Faiss](https://github.com/facebookresearch/faiss.git)和[Protocol Buffers](https://github.com/protocolbuffers/protobuf)。在安装FeatureRetrieval前，请用户确保已正确安装了这些依赖库。

> ACL（Ascend Computing Language）作为用户的API编程接口，提供Device管理、Context管理、Stream管理、内存管理、模型加载与执行、算子加载与执行、媒体数据处理等C++ API库供用户开发应用。检索FeatureRetrieval依赖开放态ACLlib的开发态（编译）和运行态（部署），所以请注意使用开放态ACLlib安装包，安装包获取和具体环境安装方法参考《CANN <version> 软件安装指南》（标准态和开放态）。
>
> 特征检索FeatureRetrieval需要的ACL包列表：
>
> | 简称             | 安装包全名                                                   | 默认安装路径                                                 |
> | ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
> | CANN             | Ascend-cann-toolkit\_{version}\_linux-{arch}.run             | /usr/local/Ascend/ascend-toolkit/latest（使用atc/opp/toolkit组件） |
> | driver           | A300-3000-npu-driver\_{version}\_{os}-aarch64.run / A300-3010-npu-driver\_{version}\_{os}-x86_64.run | /usr/local/Ascend                                            |
> | firmware         | A300-3000-npufirmware\_{version}.run / A300-3010-npufirmware\_{version}.run | /usr/local/Ascend                                            |
> | 开放态ACL        | Ascend-acllib-<version>-minios.aarch64.run                   | /usr/local/AscendMiniOs（开发态）<br>/usr/local/AscendMiniOSRun（运行态） |
> | 开放态驱动源码包 | Ascend310-driver-<version>-minios.aarch64-src.tar.gz         | FeatureRetrieval部署阶段解压使用，用于打包device侧OS镜像文件 |
>
> > Ascend-cann-device-sdk\_<version>\_linux-aarch64.zip解压得到Ascend-acllib-<version>-minios.aarch64.run和Ascend310-driver-<version>-minios.aarch64-src.tar.gz
> >
> > minios.aarch64安装包需要同时安装开发态和运行态，FeatureRetrieval编译依赖开放态ACLlib（开发态），部署依赖于开放态ACLlib（运行态）。

表2-1 硬件要求

| 环境要求       | 说明                                         |
| -------------- | -------------------------------------------- |
| 支持的硬件环境 | Atlas 800（型号3000/3010），A500PRO，200 SOC |
| 操作系统       | CentOS 7.6 / Ubuntu18.04                     |

表2-2 已安装环境依赖软件及软件版本

| 软件名称 | 软件版本                              |
| -------- | ------------------------------------- |
| GCC   | 4.8.5/7.3.0等 |
| ACL   | 20.1.0, 20.2.0, 21.1.0等（开放态） |
| Python | 3.7.5                          |
| numpy    | 1.16.0及以上                         |

表2-3 待安装环境依赖软件及软件版本

| 软件名称   | 软件版本 |
| ---------- | -------- |
| SWIG       | 3.0.12   |
| OpenBLAS   | 0.3.9    |
| Faiss      | 1.6.1    |
| protobuf   | 3.9.0    |
| googletest | 1.8.1    |

> SWIG安装指导请参考官方资料。

安装依赖前需要确保以下组件已经安装：

```shell
pkg-config autoconf automake autotools-dev m4 gfortran
# sudo apt install pkg-config autoconf automake autotools-dev m4 gfortran
```

Faiss依赖blas和lapack，我们通过安装OpenBLAS来提供blas和lapack引用（lapack已经包含在OpenBLAS中)；测试用例基于googletest构建。

Faiss的python接口依赖numpy，编译过程中需要SWIG。

FeatureRetrieval依赖Faiss，protobuf；测试用例基于googletest构建。

## 2.2 OpenBLAS安装

1. 下载OpenBLAS v0.3.9源码压缩包并解压

   ```shell
   wget https://github.com/xianyi/OpenBLAS/archive/v0.3.9.tar.gz -O OpenBLAS-0.3.9.tar.gz
   tar -xf OpenBLAS-0.3.9.tar.gz
   ```

2. 进入OpenBLAS目录

   ```shell
   cd OpenBLAS-0.3.9
   ```

3. 编译安装

   ```shell
   make FC=gfortran USE_OPENMP=1 -j
   # 默认将OpenBLAS安装在/opt/OpenBLAS目录下
   make install
   
   # 或执行如下命令可以安装在指定路径
   #make PREFIX=/your_install_path install
   ```

   > USE_OPENMP=1主要是为了解决在arm平台上Faiss调用openblas的sgemm接口死锁问题

4. 配置库路径

   ```shell
   ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so
   
   # 配置/etc/profile
   vim /etc/profile
   # 在/etc/profile中添加export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH
   source /etc/profile
   ```

## 2.3 Faiss安装

安装Faiss之前请先安装好OpenBLAS，创建软链接文件`/usr/lib/libopenblas.so`。

如果是ARM平台，编译安装Faiss前需要先[修改Faiss源码](#4.1.7 Faiss编译问题)。

- **步骤 1**   以**root**用户登录服务器操作后台（Host侧）

- **步骤 2**   把检索FeatureRetrieval源码包下载至任意目录并解压，如“`/home/HwHiAiUser`”，进入“ascendfaiss”源码目录

  ```shell
  # now in /home/HwHiAiUser directory
  mkdir FeatureRetrieval
  tar xf Ascend-featureretrieval_<version>_src.tar.gz -C FeatureRetrieval
  
  # 进入ascendfaiss源码目录
  cd FeatureRetrieval/src/ascendfaiss
  ```

- **步骤 3**   执行以下步骤编译安装Faiss

  1. 进入第三方依赖目录

     ```shell
     cd third_party
     ```

  1. 执行以下命令下载Faiss源码压缩包并解压

     ```shell
     wget https://github.com/facebookresearch/faiss/archive/v1.6.1.tar.gz
     tar -xf v1.6.1.tar.gz
     ```

  1. 进入Faiss目录

     ```shell
     cd faiss-1.6.1
     ```

  1. 执行以下命令完成Faiss的编译配置

     ```shell
     # 不需要Python接口
     ./configure --without-cuda
     
     # 需要Python接口，/usr/bin/python3.7是python解释器文件路径，可根据实际情况修改
     PYTHON=/usr/bin/python3.7 ./configure --without-cuda
     ```

     > - 如果用户需要使用GPU，则`--without-cuda`参数需要修改为`--with-cuda=<cuda install path>`，`<cuda install path>`为CUDA安装路径。同时要在命令中增加`--with-cuda-arch=<cuda编译参数>` ，如`--with-cuda-arch="-gencode=arch=compute_61,code=sm_61"`。

  1. 编译安装

     ```shell
     make -j <thread num> # <thread num>为CPU核数
     make install
     ```

  1. 编译python接口（可选）

     该步为可选项，如果第4步配置了`PYTHON`环境变量，则继续执行，否则跳过该步骤。

     执行以下操作前用户需自行安装SWIG，使用原理可参考faiss中SWIG[说明文档](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md#user-content-the-python-interface )。

     ```shell
     make -C python
     make -C python install
     ```
     > 安装到系统的PYTHON环境中，即/path/to/python3.7下面，分别位于bin目录和lib/python3.7/site-packages目录

  1. 配置系统库查找路径，返回上层目录

     动态链接依赖Faiss的程序在运行时需要知道Faiss动态库所在路径，需要将Faiss的库目录加入`LD_LIBRARY_PATH`环境变量。
     
     ```shell
     # 配置/etc/profile
     vim /etc/profile
     # 在/etc/profile中添加: export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
     # /usr/local是Faiss的安装目录,如果安装在其他目录下,将/usr/local替换为Faiss实际安装路径
     source /etc/profile
     
     cd ..
     ```

## 2.4 Protobuf安装

当前工作目录是`FeatureRetrieval/src/ascendfaiss/third_party`。

  1. 下载Protobuf v3.9.0源码压缩包并解压

     ```shell
     wget https://github.com/protocolbuffers/protobuf/releases/download/v3.9.0/protobuf-all-3.9.0.tar.gz
     tar -xf protobuf-all-3.9.0.tar.gz
     ```

  1. 进入Protobuf目录

     ```shell
     cd protobuf-3.9.0
     ```

  1. 编译安装

     安装host侧需要的protobuf二进制库和依赖头文件

     ```shell
     ./configure --prefix=/usr/local/protobuf
     make -j <thread num> # <thread num>为CPU核数
     make install
     make clean
     ```

     安装device侧需要的protobuf二进制库

     ```shell
     # hcc_path是Ascend toolkit中交叉编译器的安装路径，根据具体安装路径替换以下hcc_path路径
     hcc_path=/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc
     
     CC=${hcc_path}/bin/aarch64-target-linux-gnu-gcc CXX=${hcc_path}/bin/aarch64-target-linux-gnu-g++ CXXFLAGS=-fPIC \
     ./configure --prefix=/opt/aarch64/protobuf --host=aarch64-linux --disable-protoc
     make -j<thread num> # <thread num>为参与编译的线程数
     make install
     make clean
     ```
     
  1. 配置系统库查找路径，返回上层目录

     动态链接依赖protobuf的程序在运行时需要知道protobuf动态库所在路径，需要将protobuf的库目录加入`LD_LIBRARY_PATH`环境变量。
     
     ```shell
     # 配置/etc/profile
     vim /etc/profile
     # 在/etc/profile中添加: export LD_LIBRARY_PATH=/usr/local/protobuf/lib:$LD_LIBRARY_PATH
     source /etc/profile
     
     cd ..
     ```

## 2.5 googletest安装

当前工作目录是`FeatureRetrieval/src/ascendfaiss/third_party`。

1. 下载googletest v1.8.1源码压缩包并解压

   ```shell
   wget https://github.com/google/googletest/archive/release-1.8.1.tar.gz -O googletest-release-1.8.1.tar.gz
   tar -xf googletest-release-1.8.1.tar.gz
   ```

2. 移动googletest目录

   ```shell
   mv googletest-release-1.8.1/googletest googletest
   ```

3. 此处无需编译安装，FeatureRetrieval编译过程中编译

4. 返回上层目录，结束依赖库安装

   ```shell
   cd ..
   ```

## 2.6 FeatureRetrieval安装

当前工作目录是`FeatureRetrieval/src/ascendfaiss`。

**注意：**如果ACL包没有按[2.1 环境依赖](#2.1 环境依赖)表格中默认安装路径安装，需要先更改ACL包的安装路径或修改`ascendfaiss/acinclude/fa_check_ascend.m4`文件和`ascendfaiss/acinclude/fa_check_faiss.m4`中相关路径，设置为ACL包和对应组件的真实安装路径。相关路径变量和默认值列举如下：

| 变量名               | 对应的ACL包名/三方库名 | 默认值                                  |
| -------------------- | ---------------------- | --------------------------------------- |
| with_ascend          | CANN                   | /usr/local/Ascend/ascend-toolkit/latest |
| with_ascendminios    | 开放态ACLlib           | /usr/local/AscendMiniOs（开发态）       |
| with_ascenddriver    | driver/firmware        | /usr/local/Ascend                       |
| with_protobuf        | protobuf for host      | /usr/local/protobuf                     |
| with_protobufaarch64 | protobuf for device    | /opt/aarch64/protobuf                   |
| with_faiss           | faiss                  | /usr/local                              |

以下步骤**1~4**属于FeatureRetrieval源码和TBE算子的编译安装，可以执行当前目录下`build.sh`脚本一键安装；

```shell
bash build.sh
```

步骤**5**是将编译生成的检索可执行文件和TBE算子模型文件等打包部署到Device的文件系统，需要单独执行。

- **步骤 1** 以**root**用户登录服务器操作后台（Host侧）

- **步骤 2** 执行以下步骤编译FeatureRetrieval，用于后续使用FeatureRetrieval库

  1. 执行`cd /home/HwHiAiUser/FeatureRetrieval/src/ascendfaiss`命令进入`ascendfaiss`源码目录

  2. 执行以下命令生成配置脚本（忽略提示的“Makefile.am”的错误）

     ```shell
     ./autogen.sh
     ```

  3. 执行以下命令为`Makefile`生成配置文件`makefile.inc`

     ```shell
     ./configure
     ```

     用户也可以自行选定以下命令执行来完成相关配置：

     > - `./configure --with-ascend=<ascend-acllib-path>`
     >
     >   `<ascend-acllib-path>`为`acllib/atc/opp/toolkit`安装的目录，默认为`/usr/local/Ascend/ascend-toolkit/latest`。
     >
     > - `./configure --with-ascendminios=<ascend-minios-path>`
     >
     >   `<ascend-minios-path>`为开放态acllib安装的目录，开发环境所需的ACLlib包，默认为`/usr/local/AscendMiniOs`。
     >
     > - `./configure --with-ascenddriver=<ascend-driver-path>`
     >
     >   `<ascend-driver-path>`为driver安装的目录，默认为`/usr/local/Ascend`。
     >
     > - `./configure --with-faiss=<faiss-path>`
     >
     >   `<faiss-path>`为Faiss的安装路径，执行此命令用以使用Faiss的头文件和动态链接库。
     >
     > - `./configure --with-protobuf=<protobuf-path>`
     >
     >   `<protobuf-path>`为protobuf的安装路径，执行此命令用以指定protobuf的安装路径。
     >
     > - `./configure --with-protobufaarch64=<protobuf-aarch64-path>`
     >
     >   `<protobuf-aarch64-path>`为protobuf的aarch64版本安装路径，执行此命令用以指定protobuf的aarch64版本安装路径。
     >
     > - `PYTHON=/path/to/python3.7 ./configure`
     >
     >   `/path/to/python3.7`为用户指定版本的全路径Python解释器，执行此命令用以安装Python接口。

  4. 执行以下命令进行编译

     ```shell
     export PKG_CONFIG_PATH=/usr/local/protobuf/lib/pkgconfig
     make -j <thread num>
     ```
     
     `<thread num>`为CPU核数，不写`<thread num>`则表示使用最大核数。

- **步骤 3** 安装

  1. 安装FeatureRetrieval头文件和lib库

     ```shell
     make install  # 头文件会安装到faiss/ascend目录
     ```

  2. 编译Python接口（可选）

     执行该步骤需要保证执行`./configure`前已设置环境变量`PYTHON`，否则不需要执行第2步和第3步

     ```shell
     # 确保调用configure时已设置PYTHON环境变量: PYTHON=/path/to/python3.7 ./configure
     make -C python
     ```
     > 如果提示找不到faiss库，请检查`python/Makefile`文件中的`FAISS_INSTALL_PATH `环境变量

  3. 安装Python接口（可选）

     ```shell
     make -C python install
     ```

- **步骤 4** 算子编译

  FeatureRetrieval依赖自定义TBE算子，部署前请参考[附录4.2自定义算子](#4.2.2 FeatureRetrieval算子编译部署) 部分进行算子编译和单算子模型转换。

- **步骤 5** 准备神经网络降维模型（可选）

  可通过神经网络实现降维功能（自编码器），用户需要先训练神经网络模型，然后把该模型转换为适配昇腾平台的OM模型，并把OM模型放到 `modelpath` 目录下。

- **步骤 6** 环境部署

  执行以下命令完成环境部署

  ```shell
  bash install.sh <driver-untar-path>
  ```
  

`<driver-untar-path>`为“`Ascend310-driver-{software version}-minios.aarch64-src.tar.gz`”文件解压后的目录，例如文件在“`/usr/local/software/`”目录解压，则`<driver-untar-path>`为“`/usr/local/software/`” 。本步命令用于实现将device侧检索daemon进程文件分发到多个device，执行命令后Ascend Driver中device的文件系统会被修改，所以需要执行**“`reboot`”**命令生效。

**注意**：如果开放态的acllib(minios.aarch64，运行态)没有安装在默认路径`/usr/local/AscendMiniOSRun`下面，需要修改install.sh文件中`acllib_minios_path`变量，将其值设置为**开放态的acllib（运行态）所在目录**。

## 2.7 适配200SOC

200SOC运行环境准备请参考《Atlas 200 DK <version> 使用指南》文档，本章节假设200SOC运行环境已准备完毕。

首先在Atlas 300（型号3000/3010）环境编译FeatureRetrieval代码（参考[2.6 FeatureRetrieval安装](#2.6 FeatureRetrieval安装)）和device侧测试用例（参考[3.4 运行测试用例](#3.4 运行测试用例))。

**以运行device侧TestIndexFlat为例，具体步骤如下：**

- **步骤 1** 拷贝modelpath目录和test/ascenddaemon/TestIndexFlat到200SOC运行环境，modelpath和TestIndexFlat需要在200SOC上同一目录。

- **步骤 2** 200SOC运行环境创建软连接

  ```shell
  # ssh登录200SOC运行环境
  su - root
  mkdir -p /lib64
  ln -sf /lib/ld-linux-aarch64.so.1 /lib64/ld-linux-aarch64.so.1
  ```

- **步骤 3** 运行用例

  ```shell
  # cd 到TestIndexFlat和modelpath所在目录
  export LD_LIBRARY_PATH=/home/HwHiAiUser/Ascend/acllib/lib64:$LD_LIBRARY_PATH
  ./TestIndexFlat
  ```

  > `/home/HwHiAiUser/Ascend` 是CANN中ACLlib库的安装目录。

# 3 使用指导

## 3.1 使用约束

本章节介绍大小库检索的性能规格限制约束。

表3-1 大库性能规格约束

| 性能名称                             | 规格约束                                     |
| ------------------------------------ | -------------------------------------------- |
| 底库大小                             | <= 1亿                                       |
| 特征向量维度                         | 16的倍数                                     |
| PQ量化分段数                         | 2、4、8、12、16、20、24、32、48、64、96、128 |
| L1簇聚类中心个数（index中nlist参数） | IVF倒排链的大小，train的数据量不小于该值     |
| IVFPQ codes编码长度                  | 8bits(即分段聚类中心数为256)                 |
| 相似度度量方法                       | L2 （欧氏距离）                              |
| PCA降维                              | 原始维度和降维后的维度均为16的倍数           |
| IVFSQ 编码类型                       | 只支持SQ8                                    |

表3-2 小库性能规格约束

| 性能名称       | 规格约束                               |
| -------------- | -------------------------------------- |
| 底库大小       | 单个device尽量小于200万（512维情况下） |
| 特征向量维度   | 16的倍数                               |
| 相似度度量方法 | L2/IP （欧氏距离/内积距离）            |
| SQ 编码类型    | 只支持SQ8                              |
| INT8编码类型   | 只支持INT8                             |

## 3.2 C++接口使用

按照[2.6 FeatureRetrieval安装](#2.6 FeatureRetrieval安装) 步骤执行以后会生成静态库文件“`libascendfaiss.a`”跟动态库文件“`libascendfaiss.so`”，用户的可执行文件只需要链接静态库文件或者动态库文件即可。若使用静态库文件，需要在Makefile文件中添加`LDFLAGS`，头文件中需要包含类似以下内容：

```c++
#include <faiss/ascend/AscendIndexIVFPQ.h>
```

## 3.3 Python接口使用

为了在Python工程中将“FeatureRetrieval”作为库引用，用户需要把以下三个文件放在工程目录下，并把该工程目录添加到“PYTHONPATH”环境变量中。

```c++
__init__.py
swig_ascendfaiss.py
_swig_ascendfaiss.so
```

如果用户需要使用FeatureRetrieval库，则需要将以下代码加入用户代码中。

```python
import ascendfaiss
```

## 3.4 运行测试用例

进入“ascendfaiss”源码目录

```shell
make gtest   # 编译gtest
make test_ascend   # 编译运行host侧用例
make test_daemon   # 编译device侧用例, 需要拷贝到device侧执行
```

可以参考测试用例使用和修改各特征比对算法。

# 4 附录

## 4.1 FAQ

### 4.1.1 Ascend Device不同配置比例问题

**问题描述：**

FeatureRetrieval检索性能较弱，TopK排序性能较弱。

**解决方案：**

用户可以通过dsmi API设置Ascend device的AI CPU与Ctrl CPU三种不同的配置比例(2:6, 4:4(default), 6:2)，采用2:6可使FeatureRetrieval性能最佳；操作说明参见《DSMI API参考.chm》。

### 4.1.2 /home/HwHiAiUser目录的权限异常

**问题描述：**

若程序运行一开始出现如下错误提示，说明`/home/HwHiAiUser`目录的权限异常

```shell
[ERROR] HDC:2020-04-22-11:59:42.691.520 [drvPpcCreateDir:35] >>> create ppc dir failed, Permission denied (errno:13)
```

**解决方案：**

用户需要加入HwHiAiUser用户组，为HwHiAiUser用户组加上对“`/home/HwHiAiUser`”目录的读和执行权限

```shell
chmod g+rx /home/HwHiAiUser
```
### 4.1.3 configure时检查环境编译器出错

**问题描述：**

在执行命令`PYTHON=/path/to/python3.7 ./configure`时，**aarch64-target-linux-gnu-g++**或**aarch64-target-linux-gnu-ar**在配置检查时出现以下错误：

```shell
checking for unistd.h... yes
checking for aarch64-target-linux-gnu-g++... no
checking for aarch64-target-linux-gnu-ar... no
configure: error: Couldn't find aarch64-target-linux-gnu-g++
```

**解决方案：**

检查`aarch64-target-linux-gnu-g++`的安装路径，默认安装在`/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc`路径下，如果`toolkit/toolchain`安装在其他路径，在`--with-ascend`配置`toolkit`所在目录即可。

### 4.1.4 configure时检查环境acl.h出错

**问题描述：**

在执行`./configure`时，出现以下错误，提示找不到acl.h：

```shell
checking for unistd.h... yes
checking for aarch64-target-linux-gnu-g++... /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/linux-arrch64/do-arm64le_native/bin/aarch64-target-linux-gnu-g++
checking for aarch64-target-linux-gnu-ar... /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/linux-arrch64/do-arm64le_native/bin/aarch64-target-linux-gnu-ar
checking acl/acl.h usability... no
checking acl/acl.h presence... no
checking for acl/acl.h... no
configure: error: in `/home/featureretrieval':
configure: error: Couldn't find acl.h
See `config.log' for more details
```

 **解决方案：**

主要原因是开放态ACLlib包（开发态）安装不正确，首先查看`/usr/local/AscendMiniOs/acllib/include/acl/`路径是否存在。如果不存在，请确认安装的是开放态的ACLlib包，按照《CANN <version> 软件安装指南（开放态）》中**安装开发和运行环境**章节的安装步骤安装，将开放态ACLlib包（开发态）安装在`/usr/local/AscendMiniOs`文件夹下。

如果acllib minios.aarch64已经安装在了其他路径，则在`./configure`时通过命令`--with-ascendminios=<ascend-minios-path>`指定acllib minios.aarch64安装路径。

### 4.1.5 configure时检查环境ascend_hal.h出错

**问题描述**

在执行`./configure`时，出现以下错误，提示找不到ascend_hal.h：

```shell
checking ascend_hal.h usability... no
checking ascend_hal.h presence... no
checking for ascend_hal.h... no
configure: error: in `/home/faiss/src/ascendfaiss':
configure: error: Couldn't find ascend_hal.h
```

**解决方案**

主要原因ACL的driver包安装不正确或路径配置不正确，首先查看 `/usr/local/Ascend/driver/kernel/inc/driver` 路径是否存在或者`/usr/local/Ascend/driver`路径下kernel文件夹是否存在。如果不存在，按照《CANN <version> 软件安装指南》中**安装开发环境**章节的安装步骤安装driver包。

如果driver包已经安装在了其他路径，则在`./configure`时通过命令`--with-ascenddriver=<ascend-driver-path>`指定driver路径。

### 4.1.6 ARM平台训练速度过慢

**问题描述：**

在使用过程中发现在index的训练过程中发现出现cpu的占用率很高，但是训练速度却很慢的现象，那么很可能就触发了openblas死锁的问题。因为Faiss cpu代码的距离计算使用了openblas的sgemm，FeatureRetrieval的训练和建库代码也依赖了faiss cpu的距离计算代码。

**解决方案：**

1. 用户在编译安装OpenBLAS时，请在源码编译时增加“USE_OPENMP=1”参数，如`make FC=gfortran USE_OPENMP=1`。避免死锁导致训练速度很慢、CPU占用率很高的问题，具体可参考：https://github.com/xianyi/OpenBLAS/issues/660。
2. 若不想重装openblas，可通过设置OPENBLAS_NUM_THREADS环境变量规避，如`export OPENBLAS_NUM_THREADS=4`，具体数据可根据具体环境硬件配置调优，设置不同的值会导致不同的性能。

### 4.1.7 Faiss编译问题

本节修改仅针对arm平台

**问题描述**：

在arm环境下编译faiss可能会出现因为GCC编译器版本比较旧引发的问题。

对于python接口来说，安装后执行`import faiss`会出现 `KeyError: 'flags'`错误，需要**提前修改源码**来避免。

**解决方案**：

请按照以下步骤进行修改相关配置和源码：

1. **仅适用GCC4.8.5**。将`acinclude/ax_check_cpu.m4`文件内的`ARCH_CPUFLAGS="-march=armv8.2-a"`改成`ARCH_CPUFLAGS="-march=armv8-a"`，改完先执行aclocal和autoconf命令，再执行./configure。

   ```diff
   --- a/acinclude/ax_check_cpu.m4
   +++ b/acinclude/ax_check_cpu.m4
   @@ -13,7 +13,7 @@ AC_MSG_CHECKING([for cpu arch])
         ;;
      aarch64*-*)
   dnl This is an arch for Nvidia Xavier a proper detection would be nice.
   -      ARCH_CPUFLAGS="-march=armv8.2-a"
   +      ARCH_CPUFLAGS="-march=armv8-a"
         ;;
      *) ;;
      esac
   ```

2. **仅适用GCC4.8.5**。将`utils/distances_simd.cpp`文件的`vdups_laneq_f32`替换成`vdups_lane_f32`。

   ```diff
   --- a/utils/distances_simd.cpp
   +++ b/utils/distances_simd.cpp
   @@ -561,7 +561,7 @@ float fvec_L2sqr (const float * x,
            accu = vfmaq_f32 (accu, sq, sq);
      }
      float32x4_t a2 = vpaddq_f32 (accu, accu);
   -    return vdups_laneq_f32 (a2, 0) + vdups_laneq_f32 (a2, 1);
   +    return vdups_lane_f32 (a2, 0) + vdups_lane_f32 (a2, 1);
   }
   
   float fvec_inner_product (const float * x,
   @@ -576,7 +576,7 @@ float fvec_inner_product (const float * x,
            accu = vfmaq_f32 (accu, xi, yi);
      }
      float32x4_t a2 = vpaddq_f32 (accu, accu);
   -    return vdups_laneq_f32 (a2, 0) + vdups_laneq_f32 (a2, 1);
   +    return vdups_lane_f32 (a2, 0) + vdups_lane_f32 (a2, 1);
   }
   
   float fvec_norm_L2sqr (const float *x, size_t d)
   @@ -588,7 +588,7 @@ float fvec_norm_L2sqr (const float *x, size_t d)
            accu = vfmaq_f32 (accu, xi, xi);
      }
      float32x4_t a2 = vpaddq_f32 (accu, accu);
   -    return vdups_laneq_f32 (a2, 0) + vdups_laneq_f32 (a2, 1);
   +    return vdups_lane_f32 (a2, 0) + vdups_lane_f32 (a2, 1);
   }
   
   // not optimized for ARM
   ```

3. 由于ARM平台的/proc/cpuinfo中没有flags字段，因此将`python/faiss.py` 文件的`if "avx2" in numpy.distutils.cpuinfo.cpu.info[0]['flags']:`改成`if "avx2" in numpy.distutils.cpuinfo.cpu.info[0].get('flags', ""):`

   ```diff
   --- a/python/faiss.py
   +++ b/python/faiss.py
   @@ -28,7 +28,7 @@ def instruction_set():
               return "default"
      elif platform.system() == "Linux":
            import numpy.distutils.cpuinfo
   -        if "avx2" in numpy.distutils.cpuinfo.cpu.info[0]['flags']:
   +        if "avx2" in numpy.distutils.cpuinfo.cpu.info[0].get('flags', ""):
               return "AVX2"
            else:
               return "default"
   ```

### 4.1.8 如果在nprobe/batch增加时检索性能下降超预期，呈非线性关系

**问题描述**：

假如nprobe/batch=64的性能为QPS=100，在其他参数不变的情况下，当nprobe/batch修改为128时，QPS下降到20，呈非线性关系下降。

**解决方案**：

QPS下降多的原因可能是device侧给Index预申请的内存资源在nprobe/batch翻倍时资源不够，系统会通过aclrtmalloc再去申请内存资源导致性能下降。解决方案是增加内存池的大小（默认是128MB），可以通过修改各Index的config配置中`resourceSize`参数来调整内存池大小(默认为Byte，以Flat的Index为例)：

```c++
faiss::ascend::AscendIndexFlatConfig conf({ 0, 1, 2, 3 }, 512*1024*1024); // 512MB
// 或者使用 conf.resourceSize = 512 * 1024 * 1024;
```
### 4.1.9 protobuf配置问题

**问题描述**：

安装FeatureRetrieval，在`make -j <thread num>`过程中报错`No package 'protobuf' found `。

```shell
pkg-config --cflags protobuf  # fails if protobuf is not installed
Package protobuf was not found in the pkg-config search path.
Perhaps you should add the directory containing `protobuf.pc'
to the PKG_CONFIG_PATH environment variable
No package 'protobuf' found
```

**解决方案**：

配置PKG_CONFIG_PATH环境变量：

```shell
export PKG_CONFIG_PATH=/usr/local/protobuf/lib/pkgconfig  # 当前使用protobuf版本的lib目录
```
### 4.1.10 protobuf链接库问题

**问题描述**

在执行FeatureRetrieval的C++测试用例时出现undefined symbol错误

```shell
undefined symbol: _ZNF6google8protobuf7Message11GetTypeNameB5cxx11Ev
```

**解决方案**

需要将在[2.4节](#2.4 Protobuf安装)安装的protobuf库文件所在路径设置到环境变量LD_LIBRARY_PATH**最前面**：

```shell
 # host侧protobuf库文件默认安装在/usr/local/protobuf/lib目录
 export LD_LIBRARY_PATH=/usr/local/protobuf/lib:$LD_LIBRARY_PATH
```

### 4.1.11 FeatureRetrieval支持多index场景吗

**问题描述**

FeatureRetrieval支持单device多index场景吗？最多支持多少个index？

**解决方案**

FeatureRetrieval支持单个device上面创建多个index，在device侧，每个index共享一块内存池，内存的大小根据index的resourceSize参数，所以在多index场景下需要**确保各index的resourceSize参数大小一致**，否则会影响检索性能；**需要保证各index间串行检索**；index数需要根据业务和device侧的内存大小确定，建议不超过100个index，index数目太多会创建失败或后续特征入库失败。

## 4.2 自定义算子介绍

### 4.2.1 自定义算子简介

特征检索方案使用TIK算子开发实现特征距离计算逻辑，包含以下的自定义算子:

1. PQ距离累加算子：距离累加算子使用Vector的VCADD指令进行累加，使用多核计算。
2. PQ距离计算算子：得到长特征底库数据和待检索的长特征向量之间的L2距离。
3. PQ距离表生成算子：被同时加载至所有Ascend310芯片，然后在每颗Ascend310芯片上生成同样的距离表。
4. PCA降维算子：对长特征向量进行降维。
5. Flat距离计算算子：得到特征底库数据和待检索的特征向量之间的距离（L2/IP）。
6. SQ8距离计算算子：得到SQ量化的特征底库数据和待检索的未量化特征向量之间的距离（L2/IP）。
7. INT8距离计算算子：得到INT8量化的特征底库数据和待检索的INT8量化特征向量之间的距离（L2/COS）。

### 4.2.2 FeatureRetrieval算子编译部署

- **步骤 1**  以**HwHiAiUser**用户登录服务器操作后台（Host侧），执行以下命令进入FeatureRetrieval源码目录。

  ```shell
  cd /home/HwHiAiUser/FeatureRetrieval/src/ascendfaiss
  ```
  编译算子前需要设置`ASCEND_HOME`和`ASCEND_VERSION`环境变量，默认分别为`/usr/local/Ascend`和`ascend-toolkit/latest`

  ```shell
  export ASCEND_HOME=/usr/local/Ascend          # Ascend home path
  export ASCEND_VERSION=ascend-toolkit/latest   # atc/opp/toolkit installation path
  ```

  > `ASCEND_HOME`表示`driver/ascend-toolkit`等组件所在路径，`ASCEND_VERSION`表示当前使用的Ascend版本，如果atc安装路径是`/usr/local/Ascend/ascend-toolkit/latest`则无需设置`ASCEND_HOME`和`ASCEND_VERSION`。

- **步骤 2**   执行以下命令进入`ops`目录，并编译算子原型和算子信息。

  ```shell
  cd ops && bash ./build.sh
  ```

  > 编译过程中系统将检查“opp”算子包的安装目录是否为默认目录，如果不是用户需要自行输入“opp”算子包(`Ascend-opp-{software version}-{os.arch}.run` )的安装目录。

- **步骤 3**   执行`cd ..` 返回`ascendfaiss`源码目录。

- **步骤 4**   执行以下操作生成单算子模型。


  1. 进入tools目录。

     ```shell
     cd tools
     ```

  2. 用户有以下两种方式（单个和批量）生成算子模型文件。

     - − 批量算子模型文件生成 −

         执行以下命令，用户可以得到多组算子模型文件，用户需要自行修改命令中参数，参数说明如表4-6所示。执行命令前，用户需要更改当前目录下的`“para_table.xml”`文件，将所需的参数填入表中，参数填写示例如图4-1所示
         
     ```shell
         python run_generate_model.py -m <mode>
     ```
     
     表4-6 参数说明
         
     | 参数名称 | 参数说明   |
     | -------- | ---------- |
     | `<mode>` | 算法模式。 |
     
     > mode支持ALL以及PCAR，Flat，IVFPQ，SQ8，IVFSQ8，INT8，IVFINT8中的一种或多种，多种之间用逗号隔开，如`python run_generate_model.py -m PCAR,IVFSQ8`。默认**全选**，可以直接执行`python run_generate_model.py`。
     
     - −  单个算法算子模型文件生成  −  

         - −  IVFPQ  −  
            执行以下命令，用户可以得到一组算子模型文件（共八个算子模型文件，其中距离累加算子模型文件一个，距离计算算子模型文件六个，距离表生成算子模型文件一个），用户需要自行修改命令中参数，参数说明如表4-1所示。

            ```shell
            python ivfpq_generate_model.py -d <feature dimension> -c <coarse centroid num> -s <sub quantizers num>
            ```

            表4-1 参数说明

            | 参数名称                | 参数说明           |
            | ----------------------- | ----------------- |
            | `<feature dimension>`   | 特征向量维度D。    |
            | `<coarse centroid num>` | L1簇聚类中心个数。 |
            | `<sub quantizers num>`  | PQ分段数M。       |

            需要注意的是，特征向量维度D和PQ分段数M并不支持可以任意配置，因为PQ分段向量的维度（即subdim）受限于算子的约束。

            | 参数名称        | 参数约束值                     |
            | --------------- | ----------------------------- |
            | subdim(即`D/M`) | 4,8,16,32,48,64,80,96,112,128 |

         - −  Flat  −  
            执行以下命令，用户可以得到三十二个距离计算算子模型文件，用户需要自行修改命令中参数，参数说明如表4-2所示。

            ```shell
            python flat_generate_model.py -d <feature dimension>
            ```

            表4-2 参数说明

            | 参数名称                | 参数说明           |
            | ----------------------- | ----------------- |
            | `<feature dimension>`   | 特征向量维度D。    |

         - −  PCAR  −  
            执行以下命令，用户可以得到九个降维算子模型文件，用户需要自行修改命令中参数，参数说明如表4-3所示。

            ```shell
            python pcar_generate_model.py -i <input dimension> -o <output dimension>
            ```

            表4-3 参数说明

            | 参数名称                | 参数说明            |
            | ----------------------- | ------------------ |
            | `<input dimension>`   | 降维前特征向量维度D。  |
            | `<output dimension>`    | 降维后特征向量维度D。 |

         - −  SQ8  −  
            执行以下命令，用户可以得到二十六个SQ8距离计算算子模型文件，用户需要自行修改命令中参数，参数说明如表4-4所示。

            ```shell
            python sq8_generate_model.py -d <feature dimension>
            ```

            表4-4 参数说明

            | 参数名称                | 参数说明           |
            | ----------------------- | ----------------- |
            | `<feature dimension>`   | 特征向量维度D。    |

         - −  IVFSQ8  −  
            执行以下命令，用户可以得到一组算子模型文件（共九个算子模型文件，其中距离计算算子模型文件七个，SQ8距离计算算子模型文件两个），用户需要自行修改命令中参数，参数说明如表4-5所示。

            ```shell
            python ivfsq8_generate_model.py -d <feature dimension> -c <coarse centroid num>
            ```

            表4-5 参数说明

            | 参数名称                | 参数说明           |
            | ----------------------- | ----------------- |
            | `<feature dimension>`   | 特征向量维度D。    |
            | `<coarse centroid num>` | L1簇聚类中心个数。 |

         - −  IVFFlat  −  
            执行以下命令，用户可以得到一组算子模型文件（共八个算子模型文件），用户需要自行修改命令中参数，参数说明如表4-6所示。
         
            ```shell
         python ivfflat_generate_model.py -d <feature dimension> -c <coarse centroid num>
           ```
           
           表4-6 参数说明
           
           | 参数名称                | 参数说明           |
           | ----------------------- | ------------------ |
           | `<feature dimension>`   | 特征向量维度D。    |
           | `<coarse centroid num>` | L1簇聚类中心个数。 |
           
         
          **注意**：目前ivfflat算法性能存在问题，暂不推荐使用；我们在不久的将来优化这个检索算法。
        
        - −  INT8  −  
        **INT8和SQ8的区别主要在于：INT8由外部进行量化，Index的输入特征是int8类型；SQ8由Index内部量化，Index的输入特征是float32类型。**
          
          执行以下命令，用户可以得到一组纯int8特征按余弦距离度量的算子模型文件（共二十五个算子模型文件），用户需要自行修改命令中参数，参数说明如表4-7所示。
          
          ```shell
            python int8flat_generate_model.py -d <feature dimension>
          ```
          
            表4-7 参数说明
          
          | 参数名称              | 参数说明        |
          | --------------------- | --------------- |
          | `<feature dimension>` | 特征向量维度D。 |
          
         - −   IVFINT8  −  
            执行以下命令，用户可以得到一组纯int8特征按余弦距离度量的算子模型文件（共十七个算子模型文件），用户需要自行修改命令中参数，参数说明如表4-8所示。
            
            ```shell
            python ivfint8flat_generate_model.py -d <feature dimension> -c <coarse centroid num>
            ```
            
            表4-8 参数说明
            
            | 参数名称                | 参数说明           |
            | ----------------------- | ------------------ |
            | `<feature dimension>`   | 特征向量维度D。    |
            | `<coarse centroid num>` | L1簇聚类中心个数。 |
        
         图4-1 算子模型参数文件para_table.xml举例
        
         ```shell
         <parameter ID="1">
           <mode>IVFPQ</mode>
           <dim>96</dim>
           <nlist>16384</nlist>
           <pq>24</pq>
         </parameter>
         <parameter ID="2">
           <mode>Flat</mode>
           <dim>512</dim>
         </parameter>
         <parameter ID="3">
           <mode>PCAR</mode>
           <input_dim>512</input_dim>
           <output_dim>128</output_dim>
         </parameter>
         <parameter ID="4">
           <mode>IVFSQ8</mode>
           <dim>64</dim>
           <nlist>1024</nlist>
         </parameter>
         <parameter ID="5">
           <mode>SQ8</mode>
           <dim>128</dim>
         </parameter>
         ```

- **步骤 5**   生成的算子模型文件被保存在当前目录`op_models`文件夹，用户需要将算子模型文件拷贝/移动到源码目录下的`modelpath`目录下。

  ```shell
  mv op_models/* ../modelpath
  ```

  > 如果用户在后续的开发中，采用python脚本编译生成新的算子om文件，或修改para_table.xml文件生成新的算子om文件，则需要将这些om文件移动到modelpath目录，然后重新执行[2.6 FeatureRetrieval安装](#2.6 FeatureRetrieval安装)中[步骤5](#2.6 FeatureRetrieval安装)的环境部署操作。

## 4.3 Intel CPU加速方案（可选）

如果服务器配置的是Intel CPU，需要安装MKL加速库（Intel数学核心函数库），以加快训练速度。如果服务器配置的是其他CPU，则跳过此章节。

**MKL加速库的具体安装步骤如下：**

- **步骤 1** 从[官网](https://software.intel.com/en-us/mkl/choose-download/linux)下载`l_mkl_x.x.x.tgz`的安装包压缩文件

  > `l_mkl_x.x.x.tgz`安装包名称中的x.x.x为安装包版本号，以获取的实际包名为准

- **步骤 2** 以**root**用户登录服务器

- **步骤 3** 将步骤1中获取的安装包压缩文件上传到服务器操作系统后台任意目录（如`/home`）

- **步骤 4** 在存放压缩包文件的目录下（如`/home`）执行以下命令解压安装包，并执行cd命令进入解压后的目录

  ```shell
  tar zxvf l_mkl_x.x.x.tgz
  ```

- **步骤 5** 执行以下命令安装MKL加速库

  ```shell
  ./install.sh
  ```

  安装过程中采用默认配置，默认安装至`/opt/intel`目录下，如果用户需要安装到自定义目录，在安装过程中请输入自定义目录

- **步骤 6**   配置MKL的相关环境变量（详细说明可以参见[链接](https://stackoverflow.com/questions/17821530/executable-cannot-find-dynamically-linked-mkl-library-but-ldd-does)

  执行`vi ~/.bashrc`命令打开操作系统中当前用户的环境配置文件，并在该文件的结尾添加如下两行内容，然后执行`:wq`命令保存文件：

  ```shell
  export intel_dir=/opt/intel
  source ${intel_dir}/mkl/bin/mklvars.sh intel64
  ```

  添加完成后，执行`source ~/.bashrc`使之生效

  > `export intel_dir=/opt/intel`命令中的`“/opt/intel”`目录为默认安装目录，如果用户安装到自定义目录，那需要修改为自定义目录。

## 4.4 接口介绍

FeatureRetrieval提供建库、查询、删库等接口，部分接口详细介绍如下表格。

### 4.4.1 公共接口

#### 4.4.1.1 Index::train

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void Index::train(idx_t n, const float*  x)                  |
| 功能描述 | 根据训练集进行L1聚类中心训练和PQ量化中心训练，小库Flat、Int8算法无需调用该接口。 |
| 输入     | idx_t n：训练集中特征向量的条数  const float* x：特征向量数据 |
| 输出     | 无                                                           |
| 返回值   | 无                                                           |
| 约束说明 | 训练采用k-means进行聚类，训练集比较小可能会影响查询精度。    |

#### 4.4.1.2 Index::add

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void Index::add(Index::idx_t n, const float *x)              |
| 功能描述 | 实现建库和往底库中添加新的特征向量的功能。                   |
| 输入     | idx_t n：建库的底库的特征向量条数或要添加的特征向量条数  const float *x：特征向量数据 |
| 输出     | 无                                                           |
| 返回值   | 无                                                           |
| 约束说明 | 添加底库条数无限制，但是要保证特征向量数据内存连续的；调用add接口前必须先调用train接口。 |

#### 4.4.1.3 Index::add_with_ids

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void Index::add_with_ids(Index::idx_t n, const float \*x， const Index::idx_t \*ids) |
| 功能描述 | 实现建库和往底库中添加新的特征向量的功能，添加时，底库特征都有对应的ID。 |
| 输入     | Index::idx_tn：建库的底库的特征向量条数或要添加的特征向量条数  const float \*x：特征向量数据，const Index::idx_t \*ids：底库特征向量对应的特征ID |
| 输出     | 无                                                           |
| 返回值   | 无                                                           |
| 约束说明 | 添加底库条数无限制，但是要保证特征向量数据内存连续的；调用add接口前必须先调用train接口。 |

#### 4.4.1.4 Index::search

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void Index::search(idx_t n, const float  *x, idx_t k, float *distances, idx_t *labels) |
| 功能描述 | 特征向量查询接口，根据输入的特征向量返回最相似的k条特征的ID。 |
| 输入     | idx_t n：查询的特征向量的条数  const float *x：特征向量数据  idx_t k：需要返回的最近似的结果的个数 |
| 输出     | float *distances：查询向量与距离最近的前k个向量间的距离值  idx_t *labels：查询的距离最近的前k个向量的ID |
| 返回值   | 无                                                           |
| 约束说明 | 无                                                           |

#### 4.4.1.5 Index::remove_ids

| Name     | Description                                                |
| -------- | ---------------------------------------------------------- |
| API定义  | void Index::remove_ids(IDSelector &  sel)                  |
| 功能描述 | 删除底库中指定的特征向量的接口。                           |
| 输入     | IDSelector & sel：将要删除的索引编号                       |
| 输出     | 无                                                         |
| 返回值   | 无                                                         |
| 约束说明 | 支持指定索引的底库向量删除和一定范围的索引的底库向量删除。 |

#### 4.4.1.6 faiss::Index * index_ascend_to_cpu

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | faiss::Index \*index_ascend_to_cpu(const  faiss::Index *ascend_index) |
| 功能描述 | 把Ascend平台的Index和底库clone到CPU。                        |
| 输入     | ascend_index：AscendIndex算法索引                            |
| 输出     | 无                                                           |
| 返回值   | cpu index：cpu索引                                           |
| 约束说明 | 接口主要实现把Ascend平台算法资源拷贝到CPU，避免底库添加时进行重复训练和建库操作。 |

#### 4.4.1.7 faiss::Index * index_cpu_to_ascend

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | faiss::Index *  index_cpu_to_ascend(std::initializer_list\<int\> devices, const faiss::Index  *index, const AscendClonerOptions *options = nullptr) <br>faiss::Index *  index_cpu_to_ascend(std::vector\<int\> devices,const faiss::Index *index,  const AscendClonerOptions *options = nullptr) |
| 功能描述 | 把CPU Index资源类拷贝到Ascend平台，实现Index类恢复到Ascend平台。 |
| 输入     | std::initializer_list\<int\> devices：存储ascend芯片名称的列表<br>const faiss::Index *index：cpu index<br>const AscendClonerOptions *options =  nullptr：ascend克隆选项(可选) |
| 输出     | 无                                                           |
| 返回值   | ascend index：ascend 索引                                    |
| 约束说明 | 从本地保存的CPU  Index类恢复并clone到Ascend平台，在恢复时可避免底库添加时进行二次训练和建库操作。对于小库算法（Flat/SQ/Int8Flat）来说，使用该接口导入index_ascend_to_cpu生成的index文件时，只支持单device，多device场景可能出现特征存储变化的情况，因为小库算法在device侧不存储特征id，且在入库过程中存在数据分片传输和均分到多个device，因此相比原始数据的特征存储顺序，导入后可能会发生变化 |

#### 4.4.1.8 Index::reset

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void index::reset()                                          |
| 功能描述 | 对于大库检索而言清空所有倒排索引和底库数据，保留L1粗聚类信息和PQ量化信息。对于小库检索而言清空所有底库索引数据和相关信息。 |
| 输入     | 无                                                           |
| 输出     | 无                                                           |
| 返回值   | 无                                                           |
| 约束说明 | 无                                                           |

### 4.4.2 大库检索接口

#### 4.4.2.1 Index::setNumProbes

| Name     | Description                           |
| -------- | ------------------------------------- |
| API定义  | void index::setNumProbes(int nprobes) |
| 功能描述 | 设置每次检索时要查询的list数          |
| 输入     | int nprobes：设置的list probe数       |
| 输出     | 无                                    |
| 返回值   | 无                                    |
| 约束说明 | 无                                    |

#### 4.4.2.2 Index::getNumSubQuantizers

| Name     | Description                               |
| -------- | ----------------------------------------- |
| API定义  | int index::getNumSubQuantizers()          |
| 功能描述 | 获取采用的子量化器的数目，即PQ分段中心数M |
| 输入     | 无                                        |
| 输出     | 无                                        |
| 返回值   | 分段中心数M的大小                         |
| 约束说明 | 无                                        |

#### 4.4.2.3 Index::getBitsPerCode

| Name     | Description                 |
| -------- | --------------------------- |
| API定义  | int index::getBitsPerCode() |
| 功能描述 | 获取PQ分段向量的编码长度    |
| 输入     | 无                          |
| 输出     | 无                          |
| 返回值   | PQ分段向量的编码长度        |
| 约束说明 | 无                          |

#### 4.4.2.4  Index::getListLength

| Name     | Description                                             |
| -------- | ------------------------------------------------------- |
| API定义  | uint32_t index::getListLength(int listId, int deviceId) |
| 功能描述 | 获取指定device和list的向量条数， debug接口。            |
| 输入     | int listId：倒排表的id；int deviceId：device的id        |
| 输出     | 无                                                      |
| 返回值   | uint32_t类型的向量条数                                  |
| 约束说明 | 无                                                      |

#### 4.4.2.5 Index::getListCodesAndIds

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void index::getListCodesAndIds(int listId, int deviceId, std::vector<uint8_t>& codes, std::vector<uint32_t>& ids) |
| 功能描述 | 获取指定device和list的编码后的向量和向量的id，debug接口。    |
| 输入     | int listId：倒排表的id <br>int deviceId：device的id；        |
| 输出     | std::vector<uint8_t>& codes：获取的编码后的向量； <br>std::vector<uint32_t>& ids：获取的编码后的向量的id |
| 返回值   | 无                                                           |
| 约束说明 | 无                                                           |

#### 4.4.2.6 Index::reserveMemory

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void index::reserveMemory(size_t numVecs)                    |
| 功能描述 | 在建立底库前为底库申请预留内存，使用此接口可提高建库速度。   |
| 输入     | size_t numVecs：申请预留内存的底库数量；                     |
| 输出     | 无                                                           |
| 返回值   | 无                                                           |
| 约束说明 | 只支持IVFSQ算法                                              |

#### 4.4.2.7 Index::reclaimMemory

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | size_t index::reclaimMemory()                                |
| 功能描述 | 在保证底库数量不变的情况下，缩减底库占用的内存。             |
| 输入     | 无                                                           |
| 输出     | 无                                                           |
| 返回值   | 缩减的内存大小，单位为Byte                                   |
| 约束说明 | 只支持IVFSQ算法                                              |

#### 4.4.2.8 AscendIndexPreTransform::prependTransform

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void AscendIndexPreTransform::prependTransform\<faiss::ascend::AscendPCAMatrix\>(int dimIn, int dimOut, float eigenPower, bool randomRotation) |
| 功能描述 | 添加降维算法。                                               |
| 输入     | int dimIn: 原始维度；<br>int dimOut：降维输出维度；<br>float eigenPower：降维白化参数；<br>bool randomRotation：是否需要旋转降维矩阵 |
| 输出     | 无                                                           |
| 返回值   | 无                                                           |
| 约束说明 | 只支持PCA/PCAR降维算法和IVFSQ算法组合                        |

#### 4.4.2.9 faiss::ascend::AscendNNDimReduction

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | AscendNNDimReduction(std::vector<int> deviceList, int dimIn, int dimOut, int batchSize, std::string &modelPath) |
| 功能描述 | 神经网络降维算法。                                           |
| 输入     | std::vector<int> deviceList：推理调用的芯片ID；<br>int dimIn：降维前的维度；<br>int dimOut：降维后的维度；<br>int batchSize：模型一次推理的条数，和OM模型匹配；<br>std::string &modelPath：OM模型在device侧的路径 |
| 输出     | 无                                                           |
| 返回值   | 无                                                           |
| 约束说明 | 无                                                           |

#### 4.4.2.10 AscendNNDimReduction::infer

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void AscendNNDimReduction::infer(int n, const float *inputData, std::vector<float> &outputData) |
| 功能描述 | 神经网络降维模型推理。                                       |
| 输入     | int n：特征向量条数；<br/>const float *inputData：降维前特征向量数据；<br/>std::vector<float> &outputData：降维后特征向量数据； |
| 输出     | 无                                                           |
| 返回值   | 无                                                           |
| 约束说明 | 无                                                           |

### 4.4.3 小库检索接口

#### 4.4.3.1 Index::getBase

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void getBase(int deviceId, std::vector\<float>& xb) const    |
| 功能描述 | 实现获取添加到device上所有的长特征向量的功能（需要预先获取底库的大小并分配内存） |
| 输入     | int deviceId：芯片ID                                         |
| 输出     | std::vector\<float>& xb：特征底库数据                        |
| 返回值   | 无                                                           |
| 约束说明 | 只支持Flat算法                                               |

#### 4.4.3.2 Index::getBase

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void getBase(int deviceId, std::vector\<uint8_t>& xb) const  |
| 功能描述 | 实现获取添加到device上所有的长特征向量的功能（需要预先获取底库的大小并分配内存） |
| 输入     | int deviceId：芯片ID                                         |
| 输出     | std::vector\<uint8_t>& xb：特征底库数据                      |
| 返回值   | 无                                                           |
| 约束说明 | 只支持SQ算法                                                 |

#### 4.4.3.3 Index::getBaseSize

| Name     | Description                                        |
| -------- | -------------------------------------------------- |
| API定义  | size_t getBaseSize(int deviceId) const             |
| 功能描述 | 实现获取添加到device上所有的长特征向量条数的功能。 |
| 输入     | int deviceId：芯片ID                               |
| 输出     | 无                                                 |
| 返回值   | 指定device上的长特征向量条数                       |
| 约束说明 | 无                                                 |

#### 4.4.3.4 Index::getIdxMap

| Name     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| API定义  | void getIdxMap(int deviceId, std::vector\<Index::idx_t>& idxMap) const |
| 功能描述 | 实现获取device上的向量的索引的功能。                         |
| 输入     | int deviceId：芯片ID                                         |
| 输出     | std::vector\<Index::idx_t>& idxMap：获取的当前device中向量的索引 |
| 返回值   | 无                                                           |
| 约束说明 | 无                                                           |