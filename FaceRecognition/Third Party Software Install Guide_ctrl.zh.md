# 动态人脸Demo第三方软件编译指导

注意：部分第三方软件Device和Host程序均有依赖，需要分别编译

## 1. Device侧软件依赖

| 软件依赖 | 版本   | 下载地址                                                     | 说明                                   |
| -------- | ------ | ------------------------------------------------------------ | -------------------------------------- |
| opencv   | 4.2.0  | [Link](https://github.com/opencv/opencv/releases)            | OpenCV的基本组件，用于图像的基本处理。 |
| protobuf | 3.11.2 | [Link](https://github.com/protocolbuffers/protobuf/releases/download/v3.11.2/protobuf-cpp-3.11.2.tar.gz) | 数据序列化反序列化组件。               |

Device侧软件编译需要使用到交叉编译工具，该工具在```$ASCEND_HOME/ascend-toolkit/latest/toolkit/toolchain/hcc```里面

我们也提供cmake文件```cmake/Ascend.cmake```指向该交叉编译工具

### 1.1 OpenCV

下载完解压，按以下命令编译即可

注意：CMake版本需要3.14+

```bash
mkdir build
cd build
cmake \
-DCMAKE_TOOLCHAIN_FILE=<where you place facerecognition>/cmake/Ascend.cmake \
-DCMAKE_INSTALL_PREFIX=/opt/aarch64/opencv \
-DWITH_WEBP=OFF \
-DBUILD_opencv_world=ON ..
make -j
make install
```



### 1.2 Protobuf

下载完解压，按以下命令编译即可

```bash
CC=$ASCEND_HOME/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc \
CXX=$ASCEND_HOME/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ \
CXXFLAGS=-fPIC \
./configure \
--prefix=/opt/aarch64/protobuf \
--host=aarch64-linux \
--disable-protoc
make -j
make install
```





## 2. Host侧软件依赖

| 依赖软件      | 版本   | 下载地址                                                     | 说明                                         |
| ------------- | ------ | ------------------------------------------------------------ | -------------------------------------------- |
| opencv        | 4.2.0  | [Link](https://github.com/opencv/opencv/releases)            | OpenCV的基本组件，用于图像的基本处理         |
| protobuf      | 3.11.2 | [Link](https://github.com/protocolbuffers/protobuf/releases/download/v3.11.2/protobuf-cpp-3.11.2.tar.gz) | 数据序列化反序列化组件。                     |
| ffmpeg        | 4.2.1  | [Link](https://github.com/FFmpeg/FFmpeg/archive/n4.2.1.tar.gz) | 视频转码解码组件                             |
| uWebSockets   | 0.14.8 | [Link](https://github.com/uNetworking/uWebSockets/archive/v0.14.8.tar.gz) | websocket中的一个组件，用于构建websocker服务 |
| nlohmann/json | 3.7.3  | [Link](https://github.com/nlohmann/json/releases/download/v3.7.3/json.hpp) | json库                                       |

**注意：**其中```uWebSockets```,```nlohmann/json```两个软件仅仅web功能需要，如果不需要web功能，无需编译

Host侧库默认安装到/usr/local/下面，全部安装完成后，请设置环境变量

```bash
export PATH=/usr/local/protobuf/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/protobuf/lib:$LD_LIBRARY_PATH

export PATH=/usr/local/ffmpeg/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH

export PATH=/usr/local/opencv/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/opencv/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/opencv/lib:$LD_LIBRARY_PATH

export PATH=/usr/local/uWebSockets/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/uWebSockets/lib:$LD_LIBRARY_PATH
```



### 2.1 OpenCV

对于x86环境，下载完解压，按以下命令编译即可

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/opencv \
-DWITH_WEBP=OFF \
-DBUILD_opencv_world=ON ..
make -j8
make install
```

对于arm环境，下载完解压，按以下命令编译即可

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE \
-DENABLE_NEON=OFF \
-DCMAKE_INSTALL_PREFIX=/usr/local/opencv \
-DCMAKE_CXX_FLAGS="-march=armv8-a" \
-DWITH_WEBP=OFF \
-DBUILD_opencv_world=ON ..
make -j8
make install
```



### 2.2 Protobuf

下载完解压，按以下命令编译即可

```bash
./configure --prefix=/usr/local/protobuf
make -j
make install
```



### 2.3 FFmpeg

下载完解压，按以下命令编译即可

```bash
./configure --prefix=/usr/local/ffmpeg --enable-shared
make -j
make install
```



### 2.4 uWebSockets

下载完解压，按以下命令编译即可

```bash
make -j
make PREFIX=/usr/local/uWebSockets install
```



### 2.5 nlohmann/json

nlohmann/json 无需编译，将json.hpp文件放置到```/usr/local/nlohmann/include```即可
```
mkdir -p /usr/local/nlohmann/include
mv json.jpp /usr/local/nlohmann/include
```