# 动态人脸Demo第三方软件编译指导

## 1. 标准形态软件依赖

| 依赖软件      | 版本   | 下载地址                                                     | 说明                                         |
| ------------- | ------ | ------------------------------------------------------------ | -------------------------------------------- |
| opencv        | 4.2.0  | [Link](https://github.com/opencv/opencv/releases)            | OpenCV的基本组件，用于图像的基本处理         |
| protobuf      | 3.11.2 | [Link](https://github.com/protocolbuffers/protobuf/releases/download/v3.11.2/protobuf-cpp-3.11.2.tar.gz) | 数据序列化反序列化组件。                     |
| ffmpeg        | 4.2.1  | [Link](https://github.com/FFmpeg/FFmpeg/archive/n4.2.1.tar.gz) | 视频转码解码组件                             |
| uWebSockets   | 0.14.8 | [Link](https://github.com/uNetworking/uWebSockets/archive/v0.14.8.tar.gz) | websocket中的一个组件，用于构建websocker服务 |
| nlohmann/json | 3.7.3  | [Link](https://github.com/nlohmann/json/releases/download/v3.7.3/json.hpp) | json库                                       |

**注意：**

- 其中```uWebSockets```,```nlohmann/json```两个软件仅仅web功能需要，如果不需要web功能，无需编译。

- 如在200RC上，编译第三方软件时间过长（超过3小时以上），可考虑从其他性能较好的arm环境编译好，再移到200RC的对应路径上。


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



### 1.1 OpenCV

对于x86环境下载完解压，按以下命令编译即可

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

### 1.2 Protobuf

下载完解压，按以下命令编译即可

```bash
./configure --prefix=/usr/local/protobuf
make -j
make install
```



### 1.3 FFmpeg

下载完解压，按以下命令编译即可

```bash
./configure --prefix=/usr/local/ffmpeg --enable-shared
make -j
make install
```



### 1.4 uWebSockets

下载完解压，按以下命令编译即可

```bash
make -j
make PREFIX=/usr/local/uWebSockets install
```



### 1.5 nlohmann/json

nlohmann/json 无需编译，将json.hpp文件放置到```/usr/local/nlohmann/include```即可
```
mkdir -p /usr/local/nlohmann/include
mv json.hpp /usr/local/nlohmann/include
```
