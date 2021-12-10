[TOC]

# 1 Introduction to FeatureRetrieval

## 1.1 Application Scenarios

In application scenarios such as facial recognition and human body recognition, a neural network is usually used to extract in-depth features of a face or a human body, to obtain a semi-structured feature vector，similarity search for feature vectors can achieve the retrieval of specific targets. For example，after the feature vectors extracted from the image library are saved to the base library, the same neural network also extracts the feature vectors from target images, and compares them with the feature vectors in the base library. If the similarity is high, the system considers that the target has been found in the base library. 

Considering the retrieval precision, calculation, and storage overheads, the length of the feature vector is generally 512 dimensions (mostly float32). Based on the base library quantity, search libraries are classified into the small library (exact search) and large library (approximate search). Generally, the number of images in the small library ranges from 300,000 to 1,000,000, and the number of images in the large library can reach tens of millions or even hundreds of millions. 

The similarity retrieval framework used more in the industry is [Faiss](https://github.com/facebookresearch/faiss), big library search of FeatureRetrieval is based on Faiss framework, applying IVFPQ/IVFSQ/IVFINT8 algorithm on Ascend platform, including acceleration operators developed by TBE. The small library search inherits the FeatureRetrieval search framework to implement the Flat/SQ/INT8 algorithm for brute-force retrieval. The main implementation mode of the big and small library search (FaissAscend) is C++, with TBE operators. The optional Python interface is also provided, and Linux ARM and x86_64 platforms are supported.

## 1.2 Main Functions

Large and small library search provides functions such as feature vector query, base library addition, and base library deletion. For details, see Table 1-1.

Table 1-1 Main functions

| Function Name                          | Function Description                     |
| -------------------------------------- | ---------------------------------------- |
| Feature vector query                   | Based on the feature vector to be queried, calculates the distance between the feature vector to be queried and the vector in the base library, and returns the top K results with the highest similarity. |
| Library creation/Base library addition | Library creation is to add a base library with millions of or tens of millions of users to the Ascend platform. Base library addition is to add new feature data to the Ascend platform based on the existing base library. |
| Base library deletion                  | Deletes a feature vector corresponding to a specified index. Multiple feature vectors can be deleted. |
| Base library storage                   | Locally stores the features and indexes of the user base library. During service restoration, training and library creation are not required. |
| Base library restoration               | Restores the saved feature library and indexes to the Ascend platform. |

# 2 Installation Guide

## 2.1 Environment Dependencies

The installation of FeatureRetrieval depends on **Ascend Computing Language (ACL, Open Form)**, [Faiss](https://github.com/facebookresearch/faiss.git), and [Protocol Buffers (Protobuf)](https://github.com/protocolbuffers/protobuf). Before installing FeatureRetrieval, ensure that the dependencies have been correctly installed.

> As an API programming interface, the ACL provides C++ API libraries, such as device management, context management, stream management, memory management, model loading and execution, operator loading and execution, and media data processing, for users to develop applications. Large and small library search depends on an open compilation and operating environment. Therefore, use the Open Form ACLlib installation package. For details about how to obtain packages and install the environment, see *CANN <version> Software Installation Guide (Open Form)*.
>
> List of ACL packages required for FeatureRetrieval:
>
> | simple name                     | full name of packages                                        | default installation path                                    |
> | ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
> | CANN                            | Ascend-cann-toolkit\_{version}\_linux-{arch}.run             | /usr/local/Ascend/ascend-toolkit/latest（atc/opp/toolkit used） |
> | driver                          | A300-3000-npu-driver\_{version}\_{os}-aarch64.run / A300-3010-npu-driver\_{version}\_{os}-x86_64.run | /usr/local/Ascend                                            |
> | firmware                        | A300-3000-npufirmware\_{version}.run / A300-3010-npufirmware\_{version}.run | /usr/local/Ascend                                            |
> | Open Form ACL                   | Ascend-acllib-<version>-minios.aarch64.run                   | /usr/local/AscendMiniOs(development state)<br>/usr/local/AscendMiniOSRun(running state) |
> | Open Form driver source package | Ascend310-driver-<version>-minios.aarch64-src.tar.gz         | used after decompression during install FeatureRetrieval     |
>
> > Ascend-acllib-<version>-minios.aarch64.run and Ascend310-driver-<version>-minios.aarch64-src.tar.gz are extracted from Ascend-cann-device-sdk\_<version>\_linux-aarch64.zip.
> >
> > The minios.aarch64 installation package must be installed in both the development and running states. FeatureRetrieval compilation depends on the open-form ACLlib (development state) and deployment depends on the open-form ACLlib (running state).

Table 2-1 Hardware requirements

| Environment Requirements       | Description                                   |
| ------------------------------ | --------------------------------------------- |
| Supported Hardware Environment | Atlas 800 (Model 3000/3010)，A500PRO，200 SOC |
| OS                             | CentOS 7.6 / Ubuntu 18.04                     |

Table 2-2 Installed software and software versions required by the environment

| Software Name | Software Version                   |
| ------------- | ---------------------------------- |
| GCC           | 4.8.5/7.3.0, etc.                  |
| ACL           | 20.1.0, 20.2.0, 21.1.0 (Open Form) |
| Python        | 3.7.5                              |
| NumPy         | 1.16.0 or later                    |

Table 2-3 Software to be installed and software versions required by the environment

| Software Name | Software Version |
| ------------- | ---------------- |
| Swig          | 3.0.12           |
| OpenBLAS      | 0.3.9            |
| Faiss         | 1.6.1            |
| Protobuf      | 3.9.0            |
| googletest    | 1.8.1            |

You should make sure that the following components are installed:

```shell
pkg-config autoconf automake autotools-dev m4 gfortran
# sudo apt install pkg-config autoconf automake autotools-dev m4 gfortran
```

Faiss depends `blas` and `lapack`, we install `OpenBLAS`(which contains `blas` and `lapack`), testcases based on `googletest`.

python interface of Faiss depends `numpy`，`SWIG` used during compile python interface.

FeatureRetrieval depends `Faiss` and `protobuf`, testcases based on `googletest`.

## 2.2 OpenBLAS Installation

1. Download the OpenBLAS v0.3.9 source code package and decompress it.

   ```shell
   wget https://github.com/xianyi/OpenBLAS/archive/v0.3.9.tar.gz -O OpenBLAS-0.3.9.tar.gz
   tar -xf OpenBLAS-0.3.9.tar.gz
   ```

2. Go to the OpenBLAS directory.

   ```shell
   cd OpenBLAS-0.3.9
   ```

3. Perform compilation and installation.

   ```shell
   make FC=gfortran USE_OPENMP=1 -j
   # default indtall OpenBLAS in /opt
   make install
   
   # specify installation path
   #make PREFIX=/your_install_path install
   ```

4. configuration library path

   ```shell
   ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so
   
   # configuration /etc/profile
   vim /etc/profile
   # add [ export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH ] in /etc/profile
   source /etc/profile
   ```

## 2.3 Faiss Installation

Before installing the Faiss, install the OpenBLAS and create the soft link file `/usr/lib/libopenblas.so` first.

If installed on ARM platform, you need to [modify Faiss source code](#4.1.7 Faiss compilation issue) before compiling and installing.  

- **Step 1** Log in to the server operation background (host) as the **root** user.

- **Step 2** Download the `FeatureRetrieval` source code to any directory, for example, `/home/HwHiAiUser`. Go to the `ascendfaiss` source code directory.

  ```shell
  # now in /home/HwHiAiUser directory
  mkdir FeatureRetrieval
  tar xf Ascend-featureretrieval_<version>_src.tar.gz -C FeatureRetrieval
  
  # enter ascendfaiss work directory
  cd FeatureRetrieval/src/ascendfaiss
  ```

- **Step 3** Compile and install Faiss.

  1. Go to the third-party dependency directory.

     ```shell
     cd third_party
     ```

  2. Run the following commands to download the Faiss source code package and decompress it:

     ```shell
     wget https://github.com/facebookresearch/faiss/archive/v1.6.1.tar.gz
     tar -xf v1.6.1.tar.gz
     ```

  3. Go to the Faiss directory.

     ```shell
     cd faiss-1.6.1
     ```

  4. Run the following command to compile and configure the Faiss:

     ```shell
     # No Python interface is required.
     ./configure --without-cuda
     
     # The Python interface is required. /usr/bin/python3.7 is the path of the Python interpreter file. You can change it based on true installation path.
     PYTHON=/usr/bin/python3.7 ./configure --without-cuda
     ```

     > - If you need to use GPUs, change `--without-cuda` to `--with-cuda=<cuda install path>`, where `<cuda install path>` indicates the CUDA installation path. In addition, add the `--with-cuda-arch=<cuda compilation parameter >` to the command, for example, `--with-cuda-arch="-gencode=arch=compute_61,code=sm_61"`.

  5. Perform compilation and installation.

     ```shell
     make -j <thread num> # <thread num> indicates the number of CPU cores.
     make install
     ```

  6. Compile python interface.(Optional)

     Optional Steps. Configure the Python environment variable first. Otherwise, skip this step. Before `make` python interface, install SWIG by userself. [Working principles](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md#user-content-the-python-interface) of SWIG in faiss.

     ```shell
     make -C python
     make -C python install
     ```
     > Here we install faiss in PYTHON path, file is placed in `<bin>` and `<lib/python3.7/site-packages>` in PYTHON path.

  7. Configuration library path and return to the upper-level directory.

     If `Faiss` dynamic linked by procedure, path of the `Faiss` dynamic library needs to be known. You need to add the `Faiss` library directory to the  environment variable `LD_LIBRARY_PATH`.
     
     ```shell
     # configuration /etc/profile
     vim /etc/profile
     # add [ export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ] in /etc/profile
     # if Faiss installed in other directory, replace /usr/local with true install path
     source /etc/profile
     
     cd ..
     ```

## 2.4 Protobuf Installation

Current work directory is `FeatureRetrieval/src/ascendfaiss/third_party`.

  1. Download the Protobuf v3.9.0 source code package and decompress it.

     ```shell
     wget https://github.com/protocolbuffers/protobuf/releases/download/v3.9.0/protobuf-all-3.9.0.tar.gz
     tar -xf protobuf-all-3.9.0.tar.gz
     ```

  2. Go to the Protobuf directory.

     ```shell
     cd protobuf-3.9.0
     ```

  3. Perform compilation and installation.

     Installing the protobuf binary library and dependent header files required on the host.

     ```shell
     ./configure --prefix=/usr/local/protobuf
     make -j <thread num> # <thread num> indicates the number of CPU cores.
     make install
     make clean
     ```

     Installing the protobuf binary library required on the device.

     ```shell
     # hcc_path is installing-path of cross compiler within Ascend toolkit, represent ${hcc_path} with true installing-path
     hcc_path=/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc
     
     CC=${hcc_path}/bin/aarch64-target-linux-gnu-gcc CXX=${hcc_path}/bin/aarch64-target-linux-gnu-g++ CXXFLAGS=-fPIC \
     ./configure --prefix=/opt/aarch64/protobuf --host=aarch64-linux --disable-protoc
     make -j<thread num> # <thread num> is compiling thread number
     make install
     make clean
     ```
     
  4. Configuration library path and return to the upper-level directory.

     If `protobuf` dynamic linked by procedure, path of the `protobuf` dynamic library needs to be known. You need to add the `protobuf` library directory to the  environment variable `LD_LIBRARY_PATH`.
     
     ```shell
     # configuration /etc/profile
     vim /etc/profile
     # add [ export LD_LIBRARY_PATH=/usr/local/protobuf/lib:$LD_LIBRARY_PATH ] in /etc/profile
     source /etc/profile
     
     cd ..
     ```

## 2.5 googletest Installation

Current work directory is `FeatureRetrieval/src/ascendfaiss/third_party`.

1. Download the googletest v1.8.1 source code package and decompress it.

   ```shell
   wget https://github.com/google/googletest/archive/release-1.8.1.tar.gz -O googletest-release-1.8.1.tar.gz
   tar -xf googletest-release-1.8.1.tar.gz
   ```

2. move the googletest directory.

   ```shell
   mv googletest-release-1.8.1/googletest googletest
   ```

3. Here we need not perform compilation and installation, compiled during next step.

4. Return to upper-level directory and end depend libraries installation.

   ```
   cd ..
   ```

## 2.6 FeatureRetrieval Installation

Current work directory is `FeatureRetrieval/src/ascendfaiss`.

**Note:** If the ACL package is not installed according to the default installation path listed in [2.1 Environment Dependencies](#2.1 Environment Dependencies), change the installation path of ACL package or change the related path in the `ascendfaiss/acinclude/fa_check_ascend.m4` and `ascendfaiss/acinclude/fa_check_faiss.m4` file to the actual installation path. The following table lists the default values of path variables for ACL/3rd packages.

| variable name        | ACL/3rd packges name  | default installation path                  |
| -------------------- | --------------------- | ------------------------------------------ |
| with_ascend          | CANN(atc/opp/toolkit) | /usr/local/Ascend/ascend-toolkit/latest    |
| with_ascendminios    | Open Form ACLlib      | /usr/local/AscendMiniOs(development state) |
| with_ascenddriver    | driver/firmware       | /usr/local/Ascend                          |
| with_protobuf        | protobuf for host     | /usr/local/protobuf                        |
| with_protobufaarch64 | protobuf for device   | /opt/aarch64/protobuf                      |
| with_faiss           | faiss                 | /usr/local                                 |

Perform the following steps **1 to 4** to compile and install the FeatureRetrieval source code and TBE operator. You can run the `build.sh` script in the `src/ascendfaiss` directory of the source code to install in **one-click mode**. 

```shell
bash build.sh
```

Step **5** is to pack the executable file and TBE operator model file generated during compilation and deploy them to the device file system. You need to perform step 5 separately.

- **Step 1** Log in to the server operation background (host) as the **root** user.

- **Step 2** Compile the FeatureRetrieval library for subsequent use.

  1. Run the `cd /home/HwHiAiUser/FeatureRetrieval/src/ascendfaiss` command to go to the `ascendfaiss` source code directory.

  2. Run the following command to generate a configuration script (ignore the Makefile.am error):

     ```shell
     ./autogen.sh
     ```

  3. Run the following command to generate the `makefile.inc` configuration file for `Makefile`:

     ```shell
     ./configure
     ```

     You can also run the following commands to complete the configuration:

     > - `./configure --with-ascend=<ascend-acllib-path>`
     >
     >   `<ascend-acllib-path>` indicates the installation path of `acllib/atc/opp/toolkit`, default is `/usr/local/Ascend/ascend-toolkit/latest`.
     >
     > - `./configure --with-ascendminios=<ascend-minios-path>`
     >
     >   `<ascend-minios-path>` indicates the installation path of open-form acllib(development state), default is `/usr/local/AscendMiniOs`.
     >
     > - `./configure --with-ascenddriver=<ascend-driver-path>`
     >
     >   `<ascend-driver-path>` indicates the installation path of driver, default is `/usr/local/Ascend`.
     >
     > - `./configure --with-faiss=<faiss-path>`
     >
     >   `<faiss-path>` indicates the installation path of Faiss. Run this command to use the header file and dynamic link library of Faiss.
     >
     > - `./configure --with-protobuf=<protobuf-path>`
     >
     >   `<protobuf-path>` indicates the installation path of protobuf. Run this command to specify protobuf location.
     >
     > - `./configure --with-protobufaarch64=<protobuf-aarch64-path>`
     >
     >   `<protobuf-aarch64-path>` indicates the installation path of protobuf aarch64. Run this command to specify protobuf aarch64 location.
     >
     > - `PYTHON=/path/to/python3.7 ./configure`
     >
     >   `/path/to/python3.7` indicates the full path of Python 3.7 interpreter. Run this command to use the Python interface.

  4. Run the following commands to perform compilation:

     ```shell
     export PKG_CONFIG_PATH=/usr/local/protobuf/lib/pkgconfig
     make -j <thread num>
     ```
     
     > `<thread num>` indicates the number of CPU cores.
  
- **Step 3** Perform installation.

  1. Install the FeatureRetrieval header file and library.

     ```shell
     make install  # install file in faiss installation directory
     ```

  2. Compile the Python interface.(Optional)

     Before performing this step, ensure that the environment variable `PYTHON` has been set before running `./configure`.

     ```shell
     # make sure set PYTHON before run ./configure: PYTHON=/path/to/python3.7 ./configure
     make -C python
     ```
     > If faiss library not found during make, please check `FAISS_INSTALL_PATH ` in `python/Makefile`

  3. Install the Python interface.(Optional)

     ```shell
     make -C python install
     ```

- **Step 4** Compile the operator.

  FeatureRetrieval depends on custom operators. Before deployment, perform operator compilation and single-operator model conversion by referring to [Appendix 4.2 Custom Operators](#4.2.2 FeatureRetrieval Operator Compilation and Deployment).

- **Step 5** Prepare the neural network dimensionality reduction model.(Optional)

  The dimensionality reduction function (self-encoder) can be realized through neural network. The user needs to train the neural network model first, then convert the model into an OM model adapted to the Ascend platform, and put the OM model in the `modelpath` directory.

- **Step 6** Deploy the environment.

  Run the following command to deploy the environment:

  ```shell
  bash install.sh <driver-untar-path>
  ```
  

`<driver-untar-path>` is the directory where the `Ascend310-driver-{software version}-minios.aarch64-src.tar.gz` file is decompressed. For example, if the file is decompressed in the `/usr/local/software/` directory, `<driver-untar-path>` is `/usr/local/software/`. This command is used to distribute daemon process files to multiple devices. After this command is run, the file system of the Ascend device is modified. Therefore, you need to run the **`reboot`** command for the modification to take effect.

**Note**: If the open-form acllib (minios.aarch64, running state) is not installed in the default path `/usr/local/AscendMiniOSRun`, please modify the `acllib_minios_path` variable in install.sh file, set this parameter to **the directory where the Open Form ACLlib(running state) is located**.

## 2.7 Adaptation of 200SOC

For details to prepare the 200SOC operating environment, see the *Atlas 200 DK <version> User Guide*. This section assumes that the 200SOC operating environment has been prepared.

First compile the FeatureRetrieval code (refer to [2.6 FeatureRetrieval Installation](#2.6 FeatureRetrieval Installation)) and device testcases (refer to [3.4 Run testcases](#3.4 Run testcases)) in  Atlas 300 (Model 3000/3010) environment.

**Run TestIndexFlat on the device as an example, the specific steps are as follows**

- **Step 1** Copy the modelpath directory and test/ascenddaemon/TestIndexFlat to the 200SOC operating environment, modelpath and TestIndexFlat need to be in the same directory on 200SOC.

- **Step 2** Create soft connection in 200SOC operating environment.

  ```shell
  # Log in to the 200SOC operating environment by ssh
  su - root
  mkdir -p /lib64
  ln -sf /lib/ld-linux-aarch64.so.1 /lib64/ld-linux-aarch64.so.1
  ```

- **Step 3** Run the testcase.

  ```shell
  # cd to the directory where TestIndexFlat and modelpath are located
  export LD_LIBRARY_PATH=/home/HwHiAiUser/Ascend/acllib/lib64:$LD_LIBRARY_PATH
  ./TestIndexFlat
  ```

  > `/home/HwHiAiUser/Ascend` is the installation directory of the ACLlib library in CANN.

# 3 Usage Guide

## 3.1 Constraints

This section describes the constraints on the performance specifications of large and small library search.

Table 3-1 Constraints on large library performance specifications

| Item                                                      | Specifications                                               |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| Base library size                                         | <= 100 million                                               |
| Feature vector dimension                                  | Multiple of 16                                               |
| Number of PQ quantization trajectory centers              | 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128                 |
| Number of L1 centroids(nlist parameter in Index)          | Size of the IVF inverted list. The number of train data must be greater than or equal to this value. |
| IVFPQ code length                                         | 8 bits (that is, the number of trajectory cluster centers is 256) |
| Similarity measurement method                             | L2 (Euclidean distance)                                      |
| Principal Component Analysis for Dimensionality Reduction | Both the original and reduced dimensions are multiple of 16  |
| IVFSQ code type                                           | Only support SQ8                                             |

Table 3-2 Constraints on small library performance specifications

| Item                          | Specifications                                               |
| ----------------------------- | ------------------------------------------------------------ |
| Base library size             | for single device should less than 2 million (512 dimensions) |
| Feature vector dimension      | Multiple of 16                                               |
| Similarity measurement method | L2/IP (Euclidean distance, InnerProduct distance)            |
| SQ code type                  | Only support SQ8                                             |
| INT8 code type                | Only support INT8                                            |

## 3.2 C++ Interface Usage

After the operations in [2.6 FeatureRetrieval Installation](#2.6 FeatureRetrieval Installation) are complete, the static library file `libascendfaiss.a` and dynamic library file `libascendfaiss.so` are generated, the executable files of users only need to be linked to the static library file or dynamic library file. If the static library file is used, add `LDFLAGS` to Makefile. The header file must contain the following content:

```c++
#include <faiss/ascend/AscendIndexIVFPQ.h>
```

## 3.3 Python Interface Usage

To use FeatureRetrieval as a library reference in a Python project, you need to place the following three files in the project directory and add the project directory to the environment variable PYTHONPATH.

```c++
__init__.py
swig_ascendfaiss.py
_swig_ascendfaiss.so
```

If you need to use the FeatureRetrieval library, add the following code to the user code:

```python
import ascendfaiss
```

## 3.4 Run testcases

Go to `ascendfaiss` directory

```shell
make gtest   # compile gtest
make test_ascend   # compile and run host testcases
make test_daemon   # compile device testcases, copy to device to run
```

Use and modify feature retrieval algorithms by referring to test cases.

# 4 Appendixes

## 4.1 FAQs

### 4.1.1 Different configuration ratios of Ascend devices

**Symptom**

The search performance of FeatureRetrieval is poor.

**Solution**

You can use the DSMI API to set the ratio of AI CPUs to Ctrl CPUs of Ascend devices to 2:6, 4:4 (default), or 6:2. The ratio 2:6 can be used to achieve the optimal performance of FeatureRetrieval.

### 4.1.2 Abnormal permission of /home/HwHiAiUser

**Symptom**

If the following error message is displayed at the beginning of the program running, the permission of the `/home/HwHiAiUser` directory is abnormal.

```shell
[ERROR] HDC:2020-04-22-11:59:42.691.520 [drvPpcCreateDir:35] >>> create ppc dir failed, Permission denied (errno:13)
```

**Solution**

The user needs to be added to the **HwHiAiUser** user group, and the read and execute permissions of the `/home/HwHiAiUser` directory need to be granted to the **HwHiAiUser** user group.

```shell
chmod g+rx /home/HwHiAiUser
```
### 4.1.3 An error occurred when checking the environment compiler during configuration

**Symptom**

When the `PYTHON=/path/to/python3.7 ./configure` command is run, the following error message is displayed during the configuration check of **aarch64-target-linux-gnu-g++** or **aarch64-target-linux-gnu-ar**:

```shell
checking for unistd.h... yes
checking for aarch64-target-linux-gnu-g++... no
checking for aarch64-target-linux-gnu-ar... no
configure: error: Couldn't find aarch64-target-linux-gnu-g++
```

**Solution**

Please check installation path of `aarch64-target-linux-gnu-g++`, default path of  cross-compiler is `/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc`. If installed in other path, configure parent directory of `toolkit` in `--with-ascend`. 

### 4.1.4 An error occurred when checking the environment acl.h during configuration

**Symptom**

When the `./configure` command is run, the following error message is displayed, indicating that **acl.h** cannot be found:

```shell
checking for unistd.h... yes
checking for aarch64-target-linux-gnu-g++... /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
checking for aarch64-target-linux-gnu-ar... /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-ar
checking acl/acl.h usability... no
checking acl/acl.h presence... no
checking for acl/acl.h... no
configure: error: in `/home/heatureretrieval':
configure: error: Couldn't find acl.h
See `config.log' for more details
```

 **Solution**

The main cause is that the ACLlib(development state) package is incorrectly installed. Check whether the `/usr/local/AscendMiniOs/acllib/include/acl/` directory exists. If not, ensure that the open-form ACLlib package to be installed. Install the package in the `/usr/local/AscendMiniOs` folder by following the instructions provided in "Installing the Development and Operating Environments" in the *CANN <version> Software Installation Guide (Open Form)*.

If minios.aarch64 ACLLib(development state) has been installed in other paths，we can use arguments `--with-ascendminios=<ascend-minios-path>` during `./configure` to specify acllib minios.aarch64 installation path.

### 4.1.5 An error occurred when checking the environment ascend_hal.h during configuration

**Symptom**

When the `./configure` command is run, the following error message is displayed, indicating that **ascend_hal.h** cannot be found:

```shell
checking ascend_hal.h usability... no
checking ascend_hal.h presence... no
checking for ascend_hal.h... no
configure: error: in `/home/faiss/src/ascendfaiss':
configure: error: Couldn't find ascend_hal.h
```

 **Solution**

The main cause is that the ACL package is incorrectly installed. Check whether the `/usr/local/Ascend/driver/kernel/inc/driver` directory exists or whether the kernel folder exists in the `/usr/local/Ascend/driver` directory. If none of them exist, install the package in the `driver` package by following the instructions provided in "Development Environment Installation" in the *CANN <version> Software Installation Guide*.

If `driver` has been installed in other paths，we can use arguments `--with-ascenddriver=<ascend-driver-path>` during `./configure` to specify `driver` installation path.

### 4.1.6 Slow training speed of the ARM platform

**Symptom**

During the index training, it is found that the CPU usage is high but the training speed is slow. In this case, the OpenBlas deadlock may be triggered. The distance calculation of the Faiss CPU code uses the sgemm of OpenBlas. The training and library creation code of FeatureRetrieval also depends on the distance calculation code of the Faiss CPU.

**Solution**

1. When compiling and installing OpenBLAS, add the **USE_OPENMP=1** parameter during source code compilation, for example, `make FC=gfortran USE_OPENMP=1`. To avoid slow training speed and high CPU usage caused by deadlock, visit https://github.com/xianyi/OpenBLAS/issues/660.
2. If you do not need to reinstall OpenBlas, set the **OPENBLAS_NUM_THREADS** environment variable, for example, `export OPENBLAS_NUM_THREADS=4`. The specific data can be optimized based on the hardware configuration in the specific environment. Different values may result in different performance.

### 4.1.7 Faiss compilation issue

 This section is only suitable to ARM platform.

**Symptom**

In ARM platform, the old compiler version may result in some Faiss compilation errors.

For python, it's maybe:  `import faiss`, then `KeyError: 'flags'`. Source code needs to be modified in advance to avoid this problem. 

**Solution**

Use the following procedure to modify the configuration and source code:

1. **Only suit to GCC 4.8.5**. Change `ARCH_CPUFLAGS="-march=armv8.2-a"` in `acinclude/ax_check_cpu.m4` to `ARCH_CPUFLAGS="-march=armv8-a"`. After the change, run the **aclocal** and **autoconf** commands, and then run the **./configure** command. 

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

2. **Only suit to GCC 4.8.5**.Replace `vdups_laneq_f32` in `utils/distances_simd.cpp` with `vdups_lane_f32`. 

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

3. `/proc/cpuinfo` of the ARM platform does not contain `flags` field. Change `if "avx2" in numpy.distutils.cpuinfo.cpu.info[0]['flags']:` in `python/faiss.py` to `if "avx2" in numpy.distutils.cpuinfo.cpu.info[0].get('flags', ""):`.

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

### 4.1.8 Search performance deteriorates beyond expectations when the value of nprobe/batch increases

**Symptom**

Assume that the performance of nprobe/batch=64 is QPS=100. When other parameters remain unchanged and nprobe/batch is changed to 128, the QPS decreases to 20, which is a non-linear relationship.

**Solution**

The possible cause of the QPS decrease is that the memory resources pre-allocated by the Index on device are insufficient when nprobe/batch=128. The system uses aclrtmalloc to apply for memory resources again. The solution is to increase the size of the memory pool (128 MB by default). You can modify the parameter `resourceSize` of host Index to adjust the size of the memory pool(Unit is Byte, take IndexFlat as an example).

```c++
faiss::ascend::AscendIndexFlatConfig conf({ 0, 1, 2, 3 }, 512*1024*1024); // 512MB
// 或者使用 conf.resourceSize = 512 * 1024 * 1024;
```
### 4.1.9 Protobuf configurarion error

**Symptom**

When `make -j <thread num>` during installation of FeatureRetrieval, error message `No package 'protobuf' found ` appears.

```shell
pkg-config --cflags protobuf  # fails if protobuf is not installed
Package protobuf was not found in the pkg-config search path.
Perhaps you should add the directory containing `protobuf.pc'
to the PKG_CONFIG_PATH environment variable
No package 'protobuf' found
```

**Solution**

Configuration environment `PKG_CONFIG_PATH`:

```shell
export PKG_CONFIG_PATH=/usr/local/protobuf/lib/pkgconfig  # should be protobuf install lib path
```
### 4.1.10 Protobuf link error

**Symptom**

When `make test_ascend` in FeatureRetrieval, `undefined symbol` errors appears.

```shell
undefined symbol: _ZNF6google8protobuf7Message11GetTypeNameB5cxx11Ev
```

**Solution**

Set `library path` of protobuf installed in [chapter2.4](#2.4 Protobuf Installation) in head of environment variable `LD_LIBRARY_PATH`:

```shell
 # default installation path of libprotobuf.so is /usr/local/protobuf/lib
 export LD_LIBRARY_PATH=/usr/local/protobuf/lib:$LD_LIBRARY_PATH
```

### 4.1.11 Does FeatureRetrieval support multi-indices scenarios?

**Symptom**

Does FeatureRetrieval support single-device multi-indices scenarios? What's the maximum number of indices supported?

**Solution**

FeatureRetrieval allows multiple indices to be created on a single device. On the device side, each index shares a memory pool. The memory size is determined by the `resourceSize` parameter of the index. Therefore, in the multi-indices scenario, ensure `resourceSize` parameter values of all indices are the same. Otherwise, the retrieval performance will be affected. Ensure serial search among indices. The number of indices depends on the memory size of device and number of features every index. It is recommended that the number of indices be less than or equal to `100`. If the number of indices is too large, the creation fails or the base features cannot be added.


## 4.2 Custom Operators

### 4.2.1 Introduction to Custom Operators

FeatureRetrieval solution uses TBE operators to compute feature distance:

1. PQ distance accumulation operator: uses vector complex add (VSADD) for accumulation and multi-core calculation.
2. PQ distance calculation operator: obtains the L2 distance between the long feature library data and the long feature vector to-be-retrieved.
3. PQ distance table generation operator: is loaded to all Ascend 310 chips and generates the same distance table on each Ascend 310 chip.
4. PCA for dimensionality reduction operator: dimensionality reduction for long vectors.
5. Distance calculation operator for Flat: obtains the L2/IP distance between the feature library data and the feature vector to-be-retrieved.
6. Distance calculation operator for SQ8: obtains the L2/IP distance between the SQ quantized feature library data and the unquantized feature vector to-be-retrieved.
7. Distance calculation operator for INT8: obtains the L2/COS distance between the int8 quantized feature library data and the quantized feature vector to-be-retrieved.

### 4.2.2 FeatureRetrieval Operator Compilation and Deployment

- **Step 1** Log in to the server operation background (host) as the **HwHiAiUser** user and run the following command to go to the `FeatureRetrieval` source code directory:

  ```shell
  cd /home/HwHiAiUser/FeatureRetrieval/src/ascendfaiss
  ```
  Set `ASCEND_HOME` and `ASCEND_VERSION` environment variable before Operator Compilation. the default values are `/usr/local/Ascend` and `ascend-toolkit/latest`

  ```shell
  export ASCEND_HOME=/usr/local/Ascend          # Ascend home path
  export ASCEND_VERSION=ascend-toolkit/latest   # atc/opp/toolkit installation path
  ```

  > `ASCEND_HOME` represents installation path of `driver/ascend-toolkit`, `ASCEND_VERSION` represents current version of Ascend，if installation path of `atc` is `/usr/local/Ascend/ascend-toolkit/latest`, no need to set `ASCEND_HOME` and `ASCEND_VERSION`.

- **Step 2** Run the following command to go to the `ops` directory and compile the operator prototype and operator information:

  ```shell
  cd ops && bash ./build.sh
  ```

  > During the compilation, the system checks whether the installation directory of the opp operator package is the default directory. If it is not the default directory, you need to enter the installation directory of the opp operator package (`Ascend-opp-{software version}-{os.arch}.run`).

- **Step 3** Run the `cd..` command to return to the `ascend` source code directory.

- **Step 4** Generate a single-operator model.


  1. Go to the `tools` directory.

     ```shell
     cd tools
     ```

  2. Generate an operator model file in either of the following ways(single and batch):

     - − Generating operator model files in batches −

       Run the following command to obtain multiple groups of operator model files. You need to set the parameters in the `para_table.xml` file in the current directory. Table 4-6 lists the parameter values.

       ```shell
       python run_generate_model.py -m <mode>
       ```

       Table 4-6 Parameter description

       | Parameter | Description    |
       | --------- | -------------- |
       | <mode>    | algorithm type |

       > The value of `mode` can be `ALL`, `PCAR`, `Flat`, `IVFPQ`, `SQ8`,  `IVFSQ8`,`INT8` or `IVFINT8`. Multiple values are separated by commas (,).
       >
       > For example, `python run_generate_model.py -m PCAR,IVFSQ8`, default is `ALL`, just run `python run_generate_model.py`. 

     - − Generating a single operator model file  −  

       - −  IVFPQ  −

         Run the following command to obtain a group of operator model files (eight operator model files in total, including one distance accumulation operator model file, six distance calculation operator model files, and one distance table generation operator model file). You need to modify the parameters in the command based on the actual situation. Table 4-1 describes the parameters.

         ```shell
        python ivfpq_generate_model.py -d <feature dimension> -c <coarse centroid num> -s <sub quantizers num>
         ```

         Table 4-1 Parameter description
         
         | Parameter               | Description                          |
         | ----------------------- | ------------------------------------ |
         | `<feature dimension>`   | Feature vector dimension (D).        |
         | `<coarse centroid num>` | Number of L1 cluster centers.        |
         | `<sub quantizers num>`  | Number of PQ trajectory centers (M). |
         
         Note that the feature vector dimension D and the PQ trajectory center quantity M cannot be randomly configured because the dimension (that is, subdim) of the PQ trajectory vector is limited by operators.
         
         | Parameter      | Constraint Value                       |
         | -------------- | -------------------------------------- |
         | subdim (`D/M`) | 4, 8, 16, 32, 48, 64, 80, 96, 112, 128 |

       - −  Flat  −

         Run the following command to obtain thirty-two operator model files for distance computation. You need to modify the parameters in the command based on the actual situation. Table 4-2 describes the parameters.

         ```shell
         python flat_generate_model.py -d <feature dimension>
         ```

         Table 4-2 Parameter description

         | Parameter             | Description                   |
         | --------------------- | ----------------------------- |
         | `<feature dimension>` | Feature vector dimension (D). |

       - −  PCAR  −

         Run the following command to obtain nine operator model files for dimension reduction. You need to modify the parameters in the command based on the actual situation. Table 4-3 describes the parameters.

         ```shell
         python pcar_generate_model.py -i <input dimension> -o <output dimension>
         ```

         Table 4-3 Parameter description

         | Parameter            | Description                                                 |
         | -------------------- | ----------------------------------------------------------- |
         | `<input dimension>`  | Dimension of the feature vector before dimension reduction. |
         | `<output dimension>` | Dimension of the feature vector after dimension reduction.  |

       - −  SQ8  −

         Run the following command to obtain twenty-six operator model files for distance computation of SQ8. You need to modify the parameters in the command based on the actual situation. Table 4-4 describes the parameters.

         ```shell
         python sq8_generate_model.py -d <feature dimension>
         ```

         Table 4-4 Parameter description

         | Parameter             | Description                   |
         | --------------------- | ----------------------------- |
         | `<feature dimension>` | Feature vector dimension (D). |

       - −  IVFSQ8  −

         Run the following command to obtain a group of operator model files (nine operator model files in total, including two distance computation operator model file for SQ8, seven distance calculation operator model files). You need to modify the parameters in the command based on the actual situation. Table 4-5 describes the parameters.

         ```shell
         python ivfsq8_generate_model.py -d <feature dimension> -c <coarse centroid num>
         ```

         Table 4-5 Parameter description

         | Parameter               | Description                   |
         | ----------------------- | ----------------------------- |
         | `<feature dimension>`   | Feature vector dimension (D). |
         | `<coarse centroid num>` | Number of L1 cluster centers. |

       - −  IVFFlat  −
         Run the following command to obtain a group of operator model files(eight operator model files). You need to modify the parameters in the command based on the actual situation. Table 4-6 describes the parameters.
       
         ```shell
        python ivfflat_generate_model.py -d <feature dimension> -c <coarse centroid num>
         ```
         
         Table 4-6 Parameter description
         
         | Parameter               | Description                   |
         | ----------------------- | ----------------------------- |
         | `<feature dimension>`   | Feature vector dimension (D). |
         | `<coarse centroid num>` | Number of L1 cluster centers. |
         
         **Note:** Currently, ivfflat algorithm has performance problems and is not recommended. We optimize this retrieval algorithm in the near future.
         
       - −  INT8  −
         
         **The difference between INT8 and SQ8 is that INT8 is quantized externally, and the input feature of Index is int8 type. SQ8 is quantized by the index internal, type of input feature is float32.**
         
         Run the following command to obtain operator model files for distance computation of pure Int8 features(twenty-five operator model files). You need to modify the parameters in the command based on the actual situation. Table 4-7 describes the parameters.
         
         ```shell
         python int8flat_generate_model.py -d <feature dimension>
         ```
         
         Table 4-7 Parameter description
         
         | Parameter             | Description                   |
         | --------------------- | ----------------------------- |
         | `<feature dimension>` | Feature vector dimension (D). |
         
       - −  IVFINT8  −
       
         Run the following command to obtain a group of operator model files for distance computation of pure Int8 features(seventeen operator model files). You need to modify the parameters in the command based on the actual situation. Table 4-8 describes the parameters.
       
         ```shell
         python ivfint8flat_generate_model.py -d <feature dimension> -c <coarse centroid num>
         ```
       
         Table 4-8 Parameter description
       
         | Parameter               | Description                   |
         | ----------------------- | ----------------------------- |
         | `<feature dimension>`   | Feature vector dimension (D). |
         | `<coarse centroid num>` | Number of L1 cluster centers. |
       
       Figure 4-1 Example of operator model parameters file *para_table.xml*
       
       ```xml
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

- **Step 5** The operator model files generated are saved in the `op_models` folder in the current directory. You need to copy or move the operator model files to the `modelpath` directory in the source code directory.

  ```shell
  mv op_models/* ../modelpath
  ```

  > In subsequent development, if you use python script to compile and generate new operator om files or modify the para_table.xml file to generate new operator om files, you need to move the om files to **modelpath** directory. Then you need to perform the environment deployment operations of [Step5](#2.6 FeatureRetrieval Installation) in [2.6 FeatureRetrieval Installation](#2.6 FeatureRetrieval Installation).

## 4.3 Intel CPU Acceleration Solution (Optional)

If the server uses Intel CPUs, you need to install the Intel® Math Kernel Library (Intel® MKL) acceleration library to accelerate the training speed. If the server uses other CPUs, skip this section.

**Use the following procedure to install Intel® MKL:**

- **Step 1** Download the `l_mkl_x.x.x.tgz` installation package from the [official website](https://software.intel.com/en-us/mkl/choose-download/linux).

  > x.x.x in the `l_mkl_x.x.x.tgz` installation package name indicates the version of the installation package. Use the actual package name.

- **Step 2** Log in to the server as the **root** user.

- **Step 3** Upload the compressed installation package obtained in Step 1 to any directory (for example, `/home`) on the server OS.

- **Step 4** Run the following command in the directory (for example, `/home`) where the compressed package is stored to decompress the installation package and run the **cd** command to go to the directory where the decompressed package is stored:

  ```shell
  tar zxvf l_mkl_x.x.x.tgz
  ```

- **Step 5** Run the following command to install Intel® MKL:

  ```shell
  ./install.sh
  ```

  During the installation, the default configuration is retained. The default installation directory is `/opt/intel`. If you need to install the tool in a customized directory, specify the customized directory during the installation.

- **Step 6** Configure the environment variables of the MKL. For details, click [Link](https://stackoverflow.com/questions/17821530/executable-cannot-find-dynamically-linked-mkl-library-but-ldd-does).

  Run the `vi ~/.bashrc` command to open the environment configuration file of the current user in the OS, add the following two lines to the end of the file, and run the `:wq` command to save the file.

  ```shell
  export intel_dir=/opt/intel
  source ${intel_dir}/mkl/bin/mklvars.sh intel64
  ```

  Run the `source ~/.bashrc` command for the settings to take effect.

  > In the `export intel_dir=/opt/intel` command, `/opt/intel` is the default installation directory. If you want to install the tool in a customized directory, specify the customized directory during the installation.

## 4.4 API Description

FeatureRetrieval provides APIs for creating, querying, and deleting libraries. The following table describes some of the APIs.

### 4.4.1 Common APIs

#### 4.4.1.1 Index::train

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| API definition       | void Index::train(idx_t n, const float*  x)                  |
| Function description | This API is used to perform L1 cluster center training and PQ quantization center training based on the training set. The small library Flat/Int8 does not need to call this function. |
| Input                | idx_t n: number of feature vectors in the training set. const float* x: feature vector data. |
| Output               | None                                                         |
| Return value         | None                                                         |
| Constraint           | K-means is used for clustering. A small training set may affect the query precision. |

#### 4.4.1.2 Index::add

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| API definition       | void Index::add(Index::idx_t n, const float *x)              |
| Function description | This API is used to create a library and add feature vectors to the library. |
| Input                | idx_t n: number of feature vectors in the base library or to be added to the base library. const float *x: feature vector data. |
| Output               | None                                                         |
| Return value         | None                                                         |
| Constraint           | There is no limit on the number of base libraries to be added. However, the feature vector data in the memory must be continuous, train interface must be invoked before add interface is invoked. |

#### 4.4.1.3 Index::add_with_ids

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| API definition       | void Index::add_with_ids(Index::idx_t n, const float \*x, const Index::idx_t \*ids) |
| Function description | This API is used to create a library and add feature vectors to the library. Each feature in the library has an ID. |
| Input                | Index::idx_tn: number of feature vectors in the base library or to be added to the base library. const float \*x: feature vector data. const Index::idx_t \*ids: feature ID corresponding to the feature vector in the base library. |
| Output               | None                                                         |
| Return value         | None                                                         |
| Constraint           | There is no limit on the number of base libraries to be added. However, the feature vector data in the memory must be continuous. Only large library search supports this function, train interface must be invoked before add interface is invoked. |

#### 4.4.1.4 Index::search

| Name                 | Description                              |
| -------------------- | ---------------------------------------- |
| API definition       | void Index::search(idx_t n, const float  *x, idx_t k, float *distances, idx_t *labels) |
| Function description | This API is used to query feature vectors. The IDs of the k most similar features are returned based on the input feature vector. |
| Input                | idx_t n: number of queried feature vectors. const float *x: feature vector data. data idx_t k: number of the latest similar results to be returned. |
| Output               | float *distances: distance between the queried vector and the nearest first k vectors. idx_t *labels: ID of the nearest first k vectors. |
| Return value         | None                                     |
| Constraint           | None                                     |

#### 4.4.1.5 Index::remove_ids

| Name                 | Description                              |
| -------------------- | ---------------------------------------- |
| API definition       | void Index::remove_ids(IDSelector &  sel) |
| Function description | This API is used to delete a specified feature vector from the base library. |
| Input                | IDSelector & sel: ID of the index to be deleted. |
| Output               | None                                     |
| Return value         | None                                     |
| Constraint           | This API deletes only the base library vector of a specified index or a range of indexes. |

#### 4.4.1.6 faiss::Index * index_ascend_to_cpu

| Name                 | Description                              |
| -------------------- | ---------------------------------------- |
| API definition       | faiss::Index \*index_ascend_to_cpu(const  faiss::Index *ascend_index) |
| Function description | This API is used to clone the index and base library of the Ascend platform to a CPU. |
| Input                | ascend_index: AscendIndex algorithm index. |
| Output               | None                                     |
| Return value         | cpu index: CPU index.                    |
| Constraint           | This API is used to copy the algorithm resources of the Ascend platform to the CPU to avoid repeated training and library creation when a base library is added. |

#### 4.4.1.7 faiss::Index * index_cpu_to_ascend

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| API definition       | faiss::Index *  index_cpu_to_ascend(std::initializer_list\<int\> devices, const faiss::Index  *index, const AscendClonerOptions *options = nullptr) <br>faiss::Index *  index_cpu_to_ascend(std::vector\<int\> devices,const faiss::Index *index,  const AscendClonerOptions *options = nullptr) |
| Function description | This API is used to copy the CPU index resource class to the Ascend platform to restore the index class of the Ascend platform. |
| Input                | std::initializer_list\<int\> devices: list of Ascend chip names. <br>const faiss::Index *index：cpu index<br>const AscendClonerOptions *options =  nullptr: Ascend clone option (optional). |
| Output               | None                                                         |
| Return value         | ascend index: Ascend index.                                  |
| Constraint           | This API restores the CPU index class from the local host and clones it to the Ascend platform. During the restoration, secondary training and library creation operations are not required when a base library is added.For exact search algorithms (Flat/SQ/Int8Flat), when this API is used to import index files generated by index_ascend_to_cpu, only a single device is supported. In multi-device scenarios, feature storage position may be changed. Exact search algorithm does not store feature IDs on the device side, and data is transmitted in fragments and evenly distributed to multiple devices during import. Therefore, the feature storage sequence of original data may be changed after imported. |

#### 4.4.1.8 Index::reset

| Name                 | Description                              |
| -------------------- | ---------------------------------------- |
| API definition       | void index::reset()                      |
| Function description | This API is used to clear all inverted indexes and base database data, and retain the L1 coarse clustering information and PQ quantization information for large library, and it also be used to clear all index information of each vector for small library. |
| Input                | None                                     |
| Output               | None                                     |
| Return value         | None                                     |
| Constraint           | None                                     |

### 4.4.2 Large Library Search APIs

#### 4.4.2.1 Index::setNumProbes

| Name                 | Description                              |
| -------------------- | ---------------------------------------- |
| API definition       | void index::setNumProbes(*int* *nprobes*) |
| Function description | This API is used to set the number of lists to be queried during each search. |
| Input                | *int* *nprobes*: number of list probes.  |
| Output               | None                                     |
| Return value         | None                                     |
| Constraint           | None                                     |

#### 4.4.2.2 Index::getNumSubQuantizers

| Name                 | Description                              |
| -------------------- | ---------------------------------------- |
| API definition       | int index::getNumSubQuantizers()         |
| Function description | This API is used to obtain the number of used subquantizers, that is, the number of PQ trajectory centers (M). |
| Input                | None                                     |
| Output               | None                                     |
| Return value         | Number of trajectory centers (M)         |
| Constraint           | None                                     |

#### 4.4.2.3 Index::getBitsPerCode

| Name                 | Description                              |
| -------------------- | ---------------------------------------- |
| API definition       | int index::getBitsPerCode()              |
| Function description | This API is used to obtain the encoding length of the PQ trajectory vector. |
| Input                | None                                     |
| Output               | None                                     |
| Return value         | Encoding length of the PQ trajectory vector |
| Constraint           | None                                     |

#### 4.4.2.4  Index::getListLength

| Name                 | Description                              |
| -------------------- | ---------------------------------------- |
| API definition       | uint32_t index::getListLength(int listId, int deviceId) |
| Function description | This API is a debug API used to obtain the number of vectors of a specified device and list. |
| Input                | int listId: ID of the inverted list. int deviceId: device ID. |
| Output               | None                                     |
| Return value         | Number of vectors of the uint32_t type   |
| Constraint           | None                                     |

#### 4.4.2.5 Index::getListCodesAndIds

| Name                 | Description                              |
| -------------------- | ---------------------------------------- |
| API definition       | void index::getListCodesAndIds(int listId, int deviceId, std::vector<uint8_t>& *codes*, std::vector<uint32_t>& ids) |
| Function description | This API is a debug API used to obtain the encoded vector and vector ID of a specified device and list. |
| Input                | int listId: ID of the inverted list. <br>int deviceId: device ID. |
| Output               | std::vector<uint8_t>& codes: obtained encoded vector. <br>std::vector<uint32_t>& ids: ID of the obtained encoded vector. |
| Return value         | None                                     |
| Constraint           | None                                     |

#### 4.4.2.6 Index::reserveMemory

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| API definition       | void index::reserveMemory(size_t numVecs)                    |
| Function description | This API is used to reserve memory for the library before the library is created, which can be used to improve the feature added speed. |
| Input                | size_t numVecs: Number of features for which reserved memory is applied for. |
| Output               | None                                                         |
| Return value         | None                                                         |
| Constraint           | Only support IVFSQ                                           |

#### 4.4.2.7 Index::reclaimMemory

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| API definition       | size_t index::reclaimMemory()                                |
| Function description | Reduce the memory occupied by the base library while ensuring that the number of base libraries remains unchanged. |
| Input                | None                                                         |
| Output               | None                                                         |
| Return value         | reclaim memory size, unit is Byte                            |
| Constraint           | Only support IVFSQ                                           |

#### 4.4.2.8 AscendIndexPreTransform::prependTransform

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| API definition       | void AscendIndexPreTransform::prependTransform\<faiss::ascend::AscendPCAMatrix\>(int dimIn, int dimOut, float eigenPower, bool randomRotation) |
| Function description | Adding a dimensionality reduction algorithm.                 |
| Input                | int dimIn: dimension before reduction；<br>int dimOut: dimension after reduction；<br>float eigenPower: reduction whitening parameter；<br>bool randomRotation: whether the rotation dimension reduction matrix is required |
| Output               | None                                                         |
| Return value         | None                                                         |
| Constraint           | only support combination of PCA/PCAR and IVFSQ               |

#### 4.4.2.9 faiss::ascend::AscendNNDimReduction

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| API definition       | AscendNNDimReduction(std::vector<int> deviceList, int dimIn, int dimOut, int batchSize, std::string &modelPath) |
| Function description | Neural network dimensionality reduction algorithm.           |
| Input                | std::vector<int> deviceList: Inference chip ID；<br>int dimIn: Dimensions before dimensionality reduction；<br>int dimOut: Dimension after dimensionality reduction；<br>int batchSize: The number of inferences of the model at one time, matching the OM model；<br>std::string &modelPath: The path of the OM model on the device side |
| Output               | None                                                         |
| Return value         | None                                                         |
| Constraint           | None                                                         |

#### 4.4.2.10 AscendNNDimReduction::infer

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| API definition       | void AscendNNDimReduction::infer(int n, const float *inputData, std::vector<float> &outputData) |
| Function description | Neural network dimensionality reduction model inference。    |
| Input                | int n: Feature vector number；<br/>const float *inputData: Feature vector data before dimensionality reduction；<br/>std::vector<float> &outputData: Feature vector data after dimensionality reduction； |
| Output               | None                                                         |
| Return value         | None                                                         |
| Constraint           | None                                                         |

### 4.4.3 Small Library Search APIs

#### 4.4.3.1 Index::getBase

| Name                 | Description                              |
| -------------------- | ---------------------------------------- |
| API definition       | void getBase(int deviceId, std::vector<float>& xb) const |
| Function description | This API is used to obtain all long feature vectors added to the device. (The size of the base library needs to be obtained and memory needs to be allocated in advance.) |
| Input                | int deviceId: chip ID.                   |
| Output               | std::vector<float>& xb: feature base library data. |
| Return value         | None                                     |
| Constraint           | None                                     |

#### 4.4.3.2 Index::getBaseSize

| Name                 | Description                              |
| -------------------- | ---------------------------------------- |
| API definition       | size_t getBaseSize(int deviceId) const   |
| Function description | This API is used to obtain the number of long feature vectors added to the device. |
| Input                | int deviceId: chip ID.                   |
| Output               | None                                     |
| Return value         | Number of long feature vectors on a specified device. |
| Constraint           | None                                     |

#### 4.4.3.3 Index::getIdxMap

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| API definition       | void getIdxMap(int deviceId, std::vector\<Index::idx_t>& idxMap) const |
| Function description | This API is used to obtain the index of vectors in device.   |
| Input                | int deviceId：device ID                                      |
| Output               | std::vector\<Index::idx_t>& idxMap：index of obtain vector   |
| Return value         | None                                                         |
| Constraint           | None                                                         |

#### 4.4.3.4 Index::getIdxMap

| Name                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| API definition       | void getIdxMap(int deviceId, std::vector\<Index::idx_t>& idxMap) const |
| Function description | get index of vectors in current device                       |
| Input                | int deviceId：device id                                      |
| Output               | std::vector\<Index::idx_t>& idxMap: index of vectors in current device |
| Return value         | None                                                         |
| Constraint           | None                                                         |