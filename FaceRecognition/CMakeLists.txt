# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
cmake_minimum_required(VERSION 3.5.1)
project(FaceRecognition)

add_compile_options(-std=c++11 -fPIE -fstack-protector-all -Wall)

set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH ${PROJECT_SRC_ROOT}/cmake)

# Check environment variable
if(NOT DEFINED ENV{ASCEND_HOME})
    message(FATAL_ERROR "please define environment variable:ASCEND_HOME")  
endif()

set(DRIVER_HOME "")
if(NOT DEFINED ENV{DRIVER_HOME})
    set(DRIVER_HOME $ENV{ASCEND_HOME})
else()
    set(DRIVER_HOME $ENV{DRIVER_HOME})
endif()

if(NOT DEFINED ENV{ASCEND_VERSION})
    message(FATAL_ERROR "please define environment variable:ASCEND_VERSION")  
endif()

if(NOT DEFINED ENV{ARCH_PATTERN})
    message(FATAL_ERROR "please define environment variable:ARCH_PATTERN")  
endif()

set(CommonFolder ${PROJECT_SRC_ROOT}/Common)
set(AscendBaseFolder ${PROJECT_SRC_ROOT}/AscendBase)

set(CtrlAscendAcllibFolder /usr/local/AscendMiniOs/acllib)
set(CtrlOpenCVFolder /opt/aarch64/opencv)
set(CtrlProtobufFolder /opt/aarch64/protobuf)

set(HostOpenCVFolder /usr/local/opencv)
set(HostProtobufFolder /usr/local/protobuf)
set(HostUWebSocketsFolder /usr/local/uWebSockets)
set(HostJsonFolder /usr/local/nlohmann)
set(FFMPEG_PATH /usr/local/ffmpeg)


if(EXISTS ${HostProtobufFolder})
    set(PROTOBUF_PROTOC_EXE ${HostProtobufFolder}/bin/protoc)
    set(DATA_TRANS_PATH ${CommonFolder}/DataTrans)
    set(PROTO_PATH ${CommonFolder}/Proto)
    execute_process(COMMAND ${PROTOBUF_PROTOC_EXE} --proto_path=${PROTO_PATH} --cpp_out=${PROTO_PATH} ${PROTO_PATH}/vehicle.proto
                    COMMAND ${PROTOBUF_PROTOC_EXE} --proto_path=${DATA_TRANS_PATH} --cpp_out=${DATA_TRANS_PATH} ${DATA_TRANS_PATH}/DataTrans.proto
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(DEFINED ENABLE_CTRL_CPU)
    add_subdirectory(CtrlCPU/Device)
    add_subdirectory(CtrlCPU/Host)
else()

if(NOT DEFINED ENABLE_WEB_SERVER)
    add_subdirectory(HostCPU)
else()
    add_subdirectory(HostCPUWeb)
endif()

endif()



if(NOT DEFINED ENABLE_WEB_SERVER)
    set(ENABLE_WEB_SERVER OFF)
endif()
