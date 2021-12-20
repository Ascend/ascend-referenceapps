# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
cmake_minimum_required(VERSION 3.5.1)
project(FeatureRetrieval)

OPTION(ENABLE_TEST "build test cases" OFF)
add_compile_options(-fPIE -fstack-protector-all -fPIC -Wall)

set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH ${PROJECT_SRC_ROOT}/cmake)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SRC_ROOT})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SRC_ROOT})

# check environment variable
set(ASCEND_HOME /usr/local/Ascend)
if(DEFINED ENV{ASCEND_HOME})
    set(ASCEND_HOME $ENV{ASCEND_HOME})
endif()

set(DRIVER_HOME ${ASCEND_HOME})
if(DEFINED ENV{DRIVER_HOME})
    set(DRIVER_HOME $ENV{DRIVER_HOME})
endif()

set(ASCEND_VERSION ascend-toolkit/latest)
if(DEFINED ENV{ASCEND_VERSION})
    set(ASCEND_VERSION $ENV{ASCEND_VERSION})
endif()

set(ARCH_PATTERN "")
if(DEFINED ENV{ARCH_PATTERN})
    set(ARCH_PATTERN $ENV{ARCH_PATTERN})
endif()

set(ASCEND_MINIOS_HOME /usr/local/AscendMiniOs)
if(DEFINED ENV{ASCEND_MINIOS_HOME})
    set(ASCEND_MINIOS_HOME $ENV{ASCEND_MINIOS_HOME})
endif()

set(PROTOBUF_HOME /usr/local/protobuf)
if(DEFINED ENV{PROTOBUF_HOME})
    set(PROTOBUF_HOME $ENV{PROTOBUF_HOME})
endif()

set(PROTOBUF_AARCH64_HOME /opt/aarch64/protobuf)
if(DEFINED ENV{PROTOBUF_AARCH64_HOME})
    set(PROTOBUF_AARCH64_HOME $ENV{PROTOBUF_AARCH64_HOME})
endif()

set(FAISS_HOME /usr/local)
if(DEFINED ENV{FAISS_HOME})
    set(FAISS_HOME $ENV{FAISS_HOME})
endif()

set(CommonFolder ${PROJECT_SRC_ROOT}/common)
if(EXISTS ${PROTOBUF_HOME})
    set(PROTOBUF_PROTOC_EXE ${PROTOBUF_HOME}/bin/protoc)
    set(PROTO_PATH ${CommonFolder})
    execute_process(COMMAND ${PROTOBUF_PROTOC_EXE} --proto_path=${PROTO_PATH} --cpp_out=${PROTO_PATH} ${PROTO_PATH}/AscendIndex.proto
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OPENMP NOT FOUND")
endif()

add_subdirectory(ascend)
add_subdirectory(ascenddaemon)
