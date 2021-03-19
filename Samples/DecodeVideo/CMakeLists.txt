# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
# Add definitions ENABLE_DVPP_INTERFACE to use dvpp api
add_definitions(-DENABLE_DVPP_INTERFACE)
# project information
project(DecodeVideo)

# Check environment variable
if(NOT DEFINED ENV{ASCEND_HOME})
    message(FATAL_ERROR "please define environment variable:ASCEND_HOME")
endif()

# Find ascendbase
set(ASCEND_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../AscendBase/Base)
get_filename_component(ASCEND_BASE_ABS_DIR ${ASCEND_BASE_DIR} ABSOLUTE)
file(GLOB_RECURSE ASCEND_BASE_SRC_FILES
    ${ASCEND_BASE_ABS_DIR}/CommandParser/*cpp
    ${ASCEND_BASE_ABS_DIR}/ConfigParser/*cpp
    ${ASCEND_BASE_ABS_DIR}/ErrorCode/*cpp
    ${ASCEND_BASE_ABS_DIR}/Log/*cpp
    ${ASCEND_BASE_ABS_DIR}/FileManager/*cpp
    ${ASCEND_BASE_ABS_DIR}/DvppCommon/*cpp)

# Compile options
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -Wall)

# Skip build rpath
set(CMAKE_SKIP_BUILD_RPATH True)

# Set FFMPEG include directory and library directory
if (NOT DEFINED ENV{FFMPEG_PATH})
    message(FATAL_ERROR "Please define FFMPEG_PATH first!")
else()
    #message("FFMPEG_PATH $ENV{FFMPEG_PATH}")
    set(FFMPEG_PATH $ENV{FFMPEG_PATH})
endif()

# Set include directory and library directory
set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR}/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SRC_ROOT}/dist)

# find FFMPEG
set(CMAKE_MODULE_PATH ${PROJECT_SRC_ROOT}/CMake)
find_package(FFMPEG REQUIRED)

set(ACL_INC_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/include)
set(ACL_LIB_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/lib64/stub)

# Header path
include_directories(${ACL_INC_DIR})
include_directories(${ASCEND_BASE_DIR})
include_directories(${ASCEND_BASE_DIR}/Framework)
include_directories(${PROJECT_SRC_ROOT}/include)
include_directories(${FFMPEG_INCLUDE_DIRS})

# Src path
aux_source_directory(${PROJECT_SRC_ROOT}/src SRC_FILES)
#aux_source_directory(${PROJECT_SRC_ROOT}/../Common/src SRC_FILES)

# add host lib path
link_directories(${ACL_LIB_DIR})
SET(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath-link,${ACL_LIB_DIR}")

add_executable(main ${SRC_FILES} ${ASCEND_BASE_SRC_FILES} ${PROJECT_SRC_ROOT}/main.cpp)
target_link_libraries(main ascendcl acl_dvpp ${FFMPEG_LIBRARIES} pthread -Wl,-z,relro,-z,now,-z,noexecstack -pie -s)