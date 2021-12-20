# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
# Add definitions ENABLE_DVPP_INTERFACE to use dvpp api
add_definitions(-DENABLE_DVPP_INTERFACE)

# Add definitions USE_ACL_OP_EXECUTE_V2
if(DEFINED ENV{USE_ACL_OP_EXECUTE_V2})
add_definitions(-DUSE_ACL_OP_EXECUTE_V2)
endif()

# project information
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2")
project(InferClassification)

if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
    # Check environment variable
    if(NOT DEFINED ENV{ASCEND_HOME})
        # Set default include directory and library directory
        set(ACL_INC_DIR "C:/Program Files/HuaWei/Ascend/Acllib/include")
        set(ACL_LIB_DIR "C:/Program Files/HuaWei/Ascend/Acllib/lib64")
    else()
        set(ACL_INC_DIR $ENV{ASCEND_HOME}/Acllib/include)
        set(ACL_LIB_DIR $ENV{ASCEND_HOME}/Acllib/lib64)
    endif()
    # Set compile options
    add_compile_options(-std=c++11 /GS /SDL -Wall)
else()
    # Check environment variable
    if(NOT DEFINED ENV{ASCEND_HOME})
        message(FATAL_ERROR "please define environment variable:ASCEND_HOME")
    endif()
    set(ACL_INC_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/include)
    set(ACL_LIB_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/lib64/stub)
    # Set compile options
    add_compile_options(-std=c++11  -fPIE -fstack-protector-all -Wall)
endif()

# Find ascendbase
Set(ASCEND_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ascendbase/src/Base)
get_filename_component(ASCEND_BASE_ABS_DIR ${ASCEND_BASE_DIR} ABSOLUTE)
file(GLOB_RECURSE ASCEND_BASE_SRC_FILES
    ${ASCEND_BASE_ABS_DIR}/CommandParser/*cpp
    ${ASCEND_BASE_ABS_DIR}/ConfigParser/*cpp
    ${ASCEND_BASE_ABS_DIR}/ErrorCode/*cpp
    ${ASCEND_BASE_ABS_DIR}/Log/*cpp
    ${ASCEND_BASE_ABS_DIR}/AsynLog/*cpp
    ${ASCEND_BASE_ABS_DIR}/FileManager/*cpp
    ${ASCEND_BASE_ABS_DIR}/DvppCommon/*cpp
    ${ASCEND_BASE_ABS_DIR}/Framework/ModelProcess/*cpp
    ${ASCEND_BASE_ABS_DIR}/ResourceManager/*cpp
    ${ASCEND_BASE_ABS_DIR}/SingleOpProcess/*cpp)

# Skip build rpath
set(CMAKE_SKIP_BUILD_RPATH True)

# Set output directory
set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR}/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SRC_ROOT}/dist)

# Header path
include_directories(${ACL_INC_DIR})
include_directories(${ASCEND_BASE_DIR})
include_directories(${ASCEND_BASE_DIR}/Framework)

# Src path
aux_source_directory(${PROJECT_SRC_ROOT}/AclProcess SRC_FILES)

# add host lib path
link_directories(${ACL_LIB_DIR})

add_executable(main ${SRC_FILES} ${ASCEND_BASE_SRC_FILES} ${PROJECT_SRC_ROOT}/main.cpp)

if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
    set(CMAKE_MODULE_LINKER_FLAGS "/NXCOMPAT /DYNAMICBASE /SAFESEH")
    target_link_libraries(main libascendcl libacl_dvpp)
else ()
    target_link_libraries(main ascendcl acl_dvpp pthread -Wl,-z,relro,-z,now,-z,noexecstack -pie -s)
endif()
