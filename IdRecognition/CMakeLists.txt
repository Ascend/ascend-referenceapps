cmake_minimum_required(VERSION 3.5.1)
project(IDRecognition)

# Check environment variable
if(NOT DEFINED ENV{ASCEND_HOME})
    message(FATAL_ERROR "please define environment variable:ASCEND_HOME")
endif()

if(NOT DEFINED ENV{ASCEND_VERSION})
    message(FATAL_ERROR "please define environment variable:ASCEND_VERSION")  
endif()

if(NOT DEFINED ENV{ARCH_PATTERN})
    message(FATAL_ERROR "please define environment variable:ARCH_PATTERN")  
endif()

add_definitions(-DENABLE_DVPP_INTERFACE)
add_definitions(-DASCEND_FACE_USE_ACL)

#include
include_directories(
        /usr/include/
        ./src/AscendBase/src/Base
        ./src/AscendBase/src/Base/Framework
        ./src/Common
        ./src/Common/Module
        /usr/local/opencv/include/opencv4
)

#lib link
link_directories(
        #centos
        /usr/lib
        /usr/local/opencv/lib64
        /usr/local/lib64/
        /usr/lib64/

        #ubuntu
        /usr/local/opencv/lib
)

add_subdirectory(src/HostCPU)



