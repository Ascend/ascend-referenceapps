project(HostCPU)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../dist")
set(LIBRARY_OUTPUT_PATH "../../../output")

add_compile_options(-std=c++11 -fPIE -fstack-protector-all -O2 -Wall)#-Werror
add_definitions(-DASCEND_FACE_USE_ACL_DVPP -DASCEND_FACE_USE_ACL -DHOST_CPU_SOLUTION)

# Skip build rpath
set(CMAKE_SKIP_BUILD_RPATH True)
#include
include_directories(
        $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/acllib/include
        ${DRIVER_HOME}/driver/include/dvpp/
        ../Common
)

link_directories(
        $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/acllib/lib64/stub
        ${DRIVER_HOME}/driver/lib64
        $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/atc/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib/
)

file(GLOB_RECURSE SRCS_H
        # add cpp to ascendbasehost
        ../AscendBase/src/Base/CommandParser/CommandParser.cpp
        ../AscendBase/src/Base/ConfigParser/ConfigParser.cpp
        ../AscendBase/src/Base/DvppCommon/DvppCommon.cpp
        ../AscendBase/src/Base/ErrorCode/ErrorCode.cpp
        ../AscendBase/src/Base/FileManager/FileManager.cpp
        ../AscendBase/src/Base/Framework/ModelProcess/ModelProcess.cpp
        ../AscendBase/src/Base/Log/Log.cpp
        ../AscendBase/src/Base/ResourceManager/ResourceManager.cpp
        ../AscendBase/src/Base/Statistic/Statistic.cpp
        ../Common/Module/*.cpp
        ../Common/SystemManager/*.cpp
        ../Common/FaceFeatureLib/*.cpp
)

# add other server src files here
set(HOST_SRCS
        ./HostMain.cpp
        #add other cpp here
)
add_executable(AclIDRcgHostCPU  ${HOST_SRCS} ${SRCS_H})
target_link_libraries(AclIDRcgHostCPU
        # add libs for host
        opencv_world
        acl_dvpp
        ascendcl
        pthread
        z
        ssl
        -Wl,-z,relro,-z,now,-z,noexecstack -pie -s
)
