set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
#
set(tools $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/toolkit/toolchain/hcc)
set(CMAKE_SYSROOT ${tools}/sysroot)
set(CMAKE_C_COMPILER ${tools}/bin/aarch64-target-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${tools}/bin/aarch64-target-linux-gnu-g++)
set(CMAKE_AR ${tools}/bin/aarch64-target-linux-gnu-ar)
set(CMAKE_RANLIB ${tools}/bin/aarch64-target-linux-gnu-ranlib)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
