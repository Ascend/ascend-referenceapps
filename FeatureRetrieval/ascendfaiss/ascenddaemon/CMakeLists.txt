set(TARGET_DAEMON ascendfaissdaemon)
set(TARGET_DAEMON_STATIC ascendfaissdaemon_static)

include(${PROJECT_SRC_ROOT}/cmake/Ascend.cmake)

add_compile_options(-frename-registers -fpeel-loops)

set(CMAKE_CXX_STANDARD 14)
add_definitions(-DUSE_ACL_INTERFACE_V2)

set(ASCEND_DEVICE_DIR ${PROJECT_SRC_ROOT}/ascenddaemon)
file(GLOB_RECURSE ASCEND_DEVICE_SRC_FILE 
    ${ASCEND_DEVICE_DIR}/*.cpp
    ${ASCEND_DEVICE_DIR}/impl/*.cpp
    ${ASCEND_DEVICE_DIR}/rpc/*.cpp
    ${ASCEND_DEVICE_DIR}/utils/*.cpp
    ${CommonFolder}/*.pb.cc
    ${CommonFolder}/*.cpp
)

include_directories(
    ${PROJECT_SRC_ROOT}
    ${CommonFolder}
    ${PROTOBUF_AARCH64_HOME}/include
    ${ASCEND_MINIOS_HOME}/acllib/include
    ${DRIVER_HOME}/driver/include/dvpp
    ${DRIVER_HOME}/driver/kernel/inc/driver
)

link_directories(
    ${DRIVER_HOME}/develop/lib64
    ${ASCEND_MINIOS_HOME}/acllib/lib64
)

#lib link
set(RPATH /lib/aarch64-linux-gnu:${ASCEND_MINIOS_HOME}/acllib/lib64:${DRIVER_HOME}/develop/lib64:${DRIVER_HOME}/driver/lib64)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath-link,${RPATH}")

link_libraries(${PROTOBUF_AARCH64_HOME}/lib/libprotobuf.a)

add_library(${TARGET_DAEMON_STATIC} STATIC ${ASCEND_DEVICE_SRC_FILE})
set_target_properties(${TARGET_DAEMON_STATIC} PROPERTIES OUTPUT_NAME ${TARGET_DAEMON})

add_executable(${TARGET_DAEMON} ${ASCEND_DEVICE_SRC_FILE})
target_link_libraries(${TARGET_DAEMON}
    ${PROTOBUF_AARCH64_HOME}/lib/libprotobuf.a 
    c_sec ascendcl ascend_hal
    -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -s)

