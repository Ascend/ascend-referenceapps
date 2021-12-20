
set(TARGET_LIBRARY ascendfaiss)
set(TARGET_LIBRARY_STATIC ascendfaiss_static)

set(CMAKE_CXX_STANDARD 11)

set(ASCEND_HOST_DIR ${PROJECT_SRC_ROOT}/ascend)
file(GLOB_RECURSE ASCEND_SRC_FILE 
    ${ASCEND_HOST_DIR}/*.cpp
    ${ASCEND_HOST_DIR}/rpc/*.cpp
    ${ASCEND_HOST_DIR}/utils/*.cpp
    ${CommonFolder}/*.pb.cc
    ${CommonFolder}/*.cpp
)

file(GLOB ASCEND_SRC_HEADERS ${ASCEND_HOST_DIR}/*.h)
file(GLOB ASCEND_SRC_UTILS_HEADERS ${ASCEND_HOST_DIR}/utils/*.h)

include_directories(
    ${PROJECT_SRC_ROOT}
    ${CommonFolder}
    ${PROTOBUF_HOME}/include
    ${FAISS_HOME}/include
    ${DRIVER_HOME}/driver/include/dvpp
    ${DRIVER_HOME}/driver/kernel/inc/driver
)

link_directories(
    ${DRIVER_HOME}/driver/lib64
    ${PROTOBUF_HOME}/lib
    ${FAISS_HOME}/lib
)

add_library(${TARGET_LIBRARY} SHARED ${ASCEND_SRC_FILE})
target_link_libraries(${TARGET_LIBRARY} 
    protobuf faiss c_sec ascend_hal mmpa z dl rt
    -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -s)

add_library(${TARGET_LIBRARY_STATIC} STATIC ${ASCEND_SRC_FILE})
set_target_properties(${TARGET_LIBRARY_STATIC} PROPERTIES OUTPUT_NAME ${TARGET_LIBRARY})

INSTALL(TARGETS ${TARGET_LIBRARY} ${TARGET_LIBRARY_STATIC}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
INSTALL(FILES ${ASCEND_SRC_HEADERS} DESTINATION include/faiss/ascend)
INSTALL(FILES ${ASCEND_SRC_UTILS_HEADERS} DESTINATION include/faiss/ascend/utils)
