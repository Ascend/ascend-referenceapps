# project information
project(HostMain)

add_definitions(-DENABLE_DVPP_INTERFACE)
add_definitions(-DASCEND_FACE_USE_ACL)

# Skip build rpath
set(CMAKE_SKIP_BUILD_RPATH True)

# set the output path of the executable file
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SRC_ROOT}/dist/HostCPU)

# Find ascendbase
set(ASCEND_BASE_DIR ${AscendBaseFolder}/src/Base)
get_filename_component(ASCEND_BASE_ABS_DIR ${ASCEND_BASE_DIR} ABSOLUTE)

file(GLOB_RECURSE ASCEND_BASE_SRC_FILES
    ${ASCEND_BASE_ABS_DIR}/BlockingQueue/*cpp
    ${ASCEND_BASE_ABS_DIR}/CommandParser/*cpp
    ${ASCEND_BASE_ABS_DIR}/ConfigParser/*cpp
    ${ASCEND_BASE_ABS_DIR}/ErrorCode/*cpp
    ${ASCEND_BASE_ABS_DIR}/Log/*cpp
    ${ASCEND_BASE_ABS_DIR}/AsynLog/*cpp
    ${ASCEND_BASE_ABS_DIR}/StringEx/*cpp
    ${ASCEND_BASE_ABS_DIR}/FileManager/*cpp
    ${ASCEND_BASE_ABS_DIR}/DvppCommon/*cpp
    ${ASCEND_BASE_ABS_DIR}/Statistic/*cpp
    ${ASCEND_BASE_ABS_DIR}/Framework/ModelProcess/*cpp
    ${ASCEND_BASE_ABS_DIR}/ResourceManager/*cpp
    )

set(ASCEND_CV_DIR  ${CommonFolder}/AscendCV)
## set the ascendbase cpp file as ASCEND_CV_SRC_FILE
file(GLOB_RECURSE ASCEND_CV_SRC_FILE 
    ${ASCEND_CV_DIR}/FaceRecognition/Module/ModuleBase/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/StreamPuller/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/VideoDecoderHost/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/FaceDetection/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/FaceResize/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/MOTEmbedding/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/MOTConnection/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/QualityEvaluation/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/WarpAffine/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/VideoResize/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/FaceAttribute/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/FaceFeature/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/FaceSearch/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/ImageReader/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/ImageDecoder/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/FaceLandmark/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/FaceStock/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/FrameAlign/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/FaceDetailInfo/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/Module/RegResultHandler/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/FaceFeatureLib/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/ModuleFactory/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/ModelResource/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/FrameCache/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/StreamCache/*.cpp
    ${ASCEND_CV_DIR}/FaceRecognition/MemoryPool/*.cpp
    )

# set the AscendCV header file as ASCENDCV_INC_DIR
set(ASCEND_CV_INC_DIR 
        ${ASCEND_CV_DIR}/FaceRecognition
        ${ASCEND_CV_DIR}/FaceRecognition/Module)

# set the HostCPU cpp file as HOST_CPU_SRC_FILE
file(GLOB_RECURSE HOST_CPU_SRC_FILE 
        ${PROJECT_SRC_ROOT}/HostCPU/HostMain.cpp
        ${PROJECT_SRC_ROOT}/Common/SystemManager/*.cpp
        ${PROJECT_SRC_ROOT}/Common/DataTrans/*.cc
        ${PROJECT_SRC_ROOT}/Common/TestCV/*.cpp
        ${PROJECT_SRC_ROOT}/Common/FileEx/*.cpp
        ${PROJECT_SRC_ROOT}/Common/RegistApi/*.cpp
        ${PROJECT_SRC_ROOT}/Common/DeliveryChannel/*.cpp
        ${PROJECT_SRC_ROOT}/Common/ChannelStatus/*.cpp
        )


file(GLOB_RECURSE HOST_PROTO_SRC_FILE
    ${PROJECT_SRC_ROOT}/Common/Proto/*.cc)


# set the HostCPU header file as HOST_CPU_INC_DIR
set(HOST_CPU_INC_DIR  
        ${PROJECT_SRC_ROOT}/CtrlCPU/Host
        ${PROJECT_SRC_ROOT}/Common/)

find_package(FFMPEG REQUIRED)

# set the header file directory
## set the header file directory of acllib as ACL_INC_DIR
set(ACL_INC_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/acllib/include)  
## set all header file directory as HEADER_FILE_DIR
set(HEADER_FILE_DIR
    ${ACL_INC_DIR}
    ${ASCEND_BASE_DIR}
    ${FFMPEG_INCLUDE_DIRS}
    ${ASCEND_CV_INC_DIR}
    ${HOST_CPU_INC_DIR}
    ${HostOpenCVFolder}/include/opencv4
    ${HostProtobufFolder}/include
    ${PROJECT_SRC_ROOT}/Common/)

# set the share library directory
set(ACL_LIB_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/acllib/lib64)  
## set all share library directory as SHARED_LIB_DIR
set(SHARED_LIB_DIR
    ${ACL_LIB_DIR}
    ${HostOpenCVFolder}/lib
    ${HostOpenCVFolder}/lib64
    ${HostProtobufFolder}/lib
    )

# set the source file for target
set(SOURCE_FILE
    ${ASCEND_BASE_SRC_FILES}
    ${HOST_CPU_SRC_FILE}
    ${ASCEND_CV_SRC_FILE}
    ${HOST_PROTO_SRC_FILE}
    )

# Add link config
## Specify the directory of the indirect share library which are depend by libascendcl.so
SET(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath-link,${ACL_LIB_DIR}")

# Set the target executable file
link_directories(${SHARED_LIB_DIR})
add_executable(AclFaceRecognitionHostCPU ${SOURCE_FILE})
target_include_directories(AclFaceRecognitionHostCPU PUBLIC ${HEADER_FILE_DIR} )

target_link_libraries(AclFaceRecognitionHostCPU
    ascendcl
    acl_dvpp
    pthread
    protobuf
    opencv_world
    z
    ssl
    ${FFMPEG_LIBRARIES} -Wl,-z,relro,-z,now,-z,noexecstack -pie -s)

add_custom_command(TARGET AclFaceRecognitionHostCPU POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/model
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SRC_ROOT}/data/config ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config
        )
