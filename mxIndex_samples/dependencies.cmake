SET(MXINDEX_HOME /home/mxIndex/             CACHE STRING "")
SET(FAISS_HOME /usr/local/faiss/faiss1.7.1  CACHE STRING "")
SET(GTEST_HOME /usr/local/gtest             CACHE STRING "")
SET(DRIVER_HOME /usr/local/Ascend/driver/   CACHE STRING "")
SET(OPENBLAS /opt/OpenBLAS/                 CACHE STRING "")


IF (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  ADD_COMPILE_OPTIONS(-fPIE -fstack-protector-all -fPIC -Wall -O3 -D_FORTIFY_SOURCE=2 -frename-registers -fpeel-loops -fopenmp -march=armv8.2-a -mabi=lp64)
ELSE ()
  ADD_COMPILE_OPTIONS(-fPIE -fstack-protector-all -fPIC -Wall -O3 -D_FORTIFY_SOURCE=2 -frename-registers -fpeel-loops -fopenmp)
ENDIF ()

ADD_LINK_OPTIONS(-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -s -pie -fopenmp)

# Print key configurations
MESSAGE("-- Key Configurations: ")
MESSAGE(NOTICE "   MXINDEX_HOME is ${MXINDEX_HOME}")
MESSAGE(NOTICE "   FAISS_HOME is ${FAISS_HOME}")
MESSAGE(NOTICE "   GTEST_HOME is ${GTEST_HOME}")
MESSAGE(NOTICE "   DRIVER_HOME is ${DRIVER_HOME}")
MESSAGE(NOTICE "   OPENBLAS is ${OPENBLAS}")

INCLUDE_DIRECTORIES(
        ${MXINDEX_HOME}/include
        ${FAISS_HOME}/include
        ${GTEST_HOME}/include
        ${OPENBLAS}/include
        ${DRIVER_HOME}/include/dvpp
        /usr/local/include

)

LINK_DIRECTORIES(
        ${OPENBLAS}/lib
        ${MXINDEX_HOME}/host/lib   #需要使用标准态下面的so
        ${FAISS_HOME}/lib
        ${GTEST_HOME}/lib
        ${DRIVER_HOME}/lib64
        /usr/local/lib
)

LINK_LIBRARIES(
        faiss
        ascendfaiss
        gtest
        openblas
)




