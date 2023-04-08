#!/bin/bash

#g++ -std=c++11 -march=armv8-a -fPIC -fstack-protector-all -Wno-sign-compare -D_FORTIFY_SOURCE=2 -O3 -Wall -Wextra -DFINTEGER=int -fopenmp -I../mxIndex/include -I/usr/local/include -I/usr/local/gtest/include -I/usr/local/Ascend/driver/include/dvpp/ -o TestAscendIndexSQ TestAscendIndexSQ.cpp -L../mxIndex/host/lib -L/usr/local/lib -L/usr/local/gtest/lib -L/usr/local/Ascend/driver/lib64 -lopenblas -lfaiss -lascendfaiss -lgtest

rm -rf  ./build
mkdir build
cd build
cmake ..
make
echo "build success,please check env of MX_INDEX_MODELPATH and LD_LIBRARY_PATH before excuting the binary file in directory of build"