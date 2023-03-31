#include "IndexILFlat.h"

#include <iostream>
#include <numeric>
#include <vector>

#include "acl/acl.h"
#include "arm_fp16.h"

APP_ERROR TestComputeDistance(ascend::IndexILFlat& index, int queryN, int baseSize, float16_t* queryData)
{
std::vector<float> distances(queryN * baseSize);
auto ret = index.ComputeDistance(queryN, queryData, distances.data());
return ret;
}

APP_ERROR TestSearchByThreshold(ascend::IndexILFlat& index, int queryN, float16_t* queryData)
{
int topK = 10;
float threshold = 0.6;
std::vector<int> num(queryN);
std::vector<float> distances(queryN * topK);
std::vector<ascend::idx_t> idxs(queryN * topK);
auto ret = index.SearchByThreshold(queryN, queryData, topK, threshold,
num.data(), idxs.data(), distances.data());
return ret;
}

int main(int argc, char **argv)
{
// 0.1 Remember to set device first, please refer to CANN Application
// Software Development Guide (C&C++, Inference)
aclError aclSet = aclrtSetDevice(0);
if (aclSet) {
printf("Set device failed ,error code:%d\n", aclSet);
return 0;
}

// 0.2 construct index
const int dim = 512;
const int baseSize = 100000;
const int queryN = 64;
const int capacity = 100000;
const int resourceSize = -1;
auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;
std::vector<float16_t> base(baseSize * dim);
std::vector<ascend::idx_t> ids(baseSize);

for (size_t j = 0; j < base.size(); j++) {
base[j] = drand48();
}
std::iota(ids.begin(), ids.end(), 0);

// 1. build Index and initialize
ascend::IndexILFlat indexFlat;
auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
if (ret) {
printf("Index initialize failed ,error code:%d\n", ret);
return 0;
}

// 2. add base vectors
ret = indexFlat.AddFeatures(baseSize, base.data(), ids.data());
if (ret) {
printf("Add features failed ,error code:%d\n", ret);
return 0;
}

// 3.1 Test ComputeDistance
std::vector<float16_t> queries(queryN * dim);
for (size_t i = 0; i < queries.size(); i++) {
queries[i] = drand48();
}
ret = TestComputeDistance(indexFlat, queryN, baseSize, queries.data());
if (ret) {
printf("Compute distance failed ,error code:%d\n", ret);
return 0;
}

// 3.2 Test SearchByThreshold
ret = TestSearchByThreshold(indexFlat, queryN, queries.data());
if (ret) {
printf("Search by threshold failed ,error code:%d\n", ret);
return 0;
}

// 4. release resource
indexFlat.Finalize();
aclrtResetDevice(0);

printf("------------Demo correct--------------\n");
return 0;
}



/**
/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ -fPIC -fPIE -fstack-protector-all \
-o IndexILDemo IndexILDemo.cpp \
-fopenmp -O3 -frename-registers -fpeel-loops -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -pie -s \
-I/usr/local/AscendMiniOs/acllib/include/ \
-I../include \
-I../device/include \
-L../device/lib \
-L/usr/local/AscendMiniOs/acllib/lib64/stub \
-L/usr/local/Ascend/driver/lib64/common \
-lascendcl -lascend_hal -lc_sec -lascendfaiss_minios
*/