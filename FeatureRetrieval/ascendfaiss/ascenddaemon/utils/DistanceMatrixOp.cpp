/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ascenddaemon/utils/DistanceMatrixOp.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <sys/time.h>
#include <iostream>

namespace ascend {
namespace {
const uint KB = 1024;
const uint CACHE_LIMIT_SIZE_KB = 64;
// size limit in Byte when preloading distance table to L2 cache
const uint CACHE_LIMIT_SIZE = CACHE_LIMIT_SIZE_KB * KB;
// cache_line size in Byte
const uint CPU_CACHE_LINE = 64;

inline void DistTablePreload(const void *addr, size_t size)
{
    for (uint i = 0; i < size / CPU_CACHE_LINE; i++) {
        asm volatile("prfm pldl2keep, [%[offset], %[base]]\n"
            :
            : [ offset ] "r"(i * CPU_CACHE_LINE), [ base ] "r"(addr)
            :);
    }
}
}

DistanceMatrixOp::DistanceMatrixOp() {}

DistanceMatrixOp::~DistanceMatrixOp() {}

bool DistanceMatrixOp::exec(AscendTensor<unsigned char, DIMS_2> &code, AscendTensor<float16_t, DIMS_2> &distTable,
    AscendTensor<float16_t, DIMS_2> &distMatrix)
{
    if (!checkParams(code, distTable, distMatrix)) {
        return false;
    }
    bool ret = true;
    int M = distTable.getSize(0);

    switch (M) {
        case 2:
            computePqCodeSize2(code, distTable, distMatrix);
            break;
        case 4:
        case 8:
        case 12:
        case 20:
        case 24:
            computePqCodeSize4(code, distTable, distMatrix);
            break;
        case 16:
        case 32:
        case 48:
        case 64:
        case 96:
        case 128:
            computePqCodeSize16(code, distTable, distMatrix);
            break;
        default:
            ret = false;
    }

    return ret;
}

bool DistanceMatrixOp::checkParams(AscendTensor<unsigned char, DIMS_2> &code,
    AscendTensor<float16_t, DIMS_2> &distTable, AscendTensor<float16_t, DIMS_2> &distMatrix)
{
    if (code.data() == nullptr || distTable.data() == nullptr || distMatrix.data() == nullptr) {
        return false;
    }

    if (code.getSize(1) != distTable.getSize(0)) {
        return false;
    }

    if (code.getSize(0) > distMatrix.getSize(0)) {
        return false;
    }

    if (code.getSize(1) > distMatrix.getSize(1)) {
        return false;
    }

    return true;
}

void DistanceMatrixOp::computePqCodeSize2(AscendTensor<unsigned char, DIMS_2> &code,
    AscendTensor<float16_t, DIMS_2> &distTable, AscendTensor<float16_t, DIMS_2> &distMatrix)
{
    unsigned char *codeTablePtr = code.data();
    float16_t *distTablePtr = distTable.data();
    float16_t *matrixTablePtr = distMatrix.data();
    const int numPerList = code.getSize(0);
    const int matrixWidth = distMatrix.getSize(1);
    const int distTableWidth = distTable.getSize(1);
    int matrixI = 0;

    for (int index = 0; index < numPerList; index++) {
        unsigned short tmpVal1 = *((unsigned short *)(codeTablePtr + (index << 1)));
        unsigned char idVal1 = (tmpVal1)&0xff;
        unsigned char idVal2 = (tmpVal1 >> 8) & 0xff;

        *(matrixTablePtr + matrixI * matrixWidth + 0) = *(distTablePtr + idVal1);
        *(matrixTablePtr + matrixI * matrixWidth + 1) = *(distTablePtr + distTableWidth + idVal2);

        matrixI++;
    }
}

void DistanceMatrixOp::computePqCodeSize4(AscendTensor<unsigned char, DIMS_2> &code,
    AscendTensor<float16_t, DIMS_2> &distTable, AscendTensor<float16_t, DIMS_2> &distMatrix)
{
    unsigned char *codeTablePtr = code.data();
    float16_t *distTablePtr = distTable.data();
    float16_t *matrixTablePtr = distMatrix.data();

    const int pqM = distTable.getSize(0);
    const int numPerList = code.getSize(0);
    const int pqCentroidNum = distTable.getSize(1);
    const int matrixWidth = distMatrix.getSize(1);

    register unsigned int tmpVal1;
    register unsigned char idVal1;
    register unsigned char idVal2;
    register unsigned char idVal3;
    register unsigned char idVal4;

    register int sub, base, matrixJ;
    register int matrixI = 0;
    register float16_t dist1;
    register float16_t dist2;
    register float16_t dist3;
    register float16_t dist4;
    register float16_t *matrixBaseAddr = nullptr;

    for (int index = 0; index < numPerList; index++) {
        base = index * pqM;
        matrixBaseAddr = matrixTablePtr + matrixI * matrixWidth;
        for (sub = 0, matrixJ = 0; sub < pqM; sub = sub + 4, matrixJ = matrixJ + 4) {
            tmpVal1 = *((unsigned int *)(codeTablePtr + base + sub));
            idVal1 = (tmpVal1)&0xff;
            idVal2 = (tmpVal1 >> 8) & 0xff;
            idVal3 = (tmpVal1 >> 16) & 0xff;
            idVal4 = (tmpVal1 >> 24) & 0xff;

            dist1 = *(distTablePtr + (sub + 0) * pqCentroidNum + idVal1);
            dist2 = *(distTablePtr + (sub + 1) * pqCentroidNum + idVal2);
            dist3 = *(distTablePtr + (sub + 2) * pqCentroidNum + idVal3);
            dist4 = *(distTablePtr + (sub + 3) * pqCentroidNum + idVal4);

            *(matrixBaseAddr + matrixJ + 0) = dist1;
            *(matrixBaseAddr + matrixJ + 1) = dist2;
            *(matrixBaseAddr + matrixJ + 2) = dist3;
            *(matrixBaseAddr + matrixJ + 3) = dist4;
        }

        matrixI++;
    }
}

void DistanceMatrixOp::computePqCodeSize16(AscendTensor<unsigned char, DIMS_2> &code,
    AscendTensor<float16_t, DIMS_2> &distTable, AscendTensor<float16_t, DIMS_2> &distMatrix)
{
    unsigned char *codeTablePtr = code.data();
    float16_t *distTablePtr = distTable.data();
    float16_t *matrixTablePtr = distMatrix.data();

    if (distTable.getSizeInBytes() <= CACHE_LIMIT_SIZE) {
        DistTablePreload(distTablePtr, utils::roundUp(distTable.getSizeInBytes(), CPU_CACHE_LINE));
    }

    const int pqM = distTable.getSize(0);
    const int numPerList = code.getSize(0);
    const int pqCentroidNum = distTable.getSize(1);
    const int matrixWidth = distMatrix.getSize(1);

    unsigned long long tmpVal1;
    unsigned long long tmpVal2;
    float16_t dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8;
    float16_t dist9, dist10, dist11, dist12, dist13, dist14, dist15, dist16;
    unsigned char idVal1, idVal2, idVal3, idVal4, idVal5, idVal6, idVal7, idVal8;
    unsigned char idVal9, idVal10, idVal11, idVal12, idVal13, idVal14, idVal15, idVal16;

    int sub, base, matrixJ;
    int matrixI = 0;

    for (int index = 0; index < numPerList; index++) {
        base = index * pqM;
        for (sub = 0, matrixJ = 0; sub < pqM; base = base + 16, sub = sub + 16, matrixJ = matrixJ + 16) {
            tmpVal1 = *((unsigned long long *)(codeTablePtr + base));
            idVal1 = tmpVal1 & 0xff;
            idVal2 = (tmpVal1 >> 8) & 0xff;
            idVal3 = (tmpVal1 >> 16) & 0xff;
            idVal4 = (tmpVal1 >> 24) & 0xff;
            idVal5 = (tmpVal1 >> 32) & 0xff;
            idVal6 = (tmpVal1 >> 40) & 0xff;
            idVal7 = (tmpVal1 >> 48) & 0xff;
            idVal8 = (tmpVal1 >> 56) & 0xff;
            tmpVal2 = *((unsigned long long *)(codeTablePtr + base + 8));
            idVal9 = tmpVal2 & 0xff;
            idVal10 = (tmpVal2 >> 8) & 0xff;
            idVal11 = (tmpVal2 >> 16) & 0xff;
            idVal12 = (tmpVal2 >> 24) & 0xff;
            idVal13 = (tmpVal2 >> 32) & 0xff;
            idVal14 = (tmpVal2 >> 40) & 0xff;
            idVal15 = (tmpVal2 >> 48) & 0xff;
            idVal16 = (tmpVal2 >> 56) & 0xff;

            dist1 = *(distTablePtr + (sub + 0) * pqCentroidNum + idVal1);
            dist2 = *(distTablePtr + (sub + 1) * pqCentroidNum + idVal2);
            dist3 = *(distTablePtr + (sub + 2) * pqCentroidNum + idVal3);
            dist4 = *(distTablePtr + (sub + 3) * pqCentroidNum + idVal4);

            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 0) = dist1;
            dist5 = *(distTablePtr + (sub + 4) * pqCentroidNum + idVal5);
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 1) = dist2;
            dist6 = *(distTablePtr + (sub + 5) * pqCentroidNum + idVal6);
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 2) = dist3;
            dist7 = *(distTablePtr + (sub + 6) * pqCentroidNum + idVal7);
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 3) = dist4;
            dist8 = *(distTablePtr + (sub + 7) * pqCentroidNum + idVal8);
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 4) = dist5;
            dist9 = *(distTablePtr + (sub + 8) * pqCentroidNum + idVal9);
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 5) = dist6;
            dist10 = *(distTablePtr + (sub + 9) * pqCentroidNum + idVal10);
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 6) = dist7;
            dist11 = *(distTablePtr + (sub + 10) * pqCentroidNum + idVal11);
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 7) = dist8;
            dist12 = *(distTablePtr + (sub + 11) * pqCentroidNum + idVal12);
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 8) = dist9;
            dist13 = *(distTablePtr + (sub + 12) * pqCentroidNum + idVal13);
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 9) = dist10;
            dist14 = *(distTablePtr + (sub + 13) * pqCentroidNum + idVal14);
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 10) = dist11;
            dist15 = *(distTablePtr + (sub + 14) * pqCentroidNum + idVal15);
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 11) = dist12;
            dist16 = *(distTablePtr + (sub + 15) * pqCentroidNum + idVal16);

            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 12) = dist13;
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 13) = dist14;
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 14) = dist15;
            *(matrixTablePtr + matrixI * matrixWidth + matrixJ + 15) = dist16;
        }
        matrixI++;
    }
}
} // namespace ascend
