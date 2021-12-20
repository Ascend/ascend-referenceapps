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

#include <vector>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <faiss/ascend/rpc/AscendRpc.h>

using namespace ::faiss::ascend;

namespace {
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;

inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

// Used to seed the generator.           
inline void FastSrand(int seed)
{
    g_seed = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline int FastRand(void)
{
    const int multiplyNum = 214013;
    const int addNum = 2531011;
    const int rshiftNum = 16;
    g_seed = (multiplyNum * g_seed + addNum);
    
    return (g_seed >> rshiftNum) & FAST_RAND_MAX;
}

namespace {
    const std::vector<int> TEST_DATA_LENGTHS {
        0x1,        // 1 Byte
        0x400,      // 1 KB
        0x80000,    // 512 KB
        0x100000,   // 1 MB
        0x1000000,  // 16 MB
        0x2000000,  // 32 MB
        0x4000000,  // 64 MB
        0x6000000,  // 96 MB
    };

    const int REPEATED_TIMES = 1;
}

TEST(RpcDataIntegrity, SingleDevice)
{
    RpcError err;
    const int dev = 0;
    rpcContext ctx = nullptr;

    err = RpcCreateContext(dev, &ctx);
    ASSERT_EQ(err, RPC_ERROR_NONE);

    FastSrand(0);
    for (int r = 0; r < REPEATED_TIMES; r++) {
        for (const auto &len : TEST_DATA_LENGTHS) {
            std::vector<uint8_t> data(len);
            std::for_each(data.begin(), data.end(), [] (uint8_t &v) { v = static_cast<uint8_t>(FastRand() % 256); });
            err = RpcTestDataIntegrity(ctx, data);
            EXPECT_EQ(err, RPC_ERROR_NONE);
        }
    }

    err = RpcDestroyContext(ctx);
    ASSERT_EQ(err, RPC_ERROR_NONE);
}

TEST(RpcDataIntegrity, MultiDevices)
{
    RpcError err;
    const std::vector<int> devices {0, 1};
    std::vector<rpcContext> ctxs(devices.size());

    for (size_t i = 0; i < devices.size(); i++) {
        err = RpcCreateContext(devices[i], &ctxs[i]);
        ASSERT_EQ(err, RPC_ERROR_NONE);
    }

    FastSrand(0);
    for (int r = 0; r < REPEATED_TIMES; r++) {
        for (const auto &len : TEST_DATA_LENGTHS) {
            std::vector<uint8_t> data(len);
            std::for_each(data.begin(), data.end(), [] (uint8_t &v) { v = static_cast<uint8_t>(FastRand() % 256); });

            for (const auto ctx: ctxs) {
                err = RpcTestDataIntegrity(ctx, data);
                EXPECT_EQ(err, RPC_ERROR_NONE);
            }
        }
    }

    for (auto ctx: ctxs) {
        err = RpcDestroyContext(ctx);
        ASSERT_EQ(err, RPC_ERROR_NONE);
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
