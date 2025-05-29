// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

////////////////////////////////////////
// hmma macros
////////////////////////////////////////

/*
#define MMA_INST_OPCODE \
    "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"

#define MMA_INST(_d0, _d1, _a0, _a1, _b) \
    asm volatile(MMA_INST_OPCODE         \
                 : "=r"(_d0), "=r"(_d1)  \
                 : "r"(_a0), "r"(_a1), "r"(_b), "r"(_d0), "r"(_d1));
*/

static __device__ int shfl_cuda2maca_sync(int a0, int a1) {
    const int lane_id = __lane_id();
    const int delta = (lane_id - (lane_id / 8) * 8) * 4 + ((lane_id / 16) ^ 0x3) - lane_id;
    int x0 = __shfl_down_sync(0xffffffffffffffffull, a0, delta);
    int x1 = __shfl_down_sync(0xffffffffffffffffull, a1, delta);
    if ((lane_id / 8) % 2 == 0)
        return x0;
    return x1;
}

#define MMA_INST(_d0, _d1, _a0, _a1, _b)                                                       \
    {                                                                                          \
        int maca_a = shfl_cuda2maca_sync(_a0, _a1);                                            \
        int maca_b = shfl_cuda2maca_sync(_b, _b);                                              \
        __half2* ha = reinterpret_cast<__half2*>(&maca_a);                                     \
        __half2* hb = reinterpret_cast<__half2*>(&maca_b);                                     \
        __half2* hd0 = reinterpret_cast<__half2*>(&_d0);                                       \
        __half2* hd1 = reinterpret_cast<__half2*>(&_d1);                                       \
        __fp16 a_low = static_cast<__fp16>(float(ha->x));                                      \
        __fp16 a_high = static_cast<__fp16>(float(ha->y));                                     \
        __fp16 b_low = static_cast<__fp16>(float(hb->x));                                      \
        __fp16 b_high = static_cast<__fp16>(float(hb->y));                                     \
        __fp16 d0_low = static_cast<__fp16>(float(hd0->x));                                    \
        __fp16 d0_high = static_cast<__fp16>(float(hd0->y));                                   \
        __fp16 d1_low = static_cast<__fp16>(float(hd1->x));                                    \
        __fp16 d1_high = static_cast<__fp16>(float(hd1->y));                                   \
        auto ret = __builtin_mxc_mma_16x16x16f16({a_high, a_low, 0, 0}, {b_high, b_low, 0, 0}, \
                                                 {d0_low, d0_high, d1_low, d1_high});          \
        __half2 _hd0 = __halves2half2(ret[0], ret[1]);                                         \
        __half2 _hd1 = __halves2half2(ret[2], ret[3]);                                         \
        _d0 = *(reinterpret_cast<int*>(&_hd0));                                                \
        _d1 = *(reinterpret_cast<int*>(&_hd1));                                                \
    }

#define MMA_INST_1INT_ASCEND1(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off)  \
    {                                                                           \
        MMA_INST(_C[_C_off], _C[_C_off + _C_stride], _a0, _a1, _Bv1[_Bv1_off]); \
    }

#define MMA_INST_1INT_ASCEND2(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off)                       \
    {                                                                                                \
        MMA_INST(_C[_C_off], _C[_C_off + _C_stride], _a0, _a1, _Bv1[_Bv1_off]);                      \
        MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \
    }

#define MMA_INST_1INT_ASCEND4(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off)                       \
    {                                                                                                \
        MMA_INST(_C[_C_off], _C[_C_off + _C_stride], _a0, _a1, _Bv1[_Bv1_off]);                      \
        MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \
        MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 2]); \
        MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 3]); \
    }

#define MMA_INST_1INT_ASCEND8(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off)                       \
    {                                                                                                \
        MMA_INST(_C[_C_off], _C[_C_off + _C_stride], _a0, _a1, _Bv1[_Bv1_off]);                      \
        MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \
        MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 2]); \
        MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 3]); \
        MMA_INST(_C[_C_off + 4], _C[_C_off + _C_stride + 4], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 4]); \
        MMA_INST(_C[_C_off + 5], _C[_C_off + _C_stride + 5], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 5]); \
        MMA_INST(_C[_C_off + 6], _C[_C_off + _C_stride + 6], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 6]); \
        MMA_INST(_C[_C_off + 7], _C[_C_off + _C_stride + 7], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 7]); \
    }

#define MMA_INST_1INT_DESCEND1(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \
    {                                                                           \
        MMA_INST(_C[_C_off], _C[_C_off + _C_stride], _a0, _a1, _Bv1[_Bv1_off]); \
    }

#define MMA_INST_1INT_DESCEND2(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off)              \
    {                                                                                        \
        MMA_INST(_C[_C_off], _C[_C_off + _C_stride], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \
        MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off]);      \
    }

#define MMA_INST_1INT_DESCEND4(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off)                      \
    {                                                                                                \
        MMA_INST(_C[_C_off], _C[_C_off + _C_stride], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 3]);         \
        MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 2]); \
        MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \
        MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _a0, _a1, _Bv1[_Bv1_off]);              \
    }

#define MMA_INST_1INT_DESCEND8(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off)                      \
    {                                                                                                \
        MMA_INST(_C[_C_off], _C[_C_off + _C_stride], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 7]);         \
        MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 6]); \
        MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 5]); \
        MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 4]); \
        MMA_INST(_C[_C_off - 4], _C[_C_off + _C_stride - 4], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 3]); \
        MMA_INST(_C[_C_off - 5], _C[_C_off + _C_stride - 5], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 2]); \
        MMA_INST(_C[_C_off - 6], _C[_C_off + _C_stride - 6], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \
        MMA_INST(_C[_C_off - 7], _C[_C_off + _C_stride - 7], _a0, _a1, _Bv1[_Bv1_off]);              \
    }

#define MMA_INST_1INT_1x1(_C, _Av1, _Bv1)                                                    \
    {                                                                                        \
        MMA_INST_1INT_ASCEND1(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0); \
    }

#define MMA_INST_1INT_1x2(_C, _Av1, _Bv1)                                                    \
    {                                                                                        \
        MMA_INST_1INT_ASCEND2(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0); \
    }

#define MMA_INST_1INT_1x4(_C, _Av1, _Bv1)                                                    \
    {                                                                                        \
        MMA_INST_1INT_ASCEND4(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0); \
    }

#define MMA_INST_1INT_1x8(_C, _Av1, _Bv1)                                                    \
    {                                                                                        \
        MMA_INST_1INT_ASCEND8(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0); \
    }

#define MMA_INST_1INT_2x1(_C, _Av1, _Bv1)                                                                             \
    {                                                                                                                 \
        MMA_INST_1INT_ASCEND1(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0);                          \
        MMA_INST_1INT_DESCEND1(_C, 2, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_], _Av1[0 + _1INT_ + _1INT_X2_], _Bv1, 0); \
    }

#define MMA_INST_1INT_2x2(_C, _Av1, _Bv1)                                                                             \
    {                                                                                                                 \
        MMA_INST_1INT_ASCEND2(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0);                          \
        MMA_INST_1INT_DESCEND2(_C, 5, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_], _Av1[0 + _1INT_ + _1INT_X2_], _Bv1, 0); \
    }

#define MMA_INST_1INT_2x4(_C, _Av1, _Bv1)                                                                              \
    {                                                                                                                  \
        MMA_INST_1INT_ASCEND4(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0);                           \
        MMA_INST_1INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_], _Av1[0 + _1INT_ + _1INT_X2_], _Bv1, 0); \
    }

#define MMA_INST_1INT_2x8(_C, _Av1, _Bv1)                                                                              \
    {                                                                                                                  \
        MMA_INST_1INT_ASCEND8(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0);                           \
        MMA_INST_1INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_], _Av1[0 + _1INT_ + _1INT_X2_], _Bv1, 0); \
    }

#define MMA_INST_1INT_4x1(_C, _Av1, _Bv1)                                                                                     \
    {                                                                                                                         \
        MMA_INST_1INT_ASCEND1(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0);                                  \
        MMA_INST_1INT_DESCEND1(_C, 2, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0); \
        MMA_INST_1INT_ASCEND1(_C, 4, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0);  \
        MMA_INST_1INT_DESCEND1(_C, 6, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \
    }

#define MMA_INST_1INT_4x2(_C, _Av1, _Bv1)                                                                                      \
    {                                                                                                                          \
        MMA_INST_1INT_ASCEND2(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0);                                   \
        MMA_INST_1INT_DESCEND2(_C, 5, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0);  \
        MMA_INST_1INT_ASCEND2(_C, 8, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0);   \
        MMA_INST_1INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \
    }

#define MMA_INST_1INT_4x4(_C, _Av1, _Bv1)                                                                                      \
    {                                                                                                                          \
        MMA_INST_1INT_ASCEND4(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0);                                   \
        MMA_INST_1INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0); \
        MMA_INST_1INT_ASCEND4(_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0);  \
        MMA_INST_1INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \
    }

#define MMA_INST_1INT_4x8(_C, _Av1, _Bv1)                                                                                      \
    {                                                                                                                          \
        MMA_INST_1INT_ASCEND8(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0);                                   \
        MMA_INST_1INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0); \
        MMA_INST_1INT_ASCEND8(_C, 32, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0);  \
        MMA_INST_1INT_DESCEND8(_C, 55, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \
    }

#define MMA_INST_1INT_8x1(_C, _Av1, _Bv1)                                                                                      \
    {                                                                                                                          \
        MMA_INST_1INT_ASCEND1(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0);                                   \
        MMA_INST_1INT_DESCEND1(_C, 2, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0);  \
        MMA_INST_1INT_ASCEND1(_C, 4, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0);   \
        MMA_INST_1INT_DESCEND1(_C, 6, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0);  \
        MMA_INST_1INT_ASCEND1(_C, 8, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 4], _Av1[0 + _1INT_ + _1INT_X2_ * 4], _Bv1, 0);   \
        MMA_INST_1INT_DESCEND1(_C, 10, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 5], _Av1[0 + _1INT_ + _1INT_X2_ * 5], _Bv1, 0); \
        MMA_INST_1INT_ASCEND1(_C, 12, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 6], _Av1[0 + _1INT_ + _1INT_X2_ * 6], _Bv1, 0);  \
        MMA_INST_1INT_DESCEND1(_C, 14, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 7], _Av1[0 + _1INT_ + _1INT_X2_ * 7], _Bv1, 0); \
    }

#define MMA_INST_1INT_8x2(_C, _Av1, _Bv1)                                                                                      \
    {                                                                                                                          \
        MMA_INST_1INT_ASCEND2(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0);                                   \
        MMA_INST_1INT_DESCEND2(_C, 5, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0);  \
        MMA_INST_1INT_ASCEND2(_C, 8, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0);   \
        MMA_INST_1INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \
        MMA_INST_1INT_ASCEND2(_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 4], _Av1[0 + _1INT_ + _1INT_X2_ * 4], _Bv1, 0);  \
        MMA_INST_1INT_DESCEND2(_C, 21, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 5], _Av1[0 + _1INT_ + _1INT_X2_ * 5], _Bv1, 0); \
        MMA_INST_1INT_ASCEND2(_C, 24, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 6], _Av1[0 + _1INT_ + _1INT_X2_ * 6], _Bv1, 0);  \
        MMA_INST_1INT_DESCEND2(_C, 29, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 7], _Av1[0 + _1INT_ + _1INT_X2_ * 7], _Bv1, 0); \
    }

#define MMA_INST_1INT_8x4(_C, _Av1, _Bv1)                                                                                      \
    {                                                                                                                          \
        MMA_INST_1INT_ASCEND4(_C, 0, TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0);                                   \
        MMA_INST_1INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0); \
        MMA_INST_1INT_ASCEND4(_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0);  \
        MMA_INST_1INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \
        MMA_INST_1INT_ASCEND4(_C, 32, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 4], _Av1[0 + _1INT_ + _1INT_X2_ * 4], _Bv1, 0);  \
        MMA_INST_1INT_DESCEND4(_C, 43, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 5], _Av1[0 + _1INT_ + _1INT_X2_ * 5], _Bv1, 0); \
        MMA_INST_1INT_ASCEND4(_C, 48, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 6], _Av1[0 + _1INT_ + _1INT_X2_ * 6], _Bv1, 0);  \
        MMA_INST_1INT_DESCEND4(_C, 59, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 7], _Av1[0 + _1INT_ + _1INT_X2_ * 7], _Bv1, 0); \
    }
