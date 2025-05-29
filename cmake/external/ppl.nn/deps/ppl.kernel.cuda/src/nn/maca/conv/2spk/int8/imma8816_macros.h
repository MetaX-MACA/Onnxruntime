// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
////////////////////////////////////////
// imma macros
////////////////////////////////////////

// int8 input, int32 output
/*
#define MMA_INST_OPCODE \
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"

// operand c is omitted
#define MMA_INST(_d0, _d1, _a, _b) \
        asm volatile(MMA_INST_OPCODE:   "=r"(_d0),   "=r"(_d1): "r"(_a), "r"(_b),  "r"(_d0),   "r"(_d1));
*/

#define MMA_INST(_d0, _d1, _d2, _d3, _a, _b)                                    \
    {                                                                           \
        auto temp = __builtin_mxc_mma_16x16x16i8(_b, _a, {_d0, _d1, _d2, _d3}); \
        _d0       = temp[0];                                                    \
        _d1       = temp[1];                                                    \
        _d2       = temp[2];                                                    \
        _d3       = temp[3];                                                    \
    }

#define MMA_INST_ASCEND1(_C, _C_off, _C_stride, _a0, _B)                                                      \
    {                                                                                                         \
        MMA_INST(_C[_C_off], _C[_C_off + 1], _C[_C_off + _C_stride], _C[_C_off + _C_stride + 1], _a0, _B[0]); \
    }

#define MMA_INST_ASCEND2(_C, _C_off, _C_stride, _a0, _B)                                                              \
    {                                                                                                                 \
        MMA_INST(_C[_C_off], _C[_C_off + 1], _C[_C_off + _C_stride], _C[_C_off + _C_stride + 1], _a0, _B[0]);         \
        MMA_INST(_C[_C_off + 2], _C[_C_off + 3], _C[_C_off + _C_stride + 2], _C[_C_off + _C_stride + 3], _a0, _B[1]); \
    }

#define MMA_INST_ASCEND4(_C, _C_off, _C_stride, _a0, _B)                                                              \
    {                                                                                                                 \
        MMA_INST(_C[_C_off], _C[_C_off + 1], _C[_C_off + _C_stride], _C[_C_off + _C_stride + 1], _a0, _B[0]);         \
        MMA_INST(_C[_C_off + 2], _C[_C_off + 3], _C[_C_off + _C_stride + 2], _C[_C_off + _C_stride + 3], _a0, _B[1]); \
        MMA_INST(_C[_C_off + 4], _C[_C_off + 5], _C[_C_off + _C_stride + 4], _C[_C_off + _C_stride + 5], _a0, _B[2]); \
        MMA_INST(_C[_C_off + 6], _C[_C_off + 7], _C[_C_off + _C_stride + 6], _C[_C_off + _C_stride + 7], _a0, _B[3]); \
    }

#define MMA_INST_ASCEND8(_C, _C_off, _C_stride, _a0, _B)                                                                  \
    {                                                                                                                     \
        MMA_INST(_C[_C_off], _C[_C_off + 1], _C[_C_off + _C_stride], _C[_C_off + _C_stride + 1], _a0, _B[0]);             \
        MMA_INST(_C[_C_off + 2], _C[_C_off + 3], _C[_C_off + _C_stride + 2], _C[_C_off + _C_stride + 3], _a0, _B[1]);     \
        MMA_INST(_C[_C_off + 4], _C[_C_off + 5], _C[_C_off + _C_stride + 4], _C[_C_off + _C_stride + 5], _a0, _B[2]);     \
        MMA_INST(_C[_C_off + 6], _C[_C_off + 7], _C[_C_off + _C_stride + 6], _C[_C_off + _C_stride + 7], _a0, _B[3]);     \
        MMA_INST(_C[_C_off + 8], _C[_C_off + 9], _C[_C_off + _C_stride + 8], _C[_C_off + _C_stride + 9], _a0, _B[4]);     \
        MMA_INST(_C[_C_off + 10], _C[_C_off + 11], _C[_C_off + _C_stride + 10], _C[_C_off + _C_stride + 11], _a0, _B[5]); \
        MMA_INST(_C[_C_off + 12], _C[_C_off + 13], _C[_C_off + _C_stride + 12], _C[_C_off + _C_stride + 13], _a0, _B[6]); \
        MMA_INST(_C[_C_off + 14], _C[_C_off + 15], _C[_C_off + _C_stride + 14], _C[_C_off + _C_stride + 15], _a0, _B[7]); \
    }

#define MMA_INST_DESCEND1(_C, _C_off, _C_stride, _a0, _B)                                                              \
    {                                                                                                                  \
        MMA_INST(_C[_C_off - 1], _C[_C_off - 0], _C[_C_off + _C_stride + -1], _C[_C_off + _C_stride - 0], _a0, _B[0]); \
    }

#define MMA_INST_DESCEND2(_C, _C_off, _C_stride, _a0, _B)                                                             \
    {                                                                                                                 \
        MMA_INST(_C[_C_off - 1], _C[_C_off - 0], _C[_C_off + _C_stride - 1], _C[_C_off + _C_stride - 0], _a0, _B[1]); \
        MMA_INST(_C[_C_off - 3], _C[_C_off - 2], _C[_C_off + _C_stride - 3], _C[_C_off + _C_stride - 2], _a0, _B[0]); \
    }

#define MMA_INST_DESCEND4(_C, _C_off, _C_stride, _a0, _B)                                                             \
    {                                                                                                                 \
        MMA_INST(_C[_C_off - 1], _C[_C_off - 0], _C[_C_off + _C_stride - 1], _C[_C_off + _C_stride - 0], _a0, _B[3]); \
        MMA_INST(_C[_C_off - 3], _C[_C_off - 2], _C[_C_off + _C_stride - 3], _C[_C_off + _C_stride - 2], _a0, _B[2]); \
        MMA_INST(_C[_C_off - 5], _C[_C_off - 4], _C[_C_off + _C_stride - 5], _C[_C_off + _C_stride - 4], _a0, _B[1]); \
        MMA_INST(_C[_C_off - 7], _C[_C_off - 6], _C[_C_off + _C_stride - 7], _C[_C_off + _C_stride - 6], _a0, _B[0]); \
    }

#define MMA_INST_DESCEND8(_C, _C_off, _C_stride, _a0, _B)                                                                 \
    {                                                                                                                     \
        MMA_INST(_C[_C_off - 1], _C[_C_off - 0], _C[_C_off + _C_stride - 1], _C[_C_off + _C_stride - 0], _a0, _B[7]);     \
        MMA_INST(_C[_C_off - 3], _C[_C_off - 2], _C[_C_off + _C_stride - 3], _C[_C_off + _C_stride - 2], _a0, _B[6]);     \
        MMA_INST(_C[_C_off - 5], _C[_C_off - 4], _C[_C_off + _C_stride - 5], _C[_C_off + _C_stride - 4], _a0, _B[5]);     \
        MMA_INST(_C[_C_off - 7], _C[_C_off - 6], _C[_C_off + _C_stride - 7], _C[_C_off + _C_stride - 6], _a0, _B[4]);     \
        MMA_INST(_C[_C_off - 9], _C[_C_off - 8], _C[_C_off + _C_stride - 9], _C[_C_off + _C_stride - 8], _a0, _B[3]);     \
        MMA_INST(_C[_C_off - 11], _C[_C_off - 10], _C[_C_off + _C_stride - 11], _C[_C_off + _C_stride - 10], _a0, _B[2]); \
        MMA_INST(_C[_C_off - 13], _C[_C_off - 12], _C[_C_off + _C_stride - 13], _C[_C_off + _C_stride - 12], _a0, _B[1]); \
        MMA_INST(_C[_C_off - 15], _C[_C_off - 14], _C[_C_off + _C_stride - 15], _C[_C_off + _C_stride - 14], _a0, _B[0]); \
    }

#define MMA_INST_1x1(_C, _A, _B)                               \
    {                                                          \
        MMA_INST_ASCEND1(_C, 0, TILE_N_V2_PER_THD, _A[0], _B); \
    }

#define MMA_INST_1x2(_C, _A, _B)                               \
    {                                                          \
        MMA_INST_ASCEND2(_C, 0, TILE_N_V2_PER_THD, _A[0], _B); \
    }

#define MMA_INST_1x4(_C, _A, _B)                               \
    {                                                          \
        MMA_INST_ASCEND4(_C, 0, TILE_N_V2_PER_THD, _A[0], _B); \
    }

#define MMA_INST_1x8(_C, _A, _B)                               \
    {                                                          \
        MMA_INST_ASCEND8(_C, 0, TILE_N_V2_PER_THD, _A[0], _B); \
    }

#define MMA_INST_1x16(_C, _A, _B)                               \
    {                                                           \
        MMA_INST_ASCEND16(_C, 0, TILE_N_V2_PER_THD, _A[0], _B); \
    }

#define MMA_INST_2x1(_C, _A, _B)                                \
    {                                                           \
        MMA_INST_ASCEND1(_C, 0, TILE_N_V2_PER_THD, _A[0], _B);  \
        MMA_INST_DESCEND1(_C, 5, TILE_N_V2_PER_THD, _A[1], _B); \
    }

#define MMA_INST_2x2(_C, _A, _B)                                 \
    {                                                            \
        MMA_INST_ASCEND2(_C, 0, TILE_N_V2_PER_THD, _A[0], _B);   \
        MMA_INST_DESCEND2(_C, 11, TILE_N_V2_PER_THD, _A[1], _B); \
    }

#define MMA_INST_2x4(_C, _A, _B) \
        { \
            MMA_INST_ASCEND4  (_C, 0,TILE_N_V2_PER_THD,  _A[0], _B); \
            MMA_INST_DESCEND4 (_C, 23,TILE_N_V2_PER_THD _A[1], _B); \
        }

#define MMA_INST_2x8(_C, _A, _B)                                 \
    {                                                            \
        MMA_INST_ASCEND8(_C, 0, TILE_N_V2_PER_THD, _A[0], _B);   \
        MMA_INST_DESCEND8(_C, 47, TILE_N_V2_PER_THD, _A[1], _B); \
    }

#define MMA_INST_4x1(_C, _A, _B)                                 \
    {                                                            \
        MMA_INST_ASCEND1(_C, 0, TILE_N_V2_PER_THD, _A[0], _B);   \
        MMA_INST_DESCEND1(_C, 5, TILE_N_V2_PER_THD, _A[1], _B);  \
                                                                 \
        MMA_INST_ASCEND1(_C, 8, TILE_N_V2_PER_THD, _A[2], _B);   \
        MMA_INST_DESCEND1(_C, 13, TILE_N_V2_PER_THD, _A[3], _B); \
    }

#define MMA_INST_4x2(_C, _A, _B)                                 \
    {                                                            \
        MMA_INST_ASCEND2(_C, 0, TILE_N_V2_PER_THD, _A[0], _B);   \
        MMA_INST_DESCEND2(_C, 11, TILE_N_V2_PER_THD, _A[1], _B); \
                                                                 \
        MMA_INST_ASCEND2(_C, 16, TILE_N_V2_PER_THD, _A[2], _B);  \
        MMA_INST_DESCEND2(_C, 17, TILE_N_V2_PER_THD, _A[3], _B); \
    }

#define MMA_INST_4x4(_C, _A, _B)                                 \
    {                                                            \
        MMA_INST_ASCEND4(_C, 0, TILE_N_V2_PER_THD, _A[0], _B);   \
        MMA_INST_DESCEND4(_C, 23, TILE_N_V2_PER_THD, _A[1], _B); \
                                                                 \
        MMA_INST_ASCEND4(_C, 32, TILE_N_V2_PER_THD, _A[2], _B);  \
        MMA_INST_DESCEND4(_C, 55, TILE_N_V2_PER_THD, _A[3], _B); \
    }

#define MMA_INST_4x8(_C, _A, _B)                                  \
    {                                                             \
        MMA_INST_ASCEND8(_C, 0, TILE_N_V2_PER_THD, _A[0], _B);    \
        MMA_INST_DESCEND8(_C, 47, TILE_N_V2_PER_THD, _A[1], _B);  \
                                                                  \
        MMA_INST_ASCEND8(_C, 64, TILE_N_V2_PER_THD, _A[2], _B);   \
        MMA_INST_DESCEND8(_C, 111, TILE_N_V2_PER_THD, _A[3], _B); \
    }

#define MMA_INST_8x1(_C, _A, _B)                                 \
    {                                                            \
        MMA_INST_ASCEND1(_C, 0, TILE_N_V2_PER_THD, _A[0], _B);   \
        MMA_INST_DESCEND1(_C, 5, TILE_N_V2_PER_THD, _A[1], _B);  \
                                                                 \
        MMA_INST_ASCEND1(_C, 8, TILE_N_V2_PER_THD, _A[2], _B);   \
        MMA_INST_DESCEND1(_C, 13, TILE_N_V2_PER_THD, _A[3], _B); \
                                                                 \
        MMA_INST_ASCEND1(_C, 16, TILE_N_V2_PER_THD, _A[4], _B);  \
        MMA_INST_DESCEND1(_C, 21, TILE_N_V2_PER_THD, _A[5], _B); \
                                                                 \
        MMA_INST_ASCEND1(_C, 24, TILE_N_V2_PER_THD, _A[6], _B);  \
        MMA_INST_DESCEND1(_C, 29, TILE_N_V2_PER_THD, _A[7], _B); \
    }

#define MMA_INST_8x2(_C, _A, _B)                                 \
    {                                                            \
        MMA_INST_ASCEND2(_C, 0, TILE_N_V2_PER_THD, _A[0], _B);   \
        MMA_INST_DESCEND2(_C, 11, TILE_N_V2_PER_THD, _A[1], _B); \
                                                                 \
        MMA_INST_ASCEND2(_C, 16, TILE_N_V2_PER_THD, _A[2], _B);  \
        MMA_INST_DESCEND2(_C, 27, TILE_N_V2_PER_THD, _A[3], _B); \
                                                                 \
        MMA_INST_ASCEND2(_C, 32, TILE_N_V2_PER_THD, _A[4], _B);  \
        MMA_INST_DESCEND2(_C, 43, TILE_N_V2_PER_THD, _A[5], _B); \
                                                                 \
        MMA_INST_ASCEND2(_C, 48, TILE_N_V2_PER_THD, _A[6], _B);  \
        MMA_INST_DESCEND2(_C, 59, TILE_N_V2_PER_THD, _A[7], _B); \
    }

#define MMA_INST_8x4(_C, _A, _B)                                  \
    {                                                             \
        MMA_INST_ASCEND4(_C, 0, TILE_N_V2_PER_THD, _A[0], _B);    \
        MMA_INST_DESCEND4(_C, 23, TILE_N_V2_PER_THD, _A[1], _B);  \
                                                                  \
        MMA_INST_ASCEND4(_C, 32, TILE_N_V2_PER_THD, _A[2], _B);   \
        MMA_INST_DESCEND4(_C, 55, TILE_N_V2_PER_THD, _A[3], _B);  \
                                                                  \
        MMA_INST_ASCEND4(_C, 64, TILE_N_V2_PER_THD, _A[4], _B);   \
        MMA_INST_DESCEND4(_C, 87, TILE_N_V2_PER_THD, _A[5], _B);  \
                                                                  \
        MMA_INST_ASCEND4(_C, 96, TILE_N_V2_PER_THD, _A[6], _B);   \
        MMA_INST_DESCEND4(_C, 119, TILE_N_V2_PER_THD, _A[7], _B); \
    }
