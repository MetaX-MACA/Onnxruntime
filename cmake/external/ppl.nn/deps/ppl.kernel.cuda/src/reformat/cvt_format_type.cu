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

#include <float.h>
#include <iostream>
#include "cudakernel/reformat/reformat.h"
#include "cudakernel/common/common.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/macro.h"
#include "cvt_type_per_elems.cuh"

#include "cuda_fp16.h"
using namespace PPLCUDA;
using namespace ppl::common;

#ifdef PPLNN_USE_MACA
#define DIM 16
#else
#define DIM 32
#endif

#define LEASTCHANNEL 16

#define cvtTYPEALL(cvt_format)  \
    cvt_format(FLOAT32_INT8)    \
    cvt_format(INT8_FLOAT32)    \
    cvt_format(UINT8_FLOAT32)    \
    cvt_format(UINT8_FLOAT16)    \
    cvt_format(FLOAT32_FLOAT16) \
    cvt_format(FLOAT16_FLOAT32) \
    cvt_format(FLOAT32_INT4B)   \
    cvt_format(INT4B_FLOAT32)   \
    cvt_format(INT8_FLOAT16)    \
    cvt_format(FLOAT16_INT8)    \
    cvt_format(INT8_INT4B)      \
    cvt_format(INT8_INT8)       \
    cvt_format(INT4B_INT4B)     \
    cvt_format(INT32_INT64)     \
    cvt_format(INT64_INT32)     \
    cvt_format(INT64_FLOAT32)   \
    cvt_format(FLOAT32_INT64)   \
    cvt_format(FLOAT16_INT64)

#ifdef __MACACC__
#define OPT_CVT_FORMAT
#endif//__MACACC__

template <CVTTypeMode t_mode, CVTFormatMode mode>
__global__ void cuda_kernel_cvtformat_type(
    const void* input,
    void* output,
    ReFormatParam param)
{
}

#define cvtC16TOC8(type_mode)                                                                           \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat_type<type_mode, NHWC16_NHWC8>(                                         \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
                                                                                                        \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n+= gridDim.z) {                                              \
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        int64_t idx_h = blockIdx.y * blockDim.y + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.dst_pad && idx_h < param.n_inner) {                                           \
            int64_t dst_offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;     \
            int64_t src_offset = n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;     \
            cuda_kernel_cvt_per_elems<type_mode>(input, src_offset, output, dst_offset, param);         \
        }                                                                                               \
    }                                                                                                   \
}                                                                                                       

cvtTYPEALL(cvtC16TOC8)

#define cvtC8TOC16(type_mode)                                                                           \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat_type<type_mode, NHWC8_NHWC16>(                                         \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n+= gridDim.z) {                                              \
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        int64_t idx_h = blockIdx.y * blockDim.y + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.dst_pad && idx_h < param.n_inner) {                                           \
            int64_t dst_offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;     \
            int64_t src_offset = n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;     \
            if (idx_w < param.src_pad) {                                                                \
              cuda_kernel_cvt_per_elems<type_mode>(input, src_offset, output, dst_offset, param);       \
            } else {                                                                                    \
              cuda_kernel_set_zero_per_elems<type_mode>(output, dst_offset);                            \
            }                                                                                           \
        }                                                                                               \
    }                                                                                                   \
}                                                                                                       
cvtTYPEALL(cvtC8TOC16)

#define cvtNCTONHWC(type, type_mode)                                                             \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat_type<type_mode, NDARRAY_NHWC>(                                         \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    __shared__ type share_val[DIM][DIM + 1];                                                            \
                                                                                                        \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n+= gridDim.z) {                                              \
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        int64_t idx_h = blockIdx.y * blockDim.y + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.n_inner && idx_h < param.src_pad) {                                           \
            int64_t offset = n * param.src_pad * param.n_inner + idx_h * param.n_inner + idx_w;         \
            share_val[threadIdx.y][threadIdx.x] = ((const type*)input)[offset];                         \
        } else {                                                                                        \
            share_val[threadIdx.y][threadIdx.x] = type(0);                                              \
        }                                                                                               \
        __syncthreads();                                                                                \
                                                                                                        \
        idx_w = blockIdx.y * blockDim.y + threadIdx.x;                                                  \
        idx_h = blockIdx.x * blockDim.x + threadIdx.y;                                                  \
                                                                                                        \
        if (idx_w < param.dst_pad && idx_h < param.n_inner) {                                           \
            int64_t offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;         \
            int in_offset = threadIdx.x * (DIM + 1) + threadIdx.y;                                      \
            cuda_kernel_cvt_per_elems<type_mode>((const void*)share_val, in_offset, output, offset, param);  \
        }                                                                                               \
    }                                                                                                   \
}
cvtNCTONHWC(float, FLOAT32_INT8)
cvtNCTONHWC(char, INT8_FLOAT32)
cvtNCTONHWC(uint8_t, UINT8_FLOAT32)
cvtNCTONHWC(uint8_t, UINT8_FLOAT16)
cvtNCTONHWC(float, FLOAT32_FLOAT16)
cvtNCTONHWC(half, FLOAT16_FLOAT32)
cvtNCTONHWC(float, FLOAT32_INT4B)
cvtNCTONHWC(char, INT4B_FLOAT32)
cvtNCTONHWC(char, INT8_FLOAT16)
cvtNCTONHWC(half, FLOAT16_INT8)
cvtNCTONHWC(char, INT8_INT4B)
cvtNCTONHWC(char, INT8_INT8)
cvtNCTONHWC(char, INT4B_INT4B)
cvtNCTONHWC(float, INT32_INT64)
cvtNCTONHWC(double, INT64_INT32)
cvtNCTONHWC(double, INT64_FLOAT32)
cvtNCTONHWC(float, FLOAT32_INT64)

#define cvtNHWC8TONC(type, type_mode)                                                                   \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat_type<type_mode, NHWC_NDARRAY>(                                         \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    __shared__ type share_val[DIM][DIM + 1];                                                            \
                                                                                                        \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n += gridDim.z) {                                             \
        for (int t = blockIdx.y; t < DivUp(param.n_inner, DIM) ; t+= gridDim.y) { \
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        int64_t idx_h = t * blockDim.y + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.src_pad && idx_h < param.n_inner) {                                           \
            int64_t offset = n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;         \
            share_val[threadIdx.y][threadIdx.x] = ((const type*)input)[offset];                         \
        } else {                                                                                        \
            share_val[threadIdx.y][threadIdx.x] = (type)0;                                              \
        }                                                                                               \
        __syncthreads();                                                                                \
                                                                                                        \
        idx_w = t * blockDim.y + threadIdx.x;                                                  \
        idx_h = blockIdx.x * blockDim.x + threadIdx.y;                                                  \
                                                                                                        \
        if (idx_w < param.n_inner && idx_h < param.dst_pad) {                                           \
            int64_t offset = n * param.dst_pad * param.n_inner + idx_h * param.n_inner + idx_w;         \
            int in_offset = threadIdx.x * (DIM + 1) + threadIdx.y;                                      \
            cuda_kernel_cvt_per_elems<type_mode>((const void*)share_val, in_offset, output, offset, param);  \
        }                                                                                               \
        }\
    }                                                                                                   \
}

cvtNHWC8TONC(float, FLOAT32_INT8)
cvtNHWC8TONC(char, INT8_FLOAT32)
cvtNHWC8TONC(unsigned char, UINT8_FLOAT32)
cvtNHWC8TONC(unsigned char, UINT8_FLOAT16)
cvtNHWC8TONC(float, FLOAT32_FLOAT16)
cvtNHWC8TONC(half, FLOAT16_FLOAT32)
cvtNHWC8TONC(float, FLOAT32_INT4B)
cvtNHWC8TONC(char, INT4B_FLOAT32)
cvtNHWC8TONC(char, INT8_FLOAT16)
cvtNHWC8TONC(half, FLOAT16_INT8)
cvtNHWC8TONC(char, INT8_INT4B)
cvtNHWC8TONC(char, INT8_INT8)
cvtNHWC8TONC(char, INT4B_INT4B)
cvtNHWC8TONC(float, INT32_INT64)
cvtNHWC8TONC(double, INT64_INT32)
cvtNHWC8TONC(double, INT64_FLOAT32)
cvtNHWC8TONC(float, FLOAT32_INT64)

#define cvtN4CXTONC(type_mode)                                                                                         \
template <>                                                                                                            \
__global__ void cuda_kernel_cvtformat_type<type_mode, N4CX_NDARRAY>(                                                        \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                               \
{                                                                                                                      \
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;                                                        \
    if (tid >= param.n_inner)                                                                                          \
        return;                                                                                                        \
    const uint64_t inner_idx = tid;                                                                                    \
    const uint64_t num_inner = blockIdx.z;                                                                             \
    const uint64_t c4_idx    = blockIdx.y;                                                                             \
    _Pragma("unroll 4") for (int c_in_c4_idx = 0; c_in_c4_idx < 4; c_in_c4_idx++)                                      \
    {                                                                                                                  \
        const uint64_t c_idx       = c4_idx * 4 + c_in_c4_idx;                                                         \
        const uint64_t size        = param.n_inner;                                                                    \
        const uint64_t padChannels = gridDim.y * 4;                                                                    \
        const uint64_t numChannels = param.channel;                                                                    \
        if (c_idx < numChannels) {                                                                                     \
            const uint64_t offset    = num_inner * padChannels * size + (c4_idx * size + inner_idx) * 4 + c_in_c4_idx; \
            const uint64_t outOffset = num_inner * numChannels * size + c_idx * size + inner_idx;                      \
            cuda_kernel_cvt_per_elems<type_mode>(input, offset, output, outOffset, param);                             \
        }                                                                                                              \
    }                                                                                                                  \
}

cvtTYPEALL(cvtN4CXTONC)

#define cvtNCTON4CX(type_mode)                                                                                        \
template <>                                                                                                           \
__global__ void cuda_kernel_cvtformat_type<type_mode, NDARRAY_N4CX>(                                                       \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                              \
{                                                                                                                     \
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;                                                       \
    if (tid >= param.n_inner)                                                                                         \
        return;                                                                                                       \
    const uint64_t inner_idx = tid;                                                                                   \
    const uint64_t num_inner = blockIdx.z;                                                                            \
    const uint64_t c4_idx    = blockIdx.y;                                                                            \
    _Pragma("unroll 4") for (int c_in_c4_idx = 0; c_in_c4_idx < 4; c_in_c4_idx++)                                     \
    {                                                                                                                 \
        const uint64_t c_idx       = c4_idx * 4 + c_in_c4_idx;                                                        \
        const uint64_t size        = param.n_inner;                                                                   \
        const uint64_t padChannels = gridDim.y * 4;                                                                   \
        const uint64_t numChannels = param.channel;                                                                   \
        if (c_idx < numChannels) {                                                                                    \
            const uint64_t offset   = num_inner * padChannels * size + (c4_idx * size + inner_idx) * 4 + c_in_c4_idx; \
            const uint64_t inOffset = num_inner * numChannels * size + c_idx * size + inner_idx;                      \
            cuda_kernel_cvt_per_elems<type_mode>(input, inOffset, output, offset, param);                             \
        }                                                                                                             \
    }                                                                                                                 \
}

cvtTYPEALL(cvtNCTON4CX)

#define cvtNC1HWC0TONC(type_mode)                                                                                      \
template <>                                                                                                            \
__global__ void cuda_kernel_cvtformat_type<type_mode, NC1HWC0_NDARRAY>(                                                \
    const void* input,                                                                                                 \
    void* output,                                                                                                      \
    ReFormatParam param)                                                                                               \
{                                                                                                                      \
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;                                                        \
    if (tid >= param.n_inner)                                                                                          \
        return;                                                                                                        \
    const uint64_t inner_idx = tid;                                                                                    \
    const uint64_t num_inner = blockIdx.z;                                                                             \
    const uint64_t c16_idx    = blockIdx.y;                                                                            \
    _Pragma("unroll 16") for (int c_in_c16_idx = 0; c_in_c16_idx < 16; c_in_c16_idx++)                                 \
    {                                                                                                                  \
        const uint64_t c_idx       = c16_idx * 16 + c_in_c16_idx;                                                      \
        const uint64_t size        = param.n_inner;                                                                    \
        const uint64_t padChannels = gridDim.y * 16;                                                                   \
        const uint64_t numChannels = param.channel;                                                                    \
        if (c_idx < numChannels) {                                                                                     \
            const uint64_t offset    = num_inner * padChannels * size + (c16_idx * size + inner_idx) * 16 + c_in_c16_idx; \
            const uint64_t outOffset = num_inner * numChannels * size + c_idx * size + inner_idx;                      \
            cuda_kernel_cvt_per_elems<type_mode>(input, offset, output, outOffset, param);                             \
        }                                                                                                              \
    }                                                                                                                  \
}

cvtTYPEALL(cvtNC1HWC0TONC)

#define cvtNCTONC1HWC0(type_mode)                                                                                     \
template <>                                                                                                           \
__global__ void cuda_kernel_cvtformat_type<type_mode, NDARRAY_NC1HWC0>(                                               \
    const void* input,                                                                                                \
    void* output,                                                                                                     \
    ReFormatParam param)                                                                                              \
{                                                                                                                     \
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;                                                       \
    if (tid >= param.n_inner)                                                                                         \
        return;                                                                                                       \
    const uint64_t inner_idx = tid;                                                                                   \
    const uint64_t num_inner = blockIdx.z;                                                                            \
    const uint64_t c16_idx    = blockIdx.y;                                                                           \
    _Pragma("unroll 16") for (int c_in_c16_idx = 0; c_in_c16_idx < 16; c_in_c16_idx++)                                \
    {                                                                                                                 \
        const uint64_t c_idx       = c16_idx * 16 + c_in_c16_idx;                                                     \
        const uint64_t size        = param.n_inner;                                                                   \
        const uint64_t padChannels = gridDim.y * 16;                                                                  \
        const uint64_t numChannels = param.channel;                                                                   \
        if (c_idx < numChannels) {                                                                                    \
            const uint64_t offset   = num_inner * padChannels * size + (c16_idx * size + inner_idx) * 16 + c_in_c16_idx; \
            const uint64_t inOffset = num_inner * numChannels * size + c_idx * size + inner_idx;                      \
            cuda_kernel_cvt_per_elems<type_mode>(input, inOffset, output, offset, param);                             \
        }                                                                                                             \
    }                                                                                                                 \
}

cvtTYPEALL(cvtNCTONC1HWC0)

template <CVTTypeMode t_mode, CVTFormatMode mode>
__global__ void cuda_kernel_small_channel_cvtformat_type(
    const void* input,
    int num_elems,
    DivModFast inner_fast,
    DivModFast src_pad_fast,
    DivModFast dst_pad_fast,
    void* output,
    ReFormatParam param)
{
}

#define cvtSMCHANNELNCTONHWC8(type_mode)                                                                \
template<>                                                                                              \
__global__ void cuda_kernel_small_channel_cvtformat_type<type_mode, NDARRAY_NHWC>(                      \
    const void* input,                                                                                  \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    dst_pad_fast.divmod(tid, num_inner, c_idx);                                                         \
    inner_idx = inner_fast.mod(num_inner);                                                              \
    int outer_idx = inner_fast.div(num_inner);                                                          \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx * param.n_inner + inner_idx;         \
    if (c_idx < param.src_pad) {                                                                        \
        cuda_kernel_cvt_per_elems<type_mode>(input, offset, output, tid, param);                        \
    } else {                                                                                            \
        cuda_kernel_set_zero_per_elems<type_mode>(output, tid);                                         \
    }                                                                                                   \
}
cvtTYPEALL(cvtSMCHANNELNCTONHWC8)

#define cvtSMCHANNELNHWC8TONC(type_mode)                                                                \
template<>                                                                                              \
__global__ void cuda_kernel_small_channel_cvtformat_type<type_mode, NHWC_NDARRAY>(                      \
    const void* input,                                                                                  \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    inner_fast.divmod(tid, num_inner, inner_idx);                                                       \
    c_idx = dst_pad_fast.mod(num_inner);                                                                \
    int outer_idx = tid / (param.dst_pad * param.n_inner);                                              \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx + inner_idx * param.src_pad;         \
    cuda_kernel_cvt_per_elems<type_mode>(input, offset, output, tid, param);                            \
}
cvtTYPEALL(cvtSMCHANNELNHWC8TONC)

#define cvtSMCHANNELN4CXTONC(type_mode)                                                                          \
template <>                                                                                                      \
__global__ void cuda_kernel_small_channel_cvtformat_type<type_mode, N4CX_NDARRAY>(                               \
    const void* input,                                                                                           \
    int num_elems,                                                                                               \
    DivModFast inner_fast,                                                                                       \
    DivModFast src_pad_fast,                                                                                     \
    DivModFast dst_pad_fast,                                                                                     \
    void* output,                                                                                                \
    ReFormatParam param)                                                                                         \
{                                                                                                                \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                       \
    if (tid >= num_elems)                                                                                        \
        return;                                                                                                  \
    int inner_idx, num_inner, c_idx;                                                                             \
    inner_fast.divmod(tid, num_inner, inner_idx);                                                                \
    src_pad_fast.divmod(num_inner, num_inner, c_idx);                                                            \
    const int c4_idx           = c_idx / 4;                                                                      \
    const int c_in_c4_idx      = c_idx % 4;                                                                      \
    const uint64_t size        = param.n_inner;                                                                  \
    const uint64_t padChannels = param.src_pad;                                                                  \
    const uint64_t numChannels = param.channel;                                                                  \
    const uint64_t offset      = num_inner * padChannels * size + (c4_idx * size + inner_idx) * 4 + c_in_c4_idx; \
    const uint64_t outOffset   = num_inner * numChannels * size + c_idx * size + inner_idx;                      \
    cuda_kernel_cvt_per_elems<type_mode>(input, offset, output, outOffset, param);                               \
}
cvtTYPEALL(cvtSMCHANNELN4CXTONC)

#define cvtSMCHANNELNCTON4CX(type_mode)                                                                          \
template <>                                                                                                      \
__global__ void cuda_kernel_small_channel_cvtformat_type<type_mode, NDARRAY_N4CX>(                                    \
    const void * input,                                                                                          \
    int num_elems,                                                                                               \
    DivModFast inner_fast,                                                                                       \
    DivModFast src_pad_fast,                                                                                     \
    DivModFast dst_pad_fast,                                                                                     \
    void* output,                                                                                                \
    ReFormatParam param)                                                                                         \
{                                                                                                                \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                       \
    if (tid >= num_elems)                                                                                        \
        return;                                                                                                  \
    int inner_idx, num_inner, c_idx;                                                                             \
    inner_fast.divmod(tid, num_inner, inner_idx);                                                                \
    src_pad_fast.divmod(num_inner, num_inner, c_idx);                                                            \
    const int c4_idx           = c_idx / 4;                                                                      \
    const int c_in_c4_idx      = c_idx % 4;                                                                      \
    const uint64_t size        = param.n_inner;                                                                  \
    const uint64_t padChannels = param.dst_pad;                                                                  \
    const uint64_t numChannels = param.channel;                                                                  \
    const uint64_t offset      = num_inner * padChannels * size + (c4_idx * size + inner_idx) * 4 + c_in_c4_idx; \
    const uint64_t inOffset    = num_inner * numChannels * size + c_idx * size + inner_idx;                      \
    cuda_kernel_cvt_per_elems<type_mode>(input, inOffset, output, offset, param);                                \
}
cvtTYPEALL(cvtSMCHANNELNCTON4CX)

#define cvtSMCHANNELNC1HWC0TONC(type_mode)                                                              \
template<>                                                                                              \
__global__ void cuda_kernel_small_channel_cvtformat_type<type_mode, NC1HWC0_NDARRAY>(                   \
    const void* input,                                                                                  \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    inner_fast.divmod(tid, num_inner, inner_idx);                                                       \
    c_idx = dst_pad_fast.mod(num_inner);                                                                \
    int outer_idx = tid / (param.dst_pad * param.n_inner);                                              \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx + inner_idx * param.src_pad;         \
    cuda_kernel_cvt_per_elems<type_mode>(input, offset, output, tid, param);                            \
}
cvtTYPEALL(cvtSMCHANNELNC1HWC0TONC)

#define cvtSMCHANNELNCTONC1HWC0(type_mode)                                                              \
template <>                                                                                             \
__global__ void cuda_kernel_small_channel_cvtformat_type<type_mode, NDARRAY_NC1HWC0>(                  \
    const void* input,                                                                                  \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    dst_pad_fast.divmod(tid, num_inner, c_idx);                                                         \
    inner_idx = inner_fast.mod(num_inner);                                                              \
    int outer_idx = inner_fast.div(num_inner);                                                          \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx * param.n_inner + inner_idx;         \
    if (c_idx < param.src_pad) {                                                                        \
        cuda_kernel_cvt_per_elems<type_mode>(input, offset, output, tid, param);                        \
    } else {                                                                                            \
        cuda_kernel_set_zero_per_elems<type_mode>(output, tid);                                         \
    }                                                                                                   \
}
cvtTYPEALL(cvtSMCHANNELNCTONC1HWC0)

#define MAX_DIM 65533
template<CVTFormatMode mode>
void GenDimParam(
    ReFormatParam param,
    dim3& dimBlock,
    dim3& dimGrid)
{
    dimGrid.z = param.n_outer >= MAX_DIM ? MAX_DIM : param.n_outer;
    if (mode == NHWC_NDARRAY) {
        dimBlock.x = DIM;
        dimBlock.y = DIM;
        dimGrid.x  = DivUp(param.src_pad, DIM);
        dimGrid.y  = DivUp(param.n_inner, DIM) > MAX_DIM? MAX_DIM : DivUp(param.n_inner, DIM);
    } else if (mode == NDARRAY_NHWC) {
        dimBlock.x = DIM;
        dimBlock.y = DIM;
        dimGrid.x  = DivUp(param.n_inner, DIM);
        dimGrid.y  = DivUp(param.dst_pad, DIM);
    } else if (mode == N4CX_NDARRAY) {
        dimBlock.x = DIM;
        dimBlock.y = 1;
        dimGrid.x  = DivUp(param.n_inner, DIM);
        dimGrid.y  = param.src_pad / 4;
    } else if (mode == NDARRAY_N4CX) {
        dimBlock.x = DIM;
        dimBlock.y = 1;
        dimGrid.x  = DivUp(param.n_inner, DIM);
        dimGrid.y  = param.dst_pad / 4;
    } else if (mode == NHWC8_NHWC16){
        dimBlock.x = DIM;
        dimBlock.y = DIM;
        dimGrid.x  = DivUp(param.dst_pad, DIM);
        dimGrid.y  = DivUp(param.n_inner, DIM);
    } else if (mode == NHWC16_NHWC8){
        dimBlock.x = DIM;
        dimBlock.y = DIM;
        dimGrid.x  = DivUp(param.dst_pad, DIM);
        dimGrid.y  = DivUp(param.n_inner, DIM);
    } else if (mode == NC1HWC0_NDARRAY) {
        dimBlock.x = DIM;
        dimBlock.y = 1;
        dimGrid.x  = DivUp(param.n_inner, DIM);
        dimGrid.y  = param.src_pad / 16;
    } else if (mode == NDARRAY_NC1HWC0) {
        dimBlock.x = DIM;
        dimBlock.y = 1;
        dimGrid.x  = DivUp(param.n_inner, DIM);
        dimGrid.y  = param.dst_pad / 16;
    }
}
#define RFC8C16              \
    case NHWC8_NHWC16:         \
        RUN(NHWC8_NHWC16);     \
    case NHWC16_NHWC8:         \
        RUN(NHWC16_NHWC8);     \
    case NC1HWC0_NHWC:         \
        RUN(NC1HWC0_NHWC);     \
    case NHWC_NC1HWC0:         \
        RUN(NHWC_NC1HWC0);     \
    case NC1HWC0_NC1HWC0:         \
        RUN(NC1HWC0_NC1HWC0);

#define RFNHWC                 \
    case NDARRAY_NHWC:         \
        RUN(NDARRAY_NHWC);     \
    case NHWC_NDARRAY:         \
        RUN(NHWC_NDARRAY);

#define RFN4CX             \
    case NDARRAY_N4CX:     \
        RUN(NDARRAY_N4CX); \
    case N4CX_NDARRAY:     \
        RUN(N4CX_NDARRAY);

#define RFNC1HWC0             \
    case NDARRAY_NC1HWC0:     \
        RUN(NDARRAY_NC1HWC0); \
    case NC1HWC0_NDARRAY:     \
        RUN(NC1HWC0_NDARRAY);

#ifdef OPT_CVT_FORMAT
#define MIN(a,b) ((a) < (b)?(a):(b))
__global__ void cuda_kernel_packed_cvtformat_type_opt_f32toi8(
    const float* input,
    float4* output,
    DivModFast inner_fast,
    int num_elems,
    int stride,
    int src_pad,
    double scale,
    char zeroPoint
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= num_elems) return;
    int b = 0, hw_idx = 0;
    int offset;
    const float* ptr_input = nullptr;
    inner_fast.divmod(tid, b, hw_idx);
    offset = b * stride * src_pad + hw_idx;
    ptr_input = input + offset;
    char val[16];
    int64_t*ptr_val = (int64_t*)val;
    ptr_val[0] = 0; ptr_val[1] = 0;

    #pragma unroll
    for(int i = 0; i < LEASTCHANNEL; i++){
        if(i < src_pad){
            float data = *ptr_input;
            ptr_input += stride;
            float tmp = data * scale + zeroPoint;
            tmp = min(tmp,127);
            tmp = max(tmp,-128);
            val[i] = (signed char)(__float2int_rn(tmp));
        }
    }
    float4* dst = (float4*)val;
    output[tid] = dst[0];
}

template <CVTTypeMode mode>
__global__ void cuda_kernel_packed_cvtformat_type_opt(
    const void *input,
    void *output,
    DivModFast inner_fast,
    int num_elems,
    ReFormatParam param) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elems) return;
    char val[16];
    int64_t*ptr_val = (int64_t*)val;
    ptr_val[0] = 0; ptr_val[1] = 0;

    int b = 0, hw_idx = 0;
    inner_fast.divmod(tid, b, hw_idx);
    int offset = b * param.n_inner * param.src_pad + hw_idx;
    #pragma unroll
    for (int i = 0; i < LEASTCHANNEL; i++) {
        if(i < param.src_pad)
        {
            cuda_kernel_cvt_per_elems<mode>(input, offset, val, i, param);
            offset += param.n_inner;
        }
    }
    float4* dst = (float4*)val;
    float4* dst_out = (float4*)output;
    dst_out[tid] = dst[0];
}

//2024.4.30 add FLAOT16_INT8
template <CVTTypeMode mode>
__global__ void cuda_kernel_packed_cvtformat_type_opt1(
    const void *input,
    void *output,
    DivModFast inner_fast,
    int num_elems,
    ReFormatParam param)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elems) return;
    __shared__ float4 shared_val[256];
    int b = 0, hw_idx = 0;
    inner_fast.divmod(tid, b, hw_idx);
    int offset = b * param.n_inner * param.src_pad + hw_idx;
    #pragma unroll
    for (int i = 0; i < LEASTCHANNEL; i++)
    {
        if (i < param.src_pad)
        {
            cuda_kernel_cvt_per_elems<mode>(input, offset,&shared_val[threadIdx.x], i, param);
            offset += param.n_inner;
        }
    }
    __syncthreads();
    float4 *dst_out = (float4 *)output;
    dst_out[tid] = shared_val[threadIdx.x];
}

__global__ void cuda_kernel_cvtformat_type_opt_nhwc16_8(const int8_t *input, half* output, DivModFast dst_pad_fast, ReFormatParam param)
{
    int64_t num = blockIdx.z;
    for (int n = num; n < param.n_outer; n += gridDim.z) {
        int tid = (blockIdx.x * blockDim.x + threadIdx.x) << 3;
        int idx_w, idx_h;
        dst_pad_fast.divmod(tid, idx_h, idx_w);
        if(idx_w < param.dst_pad && idx_h < param.n_inner){
            int64_t dst_offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;
            int64_t src_offset = n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;
            const int8_t* ptr_input = input + src_offset;
            half* ptr_output = output + dst_offset;
            int64_t tmp_input = *(int64_t*)ptr_input;
            int8_t* ptr_tmp_input = (int8_t*)&tmp_input;
            float4 tmp_out;
            half* ptr_tmp_out = (half*)&tmp_out;
            #pragma unroll 8
            for(int i = 0; i < 8; i++)
            {
                cuda_kernel_cvt_per_elems<INT8_FLOAT16>((const void*)ptr_tmp_input, i, (void*)ptr_tmp_out, i, param);
            }
            *(float4*)ptr_output = tmp_out;
        }
    }
}

__global__ void cuda_kernel_cvtformat_type_opt_nhwc8_16(const float *input, int8_t* output, DivModFast dst_pad_fast, ReFormatParam param)
{
    int64_t num = blockIdx.z;
    for (int n = num; n < param.n_outer; n += gridDim.z) {
        int tid = (blockIdx.x * blockDim.x + threadIdx.x) << 4;
        int idx_w, idx_h;
        dst_pad_fast.divmod(tid, idx_h, idx_w);
        if(idx_w < param.dst_pad && idx_h < param.n_inner){
            int64_t dst_offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;
            int64_t src_offset = n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;
            float4 reg0, reg1, reg2, reg3;
            float4 reg_dst;
            int8_t*ptr_dst = (int8_t*)&reg_dst;
            #pragma unroll 16
            for(int i = 0; i < 16; i++)
            {
                ptr_dst[i] = 0;
            }
            const float4 * ptr_input = (const float4*)(input + src_offset);
            float4* ptr_output = (float4*)(output + dst_offset);
            if(idx_w < param.src_pad){
                reg0 = *ptr_input++;
                float *ptr_reg = (float*)&reg0;
                #pragma unroll 4
                for(int i = 0; i < 4; i++)
                {
                    cuda_kernel_cvt_per_elems<FLOAT32_INT8>((const void*)ptr_reg,i,(void*)ptr_dst,i,param);
                }
            }
            if(idx_w + 4 < param.src_pad){
                reg1 = *ptr_input++;
                float *ptr_reg = (float*)&reg1;
                #pragma unroll 4
                for(int i = 0; i < 4; i++)
                {
                    cuda_kernel_cvt_per_elems<FLOAT32_INT8>((const void*)ptr_reg,i,(void*)ptr_dst,i + 4,param);
                }
            }
            if(idx_w + 8 < param.src_pad){
                reg2 = *ptr_input++;
                float *ptr_reg = (float*)&reg2;
                #pragma unroll 4
                for(int i = 0; i < 4; i++)
                {
                    cuda_kernel_cvt_per_elems<FLOAT32_INT8>((const void*)ptr_reg,i,(void*)ptr_dst,i + 8,param);
                }
            }
            if(idx_w + 12 < param.src_pad){
                reg3 = *ptr_input++;
                float *ptr_reg = (float*)&reg3;
                #pragma unroll 4
                for(int i = 0; i < 4; i++)
                {
                    cuda_kernel_cvt_per_elems<FLOAT32_INT8>((const void*)ptr_reg,i,(void*)ptr_dst,i + 12,param);
                }
            }
            *ptr_output = reg_dst;
        }
    }
}

__global__ void cuda_kernel_cvtformat_type_opt_nhwc8_16_c1(const half* input, int8_t* output,DivModFast dst_pad_fast,ReFormatParam param)
{
    int64_t num = blockIdx.z;
    float scale = 1.0 / param.o_step;
    float zp = param.o_zero_point;
    for(int n = num; n < param.n_outer; n += gridDim.z){
        int tid = (blockIdx.x * blockDim.x + threadIdx.x) << 4;
        int idx_w, idx_h;
        dst_pad_fast.divmod(tid, idx_h, idx_w);
        if(idx_w < param.dst_pad && idx_h < param.n_inner){
            int64_t dst_offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;
            int64_t src_offset = n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;
            float4 reg0, reg1;
            float4 reg_dst;
            reg_dst = make_float4(0.0,0.0,0.0,0.0);
            int8_t*ptr_dst = (int8_t*)&reg_dst;
            const float4 * ptr_input = (const float4*)(input + src_offset);
            float4* ptr_output = (float4*)(output + dst_offset);
            if(idx_w < param.src_pad){
                reg0 = *ptr_input++;
                half *ptr_reg = (half*)&reg0;
                #pragma unroll 8
                for(int i = 0; i < 8; i++)
                {
                    float tmp = (float)ptr_reg[i] * scale + zp;
                    int reg_i = tmp > 127 ? 127 : tmp < -128 ? -128
                                         : (__float2int_rn(tmp));   
                    ptr_dst[i] = reg_i;
                }
            }
            if(idx_w  + 8 < param.src_pad) {
                reg1 = *ptr_input;
                half *ptr_reg = (half*)&reg1;

                ptr_dst += 8;
                #pragma unroll 8
                for(int i = 0; i < 8; i++)
                {
                    float tmp = (float)ptr_reg[i] * scale + zp;
                    int reg_i = tmp > 127 ? 127 : tmp < -128 ? -128
                                         : (__float2int_rn(tmp));   
                    ptr_dst[i] = reg_i;
                }
            }
            *ptr_output = reg_dst;
        }
    }
}

//support NDARRAY_NHWC
template<typename T1,typename T2, typename T3, typename T4, CVTTypeMode t_mode>
__global__ void cuda_kernel_cvtformat_type_opt_ndarray_nhwc(
    void* input,
    void* output,
    ReFormatParam param)
{
    constexpr int times = sizeof(T2) / sizeof(T1);
    constexpr int N = 64 / times;
    __shared__ union {
        T1 m1[64][64];
        T2 m2[64][N];
    } sm_buffer;
    int64_t num = blockIdx.z;
    int64_t stride = blockDim.x * times;
    for(int n = num; n < param.n_outer; n += gridDim.z) {
        int64_t idx_w = blockIdx.x * 64;
        int64_t idx_h = blockIdx.y * 64 + threadIdx.y;
        T1* ptr_input = (T1 *)input + n * param.src_pad * param.n_inner + idx_h * param.n_inner + idx_w;
        T3* ptr_output = (T3 *)output + n * param.dst_pad * param.n_inner + (blockIdx.x * 64 + threadIdx.y) * param.dst_pad ;

        for(int64_t i = threadIdx.x; i < N; i += blockDim.x) {
            if(idx_w + i * times < param.n_inner && idx_h < param.src_pad) {
                T2* ptr_input_v = (T2 *)ptr_input + i;
                sm_buffer.m2[threadIdx.y][i] = *ptr_input_v;
            } else {
                T2 tmp1;
                T1* ptr_tmp1 = (T1*)&tmp1;
                #pragma unroll times
                for(int j = 0; j < times; j++){
                    ptr_tmp1[j] = (T1)0;
                }
                sm_buffer.m2[threadIdx.y][i] = tmp1;
            }
        }
        __syncthreads();

        idx_w = blockIdx.y * 64;
        for(int64_t i = threadIdx.x * times; i < 64; i += stride){
            if(idx_w + i < param.dst_pad && (blockIdx.x * 64 + threadIdx.y) < param.n_inner){
                union {
                    T4 m2;
                    T3 m1[times];
                }tmp_storage;
                #pragma unroll times
                for(int j = 0; j < times; j++){
                    cuda_kernel_cvt_per_elems<t_mode>((const void*)&sm_buffer.m1[i + j][threadIdx.y],
                        0, (void*)(tmp_storage.m1 + j), 0, param);    
                }
                *(T4*)(ptr_output + idx_w + i) = tmp_storage.m2;
            }
        }
    }
}

//support NDARRAY_NHWC
template<typename T1,typename T2, typename T3, typename T4, CVTTypeMode t_mode>
__global__ void cuda_kernel_cvtformat_type_opt_ndarray_nhwc_unalign(
    void* input,
    void* output,
    ReFormatParam param)
{
    constexpr int times = sizeof(T2) / sizeof(T1);
    __shared__ union {
        T1 m1[64][64];
        T2 m2[64][64 / times];
    } sm_buffer;
    int64_t num = blockIdx.z;
    for(int n = num; n < param.n_outer; n += gridDim.z) {
        int64_t idx_w = (blockIdx.x * blockDim.x + threadIdx.x) * times;
        int64_t idx_h = blockIdx.y * blockDim.y + threadIdx.y;
        T1* ptr_input = (T1 *)(input) + n * param.src_pad * param.n_inner;
        T3* ptr_output = (T3 *)(output) + n * param.dst_pad * param.n_inner;
        #pragma unroll times
        for(int i = 0; i < times; i++){
            if(idx_w + i < param.n_inner && idx_h < param.src_pad) {
                int64_t offset = idx_h * param.n_inner + idx_w + i;
                T1 *ptr_input_v = (T1*)(ptr_input + offset);
                sm_buffer.m1[threadIdx.y][threadIdx.x*times + i] = *ptr_input_v;
            }
            else{
                T1 tmp1 = (T1)0;
                sm_buffer.m1[threadIdx.y][threadIdx.x*times + i] = tmp1;
            }
        }
        __syncthreads();

        idx_w = blockIdx.y * blockDim.y + threadIdx.x * times;
        idx_h = (blockIdx.x * blockDim.x) * times + threadIdx.y;
        
        if(idx_w < param.dst_pad && idx_h < param.n_inner)
        {
            union {
                T4 m2;
                T3 m1[times];
            }tmp_storage;
            int64_t offset = idx_h * param.dst_pad + idx_w;
            #pragma unroll times
            for(int i = 0; i < times; i++)
            {
                cuda_kernel_cvt_per_elems<t_mode>((const void*)&sm_buffer.m1[threadIdx.x*times + i][threadIdx.y],
                    0, (void*)(tmp_storage.m1 + i), 0, param);
            }
            *(T4*)(ptr_output + offset) = tmp_storage.m2;
        }
    }
}

//support NHWC_NDARRAY
template<typename T1,typename T2, typename T3, typename T4, CVTTypeMode t_mode,int N>
__global__ void cuda_kernel_cvtformat_type_opt_nhwc_ndarray(
    void* input,
    void* output,
    ReFormatParam param)
{
    constexpr int times = sizeof(T2) / sizeof(T1); 
    __shared__ union {
        T1 m1[64][64];
        T2 m2[64][N];
    } sm_buffer;
    int64_t num = blockIdx.z;
    int64_t stride = blockDim.x * times;
    for(int n = num; n < param.n_outer; n += gridDim.z) {
        int64_t idx_w = blockIdx.x * 64;
        int64_t idx_h = blockIdx.y * 64 + threadIdx.y;
        T1* ptr_input = (T1 *)input + n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;
        T3* ptr_output = (T3 *)output + n * param.dst_pad * param.n_inner + (blockIdx.x * 64 + threadIdx.y) * param.n_inner;

        for(int64_t i = threadIdx.x; i < N; i += blockDim.x) {
            if(idx_w + i * times < param.src_pad && idx_h < param.n_inner) {
                T2* ptr_input_v = (T2 *)ptr_input + i;
                sm_buffer.m2[threadIdx.y][i] = *ptr_input_v;
            } else {
                T2 tmp1;
                T1* ptr_tmp1 = (T1*)&tmp1;
                #pragma unroll times
                for(int j = 0; j < times; j++){
                    ptr_tmp1[j] = (T1)0;
                }
                sm_buffer.m2[threadIdx.y][i] = tmp1;
            }
        }
        __syncthreads();

        idx_w = blockIdx.y * 64;
        for(int64_t i = threadIdx.x * times; i < 64; i += stride) {
            if(idx_w + i < param.n_inner && (blockIdx.x * 64 + threadIdx.y) < param.dst_pad){
                union {
                    T4 m2;
                    T3 m1[times];
                }tmp_storage;
                #pragma unroll times
                for(int j = 0; j < times; j++){
                    cuda_kernel_cvt_per_elems<t_mode>((const void*)&sm_buffer.m1[i + j][threadIdx.y],
                        0, (void*)(tmp_storage.m1 + j), 0, param);    
                }
                *(T4*)(ptr_output + idx_w + i) = tmp_storage.m2;
            }
        }
    }
}

template<typename T1,typename T2, typename T3, typename T4, CVTTypeMode t_mode,int N, int SHIFT>
__global__ void cuda_kernel_cvtformat_type_opt_fast_nhwc_ndarray(
    void* input,
    void* output,
    int gridDim_z,
    ReFormatParam param)
{
    constexpr int N1 = (32 >> SHIFT) + 1;
    __shared__ union {
        T1 m1[33][N1<<SHIFT];
        T2 m2[33][N1];
    } sm_buffer;
    int num = blockIdx.z;
    for(int n = num; n < param.n_outer; n += gridDim_z) {
        int offset_x = blockIdx.x * 32;
        int offset_y = blockIdx.y * 32;
        int idx_h = offset_y + threadIdx.y;
        T1* ptr_input = (T1 *)input + n * param.src_pad * param.n_inner + idx_h * param.src_pad + offset_x;
        T3* ptr_output = (T3 *)output + n * param.dst_pad * param.n_inner + (offset_x + threadIdx.y) * param.n_inner;
        int offset = threadIdx.x << SHIFT;
        if(offset_x + offset < param.src_pad && idx_h < param.n_inner) {
            T2* ptr_input_v = (T2 *)(ptr_input + offset);
            sm_buffer.m2[threadIdx.y][threadIdx.x] = *ptr_input_v;
        } else {
            T2 tmp1;
            T1* ptr_tmp1 = (T1*)&tmp1;
            #pragma unroll N
            for(int j = 0; j < N; j++){
                ptr_tmp1[j] = (T1)0;
            }
            sm_buffer.m2[threadIdx.y][threadIdx.x] = tmp1;
        }
        
        __syncthreads();

        if(offset_y + offset < param.n_inner && (offset_x + threadIdx.y) < param.dst_pad){
            union {
                T4 m2;
                T3 m1[N];
            }tmp_storage;
            #pragma unroll N
            for(int j = 0; j < N; j++){
                cuda_kernel_cvt_per_elems<t_mode>((const void*)&sm_buffer.m1[offset + j][threadIdx.y],
                    0, (void*)(tmp_storage.m1 + j), 0, param);    
            }
            *(T4*)(ptr_output + offset_y + offset) = tmp_storage.m2;
        }
    }
}

template<typename T1,typename T2, typename T3, typename T4, CVTTypeMode t_mode,int N>
__global__ void cuda_kernel_cvtformat_type_opt_nhwc_ndarray_unalign(
    void* input,
    void* output,
    ReFormatParam param)
{
    constexpr int times = sizeof(T2) / sizeof(T1);
    __shared__ union {
        T1 m1[64][64];
        T2 m2[64][N];
    } sm_buffer;
    int64_t num = blockIdx.z;
    int64_t stride = blockDim.x * times;
    for(int n = num; n < param.n_outer; n += gridDim.z) {
        int64_t idx_w = blockIdx.x * 64;
        int64_t idx_h = blockIdx.y * 64 + threadIdx.y;
        T1* ptr_input = (T1 *)input + n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;
        T3* ptr_output = (T3 *)output + n * param.dst_pad * param.n_inner + (blockIdx.x * 64 + threadIdx.y) * param.n_inner;

        for(int64_t i = threadIdx.x; i < N; i += blockDim.x) {
            if(idx_w + i * times < param.src_pad && idx_h < param.n_inner) {
                T2* ptr_input_v = (T2 *)ptr_input + i;
                sm_buffer.m2[threadIdx.y][i] = *ptr_input_v;
            } else {
                T2 tmp1;
                T1* ptr_tmp1 = (T1*)&tmp1;
                #pragma unroll times
                for(int j = 0; j < times; j++){
                    ptr_tmp1[j] = (T1)0;
                }
                sm_buffer.m2[threadIdx.y][i] = tmp1;
            }
        }
        __syncthreads();

        idx_w = blockIdx.y * 64;
        for(int64_t i = threadIdx.x * times; i < 64; i += stride){
            if(idx_w + i < param.n_inner && (blockIdx.x * 64 + threadIdx.y) < param.dst_pad){
                union {
                    T4 m2;
                    T3 m1[times];
                }tmp_storage;
                #pragma unroll times
                for(int j = 0; j < times; j++){
                    cuda_kernel_cvt_per_elems<t_mode>((const void*)&sm_buffer.m1[i + j][threadIdx.y],
                        0, (void*)(tmp_storage.m1 + j), 0, param);    
                }
                #pragma unroll times
                for(int j = 0; j < times; j++){
                    if(idx_w + i + j < param.n_inner){
                        ptr_output[idx_w + i + j] = tmp_storage.m1[j];
                    }
                }
            }
        }
    }
}

template<typename T1,typename T2, typename T3, typename T4, CVTTypeMode t_mode,int N, int SHIFT>
__global__ void cuda_kernel_cvtformat_type_opt_fast_nhwc_ndarray_unalign(
    void* input,
    void* output,
    int gridDim_z,
    ReFormatParam param)
{
    constexpr int N1 = (32 >> SHIFT) + 1;
    __shared__ union {
        T1 m1[33][N1<<SHIFT];
        T2 m2[33][N1];
    } sm_buffer;
    int num = blockIdx.z;
    for(int n = num; n < param.n_outer; n += gridDim_z) {
        int offset_x = blockIdx.x * 32;
        int offset_y = blockIdx.y * 32;
        int idx_h = offset_y + threadIdx.y;
        T1* ptr_input = (T1 *)input + n * param.src_pad * param.n_inner + idx_h * param.src_pad + offset_x;
        T3* ptr_output = (T3 *)output + n * param.dst_pad * param.n_inner + (offset_x + threadIdx.y) * param.n_inner;
        int offset = threadIdx.x << SHIFT;
        if(offset_x + offset < param.src_pad && idx_h < param.n_inner) {
            T2* ptr_input_v = (T2 *)(ptr_input + offset);
            sm_buffer.m2[threadIdx.y][threadIdx.x] = *ptr_input_v;
        } else {
            T2 tmp1;
            T1* ptr_tmp1 = (T1*)&tmp1;
            #pragma unroll N
            for(int j = 0; j < N; j++){
                ptr_tmp1[j] = (T1)0;
            }
            sm_buffer.m2[threadIdx.y][threadIdx.x] = tmp1;
        }
        
        __syncthreads();

        if(offset_y + offset < param.n_inner && (offset_x + threadIdx.y) < param.dst_pad){
            union {
                T4 m2;
                T3 m1[N];
            }tmp_storage;
            #pragma unroll N
            for(int j = 0; j < N; j++){
                cuda_kernel_cvt_per_elems<t_mode>((const void*)&sm_buffer.m1[offset + j][threadIdx.y],
                    0, (void*)(tmp_storage.m1 + j), 0, param);    
            }
            ptr_output += offset_y + offset;
            #pragma unroll N
            for(int j = 0; j < N; j++){
                if(offset_y + j + offset < param.n_inner){
                    ptr_output[j] = tmp_storage.m1[j];
                }
            }
        }
    }
}

template <typename T1, typename T2, CVTTypeMode t_mode, CVTFormatMode mode>
void call_cvtformat_type(const void* input, void* output, dim3 block_size,dim3 grid_size, ReFormatParam param, cudaStream_t stream)
{
    switch(mode){
        case NHWC16_NHWC8:
            if((param.dst_pad & 7)==0)
            {
                block_size.x = 256; block_size.y = 1;
                grid_size.x = (param.dst_pad * param.n_inner + 2047) >> 11; grid_size.y = 1;
                cuda_kernel_cvtformat_type_opt_nhwc16_8<<<grid_size, block_size, 0, stream>>>
                    ((const int8_t*)input, (half*)output, DivModFast(param.dst_pad), param);
            }else{
                cuda_kernel_cvtformat_type<t_mode, mode>
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);    
            }
            break;
        case NHWC8_NHWC16:
            switch(t_mode) {
                case FLOAT32_INT8:
                    if((param.dst_pad & 15)==0){
                        block_size.x = 256; block_size.y = 1;
                        grid_size.x = (param.dst_pad * param.n_inner + 4095) >> 12; grid_size.y = 1;
                        cuda_kernel_cvtformat_type_opt_nhwc8_16<<<grid_size, block_size, 0, stream>>>
                            ((const float*)input, (int8_t*)output, DivModFast(param.dst_pad), param);
                    }else{
                        cuda_kernel_cvtformat_type<t_mode, mode>
                            <<<grid_size, block_size, 0, stream>>>(input, output, param);
                    }
                    break;
                case FLOAT16_INT8:
                    if((param.dst_pad & 15) == 0){
                        block_size.x = 512; block_size.y = 1;
                        grid_size.x = (param.dst_pad * param.n_inner + 8191) >> 13; grid_size.y = 1;  
                        cuda_kernel_cvtformat_type_opt_nhwc8_16_c1<<<grid_size,block_size,0,stream>>>
                            ((const half*)input, (int8_t*)output,DivModFast(param.dst_pad),param);
                    }else{
                        cuda_kernel_cvtformat_type<t_mode, mode>
                            <<<grid_size, block_size, 0, stream>>>(input, output, param);
                    }
                    break;
                default:
                    cuda_kernel_cvtformat_type<t_mode, mode>
                            <<<grid_size, block_size, 0, stream>>>(input, output, param);
            }
            break;
        case NDARRAY_NHWC:
            switch(t_mode){
                case FLOAT32_FLOAT16:
                    if((param.n_inner % 4) == 0 && (param.dst_pad % 4) == 0){
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.n_inner + 63) >> 6;
                        grid_size.y = (param.dst_pad + 63) >> 6;
                        cuda_kernel_cvtformat_type_opt_ndarray_nhwc<float,float4,half,float2,FLOAT32_FLOAT16><<<grid_size, block_size, 0, stream>>>
                            ((void*)input, output, param);
                    } else if((param.dst_pad % 4) == 0){
                        block_size.x = 8; block_size.y = 32;
                        grid_size.x = (param.n_inner + 31) >> 5;
                        grid_size.y = (param.dst_pad + 31) >> 5;
                        cuda_kernel_cvtformat_type_opt_ndarray_nhwc_unalign<float,float4,half,float2,FLOAT32_FLOAT16><<<grid_size, block_size, 0, stream>>>
                            ((void*)input, output, param);
                    } else {
                        cuda_kernel_cvtformat_type<t_mode, mode>
                            <<<grid_size, block_size, 0, stream>>>(input, output, param);
                    }
                    break;
                case FLOAT16_INT8:
                    if((param.n_inner % 8) == 0 && (param.dst_pad % 8) == 0){
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.n_inner + 63) >> 6;
                        grid_size.y = (param.dst_pad + 63) >> 6;
                        cuda_kernel_cvtformat_type_opt_ndarray_nhwc<half,float4,int8_t,float2,FLOAT16_INT8><<<grid_size, block_size, 0, stream>>>
                            ((void*)input, output, param);
                    } else if((param.n_inner % 4) == 0 && (param.dst_pad % 4) == 0) {
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.n_inner + 63) >> 6;
                        grid_size.y = (param.dst_pad + 63) >> 6;
                        cuda_kernel_cvtformat_type_opt_ndarray_nhwc<half,float2,int8_t,float,FLOAT16_INT8><<<grid_size, block_size, 0, stream>>>
                            ((void*)input, output, param);
                    } else if((param.dst_pad % 8) == 0){
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.n_inner + 63) >> 6;
                        grid_size.y = (param.dst_pad + 63) >> 6;
                       cuda_kernel_cvtformat_type_opt_ndarray_nhwc_unalign<half,float4,int8_t,float2,FLOAT16_INT8><<<grid_size, block_size, 0, stream>>>
                            ((void*)input, output, param);
                    } else {
                        cuda_kernel_cvtformat_type<t_mode, mode>
                            <<<grid_size, block_size, 0, stream>>>(input, output, param);
                    }
                    break;
                case FLOAT32_INT8:
                    if((param.n_inner % 4) == 0 && (param.dst_pad % 4) == 0) {
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.n_inner + 63) >> 6;
                        grid_size.y = (param.dst_pad + 63) >> 6;
                        cuda_kernel_cvtformat_type_opt_ndarray_nhwc<float,float4,int8_t,float,FLOAT32_INT8><<<grid_size, block_size, 0, stream>>>
                            ((void*)input, output, param);
                    } else if((param.dst_pad % 4) == 0) {
                        block_size.x = 8; block_size.y = 32;
                        grid_size.x = (param.n_inner + 31) >> 5;
                        grid_size.y = (param.dst_pad + 31) >> 5;
                        cuda_kernel_cvtformat_type_opt_ndarray_nhwc_unalign<float,float4,int8_t,float,FLOAT32_INT8><<<grid_size, block_size, 0, stream>>>
                        ((void*)input, output, param);
                    } else {
                        cuda_kernel_cvtformat_type<t_mode, mode>
                            <<<grid_size, block_size, 0, stream>>>(input, output, param);
                    }
                    break;
                case INT8_FLOAT16:
                    if((param.n_inner % 8) == 0 && (param.dst_pad % 8) == 0){
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.n_inner + 63) >> 6;
                        grid_size.y = (param.dst_pad + 63) >> 6;
                        cuda_kernel_cvtformat_type_opt_ndarray_nhwc<int8_t,float2,half,float4,INT8_FLOAT16><<<grid_size, block_size, 0, stream>>>
                            ((void*)input, output, param);
                    } else if((param.n_inner % 4) == 0 && (param.dst_pad % 4) == 0){
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.n_inner + 63) >> 6;
                        grid_size.y = (param.dst_pad + 63) >> 6;
                        cuda_kernel_cvtformat_type_opt_ndarray_nhwc<int8_t,float,half,float2,INT8_FLOAT16><<<grid_size, block_size, 0, stream>>>
                            ((void*)input, output, param);
                    } else if((param.dst_pad % 8) == 0){
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.n_inner + 63) >> 6;
                        grid_size.y = (param.dst_pad + 63) >> 6;
                        cuda_kernel_cvtformat_type_opt_ndarray_nhwc_unalign<int8_t,float2,half,float4,INT8_FLOAT16><<<grid_size, block_size, 0, stream>>>
                            ((void*)input, output, param);
                    } else {
                        cuda_kernel_cvtformat_type<t_mode, mode>
                            <<<grid_size, block_size, 0, stream>>>(input, output, param);
                    }
                    break;
                default:
                    cuda_kernel_cvtformat_type<t_mode, mode>
                        <<<grid_size, block_size, 0, stream>>>(input, output, param);
                    break;
            }
            break;
        case NHWC_NDARRAY:
            switch(t_mode) {
                case FLOAT16_FLOAT32:
                    if((param.n_inner % 4) == 0 && (param.src_pad % 4) == 0){
                        block_size.x = 8; block_size.y = 32;
                        grid_size.x = (param.src_pad + 31) >> 5;
                        grid_size.y = (param.n_inner + 31) >> 5;
                        cuda_kernel_cvtformat_type_opt_fast_nhwc_ndarray<half,float2,float,float4,FLOAT16_FLOAT32, 4, 2><<<grid_size,block_size,0,stream>>>
                            ((void*)input,output,grid_size.z,param);
                     } else if((param.src_pad % 4) == 0) {
                        block_size.x = 8; block_size.y = 32;
                        grid_size.x = (param.src_pad + 31) >> 5;
                        grid_size.y = (param.n_inner + 31) >> 5;
                        cuda_kernel_cvtformat_type_opt_fast_nhwc_ndarray_unalign<half,float2,float,float4,FLOAT16_FLOAT32, 4, 2><<<grid_size,block_size,0,stream>>>
                            ((void*)input,output,grid_size.z, param);
                    } else {
                        cuda_kernel_cvtformat_type<t_mode, mode>
                            <<<grid_size, block_size, 0, stream>>>(input, output, param);
                    }
                    break;
                case INT8_FLOAT16:
                    if((param.n_inner % 8) == 0 && (param.src_pad % 8) == 0){
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.src_pad + 63) >> 6;
                        grid_size.y = (param.n_inner + 63) >> 6;
                        cuda_kernel_cvtformat_type_opt_nhwc_ndarray<int8_t,float2,half,float4,INT8_FLOAT16,8><<<grid_size,block_size,0,stream>>>
                            ((void*)input,output,param);
                     } else if((param.src_pad % 8) == 0) {
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.src_pad + 63) >> 6;
                        grid_size.y = (param.n_inner + 63) >> 6;
                        cuda_kernel_cvtformat_type_opt_nhwc_ndarray_unalign<int8_t,float2,half,float4,INT8_FLOAT16,8><<<grid_size,block_size,0,stream>>>
                            ((void*)input,output,param);
                    } else {
                        cuda_kernel_cvtformat_type<t_mode, mode>
                            <<<grid_size, block_size, 0, stream>>>(input, output, param);
                    }
                    break;
                case FLOAT16_INT8:
                    if((param.n_inner % 8) == 0 && (param.src_pad % 8) == 0){
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.src_pad + 63) >> 6;
                        grid_size.y = (param.n_inner + 63) >> 6;
                        cuda_kernel_cvtformat_type_opt_nhwc_ndarray<half,float4,int8_t,float2,FLOAT16_INT8,8><<<grid_size,block_size,0,stream>>>
                            ((void*)input,output,param);
                     } else if((param.src_pad % 8) == 0) {
                        block_size.x = 8; block_size.y = 64;
                        grid_size.x = (param.src_pad + 63) >> 6;
                        grid_size.y = (param.n_inner + 63) >> 6;
                        cuda_kernel_cvtformat_type_opt_nhwc_ndarray_unalign<half,float4,int8_t,float2,FLOAT16_INT8,8><<<grid_size,block_size,0,stream>>>
                            ((void*)input,output,param);
                    } else {
                        cuda_kernel_cvtformat_type<t_mode, mode>
                            <<<grid_size, block_size, 0, stream>>>(input, output, param);
                    }
                    break;
                default:
                    cuda_kernel_cvtformat_type<t_mode, mode>
                        <<<grid_size, block_size, 0, stream>>>(input, output, param);
                    break;
            }
            break;
        default:
            cuda_kernel_cvtformat_type<t_mode, mode>
                <<<grid_size, block_size, 0, stream>>>(input, output, param);
         break;
    }
    return;
}

//blockSize 8x64 or 16x64
template<typename src_T, typename dst_T,int shift, int N>
__global__ void cuda_kernel_small_channel_cvtformat_type_opt_fp32_16_NDARRAY_NHWC(
    const void* input,
    int num_elems,
    void* output,
    float divPadChannel,
    ReFormatParam param)
{
    constexpr int smLength = 512 >> shift;
    __shared__ union {
        half m1[16][512];
        dst_T m2[16][smLength];
        float4 m3[16][64];
        float2 m4[16][128];
    } sm_buffer;

    int iBlockOffset = (blockIdx.x * blockDim.x) << 3;
    int tidy = threadIdx.y;
    int src_offset = blockIdx.y * param.src_pad * param.n_inner + iBlockOffset;
    int dst_offset = blockIdx.y * param.dst_pad * param.n_inner + iBlockOffset * param.dst_pad;
    float *ptr_input = (float*)input + src_offset + tidy * param.n_inner;
    half* ptr_block_output = (half*)output + dst_offset;
    int height = min(512,param.n_inner - iBlockOffset);
    int blockLength = height>>shift;
    
    if(tidy < param.src_pad) {
        for(int i = threadIdx.x; i < blockLength; i += blockDim.x)  
        {
            src_T vInput = *((src_T*)ptr_input + i);
            dst_T vOutput;
            float * ptr_reg_src = (float *)&vInput;
            half * ptr_reg_dst = (half*)&vOutput;
            #pragma unroll N
            for(int j = 0; j < N; j++)
            {
                ptr_reg_dst[j] = __float2half(ptr_reg_src[j]);
            }
            sm_buffer.m2[tidy][i] = vOutput;
        }
    } else {
        float4 reg_zero = make_float4(0.0f,0.0f,0.0f,0.0f);
        sm_buffer.m3[tidy][threadIdx.x] = reg_zero; 
    }
    __syncthreads();

    int tid = (threadIdx.y * blockDim.x + threadIdx.x)<<3;
    int out_y = tid * divPadChannel;
    int out_x = tid & (param.dst_pad - 1);
    half* ptr_output = ptr_block_output + out_y * param.dst_pad + out_x;
    if(out_y < height)
    {
        float4 dst;
        half* ptr_reg_dst = (half*)&dst;
        #pragma unroll 8
        for(int i = 0; i < 8; i++)
        {
            ptr_reg_dst[i] = sm_buffer.m1[out_x  + i][out_y];
        }
        *(float4*)ptr_output = dst;
    }
}

template<typename src_T, typename dst_T,int shift, int N>
__global__ void cuda_kernel_small_channel_cvtformat_type_opt_c4_fp32_16_NDARRAY_NHWC(
    const void* input,
    void* output,
    int block_width,
    int remain_width,
    DivModFast block_width_fast,
    DivModFast remain_width_fast,
    DivModFast dst_pad_fast,
    ReFormatParam param)
{
    __shared__ float sm_buffer[2048];

    int iBlockOffset = blockIdx.x * block_width;
    int src_offset = blockIdx.y * param.src_pad * param.n_inner + iBlockOffset;
    int dst_offset = blockIdx.y * param.dst_pad * param.n_inner + iBlockOffset * param.dst_pad;
    int dst_blockSize;
    float* ptr_input = (float*)input + src_offset;
    half* ptr_block_output = (half*)output + dst_offset;
    src_T reg_zero;
    float*ptr_zero = (float*)&reg_zero;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        ptr_zero[j] = 0.0f;
    }
    if(blockIdx.x == gridDim.x - 1) {
        int blockSize = remain_width * param.src_pad;
        int i = threadIdx.x;
        dst_blockSize = remain_width * param.dst_pad;
        blockSize = blockSize >> shift;
        dst_blockSize = dst_blockSize >> shift;
        for(; i < blockSize; i += blockDim.x) {
            int h, w;
            remain_width_fast.divmod(i<<shift, h, w);
            src_T reg_input = *(src_T*)(ptr_input + h * param.n_inner + w);
            *(src_T*)(sm_buffer + h * block_width + w) = reg_input;
        }   
        for(; i < dst_blockSize; i += blockDim.x) {
            int h, w;
            remain_width_fast.divmod(i<<shift, h, w);
            *(src_T*)(sm_buffer + h*block_width + w) = reg_zero;
        }
    } else {
        int blockSize = block_width * param.src_pad;
        int i = threadIdx.x;
        dst_blockSize = block_width * param.dst_pad;
        blockSize = blockSize >> shift;
        dst_blockSize = dst_blockSize >> shift;
        for(; i < blockSize; i += blockDim.x) {
            int h, w;
            block_width_fast.divmod(i<<shift, h, w);
            src_T reg_input = *(src_T*)(ptr_input + h * param.n_inner + w);
            *(src_T*)(sm_buffer + h * block_width + w) = reg_input;
        }
        for(; i < dst_blockSize; i += blockDim.x) {
            int h, w;
            block_width_fast.divmod(i<<shift, h, w);
            *(src_T*)(sm_buffer + h * block_width + w) = reg_zero;
        }
    }
    __syncthreads();
    dst_T reg_dst;
    half* ptr_dst = (half*)&reg_dst;
    for(int i = threadIdx.x; i < dst_blockSize; i += blockDim.x) {
        int h, w;
        dst_pad_fast.divmod(i<<shift, h, w);
        #pragma unroll N
        for(int j = 0; j < N; j++) {
            ptr_dst[j] = sm_buffer[(w + j)* block_width + h];
        }
        *(dst_T*)(ptr_block_output + (i << shift)) = reg_dst;
    }
}

template<typename src_T, typename dst_T,int dst_shift, int N>
__global__ void cuda_kernel_small_channel_cvtformat_type_opt_c4_fp32_16_unalign_NDARRAY_NHWC(
    const void* input,
    void* output,
    int block_width,
    int remain_width,
    DivModFast block_width_fast,
    DivModFast remain_width_fast,
    DivModFast dst_pad_fast,
    ReFormatParam param)
{
    __shared__ float sm_buffer[2048];
    int iBlockOffset = blockIdx.x * block_width;
    int src_offset = blockIdx.y * param.src_pad * param.n_inner + iBlockOffset;
    int dst_offset = blockIdx.y * param.dst_pad * param.n_inner + iBlockOffset * param.dst_pad;
    int dst_blockSize, blockSize;
    float* ptr_input = (float*)input + src_offset;
    half* ptr_block_output = (half*)output + dst_offset;
    float4 reg_zero = make_float4(0,0,0,0);
    src_T reg0 = *(src_T *)(&reg_zero);
    if(blockIdx.x == gridDim.x - 1) {
        dst_blockSize = remain_width * param.dst_pad;
        blockSize = remain_width * param.src_pad;
        int i = threadIdx.x;
        for(; i < blockSize; i += blockDim.x) {
            int h, w;
            remain_width_fast.divmod(i, h,w);
            float reg_input = *(ptr_input + h * param.n_inner + w);
            *(sm_buffer + h * block_width + w) = reg_input;
        }   
        for(; i < dst_blockSize; i += blockDim.x) {
            int h, w;
            remain_width_fast.divmod(i, h,w);
            *(sm_buffer + h*block_width + w) = 0;
        }
        dst_blockSize = dst_blockSize >> dst_shift; 
    } else {
        dst_blockSize = block_width * param.dst_pad;
        blockSize = block_width * param.src_pad;
        blockSize = blockSize >> dst_shift;
        dst_blockSize = dst_blockSize >> dst_shift;
        int i = threadIdx.x;
        for(; i < blockSize; i += blockDim.x) {
            int h, w;
            block_width_fast.divmod((i << dst_shift), h, w);
            float * ptr_reg_input = ptr_input + h * param.n_inner + w;
            float * ptr_sm_buffer = sm_buffer + h * block_width + w;
            #pragma unroll N
            for(int j = 0; j < N; j++) {
                float reg = *(ptr_reg_input + j);
                *(ptr_sm_buffer + j) = reg;
            }
        }
        for(; i < dst_blockSize; i += blockDim.x) {
            int h, w;
            block_width_fast.divmod((i << dst_shift), h, w);
            float * ptr_sm_buffer = sm_buffer + h * block_width + w;
            *(src_T *)(ptr_sm_buffer) = reg0;
        }
    }
    __syncthreads();

    for(int i = threadIdx.x; i < dst_blockSize; i += blockDim.x) {
        int h, w; dst_T reg_dst;
        half* ptr_dst = (half*)&reg_dst;
        dst_pad_fast.divmod((i << dst_shift), h, w);
        float * ptr_sm_buffer = sm_buffer + w * block_width + h;
        #pragma unroll N
        for(int j = 0; j < N; j++) {
            ptr_dst[j] = *ptr_sm_buffer;
            ptr_sm_buffer += block_width;
        }
        *(dst_T*)(ptr_block_output + (i << dst_shift)) = reg_dst;
    }
}

template<typename src_T, typename dst_T,int shift, int N>
__global__ void cuda_kernel_small_channel_cvtformat_type_opt_c4_fp32_8_NDARRAY_NHWC(
    const void* input,
    void* output,
    int block_width,
    int remain_width,
    DivModFast block_width_fast,
    DivModFast remain_width_fast,
    DivModFast dst_pad_fast,
    ReFormatParam param)
{
    __shared__ float sm_buffer[2048];

    int iBlockOffset = blockIdx.x * block_width;
    int src_offset = blockIdx.y * param.src_pad * param.n_inner + iBlockOffset;
    int dst_offset = blockIdx.y * param.dst_pad * param.n_inner + iBlockOffset * param.dst_pad;
    int dst_blockSize;
    float* ptr_input = (float*)input + src_offset;
    int8_t* ptr_block_output = (int8_t*)output + dst_offset;
    src_T reg_zero;
    float*ptr_zero = (float*)&reg_zero;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        ptr_zero[j] = 0.0f;
    }
    if(blockIdx.x == gridDim.x - 1) {
        int blockSize = remain_width * param.src_pad;
        int i = threadIdx.x;
        dst_blockSize = remain_width * param.dst_pad;
        blockSize = blockSize >> shift;
        dst_blockSize = dst_blockSize >> shift;
        for(; i < blockSize; i += blockDim.x) {
            int h, w;
            remain_width_fast.divmod(i<<shift, h, w);
            src_T reg_input = *(src_T*)(ptr_input + h * param.n_inner + w);
            *(src_T*)(sm_buffer + h * block_width + w) = reg_input;
        }   
        for(; i < dst_blockSize; i += blockDim.x) {
            int h, w;
            remain_width_fast.divmod(i<<shift, h, w);
            *(src_T*)(sm_buffer + h*block_width + w) = reg_zero;
        }
    } else {
        int blockSize = block_width * param.src_pad;
        int i = threadIdx.x;
        dst_blockSize = block_width * param.dst_pad;
        blockSize = blockSize >> shift;
        dst_blockSize = dst_blockSize >> shift;
        for(; i < blockSize; i += blockDim.x) {
            int h, w;
            block_width_fast.divmod(i<<shift, h, w);
            src_T reg_input = *(src_T*)(ptr_input + h * param.n_inner + w);
            *(src_T*)(sm_buffer + h * block_width + w) = reg_input;
        }
        for(; i < dst_blockSize; i += blockDim.x) {
            int h, w;
            block_width_fast.divmod(i<<shift, h, w);
            *(src_T*)(sm_buffer + h * block_width + w) = reg_zero;
        }
    }
    __syncthreads();

    float scale = 1.0 / param.o_step;
    float zp = param.o_zero_point;
    for(int i = threadIdx.x; i < dst_blockSize; i += blockDim.x) {
        int h, w; dst_T reg_dst;
        int8_t* ptr_dst = (int8_t*)&reg_dst;
        dst_pad_fast.divmod(i<<shift, h, w);
        float * ptr_sm_buffer = sm_buffer + w * block_width + h;
        #pragma unroll N
        for(int j = 0; j < N; j++) {
            int reg_i;
            float reg = *ptr_sm_buffer;
            ptr_sm_buffer += block_width;
            reg = reg * scale + zp;
            reg_i = reg > 127 ? 127 : reg < -128 ? -128
                                         : (__float2int_rn(reg));
            ptr_dst[j] = reg_i;
        }
        *(dst_T*)(ptr_block_output + (i << shift)) = reg_dst;
    }
}

template<typename src_T, typename dst_T,int dst_shift, int N>
__global__ void cuda_kernel_small_channel_cvtformat_type_opt_c4_fp32_8_unalign_NDARRAY_NHWC(
    const void* input,
    void* output,
    int block_width,
    int remain_width,
    DivModFast block_width_fast,
    DivModFast remain_width_fast,
    DivModFast dst_pad_fast,
    ReFormatParam param)
{
    __shared__ float sm_buffer[2048];
    
    int iBlockOffset = blockIdx.x * block_width;
    int src_offset = blockIdx.y * param.src_pad * param.n_inner + iBlockOffset;
    int dst_offset = blockIdx.y * param.dst_pad * param.n_inner + iBlockOffset * param.dst_pad;
    int dst_blockSize, blockSize;
    float* ptr_input = (float*)input + src_offset;
    int8_t* ptr_block_output = (int8_t*)output + dst_offset;
    float4 reg_zero = make_float4(0,0,0,0);
    src_T reg0 = *(src_T *)(&reg_zero);
    if(blockIdx.x == gridDim.x - 1) {
        dst_blockSize = remain_width * param.dst_pad;
        blockSize = remain_width * param.src_pad;
        int i = threadIdx.x;
        for(; i < blockSize; i += blockDim.x) {
            int h, w;
            remain_width_fast.divmod(i, h, w);
            float reg_input = *(ptr_input + h * param.n_inner + w);
            *(sm_buffer + h * block_width + w) = reg_input;
        }   
        for(; i < dst_blockSize; i += blockDim.x) {
            int h, w;
            remain_width_fast.divmod(i, h, w);
            *(sm_buffer + h*block_width + w) = 0;
        }
        dst_blockSize = dst_blockSize >> dst_shift;
    } else {
        dst_blockSize = block_width * param.dst_pad;
        blockSize = block_width * param.src_pad;
        blockSize = blockSize >> dst_shift;
        dst_blockSize = dst_blockSize >> dst_shift;
        int i = threadIdx.x;
        for(; i < blockSize; i += blockDim.x) {
            int h, w;
            block_width_fast.divmod(i << dst_shift, h, w);
            float * ptr_reg_input = ptr_input + h * param.n_inner + w;
            float * ptr_sm_buffer = sm_buffer + h * block_width + w;
            #pragma unroll N
            for(int j = 0; j < N; j++) {
                float reg = *(ptr_reg_input + j);
                *(ptr_sm_buffer + j) = reg;
            }
        }
        for(; i < dst_blockSize; i += blockDim.x) {
            int h, w;
            block_width_fast.divmod(i << dst_shift, h, w);
            float * ptr_sm_buffer = sm_buffer + h * block_width + w;
            *(src_T *)(ptr_sm_buffer) = reg0;
        }
    }
    __syncthreads();
    float scale = 1.0 / param.o_step;
    float zp = param.o_zero_point;
    for(int i = threadIdx.x; i < dst_blockSize; i += blockDim.x) {
        int h, w; dst_T reg_dst;
        int8_t* ptr_dst = (int8_t*)&reg_dst;
        dst_pad_fast.divmod(i<<dst_shift, h, w);
        float * ptr_sm_buffer = sm_buffer + w * block_width + h;
        #pragma unroll N
        for(int j = 0; j < N; j++) {
            int reg_i;
            float reg = *ptr_sm_buffer;
            ptr_sm_buffer += block_width;
            reg = reg * scale + zp;
            reg_i = reg > 127 ? 127 : reg < -128 ? -128
                                         : (__float2int_rn(reg));
            ptr_dst[j] = reg_i;
        }
        *(dst_T*)(ptr_block_output + (i << dst_shift)) = reg_dst;
    }
}
#endif//OPT_CVT_FORMAT

template <CVTTypeMode mode>
__global__ void cuda_kernel_packed_cvtformat_type(
    const void *input,
    void *output,
    DivModFast inner_fast,
    int num_elems,
    ReFormatParam param) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elems) return;
    char val[16];
    _Pragma("unroll")
    for (int i = 0; i < 16; i++) {
        val[i] = 0;
    }
    int b = 0, hw_idx = 0;
    inner_fast.divmod(tid, b, hw_idx);
    int offset = b * param.n_inner * param.src_pad + hw_idx;
    for (int i = 0; i < param.src_pad; i++) {
        cuda_kernel_cvt_per_elems<mode>(input, offset, val, i, param);
        offset += param.n_inner;
    }
    float4* dst = (float4*)val;
    float4* dst_out = (float4*)output;
    dst_out[tid] = dst[0];
}

void PPLCUDASmallChannelCVTPackedFormatType(cudaStream_t stream, const void *input, void *output, ReFormatParam param)
{
    dim3 dimBlock(256, 1, 1);
    int num_elems = param.out_elems / param.dst_pad;
    dim3 dimGrid(DivUp(num_elems, 256), 1, 1);
    DivModFast inner_fast(param.n_inner);
    switch (GetCVTTypeMode(param)) {
        case FLOAT32_INT8:
#ifdef OPT_CVT_FORMAT
            cuda_kernel_packed_cvtformat_type_opt_f32toi8<<<dimGrid, dimBlock, 0, stream>>>((const float*)input,(float4*)output,
                inner_fast,num_elems, param.n_inner, param.src_pad, (double(1.0)) / param.o_step, param.o_zero_point);
#else//!OPT_CVT_FORMAT
            cuda_kernel_packed_cvtformat_type<FLOAT32_INT8><<<dimGrid, dimBlock, 0, stream>>>(input, output, inner_fast, num_elems, param);
#endif//OPT_CVT_FORMAT
            break;
        case FLOAT16_INT8:
#ifdef OPT_CVT_FORMAT
            // cuda_kernel_packed_cvtformat_type_opt<FLOAT16_INT8><<<dimGrid, dimBlock, 0, stream>>>(input, output, inner_fast, num_elems, param);
            cuda_kernel_packed_cvtformat_type_opt1<FLOAT16_INT8><<<dimGrid, dimBlock, 0, stream>>>(input, output, inner_fast, num_elems, param);
#else//!OPT_CVT_FORMAT
            cuda_kernel_packed_cvtformat_type<FLOAT16_INT8><<<dimGrid, dimBlock, 0, stream>>>(input, output, inner_fast, num_elems, param);
#endif//OPT_CVT_FORMAT
            break;
        case INT8_INT8:
#ifdef OPT_CVT_FORMAT
            cuda_kernel_packed_cvtformat_type_opt<INT8_INT8><<<dimGrid, dimBlock, 0, stream>>>(input, output, inner_fast, num_elems, param);
#else//!OPT_CVT_FORMAT
            cuda_kernel_packed_cvtformat_type<INT8_INT8><<<dimGrid, dimBlock, 0, stream>>>(input, output, inner_fast, num_elems, param);
#endif//OPT_CVT_FORMAT
            break;
        default:
            break;
    }
}
void PPLCUDASmallChannelCVTFormatType(cudaStream_t stream, const void *input, void *output, ReFormatParam param)
{
    if (param.out_type == ppl::common::DATATYPE_INT8 && (param.out_format == ppl::common::DATAFORMAT_NHWC16 || param.out_format == ppl::common::DATAFORMAT_NCHW16)
        && param.in_format == ppl::common::DATAFORMAT_NDARRAY) {
        #if 0
            if(param.out_format == ppl::common::DATAFORMAT_NCHW16){
                LOG(ERROR) << "Need support func like PPLCUDASmallChannelCVTPackedFormat for NCHW16";
                return;
            }
        #endif
            PPLCUDASmallChannelCVTPackedFormatType(stream, input, output, param);
            return; 
        }
#ifdef OPT_CVT_FORMAT
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 block_size(256, 1, 1);                                                   \
        int num_elems = param.out_elems;                                              \
        dim3 grid_size(DivUp(num_elems, 256), 1, 1);                                  \
        DivModFast inner_fast(param.n_inner);                                         \
        DivModFast src_pad_fast(param.src_pad);                                       \
        DivModFast dst_pad_fast(param.dst_pad);                                       \
        switch (GetCVTTypeMode(param)) {                                              \
            case FLOAT32_INT8:                                                                  \
                switch(mode){                                                                   \
                    case NDARRAY_NHWC:                                                          \
                    if((param.dst_pad & 3) == 0) {                                                                                                                                  \
                        if((param.n_inner&3) == 0) {                                                                                                                                \
                            int block_width = 2048 / param.dst_pad;                                                                                                                 \
                            int remain_width;                                                                                                                                       \
                            block_width = block_width >> 2 << 2;                                                                                                                    \
                            dim3 blockSize(512,1,1);                                                                                                                                \
                            dim3 gridSize((param.n_inner + block_width - 1) / block_width,param.n_outer,1);                                                                         \
                            remain_width = MIN((param.n_inner - (gridSize.x - 1)*block_width), block_width);                                                                        \
                            cuda_kernel_small_channel_cvtformat_type_opt_c4_fp32_8_NDARRAY_NHWC<float4,float,2,4><<<gridSize,blockSize,0,stream>>>(                                 \
                                (const void*)input,output, block_width, remain_width, DivModFast(block_width), DivModFast(remain_width), DivModFast(param.dst_pad), param);         \
                        } else if((param.n_inner&1) == 0) {                                                                                                                         \
                            int block_width = 2048 / param.dst_pad;                                                                                                                 \
                            int remain_width;                                                                                                                                       \
                            block_width = block_width >> 1 << 1;                                                                                                                    \
                            dim3 blockSize(512,1,1);                                                                                                                                \
                            dim3 gridSize((param.n_inner + block_width - 1) / block_width,param.n_outer,1);                                                                         \
                            remain_width = MIN((param.n_inner - (gridSize.x - 1)*block_width), block_width);                                                                        \
                            cuda_kernel_small_channel_cvtformat_type_opt_c4_fp32_8_NDARRAY_NHWC<float2,half,1,2><<<gridSize,blockSize,0,stream>>>(                                  \
                                (const void*)input,output, block_width, remain_width, DivModFast(block_width), DivModFast(remain_width), DivModFast(param.dst_pad), param);         \
                        } else {                                                                                                                                                    \
                            int block_width = 2048 / param.dst_pad;                                                                                                                 \
                            block_width = block_width >> 2 << 2;                                                                                                                    \
                            int remain_width;                                                                                                                                       \
                            dim3 blockSize(512,1,1);                                                                                                                                \
                            dim3 gridSize((param.n_inner + block_width - 1) / block_width,param.n_outer,1);                                                                         \
                            remain_width = MIN((param.n_inner - (gridSize.x - 1)*block_width), block_width);                                                                        \
                            cuda_kernel_small_channel_cvtformat_type_opt_c4_fp32_8_unalign_NDARRAY_NHWC<float4,float,2,4><<<gridSize,blockSize,0,stream>>>(                         \
                                (const void*)input,output, block_width, remain_width, DivModFast(block_width), DivModFast(remain_width), DivModFast(param.dst_pad), param);         \
                        }                                                                                                                                                           \
                        return;                                                                                                                                                     \
                    }                                                                                                                                                               \
                default:                                                                                                                                                            \
                    cuda_kernel_small_channel_cvtformat_type<FLOAT32_INT8, mode>                                                                                                    \
                        <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,                                                                                        \
                        src_pad_fast, dst_pad_fast, output, param);                                                                                                                 \
                    break;                                                                                                                                                          \
                }                                                                                                                                                                   \
                break;                                                                          \
            case INT8_FLOAT32:                                                                  \
                cuda_kernel_small_channel_cvtformat_type<INT8_FLOAT32, mode>                    \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case UINT8_FLOAT32:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<UINT8_FLOAT32, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case UINT8_FLOAT16:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<UINT8_FLOAT16, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT32_FLOAT16:                                                               \
                switch(mode){                                                                   \
                    case NDARRAY_NHWC:                                                         \
                        if((param.dst_pad & 7) == 0)                                            \
                        {                                                                       \
                            float divPadChannel = 1.0 / param.dst_pad;                          \
                            if((param.n_inner & 3) == 0)                                        \
                            {                                                                   \
                                cuda_kernel_small_channel_cvtformat_type_opt_fp32_16_NDARRAY_NHWC<float4,float2,2,4>                                                                    \
                                <<<dim3((param.n_inner + 511)>>9,param.n_outer,1),dim3(64,param.dst_pad,1),0,stream>>>(input,param.n_inner,output,divPadChannel,param);                 \
                                return;                                                                                                                                                 \
                            }else {                                                                                                                                                     \
                                cuda_kernel_small_channel_cvtformat_type_opt_fp32_16_NDARRAY_NHWC<float,half,0,1>                                                                       \
                                <<<dim3((param.n_inner + 511)>>9,param.n_outer,1),dim3(64,param.dst_pad,1),0,stream>>>(input,param.n_inner,output,divPadChannel,param);                 \
                                return;                                                                                                                                                 \
                            }                                                                                                                                                           \
                        } else if((param.dst_pad & 3) == 0) {                                                                                                                           \
                            if((param.n_inner&3) == 0) {                                                                                                                                \
                                int block_width = 2048 / param.dst_pad;                                                                                                                 \
                                int remain_width;                                                                                                                                       \
                                block_width = block_width >> 2 << 2;                                                                                                                    \
                                dim3 blockSize(512,1,1);                                                                                                                                \
                                dim3 gridSize((param.n_inner + block_width - 1) / block_width,param.n_outer,1);                                                                         \
                                remain_width = MIN((param.n_inner - (gridSize.x - 1)*block_width), block_width);                                                                        \
                                cuda_kernel_small_channel_cvtformat_type_opt_c4_fp32_16_NDARRAY_NHWC<float4,float2,2,4><<<gridSize,blockSize,0,stream>>>(                               \
                                    (const void*)input,output, block_width, remain_width, DivModFast(block_width), DivModFast(remain_width), DivModFast(param.dst_pad), param);         \
                            } else if((param.n_inner&1) == 0) {                                                                                                                         \
                                int block_width = 2048 / param.dst_pad;                                                                                                                 \
                                int remain_width;                                                                                                                                       \
                                block_width = block_width >> 1 << 1;                                                                                                                    \
                                dim3 blockSize(512,1,1);                                                                                                                                \
                                dim3 gridSize((param.n_inner + block_width - 1) / block_width,param.n_outer,1);                                                                         \
                                remain_width = MIN((param.n_inner - (gridSize.x - 1)*block_width), block_width);                                                                        \
                                cuda_kernel_small_channel_cvtformat_type_opt_c4_fp32_16_NDARRAY_NHWC<float2,float,1,2><<<gridSize,blockSize,0,stream>>>(                                \
                                    (const void*)input,output, block_width, remain_width, DivModFast(block_width), DivModFast(remain_width), DivModFast(param.dst_pad), param);         \
                            } else {                                                                                                                                                    \
                                int block_width = 2048 / param.dst_pad;                                                                                                                 \
                                block_width = block_width >> 2 << 2;                                                                                                                    \
                                int remain_width;                                                                                                                                       \
                                dim3 blockSize(512,1,1);                                                                                                                                \
                                dim3 gridSize((param.n_inner + block_width - 1) / block_width,param.n_outer,1);                                                                         \
                                remain_width = MIN((param.n_inner - (gridSize.x - 1)*block_width), block_width);                                                                        \
                                cuda_kernel_small_channel_cvtformat_type_opt_c4_fp32_16_unalign_NDARRAY_NHWC<float4,float2,2,4><<<gridSize,blockSize,0,stream>>>(                       \
                                    (const void*)input,output, block_width, remain_width, DivModFast(block_width), DivModFast(remain_width), DivModFast(param.dst_pad), param);         \
                            }                                                                                                                                                           \
                            return;                                                                                                                                                     \
                        }                                                                                                                                                               \
                    default:                                                                                                                                                            \
                            cuda_kernel_small_channel_cvtformat_type<FLOAT32_FLOAT16, mode>        \
                            <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,   \
                            src_pad_fast, dst_pad_fast, output, param);                            \
                    break;                                                                     \
                }                                                                               \
                break;                                                                          \
            case FLOAT16_FLOAT32:                                                               \
                cuda_kernel_small_channel_cvtformat_type<FLOAT16_FLOAT32, mode>                 \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT32_INT4B:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<FLOAT32_INT4B, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT4B_FLOAT32:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<INT4B_FLOAT32, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT8_FLOAT16:                                                                  \
                cuda_kernel_small_channel_cvtformat_type<INT8_FLOAT16, mode>                    \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT16_INT8:                                                                  \
                cuda_kernel_small_channel_cvtformat_type<FLOAT16_INT8, mode>                    \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT8_INT4B:                                                                    \
                cuda_kernel_small_channel_cvtformat_type<INT8_INT4B, mode>                      \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT8_INT8:                                                                     \
                cuda_kernel_small_channel_cvtformat_type<INT8_INT8, mode>                       \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT4B_INT4B:                                                                   \
                cuda_kernel_small_channel_cvtformat_type<INT4B_INT4B, mode>                     \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT32_INT64:                                                                   \
                cuda_kernel_small_channel_cvtformat_type<INT32_INT64, mode>                     \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT64_INT32:                                                                   \
                cuda_kernel_small_channel_cvtformat_type<INT64_INT32, mode>                     \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT64_FLOAT32:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<INT64_FLOAT32, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT32_INT64:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<FLOAT32_INT64, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT16_INT64:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<FLOAT16_INT64, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            default:                                                                            \
                LOG(ERROR) << "Unsupport  PPLCUDASmallChannelCVTFormatType OPT_CVT for data type converter: " << (int)GetCVTTypeMode(param);           \
                break;                                                                          \
        }                                                                                       \
        return;                                                                                 \
    } while (0)
#else//!OPT_CVT_FORMAT
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 block_size(256, 1, 1);                                                   \
        int num_elems = param.out_elems;                                              \
        dim3 grid_size(DivUp(num_elems, 256), 1, 1);                                  \
        DivModFast inner_fast(param.n_inner);                                         \
        DivModFast src_pad_fast(param.src_pad);                                       \
        DivModFast dst_pad_fast(param.dst_pad);                                       \
        switch (GetCVTTypeMode(param)) {                                              \
            case FLOAT32_INT8:                                                                  \
                cuda_kernel_small_channel_cvtformat_type<FLOAT32_INT8, mode>                    \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT8_FLOAT32:                                                                  \
                cuda_kernel_small_channel_cvtformat_type<INT8_FLOAT32, mode>                    \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case UINT8_FLOAT32:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<UINT8_FLOAT32, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case UINT8_FLOAT16:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<UINT8_FLOAT16, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT32_FLOAT16:                                                               \
                cuda_kernel_small_channel_cvtformat_type<FLOAT32_FLOAT16, mode>                 \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT16_FLOAT32:                                                               \
                cuda_kernel_small_channel_cvtformat_type<FLOAT16_FLOAT32, mode>                 \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT32_INT4B:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<FLOAT32_INT4B, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT4B_FLOAT32:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<INT4B_FLOAT32, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT8_FLOAT16:                                                                  \
                cuda_kernel_small_channel_cvtformat_type<INT8_FLOAT16, mode>                    \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT16_INT8:                                                                  \
                cuda_kernel_small_channel_cvtformat_type<FLOAT16_INT8, mode>                    \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT8_INT4B:                                                                    \
                cuda_kernel_small_channel_cvtformat_type<INT8_INT4B, mode>                      \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT8_INT8:                                                                     \
                cuda_kernel_small_channel_cvtformat_type<INT8_INT8, mode>                       \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT4B_INT4B:                                                                   \
                cuda_kernel_small_channel_cvtformat_type<INT4B_INT4B, mode>                     \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT32_INT64:                                                                   \
                cuda_kernel_small_channel_cvtformat_type<INT32_INT64, mode>                     \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT64_INT32:                                                                   \
                cuda_kernel_small_channel_cvtformat_type<INT64_INT32, mode>                     \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT64_FLOAT32:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<INT64_FLOAT32, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT32_INT64:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<FLOAT32_INT64, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT16_INT64:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<FLOAT16_INT64, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            default:                                                                            \
                LOG(ERROR) << "Unsupport  PPLCUDASmallChannelCVTFormatType !OPT_CVT for data type converter: " << (int)GetCVTTypeMode(param);           \
                break;                                                                          \
        }                                                                                       \
        return;                                                                                 \
    } while (0)
#endif//OPT_CVT_FORMAT
    switch (GetCVTFormatMode(param)) {
        RFNHWC
        RFN4CX
        RFNC1HWC0
        default:
            LOG(ERROR) << "Unsupport  PPLCUDASmallChannelCVTFormatType for data format converter: " << (int)GetCVTFormatMode(param);
            return;
    }
#undef RUN
}

void PPLCUDANormalCVTFormatType(cudaStream_t stream, const void *input, void *output, ReFormatParam param)
{
#ifdef OPT_CVT_FORMAT
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 block_size(DIM, 1, 1);                                                    \
        dim3 grid_size(DIM, 1, 1);                                                     \
        GenDimParam<mode>(param, block_size, grid_size);                              \
        switch (GetCVTTypeMode(param)) {                                              \
            case FLOAT32_INT8:                                                                  \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT8" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT8" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT8" ;           \
                        return;\
                    default:\
                        break;\
                }\
                call_cvtformat_type<float, int8_t, FLOAT32_INT8, mode>(input, output, block_size, grid_size, param, stream); \
                break;                                                                          \
            case INT8_FLOAT32:                                                                  \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT8_FLOAT32, mode>                                  \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case UINT8_FLOAT32:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<UINT8_FLOAT32, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case UINT8_FLOAT16:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT16" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<UINT8_FLOAT16, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT32_FLOAT16:                                                               \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_FLOAT16" ;           \
                        return;\
                    default:\
                        break;\
                }\
                call_cvtformat_type<float, half, FLOAT32_FLOAT16, mode>(input, output, block_size, grid_size, param, stream);\
                break;                                                                          \
            case FLOAT16_FLOAT32:                                                               \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_FLOAT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                call_cvtformat_type<half, float, FLOAT16_FLOAT32, mode>(input, output, block_size, grid_size, param, stream);\
                break;                                                                          \
            case FLOAT32_INT4B:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT4B" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT4B" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT4B" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<FLOAT32_INT4B, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT4B_FLOAT32:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_FLOAT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT4B_FLOAT32, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT8_FLOAT16:                                                                  \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT16" ;           \
                        return;\
                    default:\
                        break;\
                }\
                call_cvtformat_type<int8_t, half, INT8_FLOAT16, mode>(input, output, block_size, grid_size, param, stream); \
                break;                                                                          \
            case FLOAT16_INT8:                                                                  \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_INT8" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_INT8" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_INT8" ;           \
                        return;\
                    default:\
                        break;\
                }\
                call_cvtformat_type<half, int8_t, FLOAT16_INT8, mode>(input, output, block_size, grid_size, param, stream); \
                break;                                                                          \
            case INT8_INT4B:                                                                    \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT4B" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT4B" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT4B" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT8_INT4B, mode>                                    \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT8_INT8:                                                                     \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT8" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT8" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT8" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT8_INT8, mode>                                     \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT4B_INT4B:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_INT4B" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_INT4B" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_INT4B" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT4B_INT4B, mode>                                   \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT32_INT64:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT32_INT64" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT32_INT64" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT32_INT64" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT32_INT64, mode>                                   \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT64_INT32:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_INT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_INT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_INT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT64_INT32, mode>                                   \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT64_FLOAT32:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_FLOAT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT64_FLOAT32, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT32_INT64:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT64" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT64" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT64" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<FLOAT32_INT64, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            default:                                                                            \
                LOG(ERROR) << "Unsupport  PPLCUDANormalCVTFormatType OPT_CVT for data type converter: " << (int)GetCVTTypeMode(param);           \
                break;                                                                          \
        }                                                                                       \
        return;                                                                                 \
    } while (0)
#else//!OPT_CVT_FORMAT
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 block_size(DIM, 1, 1);                                                    \
        dim3 grid_size(DIM, 1, 1);                                                     \
        GenDimParam<mode>(param, block_size, grid_size);                              \
        switch (GetCVTTypeMode(param)) {                                              \
            case FLOAT32_INT8:                                                                  \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT8" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT8" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT8" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<FLOAT32_INT8, mode>                                  \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT8_FLOAT32:                                                                  \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT8_FLOAT32, mode>                                  \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case UINT8_FLOAT32:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<UINT8_FLOAT32, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case UINT8_FLOAT16:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type UINT8_FLOAT16" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<UINT8_FLOAT16, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT32_FLOAT16:                                                               \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_FLOAT16" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<FLOAT32_FLOAT16, mode>                               \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT16_FLOAT32:                                                               \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_FLOAT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<FLOAT16_FLOAT32, mode>                               \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT32_INT4B:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT4B" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT4B" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT4B" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<FLOAT32_INT4B, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT4B_FLOAT32:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_FLOAT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT4B_FLOAT32, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT8_FLOAT16:                                                                  \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT16" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_FLOAT16" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT8_FLOAT16, mode>                                  \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT16_INT8:                                                                  \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_INT8" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_INT8" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT16_INT8" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<FLOAT16_INT8, mode>                                  \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT8_INT4B:                                                                    \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT4B" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT4B" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT4B" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT8_INT4B, mode>                                    \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT8_INT8:                                                                     \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT8" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT8" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT8_INT8" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT8_INT8, mode>                                     \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT4B_INT4B:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_INT4B" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_INT4B" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT4B_INT4B" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT4B_INT4B, mode>                                   \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT32_INT64:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT32_INT64" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT32_INT64" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT32_INT64" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT32_INT64, mode>                                   \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT64_INT32:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_INT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_INT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_INT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT64_INT32, mode>                                   \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT64_FLOAT32:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_FLOAT32" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type INT64_FLOAT32" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<INT64_FLOAT32, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT32_INT64:                                                                 \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT64" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT64" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormatType for data type FLOAT32_INT64" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat_type<FLOAT32_INT64, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            default:                                                                            \
                LOG(ERROR) << "Unsupport  for !OPT_CVT PPLCUDANormalCVTFormatType for data size" << (int)GetSizeOfDataType(param.out_type);           \
                break;                                                                          \
        }                                                                                       \
        return;                                                                                 \
    } while (0)
#endif//OPT_CVT_FORMAT
    switch (GetCVTFormatMode(param)) {
        RFC8C16
        RFNHWC
        RFN4CX
        RFNC1HWC0
        default:
            LOG(ERROR) << "Unsupport  for PPLCUDANormalCVTFormatType data format converter: " << (int)GetCVTFormatMode(param);           \
            return;
    }
#undef RUN
}

void PPLCUDACVTFormatType(
    cudaStream_t stream,
    const void* input,
    void* output,
    ReFormatParam param)
{
    if (param.channel < LEASTCHANNEL && !(GetCVTFormatMode(param) == NHWC8_NHWC16 || GetCVTFormatMode(param) == NHWC16_NHWC8 
                                          || GetCVTFormatMode(param) == NC1HWC0_NHWC || GetCVTFormatMode(param) == NHWC_NC1HWC0
                                          || GetCVTFormatMode(param) == NC1HWC0_NC1HWC0)) {
        PPLCUDASmallChannelCVTFormatType(stream, input, output, param);
    } else {
        PPLCUDANormalCVTFormatType(stream, input, output, param);
    }
}

template <CVTTypeMode type_mode>
__global__ void cuda_kernel_cvtformat_type_nc(
    const void* input,
    void* output,
    ReFormatParam param,
    bool ndarray_nhwc)
{
    int64_t idx_chl = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t idx_outer = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_chl >= param.dst_pad || idx_outer >= param.n_outer) return;
    int64_t out_offset = idx_outer * param.dst_pad + idx_chl;
    if (!ndarray_nhwc || idx_chl < param.src_pad) {
        int64_t in_offset = idx_outer * param.src_pad + idx_chl;
        cuda_kernel_cvt_per_elems<type_mode>(input, in_offset, output, out_offset, param);
    } else {
        cuda_kernel_set_zero_per_elems<type_mode>(output, out_offset);
    }   
}

void PPLCUDACVTFormatTypeNC(
    cudaStream_t stream,
    const void* input,
    void* output,
    ReFormatParam param)
{
    // only for ndarray_nhwc or nhwc_ndarray when param.inner == 1, which means just padded
    dim3 block_size, grid_size;
    block_size.x = DIM;
    block_size.y = DIM;
    grid_size.x  = DivUp(param.dst_pad, DIM);
    grid_size.y  = DivUp(param.n_outer, DIM);
    bool ndarray_nhwc = (GetCVTFormatMode(param) == NDARRAY_NHWC);
    if(!ndarray_nhwc){
        switch (GetCVTTypeMode(param)) {
            case FLOAT32_INT8:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for FLOAT32_INT8" ;
                break;
            case INT8_FLOAT32:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for INT8_FLOAT32" ;
                break;
            case UINT8_FLOAT32:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for UINT8_FLOAT32" ;
                break;
            case UINT8_FLOAT16:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for UINT8_FLOAT16" ;
                break;
            case FLOAT32_FLOAT16:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for FLOAT32_FLOAT16" ;
                break;
            case FLOAT16_FLOAT32:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for FLOAT16_FLOAT32" ;
                break;
            case FLOAT32_INT4B:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for FLOAT32_INT4B" ;
                break;
            case INT4B_FLOAT32:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for INT4B_FLOAT32" ;
                break;
            case INT8_FLOAT16:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for INT8_FLOAT16" ;
                break;
            case FLOAT16_INT8:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for FLOAT16_INT8" ;
                break;
            case INT8_INT4B:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for INT8_INT4B" ;
                break;
            case INT8_INT8:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for INT8_INT8" ;
                break;
            case INT4B_INT4B:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for INT4B_INT4B" ;
                break;
            case INT32_INT64:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for INT32_INT64" ;
                break;
            case INT64_INT32:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for INT64_INT32" ;
                break;
            case INT64_FLOAT32:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for INT64_FLOAT32" ;
                break;
            case FLOAT32_INT64:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for FLOAT32_INT64" ;
                break;
            default:
                LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for " << (int)GetCVTTypeMode(param);
                break;
        }
        return;
    }
    switch (GetCVTTypeMode(param)) {
        case FLOAT32_INT8:
            cuda_kernel_cvtformat_type_nc<FLOAT32_INT8>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case INT8_FLOAT32:
            cuda_kernel_cvtformat_type_nc<INT8_FLOAT32>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case UINT8_FLOAT32:
            cuda_kernel_cvtformat_type_nc<UINT8_FLOAT32>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case UINT8_FLOAT16:
            cuda_kernel_cvtformat_type_nc<UINT8_FLOAT16>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case FLOAT32_FLOAT16:
            cuda_kernel_cvtformat_type_nc<FLOAT32_FLOAT16>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case FLOAT16_FLOAT32:
            cuda_kernel_cvtformat_type_nc<FLOAT16_FLOAT32>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case FLOAT32_INT4B:
            cuda_kernel_cvtformat_type_nc<FLOAT32_INT4B>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case INT4B_FLOAT32:
            cuda_kernel_cvtformat_type_nc<INT4B_FLOAT32>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case INT8_FLOAT16:
            cuda_kernel_cvtformat_type_nc<INT8_FLOAT16>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case FLOAT16_INT8:
            cuda_kernel_cvtformat_type_nc<FLOAT16_INT8>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case INT8_INT4B:
            cuda_kernel_cvtformat_type_nc<INT8_INT4B>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case INT8_INT8:
            cuda_kernel_cvtformat_type_nc<INT8_INT8>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case INT4B_INT4B:
            cuda_kernel_cvtformat_type_nc<INT4B_INT4B>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case INT32_INT64:
            cuda_kernel_cvtformat_type_nc<INT32_INT64>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case INT64_INT32:
            cuda_kernel_cvtformat_type_nc<INT64_INT32>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case INT64_FLOAT32:
            cuda_kernel_cvtformat_type_nc<INT64_FLOAT32>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        case FLOAT32_INT64:
            cuda_kernel_cvtformat_type_nc<FLOAT32_INT64>
                <<<grid_size, block_size, 0, stream>>>(input, output, param, ndarray_nhwc);
            break;
        default:
            LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func  PPLCUDACVTFormatTypeNC for " << (int)GetCVTTypeMode(param);
            break;
    }
    return;
}