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

#include "cudakernel/nn/pooling_max.h"
#include "cudakernel/common/divmod_fast.h"
#include "ppl/common/types.h"
#include <cuda_fp16.h>
#include <float.h>
#ifdef PPLNN_USE_MACA
#define MAX_POOLING_OPT
#endif

#define HALF_MIN half(-65504)
#define HALF2_MIN half2(-65504, -65504)
#define PPL_CUDA_HALF2_MAX(a, b)                    \
    do {                                             \
        (a).x = __hgt((a).x, (b).x) ? (a).x : (b).x; \
        (a).y = __hgt((a).y, (b).y) ? (a).y : (b).y; \
    } while (0)
#define PPL_CUDA_MAX(a, b) a = a > b ? a : b

__device__ inline float numerical_min(float a){
    return -FLT_MAX;
}

__device__ inline int8_t numerical_min(int8_t a){
    return -128;
}

__device__ inline half2 numerical_min(half2 a){
    return HALF2_MIN;
}

__device__ inline half numerical_min(half a){
    return HALF_MIN;
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_f3s2_half(
    const half* input,
    half* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (c >= pad_channels)
        return;

    int inOff  = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    half iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy      = oy * 2 + i - pad_height;
            int ix      = ox * 2 + j - pad_width;
            bool pred   = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : HALF_MIN;
        }
    }

    // pooling max & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
#pragma unroll TILE_W
        for (int j = 0; j < TILE_W; j++) {
            half val = iregs[i * 2 + 0][j * 2 + 0];
            val      = __hgt(val, iregs[i * 2 + 0][j * 2 + 1]) ? val : iregs[i * 2 + 0][j * 2 + 1];
            val      = __hgt(val, iregs[i * 2 + 0][j * 2 + 2]) ? val : iregs[i * 2 + 0][j * 2 + 2];
            val      = __hgt(val, iregs[i * 2 + 1][j * 2 + 0]) ? val : iregs[i * 2 + 1][j * 2 + 0];
            val      = __hgt(val, iregs[i * 2 + 1][j * 2 + 1]) ? val : iregs[i * 2 + 1][j * 2 + 1];
            val      = __hgt(val, iregs[i * 2 + 1][j * 2 + 2]) ? val : iregs[i * 2 + 1][j * 2 + 2];
            val      = __hgt(val, iregs[i * 2 + 2][j * 2 + 0]) ? val : iregs[i * 2 + 2][j * 2 + 0];
            val      = __hgt(val, iregs[i * 2 + 2][j * 2 + 1]) ? val : iregs[i * 2 + 2][j * 2 + 1];
            val      = __hgt(val, iregs[i * 2 + 2][j * 2 + 2]) ? val : iregs[i * 2 + 2][j * 2 + 2];

            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = val;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s2(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

      if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy = oy * 2 + i - pad_height;
            int ix = ox * 2 + j - pad_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
        }
    }

      // pooling max & store output
#pragma unroll TILE_H
      for (int i = 0; i < TILE_H; i++) {
#pragma unroll TILE_W
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i * 2 + 0][j * 2 + 0];
            val = (val > iregs[i * 2 + 0][j * 2 + 1]) ? val : iregs[i * 2 + 0][j * 2 + 1];
            val = (val > iregs[i * 2 + 0][j * 2 + 2]) ? val : iregs[i * 2 + 0][j * 2 + 2];
            val = (val > iregs[i * 2 + 1][j * 2 + 0]) ? val : iregs[i * 2 + 1][j * 2 + 0];
            val = (val > iregs[i * 2 + 1][j * 2 + 1]) ? val : iregs[i * 2 + 1][j * 2 + 1];
            val = (val > iregs[i * 2 + 1][j * 2 + 2]) ? val : iregs[i * 2 + 1][j * 2 + 2];
            val = (val > iregs[i * 2 + 2][j * 2 + 0]) ? val : iregs[i * 2 + 2][j * 2 + 0];
            val = (val > iregs[i * 2 + 2][j * 2 + 1]) ? val : iregs[i * 2 + 2][j * 2 + 1];
            val = (val > iregs[i * 2 + 2][j * 2 + 2]) ? val : iregs[i * 2 + 2][j * 2 + 2];

            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = val;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s2(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy = oy * 2 + i - pad_height;
            int ix = ox * 2 + j - pad_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
        }
    }

      // pooling max & store output
#pragma unroll TILE_H
      for (int i = 0; i < TILE_H; i++) {
#pragma unroll TILE_W
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i * 2 + 0][j * 2 + 0];
            val = (val > iregs[i * 2 + 0][j * 2 + 1]) ? val : iregs[i * 2 + 0][j * 2 + 1];
            val = (val > iregs[i * 2 + 0][j * 2 + 2]) ? val : iregs[i * 2 + 0][j * 2 + 2];
            val = (val > iregs[i * 2 + 1][j * 2 + 0]) ? val : iregs[i * 2 + 1][j * 2 + 0];
            val = (val > iregs[i * 2 + 1][j * 2 + 1]) ? val : iregs[i * 2 + 1][j * 2 + 1];
            val = (val > iregs[i * 2 + 1][j * 2 + 2]) ? val : iregs[i * 2 + 1][j * 2 + 2];
            val = (val > iregs[i * 2 + 2][j * 2 + 0]) ? val : iregs[i * 2 + 2][j * 2 + 0];
            val = (val > iregs[i * 2 + 2][j * 2 + 1]) ? val : iregs[i * 2 + 2][j * 2 + 1];
            val = (val > iregs[i * 2 + 2][j * 2 + 2]) ? val : iregs[i * 2 + 2][j * 2 + 2];

            if (oy + i < out_height && ox + j < out_width) {
                int res = round((float(val) * in_scale) * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[outOff + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_f3s1_half(
    const half* input,
    half* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (c >= pad_channels)
        return;

    int inOff  = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    half iregs[TILE_H + 2][TILE_W + 2];

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy      = oy + i - pad_height;
            int ix      = ox + j - pad_width;
            bool pred   = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : HALF_MIN;
        }
    }

    // pooling max & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half val = iregs[i + 0][j + 0];
            val      = __hgt(val, iregs[i + 0][j + 1]) ? val : iregs[i + 0][j + 1];
            val      = __hgt(val, iregs[i + 0][j + 2]) ? val : iregs[i + 0][j + 2];
            val      = __hgt(val, iregs[i + 1][j + 0]) ? val : iregs[i + 1][j + 0];
            val      = __hgt(val, iregs[i + 1][j + 1]) ? val : iregs[i + 1][j + 1];
            val      = __hgt(val, iregs[i + 1][j + 2]) ? val : iregs[i + 1][j + 2];
            val      = __hgt(val, iregs[i + 2][j + 0]) ? val : iregs[i + 2][j + 0];
            val      = __hgt(val, iregs[i + 2][j + 1]) ? val : iregs[i + 2][j + 1];
            val      = __hgt(val, iregs[i + 2][j + 2]) ? val : iregs[i + 2][j + 2];

            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = val;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s1(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    T iregs[TILE_H + 2][TILE_W + 2];

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy = oy + i - pad_height;
            int ix = ox + j - pad_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
        }
    }

    // pooling max & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i + 0][j + 0];
            val = (val > iregs[i + 0][j + 1]) ? val : iregs[i + 0][j + 1];
            val = (val > iregs[i + 0][j + 2]) ? val : iregs[i + 0][j + 2];
            val = (val > iregs[i + 1][j + 0]) ? val : iregs[i + 1][j + 0];
            val = (val > iregs[i + 1][j + 1]) ? val : iregs[i + 1][j + 1];
            val = (val > iregs[i + 1][j + 2]) ? val : iregs[i + 1][j + 2];
            val = (val > iregs[i + 2][j + 0]) ? val : iregs[i + 2][j + 0];
            val = (val > iregs[i + 2][j + 1]) ? val : iregs[i + 2][j + 1];
            val = (val > iregs[i + 2][j + 2]) ? val : iregs[i + 2][j + 2];

            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = val;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s1(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    T iregs[TILE_H + 2][TILE_W + 2];

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy = oy + i - pad_height;
            int ix = ox + j - pad_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
        }
    }

    // pooling max & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i + 0][j + 0];
            val = (val > iregs[i + 0][j + 1]) ? val : iregs[i + 0][j + 1];
            val = (val > iregs[i + 0][j + 2]) ? val : iregs[i + 0][j + 2];
            val = (val > iregs[i + 1][j + 0]) ? val : iregs[i + 1][j + 0];
            val = (val > iregs[i + 1][j + 1]) ? val : iregs[i + 1][j + 1];
            val = (val > iregs[i + 1][j + 2]) ? val : iregs[i + 1][j + 2];
            val = (val > iregs[i + 2][j + 0]) ? val : iregs[i + 2][j + 0];
            val = (val > iregs[i + 2][j + 1]) ? val : iregs[i + 2][j + 1];
            val = (val > iregs[i + 2][j + 2]) ? val : iregs[i + 2][j + 2];

            if (oy + i < out_height && ox + j < out_width) {
                int res = round((float(val) * in_scale) * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[outOff + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
}

// #################### pooling max #######################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_common_half(
    const half* input,
    half* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (c >= pad_channels)
        return;

    int inOff  = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half res = HALF_MIN;

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                    int iy    = (oy + i) * stride_height + fy - pad_height;
                    int ix    = (ox + j) * stride_width + fx - pad_width;
                    bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    half ival = pred ? input[inOff + iy * in_width + ix] : HALF_MIN;

                    res = __hgt(res, ival) ? res : ival;
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_common_half(
    const half* input,
    half* output,
    int64_t* indices,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (c >= pad_channels)
        return;

    int inOff  = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half res = HALF_MIN;
            int64_t in_index = 0;

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                    int iy    = (oy + i) * stride_height + fy - pad_height;
                    int ix    = (ox + j) * stride_width + fx - pad_width;
                    bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    half ival = pred ? input[inOff + iy * in_width + ix] : HALF_MIN;

                    if (__hlt(res, ival)) {
                        res = ival;
                        in_index = inOff + iy * in_width + ix;
                    }
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                int64_t out_index = outOff + (oy + i) * out_width + ox + j;
                output[out_index] = res;
                indices[out_index] = in_index;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {

            T res = numerical_min(T(0));

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                int iy = (oy + i) * stride_height + fy - pad_height;
                int ix = (ox + j) * stride_width + fx - pad_width;
                bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                T ival = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));

                res = (res > ival) ? res : ival;
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
}

template <int TILE_D,int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max3d_common(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int pad_depth,
    int pad_height,
    int pad_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (c >= pad_channels) return;
    int in_imagesize = in_height * in_width;
    int out_imagesize = out_height * out_width;
    int inOff = (b * pad_channels + c) * in_depth * in_imagesize;
    int outOff = (b * pad_channels + c) * out_depth * out_imagesize;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int partH = (out_height + TILE_H - 1) / TILE_H;
    int partWH = partW * partH;

    int ox = (tx % partW) * TILE_W;
    int oy = ((tx / partW) % partH) * TILE_H;
    int od = (tx / partWH) * TILE_D;

    // register blocking for input
    for(int d = 0; d < TILE_D; d++){
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {

                T res = numerical_min(T(0));

                // read input
                for(int fd = 0; fd < kernel_depth; fd++){
                    int id = (od + d) * stride_depth + fd - pad_depth;
                    for (int fy = 0; fy < kernel_height; fy++) {
                        for (int fx = 0; fx < kernel_width; fx++) {
                            int iy = (oy + i) * stride_height + fy - pad_height;
                            int ix = (ox + j) * stride_width + fx - pad_width;
                            bool pred = (id >=0 && id < in_depth) && (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                            T ival = pred ? input[inOff + id * in_imagesize + iy * in_width + ix] : numerical_min(T(0));

                            res = (res > ival) ? res : ival;
                        }
                    }
                }

                // store output
                if (od +  d < out_depth && oy + i < out_height && ox + j < out_width) {
                    output[outOff + (od + d) * out_imagesize + (oy + i) * out_width + ox + j] = res;
                }
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {

            int res = numerical_min(T(0));

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                int iy = (oy + i) * stride_height + fy - pad_height;
                int ix = (ox + j) * stride_width + fx - pad_width;
                bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                T ival = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
                res = (res > ival) ? res : ival;
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                res = round(res * in_scale * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[outOff + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
}

template <int TILE_D, int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int pad_depth,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width * in_depth;
    int outOff = (b * pad_channels + c) * out_height * out_width * out_depth;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int partH = (out_height + TILE_H - 1) / TILE_H;

    int ox = (tx % partW) * TILE_W;
    int oy = ((tx / partW) % partH) * TILE_H;
    int oz = (tx / (partW * partH)) * TILE_D;

    // register blocking for input
    for (int k = 0; k < TILE_D; k++){
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {

                int res = numerical_min(T(0));

                // read input
                for(int fz = 0; fz < kernel_depth; fz++){
                    int iz = (oz + k) * stride_depth - pad_depth + fz;
                    for (int fy = 0; fy < kernel_height; fy++) {
                        for (int fx = 0; fx < kernel_width; fx++) {
                        int iy = (oy + i) * stride_height + fy - pad_height;
                        int ix = (ox + j) * stride_width + fx - pad_width;
                        bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width) && (iz >= 0 && iz < in_depth);
                        T ival = pred ? input[inOff + iz * in_height * in_width + iy * in_width + ix] : numerical_min(T(0));
                        res = (res > ival) ? res : ival;
                        }
                    }
                }

                // store output
                if (oy + i < out_height && ox + j < out_width && oz + k < out_depth) {
                    // res = round(res * in_scale * out_scale );
                    if(res > 127) res = 127;
                    else if( res < -128) res = -128;
                    output[outOff + (oz + k) * out_width * out_height + (oy + i) * out_width + ox + j] = res;
                }
            }
        }
    }
}

#ifdef MAX_POOLING_OPT
#define SHARED_SM 8192
template<int PAD>
__device__ __forceinline__ bool predict(int in_height, int in_width, int yy, int xx) {
    return xx >= 0 && xx < in_width && yy >= 0 && yy < in_height;
}

template<>
__device__ __forceinline__ bool predict<0>(int in_height, int in_width, int yy, int xx) {
    return xx < in_width && yy < in_height;
}

//TI means TinyImages
//Kernel process for ti kernel functions
template<typename T, int KERNEL, int PAD>
__device__ __forceinline__ T max_pooling_kernel(
    const T* s1,
    int in_height,
    int in_width,
    int iy,
    int ix,
    int in_off)
{
    T res = numerical_min(T(0));
    #pragma unroll
    for (int fy = 0; fy < KERNEL; fy++) {
        #pragma unroll
        for (int fx = 0; fx < KERNEL; fx++) {
            int xx = ix + fx - PAD;
            int yy = iy + fy - PAD;
            bool pred = predict<PAD>(in_height, in_width, yy, xx);
            T val = pred ? s1[in_off + xx + yy * in_width] : res;
            res = res > val ? res : val;
        }
    }
    return res;
}

template<typename T, int KERNEL, int PAD>
__device__ __forceinline__ T max_pooling_kernel_restricted(
    const T* s1,
    int in_height,
    int in_width,
    int iy,
    int ix,
    int in_off)
{
    int center_idx = in_off + iy * in_width + ix;
    T res = numerical_min(T(0));
    //Much faster than max_pooling_kernel as lds and max operates alternatively
    //But we need to know that out_width*STRIDE <= in_width and out_height*STRIDE <= in_height
    #pragma unroll
    for (int fy = 0; fy < KERNEL; fy++) {
        #pragma unroll
        for (int fx = 0; fx < KERNEL; fx++) {
            int xx = ix + fx - PAD;
            int yy = iy + fy - PAD;
            bool pred = predict<PAD>(in_height, in_width, yy, xx);
            int idx = pred ? in_off + xx + yy * in_width : center_idx;
            T val = s1[idx];
            res = res > val ? res : val;
        }
    }
    return res;
}

template<typename T, int KERNEL, int PAD>
__device__ __forceinline__ T max_pooling_kernel_no_edge(
    const T* s1,
    int in_height,
    int in_width,
    int iy,
    int ix,
    int in_off)
{
    T res = numerical_min(T(0));
    #pragma unroll
    for (int fy = 0; fy < KERNEL; fy++) {
        #pragma unroll
        for (int fx = 0; fx < KERNEL; fx++) {
            int xx = ix + fx - PAD;
            int yy = iy + fy - PAD;
            int idx = in_off + xx + yy * in_width;
            T val = s1[idx];
            res = res > val ? res : val;
        }
    }
    return res;
}

//TL means TinyLines
//Kernel processing for tl kernel functions
template<typename T, int KERNEL, int PAD>
__device__ __forceinline__ T max_pooling_kernel(
    const T* s1,
    int in_height,
    int in_width,
    int iyy,
    int iy,
    int ix,
    int in_off)
{
    T res = numerical_min(T(0));
    #pragma unroll
    for (int fy = 0; fy < KERNEL; fy++) {
        #pragma unroll
        for (int fx = 0; fx < KERNEL; fx++) {
            int xx = ix + fx - PAD;
            int yy = iyy + fy - PAD;
            bool pred = predict<PAD>(in_height, in_width, yy, xx);
            T val = pred ? s1[in_off + fx - PAD + (fy - PAD) * in_width] : res;
            res = res > val ? res : val;
        }
    }
    return res;
}

template<typename T, int KERNEL, int PAD, int HAVE_EDGE>
__device__ __forceinline__ T max_pooling_kernel_restricted(
    const T* s1,
    int in_height,
    int in_width,
    int iyy,
    int iy,
    int ix,
    int in_off)
{
    T res = numerical_min(T(0));
    #pragma unroll
    for (int fy = 0; fy < KERNEL; fy++) {
        #pragma unroll
        for (int fx = 0; fx < KERNEL; fx++) {
            int xx = ix + fx - PAD;
            int yy = iyy + fy - PAD;
            bool pred = predict<PAD>(in_height, in_width, yy, xx);
            int idx = pred ? in_off + fx - PAD + (fy - PAD) * in_width : in_off;
            T val = s1[idx];
            res = res > val ? res : val;
        }
    }
    return res;
}

//When no edge, we do not need to worry about edge, all pixels should be in image
template<>
__device__ __forceinline__ int8_t max_pooling_kernel_restricted<int8_t, 2, 0, 0>(
    const int8_t* s1,
    int in_height,
    int in_width,
    int iyy,
    int iy,
    int ix,
    int in_off)
{
    using T=int8_t;
    T res = s1[in_off];
    T val01 = s1[in_off+1];
    res = res > val01 ? res : val01;
    T val10 = s1[in_off+in_width];
    res = res > val10 ? res : val10;
    T val11 = s1[in_off+in_width+1];
    res = res > val11 ? res : val11;
    return res;
}

template<>
__device__ __forceinline__ int8_t max_pooling_kernel_restricted<int8_t, 2, 0, 1>(
    const int8_t* s1,
    int in_height,
    int in_width,
    int iyy,
    int iy,
    int ix,
    int in_off)
{
    using T=int8_t;
    T res = s1[in_off];
    int idx = ix + 1 < in_width ? in_off+1 : in_off;
    T val01 = s1[idx];
    res = res > val01 ? res : val01;
    idx = iyy < in_height ? in_off+in_width : in_off;
    T val10 = s1[idx];
    res = res > val10 ? res : val10;
    idx = iyy < in_height && ix + 1 < in_width ? in_off+in_width+1 : in_off;
    T val11 = s1[idx];
    res = res > val11 ? res : val11;
    return res;
}

template<>
__device__ __forceinline__ int8_t max_pooling_kernel_restricted<int8_t, 3, 0, 0>(
    const int8_t* s1,
    int in_height,
    int in_width,
    int iyy,
    int iy,
    int ix,
    int in_off)
{
    using T=int8_t;
    T res = s1[in_off];
    T val = s1[in_off+1];
    res = res > val ? res : val;
    val = s1[in_off+2];
    res = res > val ? res : val;
    val = s1[in_off+in_width];
    res = res > val ? res : val;
    val = s1[in_off+in_width+1];
    res = res > val ? res : val;
    val = s1[in_off+in_width+2];
    res = res > val ? res : val;
    val = s1[in_off+2*in_width];
    res = res > val ? res : val;
    val = s1[in_off+2*in_width+1];
    res = res > val ? res : val;
    val = s1[in_off+2*in_width+2];
    res = res > val ? res : val;
    return res;
}

template<>
__device__ __forceinline__ int8_t max_pooling_kernel_restricted<int8_t, 3, 0, 1>(
    const int8_t* s1,
    int in_height,
    int in_width,
    int iyy,
    int iy,
    int ix,
    int in_off)
{
    using T=int8_t;
    T res = s1[in_off];
    int idx = ix + 1 < in_width ? in_off+1 : in_off;
    T val = s1[idx];
    res = res > val ? res : val;
    idx = ix + 2 < in_width ? in_off+2 : in_off;
    val = s1[idx];
    res = res > val ? res : val;
    idx = iyy + 1 < in_height ? in_off+in_width : in_off;
    val = s1[idx];
    res = res > val ? res : val;
    idx = iyy + 1 < in_height && ix + 1 < in_width ? in_off+in_width+1 : in_off;
    val = s1[idx];
    res = res > val ? res : val;
    idx = iyy + 1 < in_height && ix + 2 < in_width ? in_off+in_width+2 : in_off;
    val = s1[idx];
    res = res > val ? res : val;
    idx = iyy + 2 < in_height ? in_off+in_width*2 : in_off;
    val = s1[idx];
    res = res > val ? res : val;
    idx = iyy + 2 < in_height && ix + 1 < in_width ? in_off+in_width*2+1 : in_off;
    val = s1[idx];
    res = res > val ? res : val;
    idx = iyy + 2 < in_height && ix + 2 < in_width ? in_off+in_width*2+2 : in_off;
    val = s1[idx];
    res = res > val ? res : val;
    return res;
}

template<int S>
__device__ __forceinline__ int8_t qaunt_scale(int8_t res, float s) {
    int res2 = round(res * s);
    if(res2 > 127) res2 = 127;
    else if( res2 < -128) res2 = -128;
    return res2;
}

template<>
__device__ __forceinline__ int8_t qaunt_scale<1>(int8_t res, float s) {
    return res;
}

template<int HAVE_EDGE>
__device__ __forceinline__ int calculate_load_params(
    int K, int S, int P,
    int current_image,
    int left_lines,
    int out_line_start,
    int lines_per_block,
    int in_height,
    int in_width,
    DivModFast *out_height_mod,
    int &load_start,
    int &load_count)
{
    //if there is padding, we need to fetch P lines before current lines
    int start_line_in_image = left_lines*S-P < 0 ? 0 : left_lines*S-P;
    int current_line_end = out_line_start + lines_per_block;
    int end_image = 0, end_line_out_image = 0;
    out_height_mod->divmod(current_line_end, end_image, end_line_out_image);
    //if we have edge, we have to decide how many more lines loaded
    int end_line_in_image = end_line_out_image * S + K - P - 1 >= in_height ? in_height : end_line_out_image * S + K - P - 1;
    int valid_start_address = (current_image * in_height + start_line_in_image)*in_width;
    //align start address to 4
    load_start = valid_start_address / 4 * 4;
    //align end address to 4
    int load_end = ((end_image * in_height + end_line_in_image)*in_width + 3) / 4*4;
    load_count = load_end - load_start;
    int current_in_offset =  (current_image * in_height + left_lines*S)*in_width;
    int s1_ref_offset = current_in_offset - load_start;
    return s1_ref_offset;
}

//When there is no edge, calculation is very simple
template<>
__device__ __forceinline__ int calculate_load_params<0>(
    int K, int S, int P,
    int current_image,
    int left_lines,
    int out_line_start,
    int lines_per_block,
    int in_height,
    int in_width,
    DivModFast *out_height_mod,
    int &load_start,
    int &load_count)
{
    load_start = out_line_start * S * in_width;
    load_count = lines_per_block * S * in_width;
    return 0;
}



template<int K, int S, int P, int LINES_PER_WARP, int HAVE_EDGE, int CONSTANT_SCALE=1, int RESTRICTED = 1>
__global__ void ppl_cukernel_pooling_max_common_tl_quad_int8_opt(
    const int8_t* input,
    int8_t* output,
    int num_imgs,
    int lines_per_block,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    DivModFast out_height_mod,
    float s)
{
    using T=int8_t;
    __shared__ T s1[SHARED_SM]; //IN_LINE_WIDTH * STRIDE_W * LINES_PER_BLOCK
    __shared__ T s2[SHARED_SM]; //OUT_LINE_WIDTH * LINES_PER_BLOCK
    int current_out_line_start = blockIdx.x * lines_per_block;
    int current_image = 0, left_lines = 0;
    out_height_mod.divmod(current_out_line_start, current_image, left_lines);
    if (current_image >= num_imgs) return;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x*blockDim.y*blockDim.z;
    int load_start = 0, load_count = 0;
    //if HAVE_EDGE = 1, we will have to load more lines
    //so all data kernel needs is loaded
    int s1_ref_offset = calculate_load_params<HAVE_EDGE>(
        K,S,P,current_image, left_lines, current_out_line_start, lines_per_block,
        in_height, in_width, &out_height_mod, load_start, load_count);
    const T* ip = input + load_start;
    T* op = output + current_out_line_start * out_width;
    uint32_t *sa = (uint32_t*)s1;
    for (int i = tid; i < load_count/4; i += block_size) {
        sa[i] = ((uint32_t*)ip)[i];
    }
    __syncthreads();
    const T *ss1 = s1 + s1_ref_offset;
    //Every little warp will process LINES_PER_WARP lines
    //so there is no need to calculate the position of current pixels
    //and the calculations in a loop is very expensive
    int warp_idx = threadIdx.y;
    int warp_tid = threadIdx.x;
    int oy = warp_idx;
    int inner_images,oyy;
    out_height_mod.divmod(oy+left_lines, inner_images, oyy);
    int iyy = oyy * S;
    int out_off = oy * out_width;
    int iy = inner_images * in_height + iyy - left_lines * S;
    for (int l = 0; l < LINES_PER_WARP; l++) {
        //process one line per loop
        for (int i = warp_tid; i < out_width; i+=blockDim.x) {
            int ox = i;
            int ix = ox * S;
            int idx = iy * in_width + ix;
            int8_t res = 0;
            //RESTRICTED is constant and the if-else clause will be optimized by compiler
            //if RESTRICTED is true, we can use a fixed index to replace kernel data that
            //out of image, which will accerlate shared memory loading in C500
            if (RESTRICTED)
                res = max_pooling_kernel_restricted<int8_t, K, P, HAVE_EDGE>(ss1, in_height, in_width, iyy, iy, ix, idx);
            else
                res = max_pooling_kernel<int8_t, K, P>(ss1, in_height, in_width, iyy, iy, ix, idx);
            s2[out_off+i] = qaunt_scale<CONSTANT_SCALE>(res, s);
        }
        //tiny warp process lines neighbor to other tiny warps
        oy+=blockDim.y;
        out_height_mod.divmod(oy+left_lines, inner_images, oyy);
        iyy = oyy*S;
        out_off = oy * out_width;
        iy = inner_images * in_height + iyy - left_lines * S;
    }
    __syncthreads();
    for (int i = tid; i < out_width*lines_per_block/4; i += block_size) ((uint32_t*)op)[i] = ((uint32_t*)s2)[i];
}

template<int KERNEL, int STRIDE, int PAD, int IMGS_PER_BLOCK, int HAVE_EDGE, int CONSTANT_SCALE = 1, int RESTRICTED = 1>
__device__ __forceinline__ void max_kernel_per_warp_int8(
    const int8_t* s1,
    int8_t* s2,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    DivModFast *out_width_mod,
    float s,
    int ssz,
    int osz,
    int tid)
{
    int widx = tid >> 6;
    int imgs_per_warp = IMGS_PER_BLOCK >> 2;
    int wid = tid - widx*64;
    int img_start = widx * imgs_per_warp;
    //Every warp will process imgs_per_warp/4 images
    #pragma unroll IMGS_PER_BLOCK/4
    for (int im = img_start; im < img_start + imgs_per_warp; im++) {
        int in_off = im * ssz;  //s1 offset
        int out_off = im * osz; //s2 offset
        for (int i = wid; i < osz; i += 64) {
            int oy = 0;
            int ox = 0;
            out_width_mod->divmod(i, oy, ox);
            int ix = ox * STRIDE;
            int iy = oy * STRIDE;
            int8_t res2 = numerical_min(int8_t(0));
            //HAVE_EDGE and RESTRICTED are constants and if-else will be optimized by compiler
            if (HAVE_EDGE)
                if (RESTRICTED)
                    res2 = max_pooling_kernel_restricted<int8_t, KERNEL,PAD>(s1, in_height, in_width, iy, ix, in_off);
                else
                    res2 = max_pooling_kernel<int8_t, KERNEL,PAD>(s1, in_height, in_width, iy, ix, in_off);
            else
                res2 = max_pooling_kernel_no_edge<int8_t, KERNEL,PAD>(s1, in_height, in_width, iy, ix, in_off);
            res2 = qaunt_scale<CONSTANT_SCALE>(res2, s);
            s2[out_off+i] = res2;
        }
    }
}

template<int KERNEL, int STRIDE, int PAD, int IMGS_PER_BLOCK, int HAVE_EDGE=1, int CONSTANT_SCALE=1, int RESTRICTED=1>
__global__ void ppl_cukernel_pooling_max_common_ti_quad_int8_opt(
    const int8_t* input,
    int8_t* output,
    int num_imgs,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    DivModFast out_width_mod,
    float s)
{
    using T=int8_t;
    __shared__ T s1[SHARED_SM];
    __shared__ T s2[SHARED_SM];

    int current_image = blockIdx.x * IMGS_PER_BLOCK;
    if (current_image > num_imgs) return;
    int tid = threadIdx.x;
    int block_size = 256;
    const int ssz = in_width * in_height;
    const int osz = out_width * out_height;

    const int8_t *ip = input + current_image*ssz;
    int8_t *op = output + current_image*osz;
    //Load all images into s1 using a combined load
    uint32_t *sa = (uint32_t*)s1;
    for (int i = tid; i < IMGS_PER_BLOCK*ssz/4; i+=block_size) sa[i] = ((uint32_t*)ip)[i];
    __syncthreads();
    max_kernel_per_warp_int8<KERNEL,STRIDE,PAD,IMGS_PER_BLOCK,HAVE_EDGE,CONSTANT_SCALE,RESTRICTED>(
        s1, s2, in_height, in_width, out_height, out_width, &out_width_mod, s, ssz, osz, tid);
    __syncthreads();
    //Load all pixels from s2 to output using a combined store
    for (int i = tid; i < IMGS_PER_BLOCK*osz/4; i+=block_size) ((uint32_t*)op)[i] = ((uint32_t*)s2)[i];
}

//A non-templated version to hold all situations that not templated
__global__ void ppl_cukernel_pooling_max_common_ti_int8_opt(
    const int8_t* input,
    int8_t* output,
    int num_imgs,
    int imgs_per_block,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    DivModFast out_width_mod,
    float s)
{
    using T=int8_t;
    __shared__ T s1[SHARED_SM];
    __shared__ T s2[SHARED_SM];

    int current_image = blockIdx.x * imgs_per_block;
    if (current_image > num_imgs) return;
    int tid = threadIdx.x;
    int block_size = 256;
    const int ssz = in_width * in_height;
    const int osz = out_width * out_height;

    const int8_t *ip = input + current_image*ssz;
    int8_t *op = output + current_image*osz;
    //Load all images into s1 using a combined load
    uint32_t* sa = (uint32_t*)s1;
    for (int i = tid; i < imgs_per_block*ssz/4; i+=block_size) sa[i] = ((uint32_t*)ip)[i];
    __syncthreads();
    //Every warp will calculate imgs_per_block/4 images
    int widx = tid >> 6;
    int imgs_per_warp = imgs_per_block >> 2;
    int wid = tid - widx*64;
    int img_start = widx * imgs_per_warp;
    //Check if we can use a fixed index to replace kernel data that not in image
    bool restricted = (out_width-1)*stride_width <= in_width && (out_height-1)*stride_height <= in_height;
    for (int im = img_start; im < img_start + imgs_per_warp; im++) {
        int in_off = im * ssz;  //s1 offset
        int out_off = im * osz; //s2 offset
        for (int i = wid; i < osz; i += 64) {
            int oy = 0;
            int ox = 0;
            out_width_mod.divmod(i, oy, ox);
            int ix = ox * stride_width;
            int iy = oy * stride_height;
            int center_idx = in_off + iy * in_width + ix;
            T res = numerical_min(T(0));
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                    int xx = ix + fx - pad_width;
                    int yy = iy + fy - pad_height;
                    bool pred = xx >= 0 && xx < in_width && yy >= 0 && yy < in_height;
                    if (restricted) {
                        int idx = pred ? in_off + xx + yy * in_width : center_idx;
                        T val = s1[idx];
                        res = res > val ? res : val;
                    } else {
                        T val = pred ? s1[in_off + xx + yy * in_width] : res;
                        res = res > val ? res : val;
                    }
                }
            }
            if (s != 1.0f) {
                res = qaunt_scale<0>(res, s);
            }
            s2[out_off+i] = res;
        }
    }
    __syncthreads();
    //Load all pixels from s2 to output using a combined store
    for (int i = tid; i < imgs_per_block*osz/4; i+=block_size) ((uint32_t*)op)[i] = ((uint32_t*)s2)[i];
}

template <typename T, int N>
__global__ void ppl_cukernel_pooling_max_common_float4_NHWC_flatten_opt(
    const T* input,
    T* output,
    int pad_channels,
    int in_height,
    int in_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int total,
    DivModFast channels_mod,
    DivModFast outwidth_mod,
    DivModFast outheight_mod)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= total) return;
    int bhw_idx = 0, bh_idx = 0, c_idx = 0, h_idx = 0, w_idx = 0, b_idx = 0;

    channels_mod.divmod(t_idx, bhw_idx, c_idx);
    outwidth_mod.divmod(bhw_idx, bh_idx, w_idx);
    outheight_mod.divmod(bh_idx, b_idx, h_idx);

    int in_h = (h_idx * stride_height) - padding_height;
    int in_w = (w_idx * stride_width) - padding_width;
    int in_batch_size = b_idx * in_height * in_width * pad_channels + c_idx;
    int in_off = in_batch_size + in_h * in_width * pad_channels + in_w * pad_channels;

    float4 out_val;
    half* out_val_ptr = (half*)&out_val;
    #pragma unroll N
    for (int i = 0; i < N; i++) {
        out_val_ptr[i] = HALF_MIN;
    }
    int center_idx, center_h, center_w;
    center_h = min(in_h + 1, in_height - 1);
    center_h = max(center_h, 0);
    center_w = min(in_w + 1, in_width - 1);
    center_w = max(center_w, 0);
    center_idx = in_batch_size + (center_h * in_width + center_w)* pad_channels;
    for(int i = 0; i < kernel_height; i++) {
        for (int j = 0; j < kernel_width; j++) {
            bool pred = ((in_w + j) >= 0 && (in_w + j) < in_width) && ((in_h + i) >= 0 && (in_h + i) < in_height);
            int index = pred ? (in_off + i * in_width * pad_channels + j * pad_channels) : center_idx;
            float4 src_ = input[index];
            half *src = (half*)&src_;

            #pragma unroll N
            for (int i = 0; i < N; i++) {
                out_val_ptr[i] = src[i] > out_val_ptr[i] ? src[i] : out_val_ptr[i];
            }
        }
    }
    output[t_idx] = out_val;
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common_nd_int8_f2s2p0_opt(
    const float4* input,
    int64_t* output,
    int pad_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;
    if (tx >= out_height * out_width * out_depth) return;

    int inOff = (b * pad_channels + c) * in_height * in_width * in_depth;
    int outOff = (b * pad_channels + c) * out_height * out_width * out_depth;
    int ox = tx % out_width;
    int oy = (tx / out_width) % out_height;
    int oz = tx / (out_width * out_height);
    int iy = 2 * oy;
    int iz = oz;

    int64_t res = 0;
    int8_t* res_ptr = (int8_t *)&res;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        res_ptr[i] = (int8_t)-128;
    }

    float4 first = input[inOff + iz * in_height * in_width + iy * in_width + ox];
    float4 second = input[inOff + iz * in_height * in_width + iy * in_width + ox + in_width];

    int8_t* first_ptr = (int8_t*)&first;
    int8_t* second_ptr = (int8_t*)&second;

    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        res_ptr[i] = (res_ptr[i] > first_ptr[i * 2]) ? res_ptr[i] : first_ptr[i * 2];
        res_ptr[i] = (res_ptr[i] > first_ptr[i * 2 + 1]) ? res_ptr[i] : first_ptr[i * 2 + 1];
        res_ptr[i] = (res_ptr[i] > second_ptr[i * 2]) ? res_ptr[i] : second_ptr[i * 2];
        res_ptr[i] = (res_ptr[i] > second_ptr[i * 2 + 1]) ? res_ptr[i] : second_ptr[i * 2 + 1];
    }

    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        int tmp = round(res_ptr[i] * in_scale * out_scale );
        if(tmp > 127) tmp = 127;
        else if( tmp < -128) tmp = -128;
        res_ptr[i] = tmp;
    }

    output[outOff + oz * out_height * out_width + oy * out_width + ox] = res;
}
#endif

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common(
    const T* input,
    T* output,
    int64_t *indices,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {

            T res = numerical_min(T(0));
            int64_t in_index = 0;

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                    int iy = (oy + i) * stride_height + fy - pad_height;
                    int ix = (ox + j) * stride_width + fx - pad_width;
                    bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    T ival = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
                    if (res < ival) {
                        res = ival;
                        in_index = inOff + iy * in_width + ix;
                    }
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                int64_t out_index = outOff + (oy + i) * out_width + ox + j;
                output[out_index] = res;
                indices[out_index] = in_index;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common(
    const T* input,
    T* output,
    int64_t *indices,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {

            int res = numerical_min(T(0));
            int64_t in_index = 0;

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                    int iy = (oy + i) * stride_height + fy - pad_height;
                    int ix = (ox + j) * stride_width + fx - pad_width;
                    bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    T ival = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));
                    if (res < ival) {
                        res = ival;
                        in_index = inOff + iy * in_width + ix;
                    }
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                int64_t out_index = outOff + (oy + i) * out_width + ox + j;
                res = round(res * in_scale * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[out_index] = res;
                indices[out_index] = in_index;
            }
        }
    }
}

// #################### pooling max f3s2 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_f3s2_half2_NHWC(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // register blocking for input
    half2 iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy        = oy * 2 + i - padding_height;
            int ix        = ox * 2 + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            half2 ival    = pred ? input[in_off + in_off_hw] : HALF2_MIN;
            iregs[i][j]   = ival;
        }
    }

    // pooling max & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half2 val = iregs[i * 2 + 0][j * 2 + 0];
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 0][j * 2 + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 0][j * 2 + 2]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 1][j * 2 + 0]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 1][j * 2 + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 1][j * 2 + 2]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 2][j * 2 + 0]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 2][j * 2 + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i * 2 + 2][j * 2 + 2]);

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
#endif
}

// #################### pooling max f3s1 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_f3s1_half2_NHWC(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // register blocking for input
    half2 iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy        = oy + i - padding_height;
            int ix        = ox + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            half2 ival    = pred ? input[in_off + in_off_hw] : HALF2_MIN;
            iregs[i][j]   = ival;
        }
    }
    // pooling max & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half2 val = iregs[i + 0][j + 0];
            PPL_CUDA_HALF2_MAX(val, iregs[i + 0][j + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 0][j + 2]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 1][j + 0]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 1][j + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 1][j + 2]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 2][j + 0]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 2][j + 1]);
            PPL_CUDA_HALF2_MAX(val, iregs[i + 2][j + 2]);

            if (oy + i < out_height && ox < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
#endif
}

// #################### pooling max #######################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_common_half2_NHWC(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            half2 res = HALF2_MIN;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    half2 ival    = pred ? input[in_off + in_off_hw] : HALF2_MIN;
                    PPL_CUDA_HALF2_MAX(res, ival);
                }
            }
            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = res;
            }
        }
    }
#endif
}

template <int TILE_D, int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_common_half2_NHWC(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int padding_depth,
    int padding_height,
    int padding_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * in_depth * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * out_depth * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int partH = (out_height + TILE_H - 1) / TILE_H;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = ((hw_idx / partW) % partH) * TILE_H;
    int oz    = (hw_idx / (partW * partH)) * TILE_D;

    // pooling
    for(int k = 0; k < TILE_D; k++){
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
                half2 res = HALF2_MIN;
                for (int kz = 0; kz < kernel_depth; kz++) {
                    int iz = (oz + k) * stride_depth - padding_depth + kz;
                    for (int ky = 0; ky < kernel_height; ky++) {
                        for (int kx = 0; kx < kernel_width; kx++) {
                            // load input
                            int ix        = (ox + j) * stride_width - padding_width + kx;
                            int iy        = (oy + i) * stride_height - padding_height + ky;
                            bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height) && (iz >= 0 && iz < in_depth);
                            int in_off_hw = (iz * in_height * in_width + iy * in_width + ix) * pad_channels;
                            half2 ival    = pred ? input[in_off + in_off_hw] : HALF2_MIN;
                            PPL_CUDA_HALF2_MAX(res, ival);
                        }
                    }
                }
                if (oy + i < out_height && ox + j < out_width && oz + k < out_depth) {
                    int out_off_d                           = (oz + k) * out_height * out_width * pad_channels;
                    int out_off_h                           = (oy + i) * out_width * pad_channels;
                    int out_off_w                           = (ox + j) * pad_channels;
                    output[out_off + out_off_h + out_off_w + out_off_d] = res;
                }
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_common_float4_NHWC_opt(
    const float4* input,
    float4* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            float4 res;
            half* res_ptr = (half*)&res;
            #pragma unroll 8
            for (int i = 0; i < 8; i++) {
                res_ptr[i] = HALF_MIN;
            }
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    if (pred) {
                        float4 ival = input[in_off + in_off_hw];
                        half* ival_ptr = (half*)&ival;
                        res_ptr[0] = ival_ptr[0] > res_ptr[0] ? ival_ptr[0] : res_ptr[0];
                        res_ptr[1] = ival_ptr[1] > res_ptr[1] ? ival_ptr[1] : res_ptr[1];
                        res_ptr[2] = ival_ptr[2] > res_ptr[2] ? ival_ptr[2] : res_ptr[2];
                        res_ptr[3] = ival_ptr[3] > res_ptr[3] ? ival_ptr[3] : res_ptr[3];
                        res_ptr[4] = ival_ptr[4] > res_ptr[4] ? ival_ptr[4] : res_ptr[4];
                        res_ptr[5] = ival_ptr[5] > res_ptr[5] ? ival_ptr[5] : res_ptr[5];
                        res_ptr[6] = ival_ptr[6] > res_ptr[6] ? ival_ptr[6] : res_ptr[6];
                        res_ptr[7] = ival_ptr[7] > res_ptr[7] ? ival_ptr[7] : res_ptr[7];
                    }
                }
            }
            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = res;
            }
        }
    }
#endif
}

template <int TILE_D, int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_common_float4_NHWC_opt_5D(
    const float4* input,
    float4* output,
    int batch,
    int pad_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int padding_depth,
    int padding_height,
    int padding_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * in_depth * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * out_depth * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int partH = (out_height + TILE_H - 1) / TILE_H;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = ((hw_idx / partW) % partH) * TILE_H;
    int oz    = (hw_idx / (partW * partH)) * TILE_D;

    // pooling
    for(int k = 0; k < TILE_D; k++){
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
                float4 res;
                half* res_ptr = (half*)&res;
                #pragma unroll 8
                for (int i = 0; i < 8; i++) {
                    res_ptr[i] = HALF_MIN;
                }
                for(int kz = 0; kz < kernel_depth; kz++){
                    for (int ky = 0; ky < kernel_height; ky++) {
                        for (int kx = 0; kx < kernel_width; kx++) {
                            // load input
                            int ix        = (ox + j) * stride_width - padding_width + kx;
                            int iy        = (oy + i) * stride_height - padding_height + ky;
                            int iz        = (oz + k) * stride_depth - padding_depth + kz;
                            bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height) && (iz >=0 && iz < in_depth);
                            int in_off_hw = (iz * in_width * in_height + iy * in_width + ix) * pad_channels;

                            if (pred) {
                                float4 ival = input[in_off + in_off_hw];
                                half* ival_ptr = (half*)&ival;
                                res_ptr[0] = ival_ptr[0] > res_ptr[0] ? ival_ptr[0] : res_ptr[0];
                                res_ptr[1] = ival_ptr[1] > res_ptr[1] ? ival_ptr[1] : res_ptr[1];
                                res_ptr[2] = ival_ptr[2] > res_ptr[2] ? ival_ptr[2] : res_ptr[2];
                                res_ptr[3] = ival_ptr[3] > res_ptr[3] ? ival_ptr[3] : res_ptr[3];
                                res_ptr[4] = ival_ptr[4] > res_ptr[4] ? ival_ptr[4] : res_ptr[4];
                                res_ptr[5] = ival_ptr[5] > res_ptr[5] ? ival_ptr[5] : res_ptr[5];
                                res_ptr[6] = ival_ptr[6] > res_ptr[6] ? ival_ptr[6] : res_ptr[6];
                                res_ptr[7] = ival_ptr[7] > res_ptr[7] ? ival_ptr[7] : res_ptr[7];
                            }
                        }
                    }
                }
                if (oy + i < out_height && ox + j < out_width && oz + k < out_depth) {
                    int out_off_d                           = (oz + k) * out_height * out_width * pad_channels;
                    int out_off_h                           = (oy + i) * out_width * pad_channels;
                    int out_off_w                           = (ox + j) * pad_channels;
                    output[out_off + out_off_h + out_off_w + out_off_d] = res;
                }
            }
        }
    }
#endif
}

//
template <int TILE_D, int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_common_float4_NHWC_opt_5D_test(
    const int64_t* input,
    int64_t* output,
    int batch,
    int pad_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int padding_depth,
    int padding_height,
    int padding_width)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;


    int in_off  = b_idx * in_height * in_width * in_depth * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * out_depth * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int partH = (out_height + TILE_H - 1) / TILE_H;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = ((hw_idx / partW) % partH) * TILE_H;
    int oz    = (hw_idx / (partW * partH)) * TILE_D;

    // pooling
    for(int k = 0; k < TILE_D; k++){
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
                int64_t res;
                int8_t* res_ptr = (int8_t*)&res;
                #pragma unroll 8
                for (int i = 0; i < 8; i++) {
                    res_ptr[i] = INT8_MIN;
                }
                for(int kz = 0; kz < kernel_depth; kz++){
                    for (int ky = 0; ky < kernel_height; ky++) {
                        for (int kx = 0; kx < kernel_width; kx++) {
                            // load input
                            int ix        = (ox + j) * stride_width - padding_width + kx;
                            int iy        = (oy + i) * stride_height - padding_height + ky;
                            int iz        = (oz + k) * stride_depth - padding_depth + kz;
                            bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height) && (iz >=0 && iz < in_depth);
                            int in_off_hw = (iz * in_width * in_height + iy * in_width + ix) * pad_channels;

                            if (pred) {
                                int64_t ival = input[in_off + in_off_hw];
                                int8_t* ival_ptr = (int8_t*)&ival;
                                res_ptr[0] = ival_ptr[0] > res_ptr[0] ? ival_ptr[0] : res_ptr[0];
                                res_ptr[1] = ival_ptr[1] > res_ptr[1] ? ival_ptr[1] : res_ptr[1];
                                res_ptr[2] = ival_ptr[2] > res_ptr[2] ? ival_ptr[2] : res_ptr[2];
                                res_ptr[3] = ival_ptr[3] > res_ptr[3] ? ival_ptr[3] : res_ptr[3];
                                res_ptr[4] = ival_ptr[4] > res_ptr[4] ? ival_ptr[4] : res_ptr[4];
                                res_ptr[5] = ival_ptr[5] > res_ptr[5] ? ival_ptr[5] : res_ptr[5];
                                res_ptr[6] = ival_ptr[6] > res_ptr[6] ? ival_ptr[6] : res_ptr[6];
                                res_ptr[7] = ival_ptr[7] > res_ptr[7] ? ival_ptr[7] : res_ptr[7];
                            }
                        }
                    }
                }
                int out_off_d                           = (oz + k) * out_height * out_width * pad_channels;
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                int sum_index = out_off + out_off_h + out_off_w + out_off_d;
                if (oy + i < out_height && ox + j < out_width && oz + k < out_depth) {
                    output[out_off + out_off_h + out_off_w + out_off_d] = res;
                }
            }
        }
    }
#endif
}

template <int TILE_D, int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_max_common_int64_NHWC_opt_5D_int8_2(
    const uint64_t* input,
    uint64_t* output,
    int batch,
    int pad_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int padding_depth,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int dhw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * in_depth * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * out_depth * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int partH = (out_height + TILE_H - 1) / TILE_H;
    int ox    = (dhw_idx % partW) * TILE_W;
    int oy    = ((dhw_idx / partW) % partH) * TILE_H;
    int oz    = (dhw_idx / (partW * partH)) * TILE_D;

    // pooling
    for(int k = 0; k < TILE_D; k++){
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
                int64_t res = -9187201950435737472;
                int8_t* res_ptr = (int8_t*)&res;
                for(int kz = 0; kz < kernel_depth; kz++){
                    for (int ky = 0; ky < kernel_height; ky++) {
                        for (int kx = 0; kx < kernel_width; kx++) {
                            // load input
                            int ix        = (ox + j) * stride_width - padding_width + kx;
                            int iy        = (oy + i) * stride_height - padding_height + ky;
                            int iz        = (oz + k) * stride_depth - padding_depth + kz;
                            bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height) && (iz >=0 && iz < in_depth);
                            int in_off_dhw = (iz * in_width * in_height + iy * in_width + ix) * pad_channels;

                            if (pred) {
                                int64_t ival = input[in_off + in_off_dhw];
                                int8_t* ival_ptr = (int8_t*)&ival;

                                for(int index = 0; index < 8; index++){
                                    int tmp = ival_ptr[index] > res_ptr[index] ? ival_ptr[index] : res_ptr[index];
                                    tmp = round(((float)tmp) * in_scale * out_scale);
                                    if(tmp > 127) res = 127;
                                    else if(tmp < -128) res = -128;
                                    res_ptr[index] = tmp;
                                }
                            }
                        }
                    }
                }
                if (oy + i < out_height && ox + j < out_width && oz + k < out_depth) {
                    int out_off_d                           = (oz + k) * out_height * out_width * pad_channels;
                    int out_off_h                           = (oy + i) * out_width * pad_channels;
                    int out_off_w                           = (ox + j) * pad_channels;
                    output[out_off + out_off_h + out_off_w + out_off_d] = res;
                }
            }
        }
    }
#endif
}

template <typename T, int ITER>
__global__ void ppl_cukernel_pooling_max_f2s2_half_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int pad_height,
    int pad_width) // stride is 2
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int ox = hw_idx % out_width;
    int oy = hw_idx / out_width;

    // pooling
    T res;
    half* res_ptr = reinterpret_cast<half*>(&res);
    #pragma unroll
    for (int i = 0; i < ITER; i++) res_ptr[i] = HALF_MIN;
    for (int ky = 0; ky < kernel_height; ky++) {
        for (int kx = 0; kx < kernel_width; kx++) {
            // load input
            int ix        = (ox << 1) - pad_width + kx;
            int iy        = (oy << 1) - pad_height + ky;
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
            if (pred) {
                T ival    = input[in_off + in_off_hw];
                half* ival_ptr = reinterpret_cast<half*>(&ival);
                #pragma unroll
                for (int i = 0; i < ITER; ++i) {
                    res_ptr[i] = __hgt(res_ptr[i], ival_ptr[i]) ? res_ptr[i] : ival_ptr[i];
                }
            }
        }
    }
    if (oy < out_height) {
        int out_off_h  = oy * out_width * pad_channels;
        int out_off_w  = ox * pad_channels;
        output[out_off + out_off_h + out_off_w] = res;
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T res = numerical_min(T(0));
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
                    res = res > ival ? res : ival;
                }
            }
            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common_NHWC(
    const T* input,
    T* output,
    int64_t* indices,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T res = numerical_min(T(0));
            int64_t in_index = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
                    if(res < ival) {
                        res = ival;
                        in_index = in_off + in_off_hw;
                    }
                }
            }
            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = res;
                indices[out_off + out_off_h + out_off_w] = in_index;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int res = numerical_min(T(0));
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
                    res = res > ival ? res : ival;
                }
            }
            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                res = round(res * in_scale * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[out_off + out_off_h + out_off_w] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common_NHWC(
    const T* input,
    T* output,
    int64_t* indices,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int res = numerical_min(T(0));
            int64_t in_index = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
                    if(res < ival) {
                        res = ival;
                        in_index = in_off + in_off_hw;
                    }
                }
            }
            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                res = round(res * in_scale * out_scale );
                if(res > 127) res = 127;
                else if( res < -128) res = -128;
                output[out_off + out_off_h + out_off_w] = res;
                indices[out_off + out_off_h + out_off_w] = in_index;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_intpacked_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= batch * pad_channels * out_width * out_height) return;
    int c_idx = t_idx % pad_channels;
    int hw_idx = t_idx / pad_channels;

    int h_idx = (hw_idx / out_width) % out_height;
    int w_idx = hw_idx % out_width;
    int b_idx = hw_idx / (out_height * out_width);

    int in_h = h_idx * stride_height - padding_height;
    int in_w = w_idx * stride_width - padding_width;

    int in_off = b_idx * in_height * in_width * pad_channels + in_h * in_width * pad_channels + in_w * pad_channels + c_idx;
    char4 val = {(char)-128, (char)-128, (char)-128, (char)-128};
    char4 zero = {0, 0, 0, 0};
    char4 *int_input = (char4*)input;
    int h = in_h + kernel_height -1;
    int w = in_w + kernel_width -1;
    bool pred = (w >= 0 && w < in_width) && (h >= 0 && h < in_height);
    for(int i = 0; i < kernel_height; i++) {
        for (int j = 0; j < kernel_width; j++) {
            char4 src = pred ? int_input[in_off + i * in_width * pad_channels + j * pad_channels] : zero;
            val.x = src.x > val.x ? src.x : val.x;
            val.y = src.y > val.y ? src.y : val.y;
            val.z = src.z > val.z ? src.z : val.z;
            val.w = src.w > val.w ? src.w : val.w;
        }
    }
    output[t_idx] = val;
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_intpacked_common_NHWC_opt(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale,
    int insize_channels,
    int inwidth_channels,
    int total,
    DivModFast channels_mod,
    DivModFast outwidth_mod,
    DivModFast outsize_mod,
    DivModFast outheight_mod)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= total) return;
    int hw_idx = 0, c_idx = 0, h_idx = 0, w_idx = 0, b_idx = 0;

    channels_mod.divmod(t_idx, hw_idx, c_idx);
    outwidth_mod.divmod(hw_idx, h_idx, w_idx);
    h_idx = outheight_mod.mod(h_idx);
    b_idx = outsize_mod.div(hw_idx);

    int in_h = (h_idx * stride_height) - padding_height;
    int in_w = (w_idx * stride_width) - padding_width;

    int in_off = b_idx * in_height * in_width * pad_channels + in_h * in_width * pad_channels + in_w * pad_channels + c_idx;
    int64_t *int_input = (int64_t*)input;

    // int8_t out_val = {(int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128};
    int64_t out_val = -9187201950435737472;
    int8_t *val = (int8_t*)&out_val;
    for(int i = 0; i < kernel_height; i++) {
        for (int j = 0; j < kernel_width; j++) {
            bool pred = ((in_w + j) >= 0 && (in_w + j) < in_width) && ((in_h + i) >= 0 && (in_h + i) < in_height);
            if (pred) {
                int index = in_off + i * in_width * pad_channels + j * pad_channels;
                int64_t src_ = int_input[index];
                int8_t *src = (int8_t*)&src_;

                val[0] = src[0] > val[0] ? src[0] : val[0];
                val[1] = src[1] > val[1] ? src[1] : val[1];
                val[2] = src[2] > val[2] ? src[2] : val[2];
                val[3] = src[3] > val[3] ? src[3] : val[3];
                val[4] = src[4] > val[4] ? src[4] : val[4];
                val[5] = src[5] > val[5] ? src[5] : val[5];
                val[6] = src[6] > val[6] ? src[6] : val[6];
                val[7] = src[7] > val[7] ? src[7] : val[7];
            }
        }
    }
    output[t_idx] = out_val;
}


// another method for maxpool
__global__ void ppl_cukernel_pooling_max_intpacked_common_NHWC_opt_5D(
    const int64_t* input,
    int64_t* output,
    int batch,
    int pad_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int padding_depth,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale,
    int total,
    DivModFast channels_mod,
    DivModFast out_hw_mod,
    DivModFast out_width_mod,
    DivModFast out_depth_mod,
    DivModFast outsize_mod)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= total) return;
    int dhw_idx = 0, c_idx = 0, h_idx = 0, w_idx = 0, b_idx = 0, d_idx, hw_idx;

    channels_mod.divmod(t_idx, dhw_idx, c_idx);
    out_hw_mod.divmod(dhw_idx, d_idx, hw_idx);
    out_width_mod.divmod(hw_idx, h_idx, w_idx);
    d_idx = out_depth_mod.mod(d_idx);
    b_idx = outsize_mod.div(dhw_idx);

    int in_h = (h_idx * stride_height) - padding_height;
    int in_w = (w_idx * stride_width) - padding_width;
    int in_d = (d_idx * stride_depth) - padding_depth;
    int in_off = b_idx * in_height * in_width * in_depth * pad_channels + c_idx;

    int64_t out_val;
    int8_t* val = (int8_t*)&out_val;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        val[i] = INT8_MIN;
    }
    for (int k = 0; k < kernel_depth; k++){
        for(int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
                bool pred = ((in_w + j) >= 0 && (in_w + j) < in_width) && ((in_h + i) >= 0 && (in_h + i) < in_height) && ((in_d + k) >= 0 && (in_d + k) < in_depth);
                if (pred) {
                    int inoff_dhw = (in_h + i) * in_width * pad_channels + (in_w + j) * pad_channels + (in_d + k) * in_height * in_width * pad_channels;
                    int index = in_off + inoff_dhw;
                    int64_t src_ = input[index];
                    int8_t *src = (int8_t*)&src_;

                    val[0] = src[0] > val[0] ? src[0] : val[0];
                    val[1] = src[1] > val[1] ? src[1] : val[1];
                    val[2] = src[2] > val[2] ? src[2] : val[2];
                    val[3] = src[3] > val[3] ? src[3] : val[3];
                    val[4] = src[4] > val[4] ? src[4] : val[4];
                    val[5] = src[5] > val[5] ? src[5] : val[5];
                    val[6] = src[6] > val[6] ? src[6] : val[6];
                    val[7] = src[7] > val[7] ? src[7] : val[7];
                }
            }
        }
    }
    int out_off = b_idx * out_height * out_width * out_depth * pad_channels + c_idx;
    int out_off_d                           = d_idx * out_height * out_width * pad_channels;
    int out_off_h                           = h_idx * out_width * pad_channels;
    int out_off_w                           = w_idx * pad_channels;
    int sum_index = out_off + out_off_h + out_off_w + out_off_d;
    output[t_idx] = out_val;
}

// another method for maxpool 5D
__global__ void ppl_cukernel_pooling_max_fp16_common_NHWC_opt_5D(
    const float4* input,
    float4* output,
    int batch,
    int pad_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int padding_depth,
    int padding_height,
    int padding_width,
    int total,
    DivModFast channels_mod,
    DivModFast out_hw_mod,
    DivModFast out_width_mod,
    DivModFast out_depth_mod,
    DivModFast outsize_mod)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= total) return;
    int dhw_idx = 0, c_idx = 0, h_idx = 0, w_idx = 0, b_idx = 0, d_idx, hw_idx;

    channels_mod.divmod(t_idx, dhw_idx, c_idx);
    out_hw_mod.divmod(dhw_idx, d_idx, hw_idx);
    out_width_mod.divmod(hw_idx, h_idx, w_idx);
    d_idx = out_depth_mod.mod(d_idx);
    b_idx = outsize_mod.div(dhw_idx);

    int in_h = (h_idx * stride_height) - padding_height;
    int in_w = (w_idx * stride_width) - padding_width;
    int in_d = (d_idx * stride_depth) - padding_depth;

    int in_off = b_idx * in_height * in_width * in_depth * pad_channels + c_idx;

    float4 out_val;
    half* val = (half*)&out_val;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        val[i] = HALF_MIN;
    }
    for (int k = 0; k < kernel_depth; k++){
        for(int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
                bool pred = ((in_w + j) >= 0 && (in_w + j) < in_width) && ((in_h + i) >= 0 && (in_h + i) < in_height) && ((in_d + k) >= 0 && (in_d + k) < in_depth);
                if (pred) {
                    int inoff_dhw = (in_h + i) * in_width * pad_channels + (in_w + j) * pad_channels + (in_d + k) * in_height * in_width * pad_channels;
                    int index = in_off + inoff_dhw;
                    float4 src_ = input[index];
                    half *src = (half*)&src_;

                    val[0] = src[0] > val[0] ? src[0] : val[0];
                    val[1] = src[1] > val[1] ? src[1] : val[1];
                    val[2] = src[2] > val[2] ? src[2] : val[2];
                    val[3] = src[3] > val[3] ? src[3] : val[3];
                    val[4] = src[4] > val[4] ? src[4] : val[4];
                    val[5] = src[5] > val[5] ? src[5] : val[5];
                    val[6] = src[6] > val[6] ? src[6] : val[6];
                    val[7] = src[7] > val[7] ? src[7] : val[7];

                }
            }
        }
    }
    int out_off = b_idx * out_height * out_width * out_depth * pad_channels + c_idx;
    int out_off_d                           = d_idx * out_height * out_width * pad_channels;
    int out_off_h                           = h_idx * out_width * pad_channels;
    int out_off_w                           = w_idx * pad_channels;
    int sum_index = out_off + out_off_h + out_off_w + out_off_d;

    output[t_idx] = out_val;
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_intpacked_common_NHWC_opt1(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale,
    int insize_channels,
    int inwidth_channels,
    int total,
    DivModFast channels_mod,
    DivModFast outwidth_mod,
    DivModFast outsize_mod,
    DivModFast outheight_mod)
{
    int t_idx_2 = blockIdx.x * blockDim.x + threadIdx.x;
    int t_idx =t_idx_2 * 2;
    if (t_idx >= total) return;
    int hw_idx = 0, c_idx = 0, h_idx = 0, w_idx = 0, b_idx = 0;

    channels_mod.divmod(t_idx, hw_idx, c_idx);
    outwidth_mod.divmod(hw_idx, h_idx, w_idx);
    h_idx = outheight_mod.mod(h_idx);
    b_idx = outsize_mod.div(hw_idx);

    int in_h = (h_idx * stride_height) - padding_height;
    int in_w = (w_idx * stride_width) - padding_width;

    int in_off = b_idx * in_height * in_width * pad_channels + in_h * in_width * pad_channels + in_w * pad_channels + c_idx;
    int64_t *int_input = (int64_t*)input;

    // int8_t out_val = {(int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128};
    int64_t out_val = -9187201950435737472;
    int64_t out_val2 = -9187201950435737472;
    int8_t *val = (int8_t*)&out_val;
    int8_t *val2 = (int8_t*)&out_val2;

    int center_idx = in_off + in_width * pad_channels + pad_channels;
    for(int i = 0; i < kernel_height; i++) {
        for (int j = 0; j < kernel_width; j++) {
            bool pred = ((in_w + j) >= 0 && (in_w + j) < in_width) && ((in_h + i) >= 0 && (in_h + i) < in_height);
            // if (pred) {
            //     int index = in_off + i * in_width * pad_channels + j * pad_channels;
                int index = pred ? in_off + i * in_width * pad_channels + j * pad_channels : center_idx;
                int64_t src_ = int_input[index];
                int8_t *src = (int8_t*)&src_;
                int64_t src2_ = int_input[index+1];
                int8_t *src2 = (int8_t*)&src2_;

                val[0] = src[0] > val[0] ? src[0] : val[0];
                val[1] = src[1] > val[1] ? src[1] : val[1];
                val[2] = src[2] > val[2] ? src[2] : val[2];
                val[3] = src[3] > val[3] ? src[3] : val[3];
                val[4] = src[4] > val[4] ? src[4] : val[4];
                val[5] = src[5] > val[5] ? src[5] : val[5];
                val[6] = src[6] > val[6] ? src[6] : val[6];
                val[7] = src[7] > val[7] ? src[7] : val[7];

                val2[0] = src2[0] > val2[0] ? src2[0] : val2[0];
                val2[1] = src2[1] > val2[1] ? src2[1] : val2[1];
                val2[2] = src2[2] > val2[2] ? src2[2] : val2[2];
                val2[3] = src2[3] > val2[3] ? src2[3] : val2[3];
                val2[4] = src2[4] > val2[4] ? src2[4] : val2[4];
                val2[5] = src2[5] > val2[5] ? src2[5] : val2[5];
                val2[6] = src2[6] > val2[6] ? src2[6] : val2[6];
                val2[7] = src2[7] > val2[7] ? src2[7] : val2[7];
            // }
        }
    }
    output[t_idx] = out_val;
    output[t_idx+1] = out_val2;
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_intpacked_common_NCHW16(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= batch * pad_channels * out_width * out_height) return;
    int pad_channels_c1 = 16;

    int c_idx = t_idx % pad_channels_c1;

    int hw_idx = t_idx / pad_channels_c1;
    int h_idx = (hw_idx / out_width) % out_height;
    int w_idx = hw_idx % out_width;

    int bc_idx = hw_idx / (out_height * out_width);
    // int c16_idx = bc_idx % (pad_channels/pad_channels_c1);
    // int b_idx = bc_idx / (pad_channels/pad_channels_c1);

    int in_h = h_idx * stride_height - padding_height;
    int in_w = w_idx * stride_width - padding_width;

    int in_off = bc_idx * in_height * in_width * pad_channels_c1 + in_h * in_width * pad_channels_c1 + in_w * pad_channels_c1 + c_idx;
    // int in_off = b_idx * in_height * in_width * pad_channels + c16_idx * in_height * in_width * pad_channels_c1 + in_h * in_width * pad_channels_c1 + in_w * pad_channels_c1 + c_idx;

    T res = numerical_min(T(0));
    for(int i = 0; i < kernel_height; i++) {
        for (int j = 0; j < kernel_width; j++) {
            bool pred = ((in_w + j) >= 0 && (in_w + j) < in_width) && ((in_h + i) >= 0 && (in_h + i) < in_height);
            if (pred) {
                int index = in_off + i * in_width * pad_channels_c1 + j * pad_channels_c1;
                T ival    = pred ? input[index] : numerical_min(T(0));
                res = res > ival ? res : ival;
            }
        }
    }
    output[t_idx] = res;
}

//使用DivModFast替代整型除法和求模（作为kernel参数传递）
//time=0.360
template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_intpacked_common_NCHW16_opt(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    DivModFast channels_mod,
    DivModFast outwidth_mod,
    DivModFast outsize_mod,
    DivModFast outheight_mod)
{
    int t_idx_2 = blockIdx.x * blockDim.x + threadIdx.x;
    int t_idx =t_idx_2 * 2;
    if (t_idx  >= batch * pad_channels * out_width * out_height) return;
    int pad_channels_c1 = 2;

    // int c_idx = t_idx % pad_channels_c1;
    // int hw_idx = t_idx / pad_channels_c1;

    int hw_idx = 0, c_idx = 0, h_idx =0, w_idx=0,bc_idx=0;
    channels_mod.divmod(t_idx, hw_idx, c_idx);

    outwidth_mod.divmod(hw_idx,h_idx,w_idx);

    h_idx = outheight_mod.mod(h_idx);
    // int h_idx = (hw_idx / out_width) % out_height;
    // int w_idx = hw_idx % out_width;

    bc_idx = outsize_mod.div(hw_idx);

    // int bc_idx = hw_idx / (out_height * out_width);
    // int c16_idx = bc_idx % (pad_channels/pad_channels_c1);
    // int b_idx = bc_idx / (pad_channels/pad_channels_c1);

    int in_h = h_idx * stride_height - padding_height;
    int in_w = w_idx * stride_width - padding_width;
    int in_off = bc_idx * in_height * in_width * pad_channels_c1 + in_h * in_width * pad_channels_c1 + in_w * pad_channels_c1 + c_idx;
    // int in_off = b_idx * in_height * in_width * pad_channels_c1 + c16_idx * in_height * in_width * pad_channels_c1 + in_h * in_width * pad_channels_c1 + in_w * pad_channels_c1 + c_idx;

    int64_t *int_input = (int64_t*)input;

    int64_t out_val = -9187201950435737472;
    int64_t out_val2 = -9187201950435737472;

    int8_t *val = (int8_t*)&out_val;
    int8_t *val2 = (int8_t*)&out_val2;

    //int center_idx = in_off + in_width * pad_channels_c1 + pad_channels_c1;
    for(int i = 0; i < kernel_height; i++) {
        for (int j = 0; j < kernel_width; j++) {
            bool pred = ((in_w + j) >= 0 && (in_w + j) < in_width) && ((in_h + i) >= 0 && (in_h + i) < in_height);
            if (pred) {
                int index = in_off + i * in_width * pad_channels_c1 + j * pad_channels_c1;
                //int index = pred ? in_off + i * in_width * pad_channels_c1 + j * pad_channels_c1 : center_idx;

                int64_t src_ = int_input[index];
                int8_t *src = (int8_t*)&src_;
                int64_t src2_ = int_input[index+1];
                int8_t *src2 = (int8_t*)&src2_;

                val[0] = src[0] > val[0] ? src[0] : val[0];
                val[1] = src[1] > val[1] ? src[1] : val[1];
                val[2] = src[2] > val[2] ? src[2] : val[2];
                val[3] = src[3] > val[3] ? src[3] : val[3];
                val[4] = src[4] > val[4] ? src[4] : val[4];
                val[5] = src[5] > val[5] ? src[5] : val[5];
                val[6] = src[6] > val[6] ? src[6] : val[6];
                val[7] = src[7] > val[7] ? src[7] : val[7];

                val2[0] = src2[0] > val2[0] ? src2[0] : val2[0];
                val2[1] = src2[1] > val2[1] ? src2[1] : val2[1];
                val2[2] = src2[2] > val2[2] ? src2[2] : val2[2];
                val2[3] = src2[3] > val2[3] ? src2[3] : val2[3];
                val2[4] = src2[4] > val2[4] ? src2[4] : val2[4];
                val2[5] = src2[5] > val2[5] ? src2[5] : val2[5];
                val2[6] = src2[6] > val2[6] ? src2[6] : val2[6];
                val2[7] = src2[7] > val2[7] ? src2[7] : val2[7];
            }
        }
    }
    output[t_idx] = out_val;
    output[t_idx+1] = out_val2;
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_intpacked_common_NHWC_opt2(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale,
    int insize_channels,
    int inwidth_channels,
    int total,
    DivModFast channels_mod,
    DivModFast outwidth_mod,
    DivModFast outsize_mod,
    DivModFast outheight_mod)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= total) return;
    int hw_idx = 0, c_idx = 0, h_idx = 0, w_idx = 0, b_idx = 0;

    channels_mod.divmod(t_idx, hw_idx, c_idx);
    outwidth_mod.divmod(hw_idx, h_idx, w_idx);
    h_idx = outheight_mod.mod(h_idx);
    b_idx = outsize_mod.div(hw_idx);

    int in_h = (h_idx * stride_height) - padding_height;
    int in_w = (w_idx * stride_width) - padding_width;

    int in_off = b_idx * in_height * in_width * pad_channels + in_h * in_width * pad_channels + in_w * pad_channels + c_idx;
    int64_t *int_input = (int64_t*)input;

    // int8_t out_val = {(int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128, (int8_t)-128};
    int64_t out_val = -9187201950435737472;
    int8_t *val = (int8_t*)&out_val;
    for(int i = 0; i < kernel_height; i++) {
        for (int j = 0; j < kernel_width; j++) {
            bool pred = ((in_w + j) >= 0 && (in_w + j) < in_width) && ((in_h + i) >= 0 && (in_h + i) < in_height);
            if (pred) {
                int index = in_off + i * in_width * pad_channels + j * pad_channels;
                int64_t src_ = int_input[index];
                int8_t *src = (int8_t*)&src_;

                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    int res = src[k] > val[k] ? src[k] : val[k];
                    res = round(res * in_scale * out_scale );
                    if(res > 127) res = 127;
                    else if( res < -128) res = -128;
                    val[k] = res;
                }
            }
        }
    }
    output[t_idx] = out_val;
}




template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s2_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy        = oy * 2 + i - padding_height;
            int ix        = ox * 2 + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
            iregs[i][j]   = ival;
        }
    }

    // pooling max & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i * 2 + 0][j * 2 + 0];
            PPL_CUDA_MAX(val, iregs[i * 2 + 0][j * 2 + 1]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 0][j * 2 + 2]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 1][j * 2 + 0]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 1][j * 2 + 1]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 1][j * 2 + 2]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 2][j * 2 + 0]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 2][j * 2 + 1]);
            PPL_CUDA_MAX(val, iregs[i * 2 + 2][j * 2 + 2]);

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                // int res = round(val * in_scale * out_scale );
                // if(res > 127) res = 127;
                // else if( res < -128) res = -128;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_f3s1_NHWC(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= pad_channels)
        return;
    int hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx  = blockIdx.z;

    int in_off  = b_idx * in_height * in_width * pad_channels + c_idx;
    int out_off = b_idx * out_height * out_width * pad_channels + c_idx;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (hw_idx % partW) * TILE_W;
    int oy    = (hw_idx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy        = oy + i - padding_height;
            int ix        = ox + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            T ival    = pred ? input[in_off + in_off_hw] : numerical_min(T(0));
            iregs[i][j]   = ival;
        }
    }
    // pooling max & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            T val = iregs[i + 0][j + 0];
            PPL_CUDA_MAX(val, iregs[i + 0][j + 1]);
            PPL_CUDA_MAX(val, iregs[i + 0][j + 2]);
            PPL_CUDA_MAX(val, iregs[i + 1][j + 0]);
            PPL_CUDA_MAX(val, iregs[i + 1][j + 1]);
            PPL_CUDA_MAX(val, iregs[i + 1][j + 2]);
            PPL_CUDA_MAX(val, iregs[i + 2][j + 0]);
            PPL_CUDA_MAX(val, iregs[i + 2][j + 1]);
            PPL_CUDA_MAX(val, iregs[i + 2][j + 2]);

            if (oy + i < out_height && ox < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                // int res = round(val * in_scale * out_scale );
                // if(res > 127) res = 127;
                // else if( res < -128) res = -128;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpFp16(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const half* input,
    ppl::common::TensorShape* output_shape,
    half* output,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int pad_depth,
    int pad_height,
    int pad_width)
{
    int batch, channels, pad_channels, out_depth, out_height, out_width, in_depth, in_height, in_width;
    out_depth = 1;
    in_depth = 1;
    if(output_shape->GetDimCount() == 5){
        batch        = output_shape->GetDim(0);
        channels     = output_shape->GetDim(1);
        pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
        out_depth    = output_shape->GetDim(2);
        out_height   = output_shape->GetDim(3);
        out_width    = output_shape->GetDim(4);

    }else if(output_shape->GetDimCount() == 4){
        batch        = output_shape->GetDim(0);
        channels     = output_shape->GetDim(1);
        pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
        out_height   = output_shape->GetDim(2);
        out_width    = output_shape->GetDim(3);
    }else if(output_shape->GetDimCount() == 3){
        batch = 1;
        channels = output_shape->GetDim(0);
        pad_channels = output_shape->GetDim(0) + output_shape->GetPadding1(0);
        out_height = output_shape->GetDim(1);
        out_width = output_shape->GetDim(2);
    }

    if(input_shape->GetDimCount() == 5){
        in_depth   = input_shape->GetDim(2);
        in_height  = input_shape->GetDim(3);
        in_width   = input_shape->GetDim(4);
    }else if(input_shape->GetDimCount() == 4){
        in_height    = input_shape->GetDim(2);
        in_width     = input_shape->GetDim(3);
    }else if(input_shape->GetDimCount() == 3){
        in_height = input_shape->GetDim(1);
        in_width = input_shape->GetDim(2);
    }

    bool f3 = (kernel_height == 3) && (kernel_width == 3);
    bool f2 = (kernel_height == 2) && (kernel_width == 2);
    bool s1 = (stride_height == 1) && (stride_width == 1);
    bool s2 = (stride_height == 2) && (stride_width == 2);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        if(output_shape->GetDimCount() == 5){
            int partH = (out_height + 3) /4;
            int partW = out_width;
            int partD = out_depth;
            dim3 dim_block(32,4,1);
            dim3 dim_grid;
            dim_grid.x = (partH * partW * partD + dim_block.x - 1) / dim_block.x;
            dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
            dim_grid.z = batch;
            ppl_cukernel_pooling_max3d_common<1,4, 1, __half><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_depth, in_height, in_width, out_depth, out_height,
                out_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width,
                pad_depth, pad_height, pad_width);
            return ppl::common::RC_SUCCESS;
        }
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;

        if (f3 && s1) {
            partH      = (out_height + 5) / 6;
            dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
            ppl_cukernel_pooling_max_f3s1_half<6, 1><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_max_f3s2_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        } else {
            ppl_cukernel_pooling_max_common_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        }
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
        int partH             = (out_height + 3) / 4;
        int partW             = (out_width + 0) / 1;
        int partD             = (out_depth + 0) / 1;
        int padChannelsDivide = (pad_channels >> 1);
        dim3 dim_block(32, 8, 1);
        dim3 dim_grid;
        dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (partH * partW * partD + dim_block.y - 1) / dim_block.y;
        // dim_grid.y = padChannelsDivide;
        dim_grid.z = batch;
        if (f3 && s1 && output_shape->GetDimCount() != 5) {
#ifdef MAX_POOLING_OPT
            if (pad_channels % 8 == 0) {
                int pad_channels_new = pad_channels >> 3;
                int outHW = out_height * out_width;
                int num_elems = batch * pad_channels_new * outHW;

                dim3 dim_block(256, 1, 1);
                dim3 dim_grid(1, 1, 1);
                dim_grid.x = (num_elems + dim_block.x - 1) / dim_block.x;
                ppl_cukernel_pooling_max_common_float4_NHWC_flatten_opt<float4, 8><<<dim_grid, dim_block, 0,
                                    stream>>>((const float4*)input, (float4*)output, pad_channels_new,
                                    in_height, in_width,
                                    kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width,
                                    num_elems,
                                    DivModFast(pad_channels_new),
                                    DivModFast(out_width),
                                    DivModFast(out_height));
                return ppl::common::RC_SUCCESS;
            }
#endif // MAX_POOLING_OPT
            ppl_cukernel_pooling_max_f3s1_half2_NHWC<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        } else if (f3 && s2 && output_shape->GetDimCount() != 5) {
#ifdef MAX_POOLING_OPT
            if (pad_channels % 8 == 0) {
                int pad_channels_new = pad_channels >> 3;
                int outHW = out_height * out_width;
                int num_elems = batch * pad_channels_new * outHW;

                dim3 dim_block(256, 1, 1);
                dim3 dim_grid(1, 1, 1);
                dim_grid.x = (num_elems + dim_block.x - 1) / dim_block.x;
                ppl_cukernel_pooling_max_common_float4_NHWC_flatten_opt<float4, 8><<<dim_grid, dim_block, 0,
                                    stream>>>((const float4*)input, (float4*)output, pad_channels_new,
                                    in_height, in_width,
                                    kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width,
                                    num_elems,
                                    DivModFast(pad_channels_new),
                                    DivModFast(out_width),
                                    DivModFast(out_height));
            } else {
                ppl_cukernel_pooling_max_f3s2_half2_NHWC<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
            }
#else // MAX_POOLING_OPT
            ppl_cukernel_pooling_max_f3s2_half2_NHWC<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
#endif // MAX_POOLING_OPT
        } else if (f2 && s2 && output_shape->GetDimCount() != 5) {
            int partH             = out_height;
            int partW             = out_width;
            dim3 dim_block(32, 8, 1);
            dim3 dim_grid;
            dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
            dim_grid.z = batch;
            if (pad_channels >= 256 && (pad_channels % 256 == 0)) {
                int padChannelsDivide = (pad_channels >> 3);
                dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
                ppl_cukernel_pooling_max_f2s2_half_NHWC<float4, 8><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const float4*)input, (float4*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, pad_height, pad_width);
            } else if (pad_channels < 256 && (pad_channels % 64) == 0) {
                int padChannelsDivide = (pad_channels >> 3);
                dim_block.x = 8;
                dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
                ppl_cukernel_pooling_max_f2s2_half_NHWC<float4, 8><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const float4*)input, (float4*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, pad_height, pad_width);
            } else if ((pad_channels % 8) == 0) {
                if ((pad_channels % 32) == 0) {
                    dim_block.x = 4;
                    dim_block.y = 64;
                } else if ((pad_channels % 16) == 0) {
                    dim_block.x = 2;
                    dim_block.y = 128;
                } else {
                    dim_block.x = 1;
                    dim_block.y = 256;
                }
                int padChannelsDivide = (pad_channels >> 3);
                dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
                dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
                ppl_cukernel_pooling_max_f2s2_half_NHWC<float4, 8><<<dim_grid, dim_block, 0, stream>>>((const float4*)input,
                                                (float4*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, pad_height, pad_width);

            } else {
                int padChannelsDivide = (pad_channels >> 1);
                dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
                ppl_cukernel_pooling_max_common_half2_NHWC<1, 1><<<dim_grid,
                                                                    dim_block,
                                                                    0,
                                                                    stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
            }
        } else {
            if (output_shape->GetDimCount() == 5) {
                if(pad_channels % 8 == 0) {
                    int partH = out_height;
                    int partW = out_width;
                    int partD = out_depth;
                    if ((pad_channels % 256) == 0) {
                        dim_block.x = 32;
                        dim_block.y = 8;
                    } else if ((pad_channels % 128) == 0) {
                        dim_block.x = 16;
                        dim_block.y = 16;
                    } else if ((pad_channels % 64) == 0) {
                        dim_block.x = 8;
                        dim_block.y = 32;
                    } else if ((pad_channels % 32) == 0) {
                        dim_block.x = 4;
                        dim_block.y = 64;
                    } else if ((pad_channels % 16) == 0) {
                        dim_block.x = 2;
                        dim_block.y = 128;
                    } else {
                        dim_block.x = 1;
                        dim_block.y = 256;
                    }
                    int padChannelsDivide = (pad_channels >> 3);
                    dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
                    dim_grid.y = (partH * partW * partD + dim_block.y - 1) / dim_block.y;
                    ppl_cukernel_pooling_max_common_float4_NHWC_opt_5D<1, 1, 1><<<dim_grid, dim_block, 0, stream>>>(
                                (const float4*)input, (float4*)output, batch, padChannelsDivide, in_depth, in_height, in_width,
                                out_depth, out_height, out_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width,
                                pad_depth, pad_height, pad_width);
                } else {
                    ppl_cukernel_pooling_max_common_half2_NHWC<1, 4, 1><<<dim_grid,
                                                                    dim_block,
                                                                    0,
                                                                    stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide,
                                                                    in_depth, in_height, in_width, out_depth, out_height, out_width, kernel_depth, kernel_height, kernel_width,
                                                                    stride_depth, stride_height, stride_width, pad_depth, pad_height, pad_width);
                }
                return ppl::common::RC_SUCCESS;
            }
            if ((pad_channels % 8) == 0) {
                int partH = out_height;
                int partW = out_width;
                if ((pad_channels % 256) == 0) {
                    dim_block.x = 32;
                    dim_block.y = 8;
                } else if ((pad_channels % 128) == 0) {
                    dim_block.x = 16;
                    dim_block.y = 16;
                } else if ((pad_channels % 64) == 0) {
                    dim_block.x = 8;
                    dim_block.y = 32;
                } else if ((pad_channels % 32) == 0) {
                    dim_block.x = 4;
                    dim_block.y = 64;
                } else if ((pad_channels % 16) == 0) {
                    dim_block.x = 2;
                    dim_block.y = 128;
                } else {
                    dim_block.x = 1;
                    dim_block.y = 256;
                }
                int padChannelsDivide = (pad_channels >> 3);
                dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
                dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
                ppl_cukernel_pooling_max_common_float4_NHWC_opt<1, 1><<<dim_grid, dim_block, 0, stream>>>(
                            (const float4*)input, (float4*)output, batch, padChannelsDivide, in_height, in_width,
                            out_height, out_width, kernel_height, kernel_width, stride_height, stride_width,
                            pad_height, pad_width);
            } else {
                ppl_cukernel_pooling_max_common_half2_NHWC<4, 1><<<dim_grid,
                                                                    dim_block,
                                                                    0,
                                                                    stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
            }
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpFp16(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const half* input,
    ppl::common::TensorShape* output_shape,
    half* output,
    ppl::common::TensorShape* indices_shape,
    int64_t* indices,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int batch        = output_shape->GetDim(0);
    // int channels     = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height   = output_shape->GetDim(2);
    int out_width    = output_shape->GetDim(3);
    int in_height    = input_shape->GetDim(2);
    int in_width     = input_shape->GetDim(3);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;

        ppl_cukernel_pooling_max_common_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(
            input, output, indices, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpFp32(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const float* input,
    ppl::common::TensorShape* output_shape,
    float* output,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int pad_depth,
    int pad_height,
    int pad_width)
{
    int batch, channels, pad_channels,out_depth,out_height,out_width,in_depth,in_height,in_width;
    if(output_shape->GetDimCount() == 5){
        batch        = output_shape->GetDim(0);
        channels     = output_shape->GetDim(1);
        pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
        out_depth    = output_shape->GetDim(2);
        out_height   = output_shape->GetDim(3);
        out_width    = output_shape->GetDim(4);

    }else if(output_shape->GetDimCount() == 4){
        batch        = output_shape->GetDim(0);
        channels     = output_shape->GetDim(1);
        pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
        out_height   = output_shape->GetDim(2);
        out_width    = output_shape->GetDim(3);
    }else if(output_shape->GetDimCount() == 3){
        batch = 1;
        channels = output_shape->GetDim(0);
        pad_channels = output_shape->GetDim(0) + output_shape->GetPadding1(0);
        out_height = output_shape->GetDim(1);
        out_width = output_shape->GetDim(2);
    }else{
        return ppl::common::RC_UNSUPPORTED;
    }

    if(input_shape->GetDimCount() == 5){
        in_depth   = input_shape->GetDim(2);
        in_height  = input_shape->GetDim(3);
        in_width   = input_shape->GetDim(4);
    }else if(input_shape->GetDimCount() == 4){
        in_height    = input_shape->GetDim(2);
        in_width     = input_shape->GetDim(3);
    }else if(input_shape->GetDimCount() == 3){
        in_height = input_shape->GetDim(1);
        in_width = input_shape->GetDim(2);
    }else{
        return ppl::common::RC_UNSUPPORTED;
    }

    bool f3 = (kernel_height == 3) && (kernel_width == 3);
    bool s1 = (stride_height == 1) && (stride_width == 1);
    bool s2 = (stride_height == 2) && (stride_width == 2);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        if(input_shape->GetDimCount() == 5){
            int partH = (out_height + 3) /4;
            int partW = out_width;
            int partD = out_depth;
            dim3 dim_block(32,4,1);
            dim3 dim_grid;
            dim_grid.x = (partH * partW * partD + dim_block.x - 1) / dim_block.x;
            dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
            dim_grid.z = batch;
            ppl_cukernel_pooling_max3d_common<1,4, 1, float><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_depth, in_height, in_width, out_depth, out_height,
                out_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width,
                pad_depth, pad_height, pad_width);
            return ppl::common::RC_SUCCESS;
        }
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;

        if (f3 && s1) {
            partH = (out_height + 5) / 6;
            dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
            ppl_cukernel_pooling_max_f3s1<6, 1, float><<<dim_grid, dim_block, 0, stream>>>(
              input, output, batch, pad_channels, in_height, in_width, out_height,
              out_width, kernel_height, kernel_width, stride_height, stride_width,
              pad_height, pad_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_max_f3s2<4, 1, float><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height,
                out_width, kernel_height, kernel_width, stride_height, stride_width,
                pad_height, pad_width);
        } else {
            ppl_cukernel_pooling_max_common<4, 1, float><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height,
                out_width, kernel_height, kernel_width, stride_height, stride_width,
                pad_height, pad_width);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpFp32(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const float* input,
    ppl::common::TensorShape* output_shape,
    float* output,
    ppl::common::TensorShape* indices_shape,
    int64_t* indices,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int batch = output_shape->GetDim(0);
    // int channels = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height = output_shape->GetDim(2); int out_width = output_shape->GetDim(3);
    int in_height = input_shape->GetDim(2); int in_width = input_shape->GetDim(3);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y; //per thread per chl maxpool
        dim_grid.z = batch;

        ppl_cukernel_pooling_max_common<4, 1, float><<<dim_grid, dim_block, 0, stream>>>(
            input, output, indices, batch, pad_channels, in_height, in_width, out_height,
            out_width, kernel_height, kernel_width, stride_height, stride_width,
            pad_height, pad_width);
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpInt8(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const int8_t* input,
    ppl::common::TensorShape* output_shape,
    int8_t* output,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int pad_depth,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int batch, channels, pad_channels,out_depth,out_height,out_width,in_depth,in_height,in_width;
    in_depth = 1;
    out_depth = 1;
    if(output_shape->GetDimCount() == 5){
        batch        = output_shape->GetDim(0);
        channels     = output_shape->GetDim(1);
        pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
        out_depth    = output_shape->GetDim(2);
        out_height   = output_shape->GetDim(3);
        out_width    = output_shape->GetDim(4);

    }else if(output_shape->GetDimCount() == 4){
        batch        = output_shape->GetDim(0);
        channels     = output_shape->GetDim(1);
        pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
        out_height   = output_shape->GetDim(2);
        out_width    = output_shape->GetDim(3);
    }else if(output_shape->GetDimCount() == 3){
        batch = 1;
        channels = output_shape->GetDim(0);
        pad_channels = output_shape->GetDim(0) + output_shape->GetPadding1(0);
        out_height = output_shape->GetDim(1);
        out_width = output_shape->GetDim(2);
    }

    if(input_shape->GetDimCount() == 5){
        in_depth   = input_shape->GetDim(2);
        in_height  = input_shape->GetDim(3);
        in_width   = input_shape->GetDim(4);
    }else if(input_shape->GetDimCount() == 4){
        in_height    = input_shape->GetDim(2);
        in_width     = input_shape->GetDim(3);
    }else if(input_shape->GetDimCount() == 3){
        in_height = input_shape->GetDim(1);
        in_width = input_shape->GetDim(2);
    }

    bool f3 = (kernel_height == 3) && (kernel_width == 3);
    bool f2 = (kernel_height == 2) && (kernel_width == 2);
    bool s1 = (stride_height == 1) && (stride_width == 1);
    bool s2 = (stride_height == 2) && (stride_width == 2);
    bool d1 = (kernel_depth == 1) && (stride_depth == 1);
    bool data_dim = output_shape->GetDimCount() != 5;
    if(output_shape->GetDimCount() != 5 && !d1){
        return ppl::common::RC_OTHER_ERROR;
    }

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
#ifdef MAX_POOLING_OPT
        {
            bool f2s2p0_opt = (f2 && s2 && data_dim && !pad_height && !pad_width && in_width % 16 == 0);
            if (f2s2p0_opt) {
                in_width = in_width >> 4;
                out_width = out_width >> 3;
                dim3 dim_block(256, 1, 1);
                dim3 dim_grid;
                dim_grid.x = (out_height * out_width * out_depth + dim_block.x - 1) / dim_block.x;
                dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
                dim_grid.z = batch;

                ppl_cukernel_pooling_max_common_nd_int8_f2s2p0_opt<1, 1, int64_t><<<dim_grid, dim_block, 0, stream>>>(
                    (float4*)input, (int64_t*)output, pad_channels, in_depth, in_height, in_width, out_depth, out_height,
                    out_width, in_scale, out_scale);
                return ppl::common::RC_SUCCESS;
            }
        }
        int num_imgs = batch * pad_channels;
        bool quad = (kernel_height==kernel_width) && (stride_width == stride_height) && (pad_width == pad_height);
        bool may_use_aligned_imgs_per_block = SHARED_SM / (in_width*in_height) >= 4;
        bool may_use_aligned_lines_with_stride = SHARED_SM / (stride_height * in_width) >= 4;
        bool kernel_launched = false;
        bool no_pad = pad_width == 0 && pad_height == 0;
        bool no_edge = out_width*stride_width==in_width
            && out_height*stride_height==in_height
            && stride_width == kernel_width
            && stride_height == kernel_height
            && no_pad;

        if (may_use_aligned_imgs_per_block && data_dim) {
#define CALL_TI_OPT(K,S,P,IMGS_PER_BLOCK,HAVE_EDGE,RESTRICTED) \
if (in_scale*out_scale==1.0) { \
    ppl_cukernel_pooling_max_common_ti_quad_int8_opt<K,S,P,IMGS_PER_BLOCK,HAVE_EDGE,1,RESTRICTED><<<g,b,0,stream>>>( \
        input, output, num_imgs, in_height, in_width, out_height, out_width, DivModFast(out_width), in_scale*out_scale \
    ); \
} else {\
    ppl_cukernel_pooling_max_common_ti_quad_int8_opt<K,S,P,IMGS_PER_BLOCK,HAVE_EDGE,0,RESTRICTED><<<g,b,0,stream>>>( \
        input, output, num_imgs, in_height, in_width, out_height, out_width, DivModFast(out_width), in_scale*out_scale \
    ); \
}\
kernel_launched = true; \
break;

#define CALL_TI_OPT_WITH_IMAGES(K,S,P,HAVE_EDGE,RESTRICTED) \
switch (imgs_per_block) { \
    case 4:   \
    CALL_TI_OPT(K,S,P,4,HAVE_EDGE,RESTRICTED) \
    case 8:   \
    CALL_TI_OPT(K,S,P,8,HAVE_EDGE,RESTRICTED) \
    case 16:  \
    CALL_TI_OPT(K,S,P,16,HAVE_EDGE,RESTRICTED) \
    case 32:  \
    CALL_TI_OPT(K,S,P,32,HAVE_EDGE,RESTRICTED) \
    default:  \
    CALL_TI_OPT(K,S,P,64,HAVE_EDGE,RESTRICTED) \
}
#define SELECT_TI_KERNEL(K,S,P,HAVE_EDGE,RESTRICTED) \
if (!kernel_launched && kernel_height == K && stride_height == S && pad_height == P && restricted == RESTRICTED) {\
    CALL_TI_OPT_WITH_IMAGES(K,S,P,HAVE_EDGE,RESTRICTED) \
}
            int imgs_per_block_pre = SHARED_SM/(in_width*in_height);
            int imgs_per_block = 4;
            while (imgs_per_block * 2 < imgs_per_block_pre) imgs_per_block*=2;
            if (imgs_per_block > 64) imgs_per_block = 64;
            dim3 b = dim3(256,1,1);
            dim3 g = dim3((num_imgs + imgs_per_block -1)/imgs_per_block, 1, 1);
            //Restrict means every pixel in out_image can be found in in_image, and this pixel must be in the kernel
            //so we can use it as an address base when apply kernel
            //ie. in_width=5, out_width=3, every ox*2 < in_width
            //ie. in_width=112, out_width=57, the last pixel cannot be found in in_image
            bool restricted = (out_width-1)*stride_width < in_width && (out_height-1)*stride_height < in_height;
            if (quad) {
                if (no_edge) {
                    //When kernel = 2x2 and pad = 0x0 and stride = 2x2, input_size=output_size*2
                    //Or kernel = 3x3 and pad = 0x0 and stride = 3x3, input_size=output_size*3
                    SELECT_TI_KERNEL(2,2,0,0,1)
                    SELECT_TI_KERNEL(3,3,0,0,1)
                } else {
                    SELECT_TI_KERNEL(2,2,0,1,1)
                    SELECT_TI_KERNEL(3,1,1,1,1)
                    //.ie in_width=112, out_width=57 is not restricted
                    SELECT_TI_KERNEL(3,1,1,1,0)
                    SELECT_TI_KERNEL(3,2,0,1,1)
                    SELECT_TI_KERNEL(3,2,1,1,1)
                    SELECT_TI_KERNEL(3,2,1,1,0)
                    SELECT_TI_KERNEL(5,1,2,1,1)
                    SELECT_TI_KERNEL(7,1,3,1,1)
                    SELECT_TI_KERNEL(9,1,4,1,1)
                    SELECT_TI_KERNEL(11,1,5,1,1)
                    SELECT_TI_KERNEL(13,1,6,1,1)
                }
            }
            if (!kernel_launched) {
                ppl_cukernel_pooling_max_common_ti_int8_opt<<<g, b, 0, stream>>>(
                        input, output, num_imgs, imgs_per_block, in_height, in_width, out_height,
                        out_width, kernel_height, kernel_width, stride_height, stride_width,
                        pad_height, pad_width, DivModFast(out_width), in_scale*out_scale);
                kernel_launched = true;
            }
        } else if (may_use_aligned_lines_with_stride && quad && data_dim) {
#define SELECT_TL_KERNEL(K,S,P,LINES_PER_WARP,HAVE_EDGE,RESTRICTED) \
if (!kernel_launched && kernel_height == K && stride_height == S && pad_height == P && lines_per_warp == LINES_PER_WARP && no_edge == !HAVE_EDGE && restricted == RESTRICTED) {\
    if (in_scale * out_scale == 1.0f) { \
        ppl_cukernel_pooling_max_common_tl_quad_int8_opt<K,S,P,LINES_PER_WARP, HAVE_EDGE, 1,RESTRICTED><<<g, b, 0, stream>>>( \
        input, output, num_imgs, lines_per_block, in_height, in_width, out_height, out_width, DivModFast(out_height), in_scale*out_scale); \
    } else { \
        ppl_cukernel_pooling_max_common_tl_quad_int8_opt<K,S,P,LINES_PER_WARP, HAVE_EDGE, 0,RESTRICTED><<<g, b, 0, stream>>>( \
        input, output, num_imgs, lines_per_block, in_height, in_width, out_height, out_width, DivModFast(out_height), in_scale*out_scale); \
    } \
    kernel_launched = true; \
}
            int lines_per_block_pre = SHARED_SM / (stride_height * in_width);
            int lines_per_block = 2;
            //The shared memory should large enough to fill at least lines_per_block+pad_height*2 lines
            //At the same time, we have to make the loading start/end address aligned to 4, so a 4*2 padding
            //should be reserved
            //up_edge_extending lines = pad_height
            int extended_lines = pad_height + kernel_height - pad_height - 1;
            int last_line_adjust = stride_height - 1;
            int address_adjust = in_width % 4 == 0 ? 0 : 8;
            while (lines_per_block*2 < lines_per_block_pre && (lines_per_block*stride_height*2-last_line_adjust+extended_lines)*in_width + address_adjust < SHARED_SM) lines_per_block*=2;
            if (lines_per_block >= 128) lines_per_block = 128;
            bool restricted = (out_width-1)*stride_width < in_width && (out_height-1)*stride_height < in_height;
            dim3 b = dim3(64, 4, 1);
            switch (lines_per_block) {
                case 128:
                case 64:
                case 32:
                    b = dim3(8, 32, 1);
                    break;
                case 16:
                    b = dim3(16, 16, 1);
                    break;
                default:
                    b = dim3(64, 4, 1);
            }
            dim3 g = dim3((num_imgs * out_height + lines_per_block - 1)/lines_per_block, 1, 1);
            // printf("lines_per_block=%d, last_line_adjust = %d, extended_lines = %d, address_adjust = %d, last_acquired_memory=%d\n",
            // lines_per_block, last_line_adjust, extended_lines, address_adjust, (lines_per_block*stride_height-last_line_adjust+extended_lines)*in_width + address_adjust);
            if (lines_per_block >= 4) {
                int lines_per_warp = lines_per_block / b.y;
                // printf("lines_per_warp = %d, b.y=%d\n", lines_per_warp, b.y);
#define SELECT_ALL_TL_KERNEL(K,S,P,HAVE_EDGE,RESTRICTED) \
    SELECT_TL_KERNEL(K,S,P,4,HAVE_EDGE,RESTRICTED) \
    SELECT_TL_KERNEL(K,S,P,2,HAVE_EDGE,RESTRICTED) \
    SELECT_TL_KERNEL(K,S,P,1,HAVE_EDGE,RESTRICTED)

                SELECT_ALL_TL_KERNEL(2,2,0,0,1)
                SELECT_ALL_TL_KERNEL(3,3,0,0,1)
                SELECT_ALL_TL_KERNEL(2,2,0,1,1)
                SELECT_ALL_TL_KERNEL(3,2,0,1,1)
                SELECT_ALL_TL_KERNEL(3,2,1,1,1)
                SELECT_ALL_TL_KERNEL(3,2,1,1,0)
                SELECT_ALL_TL_KERNEL(3,1,1,1,1)
                SELECT_ALL_TL_KERNEL(3,1,1,1,0)
            }
        }
        //Use original implementation
        if (!kernel_launched) {
#endif
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        int partD = (out_depth + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW * partD + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;

        if (f3 && s1 && data_dim) {
            partH = (out_height + 5) / 6;
            dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
            ppl_cukernel_pooling_max_f3s1<6, 1, int8_t><<<dim_grid, dim_block, 0, stream>>>(
              input, output, batch, pad_channels, in_height, in_width, out_height,
              out_width, kernel_height, kernel_width, stride_height, stride_width,
              pad_height, pad_width, in_scale, out_scale);
        } else if (f3 && s2 && data_dim) {
            ppl_cukernel_pooling_max_f3s2<4, 1, int8_t><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_height, in_width, out_height,
                out_width, kernel_height, kernel_width, stride_height, stride_width,
                pad_height, pad_width, in_scale, out_scale);
        } else {
            ppl_cukernel_pooling_max_common<1, 4, 1, int8_t><<<dim_grid, dim_block, 0, stream>>>(
                input, output, batch, pad_channels, in_depth, in_height, in_width, out_depth, out_height,
                out_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width,
                pad_depth, pad_height, pad_width, in_scale, out_scale);
        }
#ifdef MAX_POOLING_OPT
        }
#endif
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
#ifdef MAX_POOLING_OPT
        //when input is 5D, its dataformat is NCDHW, which is recorded as NHWC.
        if (output_shape->GetDimCount() == 5){
            dim3 dim_block(1,1,1);
            dim3 dim_grid(1,1,batch);
            int partH = out_height;
            int partW = out_width;
            int partD = out_depth;
            if ((pad_channels % 256) == 0) {
                dim_block.x = 32;
                dim_block.y = 8;
            } else if ((pad_channels % 128) == 0) {
                dim_block.x = 16;
                dim_block.y = 16;
            } else if ((pad_channels % 64) == 0) {
                dim_block.x = 8;
                dim_block.y = 32;
            } else if ((pad_channels % 32) == 0) {
                dim_block.x = 4;
                dim_block.y = 64;
            } else if ((pad_channels % 16) == 0) {
                dim_block.x = 2;
                dim_block.y = 128;
            } else if ((pad_channels % 8) == 0){
                dim_block.x = 1;
                dim_block.y = 256;
            } else {
                int pad_channels_new = pad_channels >> 3;
                int outsize = out_height * out_width * out_depth;
                int total = batch * pad_channels_new * outsize;

                dim3 d_block(256, 1, 1);
                dim3 d_grid(1, 1, 1);
                d_grid.x = (total + dim_block.x - 1) / dim_block.x;
                ppl_cukernel_pooling_max_intpacked_common_NHWC_opt_5D<<<d_grid, d_block, 0, stream>>>((const int64_t*)input, (int64_t*)output, batch, pad_channels, in_depth, in_height, in_width,
                out_depth, out_height, out_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width, pad_depth,
                pad_height, pad_width, in_scale, out_scale, total,
                DivModFast(pad_channels_new),
                DivModFast(out_width * out_height),
                DivModFast(out_width),
                DivModFast(out_depth),
                DivModFast(outsize));
                return ppl::common::RC_SUCCESS;
            }
            int padChannelsDivide = (pad_channels >> 3);
            dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
            dim_grid.y = (partH * partW * partD + dim_block.y - 1) / dim_block.y;
            float samle_scale = fabs(1.0f - (in_scale * out_scale));
            if(samle_scale <= 0.00001f){
                ppl_cukernel_pooling_max_common_float4_NHWC_opt_5D_test<1, 1, 1><<<dim_grid, dim_block, 0, stream>>>(
                        (const int64_t*)input, (int64_t*)output, batch, padChannelsDivide, in_depth, in_height, in_width,
                        out_depth, out_height, out_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width,
                        pad_depth, pad_height, pad_width);
            }else{
                ppl_cukernel_pooling_max_common_int64_NHWC_opt_5D_int8_2<1, 1, 1><<<dim_grid, dim_block, 0, stream>>>(
                            (const uint64_t*)input, (uint64_t*)output, batch, padChannelsDivide, in_depth, in_height, in_width,
                            out_depth, out_height, out_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width,
                            pad_depth, pad_height, pad_width, in_scale, out_scale);
            }
            return ppl::common::RC_SUCCESS;
        }
        int pad_channels_new = pad_channels >> 3;
        int outsize = out_height * out_width;
        int insize_channels = in_height * in_width * pad_channels_new;
        int inwidth_channels = in_width * pad_channels_new;
        int total = batch * pad_channels_new * outsize;

        dim3 dim_block(256, 1, 1);
        dim3 dim_grid(1,1,1);
        dim_grid.x = (pad_channels_new * out_height * out_width * batch + dim_block.x - 1) / dim_block.x;
        float samle_scale = fabs(1.0 - (in_scale * out_scale));
        if (samle_scale <= 0.00001f){
            ppl_cukernel_pooling_max_intpacked_common_NHWC_opt<1, 1, int64_t><<<dim_grid, dim_block, 0,
                                stream>>>((const int64_t*)input, (int64_t*)output, batch, pad_channels_new,
                                in_height, in_width, out_height, out_width,
                                kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale,
                                insize_channels,
                                inwidth_channels,
                                total,
                                DivModFast(pad_channels_new),
                                DivModFast(out_width),
                                DivModFast(outsize),
                                DivModFast(out_height));
        }else{
            ppl_cukernel_pooling_max_intpacked_common_NHWC_opt2<1, 1, int64_t><<<dim_grid, dim_block, 0,
                                stream>>>((const int64_t*)input, (int64_t*)output, batch, pad_channels_new,
                                in_height, in_width, out_height, out_width,
                                kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale,
                                insize_channels,
                                inwidth_channels,
                                total,
                                DivModFast(pad_channels_new),
                                DivModFast(out_width),
                                DivModFast(outsize),
                                DivModFast(out_height));
        }

#else
        int partH             = (out_height + 3) / 4;
        int partW             = (out_width + 0) / 1;
        dim3 dim_block(32, 8, 1);
        dim3 dim_grid;
        dim_grid.x = (pad_channels + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;
        if (f3 && s1) {
            ppl_cukernel_pooling_max_f3s1_NHWC<4, 1, int8_t><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
            // dim3 dim_block(128, 1, 1);
            // dim3 dim_grid(1,1,1);
            // dim_grid.x = ((pad_channels >> 2) * out_height * out_width * batch + dim_block.x - 1) / dim_block.x;
            // ppl_cukernel_pooling_max_intpacked_NHWC<1,1,char4><<<dim_grid,dim_block,
            //                                                     0,
            //                                                     stream>>>((const char4*)input, (char4*)output, batch, pad_channels >> 2, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_max_f3s2_NHWC<4, 1, int8_t><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
        } else if (f2 && s2) {
            dim3 dim_block(128, 1, 1);
            dim3 dim_grid(1,1,1);
            dim_grid.x = ((pad_channels >> 2) * out_height * out_width * batch + dim_block.x - 1) / dim_block.x;
            ppl_cukernel_pooling_max_intpacked_NHWC<1,1,char4><<<dim_grid,dim_block,
                                                                0,
                                                                stream>>>((const char4*)input, (char4*)output, batch, pad_channels >> 2, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
        } else {
            ppl_cukernel_pooling_max_common_NHWC<4, 1, int8_t><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
        }
#endif
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NCHW16) {
        dim3 dim_block(256,1, 1);
        dim3 dim_grid(1,1,1);

        float samle_scale = fabs(1.0 - (in_scale * out_scale));
        if (samle_scale <= 0.00001f){

            int pad_channels_c1 = 2;
            int outsize = out_height * out_width;
            dim_grid.x = ((pad_channels >> 4)  * out_height * out_width * batch + dim_block.x - 1) / dim_block.x;
            ppl_cukernel_pooling_max_intpacked_common_NCHW16_opt<1, 1, int64_t><<<dim_grid, dim_block, 0,stream>>>((const int64_t*)input, (int64_t*)output, batch, pad_channels >> 3,in_height, in_width, out_height, out_width,kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width,DivModFast(pad_channels_c1),DivModFast(out_width),DivModFast(outsize),DivModFast(out_height));
        }else{

            dim_grid.x = (batch * out_height * out_width * pad_channels + dim_block.x - 1) / dim_block.x;
            ppl_cukernel_pooling_max_intpacked_common_NCHW16<1, 1, int8_t><<<dim_grid, dim_block, 0,stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels,in_height, in_width, out_height, out_width,kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);
        }
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImpInt8(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const int8_t* input,
    ppl::common::TensorShape* output_shape,
    int8_t* output,
    ppl::common::TensorShape* indices_shape,
    int64_t* indices,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    float in_scale,
    float out_scale)
{
    int batch = output_shape->GetDim(0);
    // int channels = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height = output_shape->GetDim(2); int out_width = output_shape->GetDim(3);
    int in_height = input_shape->GetDim(2); int in_width = input_shape->GetDim(3);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y; //per thread per chl maxpool
        dim_grid.z = batch;

        ppl_cukernel_pooling_max_common<4, 1, int8_t><<<dim_grid, dim_block, 0, stream>>>(
            input, output, indices, batch, pad_channels, in_height, in_width, out_height,
            out_width, kernel_height, kernel_width, stride_height, stride_width,
            pad_height, pad_width, in_scale, out_scale);
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
            int partH             = out_height;
            int partW             = out_width;
            dim3 dim_block(32, 8, 1);
            dim3 dim_grid;
            dim_grid.x = (pad_channels + dim_block.x - 1) / dim_block.x;
            dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
            dim_grid.z = batch;
            ppl_cukernel_pooling_max_common_NHWC<1, 1, int8_t><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const int8_t*)input, (int8_t*)output, indices, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, in_scale, out_scale);
    return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input,
    ppl::common::TensorShape* output_shape,
    void* output,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int padding_depth,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return PPLCUDAMaxPoolingForwardImpFp16(
            stream, input_shape, (const half*)input, output_shape, (half*)output,
            kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width,
            padding_depth, padding_height, padding_width);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return PPLCUDAMaxPoolingForwardImpFp32(
            stream, input_shape, (const float*)input, output_shape, (float*)output,
            kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width,
            padding_depth, padding_height, padding_width);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        out_scale = 1.0f / out_scale;
        return PPLCUDAMaxPoolingForwardImpInt8(
            stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output,
            kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width,
            padding_depth, padding_height, padding_width, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAMaxPoolingForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input,
    ppl::common::TensorShape* output_shape,
    void* output,
    ppl::common::TensorShape* indices_shape,
    int64_t* indices,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int padding_depth,
    int padding_height,
    int padding_width,
    float in_scale,
    float out_scale)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return PPLCUDAMaxPoolingForwardImpFp16(
            stream, input_shape, (const half*)input, output_shape, (half*)output,
            indices_shape, indices,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return PPLCUDAMaxPoolingForwardImpFp32(
            stream, input_shape, (const float*)input, output_shape, (float*)output,
            indices_shape, indices,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        out_scale = 1.0f / out_scale;
        return PPLCUDAMaxPoolingForwardImpInt8(
            stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output,
            indices_shape, indices,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}
