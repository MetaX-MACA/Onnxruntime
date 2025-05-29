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

#include "cudakernel/nn/pooling_ave.h"
#include "ppl/common/types.h"
#include <cuda_fp16.h>
#ifdef __MACACC__
#define USE_MACA_OPTIMIZATION
#include "cudakernel/common/divmod_fast.h"
#endif

// #################### pooling ave f3s2 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s2_half(
    const half* input,
    half* output,
    int if_exclude_padding,
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
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    half iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy      = oy * 2 + i - padding_height;
            int ix      = ox * 2 + j - padding_width;
            bool pred   = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : half(0);
        }
    }
    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            half val = iregs[i * 2 + 0][j * 2 + 0];
            val      = __hadd(val, iregs[i * 2 + 0][j * 2 + 1]);
            val      = __hadd(val, iregs[i * 2 + 0][j * 2 + 2]);
            val      = __hadd(val, iregs[i * 2 + 1][j * 2 + 0]);
            val      = __hadd(val, iregs[i * 2 + 1][j * 2 + 1]);
            val      = __hadd(val, iregs[i * 2 + 1][j * 2 + 2]);
            val      = __hadd(val, iregs[i * 2 + 2][j * 2 + 0]);
            val      = __hadd(val, iregs[i * 2 + 2][j * 2 + 1]);
            val      = __hadd(val, iregs[i * 2 + 2][j * 2 + 2]);
            if (oy + i < out_height && ox + j < out_width) {
                output[out_off + (oy + i) * out_width + ox + j] = __hdiv(val, cnt);
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_f3s2(
    const T* input, T* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy = oy * 2 + i - padding_height;
            int ix = ox * 2 + j - padding_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : T(0);
        }
    }
    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            T val = iregs[i * 2 + 0][j * 2 + 0];
            val = val + iregs[i * 2 + 0][j * 2 + 1];
            val = val + iregs[i * 2 + 0][j * 2 + 2];
            val = val + iregs[i * 2 + 1][j * 2 + 0];
            val = val + iregs[i * 2 + 1][j * 2 + 1];
            val = val + iregs[i * 2 + 1][j * 2 + 2];
            val = val + iregs[i * 2 + 2][j * 2 + 0];
            val = val + iregs[i * 2 + 2][j * 2 + 1];
            val = val + iregs[i * 2 + 2][j * 2 + 2];
            if (oy + i < out_height && ox + j < out_width) {
                output[out_off + (oy + i) * out_width + ox + j] = val / cnt;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s2_int8(
    const int8_t* input, int8_t* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width, float in_scale, float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    int8_t iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy = oy * 2 + i - padding_height;
            int ix = ox * 2 + j - padding_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : 0;
        }
    }
    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            int32_t val = iregs[i * 2 + 0][j * 2 + 0];
            val = val + iregs[i * 2 + 0][j * 2 + 1];
            val = val + iregs[i * 2 + 0][j * 2 + 2];
            val = val + iregs[i * 2 + 1][j * 2 + 0];
            val = val + iregs[i * 2 + 1][j * 2 + 1];
            val = val + iregs[i * 2 + 1][j * 2 + 2];
            val = val + iregs[i * 2 + 2][j * 2 + 0];
            val = val + iregs[i * 2 + 2][j * 2 + 1];
            val = val + iregs[i * 2 + 2][j * 2 + 2];
            if (oy + i < out_height && ox + j < out_width) {
                output[out_off + (oy + i) * out_width + ox + j] = max(min((int)round((val / cnt) * in_scale / out_scale), 127), -128);
            }
        }
    }
}

// #################### pooling ave f3s1 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s1_half(
    const half* input,
    half* output,
    int if_exclude_padding,
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
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    half iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy      = oy + i - padding_height;
            int ix      = ox + j - padding_width;
            bool pred   = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width) && (c < pad_channels);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : half(0);
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            half val = iregs[i + 0][j + 0];
            val      = __hadd(val, iregs[i + 0][j + 1]);
            val      = __hadd(val, iregs[i + 0][j + 2]);
            val      = __hadd(val, iregs[i + 1][j + 0]);
            val      = __hadd(val, iregs[i + 1][j + 1]);
            val      = __hadd(val, iregs[i + 1][j + 2]);
            val      = __hadd(val, iregs[i + 2][j + 0]);
            val      = __hadd(val, iregs[i + 2][j + 1]);
            val      = __hadd(val, iregs[i + 2][j + 2]);
            if (oy + i < out_height && ox < out_width) {
                output[out_off + (oy + i) * out_width + ox] = __hdiv(val, cnt);
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_f3s1(
    const T* input, T* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    T iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy = oy + i - padding_height;
            int ix = ox + j - padding_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width) && (c < pad_channels);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : T(0);
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            T val = iregs[i + 0][j + 0];
            val = val + iregs[i + 0][j + 1];
            val = val + iregs[i + 0][j + 2];
            val = val + iregs[i + 1][j + 0];
            val = val + iregs[i + 1][j + 1];
            val = val + iregs[i + 1][j + 2];
            val = val + iregs[i + 2][j + 0];
            val = val + iregs[i + 2][j + 1];
            val = val + iregs[i + 2][j + 2];
            if (oy + i < out_height && ox < out_width) {
                output[out_off + (oy + i) * out_width + ox] = (val / cnt);
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s1_int8(
    const int8_t* input, int8_t* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width, float in_scale, float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // register blocking for input
    int8_t iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy = oy + i - padding_height;
            int ix = ox + j - padding_width;
            bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width) && (c < pad_channels);
            iregs[i][j] = pred ? input[in_off + iy * in_width + ix] : 0;
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            int32_t val = iregs[i + 0][j + 0];
            val = val + iregs[i + 0][j + 1];
            val = val + iregs[i + 0][j + 2];
            val = val + iregs[i + 1][j + 0];
            val = val + iregs[i + 1][j + 1];
            val = val + iregs[i + 1][j + 2];
            val = val + iregs[i + 2][j + 0];
            val = val + iregs[i + 2][j + 1];
            val = val + iregs[i + 2][j + 2];
            if (oy + i < out_height && ox < out_width) {
                output[out_off + (oy + i) * out_width + ox] = max(min((int)round((val / cnt) * in_scale / out_scale), 127), -128);
            }
        }
    }
}

// #################### pooling ave #######################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_common_half(
    const half* input,
    half* output,
    int if_exclude_padding,
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
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt  = 0;
            half res = half(0);
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix    = (ox + j) * stride_width - padding_width + kx;
                    int iy    = (oy + i) * stride_height - padding_height + ky;
                    bool pred = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    half ival = pred ? input[in_off + iy * in_width + ix] : half(0);
                    res       = __hadd(res, ival);

                    // cnt exclude padding
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;
            // store output
            res = __hdiv(res, cnt);
            if (ox + j < out_width && oy + i < out_height) {
                output[out_off + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_common(
    const T* input, T* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels)
        return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            T res = T(0);
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix = (ox + j) * stride_width - padding_width + kx;
                    int iy = (oy + i) * stride_height - padding_height + ky;
                    bool pred = (ix >= 0 && ix < in_width) && (iy >=0 && iy < in_height);
                    T ival = pred ? input[in_off + iy * in_width + ix] : T(0);
                    res = res + ival;

                    // cnt exclude padding
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;
            // store output
            res = res / cnt;
            if (ox + j < out_width && oy + i < out_height) {
                output[out_off + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_common_int8(
    const int8_t* input, int8_t* output, 
    int if_exclude_padding, int batch, int pad_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width, float in_scale, float out_scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (c >= pad_channels) return;

    int in_off  = (b * pad_channels + c) * in_height * in_width;
    int out_off = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;
    int ox    = (tx % partW) * TILE_W;
    int oy    = (tx / partW) * TILE_H;

    // pooling
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            int32_t res = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix = (ox + j) * stride_width - padding_width + kx;
                    int iy = (oy + i) * stride_height - padding_height + ky;
                    bool pred = (ix >= 0 && ix < in_width) && (iy >=0 && iy < in_height);
                    int8_t ival = pred ? input[in_off + iy * in_width + ix] : 0;
                    res = res + ival;

                    // cnt exclude padding
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;
            // store output
            if (ox + j < out_width && oy + i < out_height) {
                output[out_off + (oy + i) * out_width + ox + j] = max(min((int)round((res / cnt) * in_scale / out_scale), 127), -128);
            }
        }
    }
}

#ifdef USE_MACA_OPTIMIZATION
template<int HAVE_EDGE=1>
__global__ void ppl_cukernel_pooling_ave_common_int8_opt(
    const int8_t* input, int8_t* output,
    int num_images, int if_exclude_padding,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_height,
    int stride_width, int padding_height, int padding_width,
    float s, DivModFast osz_mod, DivModFast out_width_mod)
{
    uint64_t gid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x
        + (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)*blockDim.x*blockDim.y*blockDim.z;

    const int ssz = in_width * in_height;
    //every 64 threads will process 64*8 outputs pixels
    int out_off = gid / 64 * 64 * 8;
    int images_processed = 0, c_offset = 0;
    osz_mod.divmod(out_off, images_processed, c_offset);
    int tid = gid % 64;
    if (images_processed >= num_images) return;

    uint64_t c_src_start = images_processed * ssz;
    #pragma unroll 8
    for (int i = 0; i < 8 * 64; i+=64) {
        //the output index of i of this thread
        uint64_t dix = out_off + i + tid;
        int c_offset_i = c_offset + i + tid;
        int d_imgs = 0, d_current_image_offset = 0;
        osz_mod.divmod(c_offset_i, d_imgs, d_current_image_offset);
        int d_images_processed = images_processed + d_imgs;
        if (d_images_processed >= num_images) return;

        int dx = 0, dy = 0;
        out_width_mod.divmod(d_current_image_offset, dy, dx);
        uint64_t src_start = c_src_start + d_imgs * ssz;
        int sx = dx * stride_width;
        int sy = dy * stride_height;

        const int8_t* ip = input + src_start;
        //HAVE_EDGE is a constant and will be optimized by compiler
        if (HAVE_EDGE) {
            int32_t res = 0;
            int cnt = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    int iy = ky + sy - padding_height;
                    int ix = kx + sx - padding_width;
                    if (iy >= 0 && iy < in_height && ix >= 0 && ix < in_width) {
                        cnt += 1;
                        res += ip[iy*in_width + ix];
                    }
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;

            output[dix] = max(min((int)round((res / cnt) * s), 127), -128);
        } else {
            int32_t res = 0, cnt = kernel_width * kernel_height;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    int iy = ky + sy - padding_height;
                    int ix = kx + sx - padding_width;
                    res += ip[iy*in_width + ix];
                }
            }
            output[dix] = max(min((int)round((res / cnt) * s), 127), -128);
        }
    }
}

template<int P>
__device__ inline bool predict(int in_height, int in_width, int iy, int ix) {
    return iy >= 0 && iy < in_height && ix >= 0 && ix < in_width;
}

template<>
__device__ inline bool predict<0>(int in_height, int in_width, int iy, int ix) {
    return iy < in_height && ix < in_width;
}

template<int K, int P>
__device__ inline int8_t pooling_kernel(const int8_t *ip, int if_exclude_padding, int in_height, int in_width, int sy, int sx, float s) {
    int res = 0, cnt = 0;
    for (int ky = 0; ky < K; ky++) {
        for (int kx = 0; kx < K; kx++) {
            int iy = ky + sy - P;
            int ix = kx + sx - P;
            if (predict<P>(in_height, in_width, iy, ix)) {
                cnt += 1;
                res += ip[iy*in_width + ix];
            }
        }
    }

    if (!if_exclude_padding)
        cnt = K * K;

    return max(min((int)round((res / cnt) * s), 127), -128);
}

template<int K>
__device__ inline int8_t pooling_kernel_no_edge(const int8_t *ip, int in_height, int in_width, int sy, int sx, float s) {
    int res = 0, cnt = K*K;
    for (int ky = 0; ky < K; ky++) {
        for (int kx = 0; kx < K; kx++) {
            int iy = ky + sy;
            int ix = kx + sx;
            res += ip[iy*in_width + ix];
        }
    }
    return max(min((int)round((res / cnt) * s), 127), -128);
}

template<int K, int S, int P, int HAVE_EDGE=1>
__global__ void ppl_cukernel_pooling_ave_sk_int8_opt(
    const int8_t* input, int8_t* output,
    int num_images, int if_exclude_padding,
    int in_height, int in_width, int out_height, int out_width,
    float s, DivModFast osz_mod, DivModFast out_width_mod)
{
    uint64_t gid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x
        + (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)*blockDim.x*blockDim.y*blockDim.z;

    const int ssz = in_width * in_height;
    //every 64 threads will process 64*out_width outputs pixels
    int out_off = gid / 64 * 64 * 8;
    int images_processed = 0, c_offset = 0;
    osz_mod.divmod(out_off, images_processed, c_offset);
    int tid = gid % 64;
    if (images_processed >= num_images) return;

    uint64_t c_src_start = images_processed * ssz;
    #pragma unroll 8
    for (int i = 0; i < 8 * 64; i+=64) {
        //the output index of i of this thread
        uint64_t dix = out_off + i + tid;
        int c_offset_i = c_offset + i + tid;
        int d_imgs = 0, d_current_image_offset = 0;
        osz_mod.divmod(c_offset_i, d_imgs, d_current_image_offset);
        int d_images_processed = images_processed + d_imgs;
        if (d_images_processed >= num_images) return;
        int dx = 0, dy = 0;
        out_width_mod.divmod(d_current_image_offset, dy, dx);
        uint64_t src_start = c_src_start + d_imgs * ssz;
        const int8_t* ip = input + src_start;
        int sx = dx * S;
        int sy = dy * S;
        if (HAVE_EDGE)
            output[dix] = pooling_kernel<K,P>(ip, if_exclude_padding, in_height, in_width, sy, sx, s);
        else
            output[dix] = pooling_kernel_no_edge<K>(ip, in_height, in_width, sy, sx, s);
    }
}

__global__ void ppl_cukernel_pooling_ave_common_NHWC_int8_opt(
    const int8_t* input,
    int8_t* output,
    int num_elements,
    int if_exclude_padding,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int KernelH,
    int KernelW,
    int StrideH,
    int StrideW,
    int PadH,
    int PadW,
    float io_scale,
    DivModFast pad_channels_fast,
    DivModFast kernel_W_fast,
    DivModFast out_width_fast
    )
{
    int tidx = threadIdx.x;
    int tid = blockIdx.x*blockDim.x + tidx;
    if(tid >= num_elements) return;

    __shared__ int sm_sum[2048];
    int tidx1 = tidx << 3;
    #pragma unroll 8
    for(int k = 0, index = tidx1; k < 8; k++, index++)
    {
        sm_sum[index] = 0;
    }
    int start_c, hw_idx, ox, oy;
    int tid1 = tid << 3;
    int b_idx  = blockIdx.y;
    int KernelSize = KernelH * KernelW;
    pad_channels_fast.divmod(tid1, hw_idx, start_c);
    out_width_fast.divmod(hw_idx,oy,ox);
    int in_off  = b_idx * in_height * in_width * pad_channels + start_c;
    int out_off = b_idx * out_height * out_width * pad_channels + start_c;
    int cnt = 0;
    if (!if_exclude_padding)
        cnt = KernelSize;

    const int8_t *ptr_input = input + in_off;
    int8_t* ptr_output = output + out_off;

    for(int ks = 0; ks < KernelSize; ks++)
    {
        int ky, kx;
        kernel_W_fast.divmod(ks,ky,kx);
        // int ky = ks / KernelW;
        // int kx = ks % KernelW;
        int ix = ox * StrideW - PadW + kx;
        int iy = oy * StrideH - PadH + ky;
        int in_off_hw_s = (iy * in_width + ix) * pad_channels;
        bool pred = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
        int in_off_hw = (iy * in_width + ix) * pad_channels;
        if (if_exclude_padding) {
            cnt += pred;
        }
        if(pred)
        {
            int64_t tmp = *(int64_t*)(ptr_input + in_off_hw);
            #pragma unroll 8
            for(int k = 0, index = tidx1; k < 8; k++, index++)
            {
                sm_sum[index] += (int8_t)((tmp >> (8*k))&0xFF);
            }
        }
    }

    if (oy < out_height && ox < out_width) {
        int out_off_h                           = oy * out_width * pad_channels;
        int out_off_w                           = ox * pad_channels;
        int8_t* ptr_dst = ptr_output + out_off_h + out_off_w;
        __attribute__((aligned(8))) int8_t local_buffer[8];
        #pragma unroll 8
        for(int k = 0, index = tidx1; k < 8; k++, index++)
        {
            local_buffer[k] = max(min((int)round((sm_sum[index] / cnt) * io_scale), 127), -128);
        }
        *(int64_t*)ptr_dst = *(int64_t*)local_buffer;
    }
}

__global__ void ppl_cukernel_pooling_ave_f3s1_NHWC_int8_tile_opt(
    const int8_t* input,
    int8_t* output,
    int if_exclude_padding,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int PadH,
    int PadW,
    DivModFast block_height_fast,
    float io_scale
)
{
    __shared__ union{
		int8_t m1[10][10][144];
		float4 m2[10][10][9];
	} sm_buffer;

    int b_idx, h_id;
    block_height_fast.divmod(blockIdx.z, b_idx, h_id);
	int start_c = (blockIdx.x * blockDim.x + threadIdx.x)<<4;
	if(start_c >= pad_channels) return;
	int dst_h = h_id * blockDim.z;
	int dst_w = blockIdx.y * blockDim.y;
    int dst_h_size = min(out_height - dst_h, blockDim.z);
    int dst_w_size = min(out_width - dst_w, blockDim.y);
    int src_h_size = dst_h_size + 2;
    int src_w_size = dst_w_size + 2;
	int start_h = dst_h - PadH;
	int start_w = dst_w - PadW;
    int dst_cord_h = dst_h + threadIdx.z;
    int dst_cord_w = dst_w + threadIdx.y;
    const int8_t * ptr_block_input = input + b_idx * in_height * in_width * pad_channels +  start_c;
    int8_t * ptr_block_output = output + b_idx * out_height * out_width * pad_channels + (dst_cord_h * out_width + dst_cord_w)* pad_channels +  start_c;
    
    int cnt = 0;
    if (!if_exclude_padding)
        cnt = 9;
    int h = threadIdx.z;
    {
        int cord_h = start_h + h; 
        bool pred_h = cord_h >= 0 && cord_h < in_height;
        int w = threadIdx.y;
        int cord_w = start_w + w;
        const int8_t * ptr_input = ptr_block_input + (cord_h * in_width + cord_w) * pad_channels;
        if( pred_h && cord_w >= 0 && cord_w < in_width)
        {
            float4 tmp = *(float4*)(ptr_input);
            sm_buffer.m2[h][w][threadIdx.x] = tmp;
        }
        if(w == blockDim.y - 1)
        {
            cord_w++;
            if( pred_h && cord_w >= 0 && cord_w < in_width)
            {
                float4 tmp = *(float4*)(ptr_input + pad_channels);
                sm_buffer.m2[h][w + 1][threadIdx.x] = tmp;
            }
            cord_w++;
            if( pred_h && cord_w >= 0 && cord_w < in_width)
            {
                float4 tmp = *(float4*)(ptr_input + (pad_channels << 1));
                sm_buffer.m2[h][w + 2][threadIdx.x] = tmp;
            }
        }
    }

    if(h == blockDim.z - 1)
    {
        int cord_h = start_h + blockDim.z;
        int cord_h1 = cord_h + 1;
        bool pred_h = cord_h >= 0 && cord_h < in_height;
        bool pred_h1 = cord_h1 >=0 && cord_h1 < in_height;
        int w = threadIdx.y;
        int cord_w = start_w + w;
        bool pred_w = cord_w >= 0 && cord_w < in_width;
        const int8_t * ptr_input = ptr_block_input + (cord_h * in_width + cord_w) * pad_channels;

        if( pred_h && pred_w)
        {
            float4 tmp = *(float4*)(ptr_input);
            sm_buffer.m2[h + 1][w][threadIdx.x] = tmp;
        }
        if(pred_h1 && pred_w)
        {
            float4 tmp = *(float4*)(ptr_input + in_width * pad_channels);
            sm_buffer.m2[h + 2][w][threadIdx.x] = tmp;
        }
        if(w == blockDim.y - 1)
        {
            cord_w++;
            bool pred_w1 = cord_w >= 0 && cord_w < in_width;
            if( pred_h && pred_w1)
            {
                float4 tmp = *(float4*)(ptr_input + pad_channels);
                sm_buffer.m2[h + 1][w + 1][threadIdx.x] = tmp;
            }
            if( pred_h1 && pred_w1)
            {
                float4 tmp = *(float4*)(ptr_input + (in_width + 1) * pad_channels);
                sm_buffer.m2[h + 2][w + 1][threadIdx.x] = tmp;
            }
            cord_w++;
            pred_w1 = cord_w >= 0 && cord_w < in_width;
            if( pred_h && pred_w1)
            {
                float4 tmp = *(float4*)(ptr_input + 2 * pad_channels);
                sm_buffer.m2[h + 1][w + 2][threadIdx.x] = tmp;
            }
            if( pred_h1 && pred_w1)
            {
                float4 tmp = *(float4*)(ptr_input + (in_width + 2) * pad_channels);
                sm_buffer.m2[h + 2][w + 2][threadIdx.x] = tmp;
            }
        }
    }

    int32_t sum[16] = {0};
    float4 reg;
	int8_t * dst = (int8_t *)&reg;
    
    int deal_h = dst_cord_h - PadH;
    int deal_w = dst_cord_w - PadW;

    __syncthreads();

    if(dst_cord_h >= out_height || dst_cord_w >= out_width) return; 

    for(int kh = 0; kh < 3; kh ++)
    {
        int cord_h = deal_h + kh;
        int cord_w = deal_w;
        bool pred_h = cord_h >= 0 && cord_h < in_height;
        bool pred = pred_h && cord_w >=0 && cord_w < in_width;
        if(if_exclude_padding){
            cnt += pred;
        }
        if(pred){
            #pragma unroll 16
            for(int c = 0; c < 16; c++)
            {
                sum[c] += sm_buffer.m1[cord_h - start_h][cord_w - start_w][(threadIdx.x << 4) + c];
            }
        }
        cord_w++;
        pred = pred_h && cord_w >=0 && cord_w < in_width;
        if(if_exclude_padding){
            cnt += pred;
        }
        if(pred){
            #pragma unroll 16
            for(int c = 0; c < 16; c++)
            {
                sum[c] += sm_buffer.m1[cord_h - start_h][cord_w - start_w][(threadIdx.x << 4) + c];
            }
        }
        cord_w++;
        pred = pred_h && cord_w >=0 && cord_w < in_width;
        if(if_exclude_padding){
            cnt += pred;
        }
        if(pred){
            #pragma unroll 16
            for(int c = 0; c < 16; c++)
            {
                sum[c] += sm_buffer.m1[cord_h - start_h][cord_w - start_w][(threadIdx.x << 4) + c];
            }
        }
    }

    #pragma unroll 16
    for(int k = 0 ; k < 16; k++)
    {
        dst[k] = max(min((int)round((sum[k] / cnt) * io_scale), 127), -128);
    }
    
	*(float4*)(ptr_block_output) = reg;    
}

__global__ void ppl_cukernel_pooling_ave_f3s1_NHWC_int8_opt(
    const int8_t* input,
    int8_t* output,
    int num_elements,
    int if_exclude_padding,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int PadH,
    int PadW,
    float io_scale,
    DivModFast pad_channels_fast,
    DivModFast kernel_W_fast,
    DivModFast out_width_fast
    )
{
    int tidx = threadIdx.x;
    int tid = blockIdx.x*blockDim.x + tidx;
    if(tid >= num_elements) return;
    __shared__ int sm_sum[2048];
    int tidx1 = tidx << 3;
    #pragma unroll 8
    for(int k = 0, index = tidx1; k < 8; k++, index++)
    {
        sm_sum[index] = 0;
    }
    int start_c, hw_idx, ox, oy;
    int tid1 = tid << 3;
    int b_idx  = blockIdx.y;
    int KernelSize = 9;
    pad_channels_fast.divmod(tid1, hw_idx, start_c);
    out_width_fast.divmod(hw_idx,oy,ox);
    int in_off  = b_idx * in_height * in_width * pad_channels + start_c;
    int out_off = b_idx * out_height * out_width * pad_channels + start_c;
    int cnt = 0;
    if (!if_exclude_padding)
        cnt = KernelSize;

    const int8_t *ptr_input = input + in_off;
    int8_t* ptr_output = output + out_off;

    #pragma unroll
    for(int ks = 0; ks < 9; ks++)
    {
        int ky, kx;
        kernel_W_fast.divmod(ks,ky,kx);
        int ix = ox - PadW + kx;
        int iy = oy - PadH + ky;
        int in_off_hw_s = (iy * in_width + ix) * pad_channels;
        bool pred = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
        int in_off_hw = (iy * in_width + ix) * pad_channels;
        if (if_exclude_padding) {
            cnt += pred;
        }
        if(pred)
        {
            int64_t tmp = *(int64_t*)(ptr_input + in_off_hw);
            #pragma unroll 8
            for(int k = 0, index = tidx1; k < 8; k++, index++)
            {
                sm_sum[index] += (int8_t)((tmp >> (8*k))&0xFF);
            }
        }
    }

    if (oy < out_height && ox < out_width) {
        int out_off_h                           = oy * out_width * pad_channels;
        int out_off_w                           = ox * pad_channels;
        int8_t* ptr_dst = ptr_output + out_off_h + out_off_w;
        __attribute__((aligned(8))) int8_t local_buffer[8];
        #pragma unroll 8
        for(int k = 0, index = tidx1; k < 8; k++, index++)
        {
            local_buffer[k] = max(min((int)round((sm_sum[index] / cnt) * io_scale), 127), -128);
        }
        *(int64_t*)ptr_dst = *(int64_t*)local_buffer;
    }
}

__global__ void ppl_cukernel_pooling_ave_f3s2_NHWC_int8_opt(
    const int8_t* input,
    int8_t* output,
    int num_elements,
    int if_exclude_padding,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int PadH,
    int PadW,
    float io_scale,
    DivModFast pad_channels_fast,
    DivModFast kernel_W_fast,
    DivModFast out_width_fast
    )
{
    int tidx = threadIdx.x;
    int tid = blockIdx.x*blockDim.x + tidx;
    if(tid >= num_elements) return;

    __shared__ int sm_sum[2048];
    int tidx1 = tidx << 3;
    #pragma unroll 8
    for(int k = 0, index = tidx1; k < 8; k++, index++)
    {
        sm_sum[index] = 0;
    }
    int start_c, hw_idx, ox, oy;
    int tid1 = tid << 3;
    int b_idx  = blockIdx.y;
    int KernelSize = 9;
    pad_channels_fast.divmod(tid1, hw_idx, start_c);
    out_width_fast.divmod(hw_idx,oy,ox);
    int in_off  = b_idx * in_height * in_width * pad_channels + start_c;
    int out_off = b_idx * out_height * out_width * pad_channels + start_c;
    int cnt = 0;
    if (!if_exclude_padding)
        cnt = KernelSize;

    const int8_t *ptr_input = input + in_off;
    int8_t* ptr_output = output + out_off;
    #pragma unroll
    for(int ks = 0; ks < 9; ks++)
    {
        int ky, kx;
        kernel_W_fast.divmod(ks,ky,kx);
        int ix = (ox << 1) - PadW + kx;
        int iy = (oy << 1)  - PadH + ky;
        int in_off_hw_s = (iy * in_width + ix) * pad_channels;
        bool pred = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
        int in_off_hw = (iy * in_width + ix) * pad_channels;
        if (if_exclude_padding) {
            cnt += pred;
        }
        if(pred)
        {
            int64_t tmp = *(int64_t*)(ptr_input + in_off_hw);
            #pragma unroll 8
            for(int k = 0, index = tidx1; k < 8; k++, index++)
            {
                sm_sum[index] += (int8_t)((tmp >> (8*k))&0xFF);
            }
        }
    }

    if (oy < out_height && ox < out_width) {
        int out_off_h                           = oy * out_width * pad_channels;
        int out_off_w                           = ox * pad_channels;
        int8_t* ptr_dst = ptr_output + out_off_h + out_off_w;
        __attribute__((aligned(8))) int8_t local_buffer[8];
        #pragma unroll 8
        for(int k = 0, index = tidx1; k < 8; k++, index++)
        {
            local_buffer[k] = max(min((int)round((sm_sum[index] / cnt) * io_scale), 127), -128);
        }
        *(int64_t*)ptr_dst = *(int64_t*)local_buffer;
    }
}

__global__ void ppl_cukernel_pooling_ave_common_half_NHWC_opt(    
    const half* input,
    half* output,
    int num_elements,  
    int if_exclude_padding,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int KernelH,
    int KernelW,
    int StrideH,
    int StrideW,
    int PadH,
    int PadW,
    DivModFast pad_channels_fast,
    DivModFast kernel_W_fast,
    DivModFast out_width_fast
    )
{
    int tidx = threadIdx.x;
    int tid = blockIdx.x*blockDim.x + tidx;
    if(tid >= num_elements) return;

    __shared__ float sm_sum[2048];
    int tidx1 = tidx << 3;
    #pragma unroll 8
    for(int k = 0, index = tidx1; k < 8; k++, index++)
    {
        sm_sum[index] = float(0.f);
    }
    int start_c, hw_idx, ox, oy;
    int tid1 = tid << 3;
    int b_idx  = blockIdx.y;
    int KernelSize = KernelH * KernelW;
    pad_channels_fast.divmod(tid1, hw_idx, start_c);
    out_width_fast.divmod(hw_idx,oy,ox);
    int in_off  = b_idx * in_height * in_width * pad_channels + start_c;
    int out_off = b_idx * out_height * out_width * pad_channels + start_c;
    int cnt = 0;
    if (!if_exclude_padding)
        cnt = KernelSize;
    
    const half *ptr_input = input + in_off;
    half* ptr_output = output + out_off;   
    
    for(int ks = 0; ks < KernelSize; ks++)
    {
        int ky, kx;
        kernel_W_fast.divmod(ks,ky,kx);
        int ix = ox * StrideW - PadW + kx;
        int iy = oy * StrideH - PadH + ky;
        int in_off_hw_s = (iy * in_width + ix) * pad_channels;
        bool pred = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
        int in_off_hw = (iy * in_width + ix) * pad_channels;
        if (if_exclude_padding) {
            cnt += pred;
        }
         
        if(pred)
        {
            float4 tmp = *(float4*)(ptr_input + in_off_hw);
            half *ptr_tmp = (half*)&tmp;
            #pragma unroll 8
            for(int k = 0, index = tidx1; k < 8; k++, index++)
            {
                sm_sum[index] += (float)ptr_tmp[k];
            }
        }
    }

    if (oy < out_height && ox < out_width) {
        int out_off_h                           = oy * out_width * pad_channels;
        int out_off_w                           = ox * pad_channels;
        half* ptr_dst = ptr_output + out_off_h + out_off_w;
        
        __attribute__((aligned(16))) half local_buffer[8];
        #pragma unroll 8
        for(int k = 0, index = tidx1; k < 8; k++, index++)
        {
            local_buffer[k] = (half)(sm_sum[index]/cnt);  
        }
        *(float4*)ptr_dst = *(float4*)local_buffer;        
    }
}
#endif//USE_MACA_OPTIMIZATION

// #################### pooling ave f3s2 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s2_half2_NHWC(
    const half2* input,
    half2* output,
    int if_exclude_padding,
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
            half2 ival    = pred ? input[in_off + in_off_hw] : half2(0.f, 0.f);
            iregs[i][j]   = ival;
        }
    }

    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            half2 val = iregs[i * 2 + 0][j * 2 + 0];
            val       = __hadd2(val, iregs[i * 2 + 0][j * 2 + 1]);
            val       = __hadd2(val, iregs[i * 2 + 0][j * 2 + 2]);
            val       = __hadd2(val, iregs[i * 2 + 1][j * 2 + 0]);
            val       = __hadd2(val, iregs[i * 2 + 1][j * 2 + 1]);
            val       = __hadd2(val, iregs[i * 2 + 1][j * 2 + 2]);
            val       = __hadd2(val, iregs[i * 2 + 2][j * 2 + 0]);
            val       = __hadd2(val, iregs[i * 2 + 2][j * 2 + 1]);
            val       = __hadd2(val, iregs[i * 2 + 2][j * 2 + 2]);

            val.x = __hdiv(val.x, cnt);
            val.y = __hdiv(val.y, cnt);

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_f3s2_NHWC(
    const T* input,
    T* output,
    int if_exclude_padding,
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

    // register blocking for input
    T iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy        = oy * 2 + i - padding_height;
            int ix        = ox * 2 + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            T ival    = pred ? input[in_off + in_off_hw] : T(0);
            iregs[i][j]   = ival;
        }
    }

    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            T val = iregs[i * 2 + 0][j * 2 + 0];
            val       = val + iregs[i * 2 + 0][j * 2 + 1];
            val       = val + iregs[i * 2 + 0][j * 2 + 2];
            val       = val + iregs[i * 2 + 1][j * 2 + 0];
            val       = val + iregs[i * 2 + 1][j * 2 + 1];
            val       = val + iregs[i * 2 + 1][j * 2 + 2];
            val       = val + iregs[i * 2 + 2][j * 2 + 0];
            val       = val + iregs[i * 2 + 2][j * 2 + 1];
            val       = val + iregs[i * 2 + 2][j * 2 + 2];

            val = val / cnt;

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s2_NHWC_int8(
    const int8_t* input,
    int8_t* output,
    int if_exclude_padding,
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
    int8_t iregs[TILE_H * 2 + 1][TILE_W * 2 + 1];
    for (int i = 0; i < 2 * TILE_H + 1; i++) {
        for (int j = 0; j < 2 * TILE_W + 1; j++) {
            int iy        = oy * 2 + i - padding_height;
            int ix        = ox * 2 + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            int8_t ival    = pred ? input[in_off + in_off_hw] : 0;
            iregs[i][j]   = ival;
        }
    }

    // pooling ave & store output
#pragma unroll TILE_H
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) * 2 + fy - padding_height;
                        int ix = (ox + j) * 2 + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            int32_t val = iregs[i * 2 + 0][j * 2 + 0];
            val       = val + iregs[i * 2 + 0][j * 2 + 1];
            val       = val + iregs[i * 2 + 0][j * 2 + 2];
            val       = val + iregs[i * 2 + 1][j * 2 + 0];
            val       = val + iregs[i * 2 + 1][j * 2 + 1];
            val       = val + iregs[i * 2 + 1][j * 2 + 2];
            val       = val + iregs[i * 2 + 2][j * 2 + 0];
            val       = val + iregs[i * 2 + 2][j * 2 + 1];
            val       = val + iregs[i * 2 + 2][j * 2 + 2];

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = max(min((int)round((val / cnt) * in_scale / out_scale), 127), -128);
            }
        }
    }
}

// #################### pooling ave f3s1 ##################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s1_half2_NHWC(
    const half2* input,
    half2* output,
    int if_exclude_padding,
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
            half2 ival    = pred ? input[in_off + in_off_hw] : half2(0.f, 0.f);
            iregs[i][j]   = ival;
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            half2 val = iregs[i + 0][j + 0];
            val       = __hadd2(val, iregs[i + 0][j + 1]);
            val       = __hadd2(val, iregs[i + 0][j + 2]);
            val       = __hadd2(val, iregs[i + 1][j + 0]);
            val       = __hadd2(val, iregs[i + 1][j + 1]);
            val       = __hadd2(val, iregs[i + 1][j + 2]);
            val       = __hadd2(val, iregs[i + 2][j + 0]);
            val       = __hadd2(val, iregs[i + 2][j + 1]);
            val       = __hadd2(val, iregs[i + 2][j + 2]);
            val.x     = __hdiv(val.x, cnt);
            val.y     = __hdiv(val.y, cnt);

            if (oy + i < out_height && ox < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_f3s1_NHWC(
    const T* input,
    T* output,
    int if_exclude_padding,
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

    // register blocking for input
    T iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy        = oy + i - padding_height;
            int ix        = ox + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            T ival    = pred ? input[in_off + in_off_hw] : T(0);
            iregs[i][j]   = ival;
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            T val = iregs[i + 0][j + 0];
            val       = val + iregs[i + 0][j + 1];
            val       = val + iregs[i + 0][j + 2];
            val       = val + iregs[i + 1][j + 0];
            val       = val + iregs[i + 1][j + 1];
            val       = val + iregs[i + 1][j + 2];
            val       = val + iregs[i + 2][j + 0];
            val       = val + iregs[i + 2][j + 1];
            val       = val + iregs[i + 2][j + 2];
            val       = val / cnt;

            if (oy + i < out_height && ox < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = val;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_f3s1_NHWC_int8(
    const int8_t* input,
    int8_t* output,
    int if_exclude_padding,
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
    int8_t iregs[TILE_H + 2][TILE_W + 2];
    for (int i = 0; i < TILE_H + 2; i++) {
        for (int j = 0; j < TILE_W + 2; j++) {
            int iy        = oy + i - padding_height;
            int ix        = ox + j - padding_width;
            bool pred     = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
            int in_off_hw = (iy * in_width + ix) * pad_channels;
            int8_t ival    = pred ? input[in_off + in_off_hw] : 0;
            iregs[i][j]   = ival;
        }
    }
    // pooling ave & store output
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {
            int cnt = 0;
            if (if_exclude_padding) {
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        int iy = (oy + i) + fy - padding_height;
                        int ix = (ox + j) + fx - padding_width;
                        cnt += (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                    }
                }
            } else {
                cnt = 9;
            }
            int32_t val = iregs[i + 0][j + 0];
            val       = val + iregs[i + 0][j + 1];
            val       = val + iregs[i + 0][j + 2];
            val       = val + iregs[i + 1][j + 0];
            val       = val + iregs[i + 1][j + 1];
            val       = val + iregs[i + 1][j + 2];
            val       = val + iregs[i + 2][j + 0];
            val       = val + iregs[i + 2][j + 1];
            val       = val + iregs[i + 2][j + 2];

            if (oy + i < out_height && ox < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = max(min((int)round((val / cnt) * in_scale / out_scale), 127), -128);
            }
        }
    }
}

// #################### pooling ave #######################
template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_common_half2_NHWC(
    const half2* input,
    half2* output,
    int if_exclude_padding,
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
            half2 res = half2(0.f, 0.f);
            int cnt   = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    half2 ival    = pred ? input[in_off + in_off_hw] : half2(0.f, 0.f);
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                    res = __hadd2(res, ival);
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;

            res.x = __hdiv(res.x, cnt);
            res.y = __hdiv(res.y, cnt);

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = res;
            }
        }
    }
#endif
}

template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_ave_common_NHWC(
    const T* input,
    T* output,
    int if_exclude_padding,
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
            T res = T(0);
            int cnt   = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    T ival    = pred ? input[in_off + in_off_hw] : T(0);
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                    res = res + ival;
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;

            res = res / cnt;

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = res;
            }
        }
    }
}

template <int TILE_H, int TILE_W>
__global__ void ppl_cukernel_pooling_ave_common_NHWC_int8(
    const int8_t* input,
    int8_t* output,
    int if_exclude_padding,
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
            int32_t res = 0;
            int cnt   = 0;
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // load input
                    int ix        = (ox + j) * stride_width - padding_width + kx;
                    int iy        = (oy + i) * stride_height - padding_height + ky;
                    bool pred     = (ix >= 0 && ix < in_width) && (iy >= 0 && iy < in_height);
                    int in_off_hw = (iy * in_width + ix) * pad_channels;
                    int8_t ival    = pred ? input[in_off + in_off_hw] : 0;
                    if (if_exclude_padding) {
                        cnt += pred;
                    }
                    res = res + ival;
                }
            }

            if (!if_exclude_padding)
                cnt = kernel_height * kernel_width;

            if (oy + i < out_height && ox + j < out_width) {
                int out_off_h                           = (oy + i) * out_width * pad_channels;
                int out_off_w                           = (ox + j) * pad_channels;
                output[out_off + out_off_h + out_off_w] = max(min((int)round((res / cnt) * in_scale / out_scale), 127), -128);
            }
        }
    }
}

ppl::common::RetCode PPLCUDAAvePoolingForwardImpFp16(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const half* input,
    ppl::common::TensorShape* output_shape,
    half* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int if_exclude_padding)
{
    int batch        = output_shape->GetDim(0);
    int channels     = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height   = output_shape->GetDim(2);
    int out_width    = output_shape->GetDim(3);
    int in_height    = input_shape->GetDim(2);
    int in_width     = input_shape->GetDim(3);

    bool f3 = (kernel_height == 3) && (kernel_width == 3);
    bool s1 = (stride_height == 1) && (stride_width == 1);
    bool s2 = (stride_height == 2) && (stride_width == 2);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;

        if (f3 && s1) {
            ppl_cukernel_pooling_ave_f3s1_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(input, output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_ave_f3s2_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(input, output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else {
            ppl_cukernel_pooling_ave_common_half<4, 1><<<dim_grid, dim_block, 0, stream>>>(input, output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        }
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 || 
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
#ifdef USE_MACA_OPTIMIZATION
            if((pad_channels & 7) == 0)
            {
                dim3 dim_grid;
                dim3 dim_block(256,1,1);
                int channels_per_block = pad_channels >> 3;
                int num = out_width*out_height*channels_per_block;
                dim_grid.x = (num + 255) / 256;
                dim_grid.y = batch;
                dim_grid.z = 1;
                ppl_cukernel_pooling_ave_common_half_NHWC_opt<<<dim_grid,
                                                            dim_block,
                                                            0,
                                                            stream
                                                            >>>((const half*)input, (half*)output, num, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width , kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, DivModFast(pad_channels), DivModFast(kernel_width), DivModFast(out_width));
                return ppl::common::RC_SUCCESS;   
            }
#endif//USE_MACA_OPTIMIZATION
        int partH             = (out_height + 3) / 4;
        int partW             = (out_width + 0) / 1;
        int padChannelsDivide = (pad_channels >> 1);
        dim3 dim_block(32, 8, 1);
        dim3 dim_grid;
        dim_grid.x = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
        // dim_grid.y = padChannelsDivide;
        dim_grid.z = batch;
        if (f3 && s1) {
            ppl_cukernel_pooling_ave_f3s1_half2_NHWC<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const half2*)input, (half2*)output, if_exclude_padding, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_ave_f3s2_half2_NHWC<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const half2*)input, (half2*)output, if_exclude_padding, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else {
            ppl_cukernel_pooling_ave_common_half2_NHWC<4, 1><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const half2*)input, (half2*)output, if_exclude_padding, batch, padChannelsDivide, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

template<typename T>
ppl::common::RetCode PPLCUDAAvePoolingForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const T* input,
    ppl::common::TensorShape* output_shape,
    T* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int if_exclude_padding)
{
     int batch, channels, pad_channels,out_height,out_width,in_height,in_width;
    if(output_shape->GetDimCount() == 4){
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

    if(input_shape->GetDimCount() == 4){
        in_height    = input_shape->GetDim(2);
        in_width     = input_shape->GetDim(3);
    }else if(input_shape->GetDimCount() == 3){
        in_height = input_shape->GetDim(1);
        in_width = input_shape->GetDim(2);
    }

    bool f3 = (kernel_height == 3) && (kernel_width == 3);
    bool s1 = (stride_height == 1) && (stride_width == 1);
    bool s2 = (stride_height == 2) && (stride_width == 2);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;

        if (f3 && s1) {
            ppl_cukernel_pooling_ave_f3s1<4, 1, T><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_ave_f3s2<4, 1, T><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else {
            ppl_cukernel_pooling_ave_common<4, 1, T><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width);
        }
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        int partH             = (out_height + 3) / 4; //tile
        int partW             = (out_width + 0) / 1;
        dim3 dim_block(32, 8, 1);
        dim3 dim_grid;
        dim_grid.x = (pad_channels + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;
        if (f3 && s1) {
            ppl_cukernel_pooling_ave_f3s1_NHWC<4, 1, T><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const T*)input, (T*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_ave_f3s2_NHWC<4, 1, T><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const T*)input, (T*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        } else {
            ppl_cukernel_pooling_ave_common_NHWC<4, 1, T><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const T*)input, (T*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAAvePoolingForwardImpInt8(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const int8_t* input,
    ppl::common::TensorShape* output_shape,
    int8_t* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int if_exclude_padding,
    float in_scale,
    float out_scale)
{
    int batch        = output_shape->GetDim(0);
    int channels     = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height   = output_shape->GetDim(2);
    int out_width    = output_shape->GetDim(3);
    int in_height    = input_shape->GetDim(2);
    int in_width     = input_shape->GetDim(3);

    bool f3 = (kernel_height == 3) && (kernel_width == 3);
    bool s1 = (stride_height == 1) && (stride_width == 1);
    bool s2 = (stride_height == 2) && (stride_width == 2);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
#ifdef USE_MACA_OPTIMIZATION
        int num_images = batch * ((pad_channels + 3) / 4) * 4;
        bool kernel_launched = false;
        bool quad = (kernel_height==kernel_width) && (stride_width == stride_height) && (padding_width == padding_height);
        bool no_pad = padding_width == 0 && padding_height == 0;
        bool no_edge = out_width*stride_width==in_width
            && out_height*stride_height==in_height
            && stride_width == kernel_width
            && stride_height == kernel_height
            && no_pad;
        //ie. in_width = 12, out_width = 5, k = 3, stride = 2, pad = 0, max kernel width is 4 * 2 + 3 = 11 < 12
        bool no_edge2 = no_pad && in_width == out_width*stride_width+kernel_width-1 && in_height == out_height*stride_height+kernel_height-1;
        no_edge |= no_edge2;
        dim3 dim_block(64, 4, 1);
        dim3 dim_grid;
        uint64_t block_data = dim_block.x * dim_block.y * dim_block.z * 8;
        dim_grid.x = 1;
        dim_grid.y = (((uint64_t)num_images * out_width * out_height + block_data - 1) / block_data + batch - 1) / batch;
        dim_grid.y = (dim_grid.y + 4 - 1) / 4 * 4;
        if (dim_grid.y == 0) dim_grid.y = 1;
        dim_grid.z = batch;
#define SELECT_KERNEL(K,S,P,HAVE_EDGE) \
if (kernel_height == K && stride_height == S && padding_height == P && no_edge == !HAVE_EDGE) {\
    ppl_cukernel_pooling_ave_sk_int8_opt<K,S,P,HAVE_EDGE><<<dim_grid, dim_block, \
                0, stream>>>(input, output, num_images, if_exclude_padding, \
                in_height, in_width, out_height, out_width, in_scale/out_scale, \
                DivModFast(out_height*out_width), DivModFast(out_width)); \
    kernel_launched = true; \
}
        bool valid_pixels = ((uint64_t)num_images * out_width * out_height)%(64*8) == 0;
        if (quad && valid_pixels) {
            SELECT_KERNEL(2,2,0,0)
            SELECT_KERNEL(2,2,0,1)
            SELECT_KERNEL(3,1,0,0)
            SELECT_KERNEL(3,1,0,1)
            SELECT_KERNEL(3,1,1,1)
            SELECT_KERNEL(3,2,0,0)
            SELECT_KERNEL(3,2,0,1)
            SELECT_KERNEL(3,2,1,1)
            SELECT_KERNEL(8,8,0,0)
        }
        if (!kernel_launched && valid_pixels) {
            if (no_edge)
                ppl_cukernel_pooling_ave_common_int8_opt<0><<<dim_grid, dim_block, 0, stream>>>(input, output, num_images, 
                if_exclude_padding,in_height, in_width, out_height, out_width,
                kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width,
                in_scale/out_scale, DivModFast(out_height*out_width), DivModFast(out_width));
            else
                ppl_cukernel_pooling_ave_common_int8_opt<1><<<dim_grid, dim_block, 0, stream>>>(input, output, num_images, 
                if_exclude_padding, in_height, in_width, out_height, out_width,
                kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width,
                in_scale/out_scale, DivModFast(out_height*out_width), DivModFast(out_width));
            kernel_launched = true;
        }
        if (!kernel_launched) {
#endif
        // thread layout
        int partH = (out_height + 3) / 4;
        int partW = (out_width + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;

        if (f3 && s1) {
            ppl_cukernel_pooling_ave_f3s1_int8<4, 1><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        } else if (f3 && s2) {
            ppl_cukernel_pooling_ave_f3s2_int8<4, 1><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        } else {
            ppl_cukernel_pooling_ave_common_int8<4, 1><<<dim_grid, dim_block,
                0, stream>>>(input, output, if_exclude_padding, batch, pad_channels,
                in_height, in_width, out_height, out_width, kernel_height,
                kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        }
#ifdef USE_MACA_OPTIMIZATION
        }
#endif
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
        int partH             = (out_height + 3) / 4; //tile
        int partW             = (out_width + 0) / 1;
        dim3 dim_block(32, 8, 1);
        dim3 dim_grid;
        dim_grid.x = (pad_channels + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (partH * partW + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;
        if (f3 && s1) {
#ifdef USE_MACA_OPTIMIZATION
            if((pad_channels & 15) == 0)
            {
                float io_scale = in_scale / out_scale;
                dim3 dim_block(8,8,8);
                int channels_per_block = pad_channels >> 4;
                dim3 dim_grid;
                int block_height = (out_height + 7)>>3;
                dim_grid.x = (pad_channels + 127)>>7;
                dim_grid.y = (out_width + 7)>>3;
                dim_grid.z = block_height * batch;
                ppl_cukernel_pooling_ave_f3s1_NHWC_int8_tile_opt<<<dim_grid,
                                                            dim_block,
                                                            0,
                                                            stream
                                                            >>>((const int8_t*)input, (int8_t*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width,
                                                            padding_height, padding_width,DivModFast(block_height),io_scale);
                return ppl::common::RC_SUCCESS;
            }
#endif//USE_MACA_OPTIMIZATION
            ppl_cukernel_pooling_ave_f3s1_NHWC_int8<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const int8_t*)input, (int8_t*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        } else if (f3 && s2) {
#ifdef USE_MACA_OPTIMIZATION
            if((pad_channels&7) == 0)
            {
                float io_scale = in_scale / out_scale;
                dim3 dim_block(256,1,1);
                int channels_per_block = pad_channels >> 3;
                int num = out_width*out_height*channels_per_block;
                dim_grid.x = (num + 255) / 256;
                dim_grid.y = batch;
                dim_grid.z = 1;
                ppl_cukernel_pooling_ave_f3s2_NHWC_int8_opt<<<dim_grid,
                                                            dim_block,
                                                            0,
                                                            stream
                                                        >>>((const int8_t*)input, (int8_t*)output, num, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width,
                                                                padding_height, padding_width, io_scale, DivModFast(pad_channels), DivModFast(kernel_width), DivModFast(out_width));
                return ppl::common::RC_SUCCESS;
            }
#endif//USE_MACA_OPTIMIZATION
            ppl_cukernel_pooling_ave_f3s2_NHWC_int8<4, 1><<<dim_grid,
                                                              dim_block,
                                                              0,
                                                              stream>>>((const int8_t*)input, (int8_t*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        } else {
#ifdef USE_MACA_OPTIMIZATION
            if((pad_channels & 7) == 0)
            {
                float io_scale = in_scale / out_scale;
                dim3 dim_block(256,1,1);
                int channels_per_block = pad_channels >> 3;
                int num = out_width*out_height*channels_per_block;
                dim_grid.x = (num + 255) / 256;
                dim_grid.y = batch;
                dim_grid.z = 1;
                ppl_cukernel_pooling_ave_common_NHWC_int8_opt<<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream
                                                                >>>((const int8_t*)input, (int8_t*)output, num, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width , kernel_height, kernel_width, 
                                                                    stride_height, stride_width, padding_height, padding_width, io_scale, DivModFast(pad_channels), DivModFast(kernel_width), DivModFast(out_width));
                return ppl::common::RC_SUCCESS;
            }
#endif//USE_MACA_OPTIMIZATION
            ppl_cukernel_pooling_ave_common_NHWC_int8<4, 1><<<dim_grid,
                                                            dim_block,
                                                            0,
                                                            stream>>>((const int8_t*)input, (int8_t*)output, if_exclude_padding, batch, pad_channels, in_height, in_width, out_height, out_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, in_scale, out_scale);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAAvePoolingForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input,
    ppl::common::TensorShape* output_shape,
    void* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int if_exclude_padding,
    float in_scale,
    float out_scale)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return PPLCUDAAvePoolingForwardImpFp16(
            stream, input_shape, (const half*)input, output_shape, (half*)output, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, if_exclude_padding);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return PPLCUDAAvePoolingForwardImp<float>(
            stream, input_shape, (const float*)input, output_shape, (float*)output,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width, if_exclude_padding);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        return PPLCUDAAvePoolingForwardImpInt8(
            stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width, if_exclude_padding, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}
