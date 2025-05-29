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

#include "cudakernel/nn/resize.h"
#include "ppl/common/types.h"
#include "cudakernel/common/divmod_fast.h"
#ifdef __MACACC__
#define OPT_RESIZE
#endif//__MACACC__
// void printtoHost(float *dev,int batchsize,int channel,int height,int width, int format);

static void GetNumBlocks(int32_t& block_size, dim3& grid_size, int32_t& channels_per_piece,
    const int32_t num_threads, const int32_t channels) {
    int dev = 0;
    cudaGetDevice(&dev);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
#ifdef OPT_RESIZE
    sm_count <<= 5;
#endif//OPT_RESIZE
    // guarantee enough blocks
    block_size = 32;
    while ((num_threads / block_size) >= 4 && block_size < 256) {
        block_size = block_size << 1;
    }
    grid_size.x = (num_threads + block_size - 1) / block_size;
    channels_per_piece = 8;
    if (channels <= channels_per_piece) {
        channels_per_piece = channels;
        grid_size.y = 1;
        return;
    }
    int32_t expected_blocks = sm_count * 4;
    grid_size.y = (channels + channels_per_piece - 1) / channels_per_piece;
    while (channels_per_piece < channels &&
        grid_size.x * grid_size.y > expected_blocks) {
        channels_per_piece = channels_per_piece << 1;
        grid_size.y = (channels + channels_per_piece - 1) / channels_per_piece;
    }
    if (channels_per_piece > channels) {
        channels_per_piece = channels;
        grid_size.y = (channels + channels_per_piece - 1) / channels_per_piece;
    }
}

struct half8_ {
    half x0;
    half y0;
    half z0;
    half w0;
    half x1;
    half y1;
    half z1;
    half w1;
};

// static inline __device__ float cudaComputeSourceIndexCubic(
//     float scale,
//     int dstIndex)
// {
//     float srcIdx = scale * (dstIndex + 0.5) - 0.5;
//     return srcIdx;
// }
//
// static inline __device__ float cudaComputeSourceIndexNearest(
//     float scale,
//     int dstIndex,
//     int transform_mode)
// {
//     float srcIdx = 0.f;
//     if (transform_mode == 3) {
//         srcIdx = scale * dstIndex;
//     } else {
//         srcIdx = scale * (dstIndex + 0.5) - 0.5;
//     }
//     return (srcIdx < 0) ? 0.f : srcIdx;
// }
//
// static inline __device__ float cudaComputeSourceIndexBilinear(
//     float scale,
//     int dstIndex)
// {
//     float srcIdx = scale * (dstIndex + 0.5) - 0.5;
//     return (srcIdx < 0) ? 0.f : srcIdx;
// }

static __device__ __forceinline__ float cudaComputeSourceIndex(
    float scale,
    int dstIndex,
    int transform_mode)
{
    float srcIdx = 0.f;
    if (transform_mode == 0 || transform_mode == 1) {
        srcIdx = scale * (dstIndex + 0.5) - 0.5;
    } else if (transform_mode == 2 || transform_mode == 3) {
        srcIdx = scale * dstIndex;
    } else {
        srcIdx = scale * (dstIndex + 0.5);
    }
    return (srcIdx < 0) ? 0.f : srcIdx;
}

static inline __device__ double cudaComputeSourceSTDIndex(
    double scale,
    int dstIndex,
    int transform_mode)
{
    double srcIdx = 0.f;
    if (transform_mode == 0 || transform_mode == 1) {
        srcIdx = scale * (dstIndex + 0.5) - 0.5;
    } else if (transform_mode == 2 || transform_mode == 3) {
        srcIdx = scale * dstIndex;
    } else {
        srcIdx = scale * (dstIndex + 0.5);
    }
    return (srcIdx < 0) ? 0.f : srcIdx;
}

static inline __device__ float cudaComputeBilinearSourceIndex(
    float scale,
    float roi_0,
    float roi_1,
    int dstIndex,
    int output_width,
    int input_width,
    int transform_mode)
{
    float srcIdx = 0.f;
    if (transform_mode == 0 || transform_mode == 1) {
        if(transform_mode == 1&&output_width == 1){
            srcIdx = - 0.5;
        }else{
            srcIdx = scale * (dstIndex + 0.5) - 0.5;
        }
    } else if (transform_mode == 2 || transform_mode == 3) {
        srcIdx = scale * dstIndex;
    } else if (transform_mode == 5){
        if(output_width == 1){
            srcIdx = (roi_1 - roi_0)*(input_width - 1)/2;
        }else{
            srcIdx = (roi_1 - roi_0)*scale*dstIndex;
        }
        srcIdx += roi_0 * (input_width - 1);
    }else {
        srcIdx = scale * (dstIndex + 0.5);
    }
    return (srcIdx < 0) ? 0.f : srcIdx;;
}

static __device__ __forceinline__ float cubic_convolution1(
    float x,
    float A)
{
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}

static __device__ __forceinline__ float cubic_convolution2(
    float x,
    float A)
{
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

static __device__ __forceinline__ void get_cubic_resize_coefficients(
    float coeffs[4],
    float t,
    float cubic_coeff)
{
    float A = cubic_coeff;

    float x1  = t;
    coeffs[0] = cubic_convolution2(x1 + 1.0, A);
    coeffs[1] = cubic_convolution1(x1, A);

    // opposite coefficients
    float x2  = 1.0 - t;
    coeffs[2] = cubic_convolution1(x2, A);
    coeffs[3] = cubic_convolution2(x2 + 1.0, A);
}

template <typename T>
static __device__ inline T cubic_interplote(float frac0, T data0, float frac1, T data1, float frac2, T data2, float frac3, T data3);

template<typename T>
static __device__ inline T cubic_interplote(float frac0, T data0, float frac1, T data1, float frac2, T data2, float frac3, T data3) {
    T res;
    res = frac0 * data0 + frac1 * data1 +
          frac2 * data2 + frac3 * data3;
    return res;
}


__device__ inline float cubic_interplote_float(float frac0, float data0, float frac1, float data1, float frac2, float data2, float frac3, float data3) {
    float res;
    res = frac0 * data0 + frac1 * data1 +
          frac2 * data2 + frac3 * data3;
    return res;
}

template <>
__device__ inline half cubic_interplote<half>(float frac0, half data0, float frac1, half data1, float frac2, half data2, float frac3, half data3)
{
    half res;
    res = frac0 * __half2float(data0) + frac1 * __half2float(data1) +
          frac2 * __half2float(data2) + frac3 * __half2float(data3);
    return res;
}

// template <>
// __device__ inline half8_ cubic_interplote<half8_>(float frac0, half8_ data0, float frac1, half8_ data1, float frac2, half8_ data2, float frac3, half8_ data3)
// {
//     half8_ res;
//     res.x0 = frac0 * __half2float(data0.x0) + frac1 * __half2float(data1.x0) +
//              frac2 * __half2float(data2.x0) + frac3 * __half2float(data3.x0);
//     res.y0 = frac0 * __half2float(data0.y0) + frac1 * __half2float(data1.y0) +
//              frac2 * __half2float(data2.y0) + frac3 * __half2float(data3.y0);
//     res.z0 = frac0 * __half2float(data0.z0) + frac1 * __half2float(data1.z0) +
//              frac2 * __half2float(data2.z0) + frac3 * __half2float(data3.z0);
//     res.w0 = frac0 * __half2float(data0.w0) + frac1 * __half2float(data1.w0) +
//              frac2 * __half2float(data2.w0) + frac3 * __half2float(data3.w0);
//     res.x1 = frac0 * __half2float(data0.x1) + frac1 * __half2float(data1.x1) +
//              frac2 * __half2float(data2.x1) + frac3 * __half2float(data3.x1);
//     res.y1 = frac0 * __half2float(data0.y1) + frac1 * __half2float(data1.y1) +
//              frac2 * __half2float(data2.y1) + frac3 * __half2float(data3.y1);
//     res.z1 = frac0 * __half2float(data0.z1) + frac1 * __half2float(data1.z1) +
//              frac2 * __half2float(data2.z1) + frac3 * __half2float(data3.z1);
//     res.w1 = frac0 * __half2float(data0.w1) + frac1 * __half2float(data1.w1) +
//              frac2 * __half2float(data2.w1) + frac3 * __half2float(data3.w1);
//     return res;
// }

// template <typename T>
// static __device__ __forceinline__ T cubic_interp1d(
//     T x0,
//     T x1,
//     T x2,
//     T x3,
//     float t,
//     float cubic_coeff)
// {
//     float coeffs[4];
//     get_cubic_resize_coefficients(coeffs, t, cubic_coeff);

//     return cubic_interplote<T>(coeffs[0], x0, coeffs[1], x1, coeffs[2], x2, coeffs[3], x3);
// }

template <typename T>
static __device__ __forceinline__ T cubic_interp1dx(
    T x0,
    T x1,
    T x2,
    T x3,
    int w0,
    int w1,
    int w2,
    int w3,
    int input_width,
    int exclude_outside,
    float t,
    float cubic_coeff)
{
    float coeffs[4];
    get_cubic_resize_coefficients(coeffs, t, cubic_coeff);
    if(exclude_outside)
    {
        if(w0 < 0 || w0 >= input_width)
            coeffs[0] = 0;
        if(w1 < 0 || w1 >= input_width)
            coeffs[1] = 0;
        if(w2 < 0 || w2 >= input_width)
            coeffs[2] = 0;
        if(w3 < 0 || w3 >= input_width)
            coeffs[3] = 0;
        float sum = coeffs[0] + coeffs[1] + coeffs[2] + coeffs[3];
        sum == 0? sum = 1: 1.0 / sum;
        for(int i = 0; i < 4; i++)
        {
            coeffs[i] = coeffs[i] / sum;
        }
    }
    return cubic_interplote<T>(coeffs[0], x0, coeffs[1], x1, coeffs[2], x2, coeffs[3], x3);
}

static __device__ __forceinline__ float cubic_interp1d_float(
    float x0,
    float x1,
    float x2,
    float x3,
    float t,
    float cubic_coeff)
{
    float coeffs[4];
    get_cubic_resize_coefficients(coeffs, t, cubic_coeff);

    return cubic_interplote_float(coeffs[0], x0, coeffs[1], x1, coeffs[2], x2, coeffs[3], x3);
}

template <typename T>
__device__ __forceinline__ static T resize_get_value_bounded(
    const T* data,
    int height,
    int width,
    int access_c,
    int y,
    int x)
{
    int access_y = max(min(y, height - 1), 0);
    int access_x = max(min(x, width - 1), 0);
    return data[access_c * height * width + access_y * width + access_x];
}

template <typename T>
__device__ __forceinline__ static T resize_get_value_bounded_nhwc(
    const T* data,
    int height,
    int width,
    int channels,
    int access_c,
    int y,
    int x)
{
    int access_y = max(min(y, height - 1), 0);
    int access_x = max(min(x, width - 1), 0);
    return data[(access_y * width + access_x) * channels + access_c];
}

template<typename T>
__device__ inline T bilinear_interplote(float frac_w0, float frac_w1,
    float frac_h0, float frac_h1, T data0, T data1, T data2, T data3) {
    T res;
    res = frac_h0 * (frac_w0 * data0 + frac_w1 * data1) +
          frac_h1 * (frac_w0 * data2 + frac_w1 * data3);
    return res;
}

template<>
__device__ inline int8_t bilinear_interplote<int8_t>(float frac_w0, float frac_w1,
    float frac_h0, float frac_h1, int8_t data0, int8_t data1, int8_t data2, int8_t data3) {
    int8_t res;
    res = round(frac_h0 * (frac_w0 * data0 + frac_w1 * data1) +
          frac_h1 * (frac_w0 * data2 + frac_w1 * data3));
    return res;
}

template <>
__device__ inline half bilinear_interplote<half>(float frac_w0, float frac_w1, float frac_h0, float frac_h1, half data0, half data1, half data2, half data3)
{
    half res;
    res = frac_h0 * (frac_w0 * __half2float(data0) + frac_w1 * __half2float(data1)) +
          frac_h1 * (frac_w0 * __half2float(data2) + frac_w1 * __half2float(data3));
    return res;
}

template <>
__device__ inline half8_ bilinear_interplote<half8_>(float frac_w0, float frac_w1, float frac_h0, float frac_h1, half8_ data0, half8_ data1, half8_ data2, half8_ data3)
{
    half8_ res;
    res.x0 = frac_h0 * (frac_w0 * __half2float(data0.x0) + frac_w1 * __half2float(data1.x0)) +
             frac_h1 * (frac_w0 * __half2float(data2.x0) + frac_w1 * __half2float(data3.x0));
    res.y0 = frac_h0 * (frac_w0 * __half2float(data0.y0) + frac_w1 * __half2float(data1.y0)) +
             frac_h1 * (frac_w0 * __half2float(data2.y0) + frac_w1 * __half2float(data3.y0));
    res.z0 = frac_h0 * (frac_w0 * __half2float(data0.z0) + frac_w1 * __half2float(data1.z0)) +
             frac_h1 * (frac_w0 * __half2float(data2.z0) + frac_w1 * __half2float(data3.z0));
    res.w0 = frac_h0 * (frac_w0 * __half2float(data0.w0) + frac_w1 * __half2float(data1.w0)) +
             frac_h1 * (frac_w0 * __half2float(data2.w0) + frac_w1 * __half2float(data3.w0));
    res.x1 = frac_h0 * (frac_w0 * __half2float(data0.x1) + frac_w1 * __half2float(data1.x1)) +
             frac_h1 * (frac_w0 * __half2float(data2.x1) + frac_w1 * __half2float(data3.x1));
    res.y1 = frac_h0 * (frac_w0 * __half2float(data0.y1) + frac_w1 * __half2float(data1.y1)) +
             frac_h1 * (frac_w0 * __half2float(data2.y1) + frac_w1 * __half2float(data3.y1));
    res.z1 = frac_h0 * (frac_w0 * __half2float(data0.z1) + frac_w1 * __half2float(data1.z1)) +
             frac_h1 * (frac_w0 * __half2float(data2.z1) + frac_w1 * __half2float(data3.z1));
    res.w1 = frac_h0 * (frac_w0 * __half2float(data0.w1) + frac_w1 * __half2float(data1.w1)) +
             frac_h1 * (frac_w0 * __half2float(data2.w1) + frac_w1 * __half2float(data3.w1));
    return res;
}

#ifdef OPT_RESIZE
#define MIN(a,b) ((a) < (b) ? (a):(b))
typedef __NATIVE_VECTOR__(2, float) v2f;
static float hostComputeSourceIndex(
    float scale,
    int dstIndex,
    int transform_mode)
{
    float srcIdx = 0.f;
    if (transform_mode == 0 || transform_mode == 1) {
        srcIdx = scale * (dstIndex + 0.5) - 0.5;
    } else if (transform_mode == 2 || transform_mode == 3) {
        srcIdx = scale * dstIndex;
    } else {
        srcIdx = scale * (dstIndex + 0.5);
    }
    return (srcIdx < 0) ? 0.f : srcIdx;
}

template <typename T>
__global__ void ppl_cukernel_resize_bilinear_int8_opt(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale,
    DivModFast out_width_fast,
    int transform_mode
)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < num_threads)
    {
        int h2, w2;
        out_width_fast.divmod(index, h2, w2);
        const float h1r = cudaComputeSourceIndex(h_scale, h2, transform_mode);

        const int h1         = h1r;
        const int h1p        = (h1 < in_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = 1.f - h1lambda;
        const float w1r      = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1         = w1r;
        const int w1p        = (w1 < in_width - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = 1.f - w1lambda;
        const float hw00lambda = h0lambda*w0lambda;
        const float hw01lambda = h0lambda*w1lambda;
        const float hw10lambda = h1lambda*w0lambda;
        const float hw11lambda = h1lambda*w1lambda;
        int imageSize = in_height*in_width;
        int outImageSize = out_height*out_width;
        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = input + start_c * imageSize + max(h1,0)*in_width + max(w1,0);
        T* pos2 = output + start_c * outImageSize + h2 * out_width + w2;
        const T* pos1_next_line = pos1 + h1p*in_width;

        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) { //右边一个和下边一个
            // pos2[0] = h0lambda * (w0lambda * pos1[0] +
            // w1lambda * pos1[w1p]) +
            // h1lambda * (w0lambda * pos1[h1p * in_width] +
            // w1lambda * pos1[h1p * in_width + w1p]);
            T src00 = pos1[0];
            T src01 = pos1[w1p];
            T src10 = pos1_next_line[0];
            T src11 = pos1_next_line[w1p];
            int32_t temp = round(hw00lambda*src00 + hw01lambda*src01 + hw10lambda*src10 + hw11lambda*src11);
            temp = round(temp * in_scale * __builtin_mxc_rcpf(out_scale));
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            pos2[0] = temp;
            pos1 += imageSize;
            pos1_next_line += imageSize;
            pos2 += outImageSize;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_int8_opt(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    int inSize,
    T* output,
    int out_height,
    int out_width,
    int outSize,
    float div_scale,
    DivModFast out_width_fast,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        // const int w2 = index % out_width; // 0:out_width-1
        // const int h2 = index / out_width; // 0:out_height-1
        int h2, w2;
        // h2 = index*r_out_width;
        // w2 = index - index*r_out_width;
        out_width_fast.divmod(index, h2, w2);
        //const float h1r = h_scale * h2;
        const float h1r = cudaComputeSourceIndex(h_scale, h2, transform_mode);
        const int h1    = h1r;

        //const float w1r = w_scale * w2;
        const float w1r = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1    = w1r;

        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = &input[start_c * inSize + h1 * in_width + w1];
        T* pos2       = &output[start_c * outSize + h2 * out_width + w2];
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            T input = pos1[0];
            int32_t temp = round(input * div_scale);
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            pos2[0] = temp;
            pos1 += inSize;
            pos2 += outSize;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_round_prefer_floor_int8_opt(
    int num_threads,
    double h_scale,
    double w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    int inSize,
    T* output,
    int out_height,
    int out_width,
    int outSize,
    DivModFast out_width_fast,
    float scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        // const int w2 = index % out_width; // 0:out_width-1
        // const int h2 = index / out_width; // 0:out_height-1
        //const float h1r = h_scale * h2;
        int h2, w2;
        out_width_fast.divmod(index, h2, w2);
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1rfloor = floor(h1r);
        const int h1    = h1r - h1rfloor <= 0.5 ? h1rfloor: h1rfloor + 1;

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1rfloor = floor(w1r);
        const int w1    = w1r - w1rfloor <= 0.5 ? w1rfloor: w1rfloor + 1;

        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = &input[start_c * inSize + min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)];
        T* pos2       = &output[start_c * outSize + h2 * out_width + w2];
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            int32_t temp = round(pos1[0] * scale);
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            pos2[0] = temp;
            pos1 += inSize;
            pos2 += outSize;
        }
    }
}

template<typename T,int N,int ShiftN>
__global__ void ppl_cukernel_resize_nearest_int8_opt_2x(
    const int8_t* input,
    int in_height,
    int in_width,
    int8_t* output,
    int out_widthN,
    int block_line,
    float div_scale,
    DivModFast out_widthN_fast)
{
    int Height_Offset = blockIdx.x * block_line;
    int offset = Height_Offset * in_width;
    const int8_t * ptr_input = input + offset;
    int8_t * ptr_output = output + (offset<<2);
    __shared__ union {
        int8_t m1[4096];
        T m2[4096 >> ShiftN];
    }sm_input;

    int block_real_line = min(block_line, in_height - Height_Offset);
    if(block_real_line <= 0) return;
    int blockSize = block_real_line * in_width;
    int blockSizeN = (blockSize + (1 << ShiftN) - 1) >> ShiftN;
    for(int id = threadIdx.x; id < blockSizeN; id += blockDim.x){
        sm_input.m2[id] = *((T*)ptr_input + id);
    }
    __syncthreads();
    
    int dstBlockSizeN = (block_real_line << 1) * out_widthN;
    for(int id = threadIdx.x; id < dstBlockSizeN; id += blockDim.x){
        int h, w;
        out_widthN_fast.divmod(id, h, w);
        int h_offset = h * out_widthN;
        int dst_index = h_offset + w;
        int src_index = (h >> 1) * in_width;
        int w_shift = w << ShiftN;
        T reg_dst;
        int8_t*ptr_reg_dst = (int8_t*)&reg_dst;
        #pragma unroll N
        for(int k = 0; k < N; k++){
            int32_t temp = round(sm_input.m1[src_index + ((w_shift + k)>>1)] * div_scale);
            temp = min(temp,127);
            temp = max(temp,-128);
            ptr_reg_dst[k] = temp;
        }
        *((T*)ptr_output + dst_index) = reg_dst;
    }
}

__global__ void ppl_cukernel_resize_nearest_int8_opt_4x(
    const int8_t* input,
    int in_height,
    int in_width,
    int8_t* output,
    int out_widthN,
    int block_line,
    float div_scale,
    DivModFast out_widthN_fast)
{
    int Height_Offset = blockIdx.x * block_line;
    int offset = Height_Offset * in_width;
    const int8_t * ptr_input = input + offset;
    int8_t * ptr_output = output + (offset << 4);
    __shared__ union {
        int8_t m1[3276];
        int32_t m2[819];
    }sm_input;

    int block_real_line = min(block_line, in_height - Height_Offset);
    int blockSize = block_real_line * in_width;
   
    int blockSizeN = (blockSize + 3) >> 2;
    for(int id = threadIdx.x; id < blockSizeN; id += blockDim.x){
        sm_input.m2[id] = *((int32_t*)ptr_input + id);
    }
    __syncthreads();
    
    int dstBlockSizeN = blockSize << 2;
    for(int id = threadIdx.x; id < dstBlockSizeN; id += blockDim.x){
        int h, w;
        out_widthN_fast.divmod(id, h, w);
        int h_offset = h * out_widthN;
        int dst_index = h_offset + w;
        int src_index = (h >> 2) * in_width;
        int32_t reg_dst;
        int8_t* ptr_reg_dst = (int8_t*)&reg_dst;
        #pragma unroll 4
        for(int k = 0; k < 4; k++){
            int32_t temp = round(sm_input.m1[src_index + w] * div_scale);
            temp = min(temp,127);
            temp = max(temp,-128);
            ptr_reg_dst[k] = temp;
        }
        *((int32_t*)ptr_output + dst_index) = reg_dst;
    }
}

__global__ void ppl_cukernel_resize_nearest_int8_opt_8x(
    const int8_t* input,
    int in_height,
    int in_width,
    int8_t* output,
    int out_widthN,
    int block_line,
    float div_scale,
    DivModFast out_widthN_fast)
{
    int Height_Offset = blockIdx.x * block_line;
    int offset = Height_Offset * in_width;
    const int8_t * ptr_input = input + offset;
    int8_t * ptr_output = output + (offset << 6);
    __shared__ union {
        int8_t m1[1820];
        int32_t m2[455];
    }sm_input;

    int block_real_line = min(block_line, in_height - Height_Offset);
    int blockSize = block_real_line * in_width;
   
    int blockSizeN = (blockSize + 3) >> 2;
    for(int id = threadIdx.x; id < blockSizeN; id += blockDim.x){
        sm_input.m2[id] = *((int32_t*)ptr_input + id);
    }
    __syncthreads();
    
    int dstBlockSizeN = (block_real_line << 3) * out_widthN;
    for(int id = threadIdx.x; id < dstBlockSizeN; id += blockDim.x){
        int h, w;
        out_widthN_fast.divmod(id, h, w);
        int h_offset = h * out_widthN;
        int dst_index = h_offset + w;
        int src_index = (h >> 3) * in_width;
        int w_shift = w << 2;
        int32_t reg_dst;
        int8_t* ptr_reg_dst = (int8_t*)&reg_dst;
        #pragma unroll 4
        for(int k = 0; k < 4; k++){
            int32_t temp = round(sm_input.m1[src_index + ((w_shift + k)>>3)] * div_scale);
            temp = min(temp,127);
            temp = max(temp,-128);
            ptr_reg_dst[k] = temp;
        }
        *((int32_t*)ptr_output + dst_index) = reg_dst;
    }
}

__global__ void ppl_cukernel_resize_nearest_nhwc_int8_opt_t3(
    int num_elements,
    int h_scale,
    int w_scale,
    int channels,
    const float4* input,
    int in_height,
    int in_width,
    float4* output,
    int out_height,
    int out_width,
    float scale,
    DivModFast channels_fast,
    DivModFast in_width_fast)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= num_elements) return;

    int bidx = blockIdx.z;
    int iSize = in_height * in_width;
    int oSize = out_height * out_width;
    int iChannelSize = iSize * channels;
    int oChannelSize = oSize * channels;
    int oCW = channels * out_width;
    const float4* ptr_block_input = input + bidx * iChannelSize;
    int hw_idx, c_idx;
    channels_fast.divmod(index, hw_idx, c_idx);
    int ih, iw;
    in_width_fast.divmod(hw_idx, ih, iw);
    int out_h_s = ih * h_scale;
    int out_w_s = iw * w_scale;
    float4* ptr_block_output = output + bidx * oChannelSize + (out_h_s * out_width + out_w_s) * channels + c_idx;
    float4 reg_input = *(ptr_block_input + index);
    int8_t *ptr_reg = (int8_t*)&reg_input;
    #pragma unroll 16
    for(int k = 0; k < 16; k++){
        int32_t res = round(ptr_reg[k] * scale);
        res = min(res,127);
        res = max(res,-128);
        ptr_reg[k] = res;
    }

    for(int i = 0; i < h_scale; i++){
        for(int j = 0; j < w_scale; j++){
            if(out_h_s + i < out_height && out_w_s + j < out_width){
                *(ptr_block_output + (i*out_width + j)*channels) = reg_input;
            }
        }
    }
}

// __global__ void ppl_cukernel_resize_nearest_nhwc_opt_t3(
//     int num_elements,
//     int h_scale,
//     int w_scale,
//     int channels,
//     const float4* input,
//     int in_height,
//     int in_width,
//     float4* output,
//     int out_height,
//     int out_width,
//     DivModFast channels_fast,
//     DivModFast in_width_fast)
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if(index >= num_elements) return;

//     int bidx = blockIdx.z;
//     int iSize = in_height * in_width;
//     int oSize = out_height * out_width;
//     int iChannelSize = iSize * channels;
//     int oChannelSize = oSize * channels;
//     int oCW = channels * out_width;
//     const float4* ptr_block_input = input + bidx * iChannelSize;
//     int hw_idx, c_idx;
//     channels_fast.divmod(index, hw_idx, c_idx);
//     int ih, iw;
//     in_width_fast.divmod(hw_idx, ih, iw);
//     int out_h_s = ih * h_scale;
//     int out_w_s = iw * w_scale;
//     float4* ptr_block_output = output + bidx * oChannelSize + (out_h_s * out_width + out_w_s) * channels + c_idx;
//     float4 reg_input = *(ptr_block_input + index);

//     for(int i = 0; i < h_scale; i++){
//         for(int j = 0; j < w_scale; j++){
//             if(out_h_s + i < out_height && out_w_s + j < out_width){
//                 *(ptr_block_output + (i*out_width + j)*channels) = reg_input;
//             }
//         }
//     }
// }

template <typename T, typename T1, int shiftN>
__global__ void ppl_cukernel_resize_nearest_round_prefer_floor_nhwc_opt(
    int num_elems,
    float h_scale,
    float w_scale,
    int pad_channels,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode,
    DivModFast pad_channels_fast,
    DivModFast out_width_fast,
    DivModFast hw_fast
    )
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x) << shiftN;
    if(id >= num_elems) return;
    int nhw_id, hw_id, n, cid, h, w;
    pad_channels_fast.divmod(id, nhw_id, cid);
    hw_fast.divmod(nhw_id, n, hw_id);
    out_width_fast.divmod(hw_id, h, w);
    int inSize = in_height * in_width;
    int oSize = out_height * out_width;
    const T* ptr_block_input = input + n * inSize * pad_channels;
    T* ptr_block_output = output + id;
    
    const double h1r = cudaComputeSourceSTDIndex(h_scale, h, transform_mode);
    const int h1rfloor = floor(h1r);
    const int h1    = h1r - h1rfloor <= 0.5 ? h1rfloor: h1rfloor + 1;

    //const float w1r = w_scale * w2;
    const double w1r = cudaComputeSourceSTDIndex(w_scale, w, transform_mode);
    const int w1rfloor = floor(w1r);
    const int w1    = w1r - w1rfloor <= 0.5 ? w1rfloor: w1rfloor + 1;
    ptr_block_input += (min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)) * pad_channels + cid;
    T1 src;
    src = *(T1*)ptr_block_input;
    *(T1*)ptr_block_output = src;
}

static int condition(int in_height, int out_height, int in_width, int out_width, float h_scale, float w_scale){
    if((in_height << 1) == out_height && (in_width << 1) == out_width && abs(h_scale - 0.5) < 0.00000001 && abs(w_scale - 0.5) < 0.00000001){
        return 1;
    }else if((in_height<<2) == out_height && (in_width << 2) == out_width && abs(h_scale - 0.25) < 0.00000001 && abs(w_scale - 0.25) < 0.00000001){
        return 2;
    }else if((in_height<<3) == out_height && (in_width << 3) == out_width && abs(h_scale - 0.125) < 0.00000001 && abs(w_scale - 0.125) < 0.00000001){
        return 3;
    }
    return 0;
}
static int condition_nhwc(int in_height, int out_height, int in_width, int out_width, float h_scale, float w_scale){
    if(int((out_height + in_height - 1)/ in_height) == 2 && int((out_width + in_width - 1)/ in_width) == 2){
        return 1;
    }else if((in_height<<2) == out_height && (in_width << 2) == out_width && abs(h_scale - 0.25) < 0.00000001 && abs(w_scale - 0.25) < 0.00000001){
        return 2;
    }else if((in_height<<3) == out_height && (in_width << 3) == out_width && abs(h_scale - 0.125) < 0.00000001 && abs(w_scale - 0.125) < 0.00000001){
        return 3;
    }
    return 0;
}

template<typename T,int shiftN,int N>
__global__ void ppl_cukernel_resize_bilinear_nhwc_int8_opt(
    int num_elems,
    float h_scale,
    float w_scale,
    int pad_channels,
    const int8_t* input,
    int in_height,
    int in_width,
    int8_t* output,
    int out_height,
    int out_width,
    float scale,
    int transform_mode,
    DivModFast pad_channels_fast,
    DivModFast out_width_fast,
    DivModFast hw_fast
)
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x) << shiftN;
    if(id >= num_elems) return;
    int nhw_id, hw_id, n, cid, h, w;
    pad_channels_fast.divmod(id, nhw_id, cid);
    hw_fast.divmod(nhw_id, n, hw_id);
    out_width_fast.divmod(hw_id, h, w);

    int inSize = in_height * in_width;
    int oSize = out_height * out_width;
    const int8_t* ptr_block_input = input + n * inSize * pad_channels;
    int8_t* ptr_block_output = output + id;

    float h1r = cudaComputeSourceIndex(h_scale, h, transform_mode);
    int h1 = h1r;
    int h1p = (h1 < in_height - 1) ? 1 : 0;
    float w1r = cudaComputeSourceIndex(w_scale, w, transform_mode);
    int w1 = w1r;
    int w1p = (w1 < in_width - 1) ? 1 : 0;
    float w1lambda = w1r - w1;
    float h1lambda = h1r - h1;
    float w0lambda = 1.f - w1lambda;
    float h0lambda = 1.f - h1lambda;
    ptr_block_input += (max(h1,0) * in_width + max(w1,0)) * pad_channels + cid;
    const int8_t* pos00 = ptr_block_input;
    const int8_t* pos01 = ptr_block_input + w1p * pad_channels;
    const int8_t* pos10 = pos00 + h1p*in_width * pad_channels;
    const int8_t* pos11 = pos10 + w1p * pad_channels;
    T dst;
    T src00, src01, src10,src11;
    src00 = *(T*)pos00;
    
    if(w1p) {
        src01 = *(T*)pos01;
    }

    if(h1p) {
        src10 = *(T*)pos10;
        if(w1p){
            src11 = *(T*)pos11;
        }
    }

    int8_t *ptr_src00 = (int8_t*)&src00;
    int8_t *ptr_src01 = (int8_t*)&src01;
    int8_t *ptr_src10 = (int8_t*)&src10;
    int8_t *ptr_src11 = (int8_t*)&src11;
    int8_t *ptr_dst = (int8_t*)&dst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        int32_t temp = 0;
        if(w1p && h1p){
            temp = bilinear_interplote<int8_t>(w0lambda, w1lambda, h0lambda, h1lambda,
                ptr_src00[i], ptr_src01[i], ptr_src10[i], ptr_src11[i]);
        } else if (w1p) {
            temp = w0lambda * ptr_src00[i] + w1lambda * ptr_src01[i];
        } else if(h1p){
            temp = h0lambda * ptr_src00[i] + h1lambda * ptr_src10[i];
        } else {
            temp = ptr_src00[i];
        }
        
        temp = round(temp * scale);
        temp = min(temp,127);
        temp = max(temp,-128);
        ptr_dst[i] = temp;
    }
    *(T*)ptr_block_output = dst;
}

__global__ void ppl_cukernel_resize_bilinear_nhwc_int8_sm_opt(
    float h_scale,
    float w_scale,
    int pad_channels,
    const int8_t* input,
    int in_height,
    int in_width,
    int8_t* output,
    int out_height,
    int out_width,
    int iTileHeight,
    int iTileWidth,
    float scale,
    int transform_mode,
    DivModFast blocky_fast)
{
    int blockChannel = blockDim.x << 4;
    int blockWidth = blockDim.y;
    int blockHeight = blockDim.z;
    int blockWC = blockChannel * blockWidth;
    int cid = (threadIdx.x + blockIdx.x * blockDim.x) << 4;
    int by, n;
    int idx = threadIdx.y + blockIdx.y * blockDim.y;
    blocky_fast.divmod(blockIdx.z, n, by);
    int start_y = by * blockHeight;
    int idy = start_y + threadIdx.z;
    if(idx >= out_width || idy >= out_height || cid >= pad_channels) return;

    int iSize = in_height * in_width;
    int oSize = out_height * out_width;
    int8_t * ptr_block_output = output + n * oSize * pad_channels + (idy * out_width + idx) * pad_channels + cid;
    const int8_t* ptr_block_input = input + n * iSize * pad_channels + cid;
    float h1r = cudaComputeSourceIndex(h_scale, start_y, transform_mode);
    int h1_s = h1r;
    float w1r = cudaComputeSourceIndex(w_scale, idx, transform_mode);
    int w1 = w1r;
    int w1p = (w1 < in_width - 1) ? 1:0;
    float w1lambda = w1r - w1;
    float w0lambda = 1.f - w1lambda;
    __shared__ float sm_buffer[4096];
    
    int iValidHeight = min(iTileHeight, in_height - h1_s);
    if(threadIdx.z < iValidHeight) {
        float* ptr_sm = sm_buffer + threadIdx.z * blockWC + threadIdx.y * blockChannel + (threadIdx.x << 4);
        int offset = (h1_s + threadIdx.z)*in_width + w1;
        if(w1p){
            float4 reg_buffer0, reg_buffer1;
            int8_t* buffer0 = (int8_t*)&reg_buffer0;
            int8_t* buffer1 = (int8_t*)&reg_buffer1;
            reg_buffer0 = *(float4*)(ptr_block_input + offset * pad_channels);
            reg_buffer1 = *(float4*)(ptr_block_input + (offset + w1p) * pad_channels);
            #pragma unroll 16
            for(int i = 0; i < 16; i++) {
                ptr_sm[i] = w0lambda * buffer0[i] + w1lambda * buffer1[i];
            }
        } else {
            float4 reg_buffer, reg_buffer1;
            int8_t* buffer = (int8_t*)&reg_buffer;
            reg_buffer = *(float4*)(ptr_block_input + offset * pad_channels);
            #pragma unroll 16
            for(int i = 0; i < 16; i++) {
                ptr_sm[i] = buffer[i];
            }
        }
    }   
    __syncthreads();
    
    float h2r = cudaComputeSourceIndex(h_scale, idy, transform_mode);
    int h2 = h2r;
    int h2p = (h2 < in_height - 1) ? 1 : 0;
    float h1lambda = (h2r - h2)*scale;
    float h0lambda = (1.f - h1lambda)*scale;
    float4 reg_dst;
    int8_t* dst = (int8_t*)&reg_dst;
    float * ptr_sm = sm_buffer + (h2 - h1_s) * blockWC + threadIdx.y * blockChannel + (threadIdx.x << 4);
    if(h2p) {
        float* ptr_next_sm = ptr_sm + blockWC;
        #pragma unroll 16
        for(int i = 0; i < 16; i++) {
            float reg_f;
            int32_t temp = 0;
            reg_f = h0lambda * ptr_sm[i] + h1lambda * ptr_next_sm[i];
            temp = round(reg_f);
            temp = min(temp,127);
            temp = max(temp,-128);
            dst[i] = temp;
        }
    } else {
        #pragma unroll 16
        for(int i = 0; i < 16; i++) {
            float reg_f;
            int reg_s;
            reg_f = ptr_sm[i];
            reg_s = round(reg_f * scale);
            reg_s = min(reg_s,127);
            reg_s = max(reg_s,-128);
            dst[i] = reg_s;
        }
    }
    *(float4*)ptr_block_output = reg_dst;
}

template <typename T, int shiftN, int N>
__global__ void ppl_cukernel_resize_nearest_round_prefer_floor_nhwc_int8_opt(
    int num_elems,
    double h_scale,
    double w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const int8_t* input,
    int in_height,
    int in_width,
    int8_t* output,
    int out_height,
    int out_width,
    float scale,
    int transform_mode,
    DivModFast pad_channels_fast,
    DivModFast out_width_fast,
    DivModFast hw_fast
    )
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x) << shiftN;
    if(id >= num_elems) return;
    int nhw_id, hw_id, n, cid, h, w;
    pad_channels_fast.divmod(id, nhw_id, cid);
    hw_fast.divmod(nhw_id, n, hw_id);
    out_width_fast.divmod(hw_id, h, w);
    int inSize = in_height * in_width;
    int oSize = out_height * out_width;
    const int8_t* ptr_block_input = input + n * inSize * pad_channels;
    int8_t* ptr_block_output = output + id;
    const double h1r = cudaComputeSourceSTDIndex(h_scale, h, transform_mode);
    const int h1rfloor = floor(h1r);
    const int h1    = h1r - h1rfloor <= 0.5 ? h1rfloor: h1rfloor + 1;
    const double w1r = cudaComputeSourceSTDIndex(w_scale, w, transform_mode);
    const int w1rfloor = floor(w1r);
    const int w1    = w1r - w1rfloor <= 0.5 ? w1rfloor: w1rfloor + 1;
    ptr_block_input += (min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)) * pad_channels + cid;
    T dst, src;
    src = *(T*)ptr_block_input;
    int8_t* ptr_src = (int8_t*)&src;
    int8_t* ptr_dst = (int8_t*)&dst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        int32_t temp = round(ptr_src[i] * scale);
        temp = min(temp,127);
        temp = max(temp,-128);
        ptr_dst[i] = temp;
    }
    *(T*)ptr_block_output = dst;
}

template <typename T, int shiftN, int N>
__global__ void ppl_cukernel_resize_nearest_nhwc_int8_opt(
    int num_elems,
    float h_scale,
    float w_scale,
    int pad_channels,
    const int8_t* input,
    int in_height,
    int in_width,
    int8_t* output,
    int out_height,
    int out_width,
    float scale,
    int transform_mode,
    DivModFast pad_channels_fast,
    DivModFast out_width_fast,
    DivModFast hw_fast
    )
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x) << shiftN;
    if(id >= num_elems) return;
    int nhw_id, hw_id, n, cid, h, w;
    pad_channels_fast.divmod(id, nhw_id, cid);
    hw_fast.divmod(nhw_id, n, hw_id);
    out_width_fast.divmod(hw_id, h, w);
    int inSize = in_height * in_width;
    int oSize = out_height * out_width;
    const int8_t* ptr_block_input = input + n * inSize * pad_channels;
    int8_t* ptr_block_output = output + id;
    const float h1r = cudaComputeSourceIndex(h_scale, h, transform_mode);
    const int h1    = h1r;
    const float w1r = cudaComputeSourceIndex(w_scale, w, transform_mode);
    const int w1    = w1r;
    ptr_block_input += (h1 * in_width + w1) * pad_channels + cid;
    T dst, src;
    src = *(T*)ptr_block_input;
    int8_t* ptr_src = (int8_t*)&src;
    int8_t* ptr_dst = (int8_t*)&dst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        int32_t temp = round(ptr_src[i] * scale);
        temp = min(temp,127);
        temp = max(temp,-128);
        ptr_dst[i] = temp;
    }
    *(T*)ptr_block_output = dst;
}

template <typename T, typename T1, int shiftN>
__global__ void ppl_cukernel_resize_nearest_nhwc_opt(
    int num_elems,
    float h_scale,
    float w_scale,
    int pad_channels,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode,
    DivModFast pad_channels_fast,
    DivModFast out_width_fast,
    DivModFast hw_fast
    )
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x) << shiftN;
    if(id >= num_elems) return;
    int nhw_id, hw_id, n, cid, h, w;
    pad_channels_fast.divmod(id, nhw_id, cid);
    hw_fast.divmod(nhw_id, n, hw_id);
    out_width_fast.divmod(hw_id, h, w);
    int inSize = in_height * in_width;
    int oSize = out_height * out_width;
    const T* ptr_block_input = input + n * inSize * pad_channels;
    T* ptr_block_output = output + id;
    const float h1r = cudaComputeSourceIndex(h_scale, h, transform_mode);
    const int h1    = h1r;
    const float w1r = cudaComputeSourceIndex(w_scale, w, transform_mode);
    const int w1    = w1r;
    ptr_block_input += (h1 * in_width + w1) * pad_channels + cid;
    T1 src;
    src = *(T1*)ptr_block_input;
    *(T1*)ptr_block_output = src;
}

//out_height:nxcxh,out_height_fast:h, T origin type shiftN1 = sizeof(float4) / sizeof(T)
template<typename T, int shiftN, int shiftN1>
__global__ void ppl_cukernel_resize_nearest_opt_t3(
    const T* input,
    float h_scale,
    float w_scale,
    int in_height,
    int in_width,
    T* output,
    int out_width,
    int block_line,
    int total_height,
    DivModFast out_height_fast,
    DivModFast out_width_fast)
{
    int start_h = blockIdx.x * block_line;
    int out_block_height = min(block_line, total_height - start_h);
    int end_h = start_h + out_block_height - 1;
    if(out_block_height <= 0) return;
    int out_h_start, nc_start, out_h_end, nc_end;
    out_height_fast.divmod(start_h, nc_start, out_h_start);
    out_height_fast.divmod(end_h, nc_end, out_h_end);
    int ih_start = out_h_start * h_scale;
    int in_h_start = nc_start * in_height + ih_start;
    int ih_end = out_h_end * h_scale;
    int in_h_end = nc_end * in_height + ih_end;
    int out_block_size = out_block_height * out_width;
    int in_block_size = (in_h_end - in_h_start + 1) * in_width;
    int in_offset = in_h_start * in_width;
    int in_offset_align = in_offset & 0xFFFFFFF0;
    int offset0 = in_offset - in_offset_align;
    in_block_size += offset0;
    in_block_size = (in_block_size + (1 << shiftN1) - 1) >> shiftN1;
    const T *ptr_block_input = (const T*)input + in_offset_align;
    int out_offset = start_h * out_width;
    T *ptr_block_output = output + out_offset;
    __shared__ union {
        int8_t m1[8192];
        float4 m2[512];
        T m3[8192 >> shiftN];
    }sm_input,sm_output;

    for(int i = threadIdx.x; i < in_block_size; i += blockDim.x) {
        sm_input.m2[i] = *((float4 *)ptr_block_input  + i);
    }
    __syncthreads();

    T *ptr_sm_input = sm_input.m3 + offset0;
    
    for(int i = threadIdx.x; i < out_block_size; i += blockDim.x) {
        int nchid,w_id;
        out_width_fast.divmod(out_offset + i, nchid, w_id);
        int nc, out_h;
        out_height_fast.divmod(nchid,nc,out_h);
        int in_h = out_h * h_scale;
        int local_in_w = w_id * w_scale;
        int local_in_h = nc*in_height + in_h - in_h_start;
        T tmp;
        tmp = ptr_sm_input[local_in_h* in_width + local_in_w];
        sm_output.m3[i] = tmp;
    }
    __syncthreads();
    int out_block_sizeN1 = out_block_size >> shiftN1;
    for(int i = threadIdx.x; i < out_block_sizeN1; i += blockDim.x){
        *((float4*)ptr_block_output + i) = sm_output.m2[i];
    }
}

template<typename T, typename T1, int N, int shiftN>
__global__  void ppl_cukernel_resize_bilinear_nhwc_opt(
    int num_elems,
    float h_scale,
    float w_scale,
    float roi_0,
    float roi_1,
    float roi_2,
    float roi_3,
    float extrapolation_value,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode,
    DivModFast pad_channels_fast,
    DivModFast out_width_fast,
    DivModFast hw_fast
)
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x) << shiftN;
    if(id >= num_elems) return;
    int nhw_id, hw_id, n, cid, h, w;
    pad_channels_fast.divmod(id, nhw_id, cid);
    hw_fast.divmod(nhw_id, n, hw_id);
    out_width_fast.divmod(hw_id, h, w);

    int inSize = in_height * in_width;
    int oSize = out_height * out_width;
    const T* ptr_block_input = input + n * inSize * pad_channels;
    T* ptr_block_output = output + id;

    const float h1r = cudaComputeBilinearSourceIndex(h_scale, roi_2, roi_3, h, out_height, in_height, transform_mode);
    const int h1    = floor(h1r);
    const int h1p   = h1 < 0 ? 0: (h1 < in_height - 1) ? 1 : 0;
    const float w1r = cudaComputeBilinearSourceIndex(w_scale,roi_0,roi_1, w, out_width, in_width, transform_mode);
    const int w1    = floor(w1r);
    const int w1p   = w1 < 0 ? 0 : (w1 < in_width - 1) ? 1 : 0;
    float w1lambda = w1r - w1;
    float h1lambda = h1r - h1;
    float w0lambda = 1.f - w1lambda;
    float h0lambda = 1.f - h1lambda;
    ptr_block_input += (max(h1,0) * in_width + max(w1,0)) * pad_channels + cid;
    const T* pos00 = ptr_block_input;
    const T* pos01 = ptr_block_input + w1p * pad_channels;
    const T* pos10 = pos00 + h1p*in_width * pad_channels;
    const T* pos11 = pos10 + w1p * pad_channels;
    T1 dst;
    T1 src00, src01, src10,src11;
    src00 = *(T1*)pos00;
    
    if(w1p) {
        src01 = *(T1*)pos01;
    }

    if(h1p) {
        src10 = *(T1*)pos10;
        if(w1p){
            src11 = *(T1*)pos11;
        }
    }
    if(transform_mode == 5) {
        if(h1r < 0 || h1r >= in_height - 1 || w1r < 0 || w1r >= in_width - 1){
            T* ptr_value = (T*)&dst;
            #pragma unroll N
            for(int i = 0; i < N; i++) {
                ptr_value[i] = extrapolation_value;
            }
            *(T1*)ptr_block_output = dst;
            return;    
        }
    }
    T *ptr_src00 = (T*)&src00;
    T *ptr_src01 = (T*)&src01;
    T *ptr_src10 = (T*)&src10;
    T *ptr_src11 = (T*)&src11;
    T *ptr_dst = (T*)&dst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        T temp = 0;
        if(w1p && h1p){
            temp = bilinear_interplote<T>(w0lambda, w1lambda, h0lambda, h1lambda,
                ptr_src00[i], ptr_src01[i], ptr_src10[i], ptr_src11[i]);
        } else if (w1p) {
            temp = w0lambda * (float)(ptr_src00[i]) + w1lambda * (float)(ptr_src01[i]);
        } else if(h1p){
            temp = h0lambda * (float)(ptr_src00[i]) + h1lambda * (float)(ptr_src10[i]);
        } else {
            temp = ptr_src00[i];
        }
        
        ptr_dst[i] = temp;
    }
    *(T1*)ptr_block_output = dst;
}

// template<typename ELE_T,typename dst_T, int dst_N, int dst_shiftN>
// __global__ void ppl_cukernel_resize_bilinear_opt_line_t1(
//     const ELE_T* input,
//     ELE_T* output,
//     float h_scale,
//     float w_scale,
//     float roi_0,
//     float roi_1,
//     float roi_2,
//     float roi_3,
//     float extrapolation_value,
//     int in_height,
//     int in_hc,
//     int in_width,
//     int out_height,
//     int out_hc,
//     int out_width,
//     int block_line,
//     int transform_mode,
//     DivModFast out_width_fast,
//     DivModFast out_height_fast
// )
// {
//     int inSize = in_hc * in_width;
//     int outSize = out_hc * out_width;
//     int out_h_start = blockIdx.x * block_line;
//     int out_block_height = min(block_line, out_hc - out_h_start);
//     int out_h_end = out_h_start + out_block_height - 1;
//     if(out_block_height <= 0) return;
//     int oh_s, ow_s, oc_s;
//     int oh_e, ow_e, oc_e;
//     out_height_fast.divmod(out_h_start, oc_s, oh_s);
//     out_height_fast.divmod(out_h_end, oc_e, oh_e);

//     // const float h1_f = cudaComputeBilinearSourceIndex(h_scale,roi_2,roi_3, oh_s, out_height, in_height, transform_mode);
//     const float h1_f = max(h_scale * (oh_s + 0.5) - 0.5,0);
//     const int in_h_start = floor(h1_f);
//     // const float h2_f = cudaComputeBilinearSourceIndex(h_scale,roi_2,roi_3, oh_e, out_height, in_height, transform_mode);
//     const float h2_f = max(h_scale *(oh_e + 0.5) - 0.5, 0);
//     const int in_h_end = floor(h2_f);
//     int h3 = min(in_h_end + 1, in_height - 1);
//     int in_channel_height = (oc_e - oc_s) * in_height + h3 - in_h_start + 1;
//     unsigned int in_offset = blockIdx.y * inSize + (oc_s * in_height + in_h_start) * in_width;
    
//     int out_block_offset = out_h_start * out_width;
//     const ELE_T * ptr_block_input = input + in_offset;

//     ELE_T *ptr_block_output = output + blockIdx.y * outSize + out_block_offset;
//     __shared__ half sum_buffer[8192];
//     int out_block_size = out_block_height * out_width;
//     int buffer_size = in_channel_height * out_width;
    
//     for(int i = threadIdx.x; i < buffer_size; i += blockDim.x)
//     {
//         int local_h, local_w;
//         out_width_fast.divmod(i, local_h, local_w);
//         float w1r  = max(w_scale * (local_w + 0.5) - 0.5, 0);
//         int w1 = floor(w1r);
//         const float w1lambda = w1r - w1;
//         float src0 = ptr_block_input[local_h * in_width + w1];
//         int w1n = min(in_width - 1, w1 + 1);
//         float src1 = ptr_block_input[local_h * in_width + w1n];
//         float temp0 = src0 + w1lambda * (src1 - src0);
//         sum_buffer[i] = temp0;
//     }
//     __syncthreads();
//     int out_block_sizeN = out_block_size >> dst_shiftN;
//     for(int i = threadIdx.x; i < out_block_sizeN; i += blockDim.x){
//         int t, h, w, c;
//         int d_offset = i << dst_shiftN;
//         out_width_fast.divmod(d_offset + out_block_offset, t, w);
//         out_height_fast.divmod(t, c, h);
//         const float h1r = max(h_scale * (h + 0.5) - 0.5,0);
//         const int h1 = floor(h1r);
//         float h1lambda = h1r - h1;
//         union {
//             ELE_T m1[dst_N];
//             dst_T m2;
//         } temp_storage;
//         int c_offset = (c - oc_s) * in_height;
//         int local_h = c_offset + h1 - in_h_start;
//         int next_h = c_offset + min(h1 + 1,in_height - 1) - in_h_start;
//         int first_offset = local_h * out_width;
//         int next_offset = next_h * out_width;
//         half* ptr_line_0 = sum_buffer + first_offset + w;
//         half* ptr_line_1 = sum_buffer + next_offset + w;
//         #pragma unroll dst_N
//         for(int j = 0; j < dst_N; ++j)
//         {
//             float temp0 = ptr_line_0[j];
//             float temp1 = ptr_line_1[j];
//             ELE_T temp = temp0 + h1lambda*(temp1 - temp0);
//             temp_storage.m1[j] = temp;
//         }
//         *(dst_T*)(ptr_block_output + d_offset) = temp_storage.m2;
//     }
// }

// template<typename ELE_T,typename dst_T, int dst_N, int dst_shiftN>
// __global__ void ppl_cukernel_resize_bilinear_opt_line_t2(
//     const ELE_T* input,
//     ELE_T* output,
//     float h_scale,
//     float w_scale,
//     float roi_0,
//     float roi_1,
//     float roi_2,
//     float roi_3,
//     float extrapolation_value,
//     int in_height,
//     int in_hc,
//     int in_width,
//     int out_height,
//     int out_hc,
//     int out_width,
//     int block_line,
//     int transform_mode,
//     DivModFast out_width_fast,
//     DivModFast out_height_fast
// )
// {
//     int inSize = in_hc * in_width;
//     int outSize = out_hc * out_width;
//     int out_h_start = blockIdx.x * block_line;
//     int out_block_height = min(block_line, out_hc - out_h_start);
//     int out_h_end = out_h_start + out_block_height - 1;
//     if(out_block_height <= 0) return;
//     int oh_s, ow_s, oc_s;
//     int oh_e, ow_e, oc_e;
//     out_height_fast.divmod(out_h_start, oc_s, oh_s);
//     out_height_fast.divmod(out_h_end, oc_e, oh_e);

//     // const float h1_f = cudaComputeBilinearSourceIndex(h_scale,roi_2,roi_3, oh_s, out_height, in_height, transform_mode);
//     const float h1_f = max(h_scale * oh_s,0);
//     const int in_h_start = floor(h1_f);
//     // const float h2_f = cudaComputeBilinearSourceIndex(h_scale,roi_2,roi_3, oh_e, out_height, in_height, transform_mode);
//     const float h2_f = max(h_scale *oh_e, 0);
//     const int in_h_end = floor(h2_f);
//     int h3 = min(in_h_end + 1, in_height - 1);
//     int in_channel_height = (oc_e - oc_s) * in_height + h3 - in_h_start + 1;
//     unsigned int in_offset = blockIdx.y * inSize + (oc_s * in_height + in_h_start) * in_width;
    
//     int out_block_offset = out_h_start * out_width;
//     const ELE_T * ptr_block_input = input + in_offset;

//     ELE_T *ptr_block_output = output + blockIdx.y * outSize + out_block_offset;
//     __shared__ half sum_buffer[8192];
//     int out_block_size = out_block_height * out_width;
//     int buffer_size = in_channel_height * out_width;
    
//     for(int i = threadIdx.x; i < buffer_size; i += blockDim.x)
//     {
//         int local_h, local_w;
//         out_width_fast.divmod(i, local_h, local_w);
//         float w1r  = max(w_scale * local_w, 0);
//         int w1 = floor(w1r);
//         const float w1lambda = w1r - w1;
//         float src0 = ptr_block_input[local_h * in_width + w1];
//         int w1n = min(in_width - 1, w1 + 1);
//         float src1 = ptr_block_input[local_h * in_width + w1n];
//         float temp0 = src0 + w1lambda * (src1 - src0);
//         sum_buffer[i] = temp0;
//     }
//     __syncthreads();
//     int out_block_sizeN = out_block_size >> dst_shiftN;
//     for(int i = threadIdx.x; i < out_block_sizeN; i += blockDim.x){
//         int t, h, w, c;
//         int d_offset = i << dst_shiftN;
//         out_width_fast.divmod(d_offset + out_block_offset, t, w);
//         out_height_fast.divmod(t, c, h);
//         const float h1r = max(h_scale * h,0);
//         const int h1 = floor(h1r);
//         float h1lambda = h1r - h1;
//         union {
//             ELE_T m1[dst_N];
//             dst_T m2;
//         } temp_storage;
//         int c_offset = (c - oc_s) * in_height;
//         int local_h = c_offset + h1 - in_h_start;
//         int next_h = c_offset + min(h1 + 1,in_height - 1) - in_h_start;
//         int first_offset = local_h * out_width;
//         int next_offset = next_h * out_width;
//         half* ptr_line_0 = sum_buffer + first_offset + w;
//         half* ptr_line_1 = sum_buffer + next_offset + w;
//         #pragma unroll dst_N
//         for(int j = 0; j < dst_N; ++j)
//         {
//             float temp0 = ptr_line_0[j];
//             float temp1 = ptr_line_1[j];
//             ELE_T temp = temp0 + h1lambda*(temp1 - temp0);
//             temp_storage.m1[j] = temp;
//         }
//         *(dst_T*)(ptr_block_output + d_offset) = temp_storage.m2;
//     }
// }

// template<typename ELE_T,typename dst_T, int dst_N, int dst_shiftN>
// __global__ void ppl_cukernel_resize_bilinear_opt_line_t4(
//     const ELE_T* input,
//     ELE_T* output,
//     float h_scale,
//     float w_scale,
//     float roi_0,
//     float roi_1,
//     float roi_2,
//     float roi_3,
//     float extrapolation_value,
//     int in_height,
//     int in_hc,
//     int in_width,
//     int out_height,
//     int out_hc,
//     int out_width,
//     int block_line,
//     int transform_mode,
//     DivModFast out_width_fast,
//     DivModFast out_height_fast
// )
// {
//     int inSize = in_hc * in_width;
//     int outSize = out_hc * out_width;
//     int out_h_start = blockIdx.x * block_line;
//     int out_block_height = min(block_line, out_hc - out_h_start);
//     int out_h_end = out_h_start + out_block_height - 1;
//     if(out_block_height <= 0) return;
//     int oh_s, ow_s, oc_s;
//     int oh_e, ow_e, oc_e;
//     out_height_fast.divmod(out_h_start, oc_s, oh_s);
//     out_height_fast.divmod(out_h_end, oc_e, oh_e);

//     // const float h1_f = cudaComputeBilinearSourceIndex(h_scale,roi_2,roi_3, oh_s, out_height, in_height, transform_mode);
//     const float h1_f = max(h_scale * (oh_s + 0.5),0);
//     const int in_h_start = floor(h1_f);
//     // const float h2_f = cudaComputeBilinearSourceIndex(h_scale,roi_2,roi_3, oh_e, out_height, in_height, transform_mode);
//     const float h2_f = max(h_scale * (oh_e + 0.5), 0);
//     const int in_h_end = floor(h2_f);
//     int h3 = min(in_h_end + 1, in_height - 1);
//     int in_channel_height = (oc_e - oc_s) * in_height + h3 - in_h_start + 1;
//     unsigned int in_offset = blockIdx.y * inSize + (oc_s * in_height + in_h_start) * in_width;
    
//     int out_block_offset = out_h_start * out_width;
//     const ELE_T * ptr_block_input = input + in_offset;

//     ELE_T *ptr_block_output = output + blockIdx.y * outSize + out_block_offset;
//     __shared__ half sum_buffer[8192];
//     int out_block_size = out_block_height * out_width;
//     int buffer_size = in_channel_height * out_width;
    
//     for(int i = threadIdx.x; i < buffer_size; i += blockDim.x)
//     {
//         int local_h, local_w;
//         out_width_fast.divmod(i, local_h, local_w);
//         float w1r  = max(w_scale * (local_w + 0,5), 0);
//         int w1 = floor(w1r);
//         const float w1lambda = w1r - w1;
//         float src0 = ptr_block_input[local_h * in_width + w1];
//         int w1n = min(in_width - 1, w1 + 1);
//         float src1 = ptr_block_input[local_h * in_width + w1n];
//         float temp0 = src0 + w1lambda * (src1 - src0);
//         sum_buffer[i] = temp0;
//     }
//     __syncthreads();
//     int out_block_sizeN = out_block_size >> dst_shiftN;
//     for(int i = threadIdx.x; i < out_block_sizeN; i += blockDim.x){
//         int t, h, w, c;
//         int d_offset = i << dst_shiftN;
//         out_width_fast.divmod(d_offset + out_block_offset, t, w);
//         out_height_fast.divmod(t, c, h);
//         const float h1r = max(h_scale * h,0);
//         const int h1 = floor(h1r);
//         float h1lambda = h1r - h1;
//         union {
//             ELE_T m1[dst_N];
//             dst_T m2;
//         } temp_storage;
//         int c_offset = (c - oc_s) * in_height;
//         int local_h = c_offset + h1 - in_h_start;
//         int next_h = c_offset + min(h1 + 1,in_height - 1) - in_h_start;
//         int first_offset = local_h * out_width;
//         int next_offset = next_h * out_width;
//         half* ptr_line_0 = sum_buffer + first_offset + w;
//         half* ptr_line_1 = sum_buffer + next_offset + w;
//         #pragma unroll dst_N
//         for(int j = 0; j < dst_N; ++j)
//         {
//             float temp0 = ptr_line_0[j];
//             float temp1 = ptr_line_1[j];
//             ELE_T temp = temp0 + h1lambda*(temp1 - temp0);
//             temp_storage.m1[j] = temp;
//         }
//         *(dst_T*)(ptr_block_output + d_offset) = temp_storage.m2;
//     }
// }

template <typename T>
__global__ void ppl_cukernel_resize_bilinear_opt(
    int num_threads,
    float h_scale,
    float w_scale,
    float roi_0,
    float roi_1,
    float roi_2,
    float roi_3,
    float extrapolation_value,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode,
    DivModFast out_width_fast)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        int h2, w2;
        out_width_fast.divmod(index, h2, w2);
        const float h1r = cudaComputeBilinearSourceIndex(h_scale,roi_2,roi_3, h2, out_height, in_height, transform_mode);
        const int h1         = floor(h1r);
        const int h1p        = h1 < 0 ? 0 : (h1 < in_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = 1.f - h1lambda;
        const float w1r      = cudaComputeBilinearSourceIndex(w_scale,roi_0,roi_1, w2, out_width, in_width, transform_mode);
        const int w1         = floor(w1r);
        const int w1p        = w1 < 0 ? 0 : (w1 < in_width - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = 1.f - w1lambda;
        const float hw00lambda = h0lambda*w0lambda;
        const float hw01lambda = h0lambda*w1lambda;
        const float hw10lambda = h1lambda*w0lambda;
        const float hw11lambda = h1lambda*w1lambda;
        int imageSize = in_height*in_width;
        int outImageSize = out_height*out_width;
        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = input + start_c * imageSize + max(h1,0)*in_width + max(w1,0);
        T* pos2 = output + start_c * outImageSize + index;
        const T* pos1_next_line = pos1 + h1p*in_width;

        if(transform_mode == 5){
            if(h1r < 0 || h1r >= in_height - 1 || w1r < 0 || w1r >= in_width - 1 ){
                for (int c = 0;
                    (start_c + c) < channels && c < channels_per_piece; ++c) { //右边一个和下边一个
                    pos2[0] = extrapolation_value;
                    pos2 += outImageSize;
                }
                return;
            }
        }
    
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) { //右边一个和下边一个
            T src00 = pos1[0];
            T src01 = pos1[w1p];
            T src10 = pos1_next_line[0];
            T src11 = pos1_next_line[w1p];
            T temp = hw00lambda*(float)src00 + hw01lambda*(float)src01 + hw10lambda*(float)src10 + hw11lambda*(float)src11;

            pos2[0] = temp;
            pos1 += imageSize;
            pos1_next_line += imageSize;
            pos2 += outImageSize;
        }
    }
}

__global__ void ppl_cukernel_resize_bilinear_nhwc_sm_opt(
    float h_scale,
    float w_scale,
    float roi_0,
    float roi_1,
    float roi_2,
    float roi_3,
    float extrapolation_value,
    int pad_channels,
    const half* input,
    int in_height,
    int in_width,
    half* output,
    int out_height,
    int out_width,
    int iTileHeight,
    int iTileWidth,
    int transform_mode,
    DivModFast blocky_fast)
{
    int by, n;
    int blockChannel = blockDim.x << 3;
    int blockWidth = blockDim.y;
    int blockHeight = blockDim.z;
    int blockWC = blockChannel * blockWidth;
    int cid = (threadIdx.x + blockIdx.x * blockDim.x) << 3;
    blocky_fast.divmod(blockIdx.z, n, by);
    int start_y = by * blockHeight;
    int start_x = blockIdx.y * blockDim.y;
    int idx = threadIdx.y + start_x;
    int idy = start_y + threadIdx.z;
    if(idx >= out_width || idy >= out_height || cid >= pad_channels) return;
    int iSize = in_height * in_width;
    int oSize = out_height * out_width;
    half * ptr_block_output = output + n * oSize * pad_channels + (idy * out_width + idx) * pad_channels + cid;
    const half* ptr_block_input = input + n * iSize * pad_channels + cid;
    const float h1r = cudaComputeBilinearSourceIndex(h_scale, roi_2, roi_3, start_y, out_height, in_height, transform_mode);
    int h1_s = floor(h1r);
    float w1r = cudaComputeBilinearSourceIndex(w_scale,roi_0,roi_1, idx, out_width, in_width, transform_mode);
    int w1 = floor(w1r);
    int w1p = (w1 < in_width - 1) ? 1:0;
    float w1lambda = w1r - w1;
    float w0lambda = 1.f - w1lambda;
    v2f vw0, vw1;
    vw0[0] = w0lambda; vw0[1] = w0lambda;
    vw1[0] = w1lambda; vw1[1] = w1lambda;
    __shared__ float sm_buffer[4096];
    half *ptr_sm_input = (half*)(sm_buffer + iTileHeight * blockWC);

    int iValidHeight = min(iTileHeight, in_height - h1_s);
    int hw_offset = threadIdx.y * blockChannel + (threadIdx.x<<3);
    if(threadIdx.z < iValidHeight) {
        float *ptr_sm = sm_buffer + threadIdx.z * blockWC + hw_offset;
        int offset = (h1_s + threadIdx.z)*in_width + w1;
        if(w1p){
            float4 reg_buffer0, reg_buffer1;
            half* buffer0 = (half*)&reg_buffer0;
            half* buffer1 = (half*)&reg_buffer1;
            reg_buffer0 = *(float4*)(ptr_block_input + offset * pad_channels);
            reg_buffer1 = *(float4*)(ptr_block_input + (offset + w1p) * pad_channels);
            #pragma unroll 4
            for(int i = 0; i < 8; i += 2) {
                v2f vdst, vsrc0, vsrc1;
                vdst[0] = 0.0; vdst[1] = 0.0;
                vsrc0[0] = buffer0[i]; vsrc0[1] = buffer0[i + 1];
                vsrc1[0] = buffer1[i]; vsrc1[1] = buffer1[i + 1];
                vdst = __builtin_mxc_pk_fma_f32(vsrc0, vw0, vdst);
                vdst = __builtin_mxc_pk_fma_f32(vsrc1, vw1, vdst);
                *(float2*)(ptr_sm + i) = *(float2*)&vdst;
            }
        } else {
            float4 reg_buffer, reg_buffer1;
            half* buffer = (half*)&reg_buffer;
            reg_buffer = *(float4*)(ptr_block_input + offset * pad_channels);
            #pragma unroll 8
            for(int i = 0; i < 8; i++) {
                ptr_sm[i] = buffer[i];
            }
        }
    }
    __syncthreads();
    
    float h2r = cudaComputeBilinearSourceIndex(h_scale, roi_2, roi_3, idy, out_height, in_height, transform_mode);
    int h2 = h2r;
    int h2p = (h2 < in_height - 1) ? 1 : 0;
    float h1lambda = h2r - h2;
    float h0lambda = 1.f - h1lambda;
    v2f vh0, vh1;
    vh0[0] = h0lambda; vh0[1] = h0lambda;
    vh1[0] = h1lambda; vh1[1] = h1lambda;
    float4 reg_dst;
    half* dst = (half*)&reg_dst;
    float *ptr_sm = sm_buffer + (h2 - h1_s) * blockWC + hw_offset;
    if(h2p) {
        float* ptr_next_sm = ptr_sm + h2p * blockWC;
        #pragma unroll 4
        for(int i = 0; i < 8; i += 2) {
            v2f vdst, vsrc0, vsrc1;
            vdst[0] = 0.0; vdst[1] = 0.0;
            vsrc0[0] = ptr_sm[i]; vsrc0[1] = ptr_sm[i + 1];
            vsrc1[0] = ptr_next_sm[i]; vsrc1[1] = ptr_next_sm[i + 1];
            vdst = __builtin_mxc_pk_fma_f32(vsrc0, vh0, vdst);
            vdst = __builtin_mxc_pk_fma_f32(vsrc1, vh1, vdst);
            dst[i] = vdst[0];
            dst[i + 1] = vdst[1];
        }
    } else {
        #pragma unroll 8
        for(int i = 0; i < 8; i++) {
            half temp;
            temp = ptr_sm[i];
            dst[i] = temp;
        }
    }
    *(float4*)ptr_block_output = reg_dst;
}
#endif//OPT_RESIZE

template <typename T>
__global__ void ppl_cukernel_resize_bilinear_int8(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const float h1r = cudaComputeSourceIndex(h_scale, h2, transform_mode);

        const int h1         = h1r;
        const int h1p        = (h1 < in_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = 1.f - h1lambda;

        //const float w1r = w_scale * w2;
        const float w1r      = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1         = w1r;
        const int w1p        = (w1 < in_width - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = 1.f - w1lambda;

        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = &input[start_c * in_height * in_width + max(h1,0) * in_width + max(w1,0)];
        T* pos2       = &output[start_c * out_height * out_width + h2 * out_width + w2];
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) { //右边一个和下边一个
            // pos2[0] = h0lambda * (w0lambda * pos1[0] +
            // w1lambda * pos1[w1p]) +
            // h1lambda * (w0lambda * pos1[h1p * in_width] +
            // w1lambda * pos1[h1p * in_width + w1p]);
            int32_t temp = bilinear_interplote<T>(w0lambda, w1lambda, h0lambda, h1lambda, pos1[0], pos1[w1p], pos1[h1p * in_width], pos1[h1p * in_width + w1p]);
            temp = round(temp * in_scale / out_scale);
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            pos2[0] = temp;
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_bilinear(
    int num_threads,
    float h_scale,
    float w_scale,
    float roi_0,
    float roi_1,
    float roi_2,
    float roi_3,
    float extrapolation_value,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const float h1r = cudaComputeBilinearSourceIndex(h_scale,roi_2,roi_3, h2, out_height, in_height, transform_mode);

        const int h1         = floor(h1r);
        const int h1p        = h1 < 0 ? 0 : (h1 < in_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = 1.f - h1lambda;

        //const float w1r = w_scale * w2;
        const float w1r      = cudaComputeBilinearSourceIndex(w_scale,roi_0,roi_1, w2, out_width, in_width, transform_mode);
        const int w1         = floor(w1r);
        const int w1p        = w1 < 0 ? 0 : (w1 < in_width - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = 1.f - w1lambda;

        int start_c = blockIdx.y * channels_per_piece;
        T* pos2       = &output[start_c * out_height * out_width + h2 * out_width + w2];
        if(transform_mode == 5){
            if(h1r < 0 || h1r >= in_height - 1 || w1r < 0 || w1r >= in_width - 1 ){
                for (int c = 0;
                    (start_c + c) < channels && c < channels_per_piece; ++c) { //右边一个和下边一个

                    pos2[0] = extrapolation_value;
                    pos2 += out_width * out_height;
                }
                return;
            }
        }
        const T* pos1 = &input[start_c * in_height * in_width + max(h1,0) * in_width + max(w1,0)];

        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) { //右边一个和下边一个
            // pos2[0] = h0lambda * (w0lambda * pos1[0] +
            // w1lambda * pos1[w1p]) +
            // h1lambda * (w0lambda * pos1[h1p * in_width] +
            // w1lambda * pos1[h1p * in_width + w1p]);
            pos2[0] = bilinear_interplote<T>(w0lambda, w1lambda, h0lambda, h1lambda, pos1[0], pos1[w1p], pos1[h1p * in_width], pos1[h1p * in_width + w1p]);
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_bilinear_nhwc(
    int num_threads,
    float h_scale,
    float w_scale,
    float roi_0,
    float roi_1,
    float roi_2,
    float roi_3,
    float extrapolation_value,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const float h1r = cudaComputeBilinearSourceIndex(h_scale, roi_2, roi_3, h2, out_height, in_height, transform_mode);

        const int h1         = floor(h1r);
        const int h1p        = h1 < 0 ? 0: (h1 < in_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = 1.f - h1lambda;

        //const float w1r = w_scale * w2;
        const float w1r      = cudaComputeBilinearSourceIndex(w_scale,roi_0,roi_1, w2, out_width, in_width, transform_mode);
        const int w1         = floor(w1r);
        const int w1p        = w1 < 0 ? 0 : (w1 < in_width - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = 1.f - w1lambda;

        const T* pos1 = &input[bidx * in_height * in_width * pad_channels + (max(h1,0) * in_width + max(w1,0)) * pad_channels];
        T* pos2       = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        int start_c = blockIdx.y * channels_per_piece;

        if(transform_mode == 5){
            if(h1r < 0 || h1r >= in_height - 1 || w1r < 0 || w1r >= in_width - 1){
                for(int c = start_c; c < channels && (c - start_c) < channels_per_piece; ++c) {
                    pos2[c] = extrapolation_value;
                }
                return;
            }
        }
        for (int c = start_c;
            c < channels && (c - start_c) < channels_per_piece; ++c) { //右边一个和下边一个
            pos2[c] = bilinear_interplote<T>(w0lambda, w1lambda, h0lambda, h1lambda,
                pos1[c], pos1[w1p * pad_channels + c], pos1[(h1p * in_width) * pad_channels + c], pos1[(h1p * in_width + w1p) * pad_channels + c]);
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_bilinear_nhwc_int8(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const float h1r = cudaComputeSourceIndex(h_scale, h2, transform_mode);

        const int h1         = h1r;
        const int h1p        = (h1 < in_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = 1.f - h1lambda;

        //const float w1r = w_scale * w2;
        const float w1r      = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1         = w1r;
        const int w1p        = (w1 < in_width - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = 1.f - w1lambda;

        const T* pos1 = &input[bidx * in_height * in_width * pad_channels + (max(h1,0) * in_width + max(w1,0)) * pad_channels];
        T* pos2       = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        int start_c = blockIdx.y * channels_per_piece;
        for (int c = start_c;
            c < channels && (c - start_c) < channels_per_piece; ++c) { //右边一个和下边一个
            int32_t temp = bilinear_interplote<T>(w0lambda, w1lambda, h0lambda, h1lambda,
                pos1[c], pos1[w1p * pad_channels + c], pos1[(h1p * in_width) * pad_channels + c], pos1[(h1p * in_width + w1p) * pad_channels + c]);
            int32_t res = round(temp * in_scale / out_scale);
            if(res > 127) res = 127;
            if(res < -128) res = -128;
            pos2[c] = res;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_int8(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const float h1r = cudaComputeSourceIndex(h_scale, h2, transform_mode);
        const int h1    = h1r;

        //const float w1r = w_scale * w2;
        const float w1r = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1    = w1r;

        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = &input[start_c * in_height * in_width + h1 * in_width + w1];
        T* pos2       = &output[start_c * out_height * out_width + h2 * out_width + w2];
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            int32_t temp = round(pos1[0] * in_scale / out_scale);
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            pos2[0] = temp;
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_round_prefer_floor_int8(
    int num_threads,
    double h_scale,
    double w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1rfloor = floor(h1r);
        const int h1    = h1r - h1rfloor <= 0.5 ? h1rfloor: h1rfloor + 1;

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1rfloor = floor(w1r);
        const int w1    = w1r - w1rfloor <= 0.5 ? w1rfloor: w1rfloor + 1;

        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = &input[start_c * in_height * in_width + min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)];
        T* pos2       = &output[start_c * out_height * out_width + h2 * out_width + w2];
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            int32_t temp = round(pos1[0] * in_scale / out_scale);
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            pos2[0] = temp;
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_round_prefer_ceil_int8(
    int num_threads,
    double h_scale,
    double w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1rfloor = floor(h1r);
        const int h1    = h1r - h1rfloor < 0.5 ? h1rfloor: h1rfloor + 1;

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1rfloor = floor(w1r);
        const int w1    = w1r - w1rfloor < 0.5 ? w1rfloor: w1rfloor + 1;

        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = &input[start_c * in_height * in_width + min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)];
        T* pos2       = &output[start_c * out_height * out_width + h2 * out_width + w2];
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            int32_t temp = round(pos1[0] * in_scale / out_scale);
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            pos2[0] = temp;
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_ceil_int8(
    int num_threads,
    double h_scale,
    double w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1    = ceil(h1r);

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1    = ceil(w1r);

        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = &input[start_c * in_height * in_width + min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)];
        T* pos2       = &output[start_c * out_height * out_width + h2 * out_width + w2];
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            int32_t temp = round(pos1[0] * in_scale / out_scale);
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            pos2[0] = temp;
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const float h1r = cudaComputeSourceIndex(h_scale, h2, transform_mode);
        const int h1    = h1r;

        //const float w1r = w_scale * w2;
        const float w1r = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1    = w1r;

        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = &input[start_c * in_height * in_width + h1 * in_width + w1];
        T* pos2       = &output[start_c * out_height * out_width + h2 * out_width + w2];
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            pos2[0] = pos1[0];
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_round_prefer_floor(
    int num_threads,
    double h_scale,
    double w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1rfloor = floor(h1r);
        const int h1    = h1r - h1rfloor <= 0.5 ? h1rfloor: h1rfloor + 1;

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1rfloor = floor(w1r);
        const int w1    = w1r - w1rfloor <= 0.5 ? w1rfloor: w1rfloor + 1;

        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = &input[start_c * in_height * in_width + min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)];
        T* pos2       = &output[start_c * out_height * out_width + h2 * out_width + w2];
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            pos2[0] = pos1[0];
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_round_prefer_ceil(
    int num_threads,
    double h_scale,
    double w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1rfloor = floor(h1r);
        const int h1    = h1r - h1rfloor < 0.5 ? h1rfloor: h1rfloor + 1;

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1rfloor = floor(w1r);
        const int w1    = w1r - w1rfloor < 0.5 ? w1rfloor: w1rfloor + 1;

        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = &input[start_c * in_height * in_width + min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)];
        T* pos2       = &output[start_c * out_height * out_width + h2 * out_width + w2];
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            pos2[0] = pos1[0];
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_ceil(
    int num_threads,
    double h_scale,
    double w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1    = ceil(h1r);

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1    = ceil(w1r);

        int start_c = blockIdx.y * channels_per_piece;
        const T* pos1 = &input[start_c * in_height * in_width + min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)];
        T* pos2       = &output[start_c * out_height * out_width + h2 * out_width + w2];
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            pos2[0] = pos1[0];
            pos1 += in_width * in_height;
            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_round_prefer_floor_nhwc(
    int num_threads,
    double h_scale,
    double w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1rfloor = floor(h1r);
        const int h1    = h1r - h1rfloor <= 0.5 ? h1rfloor: h1rfloor + 1;

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1rfloor = floor(w1r);
        const int w1    = w1r - w1rfloor <= 0.5 ? w1rfloor: w1rfloor + 1;

        const T* pos1 = &input[bidx * in_height * in_width * pad_channels + (min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)) * pad_channels];
        T* pos2       = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        int start_c = blockIdx.y * channels_per_piece;
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            pos2[start_c + c] = pos1[start_c + c];
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_round_prefer_ceil_nhwc(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1rfloor = floor(h1r);
        const int h1    = h1r - h1rfloor < 0.5 ? h1rfloor: h1rfloor + 1;

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1rfloor = floor(w1r);
        const int w1    = w1r - w1rfloor < 0.5 ? w1rfloor: w1rfloor + 1;

        const T* pos1 = &input[bidx * in_height * in_width * pad_channels + (min(max(h1,0), in_height - 1) * in_width + min(max(w1,0),in_width - 1)) * pad_channels];
        T* pos2       = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        int start_c = blockIdx.y * channels_per_piece;
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            pos2[start_c + c] = pos1[start_c + c];
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_ceil_nhwc(
    int num_threads,
    double h_scale,
    double w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1    = ceil(h1r);

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1    = ceil(w1r);

        const T* pos1 = &input[bidx * in_height * in_width * pad_channels + (min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)) * pad_channels];
        T* pos2       = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        int start_c = blockIdx.y * channels_per_piece;
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            pos2[start_c + c] = pos1[start_c + c];
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_nhwc(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const float h1r = cudaComputeSourceIndex(h_scale, h2, transform_mode);
        const int h1    = h1r;

        //const float w1r = w_scale * w2;
        const float w1r = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1    = w1r;

        const T* pos1 = &input[bidx * in_height * in_width * pad_channels + (h1 * in_width + w1) * pad_channels];
        T* pos2       = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        int start_c = blockIdx.y * channels_per_piece;
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            pos2[start_c + c] = pos1[start_c + c];
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_cubic_nwhc(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float cubic_coeff,
    int transform_mode,
    int exclude_outside)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        const float h1r      = cudaComputeSourceIndex(h_scale, h2, transform_mode);
        const int h1         = floorf(h1r);
        const float h1lambda = h1r - h1;

        const float w1r      = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1         = floorf(w1r);
        const float w1lambda = w1r - w1;

        T* pos2 = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        const T* pos1 = input + bidx * in_height * in_width * pad_channels;
        int start_c = blockIdx.y * channels_per_piece;
        for (int c = start_c;
            c < channels && (c - start_c) < channels_per_piece; ++c) {
            T coefficients[4];

            for (int k = 0; k < 4; k++) {
                coefficients[k] = cubic_interp1dx<T>(
                    resize_get_value_bounded_nhwc(
                        pos1, in_height, in_width, pad_channels, c, h1 - 1 + k, w1 - 1),
                    resize_get_value_bounded_nhwc(
                        pos1, in_height, in_width, pad_channels, c, h1 - 1 + k, w1 + 0),
                    resize_get_value_bounded_nhwc(
                        pos1, in_height, in_width, pad_channels, c, h1 - 1 + k, w1 + 1),
                    resize_get_value_bounded_nhwc(
                        pos1, in_height, in_width, pad_channels, c, h1 - 1 + k, w1 + 2),
                    w1 -1,
                    w1,
                    w1 + 1,
                    w1 + 2,
                    in_width,
                    exclude_outside,
                    w1lambda,
                    cubic_coeff);
            }
            pos2[c] = cubic_interp1dx<T>(
                coefficients[0],
                coefficients[1],
                coefficients[2],
                coefficients[3],
                h1 - 1,
                h1,
                h1 + 1,
                h1 + 2,
                in_height,
                exclude_outside,
                h1lambda,
                cubic_coeff);
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_round_prefer_floor_nhwc_int8(
    int num_threads,
    double h_scale,
    double w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1rfloor = floor(h1r);
        const int h1    = h1r - h1rfloor <= 0.5 ? h1rfloor: h1rfloor + 1;

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1rfloor = floor(w1r);
        const int w1    = w1r - w1rfloor <= 0.5 ? w1rfloor: w1rfloor + 1;

        const T* pos1 = &input[bidx * in_height * in_width * pad_channels + (min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)) * pad_channels];
        T* pos2       = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        int start_c = blockIdx.y * channels_per_piece;
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            int32_t temp = round(pos1[start_c + c] * in_scale / out_scale);
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            pos2[start_c + c] = temp;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_round_prefer_ceil_nhwc_int8(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1rfloor = floor(h1r);
        const int h1    = h1r - h1rfloor < 0.5 ? h1rfloor: h1rfloor + 1;

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1rfloor = floor(w1r);
        const int w1    = w1r - w1rfloor < 0.5 ? w1rfloor: w1rfloor + 1;

        const T* pos1 = &input[bidx * in_height * in_width * pad_channels + (min(max(h1,0), in_height - 1) * in_width + min(max(w1,0),in_width - 1)) * pad_channels];
        T* pos2       = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        int start_c = blockIdx.y * channels_per_piece;
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            int32_t temp = round(pos1[start_c + c] * in_scale / out_scale);
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            pos2[start_c + c] = temp;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_ceil_nhwc_int8(
    int num_threads,
    double h_scale,
    double w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const double h1r = cudaComputeSourceSTDIndex(h_scale, h2, transform_mode);
        const int h1    = ceil(h1r);

        //const float w1r = w_scale * w2;
        const double w1r = cudaComputeSourceSTDIndex(w_scale, w2, transform_mode);
        const int w1    = ceil(w1r);

        const T* pos1 = &input[bidx * in_height * in_width * pad_channels + (min(max(h1,0),in_height - 1) * in_width + min(max(w1,0),in_width - 1)) * pad_channels];
        T* pos2       = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        int start_c = blockIdx.y * channels_per_piece;
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            int32_t temp = round(pos1[start_c + c] * in_scale / out_scale);
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            pos2[start_c + c] = temp;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_nearest_nhwc_int8(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        //const float h1r = h_scale * h2;
        const float h1r = cudaComputeSourceIndex(h_scale, h2, transform_mode);
        const int h1    = h1r;

        //const float w1r = w_scale * w2;
        const float w1r = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1    = w1r;

        const T* pos1 = &input[bidx * in_height * in_width * pad_channels + (h1 * in_width + w1) * pad_channels];
        T* pos2       = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        int start_c = blockIdx.y * channels_per_piece;
        for (int c = 0;
            (start_c + c) < channels && c < channels_per_piece; ++c) {
            int32_t res = round(pos1[start_c + c] * in_scale / out_scale);
            if(res > 127) res = 127;
            if(res < -128) res = -128;
            pos2[start_c + c] = res;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_cubic_nhwc_int8(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int pad_channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float cubic_coeff,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.z;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        const float h1r      = cudaComputeSourceIndex(h_scale, h2, transform_mode);
        const int h1         = floorf(h1r);
        const float h1lambda = h1r - h1;

        const float w1r      = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1         = floorf(w1r);
        const float w1lambda = w1r - w1;

        T* pos2 = &output[bidx * out_height * out_width * pad_channels + (h2 * out_width + w2) * pad_channels];
        const T* pos1 = input + bidx * in_height * in_width * pad_channels;
        int start_c = blockIdx.y * channels_per_piece;
        for (int c = start_c;
            c < channels && (c - start_c) < channels_per_piece; ++c) {
            float coefficients[4];

            for (int k = 0; k < 4; k++) {
                coefficients[k] = cubic_interp1d_float(
                    resize_get_value_bounded_nhwc(
                        pos1, in_height, in_width, pad_channels, c, h1 - 1 + k, w1 - 1),
                    resize_get_value_bounded_nhwc(
                        pos1, in_height, in_width, pad_channels, c, h1 - 1 + k, w1 + 0),
                    resize_get_value_bounded_nhwc(
                        pos1, in_height, in_width, pad_channels, c, h1 - 1 + k, w1 + 1),
                    resize_get_value_bounded_nhwc(
                        pos1, in_height, in_width, pad_channels, c, h1 - 1 + k, w1 + 2),
                    w1lambda,
                    cubic_coeff);
            }
            float temp = cubic_interp1d_float(
                coefficients[0],
                coefficients[1],
                coefficients[2],
                coefficients[3],
                h1lambda,
                cubic_coeff);
            int32_t result = round(temp * in_scale / out_scale);
            if(result > 127) result = 127;
            if(result < -128) result = -128;
            pos2[c] = result;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_cubic_int8(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float cubic_coeff,
    float in_scale,
    float out_scale,
    int transform_mode)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        const float h1r      = cudaComputeSourceIndex(h_scale, h2, transform_mode);
        const int h1         = floorf(h1r);
        const float h1lambda = h1r - h1;

        const float w1r      = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1         = floorf(w1r);
        const float w1lambda = w1r - w1;

        int start_c = blockIdx.y * channels_per_piece;
        T* pos2 = &output[start_c * out_height * out_width +h2 * out_width + w2];
        for (int c = start_c;
            start_c < channels && (c - start_c) < channels_per_piece; ++c) {
            float coefficients[4];

            for (int k = 0; k < 4; k++) {
                coefficients[k] = cubic_interp1d_float(
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 - 1),
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 + 0),
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 + 1),
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 + 2),
                    w1lambda,
                    cubic_coeff);
            }
            float temp = cubic_interp1d_float(
                coefficients[0],
                coefficients[1],
                coefficients[2],
                coefficients[3],
                h1lambda,
                cubic_coeff);
            int32_t res = round(temp * in_scale / out_scale);
            if(res > 127) res = 127;
            if(res < -128) res = -128;
            pos2[0] = res;

            pos2 += out_width * out_height;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_resize_cubic(
    int num_threads,
    float h_scale,
    float w_scale,
    int channels,
    int channels_per_piece,
    const T* input,
    int in_height,
    int in_width,
    T* output,
    int out_height,
    int out_width,
    float cubic_coeff,
    int transform_mode,
    int exclude_outside)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_threads) {
        const int w2 = index % out_width; // 0:out_width-1
        const int h2 = index / out_width; // 0:out_height-1

        const float h1r      = cudaComputeSourceIndex(h_scale, h2, transform_mode);
        const int h1         = floorf(h1r);
        const float h1lambda = h1r - h1;

        const float w1r      = cudaComputeSourceIndex(w_scale, w2, transform_mode);
        const int w1         = floorf(w1r);
        const float w1lambda = w1r - w1;

        int start_c = blockIdx.y * channels_per_piece;
        T* pos2 = &output[start_c * out_height * out_width +h2 * out_width + w2];
        for (int c = start_c;
            c < channels && (c - start_c) < channels_per_piece; ++c) {
            T coefficients[4];

            for (int k = 0; k < 4; k++) {
                coefficients[k] = cubic_interp1dx<T>(
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 - 1),
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 + 0),
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 + 1),
                    resize_get_value_bounded(
                        input, in_height, in_width, c, h1 - 1 + k, w1 + 2),
                    w1 -1,
                    w1,
                    w1 + 1,
                    w1 + 2,
                    in_width,
                    exclude_outside,
                    w1lambda,
                    cubic_coeff);
            }
            pos2[0] = cubic_interp1dx<T>(
                coefficients[0],
                coefficients[1],
                coefficients[2],
                coefficients[3],
                h1 - 1,
                h1,
                h1 + 1,
                h1 + 2,
                in_height,
                exclude_outside,
                h1lambda,
                cubic_coeff);

            pos2 += out_width * out_height;
        }
    }
}

static inline float hostComputeAreaScale(int input_size, int output_size, int mode)
{
    if (input_size == output_size) return 1.f;
    if (mode == 2) {
        return float(input_size - 1) / (output_size - 1);
    } else if (mode == 1 && output_size < 1) {
        return 0.f;
    } else {
        return float(input_size) / output_size;
    }
}

static inline double hostComputeAreaScaleTD(int input_size, int output_size, int mode)
{
    if (input_size == output_size) return 1.f;
    if (mode == 2 || mode == 5) {
        if(output_size == 1) {
            return 0;
        }else{
            return double(input_size - 1) / (output_size - 1);
        }
    } else if (mode == 1 && output_size < 1) {
        return 0.f;
    } else {
        return double(input_size) / output_size;
    }
}
// coordinate_transformation_mode definition
// {"half_pixel", 0}, {"pytorch_half_pixel", 1}, {"align_corners", 2},
// {"asymmetric", 3}, {"tf_half_pixel_for_nn", 4}, {"tf_crop_and_resize", 5}
// interpolation mode
// {"nearest", 0}, {"linear", 1}, {"cubic", 2}
template <typename T>
ppl::common::RetCode ppl_resize_forward(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const T* input,
    const ppl::common::TensorShape* output_shape,
    T* output,
    bool scale_pre_set,
    float h_scale_pre,
    float w_scale_pre,
    float roi_0,
    float roi_1,
    float roi_2,
    float roi_3,
    float extrapolation_value,
    int transform_mode,
    int inter_mode,
    float cubic_coeff,
    int nearest_mode,
    int exclude_outside)
{
    // if (transform_mode == 5) return ppl::common::RC_UNSUPPORTED;
    int dim_count  = output_shape->GetDimCount();
    int out_height = output_shape->GetDim(2), out_width = 1;
    int in_height = input_shape->GetDim(2), in_width = 1;
    for (int it = 3; it < dim_count - 1; ++it) {
        out_height *= output_shape->GetDim(it);
        in_height *= input_shape->GetDim(it);
    }
    if (dim_count >= 4) {
        out_width    = output_shape->GetDim(dim_count - 1);
        in_width     = input_shape->GetDim(dim_count - 1);
    }
    int channels = output_shape->GetDim(0) * output_shape->GetDim(1);

    double h_scale = 0.f, w_scale = 0.f;
    if (scale_pre_set) {
        h_scale = h_scale_pre;
        w_scale = w_scale_pre;
    } else {
        h_scale = hostComputeAreaScaleTD(in_height, out_height, transform_mode);
        w_scale = hostComputeAreaScaleTD(in_width, out_width, transform_mode);
    }

    int num_threads = out_height * out_width;
    int block_size  = 256; dim3 grid_size(1, 1, 1); int channels_per_piece = 1;
    GetNumBlocks(block_size, grid_size, channels_per_piece, num_threads, channels);
    if (inter_mode == 0) {
        if(nearest_mode == 0){
            ppl_cukernel_resize_nearest_round_prefer_floor<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
        }else if(nearest_mode == 1){
            ppl_cukernel_resize_nearest_round_prefer_ceil<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
        }else if(nearest_mode == 3 ){
            ppl_cukernel_resize_nearest_ceil<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
        }else{
#ifdef OPT_RESIZE
            if (transform_mode == 2 || transform_mode == 3){
                int block_line = ((8192 / sizeof(T) - 16) / out_width) >> 4 <<4;
                if(block_line >= 1 && h_scale < 1 && w_scale < 1){
                    int total_height = out_height * channels;
                    if(sizeof(T) == 4){
                        grid_size.x = (total_height + block_line - 1) / block_line;
                        grid_size.y = 1;
                        ppl_cukernel_resize_nearest_opt_t3<T,2,2><<<grid_size, block_size, 0, stream>>>(
                            (const T*)input, h_scale, w_scale, in_height, in_width, output, out_width, block_line,total_height, DivModFast(out_height),DivModFast(out_width));
                        return ppl::common::RC_SUCCESS;
                    } else if(sizeof(T) == 2){
                        grid_size.x = (total_height + block_line - 1) / block_line;
                        grid_size.y = 1;
                        ppl_cukernel_resize_nearest_opt_t3<T,1,3><<<grid_size, block_size, 0, stream>>>(
                            input, h_scale, w_scale, in_height, in_width, output, out_width, block_line,total_height, DivModFast(out_height),DivModFast(out_width));
                        return ppl::common::RC_SUCCESS;
                    } else {
                        ppl_cukernel_resize_nearest<T><<<grid_size, block_size, 0, stream>>>(
                            num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
                        return ppl::common::RC_SUCCESS;
                    }
                }
            }
#endif//OPT_RESIZE 
            {
                ppl_cukernel_resize_nearest<T><<<grid_size, block_size, 0, stream>>>(
                    num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
            }
        }
    } else if (inter_mode == 1) {
#ifdef OPT_RESIZE
        // if(transform_mode != 5 && sizeof(T) == 2)
        // {
        //     int out_channel_height = output_shape->GetDim(1) * out_height;
        //     int in_channel_height = output_shape->GetDim(1) * in_height;
        //     int block_line = MIN(((8192 - 16) / out_width),out_channel_height);
        //     block_line = block_line >> 3 << 3;
        //     if(block_line >= 1 && (out_width & 1) == 0)
        //     {
        //         block_size=256;
        //         grid_size.x = (out_channel_height + block_line - 1)  / block_line; 
        //         grid_size.y = output_shape->GetDim(0);
        //         grid_size.z = 1;
        //         if((transform_mode == 1 || transform_mode == 0) && out_width != 1)
        //         {
        //             if((out_width & 7) == 0)
        //             {
        //                 ppl_cukernel_resize_bilinear_opt_line_t1<T, float4, 8, 3><<<grid_size, block_size, 0, stream>>>(
        //                     input,output, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value, in_height, in_channel_height, in_width, out_height, out_channel_height, 
        //                     out_width, block_line,transform_mode,DivModFast(out_width),DivModFast(out_height));
        //             } else if((out_width & 3) == 0) {
        //                 ppl_cukernel_resize_bilinear_opt_line_t1<T, float2, 4, 2><<<grid_size, block_size, 0, stream>>>(
        //                     input,output, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value, in_height, in_channel_height, in_width, out_height, out_channel_height, 
        //                     out_width, block_line,transform_mode,DivModFast(out_width),DivModFast(out_height));
        //             } 
        //             else if((out_width&1) == 0) {
        //                 ppl_cukernel_resize_bilinear_opt_line_t1<T, float, 2, 1><<<grid_size, block_size, 0, stream>>>(
        //                     input,output, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value, in_height, in_channel_height, in_width, out_height, out_channel_height, 
        //                     out_width, block_line,transform_mode,DivModFast(out_width),DivModFast(out_height));
        //             }
        //         } else if((transform_mode == 2 || transform_mode == 3)) {
        //             if((out_width & 7) == 0)
        //             {
        //                 ppl_cukernel_resize_bilinear_opt_line_t2<T, float4, 8, 3><<<grid_size, block_size, 0, stream>>>(
        //                     input,output, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value, in_height, in_channel_height, in_width, out_height, out_channel_height, 
        //                     out_width, block_line,transform_mode,DivModFast(out_width),DivModFast(out_height));
        //             } else if((out_width & 3) == 0) {
        //                 ppl_cukernel_resize_bilinear_opt_line_t2<T, float2, 4, 2><<<grid_size, block_size, 0, stream>>>(
        //                     input,output, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value, in_height, in_channel_height, in_width, out_height, out_channel_height, 
        //                     out_width, block_line,transform_mode,DivModFast(out_width),DivModFast(out_height));
        //             } 
        //             else if((out_width&1) == 0) {
        //                 ppl_cukernel_resize_bilinear_opt_line_t2<T, float, 2, 1><<<grid_size, block_size, 0, stream>>>(
        //                     input,output, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value, in_height, in_channel_height, in_width, out_height, out_channel_height, 
        //                     out_width, block_line,transform_mode,DivModFast(out_width),DivModFast(out_height));
        //             }
        //         } else {
        //             if((out_width & 7) == 0)
        //             {
        //                 ppl_cukernel_resize_bilinear_opt_line_t4<T, float4, 8, 3><<<grid_size, block_size, 0, stream>>>(
        //                     input,output, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value, in_height, in_channel_height, in_width, out_height, out_channel_height, 
        //                     out_width, block_line,transform_mode,DivModFast(out_width),DivModFast(out_height));
        //             } else if((out_width & 3) == 0) {
        //                 ppl_cukernel_resize_bilinear_opt_line_t4<T, float2, 4, 2><<<grid_size, block_size, 0, stream>>>(
        //                     input,output, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value, in_height, in_channel_height, in_width, out_height, out_channel_height, 
        //                     out_width, block_line,transform_mode,DivModFast(out_width),DivModFast(out_height));
        //             } 
        //             else if((out_width&1) == 0) {
        //                 ppl_cukernel_resize_bilinear_opt_line_t4<T, float, 2, 1><<<grid_size, block_size, 0, stream>>>(
        //                     input,output, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value, in_height, in_channel_height, in_width, out_height, out_channel_height, 
        //                     out_width, block_line,transform_mode,DivModFast(out_width),DivModFast(out_height));
        //             }
        //         }
        //         return ppl::common::RC_SUCCESS;
        //     }
        // }
        ppl_cukernel_resize_bilinear_opt<T><<<grid_size, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode,DivModFast(out_width));
        return ppl::common::RC_SUCCESS;
#endif//OPT_RESIZE
        ppl_cukernel_resize_bilinear<T><<<grid_size, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
    } else if (inter_mode == 2) {
        ppl_cukernel_resize_cubic<T><<<grid_size, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, cubic_coeff, transform_mode,exclude_outside);
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T>
ppl::common::RetCode ppl_resize_forward_nhwc(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const T* input,
    const ppl::common::TensorShape* output_shape,
    T* output,
    bool scale_pre_set,
    float h_scale_pre,
    float w_scale_pre,
    float roi_0,
    float roi_1,
    float roi_2,
    float roi_3,
    float extrapolation_value,
    int transform_mode,
    int inter_mode,
    float cubic_coeff,
    int nearest_mode,
    int exclude_outside)
{
    // if (transform_mode == 5) return ppl::common::RC_UNSUPPORTED;
    int dim_count  = output_shape->GetDimCount();
    int out_height = output_shape->GetDim(2), out_width = 1;
    int in_height = input_shape->GetDim(2), in_width = 1;
    for (int it = 3; it < dim_count - 1; ++it) {
        out_height *= output_shape->GetDim(it);
        in_height *= input_shape->GetDim(it);
    }
    if (dim_count >= 4) {
        out_width    = output_shape->GetDim(dim_count - 1);
        in_width     = input_shape->GetDim(dim_count - 1);
    }
    int channels = output_shape->GetDim(1);
    int pad_channels = channels + output_shape->GetPadding1(1);
    double h_scale = 0.f, w_scale = 0.f;
    if (scale_pre_set) {
        h_scale = h_scale_pre;
        w_scale = w_scale_pre;
    } else {
        h_scale = hostComputeAreaScaleTD(in_height, out_height, transform_mode);
        w_scale = hostComputeAreaScaleTD(in_width, out_width, transform_mode);
    }
    
    int num_threads = out_height * out_width;
    int block_size  = 256; dim3 grid_size(1, 1, 1); int channels_per_piece = 1;
    GetNumBlocks(block_size, grid_size, channels_per_piece, num_threads, channels);
    grid_size.z =  output_shape->GetDim(0);
    if (inter_mode == 0) {
        if(nearest_mode == 0){
#ifdef OPT_RESIZE
            int num_elems = out_height * out_width * pad_channels * output_shape->GetDim(0);
            if(2 == sizeof(T) && 0 == (pad_channels & 7)) {
                int blockSize = 256;
                int gridSize = (num_elems + 2047) >> 11;
                ppl_cukernel_resize_nearest_round_prefer_floor_nhwc_opt<T,float4,3><<<gridSize, blockSize, 0, stream>>>(
                    num_elems, h_scale, w_scale, pad_channels, input, in_height, in_width, output, out_height, out_width, transform_mode
                    ,DivModFast(pad_channels),DivModFast(out_width),DivModFast(out_width*out_height));
            } else if(4 == sizeof(T) && 0 == (pad_channels & 3)) {
                int blockSize = 256;
                int gridSize = (num_elems + 1023) >> 10;
                ppl_cukernel_resize_nearest_round_prefer_floor_nhwc_opt<T,float4,2><<<gridSize, blockSize, 0, stream>>>(
                    num_elems, h_scale, w_scale, pad_channels, input, in_height, in_width, output, out_height, out_width, transform_mode
                    ,DivModFast(pad_channels),DivModFast(out_width),DivModFast(out_width*out_height));
            } else {
                ppl_cukernel_resize_nearest_round_prefer_floor_nhwc<T><<<grid_size, block_size, 0, stream>>>(
                    num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
            }
#else//!OPT_RESIZE
            ppl_cukernel_resize_nearest_round_prefer_floor_nhwc<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
#endif//OPT_RESIZE
        } else if(nearest_mode == 1) {
            ppl_cukernel_resize_nearest_round_prefer_ceil_nhwc<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
        } else if(nearest_mode == 3) {
            ppl_cukernel_resize_nearest_ceil_nhwc<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
        }else{
#ifdef OPT_RESIZE
            // if(3 == transform_mode || 2 == transform_mode) {
            //     if(condition_nhwc(in_height, out_height, in_width, out_width, h_scale, w_scale)) {
            //         int N = 16 / sizeof(T);
            //         if(0 == (pad_channels & (N - 1))) {
            //             int channelsN = pad_channels / N;
            //             int inSize = in_height * in_width;
            //             int num_elements = inSize * channelsN;
            //             block_size = 256;
            //             grid_size.x =  (num_elements + block_size - 1) / block_size;
            //             grid_size.y = 1; 
            //             ppl_cukernel_resize_nearest_nhwc_opt_t3<<<grid_size, block_size, 0, stream>>>(
            //                 num_elements, (out_height + in_height - 1) / in_height, (out_width + in_width - 1) / in_width, channelsN, (const float4*)input,
            //                 in_height, in_width, (float4*)output, out_height, out_width, DivModFast(channelsN), DivModFast(in_width)
            //             );
            //             return ppl::common::RC_SUCCESS;        
            //         }
            //     }
            // }
            int batch = output_shape->GetDim(0);
            int num_elems = out_height * out_width * pad_channels * batch;
            if(2 == sizeof(T) && 0 == (pad_channels & 7)) {
                int blockSize = 256;
                int gridSize = (num_elems + 2047) >> 11;
                ppl_cukernel_resize_nearest_nhwc_opt<T,float4,3><<<gridSize, blockSize, 0, stream>>>(
                    num_elems, h_scale, w_scale, pad_channels, input, in_height, in_width, output, out_height, out_width, transform_mode
                    ,DivModFast(pad_channels),DivModFast(out_width),DivModFast(out_width*out_height));
            } else if(4 == sizeof(T) && 0 == (pad_channels & 3)) {
                int blockSize = 256;
                int gridSize = (num_elems + 1023) >> 10;
                ppl_cukernel_resize_nearest_nhwc_opt<T,float4,2><<<gridSize, blockSize, 0, stream>>>(
                    num_elems, h_scale, w_scale, pad_channels, input, in_height, in_width, output, out_height, out_width, transform_mode
                    ,DivModFast(pad_channels),DivModFast(out_width),DivModFast(out_width*out_height));
            } else {
                ppl_cukernel_resize_nearest_nhwc<T><<<grid_size, block_size, 0, stream>>>(
                    num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
            }
#else//!OPT_RESIZE
            ppl_cukernel_resize_nearest_nhwc<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
#endif//OPT_RESIZE
        }
    } else if (inter_mode == 1) {
#ifdef OPT_RESIZE
        int batch = output_shape->GetDim(0);
        int num_elems = out_height * out_width * pad_channels * batch;
        if(2 == sizeof(T) && 0 == (pad_channels & 7)) {
            if(h_scale < 1 && w_scale < 1 && transform_mode != 5)
            {
                dim3 blockSize(1,1,1);
                blockSize.x = MIN((pad_channels >> 3), 8);
                blockSize.y = 4;
                blockSize.z = 512 / blockSize.x / blockSize.y;
                int blocky = (out_height + blockSize.z - 1) / blockSize.z;
                dim3 gridSize((pad_channels + blockSize.x * 8 - 1) / (blockSize.x * 8),(out_width + 3)>>2,batch*blocky);
                float temp = hostComputeSourceIndex(h_scale, blockSize.z, transform_mode);
                int iTileWidth,iTileHeight;
                iTileHeight = MIN(temp + 4, in_height);
                temp = hostComputeSourceIndex(w_scale,blockSize.y,transform_mode);
                iTileWidth = MIN(temp + 4, in_width);
                ppl_cukernel_resize_bilinear_nhwc_sm_opt<<<gridSize,blockSize,0,stream>>>(h_scale,w_scale, roi_0, roi_1, roi_2, roi_3, extrapolation_value, pad_channels,(const half*)input,in_height,in_width,
                (half*)output,out_height,out_width,iTileHeight,iTileWidth,transform_mode,DivModFast(blocky));
                return ppl::common::RC_SUCCESS;    
            }
            int blockSize = 256;
            int gridSize = (num_elems + 2047) >> 11;
            ppl_cukernel_resize_bilinear_nhwc_opt<T,float4,8,3><<<gridSize, blockSize, 0, stream>>>(
                num_elems, h_scale, w_scale, roi_0, roi_1, roi_2, roi_3, extrapolation_value, channels,
                pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, 
                out_width, transform_mode,DivModFast(pad_channels),DivModFast(out_width),DivModFast(out_width*out_height));
            return ppl::common::RC_SUCCESS;
        } else if(4 == sizeof(T) && 0 == (pad_channels & 3)) {
            int blockSize = 256;
            int gridSize = (num_elems + 1023) >> 10;
            ppl_cukernel_resize_bilinear_nhwc_opt<T,float4,4,2><<<gridSize, blockSize, 0, stream>>>(
                num_elems, h_scale, w_scale, roi_0, roi_1, roi_2, roi_3, extrapolation_value, channels,
                pad_channels, channels_per_piece, input, in_height, in_width, output, out_height,
                out_width, transform_mode,DivModFast(pad_channels),DivModFast(out_width),DivModFast(out_width*out_height));
            return ppl::common::RC_SUCCESS;
        }
#endif//OPT_RESIZE
        ppl_cukernel_resize_bilinear_nhwc<T><<<grid_size, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, roi_0, roi_1, roi_2, roi_3, extrapolation_value, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, transform_mode);
    } else if (inter_mode == 2) {
        ppl_cukernel_resize_cubic_nwhc<T><<<grid_size, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, cubic_coeff, transform_mode,exclude_outside);
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T>
ppl::common::RetCode ppl_resize_forward_nhwc_int8(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const T* input,
    const ppl::common::TensorShape* output_shape,
    T* output,
    bool scale_pre_set,
    float h_scale_pre,
    float w_scale_pre,
    int transform_mode,
    int inter_mode,
    float cubic_coeff,
    int nearest_mode,
    float in_scale,
    float out_scale)
{
    if (transform_mode == 5) return ppl::common::RC_UNSUPPORTED;
    int dim_count  = output_shape->GetDimCount();
    int out_height = output_shape->GetDim(2), out_width = 1;
    int in_height = input_shape->GetDim(2), in_width = 1;
    for (int it = 3; it < dim_count - 1; ++it) {
        out_height *= output_shape->GetDim(it);
        in_height *= input_shape->GetDim(it);
    }
    if (dim_count >= 4) {
        out_width    = output_shape->GetDim(dim_count - 1);
        in_width     = input_shape->GetDim(dim_count - 1);
    }
    int channels = output_shape->GetDim(1);
    int pad_channels = channels + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);

    float h_scale = 0.f, w_scale = 0.f;
    if (scale_pre_set) {
        h_scale = h_scale_pre;
        w_scale = w_scale_pre;
    } else {
        h_scale = hostComputeAreaScale(in_height, out_height, transform_mode);
        w_scale = hostComputeAreaScale(in_width, out_width, transform_mode);
    }
    int num_threads = out_height * out_width;
    int block_size  = 256; dim3 grid_size(1, 1, 1); int channels_per_piece = 1;
    GetNumBlocks(block_size, grid_size, channels_per_piece, num_threads, channels);
    grid_size.z =  output_shape->GetDim(0);
    if (inter_mode == 0) {
        if(nearest_mode == 0){
#ifdef OPT_RESIZE
            if(1 == condition(in_height, out_height, in_width, out_width, h_scale, w_scale) && transform_mode >=0 && transform_mode <= 3){
                int channelsN = pad_channels >> 4;
                int inSize = in_height * in_width;
                int num_elements = inSize * channelsN;
                block_size = 256;
                float scale = in_scale / out_scale;
                grid_size.x =  (num_elements + block_size - 1) / block_size;
                grid_size.y = 1; 
                ppl_cukernel_resize_nearest_nhwc_int8_opt_t3<<<grid_size, block_size, 0, stream>>>(
                    num_elements, out_height / in_height, out_width / in_width, channelsN, (const float4*)input,
                    in_height, in_width, (float4*)output, out_height, out_width, scale, DivModFast(channelsN), DivModFast(in_width)
                );
                return ppl::common::RC_SUCCESS;
            }
            if(0 == (pad_channels & 15)){
                float scale = in_scale / out_scale;
                int batch = output_shape->GetDim(0);
                int num_elems = out_height*out_width*pad_channels*batch;
                int blockSize = 256;
                int gridSize = (num_elems + 4095) >> 12;
                ppl_cukernel_resize_nearest_round_prefer_floor_nhwc_int8_opt<float4,4,16><<<gridSize, blockSize, 0, stream>>>(
                    num_elems, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, scale, transform_mode
                    ,DivModFast(pad_channels),DivModFast(out_width),DivModFast(out_width*out_height));
            }else{
                ppl_cukernel_resize_nearest_round_prefer_floor_nhwc_int8<T><<<grid_size, block_size, 0, stream>>>(
                    num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
            }
#else//!OPT_RESIZE
            ppl_cukernel_resize_nearest_round_prefer_floor_nhwc_int8<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
#endif//OPT_RESIZE
        } else if(nearest_mode == 1) {
            ppl_cukernel_resize_nearest_round_prefer_ceil_nhwc_int8<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
        } else if(nearest_mode == 3) {
            ppl_cukernel_resize_nearest_ceil_nhwc_int8<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
        } else{
#ifdef OPT_RESIZE
            if(3 == transform_mode || 2 == transform_mode){
                if(0 == (pad_channels&15) && condition_nhwc(in_height, out_height, in_width, out_width, h_scale, w_scale)){
                    int channelsN = pad_channels >> 4;
                    int inSize = in_height * in_width;
                    int num_elements = inSize * channelsN;
                    block_size = 256;
                    float scale = in_scale / out_scale;
                    grid_size.x =  (num_elements + block_size - 1) / block_size;
                    grid_size.y = 1; 
                    ppl_cukernel_resize_nearest_nhwc_int8_opt_t3<<<grid_size, block_size, 0, stream>>>(
                        num_elements, (out_height + in_height - 1) / in_height, (out_width + in_width - 1) / in_width, channelsN, (const float4*)input,
                        in_height, in_width, (float4*)output, out_height, out_width, scale, DivModFast(channelsN), DivModFast(in_width)
                    );
                    return ppl::common::RC_SUCCESS;        
                }
            }
            if(0 == (pad_channels & 15)){
                float scale = in_scale / out_scale;
                int batch = output_shape->GetDim(0);
                int num_elems = out_height*out_width*pad_channels*batch;
                int blockSize = 256;
                int gridSize = (num_elems + 4095) >> 12;
                ppl_cukernel_resize_nearest_nhwc_int8_opt<float4,4,16><<<gridSize, blockSize, 0, stream>>>(
                    num_elems, h_scale, w_scale, pad_channels, input, in_height, in_width, output, out_height, out_width, scale, transform_mode
                    ,DivModFast(pad_channels),DivModFast(out_width),DivModFast(out_width*out_height));
            } else {
                ppl_cukernel_resize_nearest_nhwc_int8<T><<<grid_size, block_size, 0, stream>>>(
                    num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
            }
#else//!OPT_RESIZE
            ppl_cukernel_resize_nearest_nhwc_int8<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
#endif//OPT_RESIZE
        }
    } else if (inter_mode == 1) {
#ifdef OPT_RESIZE
        if(0 == (pad_channels & 15)){
           float scale = in_scale / out_scale;
            int batch = output_shape->GetDim(0);
            if(h_scale < 1 && w_scale < 1) {
                int iTileWidth, iTileHeight;
                dim3 blockSize(1,1,1);
                blockSize.x = MIN((pad_channels >> 4), 8);
                blockSize.y = 4;
                blockSize.z = 256 / blockSize.x / blockSize.y;
                int blocky = (out_height + blockSize.z - 1) / blockSize.z;
                float temp = hostComputeSourceIndex(h_scale, blockSize.z, transform_mode);
                iTileHeight = MIN(temp + 4, in_height);
                temp = hostComputeSourceIndex(w_scale, blockSize.y, transform_mode);
                iTileWidth = MIN(temp + 4, in_width);
                dim3 gridSize((pad_channels + blockSize.x * 16 - 1) / (blockSize.x * 16),(out_width + 3)>>2,batch*blocky);
                ppl_cukernel_resize_bilinear_nhwc_int8_sm_opt<<<gridSize,blockSize,0,stream>>>(h_scale,w_scale,pad_channels,(const int8_t*)input,in_height,in_width,
                    (int8_t*)output,out_height,out_width,iTileHeight,iTileWidth,scale,transform_mode,DivModFast(blocky));
                return ppl::common::RC_SUCCESS;
            }
            int num_elems = out_height*out_width*pad_channels*batch;
            int blockSize = 256;
            int gridSize = (num_elems + 4095) >> 12;
            ppl_cukernel_resize_bilinear_nhwc_int8_opt<float4,4,16><<<gridSize,blockSize,0,stream>>>(num_elems,h_scale,w_scale,pad_channels,input,in_height,in_width,
                output,out_height,out_width,scale,transform_mode,DivModFast(pad_channels),DivModFast(out_width),DivModFast(out_width*out_height));
            return ppl::common::RC_SUCCESS;
        }
#endif//OPT_RESIZE
        ppl_cukernel_resize_bilinear_nhwc_int8<T><<<grid_size, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
    } else if (inter_mode == 2) {
        ppl_cukernel_resize_cubic_nhwc_int8<T><<<grid_size, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, channels, pad_channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, cubic_coeff, in_scale, out_scale, transform_mode);
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T>
ppl::common::RetCode ppl_resize_forward_int8(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const T* input,
    const ppl::common::TensorShape* output_shape,
    T* output,
    bool scale_pre_set,
    float h_scale_pre,
    float w_scale_pre,
    int transform_mode,
    int inter_mode,
    float cubic_coeff,
    int nearest_mode,
    float in_scale,
    float out_scale)
{
    if (transform_mode == 5) return ppl::common::RC_UNSUPPORTED;
    int dim_count  = output_shape->GetDimCount();
    int out_height = 1, out_width = 1;
    int in_height = 1, in_width = 1;
    for (int it = 2; it < dim_count - 1; ++it) {
        out_height *= output_shape->GetDim(it);
        in_height *= input_shape->GetDim(it);
    }
    out_width    = output_shape->GetDim(dim_count - 1);
    in_width     = input_shape->GetDim(dim_count - 1);
    int channels = output_shape->GetDim(0) * output_shape->GetDim(1);

    float h_scale = 0.f, w_scale = 0.f;
    if (scale_pre_set) {
        h_scale = h_scale_pre;
        w_scale = w_scale_pre;
    } else {
        h_scale = hostComputeAreaScale(in_height, out_height, transform_mode);
        w_scale = hostComputeAreaScale(in_width, out_width, transform_mode);
    }
    int num_threads = out_height * out_width;
    int block_size  = 256; dim3 grid_size(1, 1, 1); int channels_per_piece = 1;
    GetNumBlocks(block_size, grid_size, channels_per_piece, num_threads, channels);

    if (inter_mode == 0) {
        if(nearest_mode == 0){
#ifdef OPT_RESIZE
            float scale = in_scale / out_scale;
            if(1 == condition(in_height, out_height, in_width, out_width, h_scale, w_scale) && transform_mode >=0 && transform_mode <= 3) {
                int blockSize = 256;
                int64_t taskHeight = in_height * channels;
                if((out_width & 15)==0) {
                    int block_line = ((blockSize << 4) / in_width);
                    if(((block_line*in_width)&3)!=0){
                        block_line = block_line >> 1 << 1;
                    }
                    if(block_line >= 1){
                        int out_widthN = out_width>>4;
                        int64_t gridSize = (taskHeight + block_line - 1) / block_line;
                        ppl_cukernel_resize_nearest_int8_opt_2x<float4,16,4><<<gridSize, blockSize, 0, stream>>>(
                            (const int8_t*)input, taskHeight, in_width, (int8_t*)output, out_widthN, block_line, scale, DivModFast(out_widthN));
                        
                        return ppl::common::RC_SUCCESS;
                    }
                } else if(((out_width & 7) == 0)) {
                    int block_line = ((blockSize << 3) / in_width);
                    if(((block_line * in_width)&7)!=0){
                        block_line = block_line >> 1 << 1;
                    }
                    if(block_line >= 1){
                        int out_widthN = out_width >> 3;
                        int64_t gridSize = (taskHeight + block_line - 1) / block_line;
                        ppl_cukernel_resize_nearest_int8_opt_2x<float2,8,3><<<gridSize, blockSize, 0, stream>>>(
                            (const int8_t*)input, taskHeight, in_width, (int8_t*)output, out_widthN, block_line, scale, DivModFast(out_widthN));
                        return ppl::common::RC_SUCCESS;
                    }
                } else if((out_width & 3) == 0) {
                    int block_line = ((blockSize << 2) / in_width);
                    if(((block_line*in_width)&3)!=0){
                        block_line = block_line >> 2 << 2;
                    }
                    if(block_line >= 1){
                        int out_widthN = out_width >> 2;
                        int64_t gridSize = (taskHeight + block_line - 1) / block_line;
                        ppl_cukernel_resize_nearest_int8_opt_2x<int32_t,4,2><<<gridSize, blockSize, 0, stream>>>(
                            (const int8_t*)input, taskHeight, in_width, (int8_t*)output, out_widthN, block_line, scale, DivModFast(out_widthN));
                        return ppl::common::RC_SUCCESS;
                    }
                } else if((out_width&1)==0) {
                    int block_line = (4096 / in_width) >>1 <<1;
                    if(block_line >= 1){
                        int out_widthN = out_width >> 1;
                        int64_t gridSize = (taskHeight + block_line - 1) / block_line;
                        ppl_cukernel_resize_nearest_int8_opt_2x<int16_t,2,1><<<gridSize, blockSize, 0, stream>>>(
                             (const int8_t*)input, taskHeight, in_width, (int8_t*)output, out_widthN, block_line, scale, DivModFast(out_widthN));
                        return ppl::common::RC_SUCCESS;
                    }
                }
            }

            int inSize = in_height * in_width;
            int outSize = out_height * out_width;
            DivModFast out_width_fast(out_width);
            ppl_cukernel_resize_nearest_round_prefer_floor_int8_opt<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, inSize, output, out_height, out_width, outSize, out_width_fast, scale, transform_mode);

#else//OPT_RESIZE
            ppl_cukernel_resize_nearest_round_prefer_floor_int8<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
#endif//OPT_RESIZE
        }else if(nearest_mode == 1){
            ppl_cukernel_resize_nearest_round_prefer_ceil_int8<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
        }else if(nearest_mode == 3 ){
            ppl_cukernel_resize_nearest_ceil_int8<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
        }else{
#ifdef OPT_RESIZE
            if(3 == transform_mode && sizeof(T) == 1){
                float scale = in_scale / out_scale;
                if(1 == condition(in_height, out_height, in_width, out_width, h_scale, w_scale)){
                    int blockSize = 256;
                    int64_t taskHeight = in_height * channels;
                    if((out_width & 15)==0){
                        int block_line = ((blockSize << 4) / in_width);
                        if(((block_line*in_width)&3)!=0){
                            block_line = block_line >> 1 << 1;
                        }
                        if(block_line >= 1){
                            int out_widthN = out_width>>4;
                            int64_t gridSize = (taskHeight + block_line - 1) / block_line;
                            ppl_cukernel_resize_nearest_int8_opt_2x<float4,16,4><<<gridSize, blockSize, 0, stream>>>(
                                (const int8_t*)input, taskHeight, in_width, (int8_t*)output, out_widthN, block_line, scale, DivModFast(out_widthN));
                            return ppl::common::RC_SUCCESS;
                        }
                    } else if(((out_width & 7) == 0)){
                        int block_line = ((blockSize << 3) / in_width);
                        if(((block_line * in_width)&7)!=0){
                            block_line = block_line >> 1 << 1;
                        }
                        if(block_line >= 1){
                            int out_widthN = out_width >> 3;
                            int64_t gridSize = (taskHeight + block_line - 1) / block_line;
                            ppl_cukernel_resize_nearest_int8_opt_2x<float2,8,3><<<gridSize, blockSize, 0, stream>>>(
                                (const int8_t*)input, taskHeight, in_width, (int8_t*)output, out_widthN, block_line, scale, DivModFast(out_widthN));
                            return ppl::common::RC_SUCCESS;
                        }
                    } else if((out_width & 3) == 0){
                        int block_line = ((blockSize << 2) / in_width);
                        if(((block_line*in_width)&3)!=0){
                            block_line = block_line >> 2 << 2;
                        }
                        if(block_line >= 1){
                            int out_widthN = out_width >> 2;
                            int64_t gridSize = (taskHeight + block_line - 1) / block_line;
                            ppl_cukernel_resize_nearest_int8_opt_2x<int32_t,4,2><<<gridSize, blockSize, 0, stream>>>(
                                (const int8_t*)input, taskHeight, in_width, (int8_t*)output, out_widthN, block_line, scale, DivModFast(out_widthN));
                            return ppl::common::RC_SUCCESS;
                        }
                    } else if((out_width&1)==0){
                        int block_line = (4096 / in_width) >>1 <<1;
                        if(block_line >= 1){
                            int out_widthN = out_width >> 1;
                            int64_t gridSize = (taskHeight + block_line - 1) / block_line;
                            ppl_cukernel_resize_nearest_int8_opt_2x<int16_t,2,1><<<gridSize, blockSize, 0, stream>>>(
                                 (const int8_t*)input, taskHeight, in_width, (int8_t*)output, out_widthN, block_line, scale, DivModFast(out_widthN));
                            return ppl::common::RC_SUCCESS;
                        }
                    }
                } else if(2 == condition(in_height, out_height, in_width, out_width, h_scale, w_scale)) {
                    int blockSize = 256;
                    int block_line = (3276 / in_width);
                    if(((block_line*in_width)&3) !=0){
                        block_line = block_line >> 2 << 2;
                    }
                    if(block_line >= 1 && ((out_width & 3) == 0)){
                        int out_widthN = out_width >> 2;
                        int64_t taskHeight = in_height * channels;
                        int64_t gridSize = (taskHeight + block_line - 1) / block_line;
                        ppl_cukernel_resize_nearest_int8_opt_4x<<<gridSize, blockSize, 0, stream>>>(
                            (const int8_t*)input, taskHeight, in_width, (int8_t*)output, out_widthN, block_line, scale, DivModFast(out_widthN));
                        return ppl::common::RC_SUCCESS;
                    }
                } else if(3 == condition(in_height, out_height, in_width, out_width, h_scale, w_scale)) {
                    int blockSize = 256;
                    int block_line = 1820 / in_width;
                    if(((block_line*in_width)&3)!=0){
                        block_line = block_line >> 2 << 2;
                    }
                    if(block_line >= 1 && ((out_width & 3) == 0)){
                        int out_widthN = out_width>>2;
                        int64_t taskHeight = in_height * channels;
                        int64_t gridSize = (taskHeight + block_line - 1) / block_line;
                        ppl_cukernel_resize_nearest_int8_opt_8x<<<gridSize, blockSize, 0, stream>>>(
                            (const int8_t*)input, taskHeight, in_width, (int8_t*)output, out_widthN, block_line, scale, DivModFast(out_widthN));
                        return ppl::common::RC_SUCCESS;
                    }
                }
            }
            int inSize = in_height * in_width;
            int outSize = out_height * out_width;
            float scale = in_scale / out_scale;
            DivModFast out_width_fast(out_width);
            ppl_cukernel_resize_nearest_int8_opt<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, inSize, output, out_height, out_width, outSize, scale, out_width_fast,transform_mode);
#else//!OPT_RESIZE
            ppl_cukernel_resize_nearest_int8<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
#endif//OPT_RESIZE
        }
    } else if (inter_mode == 1) {
#ifdef OPT_RESIZE
        {
            DivModFast out_width_fast(out_width);
            ppl_cukernel_resize_bilinear_int8_opt<T><<<grid_size, block_size, 0, stream>>>(
                num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, out_width_fast, transform_mode);
        }
#else//!OPT_RESIZE
        ppl_cukernel_resize_bilinear_int8<T><<<grid_size, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, in_scale, out_scale, transform_mode);
#endif//OPT_RESIZE
    } else if (inter_mode == 2) {
        ppl_cukernel_resize_cubic_int8<T><<<grid_size, block_size, 0, stream>>>(
            num_threads, h_scale, w_scale, channels, channels_per_piece, input, in_height, in_width, output, out_height, out_width, cubic_coeff, in_scale, out_scale, transform_mode);
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAResizeForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    bool scale_pre_set,
    float h_scale,
    float w_scale,
    float roi_0,
    float roi_1,
    float roi_2,
    float roi_3,
    int exclude_outside,
    int transform_mode,
    int inter_mode,
    float cubic_coeff,
    int nearest_mode,
    float in_scale,
    float out_scale,
    float extrapolation_value)
{
    // special case, just copy
    int dim_count  = output_shape->GetDimCount();
    int out_height = output_shape->GetDim(2), out_width = 1;
    int in_height = input_shape->GetDim(2), in_width = 1;
    for (int it = 3; it < dim_count - 1; ++it) {
        out_height *= output_shape->GetDim(it);
        in_height *= input_shape->GetDim(it);
    }
    if (dim_count >= 4) {
        out_width    = output_shape->GetDim(dim_count - 1);
        in_width     = input_shape->GetDim(dim_count - 1);
    }
    if (out_height == in_height && out_width == in_width) {
        cudaMemcpyAsync(output, input, input_shape->CalcBytesIncludingPadding(), cudaMemcpyDeviceToDevice, stream);
    }
    // common case
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            return ppl_resize_forward<half>(stream, input_shape, (const half*)input, output_shape, (half*)output, scale_pre_set, h_scale, w_scale,roi_0,roi_1,roi_2,roi_3,extrapolation_value, transform_mode, inter_mode, cubic_coeff,nearest_mode,exclude_outside);
        } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
            return ppl_resize_forward_nhwc<half>(stream, input_shape, (const half*)input, output_shape, (half*)output, scale_pre_set, h_scale, w_scale,roi_0,roi_1,roi_2,roi_3,extrapolation_value,transform_mode, inter_mode, cubic_coeff,nearest_mode,exclude_outside);
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        if(output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            return ppl_resize_forward<float>(stream, input_shape, (const float*)input, output_shape, (float*)output, scale_pre_set, h_scale, w_scale, roi_0,roi_1,roi_2,roi_3,extrapolation_value,transform_mode, inter_mode, cubic_coeff,nearest_mode,exclude_outside);
        } else if(output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC || output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 || output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
            return ppl_resize_forward_nhwc<float>(stream, input_shape, (const float*)input, output_shape, (float*)output, scale_pre_set, h_scale, w_scale,roi_0,roi_1,roi_2,roi_3,extrapolation_value,transform_mode, inter_mode, cubic_coeff,nearest_mode,exclude_outside);
        }
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            return ppl_resize_forward_int8<int8_t>(stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output, scale_pre_set, h_scale, w_scale, transform_mode, inter_mode, cubic_coeff, nearest_mode, in_scale, out_scale);
        } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
            return ppl_resize_forward_nhwc_int8<int8_t>(stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output, scale_pre_set, h_scale, w_scale, transform_mode, inter_mode, cubic_coeff, nearest_mode, in_scale, out_scale);
        } else{
            return ppl::common::RC_UNSUPPORTED;    
        }
    }else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

// void printtoHost(float *dev,int batchsize,int channel,int height,int width, int format)
// {
//     float *cpu = nullptr;
//     cudaMallocHost((void**)&cpu,batchsize*channel*width*height*sizeof(float));
//     cudaMemcpy(cpu, dev,sizeof(float)*batchsize*channel*width*height, cudaMemcpyDeviceToHost);
//     if(format == 1){
//         for(int b = 0; b < batchsize; b++){
//             for(int c = 0; c < channel; c++){
//                 for(int h = 0; h < height; h++){
//                     for(int w = 0; w < width; w++){
//                         printf("%f,",cpu[b*channel*height*width + c*height*width + h*width + w]);
//                     }
//                     printf("\n");
//                 }
//                 printf("=======next channel=====\n");
//             }
//             printf("=======next batch=====\n");
//         }
//     }
//     else{
//         for(int b = 0; b < batchsize; b++){
//             for(int c = 0; c < channel; c++){
//                 for(int h = 0; h < height; h++){
//                     for(int w = 0; w < width; w++){
//                         printf("%f,",cpu[b*channel*height*width + (h*width + w)*channel + c]);
//                     }
//                     printf("\n");
//                 }
//                 printf("=======next channel=====\n");
//             }
//             printf("=======next batch=====\n");
//         }
//     }
//     cudaFreeHost((void*)cpu);
// }
