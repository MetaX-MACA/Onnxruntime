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

#include "cudakernel/nn/global_pooling_ave.h"
#include "ppl/common/types.h"
#include <cuda_fp16.h>
#ifdef PPLNN_USE_MACA
#define GAP_OPT
#endif//PPLNN_USE_MACA

#ifdef GAP_OPT
//FIXME:
//global average pooling nhwc在int8_t类型和half使用了两种优化方式
//这是由于多个时间段优化，互相之间并未更好的借鉴的结果
//int8_t的优化集中在C2N, line, dbuf以及32N上，尽可能一个block内完成channel维度的计算
//half的优化，会尝试分割pad_channels，CS表示将channel维度进行拆分，意味着每个block
//的数据并不连续，因此需要保证每个循环能够有足够的数据；HWS表示将HW维度进行分割，为了
//更好的进行reduce计算，只会考虑HW中有2^N余数的情况。CS=1或者HWS=1的时候会有特定函数
//进行优化。
//当前优化还存在一些问题：
//1. 当batch size比较小的时候，会造成很多优化失效
//2. 部分shape情况下，不管是int8或者half都可能进行负优化
//3. get_cs_and_hws函数过于复杂，且各种判断条件是从100多条数据中提取而成
//4. int8 32N优化中，过于强调寄存器的使用，而忽略了线程数量的增加，可能减少并行性
template <uint32_t block_size, typename T>
__device__ __forceinline__ void WarpReduceAdd(volatile T *sdata, uint32_t tid) {
    if (block_size >= 128) sdata[tid] += sdata[tid + 64];
    if (block_size >= 64) sdata[tid] += sdata[tid + 32];
    if (block_size >= 32) sdata[tid] += sdata[tid + 16];
    if (block_size >= 16) sdata[tid] += sdata[tid + 8];
    if (block_size >= 8) sdata[tid] += sdata[tid + 4];
    if (block_size >= 4) sdata[tid] += sdata[tid + 2];
    if (block_size >= 2) sdata[tid] += sdata[tid + 1];
}

template<int PAD_CHANNELS>
__launch_bounds__(1024)
__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_Opt_C2N(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    float scale
) {
    const int8_t* ptr_block_input = input + blockIdx.x * PAD_CHANNELS * HW;
    int8_t* ptr_block_output = output + blockIdx.x * PAD_CHANNELS;
    const float4 *ptr_input = (const float4*)(ptr_block_input);

    //Every thread will read 4 bytes per loop, aka process 16 images
    //A block have 512 threads, so, each loop, a block can handle
    //512 * 16 pixels = 4 channels of 16 images
    __shared__ int32_t results[PAD_CHANNELS]; //store final result

    union {
        float4 packed;
        int8_t unpacked[16];
    } mediate;
    int32_t temp_result[16] = {0};

    for (int i = threadIdx.x; i < HW * PAD_CHANNELS / sizeof(float4); i+=blockDim.x) {
        mediate.packed = ptr_input[i];
        for (int j = 0; j < 16; j++) temp_result[j] += mediate.unpacked[j];
    }

    for (int i = threadIdx.x; i < PAD_CHANNELS; i+= blockDim.x) {
        results[i] = 0;
    }

    __syncthreads();

    //Add thread 0, thread 128, thread 256 and thread 384
    //first loop, thread 0 write 0, thread 128 write 1, thread 256 write 2, thread 384 write 3
    //second loop, thread 0 write 1, thread 128 write 2, thread 256 write 3, thread 384 write 4
    //...
    //sixteenth loop, thread 0 write 15, thread 128 write 0, thread 256 write 1, thread 384 write 2

    //Add thread 1, thread 129, thread 257 and thread 385
    int bid = threadIdx.x / (PAD_CHANNELS / 16);
    int b_start = (threadIdx.x - (bid * (PAD_CHANNELS / 16))) << 4;
    for (int i = 0; i < 16; i++) {
        int write_idx = b_start + (i + bid) % 16;
        results[write_idx] += temp_result[(i + bid) % 16];
        __syncthreads();
    }
    float4* op = (float4*)ptr_block_output;

    for (int i = threadIdx.x; i < (PAD_CHANNELS / 16); i+=blockDim.x) {
        for (int j = 0; j < 16; j++) {
            int32_t temp = round(float(results[(i<<4) + j]) / HW * scale);
            temp = min(temp,127);
            temp = max(temp,-128);
            mediate.unpacked[j] = temp;
        }
        op[i] = mediate.packed;
    }
}

template<typename T>
__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_Opt_line(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    int line_pb,
    int channelCount,
    float scale
)
{
    constexpr int times = sizeof(T);
    constexpr int SM_SIZE = 8192;
    constexpr int block_size = SM_SIZE / times;
    int inputCount = channelCount * HW;
    int offset = line_pb * channelCount;
    int tid = threadIdx.x;
    int height = HW - line_pb + 1;
    int h = line_pb;
    int stride = pad_channels << 1;
    __shared__ union{
        int8_t m1[SM_SIZE];
        T m2[block_size];
    } sm_input;

    __shared__ int sum_buffer[2048];

    const int8_t* ptr_block_input = input + blockIdx.x * pad_channels * HW;
    int8_t* ptr_block_output = output + blockIdx.x * pad_channels;
    const T* ptr_input = (const T*)ptr_block_input;

    for(int i = tid; i < block_size; i += blockDim.x){
        if(i < inputCount){
            sm_input.m2[i] = *(ptr_input + i);
        }
    }
    __syncthreads();
    ptr_input += offset;

    for(int i = tid; i < pad_channels; i += blockDim.x){
        sum_buffer[i] = sm_input.m1[i] + sm_input.m1[pad_channels + i];
    }
    int offset1 = stride;
    for(int j = 2; j < line_pb; j += 2){
        for(int i = tid; i < pad_channels; i += blockDim.x){
            sum_buffer[i] += sm_input.m1[offset1 + i] + sm_input.m1[offset1 + pad_channels + i];
        }
        offset1 += stride;
    }
    __syncthreads();
    for(; h < height; h += line_pb){
        for(int i = tid; i < block_size; i += blockDim.x){
            if(i < inputCount){
                sm_input.m2[i] = *(ptr_input + i);
            }
        }
        __syncthreads();
        ptr_input += offset;
        int n = 0;
        for(int j = 0; j < line_pb; j+=2){
            for(int i = tid; i < pad_channels; i += blockDim.x){
                sum_buffer[i] += sm_input.m1[n + pad_channels + i] + sm_input.m1[n + i];
            }
            n += stride;
        }
        __syncthreads();
    }

    for(; h < HW; h += line_pb){
        for(int i = tid; i < block_size; i += blockDim.x){
            if(i < inputCount){
                sm_input.m2[i] = *(ptr_input + i);
            }
        }
        __syncthreads();
        ptr_input += offset;
        for(int j = 0; j < line_pb; j++){
            if(h + j < HW){
                for(int i = tid; i < pad_channels; i += blockDim.x){
                    sum_buffer[i] += sm_input.m1[j*pad_channels + i];
                }
            }
        }
        __syncthreads();
    }

    for(int i = tid; i < pad_channels; i += blockDim.x){
        int32_t temp = round(float(sum_buffer[i]) / HW * scale);
        temp = min(temp,127);
        temp = max(temp,-128);
        sm_input.m1[i] = temp;
    }
    __syncthreads();
    for(int i = tid; i < channelCount; i += blockDim.x){
        *((T*)ptr_block_output + i) = *((T*)sm_input.m1 + i);
    }
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_Opt_dbuf(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    int lines_per_loop,
    float scale) {
    using T = float4;
    using MT = uint64_t;
    constexpr int IBUF_SZ = 8192;
    constexpr int SZ = IBUF_SZ/2;

    const T* block_input = (T*)(input + pad_channels * HW * blockIdx.x);
    __shared__ union {
        int8_t unpacked[IBUF_SZ];
        T packed[IBUF_SZ/sizeof(T)];
        MT mediate[IBUF_SZ/sizeof(MT)];
    } ibuf; //8K
    __shared__ int sum_buffer[SZ];  //16K

    for (int i = threadIdx.x; i < SZ; i += blockDim.x) {
        sum_buffer[i] = 0;
    }
    int line = 0;
    union {
        uint64_t packed;
        int8_t unpacked[8];
    } i0, i1;
    const int stride_lines_first_loop = lines_per_loop * 2;
    const int stride_first_loop = stride_lines_first_loop * pad_channels;
    const int read_pack_size = stride_first_loop / sizeof(T);
    for (int l = 0 ; l + stride_lines_first_loop <= HW; l += stride_lines_first_loop) {
        if (threadIdx.x < read_pack_size)  {
            //VERY IMPORTANT:
            //A const qualifier is needed for tmp
            //or the compiler will use stp to store tmp
            const float4 tmp = block_input[threadIdx.x];
            block_input += read_pack_size;
            line += stride_lines_first_loop;
            //Before arrive of tmp, do some calculations
            int i0_start = threadIdx.x;
            int i1_start = threadIdx.x + read_pack_size;
            int res[8];
            int res2[8];

            ibuf.packed[threadIdx.x] = tmp;

            __syncthreads();
            //Do a reduce in ibuf.mediate, which means split the read data into two parts
            //and add them
            i0.packed = ibuf.mediate[i0_start];
            i1.packed = ibuf.mediate[i1_start];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                res[i] = sum_buffer[i + threadIdx.x * 8];
                res2[i] = i0.unpacked[i] + i1.unpacked[i];
                res[i] += res2[i];
            }
            //Following shared buffer store will translate into a sts_b128
            #pragma unroll
            for (int i = 0; i < 8; i++) sum_buffer[threadIdx.x*8 + i] = res[i];
        } else {
            block_input += read_pack_size;
            line += stride_lines_first_loop;
        }
    }

    //if there are more lines than lines_per_loop, we should process it
    if (HW-line >= lines_per_loop) {
        if (threadIdx.x < lines_per_loop * pad_channels / sizeof(T))
            ibuf.packed[threadIdx.x] = block_input[threadIdx.x];
        __syncthreads();
        for (int i = threadIdx.x; i < lines_per_loop * pad_channels; i+=blockDim.x)
            sum_buffer[i] += ibuf.unpacked[i];
        line += lines_per_loop;
        block_input += lines_per_loop * pad_channels / sizeof(T);
    }

    const int rest_data_size = HW * pad_channels - line * pad_channels;
    for (int i = threadIdx.x; i < rest_data_size / sizeof(T); i+=blockDim.x) {
        ibuf.packed[i] = block_input[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rest_data_size; i+=blockDim.x) {
        sum_buffer[i] += ibuf.unpacked[i];
    }
    __syncthreads();

    //Reduce on all channels into the first channel
    for (int i = lines_per_loop; i > 1; ) {
        if ( i % 2 == 1) {
            for (int j = threadIdx.x; j < pad_channels; j+=blockDim.x) {
                sum_buffer[j] += sum_buffer[j+(i-1)*pad_channels];
            }
            __syncthreads();
            i -= 1;
        }
        int stride = i / 2 * pad_channels;
        for (int j = threadIdx.x; j < stride; j+=blockDim.x)
            sum_buffer[j] += sum_buffer[j+stride];
        __syncthreads();
        i /= 2;
    }

    for (int tid = threadIdx.x; tid < pad_channels; tid+=blockDim.x) {
        int32_t temp = round(float(sum_buffer[tid]) / HW * scale);
        temp = min(temp, 127);
        temp = max(temp,-128);
        output[pad_channels  * blockIdx.x + tid] = temp;
    }
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_Opt(
    const int64_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    float scale)
{
    int c        = blockIdx.y * blockDim.y + threadIdx.y;
    if(c >= pad_channels)
        return;
    int b_offset = blockIdx.z * pad_channels;
    const int64_t* ptr_input = input + b_offset * HW + c;
    int8_t* ptr_output = output + ((b_offset + c)<<3);
    int32_t res[8];
    #pragma unroll 8
    for(int i = 0; i < 8; i++)
    {
        res[i] = 0;
    }

    int64_t ival = 0;
    int8_t *ptr_ival = (int8_t*)&ival;
    for(int i = threadIdx.x; i < HW; i += blockDim.x){
        ival = *(ptr_input + i * pad_channels);
        #pragma unroll 8
        for(int j = 0; j < 8; j++)
        {
            res[j] += ptr_ival[j];
        }
    }
    //hw x channel
    __shared__ int32_t sum_buffer[64][33];
    #pragma unroll 8
    for(int i = 0; i < 8; i++)
    {
        sum_buffer[(threadIdx.y<<3) + i][threadIdx.x] = res[i];
    }
    __syncthreads();

    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3)],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 1],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 2],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 3],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 4],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 5],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 6],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 7],threadIdx.x);
    __syncthreads();

    if(threadIdx.x < 8)
    {
        int32_t temp = round(float(sum_buffer[(threadIdx.y<<3) + threadIdx.x][0]) / HW * scale);
        temp = min(temp, 127);
        temp = max(temp,-128);
        ptr_output[threadIdx.x] = temp;
    }
}

template<int N>  // 4
__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_32N(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int DHW,
    float scale
) {
    int bid = blockIdx.x;
    const int8_t* input_start = input + bid * DHW * pad_channels;
    int8_t* output_start = output + bid * pad_channels;
    int sum_buffer[N] = {0};
    int stride = blockDim.x;

    for (int l = 0; l < DHW; l++) {
        #pragma unroll N
        for (int i = 0; i < N; i++) {
            sum_buffer[i] += input_start[i * stride + threadIdx.x];
        }
        input_start += pad_channels;
    }
    #pragma unroll N
    for (int i = 0; i < N; i++) {
        int32_t temp = round(float(sum_buffer[i]) / DHW * scale);
        temp = min(temp,127);
        temp = max(temp,-128);
        output_start[i*stride+threadIdx.x] = temp;
    }
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_SPLIT_x4(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    float scale
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int sid = threadIdx.y;
    int BLOCK = blockDim.x;
    const int8_t* input_start = input + bid * HW * pad_channels + (sid * BLOCK + tid) * 4;
    int8_t* output_start = output + bid * pad_channels + (sid * BLOCK + tid) * 4;
    int sum_buffer[4] = {0};
    union {
        int8_t unpacked[4];
        uint32_t packed;
    } package;

    for (int l = 0; l < HW; l++) {
        package.packed = *((const uint32_t*)input_start);
        for (int i = 0; i < 4; i++)
            sum_buffer[i] += package.unpacked[i];
        input_start += pad_channels;
    }

    for (int i = 0; i < 4; i++) {
        int32_t temp = round(float(sum_buffer[i]) / HW * scale);
        temp = min(temp,127);
        temp = max(temp,-128);
        package.unpacked[i] = temp;
    }
    *((uint32_t*)output_start) = package.packed;
}
#endif//GAP_OPT

template<typename T>
__global__ void ppl_cukernel_pooling_ave_global_shuffle(
      const T* input,
      T* output,
      int batch,
      int pad_channels,
      int DHW)
{
    int c  = (blockIdx.y * blockDim.y + threadIdx.y);
    int bc = blockIdx.z * pad_channels + c;
    if (c >= pad_channels)
        return;

    T res = T(0);
    for (int i = 0; i < DHW; i += 64) {
        bool pred0 = i + threadIdx.x * 2 + 0 < DHW;
        bool pred1 = i + threadIdx.x * 2 + 1 < DHW;
        T ival0 = pred0 ? input[bc * DHW + 2 * threadIdx.x + i + 0] : T(0);
        T ival1 = pred1 ? input[bc * DHW + 2 * threadIdx.x + i + 1] : T(0);
        T val = ival0 + ival1;
        res = res + val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        T val = __shfl_down_sync(0xffffffff, res, offset);
#else
        T val = __shfl_down(res, offset);
#endif
        res = res + val;
    }

    // store output
    if (threadIdx.x == 0)
        output[bc] = res / DHW;
}


__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8(
      const int8_t* input,
      int8_t* output,
      int batch,
      int pad_channels,
      int HW, float in_scale, float out_scale)
{
    int c = (blockIdx.y * blockDim.y + threadIdx.y);
    int bc = blockIdx.z * pad_channels + c;
    if (c >= pad_channels) return;

    int32_t res = int32_t(0);
    for (int i = 0; i < HW; i += 64) {
        bool pred0 = i + threadIdx.x * 2 + 0 < HW;
        bool pred1 = i + threadIdx.x * 2 + 1 < HW;
        int8_t ival0 = pred0 ? input[bc * HW + 2 * threadIdx.x + i + 0] : int8_t(0);
        int8_t ival1 = pred1 ? input[bc * HW + 2 * threadIdx.x + i + 1] : int8_t(0);
        int32_t val = ival0 + ival1;
        res = res + val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        int32_t val = __shfl_down_sync(0xffffffff, res, offset);
#else
        int32_t val = __shfl_down(res, offset);
#endif
        res = res + val;
    }

    // store output
    if (threadIdx.x == 0) {
        int64_t temp= round((float(res) / HW * in_scale) / out_scale);
        if (temp > 127)         temp = 127;
        else if (temp < -128)   temp = -128;
        output[bc] = temp;
    }
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_opt(
      const int8_t* input,
      int8_t* output,
      int batch,
      int pad_channels,
      int HW, float scale)
{
    int c = (blockIdx.y * blockDim.y + threadIdx.y);
    int bc = blockIdx.z * pad_channels + c;
    if (c >= pad_channels) return;

    int32_t res = int32_t(0);
    for (int i = threadIdx.x; i < HW; i+=32)
    {
        int8_t value = input[bc * HW + i];
        res += value;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        int32_t val = __shfl_down_sync(0xffffffff, res, offset);
#else
        int32_t val = __shfl_down(res, offset);
#endif
        res = res + val;
    }

    // store output
    if (threadIdx.x == 0) {
        // int64_t temp= round(float(res) / HW * in_scale) / out_scale);
        int64_t temp= round(float(res) * scale);
        if (temp > 127)         temp = 127;
        else if (temp < -128)   temp = -128;
        output[bc] = temp;
    }
}

template <typename srcT, int Iter>
__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_HW_even_opt(
      const int8_t* input,
      int8_t* output,
      int batch,
      int pad_channels,
      int HW, float scale)
{
    int c = (blockIdx.y * blockDim.y + threadIdx.y);
    int bc = blockIdx.z * pad_channels + c;
    if (c >= pad_channels) return;

    int32_t res = int32_t(0);
    const srcT *input_srcT = (const srcT *)(input + bc * HW);
    union{
        int8_t x[Iter];
        srcT y;
    } data;

    for (int i = threadIdx.x; i < (HW / Iter) ;i+=32){
        data.y = input_srcT[i];
        #pragma unroll
        for (int it =0;it<Iter;it++)
        {
            res += data.x[it];
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        int32_t val = __shfl_down_sync(0xffffffff, res, offset);
#else
        int32_t val = __shfl_down(res, offset);
#endif
        res = res + val;
    }

    // store output
    if (threadIdx.x == 0) {
        // int64_t temp= round(float(res) / HW * in_scale) / out_scale);
        int64_t temp= round(float(res) * scale);
        if (temp > 127)         temp = 127;
        else if (temp < -128)   temp = -128;
        output[bc] = temp;
    }
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_2image(
      const int8_t* input,
      int8_t* output,
      int batch,
      int pad_channels,
      int HW, float scale)
{
    int bc = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    if (bc >= pad_channels * batch) return;

    int32_t res = int32_t(0);

    // 64 thread read two image
    int32_t coeff = threadIdx.x >= 32 ? 1: 0;
    int64_t start_pos = coeff * (HW & (~0x1));
    const int16_t *input_16 = (const int16_t *)(input + bc * HW + start_pos);
    union
    {
        int8_t x[2];
        int16_t y;
    } data;

    for (int i = threadIdx.x; i < (HW + 1) / 2 + 32 *coeff ; i += 32) {
        data.y = input_16[i - 32 * coeff];
        res += data.x[0] + data.x[1];
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        int32_t val = __shfl_down_sync(0xffffffff, res, offset);
#else
        int32_t val = __shfl_down(res, offset);
#endif
        res = res + val;
    }

    // store output
    if (threadIdx.x == 0 || threadIdx.x == 32) {
        res -= (HW & 0x1) ? input[bc * HW + HW - coeff] : 0;
        int64_t temp= round(float(res) * scale);
        if (temp > 127)         temp = 127;
        else if (temp < -128)   temp = -128;
        output[bc + coeff] = temp;
    }
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_half(
    const half* input,
    half* output,
    int batch,
    int pad_channels,
    int HW)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c  = (blockIdx.y * blockDim.y + threadIdx.y);
    int bc = blockIdx.z * pad_channels + c;
    if (c >= pad_channels)
        return;

    float res = 0.f;
    for (int i = 0; i < HW; i += 64) {
        bool pred0 = i + threadIdx.x * 2 + 0 < HW;
        bool pred1 = i + threadIdx.x * 2 + 1 < HW;
        half ival0 = pred0 ? input[bc * HW + 2 * threadIdx.x + i + 0] : half(0);
        half ival1 = pred1 ? input[bc * HW + 2 * threadIdx.x + i + 1] : half(0);
        float val  = __half2float(__hadd(ival0, ival1));
        res        += val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        float val = __shfl_down_sync(0xffffffff, res, offset);
#else
        float val = __shfl_down(res, offset);
#endif
        res +=  val;
    }

    // store output
    if (threadIdx.x == 0)
        output[bc] = __float2half(res / HW);
#endif
}

static __device__ float2 __f2add(float2 val0, float2 val1) {
    float2 res{0.f, 0.f};
    res.x = val0.x + val1.x;
    res.y = val0.y + val1.y;
    return res;
}

template<int TILE_C, int TILE_HW>
__global__ void ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int HW)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c        = blockIdx.x * blockDim.x + threadIdx.x;
    int b_offset = blockIdx.z * pad_channels;
    if (c >= pad_channels)
        return;

    float2 res = float2{0.f, 0.f};
    // main loop
    for (int i = threadIdx.y; i < HW; i += blockDim.y) {
        half2 ival = input[b_offset * HW + i * pad_channels + c];
        res        = __f2add(res, __half22float2(ival));
    }
    __shared__ float2 sum_buffer[TILE_HW][TILE_C];
    sum_buffer[threadIdx.y][threadIdx.x] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i = (i >> 1)) {
        if (threadIdx.y < i) {
            float2 res                           = sum_buffer[threadIdx.y + i][threadIdx.x];
            res                                  = __f2add(res, sum_buffer[threadIdx.y][threadIdx.x]);
            sum_buffer[threadIdx.y][threadIdx.x] = res;
            __syncthreads();
        }
    }
    // store output
    if (threadIdx.y == 0) {
        float2 res = sum_buffer[threadIdx.y][threadIdx.x];
        res.x = res.x / HW;
        res.y = res.y / HW;
        output[b_offset + c] = __float22half2_rn(res);
    }
#endif
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_atomic(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int HW)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 10 // atomicAdd half2 supported on cuda10
    int c        = blockIdx.x * blockDim.x + threadIdx.x;
    int b_offset = blockIdx.z * pad_channels;
    if (c >= pad_channels)
        return;

    float2 res = float2{0.f, 0.f};
    // main loop
    for (int i = threadIdx.y + blockDim.y * blockIdx.y;
                i < HW; i += blockDim.y * gridDim.y) {
        half2 ival = input[b_offset * HW + i * pad_channels + c];
        res        = __f2add(res, __half22float2(ival));
    }
    __shared__ float2 sum_buffer[32][8];
    sum_buffer[threadIdx.y][threadIdx.x] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i = (i >> 1)) {
        if (threadIdx.y < i) {
            float2 res                           = sum_buffer[threadIdx.y + i][threadIdx.x];
            res                                  = __f2add(res, sum_buffer[threadIdx.y][threadIdx.x]);
            sum_buffer[threadIdx.y][threadIdx.x] = res;
            __syncthreads();
        }
    }
    // store output
    if (threadIdx.y == 0) {
        float2 res = sum_buffer[threadIdx.y][threadIdx.x];
        res.x = res.x / HW;
        res.y = res.y / HW;
        atomicAdd(&output[b_offset + c], __float22half2_rn(res));
    }
#endif
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    float in_scale,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c        = blockIdx.x * blockDim.x + threadIdx.x;
    int b_offset = blockIdx.z * pad_channels;
    if (c >= pad_channels)
        return;

    int64_t res = 0;
    // main loop
    for (int i = threadIdx.y; i < HW; i += blockDim.y) {
        int8_t ival = input[b_offset * HW + i * pad_channels + c];
        res = res + ival;
    }
    __shared__ int64_t sum_buffer[8][32];
    sum_buffer[threadIdx.y][threadIdx.x] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i = (i >> 1)) {
        if (threadIdx.y < i) {
            int64_t res                            = sum_buffer[threadIdx.y + i][threadIdx.x];
            res                                   = res + sum_buffer[threadIdx.y][threadIdx.x];
            sum_buffer[threadIdx.y][threadIdx.x] = res;
            __syncthreads();
        }
    }
    // store output
    if (threadIdx.y == 0) {
        int64_t temp = round((float(sum_buffer[threadIdx.y][threadIdx.x]) / HW * in_scale) / out_scale);
        if (temp > 127)         temp = 127;
        else if (temp < -128)   temp = -128;
        output[b_offset + c] = temp;
    }

#endif
}

#ifdef GAP_OPT
//CS != 1 and HWS != 1
//CHANNEL 被平分为CS份，HW倍平分为HWS份
__launch_bounds__(1024)
__global__ void ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_S(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int DHW,
    float scale
) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int CS = gridDim.y, DHWS = blockDim.y;

    const int channels_per_tiny_block = DHW / DHWS;
    //每个loop读取的数据量, 即 C/CS个数据
    const int read_size_per_loop = pad_channels / CS;
    //每次读完C/CS数据后，需要跳多少地址
    const int stride = pad_channels;
    float2 sum_buffer;
    sum_buffer.x = 0;
    sum_buffer.y = 0;
    //bx表示现在所处的batch， by, 表示现在所处的tiny channel索引, ty *
    int src_offset =  bx * pad_channels * DHW + ty * channels_per_tiny_block * pad_channels + by * read_size_per_loop;
    for (int i = 0; i < channels_per_tiny_block; i++) {
        //for (int t = tx; t < read_size_per_loop; t += blockDim.x) {
            half2 val = input[src_offset+tx];
            sum_buffer = __f2add(sum_buffer, __half22float2(val));
        //}
        src_offset += stride;
    }

    __shared__ float2 all_data[1024];
    //Every thread store sum_buffer to all_data
    int tid = tx + ty * blockDim.x;
    all_data[tid] = sum_buffer;
    __syncthreads();
    int s = DHWS;
    while (ty < s / 2 && s > 1) {
        auto r0 = all_data[tx + ty * blockDim.x];
        auto rs = all_data[tx + (ty + s / 2) * blockDim.x];
        r0 = __f2add(r0, rs);
        all_data[tx + ty * blockDim.x] = r0;
        __syncthreads();
        s /= 2;
    }

    const int out_offset = bx * pad_channels + by * read_size_per_loop;
    if (ty == 0) {
        float2 res = all_data[tx];
        res.x *= scale;
        res.y *= scale;
        output[out_offset + tx] = __float22half2_rn(res);
    }
}

//CS=1的时候，block读取数据是连续的
__launch_bounds__(1024)
__global__ void ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_S_CS1(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int HW,
    float scale
) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x;
    int HWS = blockDim.y;

    const int channels_per_tiny_block = HW / HWS;
    //每个loop读取的数据量, 即 C/CS个数据
    const int read_size_per_loop = pad_channels;
    float2 sum_buffer;
    sum_buffer.x = 0;
    sum_buffer.y = 0;
    //blockDim.x == pad_channels
    int src_offset =  bx * pad_channels * HW + ty * channels_per_tiny_block * pad_channels;
    for (int i = tx; i < channels_per_tiny_block * read_size_per_loop; i+=blockDim.x) {
        //for (int t = tx; t < read_size_per_loop; t += blockDim.x) {
            half2 val = input[src_offset+i];
            sum_buffer = __f2add(sum_buffer, __half22float2(val));
        //}
        //src_offset += stride;
    }

    __shared__ float2 all_data[1024];
    //Every thread store sum_buffer to all_data
    int tid = tx + ty * blockDim.x;
    all_data[tid] = sum_buffer;
    __syncthreads();
    int s = HWS;
    while (ty < s / 2 && s > 1) {
        auto r0 = all_data[tx + ty * blockDim.x];
        auto rs = all_data[tx + (ty + s / 2) * blockDim.x];
        r0 = __f2add(r0, rs);
        all_data[tx + ty * blockDim.x] = r0;
        __syncthreads();
        s /= 2;
    }

    const int out_offset = bx * pad_channels;
    if (ty == 0) {
        float2 res = all_data[tx];
        res.x *= scale;
        res.y *= scale;
        output[out_offset + tx] = __float22half2_rn(res);
    }
}

//HWS=1的时候，不需要shared buffer
__launch_bounds__(1024)
__global__ void ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_S_HWS1(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int HW,
    float scale
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int CS = gridDim.y;

    const int channels_per_tiny_block = HW;
    //每个loop读取的数据量, 即 C/CS个数据
    const int read_size_per_loop = pad_channels / CS;
    //每次读完C/CS数据后，需要跳多少地址
    const int stride = pad_channels;
    float2 sum_buffer;
    sum_buffer.x = 0;
    sum_buffer.y = 0;
    //bx表示现在所处的batch， by, 表示现在所处的tiny channel索引, ty *
    int src_offset =  bx * pad_channels * HW + by * read_size_per_loop;
    for (int i = 0; i < channels_per_tiny_block; i++) {
        //for (int t = tx; t < read_size_per_loop; t += blockDim.x) {
            half2 val = input[src_offset+tx];
            sum_buffer = __f2add(sum_buffer, __half22float2(val));
        //}
        src_offset += stride;
    }

    const int out_offset = bx * pad_channels + by * read_size_per_loop;
    sum_buffer.x *= scale;
    sum_buffer.y *= scale;
    output[out_offset + tx] = __float22half2_rn(sum_buffer);
}

//CS != 1 and HWS != 1
//CHANNEL 被平分为CS份，HW倍平分为HWS份
__launch_bounds__(1024)
__global__ void ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_S_CS1_HWS1(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int HW,
    float scale
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    //每个loop读取的数据量, 即 C/CS个数据
    const int read_size_per_loop = pad_channels;
    float2 sum_buffer;
    sum_buffer.x = 0;
    sum_buffer.y = 0;
    //bx表示现在所处的batch， by, 表示现在所处的tiny channel索引, ty *
    int src_offset =  bx * pad_channels * HW;
    for (int i = tx; i < HW * read_size_per_loop; i += blockDim.x) {
        //for (int t = tx; t < read_size_per_loop; t += blockDim.x) {
            half2 val = input[src_offset + i];
            sum_buffer = __f2add(sum_buffer, __half22float2(val));
        //}
        //src_offset += stride;
    }

    const int out_offset = bx * pad_channels;
    sum_buffer.x *= scale;
    sum_buffer.y *= scale;
    output[out_offset + tx] = __float22half2_rn(sum_buffer);
}
#endif

#ifdef GAP_OPT
int max_diploit_of_e2n(int v) {
    int n = 0;
    while (v % 2 == 0 && v >= 1) {
        n++;
        v >>= 1;
    }
    return n;
}

int e2n(int n) {
    return 1 << n;
}

bool get_cs_and_hs(int &CS, int &DHWS, int batch, int in_width, int in_height, int in_depth, int pad_channels) {
    int DHW = in_width * in_height * in_depth;
    CS = 1;
    DHWS = 1;
    int DHW_MIN = 128;
    if (pad_channels <= 16) {
        if (pad_channels != 4 && pad_channels != 8 && pad_channels != 16) return false;
        if (pad_channels * DHW < 2048) return false;
        int n = max_diploit_of_e2n(DHW);
        //如果DHW不是2^n的整数倍，不能够进行DHW的分割，否则不方便进行reduce计算
        if (n == 0) return true;
        n = e2n(n);
        while (n * pad_channels > 1024) {
            n >>= 1;
        }
        DHWS = n;
        return true;
    } else if (pad_channels <= 512) {
        int n = max_diploit_of_e2n(pad_channels);
        if (e2n(n) == pad_channels) {
            //pad_channels = 32, 64, 128, 256, 512
            int diploit_dhw_n = max_diploit_of_e2n(DHW);
            if (diploit_dhw_n == 0) {
                if (DHW < DHW_MIN) return false;
                if (DHW >= DHW_MIN && DHW <= 256 && batch >= 64) {
                    //数据量较小时可以进行优化
                    return true;
                } else if (pad_channels >= 128) {
                    CS = pad_channels / 128;
                    return true;
                }
                return false;
            }
            if (pad_channels == 512) {
                if (batch <= 64) {
                    CS=4;
                    if (diploit_dhw_n > 3) diploit_dhw_n = 3;
                    DHWS = e2n(diploit_dhw_n);
                    return true;
                } else {
                    DHWS=2;
                    return true;
                }
            } else if (pad_channels == 256) {
                if (batch <= 64) {
                    CS=2;
                    if (diploit_dhw_n > 3) diploit_dhw_n = 3;
                    DHWS = e2n(diploit_dhw_n);
                    return true;
                } else {
                    if (diploit_dhw_n > 2) diploit_dhw_n = 2;
                    DHWS = e2n(diploit_dhw_n);
                    return true;
                }
            } else if (pad_channels == 128) {
                if (diploit_dhw_n > 3) diploit_dhw_n = 3;
                DHWS = e2n(diploit_dhw_n);
                return true;
            }
            if (DHW >= 256 && pad_channels <= 64) {
                CS = 1;
                while ((pad_channels << diploit_dhw_n) > 1024 && diploit_dhw_n > 0) diploit_dhw_n--;
                if (diploit_dhw_n == 0) return false;
                DHWS = e2n(diploit_dhw_n);
                return true;
            }
            return false;
        } else {
            int diploit_dhw_n = max_diploit_of_e2n(DHW);
            if (n < 7) {
                if (DHW >= 256 && pad_channels <= 64) {
                    CS = 1;
                    while ((pad_channels << diploit_dhw_n) > 1024 && diploit_dhw_n > 0) diploit_dhw_n--;
                    if (diploit_dhw_n == 0) return false;
                    DHWS = e2n(diploit_dhw_n);
                    return true;
                }
                return false;
            }
            CS = pad_channels >> n;
            while (n + diploit_dhw_n > 10 && diploit_dhw_n > 0) --diploit_dhw_n;
            DHWS = e2n(diploit_dhw_n);
            if (diploit_dhw_n == 0 && DHW < DHW_MIN) return false;
            return true;
        }
    } else {
        int n = max_diploit_of_e2n(pad_channels);  //10
        if (e2n(n) == pad_channels) {
            int diploit_dhw_n = max_diploit_of_e2n(DHW); //8
            int dhwsn = 0;
            int nn = n - 10;
            n = 10;
            while (n > 7 && (DHW >> dhwsn) > 256 && dhwsn < diploit_dhw_n) {
                ++dhwsn;
                --n;
            }
            CS = e2n(nn+10-n);
            DHWS = e2n(dhwsn);
            return true;
        } else {
            if (n < 7) {
                return false;
            }

            CS = pad_channels >> n;
            int diploit_dhw_n = max_diploit_of_e2n(DHW);
            while (n + diploit_dhw_n > 10 && diploit_dhw_n > 0) --diploit_dhw_n;
            DHWS = e2n(diploit_dhw_n);
            return true;
        }
    }
}
#endif

__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_N16CX(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    float in_scale,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_blk = blockIdx.x; //c_blk = c1_idx
    int c_idx = threadIdx.x; //c0_idx
    int b = blockIdx.z;

    int channels_per_block = 16;
    int b_offset = b * pad_channels;

    if (c_blk * channels_per_block + c_idx >= pad_channels) return;

    int64_t res = 0;

    for (int i = threadIdx.y; i < HW; i += blockDim.y) {
        int input_idx = b * HW * pad_channels  + c_blk * HW * channels_per_block + i * channels_per_block + c_idx;
        int8_t ival = input[input_idx];
        res += ival;
    }

    __shared__ int64_t sum_buffer[8][16];
    sum_buffer[threadIdx.y][c_idx] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i >>= 1) {
        if (threadIdx.y < i) {
            sum_buffer[threadIdx.y][c_idx] += sum_buffer[threadIdx.y + i][c_idx];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        int64_t temp = round((float(sum_buffer[0][c_idx]) / HW * in_scale) / out_scale);
        temp = max(min(temp, int64_t(127)), int64_t(-128));
        int output_idx = b * pad_channels + c_blk * channels_per_block + c_idx;
        output[output_idx] = static_cast<int8_t>(temp);
    }
#endif
}

__global__ void ppl_cukernel_pooling_ave_global_shuffle_int8_N16CX_opt(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    float in_scale,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c_blk = blockIdx.x;
    int c_idx = threadIdx.x;
    int b = blockIdx.z;
    int channels_per_block = 16;
    int b_offset = b * pad_channels;

    if (c_blk * channels_per_block + c_idx >= pad_channels) return;

    int64_t res = 0;

    for (int i = threadIdx.y; i < HW; i += blockDim.y) {
        int input_idx = b * HW * pad_channels + c_blk * HW * channels_per_block + i* channels_per_block + c_idx;
        res += static_cast<int64_t>(input[input_idx]);
    }

    __shared__ int64_t sum_buffer[8][16];
    sum_buffer[threadIdx.y][c_idx] = res;
    __syncthreads();

    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        if (threadIdx.y < offset) {
            sum_buffer[threadIdx.y][c_idx] += sum_buffer[threadIdx.y + offset][c_idx];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        int32_t final_res = sum_buffer[0][c_idx];
        float temp = roundf((static_cast<float>(final_res) / HW) * in_scale / out_scale);
        temp = fminf(fmaxf(temp, -128.0f), 127.0f);
        int output_idx = b * pad_channels + c_blk * channels_per_block + c_idx;
        output[output_idx] = static_cast<int8_t>(temp);
    }
#endif
}

__global__ void ppl_cukernel_pooling_ave_g1obal_shuffle_int8_N16CX_channels_2N(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    float scale,
    int in_dex) {
    int c1_num = pad_channels / 16;
    const int8_t* ptr_block_input = input + blockIdx.x * pad_channels * HW;
    int8_t* ptr_block_output = output + blockIdx.x * pad_channels;
    const float4 *ptr_input = (const float4*)(ptr_block_input);

    //Every thread will read 4 bytes per loop, aka process 16 images
    //A block have 512 threads, so, each loop, a block can handle
    //512 * 16 pixels = 4 channels of 16 images
    __shared__ int32_t results[4096]; //store final result

    union {
        float4 packed;
        int8_t unpacked[16];
    } mediate;
    int32_t temp_result[16] = {0};

    for (int i = threadIdx.x % in_dex; i < HW ; i+=in_dex){
        mediate.packed = ptr_input[ ((threadIdx.x/in_dex)%c1_num)*HW + i];
        for (int j = 0; j < 16; j++) temp_result[j] += mediate.unpacked[j];
    }

    /*for (int i = threadIdx.x; i< HW * c1_num; i+=blockDim.x) {
        mediate.packed = ptr_input[ i / c1_num + (i % c1_num)* HW];
        //.packed = ptr_input[ i];
        for (int j = 0; j < 16; j++) temp_result[j] += mediate.unpacked[j];
    }*/

    for (int i = threadIdx.x; i < pad_channels; i+= blockDim.x) {
        results[i] = 0;
    }

    __syncthreads();

    for (int i = 0; i < 16; i++) {
        int write_idx = (threadIdx.x/in_dex) * 16 + ((threadIdx.x % in_dex) + i) % 16;
        results[write_idx] += temp_result[((threadIdx.x % in_dex) + i) % 16];
        __syncthreads();
    }

    /*int bid = threadIdx.x / (PAD_CHANNELS / 16);
    int b_start = (threadIdx.x - (bid * (PAD_CHANNELS / 16))) << 4;
    for (int i = 0; i < 16; i++) {
        int write_idx = b_start + (i + bid) % 16;
        results[write_idx] += temp_result[(i + bid) % 16];
        __syncthreads();
    }*/
    float4* op = (float4*)ptr_block_output;

   /* for(int i = threadIdx.x; i < (pad_channels / 16); i += blockDim.x) {
    int32_t temp;
    int offset = i << 4;
    //int packed_value = 0;
    for (int j = 0; j < 16; j++) {
        float temp =round( float(results[offset + j]) / HW * scale);
        //temp = round(val);
        temp = min(temp, 127);
        temp = max(temp, -128);
        //packed_value |= (temp & 0xFF) << (j * 8);
        mediate.unpacked[j] = temp;
    }
    //op[i] = packed_value;
    op[i] = mediate.packed;
}*/
    for (int i = threadIdx.x; i < (pad_channels / 16); i+=blockDim.x) {
        for (int j = 0; j < 16; j++) {
            int32_t temp = round(float(results[(i<<4) + j]) / HW * scale);
            temp = min(temp,127);
            temp = max(temp,-128);
            mediate.unpacked[j] = temp;
        }
        op[i] = mediate.packed;
    }
}

ppl::common::RetCode PPLCUDAGlobalAvePoolingForwardImpFp16(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const half* input,
    ppl::common::TensorShape* output_shape,
    half* output)
{
    int batch        = output_shape->GetDim(0);
    // int channels = output_shape.GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    // int out_height = output_shape.GetDim(2); int out_width = output_shape.GetDim(3);
    int in_height    = input_shape->GetDim(2);
    int in_width = 1;
    int in_depth = 1;
    if(input_shape->GetDimCount() == 5)
    {
        in_depth = input_shape->GetDim(2);
        in_height = input_shape->GetDim(3);
        in_width = input_shape->GetDim(4);
    }
    else if(input_shape->GetDimCount() == 4)
    {
        in_width = input_shape->GetDim(3);
    }
    else if(input_shape->GetDimCount() == 3)
    {

    }
    else
    {
        return ppl::common::RC_UNSUPPORTED;
    }
    int in_dhw = in_height * in_width * in_depth;

    dim3 dim_grid(1, 1, batch);
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        dim3 dim_block(32, 4, 1);
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        ppl_cukernel_pooling_ave_global_shuffle_half<<<dim_grid, dim_block, 0, stream>>>((const half*)input, (half*)output, batch, pad_channels, in_dhw);
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
        // use half2, default padded
        dim3 dim_block(32, 8, 1); // (c, hw, 1)
        int padChannelsDivide = pad_channels >> 1; // half2
        int channel_blocks    = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
        constexpr int block_threshold = 64;
        constexpr int hw_threshold = 128;
#ifdef GAP_OPT
        {
            int CS = 1;
            int HWS = 1;
            if (get_cs_and_hs(CS, HWS, batch, in_width, in_height, in_depth, padChannelsDivide)) {
                dim3 dim_grid, dim_block;
                dim_grid.x = batch;
                dim_grid.y = CS;
                dim_grid.z = 1;
                dim_block.x = padChannelsDivide / CS;
                dim_block.y = HWS;
                dim_block.z = 1;
                if (CS == 1 && HWS == 1) {
                    ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_S_CS1_HWS1<<<dim_grid, dim_block, 0, stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_dhw, 1.0/in_dhw);
                } else if (CS == 1) {
                    ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_S_CS1<<<dim_grid, dim_block, 0, stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_dhw, 1.0/in_dhw);
                } else if (HWS == 1) {
                    ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_S_HWS1<<<dim_grid, dim_block, 0, stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_dhw, 1.0/in_dhw);
                } else {
                    ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_S<<<dim_grid, dim_block, 0, stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_dhw, 1.0/in_dhw);
                }
                return ppl::common::RC_SUCCESS;
            }
        }
#endif
        if (channel_blocks * batch < block_threshold && in_dhw > hw_threshold) {
            dim3 dim_block(8, 32, 1); // (c, hw, 1)
            int channel_blocks    = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
            dim3 dim_grid(channel_blocks, 1, batch);
        #ifdef PPLNN_USE_MACA //The result of atomicAdd function is not correct now, TO avoid it
            ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC<8, 32><<<dim_grid,
                                                               dim_block,
                                                               0,
                                                               stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_dhw);
        #else
            if (channel_blocks * batch < block_threshold) {
                int hw_blocks_limit = 8;
                int hw_blocks =  (in_dhw + dim_block.y - 1) / dim_block.y;
                hw_blocks = hw_blocks_limit < hw_blocks ? hw_blocks_limit : hw_blocks;
                dim3 dim_grid(channel_blocks, hw_blocks, batch);
                cudaMemsetAsync(output, 0, output_shape->CalcBytesIncludingPadding(), stream);
                ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC_atomic<<<dim_grid,
                                                                    dim_block,
                                                                    0,
                                                                    stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_dhw);
            } else {
                ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC<8, 32><<<dim_grid,
                                                                    dim_block,
                                                                    0,
                                                                    stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_dhw);
            }
        #endif
        } else {

            dim3 dim_grid(channel_blocks, 1, batch);
            ppl_cukernel_pooling_ave_global_shuffle_half2_NHWC<32, 8><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_dhw);
        }
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAGlobalAvePoolingForwardImpFp32(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const float* input,
    ppl::common::TensorShape* output_shape,
    float* output)
{
    int batch        = output_shape->GetDim(0);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int in_height    = input_shape->GetDim(2);
    int in_width = 1;
    int in_depth = 1;
    if(input_shape->GetDimCount() == 5)
    {
        in_depth = input_shape->GetDim(2);
        in_height = input_shape->GetDim(3);
        in_width = input_shape->GetDim(4);
    }
    else if(input_shape->GetDimCount() == 4)
    {
        in_width = input_shape->GetDim(3);
    }
    else if(input_shape->GetDimCount() == 3)
    {

    }
    else
    {
        return ppl::common::RC_UNSUPPORTED;
    }
    int in_dhw = in_height * in_width * in_depth;

    dim3 dim_block(32, 4, 1);
    dim3 dim_grid(1, 1, batch);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
        ppl_cukernel_pooling_ave_global_shuffle<float><<<dim_grid, dim_block,
            0, stream>>>((const float*)input, (float*)output, batch, pad_channels,
            in_dhw);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAGlobalAvePoolingForwardImpInt8(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const int8_t* input,
    ppl::common::TensorShape* output_shape,
    int8_t* output,
    float in_scale,
    float out_scale
) {

    int batch = output_shape->GetDim(0);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int in_height = input_shape->GetDim(2);
    int in_width = 1;
    int in_depth = 1;
    if(input_shape->GetDimCount() == 5)
    {
        in_depth = input_shape->GetDim(2);
        in_height = input_shape->GetDim(3);
        in_width = input_shape->GetDim(4);
    }
    else if(input_shape->GetDimCount() == 4)
    {
        in_width = input_shape->GetDim(3);
    }
    else if(input_shape->GetDimCount() == 3)
    {

    }
    else
    {
        return ppl::common::RC_UNSUPPORTED;
    }
    dim3 dim_block(32, 4, 1);
    dim3 dim_grid(1, 1, batch);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        int DHW = in_height * in_width * in_depth;
        float scale = in_scale / (out_scale * DHW );

        if (DHW & 0x1){
            // HW large and pad_channels * batch is even
            if (DHW > 64 && !((pad_channels * batch) & 0x1)){
                dim3 dim_block(64, 4, 1);
                dim_grid.y = (pad_channels * batch + (dim_block.y * 2) - 1) / (dim_block.y * 2) ;
                dim_grid.z = 1;
                ppl_cukernel_pooling_ave_global_shuffle_int8_2image<<<dim_grid, dim_block,
                    0, stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels,
                    DHW,  scale);
            }
            else{
                // normal method
                dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
                ppl_cukernel_pooling_ave_global_shuffle_int8_opt<<<dim_grid, dim_block,
                    0, stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels,
                    DHW,  scale);
            }
        } else if (DHW & 0x2){
            dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
            ppl_cukernel_pooling_ave_global_shuffle_int8_HW_even_opt<int16_t, 2><<<dim_grid, dim_block,
                0, stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels,
                DHW,  scale);
        } else if (DHW & 0x4){
            dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
            ppl_cukernel_pooling_ave_global_shuffle_int8_HW_even_opt<int32_t, 4><<<dim_grid, dim_block,
                0, stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels,
                DHW,  scale);
        } else {
            dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
            ppl_cukernel_pooling_ave_global_shuffle_int8_HW_even_opt<int64_t, 8><<<dim_grid, dim_block,
                0, stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels,
                DHW,  scale);
        }
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
#ifdef GAP_OPT
        int DHW = in_height * in_width * in_depth;
        float scale = in_scale / out_scale;
        bool kernel_launched = false;
        bool last_rejected = false;
#if 1
#define CALL_32N(SPLITS) \
    ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_32N<SPLITS><<<dim_grid, dim_block>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, scale); \
    kernel_launched=true;

#define CALL_SPLIT \
    ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_SPLIT_x4<<<dim_grid, dim_block>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, scale); \
    kernel_launched = true;

#define CALL_N_OR_SPLIT(N,R) \
    if (R > 8 && N * R <= 1024) { \
        dim_block.x = N / 4; \
        dim_block.y = R; \
        if (DHW * pad_channels / N / R <= 1600) { \
            CALL_SPLIT \
        } else {    \
            last_rejected = true; \
        } \
    } else { \
        if (DHW * pad_channels / N <= 1600) { \
            CALL_32N(R) \
        } else { \
            last_rejected = true; \
        } \
    } \
    break;

#define CASE(N,R)   \
    case R*N:       \
        CALL_N_OR_SPLIT(N, R)

    if (!kernel_launched && !last_rejected && pad_channels % 512 == 0) {
        dim3 dim_block(512, 1, 1);
        dim3 dim_grid(batch, 1, 1);

        switch (pad_channels) {
            CASE(512,1)
            CASE(512,2)
            CASE(512,3)
            CASE(512,4)
            CASE(512,5)
            CASE(512,6)
            CASE(512,7)
            CASE(512,8)
            CASE(512,9)
            CASE(512,10)
            CASE(512,11)
            CASE(512,12)
            CASE(512,13)
            CASE(512,14)
            CASE(512,15)
            CASE(512,16)
            CASE(512,17)
            CASE(512,18)
            CASE(512,19)
            CASE(512,20)
        }
    }

#define SWITCH_CASE(N) \
    switch (pad_channels) { \
        CASE(N,1)   \
        CASE(N,3)   \
        CASE(N,5)   \
        CASE(N,7)   \
        CASE(N,9)   \
        CASE(N,11)   \
        CASE(N,13)   \
        CASE(N,15)   \
        CASE(N,17)   \
        CASE(N,19)   \
        CASE(N,21)   \
        CASE(N,23)   \
        CASE(N,25)   \
        CASE(N,27)   \
        CASE(N,29)   \
        CASE(N,31)   \
        CASE(N,33)   \
        CASE(N,35)   \
        CASE(N,37)   \
        CASE(N,39)   \
    }

        if (!kernel_launched && !last_rejected && pad_channels % 256 == 0) {
            dim3 dim_block(256, 1, 1);
            dim3 dim_grid(batch, 1, 1);
            SWITCH_CASE(256)
        }

        if (!kernel_launched && !last_rejected && pad_channels % 128 == 0) {
            dim3 dim_block(128, 1, 1);
            dim3 dim_grid(batch, 1, 1);
            SWITCH_CASE(128)
        }

        if (!kernel_launched && !last_rejected && pad_channels % 64 == 0) {
            dim3 dim_block(64, 1, 1);
            dim3 dim_grid(batch, 1, 1);
            SWITCH_CASE(64)
        }

        if (!kernel_launched && !last_rejected && pad_channels % 32 == 0) {
            dim3 dim_block(32, 1, 1);
            dim3 dim_grid(batch, 1, 1);
            SWITCH_CASE(32)
        }
#undef CALL_32N
#undef CASE
#undef SWITCH_CASE
#undef CALL_N_OR_SPLIT
        if (kernel_launched) return ppl::common::RC_SUCCESS;
#endif
        if((pad_channels == 512 || pad_channels == 1024 || pad_channels == 2048 || pad_channels == 4096)){
            dim3 dim_block(1024, 1, 1);
            dim3 dim_grid(batch, 1, 1);
            if (DHW < 256) dim_block.x /= 2;
#define CALL_C2N(PAD_CHANNELS) \
            ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_Opt_C2N<PAD_CHANNELS><<<dim_grid, \
                                                                dim_block, \
                                                                0, \
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, scale)
            if (pad_channels == 512) {
                CALL_C2N(512);
            } else if (pad_channels == 1024) {
                CALL_C2N(1024);
            } else if (pad_channels == 2048) {
                CALL_C2N(2048);
            } else if (pad_channels == 4096) {
                CALL_C2N(4096);
            }
        } else if(pad_channels < 2048 && pad_channels >= 1024 && (pad_channels & 7) == 0){
            dim3 dim_block(512, 1, 1);
            dim3 dim_grid(batch, 1, 1);
            int line_pb = 8192 / pad_channels;
            line_pb = line_pb >> 1 << 1;
            if((pad_channels & 15) == 0){
                int channelCount = pad_channels / 16;
                ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_Opt_line<float4><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, line_pb, channelCount, scale);
            }else if((pad_channels & 7) == 0){
                int channelCount = pad_channels / 8;
                ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_Opt_line<int64_t><<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, line_pb, channelCount, scale);
            }
        } else if ((pad_channels & 15) == 0 && pad_channels < 1024 && DHW > 4096 / pad_channels * 2) {
            dim3 dim_block(512, 1, 1);
            dim3 dim_grid(batch, 1, 1);
            int lines_per_loop = 4096 / pad_channels;
            ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_Opt_dbuf<<<dim_grid, dim_block, 0, stream>>>(
                (const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, lines_per_loop, scale);
        } else {
            dim3 dim_block(32,8,1);
            int pad_channels1 = pad_channels / 8;
            int channel_blocks = (pad_channels1 + dim_block.y - 1) / (dim_block.y);
            dim3 dim_grid(1, channel_blocks, batch);
            ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC_Opt<<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const int64_t*)input, (int8_t*)output, batch, pad_channels1, DHW, scale);
        }
#else//!GAP_OPT
        dim3 dim_block(32, 8, 1);
        int channel_blocks    = (pad_channels + dim_block.x - 1) / dim_block.x;
        dim3 dim_grid(channel_blocks, 1, batch);

        ppl_cukernel_pooling_ave_global_shuffle_int8_NHWC<<<dim_grid,
                                                             dim_block,
                                                             0,
                                                             stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, in_height * in_width, in_scale, out_scale);
#endif//GAP_OPT
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NCHW16) {
       /* dim3 dim_block(16, 8, 1);
        dim3 dim_grid((pad_channels + 15) / 16, 1, batch);
        ppl_cukernel_pooling_ave_global_shuffle_int8_N16CX<<<dim_grid, dim_block, 0, stream>>>(
                                                            (const int8_t*)input, (int8_t*)output, batch, pad_channels, in_height * in_width, in_scale, out_scale);
    dim3 dim_block(1024, 1, 1);
        dim3 dim_grid(batch, 1, 1);
        int HW =in_height*in_width;
        float scale = in_scale / out_scale;
        if (HW < 256) dim_block.x /= 2;
        #define CALL_C2NN(PAD_CHANNELS)\
         ppl_cukernel_pooling_ave_g1obal_shuffle_int8_N16CX_opt3<PAD_CHANNELS><<<dim_grid, \
                                                                dim_block, \
                                                                0, \
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, HW, scale);
        if (pad_channels == 2048) {
                CALL_C2NN(2048);
	}*/
    #ifdef GAP_OPT
        int DHW = in_height * in_width * in_depth;
        float scale = in_scale / out_scale;
        if(pad_channels == 512 || pad_channels == 1024 || pad_channels == 2048 || pad_channels == 4096){
            dim3 dim_block(1024, 1, 1);
            dim3 dim_grid(batch, 1, 1);
            if (DHW < 256) dim_block.x /= 2;
            //#define CALL_C2N(PAD_CHANNELS) \
            /* ppl_cukernel_pooling_ave_g1obal_shuffle_int8_N16CX_opt3<PAD_CHANNELS><<<dim_grid, \
                                                                dim_block, \
                                                                0, \
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, HW, scale)*/
            if (pad_channels == 512) {
                //CALL_C2N(512);
               /* int PAD_CHANNELS=512;*/
                int in_dex=dim_block.x/(pad_channels/16);
                 ppl_cukernel_pooling_ave_g1obal_shuffle_int8_N16CX_channels_2N<<<dim_grid, \
                                                                dim_block, \
                                                                0, \
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, scale,in_dex);
            } else if (pad_channels == 1024) {
                /*CALL_C2N(1024);
                int PAD_CHANNELS=1024;*/
                int in_dex=dim_block.x/(pad_channels/16);
                 ppl_cukernel_pooling_ave_g1obal_shuffle_int8_N16CX_channels_2N<<<dim_grid, \
                                                                dim_block, \
                                                                0, \
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, scale,in_dex);
            } else if (pad_channels == 2048) {
                /*CALL_C2N(2048);
                int PAD_CHANNELS=2048;*/
                int in_dex=dim_block.x/(pad_channels/16);
                 ppl_cukernel_pooling_ave_g1obal_shuffle_int8_N16CX_channels_2N<<<dim_grid, \
                                                                dim_block, \
                                                                0, \
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, scale,in_dex);
            } else if (pad_channels == 4096) {
                /*CALL_C2N(4096);
                int PAD_CHANNELS=2048;*/
                int in_dex=dim_block.x/(pad_channels/16);
                 ppl_cukernel_pooling_ave_g1obal_shuffle_int8_N16CX_channels_2N<<<dim_grid, \
                                                                dim_block, \
                                                                0, \
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, scale,in_dex);
            }

        } else {
             dim3 dim_block(1024, 1, 1);
             dim3 dim_grid(batch, 1, 1);
             if (DHW < 256) dim_block.x /= 8;
            ppl_cukernel_pooling_ave_global_shuffle_int8_N16CX_opt<<<dim_grid,
                                                                dim_block,
                                                                0,
                                                                stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, in_scale, out_scale);
        }
    #else//!GAP_OPT
        dim3 dim_block(1024, 1, 1);
        dim3 dim_grid(batch, 1, 1);
        ppl_cukernel_pooling_ave_global_shuffle_int8_N16CX<<<dim_grid, dim_block, 0, stream>>>(
                                                            (const int8_t*)input, (int8_t*)output, batch, pad_channels, DHW, in_scale, out_scale);
    #endif//GAP_OPT
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAGlobalAvePoolingForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input,
    ppl::common::TensorShape* output_shape,
    void* output, float in_scale, float out_scale)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return PPLCUDAGlobalAvePoolingForwardImpFp16(
            stream, input_shape, (const half*)input, output_shape, (half*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        return PPLCUDAGlobalAvePoolingForwardImpFp32(
            stream, input_shape, (const float*)input, output_shape, (float*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        return PPLCUDAGlobalAvePoolingForwardImpInt8(
            stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}
