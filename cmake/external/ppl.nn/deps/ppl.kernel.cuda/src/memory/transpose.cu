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

#include "cudakernel/memory/transpose.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "cudakernel/common/common.h"
#include <cuda_fp16.h>
#define DIM     32
#define MAX_DIM 65533

struct FastTransposeParam {
    int64_t n_outer  = 1;
    int64_t n_height = 1;
    int64_t n_width  = 1;
    int64_t n_inner  = 1;
    void reset()
    {
        n_outer  = 1;
        n_height = 1;
        n_width  = 1;
        n_inner  = 1;
    }
    int GetMaxPower2(const int max_value = 8)
    {
        for (int ret = max_value; ret >= 1; ret /= 2) {
            if (n_height % ret == 0 && n_width % ret == 0)
                return ret;
        }
        return 1;
    }
};

template <typename T>
__global__ void cuda_kernel_fast_trans(
    const T *input,
    FastTransposeParam param,
    T *output)
{
    __shared__ T share_val[DIM][DIM + 1];
    int64_t num = blockIdx.z;
    for (int n = num; n < param.n_outer; n += gridDim.z) {
        for (int t = blockIdx.y; t < DivUp(param.n_height, 32); t += gridDim.y) {
            int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t idx_h = t * blockDim.y + threadIdx.y;

            if (idx_w < param.n_width && idx_h < param.n_height) {
                int64_t offset                      = n * param.n_height * param.n_width + idx_h * param.n_width + idx_w;
                share_val[threadIdx.y][threadIdx.x] = input[offset];
            } else {
                share_val[threadIdx.y][threadIdx.x] = (T)0;
            }
            __syncthreads();
            idx_w = t * blockDim.y + threadIdx.x;
            idx_h = blockIdx.x * blockDim.x + threadIdx.y;
            if (idx_w < param.n_height && idx_h < param.n_width) {
                int64_t offset = n * param.n_height * param.n_width + idx_h * param.n_height + idx_w;
                output[offset] = share_val[threadIdx.x][threadIdx.y];
            }
        }
    }
}

template <typename T>//去掉内层循环
__global__ void cuda_kernel_fast_trans_opt(
    const T *input,
    FastTransposeParam param,
    T *output)
{
    __shared__ T share_val[DIM][DIM + 1];
    int64_t num = blockIdx.z;
    int t = blockIdx.y;
    for (int n = num; n < param.n_outer; n += gridDim.z) {
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;
        int64_t idx_h = t * blockDim.y + threadIdx.y;

        if (idx_w < param.n_width && idx_h < param.n_height) {
            int64_t offset                      = n * param.n_height * param.n_width + idx_h * param.n_width + idx_w;
            share_val[threadIdx.y][threadIdx.x] = input[offset];
        } else {
            share_val[threadIdx.y][threadIdx.x] = (T)0;
        }
        __syncthreads();
        idx_w = t * blockDim.y + threadIdx.x;
        idx_h = blockIdx.x * blockDim.x + threadIdx.y;
        if (idx_w < param.n_height && idx_h < param.n_width) {
            int64_t offset = n * param.n_height * param.n_width + idx_h * param.n_height + idx_w;
            output[offset] = share_val[threadIdx.x][threadIdx.y];
        }
    }
}

template <typename T1, typename T2>
__global__ void cuda_kernel_fast_trans_op(
    const T1 *input,
    FastTransposeParam param,
    T1 *output)
{
    constexpr int times = sizeof(T2) / sizeof(T1);
    __shared__ union {
        T1 m1[DIM][DIM + times];
        T2 m2[DIM][DIM / times + 1];
    } share_val;

    int64_t num = blockIdx.z;
    for (int n = num; n < param.n_outer; n += gridDim.z) {
        const T2 *input_T2 = (T2 *)(input + n * param.n_height * param.n_width);
        T2 *output_T2      = (T2 *)(output + n * param.n_height * param.n_width);

        for (int t = blockIdx.y; t < DivUp(param.n_height, DIM); t += gridDim.y) {
            int64_t idx_w = times * blockIdx.x * blockDim.x + times * threadIdx.x;
            int64_t idx_h = t * blockDim.y + threadIdx.y;
            if (idx_w < param.n_width && idx_h < param.n_height) {
                int64_t offset                         = idx_h * param.n_width + idx_w;
                share_val.m2[threadIdx.y][threadIdx.x] = input_T2[offset / times];
            } else {
                share_val.m2[threadIdx.y][threadIdx.x] = (int16_t)0;
            }
            __syncthreads();

            idx_w = t * blockDim.y + threadIdx.x * times;
            idx_h = times * blockIdx.x * blockDim.x + threadIdx.y;

            if (idx_w < param.n_height && idx_h < param.n_width) {
                int64_t offset = idx_h * param.n_height + idx_w;
                union {
                    T2 m2;
                    T1 m1[times];
                } tmp_storage;

#pragma unroll
                for (int i = 0; i < times; i++) {
                    tmp_storage.m1[i] = share_val.m1[times * threadIdx.x + i][threadIdx.y];
                }
                
                output_T2[offset / times] = tmp_storage.m2;
            }
        }
    }
}

__global__ __launch_bounds__(1024) void cuda_kernel_fast_trans_opt_int8_unalign(const int8_t* input, FastTransposeParam param, int8_t* output, int64_t gridDim_z, int64_t gridDim_y, int64_t blockDim_x, int64_t blockDim_y) {
    __shared__ union{
        int8_t m1[128][128];
        float4 m2[128][8];
    } sm_buffer;
    int64_t num = blockIdx.z;
    int block_num = (param.n_height + 15) >> 4;
    int64_t image_size = param.n_height * param.n_width;
    
    for(int n = num; n < param.n_outer; n += gridDim_z) {
        for(int t = blockIdx.y; t < block_num; t += gridDim_y) {
            int64_t idx_w = (blockIdx.x * blockDim_x + threadIdx.x) << 4;
            int64_t idx_h = t * blockDim_y + threadIdx.y;
            const int8_t * ptr_input = (const int8_t *)(input + n * image_size);
            int8_t * ptr_output      = (int8_t *)(output + n * image_size);
            if (idx_w < param.n_width && idx_h < param.n_height) {
                int64_t offset                         = idx_h * param.n_width + idx_w;
                sm_buffer.m2[threadIdx.y][threadIdx.x] = *(float4*)(ptr_input + offset);
            }
            __syncthreads();
            int offset_x = threadIdx.x << 4;
            idx_w = t * blockDim_y + offset_x;
            idx_h = blockIdx.x * (blockDim_x << 4) + threadIdx.y;

            if (idx_h < param.n_width) {
                int64_t offset = idx_h * param.n_height + idx_w;
                ptr_output = ptr_output + offset;
                for(int i = 0, w_offset = idx_w; i < 16 && w_offset < param.n_height; i++, w_offset++){
                    int8_t* ptr_sm_buffer = sm_buffer.m1[offset_x + i];
                    *(ptr_output + i) = ptr_sm_buffer[threadIdx.y];
                }
            }
        }
    }
}

__global__ void cuda_kernel_fast_trans_opt_fp16_unalign(const half* input, FastTransposeParam param, half* output, int64_t gridDim_z, int64_t gridDim_y, int64_t blockDim_x, int64_t blockDim_y) {
    __shared__ union{
        half m1[64][72];
        float4 m2[64][9];
    } sm_buffer;
    int64_t num = blockIdx.z;
    int block_num = (param.n_height + 7) >> 3;
    int64_t image_size = param.n_height * param.n_width;
    
    for(int n = num; n < param.n_outer; n += gridDim_z) {
        for(int t = blockIdx.y; t < block_num; t += gridDim_y) {
            int64_t idx_w = (blockIdx.x * blockDim_x + threadIdx.x) << 3;
            int64_t idx_h = t * blockDim_y + threadIdx.y;
            const half * ptr_input = (const half *)(input + n * image_size);
            half * ptr_output      = (half *)(output + n * image_size);
            if (idx_w < param.n_width && idx_h < param.n_height) {
                int64_t offset                         = idx_h * param.n_width + idx_w;
                sm_buffer.m2[threadIdx.y][threadIdx.x] = *(float4*)(ptr_input + offset);
            }
            __syncthreads();
            int offset_x = threadIdx.x << 3;
            idx_w = t * blockDim_y + offset_x;
            idx_h = blockIdx.x * (blockDim_x << 3) + threadIdx.y;

            if (idx_h < param.n_width) {
                int64_t offset = idx_h * param.n_height + idx_w;
                ptr_output = ptr_output + offset;
                for(int i = 0, w_offset = idx_w; i < 8 && w_offset < param.n_height; i++, w_offset++){
                    half* ptr_sm_buffer = sm_buffer.m1[offset_x + i];
                    *(ptr_output + i) = ptr_sm_buffer[threadIdx.y];
                }
            }
        }
    }
}

#ifdef PPLNN_USE_MACA
template<typename T, int SHIFT>
__global__ void cuda_kernel_fast_trans_shift_rows(
    const int32_t num_elems,
    const T *input,
    FastTransposeParam param,
    DivModFast h_mod,
    DivModFast w_mod,
    T *output)
{
    //In this kernel, we will only load data from no more than 2 images
    //So many calculations can be omited.
    constexpr int MEM = 8192;
    __shared__ union {
        T unpacked[MEM/sizeof(T)];
        float4 packed[MEM/sizeof(float4)];
    } ibuf;
    __shared__ T dbuf[MEM/sizeof(T)];

    int elem_start = (blockIdx.x << SHIFT) * param.n_width;
    if (elem_start > num_elems) return;

    //Load from src image, directly use maximum bandwidth
    const float4* block_input = (const float4*)(input + elem_start);
    constexpr int f4w = sizeof(float4) / sizeof(T);
    if (threadIdx.x < (param.n_width << SHIFT) / f4w) ibuf.packed[threadIdx.x] = block_input[threadIdx.x];

    __syncthreads();
    //Exchange ibuf data into dbuf
    //The dbuf is almost the same as we will write
    {
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        for (int i = t_bid; i < param.n_width; i += (blockDim.x >> SHIFT)) {
            int src_idx = i + t_tid * param.n_width;
            int dest_idx = (i << SHIFT) + t_tid;
            dbuf[dest_idx] = ibuf.unpacked[src_idx];
        }
    }

    //To see where the data will store to
    int img0, img0_start_row, img0_rows, img1;
    h_mod.divmod(blockIdx.x << SHIFT, img0, img0_start_row);
    if (img0_start_row + (1 << SHIFT) >= param.n_height) {
        img1 = img0 + 1;
        img0_rows = param.n_height - img0_start_row;
    } else {
        img0_rows = (1 << SHIFT);
    }
    __syncthreads();

    //When storing in one image, an if clause can be omited
    if (img0_rows == (1 << SHIFT)) {
        T *img_output = output + img0 * param.n_height * param.n_width;
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        for (int i = t_bid; i < param.n_width; i += (blockDim.x >> SHIFT)) {
            img_output[i * param.n_height + t_tid + img0_start_row] = dbuf[(i << SHIFT) + t_tid];
        }
    } else {
        T *img0_output = output + img0 * param.n_height * param.n_width;
        T *img1_output = output + img1 * param.n_height * param.n_width;
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        for (int i = t_bid; i < param.n_width; i += (blockDim.x >> SHIFT)) {
            if (t_tid < img0_rows)
            img0_output[i * param.n_height + t_tid + img0_start_row] = dbuf[(i << SHIFT) + t_tid];
            else
            img1_output[i * param.n_height + t_tid - img0_rows] = dbuf[(i << SHIFT) + t_tid];
        }
    }
}

template<typename T, int SHIFT>
__global__ void cuda_kernel_fast_trans_shift_rows_with_tail(
    const int32_t num_elems,
    const T *input,
    FastTransposeParam param,
    DivModFast h_mod,
    DivModFast w_mod,
    T *output)
{
    //Some times, we can not load 1 << SHIFT rows in all blocks
    //Most of the logic is the same as cuda_kernel_fast_trans_shift_rows
    //but we have to identify if we read the data end and data is less
    //than 1 << SHIFT rows
    constexpr int MEM = 8192;
    __shared__ union {
        T unpacked[MEM/sizeof(T)];
        float4 packed[MEM/sizeof(float4)];
    } ibuf;
    __shared__ T dbuf[MEM/sizeof(T)];

    int elem_start = (blockIdx.x << SHIFT) * param.n_width;
    if (elem_start > num_elems) return;
    const float4* block_input = (const float4*)(input + elem_start);
    constexpr int f4w = sizeof(float4) / sizeof(T);
    int block_lines = param.n_height * param.n_outer - (blockIdx.x << SHIFT) >= (1 << SHIFT) ? (1 << SHIFT) : param.n_height * param.n_outer - (blockIdx.x << SHIFT);
    if (threadIdx.x < ((block_lines * param.n_width) / f4w)) ibuf.packed[threadIdx.x] = block_input[threadIdx.x];

    __syncthreads();
    {
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        for (int i = t_bid; i < param.n_width; i += (blockDim.x >> SHIFT)) {
            if (t_tid < block_lines) {
                int src_idx = i + t_tid * param.n_width;
                int dest_idx = (i << SHIFT) + t_tid;
                dbuf[dest_idx] = ibuf.unpacked[src_idx];
            }
        }
    }

    int img0, img0_start_row, img0_rows, img1;
    h_mod.divmod(blockIdx.x << SHIFT, img0, img0_start_row);
    if (img0_start_row + (1 << SHIFT) >= param.n_height) {
        img1 = img0 + 1;
        img0_rows = param.n_height - img0_start_row;
    } else {
        img0_rows = (1 << SHIFT);
    }
    __syncthreads();

    if (img0_rows == (1 << SHIFT)) {
        T *img_output = output + img0 * param.n_height * param.n_width;
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        for (int i = t_bid; i < param.n_width; i += (blockDim.x >> SHIFT)) {
            if (t_tid < block_lines) {
                img_output[i * param.n_height + t_tid + img0_start_row] = dbuf[(i << SHIFT) + t_tid];
            }
        }
    } else {
        T *img0_output = output + img0 * param.n_height * param.n_width;
        T *img1_output = output + img1 * param.n_height * param.n_width;
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        for (int i = t_bid; i < param.n_width; i += (blockDim.x >> SHIFT)) {
            if (t_tid < block_lines) {
                if (t_tid < img0_rows)
                img0_output[i * param.n_height + t_tid + img0_start_row] = dbuf[(i << SHIFT) + t_tid];
                else
                img1_output[i * param.n_height + t_tid - img0_rows] = dbuf[(i << SHIFT) + t_tid];
            }
        }
    }
}

template<typename T, int SHIFT>
__global__ void cuda_kernel_fast_trans_shift_rows_mi(
    const int32_t num_elems,
    const T *input,
    FastTransposeParam param,
    DivModFast h_mod,
    DivModFast w_mod,
    T *output)
{
    //Same logic with cuda_kernel_fast_trans_shift_rows except that
    //we need to process multi-image in one block
    constexpr int MEM = 8192;
    __shared__ union {
        T unpacked[MEM/sizeof(T)];
        float4 packed[MEM/sizeof(float4)];
    } ibuf;
    __shared__ T dbuf[MEM/sizeof(T)];
    int elem_start = (blockIdx.x << SHIFT) * param.n_width;
    if (elem_start >= num_elems) return;
    const float4* block_input = (const float4*)(input + elem_start);
    //Load at most 1 << SHIFT rows of original image, SHIFT should >= 4 and <= 9
    //Load input image data 16 pixels each time
    constexpr int f4w = sizeof(float4) / sizeof(T);
    if (threadIdx.x < (param.n_width << SHIFT) / f4w) ibuf.packed[threadIdx.x] = block_input[threadIdx.x];
    __syncthreads();
    //Re-arrange data into output order
    //Attention: 我们使用了D-BUFF机制来对数据进行旋转，而不是在load阶段就进行数据旋转，主要原因如下：
    //1. 为了增加最终写入结果时CACHELINE的命中概率(CACHE LINE耗尽的时候，读写指令均无法发射)，我们需要
    //   对数据进行事先的transpose
    //2. 如果直接在读input的时候，就将数据旋转后放入shared memory，会导致严重的bank conflict，导致STS
    //   指令只能够延迟发射（如读入一个float4大小的数据，实际有16个数据，需要写入到分布在16行的16个位置）
    //3. 从shared memory读入数据要比从global memory读入数据效率高
    //4. 从shared memory读入列数据，以行的方式存入到第二个shared memory buffer，产生的bank conflict会
    //   极大的减少，读数据的效率也增加
    //5. 针对sizeof(T) < 16的情况，使用D-BUFF机制均能带来收益
    {
        //Divide block threads into blockDim.x >> SHIFT tiny blocks
        //Which means when SHIFT == 4, TinyBlocks = 32
        //                 SHIFT == 5, TinyBlocks = 16
        //                 SHIFT == 6, TinyBlocks = 8
        //                 SHIFT == 7, TinyBlocks = 4
        //                 SHIFT == 8, TinyBlocks = 2
        //                 SHIFT == 9, TinyBlocks = 1
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        //Src shape h = N, w = param.n_width
        //Dest shape h = n_width, w = N
        for (int i = t_bid; i < param.n_width; i += (blockDim.x  >> SHIFT)) {
            int src_idx = i + t_tid * param.n_width;
            int dest_idx = (i << SHIFT) + t_tid;
            dbuf[dest_idx] = ibuf.unpacked[src_idx];
        }
    }

    __syncthreads();
    //在shared memory中已经排序好的数据，可能跨越了多个img
    //如对于源数据  w = 9, h = 200,  每次总共读入512行数据，跨越了3张图片
    //则读入的ibuf数据为(h x w) 512 * 9, dbuf数据为 9 * 512
    //数据分布如下：
    /*
        [0, 1, 2, ..., 199], [200, ..., 399], [400, ..., 511]    line 0
        [512, ...,     711], [712, ..., 911], [912, ..., 1111]   line 1
        ...
        [512*8, ..., ] ...                    [ ...,  512 * 9 - 1] line 8
        img0                 img1             img2
    */
    int t_bid = threadIdx.x >> SHIFT;
    int t_tid = threadIdx.x - (t_bid << SHIFT);
    for (int i = t_bid; i < param.n_width; i += (blockDim.x  >> SHIFT)) {
        int img0, img0_start_row;
        T val = dbuf[(i << SHIFT) + t_tid];
        h_mod.divmod((blockIdx.x << SHIFT) + t_tid, img0, img0_start_row);
        int d_w = img0_start_row;
        int d_h = i;
        int offset = img0 * param.n_height * param.n_width + d_h * param.n_height + d_w;
        output[offset] = val;
    }
}

template<typename T, int SHIFT>
__global__ void cuda_kernel_fast_trans_shift_rows_with_tail_mi(
    const int32_t num_elems,
    const T *input,
    FastTransposeParam param,
    DivModFast h_mod,
    DivModFast w_mod,
    T *output)
{
    //Same logic with cuda_kernel_fast_trans_shift_rows_mi except
    //that the last data processing block will process less than
    //1 << SHIFT rows
    constexpr int MEM = 8192;
    __shared__ union {
        T unpacked[MEM/sizeof(T)];
        float4 packed[MEM/sizeof(float4)];
    } ibuf;
    __shared__ T dbuf[MEM/sizeof(T)];
    int elem_start = (blockIdx.x << SHIFT) * param.n_width;
    if (elem_start > num_elems) return;
    int block_lines = param.n_height * param.n_outer - (blockIdx.x << SHIFT) >= (1 << SHIFT) ? (1 << SHIFT) : param.n_height * param.n_outer - (blockIdx.x << SHIFT);
    const float4* block_input = (const float4*)(input + elem_start);
    //Load at most 1 << SHIFT rows of original image, SHIFT should >= 4 and <= 9
    //Load input image data 16 pixels each time
    constexpr int f4w = sizeof(float4) / sizeof(T);
    if (threadIdx.x < (param.n_width * block_lines) / f4w) ibuf.packed[threadIdx.x] = block_input[threadIdx.x];
    __syncthreads();
    //Re-arrange data into output order
    //Attention: 我们使用了D-BUFF机制来对数据进行旋转，而不是在load阶段就进行数据旋转，主要原因如下：
    //1. 为了增加最终写入结果时CACHELINE的命中概率(CACHE LINE耗尽的时候，读写指令均无法发射)，我们需要
    //   对数据进行事先的transpose
    //2. 如果直接在读input的时候，就将数据旋转后放入shared memory，会导致严重的bank conflict，导致STS
    //   指令只能够延迟发射（如读入一个float4大小的数据，实际有16个数据，需要写入到分布在16行的16个位置）
    //3. 从shared memory读入数据要比从global memory读入数据效率高
    //4. 从shared memory读入列数据，以行的方式存入到第二个shared memory buffer，产生的bank conflict会
    //   极大的减少，读数据的效率也增加
    //5. 针对sizeof(T) < 16的情况，使用D-BUFF机制均能带来收益
    {
        //Divide block threads into blockDim.x >> SHIFT tiny blocks
        //Which means when SHIFT == 4, TinyBlocks = 32
        //                 SHIFT == 5, TinyBlocks = 16
        //                 SHIFT == 6, TinyBlocks = 8
        //                 SHIFT == 7, TinyBlocks = 4
        //                 SHIFT == 8, TinyBlocks = 2
        //                 SHIFT == 9, TinyBlocks = 1
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        //Src shape h = N, w = param.n_width
        //Dest shape h = n_width, w = N
        for (int i = t_bid; i < param.n_width; i += (blockDim.x  >> SHIFT)) {
            if (t_tid < block_lines) {
                int src_idx = i + t_tid * param.n_width;
                int dest_idx = (i << SHIFT) + t_tid;
                dbuf[dest_idx] = ibuf.unpacked[src_idx];
            }
        }
    }

    __syncthreads();
    //在shared memory中已经排序好的数据，可能跨越了多个img
    //如对于源数据  w = 9, h = 200,  每次总共读入512行数据，跨越了3张图片
    //则读入的ibuf数据为(h x w) 512 * 9, dbuf数据为 9 * 512
    //数据分布如下:
    /*
        [0, 1, 2, ..., 199], [200, ..., 399], [400, ..., 511]    line 0
        [512, ...,     711], [712, ..., 911], [912, ..., 1111]   line 1
        ...
        [512*8, ..., ] ...                    [ ...,  512 * 9 - 1] line 8
        img0                 img1             img2
    */
    int t_bid = threadIdx.x >> SHIFT;
    int t_tid = threadIdx.x - (t_bid << SHIFT);
    for (int i = t_bid; i < param.n_width; i += (blockDim.x  >> SHIFT)) {
        if (t_tid < block_lines) {
            int img0, img0_start_row;
            T val = dbuf[(i << SHIFT) + t_tid];
            h_mod.divmod((blockIdx.x << SHIFT) + t_tid, img0, img0_start_row);
            int d_w = img0_start_row;
            int d_h = i;
            int offset = img0 * param.n_height * param.n_width + d_h * param.n_height + d_w;
            output[offset] = val;
        }
    }
}


//For width == 1 or height == 1
//We only need to apply a memcpy
template<typename T>
__global__ void cuda_kernel_fast_trans_b1(
    const int32_t num_elems,
    const T *input,
    FastTransposeParam param,
    T *output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_elems) return;
    output[idx] = input[idx];
}


template<typename T, int SHIFT>
__global__ void cuda_kernel_fast_trans_shift_cols(
    const int32_t num_elems,
    const T *input,
    FastTransposeParam param,
    DivModFast h_mod,
    DivModFast w_mod,
    T *output)
{
    //Load data from at most two images per block
    //Each block will load 1 << SHIFT cols
    constexpr int MEM = 8192;
    __shared__ union {
        T unpacked[MEM/sizeof(T)];
        float4 packed[MEM/sizeof(float4)];
    } dbuf;
    __shared__ T ibuf[MEM/sizeof(T)];
    int elem_start = (blockIdx.x << SHIFT) * param.n_height;
    if (elem_start > num_elems) return;
    //Identify if we will load data from one or two images
    int img0, img0_start_col, img0_cols, img1;
    w_mod.divmod(blockIdx.x << SHIFT, img0, img0_start_col);
    if (img0_start_col + (1 << SHIFT) >= param.n_width) {
        img1 = img0 + 1;
        img0_cols = param.n_width - img0_start_col;
    } else {
        img0_cols = (1 << SHIFT);
    }
    if (img0_cols == (1 << SHIFT)) {
        const T *img_input = input + img0 * param.n_height * param.n_width;
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        for (int i = t_bid; i < param.n_height; i += (blockDim.x >> SHIFT)) {
            ibuf[(i << SHIFT) + t_tid] = img_input[i * param.n_width + t_tid + img0_start_col];
        }
    } else {
        const T *img0_input = input + img0 * param.n_height * param.n_width;
        const T *img1_input = input + img1 * param.n_height * param.n_width;
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        for (int i = t_bid; i < param.n_height; i += (blockDim.x >> SHIFT)) {
            if (t_tid < img0_cols)
            ibuf[(i << SHIFT) + t_tid] = img0_input[i * param.n_width + t_tid + img0_start_col];
            else
            ibuf[(i << SHIFT) + t_tid] = img1_input[i * param.n_width + t_tid - img0_cols];
        }
    }

    __syncthreads();
    //Exchange original data into final output
    for (int i = threadIdx.x; i < param.n_height << SHIFT; i+=blockDim.x) {
        int src_w,src_h;
        h_mod.divmod(i, src_w, src_h);
        int src_pixel = src_w + (src_h << SHIFT);
        dbuf.unpacked[i] = ibuf[src_pixel];
    }
    __syncthreads();
    //Directly output as a combined data type
    float4* block_output = (float4*)(output + (blockIdx.x << SHIFT) * param.n_height);
    for (int i = threadIdx.x; i < (param.n_height << SHIFT) / (sizeof(float4) / sizeof(T)); i+=blockDim.x) {
        block_output[i] = dbuf.packed[i];
    }
}

template<typename T, int SHIFT>
__global__ void cuda_kernel_fast_trans_shift_cols_with_tail(
    const int32_t num_elems,
    const T *input,
    FastTransposeParam param,
    DivModFast h_mod,
    DivModFast w_mod,
    T *output)
{
    //Same logic as cuda_kernel_fast_trans_shift_cols expect
    //that not all blocks will process exactly 1 << SHIFT cols
    constexpr int MEM = 8192;
    __shared__ union {
        T unpacked[MEM/sizeof(T)];
        float4 packed[MEM/sizeof(float4)];
    } dbuf;
    __shared__ T ibuf[MEM/sizeof(T)];
    int elem_start = (blockIdx.x << SHIFT) * param.n_height;
    if (elem_start > num_elems) return;
    int block_cols = param.n_width * param.n_outer - (blockIdx.x << SHIFT) >= (1 << SHIFT)
        ? (1 << SHIFT) : param.n_width * param.n_outer - (blockIdx.x << SHIFT);
    int img0, img0_start_col, img0_cols, img1;
    w_mod.divmod(blockIdx.x << SHIFT, img0, img0_start_col);
    if (img0_start_col + (1 << SHIFT) >= param.n_width) {
        img1 = img0 + 1;
        img0_cols = param.n_width - img0_start_col;
    } else {
        img0_cols = (1 << SHIFT);
    }
    if (img0_cols == (1 << SHIFT)) {
        const T *img_input = input + img0 * param.n_height * param.n_width;
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        for (int i = t_bid; i < param.n_height; i += (blockDim.x >> SHIFT)) {
            ibuf[(i << SHIFT) + t_tid] = img_input[i * param.n_width + t_tid + img0_start_col];
        }
    } else {
        const T *img0_input = input + img0 * param.n_height * param.n_width;
        const T *img1_input = input + img1 * param.n_height * param.n_width;
        int t_bid = threadIdx.x >> SHIFT;
        int t_tid = threadIdx.x - (t_bid << SHIFT);
        for (int i = t_bid; i < param.n_height; i += (blockDim.x >> SHIFT)) {
            if (t_tid < block_cols) {
                if (t_tid < img0_cols)
                ibuf[(i << SHIFT) + t_tid] = img0_input[i * param.n_width + t_tid + img0_start_col];
                else
                ibuf[(i << SHIFT) + t_tid] = img1_input[i * param.n_width + t_tid - img0_cols];
            }
        }
    }

    __syncthreads();
    for (int i = threadIdx.x; i < param.n_height * block_cols; i+=blockDim.x) {
        int src_w,src_h;
        h_mod.divmod(i, src_w, src_h);
        int src_pixel = src_w + (src_h << SHIFT);
        dbuf.unpacked[i] = ibuf[src_pixel];
    }
    __syncthreads();

    float4* block_output = (float4*)(output + (blockIdx.x << SHIFT) * param.n_height);
    for (int i = threadIdx.x; i < (param.n_height * block_cols) / (sizeof(float4) / sizeof(T)); i+=blockDim.x) {
        block_output[i] = dbuf.packed[i];
    }
}

template<class T>
bool run_kernel_opt(FastTransposeParam param, const T* input, T* output, cudaStream_t stream) {
    int num_elems = param.n_outer * param.n_width * param.n_height;
    bool launched = false;
    if ((param.n_width == 1 || param.n_height == 1) && num_elems % (16 / sizeof(T)) == 0) {
        dim3 dim_block(256,1,1);
        dim3 dim_grid(DivUp(num_elems, (16 / sizeof(T))*256), 1, 1);
        cuda_kernel_fast_trans_b1<<<dim_grid, dim_block, 0, stream>>>(num_elems, (const float4*)input, param, (float4*)output);
        return true;
    }
    int power2 = param.GetMaxPower2(8 / sizeof(T));
    if (sizeof(T) >= sizeof(float4)) return false;
    //In this condition, cuda_kernel_fast_trans_op have better performance
    if (sizeof(T) == sizeof(int8_t) && power2 >= 4) return false;
    if (num_elems % (16 / sizeof(T)) == 0) {
        int lines = param.n_height * param.n_outer;
        int lines_per_block = 8192 / param.n_width / sizeof(T);
        int N = 1;
        //Trying to load 1 << N lines per block
        while (true) if ((1 << (N + 1)) > lines_per_block) break; else N += 1;
        if (N > 9) N = 9;
        bool multi_img = (1 << N) > param.n_height;
        bool no_tail = lines % (1 << N) == 0;
        if (N >= 4) {
            dim3 dim_block(512,1,1);
            dim3 dim_grid(DivUp(param.n_height * param.n_outer, 1 << N), 1, 1);
            if (!multi_img) {
                if (no_tail) {
#define CASE(SHIFT) \
    case SHIFT: \
        cuda_kernel_fast_trans_shift_rows<T, SHIFT><<<dim_grid, dim_block, 0, stream>>>(num_elems, input, param, DivModFast(param.n_height), DivModFast(param.n_width), output); \
        return true;
                    switch (N) {
                        CASE(4)
                        CASE(5)
                        CASE(6)
                        CASE(7)
                        CASE(8)
                        CASE(9)
                    }
#undef CASE
                } else {
#define CASE(SHIFT) \
    case SHIFT: \
        cuda_kernel_fast_trans_shift_rows_with_tail<T, SHIFT><<<dim_grid, dim_block, 0, stream>>>(num_elems, input, param, DivModFast(param.n_height), DivModFast(param.n_width), output); \
        return true;
                    switch (N) {
                        CASE(4)
                        CASE(5)
                        CASE(6)
                        CASE(7)
                        CASE(8)
                        CASE(9)
                    }
#undef CASE
                }
            } else {
                if (no_tail) {
#define CASE(SHIFT) \
    case SHIFT: \
        cuda_kernel_fast_trans_shift_rows_mi<T, SHIFT><<<dim_grid, dim_block, 0, stream>>>(num_elems, input, param, DivModFast(param.n_height), DivModFast(param.n_width), output); \
        return true;
                    switch (N) {
                        CASE(4)
                        CASE(5)
                        CASE(6)
                        CASE(7)
                        CASE(8)
                        CASE(9)
                    }
#undef CASE
                } else {
#define CASE(SHIFT) \
    case SHIFT: \
        cuda_kernel_fast_trans_shift_rows_with_tail_mi<T, SHIFT><<<dim_grid, dim_block, 0, stream>>>(num_elems, input, param, DivModFast(param.n_height), DivModFast(param.n_width), output); \
        return true;
                    switch (N) {
                        CASE(4)
                        CASE(5)
                        CASE(6)
                        CASE(7)
                        CASE(8)
                        CASE(9)
                    }
#undef CASE
                }
            }
        } else {
            int cols = param.n_width * param.n_outer;
            int cols_per_block = 8192 / param.n_height / sizeof(T);
            int N = 1;
            //Trying to load 1 << N lines per block
            while (true) if ((1 << (N + 1)) > cols_per_block) break; else N += 1;
            if (N > 9) N = 9;
            bool multi_img = (1 << N) > param.n_width;
            if (multi_img) return false;
            bool no_tail = cols % (1 << N) == 0;

            if (N >= 4) {
                dim3 dim_block(512,1,1);
                dim3 dim_grid(DivUp(param.n_width * param.n_outer, 1 << N), 1, 1);
                if (no_tail) {
#define CASE(SHIFT) \
    case SHIFT: \
            cuda_kernel_fast_trans_shift_cols<T, SHIFT><<<dim_grid, dim_block, 0, stream>>>(num_elems, input, param, DivModFast(param.n_height), DivModFast(param.n_width), output); \
        return true;
                    switch (N) {
                        CASE(4)
                        CASE(5)
                        CASE(6)
                        CASE(7)
                        CASE(8)
                        CASE(9)
                    }
#undef CASE
                } else {
#define CASE(SHIFT) \
    case SHIFT: \
        cuda_kernel_fast_trans_shift_cols_with_tail<T, SHIFT><<<dim_grid, dim_block, 0, stream>>>(num_elems, input, param, DivModFast(param.n_height), DivModFast(param.n_width), output); \
        return true;
                    switch (N) {
                        CASE(4)
                        CASE(5)
                        CASE(6)
                        CASE(7)
                        CASE(8)
                        CASE(9)
                    }
#undef CASE
                }
            }
        }
    }
    return launched;
}
#endif

bool FastTransposeSupport(
    FastTransposeParam *fast_param,
    const ppl::common::TensorShape *input_shape,
    TransposeKernelParam param,
    const ppl::common::TensorShape *output_shape)
{
    if (input_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY ||
        output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        return false;
    }
    fast_param->reset();
    int num_dims = input_shape->GetDimCount();
    for (int i = 0; i < num_dims; i++) {
        if (param.perm[i] == i) {
            fast_param->n_outer *= input_shape->GetDim(i);
            continue;
        } else {
            fast_param->n_height = input_shape->GetDim(i);
            for (int j = i + 1; j < num_dims; j++) {
                if (param.perm[j - 1] == j) {
                    fast_param->n_width *= input_shape->GetDim(j);
                } else {
                    return false;
                }
            }
            break;
        }
    }
    return true;
}

bool FastTransposeSupport2(
    FastTransposeParam *fast_param,
    const ppl::common::TensorShape *input_shape,
    TransposeKernelParam param,
    const ppl::common::TensorShape *output_shape)
{
    if (input_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY ||
        output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        return false;
    }
    fast_param->reset();
    int num_dims = input_shape->GetDimCount();
    for (int i = 0; i < num_dims; i++) {
        if (param.perm[i] == i) {
            fast_param->n_outer *= input_shape->GetDim(i);
            continue;
        } else {
            fast_param->n_width = input_shape->GetDim(num_dims - 1);
            fast_param->n_height = 1;
            for (int j = i; j < num_dims - 1; j++) {
                if (param.perm[j + 1] == j) {
                    fast_param->n_height *= input_shape->GetDim(j);
                } else {
                    return false;
                }
            }
            break;
        }
    }
    return true;
}
ppl::common::RetCode PPLCUDATransposeFastForwardImp(
    cudaStream_t stream,
    FastTransposeParam param,
    const ppl::common::TensorShape *input_shape,
    const void *input,
    const ppl::common::TensorShape *output_shape,
    void *output)
{
#ifdef PPLNN_USE_MACA
    if (input_shape->GetDimCount() >= 3) {
	int dims = input_shape->GetDimCount();
        //1. if we deem the last two dim as one dim, we can use opt
        //param.n_width = input.rdim0 * input.rdim1 = output.rdim1*output.rdim2
        //param.n_height = input.rdim2 = output.rdim0
        //2. if we just transpose the last two dims
        int input_rdim0 = input_shape->GetDim(dims-1);
        int input_rdim1 = input_shape->GetDim(dims-2);
        int input_rdim2 = input_shape->GetDim(dims-3);
        int output_rdim0 = output_shape->GetDim(dims-1);
        int output_rdim1 = output_shape->GetDim(dims-2);
        int output_rdim2 = output_shape->GetDim(dims-3);
        bool case1_valid = param.n_width == input_rdim0 * input_rdim1 && param.n_width == output_rdim1 * output_rdim2
            && param.n_height == input_rdim2 && param.n_height == output_rdim0;
        bool case2_valid = param.n_height == input_rdim1 && param.n_width == input_rdim0
            && param.n_height == output_rdim0 && param.n_width == output_rdim1;
        if (case1_valid || case2_valid) {
            switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
                case sizeof(int8_t):
                    if (run_kernel_opt<int8_t>(param, (const int8_t*)input, (int8_t*)output, stream)) return ppl::common::RC_SUCCESS;
                    break;
                case sizeof(int16_t):
                    if (run_kernel_opt<int16_t>(param, (const int16_t*)input, (int16_t*)output, stream)) return ppl::common::RC_SUCCESS;
                    break;
                case sizeof(int32_t):
                    if (run_kernel_opt<int32_t>(param, (const int32_t*)input, (int32_t*)output, stream)) return ppl::common::RC_SUCCESS;
                    break;
                case sizeof(int64_t):
                    if (run_kernel_opt<int64_t>(param, (const int64_t*)input, (int64_t*)output, stream)) return ppl::common::RC_SUCCESS;
                    break;
            }
        }
    }
#endif
    dim3 dim_block(DIM, DIM, 1);
    int dimz = param.n_outer >= MAX_DIM ? MAX_DIM : param.n_outer;
    dim3 dim_grid(DivUp(param.n_width, DIM), DivUp(param.n_height, DIM), dimz);

#define SWITCH_CASE(TYPE)                                           \
    case sizeof(TYPE): {                                            \
        cuda_kernel_fast_trans_opt<<<dim_grid, dim_block, 0, stream>>>( \
            (const TYPE *)input, param, (TYPE *)output);            \
        return ppl::common::RC_SUCCESS;                             \
    }
    switch(ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        case 1:
            if((param.n_width & 15) == 0 && (param.n_height&15)!=0) {
                dim3 dimGrid(DivUp(param.n_width, 128), DivUp(param.n_height, 128), dimz);
                dim3 blockSize(8, 128, 1);
                cuda_kernel_fast_trans_opt_int8_unalign<<<dimGrid, blockSize, 0, stream>>>(
                (const int8_t *)input, param, (int8_t *)output,dimGrid.z,dimGrid.y,blockSize.x,blockSize.y);
                return ppl::common::RC_SUCCESS;
            }
            break;
        case 2:
            if((param.n_width&7) == 0 && (param.n_height&7)!=0) {
                dim3 dimGrid(DivUp(param.n_width, 64), DivUp(param.n_height, 64), dimz);
                dim3 blockSize(8, 64, 1);
                cuda_kernel_fast_trans_opt_fp16_unalign<<<dimGrid, blockSize, 0, stream>>>(
                (const half *)input, param, (half *)output,dimGrid.z,dimGrid.y,blockSize.x,blockSize.y);
                return ppl::common::RC_SUCCESS;
            }
        break;
        default:
        break;
    }
    int power2 = param.GetMaxPower2(8 / ppl::common::GetSizeOfDataType(input_shape->GetDataType()));
    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        case sizeof(int8_t):
            switch (power2) {
                case 8:
                    dim_block.x /= 8;
                    cuda_kernel_fast_trans_op<int8_t, int64_t><<<dim_grid, dim_block, 0, stream>>>(
                        (const int8_t *)input, param, (int8_t *)output);
                    break;
                case 4:
                    dim_block.x /= 4;
                    cuda_kernel_fast_trans_op<int8_t, int32_t><<<dim_grid, dim_block, 0, stream>>>(
                        (const int8_t *)input, param, (int8_t *)output);
                    break;
                case 2:
                    dim_block.x /= 2;
                    cuda_kernel_fast_trans_op<int8_t, int16_t><<<dim_grid, dim_block, 0, stream>>>(
                        (const int8_t *)input, param, (int8_t *)output);
                    break;
                default:
                    cuda_kernel_fast_trans_opt<<<dim_grid, dim_block, 0, stream>>>(
                        (const int8_t *)input, param, (int8_t *)output);
                    break;
            }
            return ppl::common::RC_SUCCESS;
            // SWITCH_CASE(int8_t);
        //add for:trans224 in RFBfp16 perm[0,2,3,1] inputshape[64,486,64,64].the case can't access run_kernel_opt option.
        case sizeof(int16_t):
            switch(power2){
                case 2:
                    dim_block.x /= 2;
                    cuda_kernel_fast_trans_op<int16_t, int32_t><<<dim_grid, dim_block, 0, stream>>>(
                        (const int16_t *)input, param, (int16_t *)output);
                    break;
                default:
                    cuda_kernel_fast_trans_opt<<<dim_grid, dim_block, 0, stream>>>(
                        (const int16_t *)input, param, (int16_t *)output);
                    break;
            }
            return ppl::common::RC_SUCCESS;
            //SWITCH_CASE(int16_t); // power2: 4 2
            SWITCH_CASE(int32_t); // power2: 2
            SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#undef SWITCH_CASE
}

template <typename T>
__global__ void cuda_kernel_middle_trans(
    const T *input,
    int64_t num_elems,
    FastTransposeParam param,
    T *output)
{
    int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_elems)
        return;
    int inner_idx  = tid % param.n_inner;
    int width_idx  = (tid / param.n_inner) % param.n_width;
    int height_idx = (tid / (param.n_inner * param.n_width)) % param.n_height;
    int outer_idx  = tid / (param.n_inner * param.n_width * param.n_height);

    int64_t offset = outer_idx * param.n_inner * param.n_width * param.n_height + height_idx * param.n_inner +
                     width_idx * param.n_height * param.n_inner + inner_idx;
    output[offset] = input[tid];
}

bool MiddleFastTransposeSupport(
    FastTransposeParam *fast_param,
    const ppl::common::TensorShape *input_shape,
    TransposeKernelParam param,
    const ppl::common::TensorShape *output_shape)
{
    if (input_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY ||
        output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        return false;
    }
    fast_param->reset();
    int num_dims    = input_shape->GetDimCount();
    int height_axis = 0;
    int width_axis  = num_dims - 1;
    for (int i = 0; i < num_dims && param.perm[i] == i; fast_param->n_outer *= input_shape->GetDim(i), height_axis = i + 1, i++)
        ;
    for (int i = num_dims - 1; i >= 0 && param.perm[i] == i; fast_param->n_inner *= input_shape->GetDim(i), width_axis = i - 1, i--)
        ;
    if (width_axis <= height_axis)
        return false;
    fast_param->n_height *= input_shape->GetDim(height_axis);
    fast_param->n_width *= input_shape->GetDim(width_axis);
    if (width_axis - height_axis != 1)
        return false;
    return true;
}

ppl::common::RetCode PPLCUDATransposeMiddleFastForwardImp(
    cudaStream_t stream,
    FastTransposeParam param,
    const ppl::common::TensorShape *input_shape,
    const void *input,
    const ppl::common::TensorShape *output_shape,
    void *output)
{
    const int block_size = 256;
    dim3 dim_block(block_size, 1, 1);
    int64_t num_elems = output_shape->CalcElementsIncludingPadding();
    dim3 dim_grid(DivUp(num_elems, block_size), 1, 1);

#define SWITCH_CASE(TYPE)                                             \
    case sizeof(TYPE): {                                              \
        cuda_kernel_middle_trans<<<dim_grid, dim_block, 0, stream>>>( \
            (const TYPE *)input, num_elems, param, (TYPE *)output);   \
        return ppl::common::RC_SUCCESS;                               \
    }

    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        SWITCH_CASE(int8_t);
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#undef SWITCH_CASE
}

template <typename T>
__global__ void ppl_cukernel_transpose(
    int64_t num_elems,
    int num_dims,
    GArray<DivModFast> input_strides_fast,
    GArray<int64_t> output_flip_strides,
    const T *input,
    T *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int64_t output_offset = 0;
    int idx, remain = index;
    for (int it = 0; it < num_dims; ++it) {
        input_strides_fast[it].divmod(remain, idx, remain);
        output_offset += idx * output_flip_strides[it];
    }
    output[output_offset] = input[index];
}

template <typename T>
__global__ void ppl_cukernel_transpose_nhwc(
    int64_t num_elems,
    int num_dims,
    GArray<DivModFast> input_strides_fast,
    GArray<int64_t> input_strides,
    GArray<int64_t> output_flip_strides,
    const T *input,
    T *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int64_t input_offset  = 0;
    int64_t output_offset = 0;
    int idx, remain = index;
    for (int it = 0; it < num_dims; ++it) {
        input_strides_fast[it].divmod(remain, idx, remain);
        input_offset += idx * input_strides[it];
        output_offset += idx * output_flip_strides[it];
    }
    output[output_offset] = input[input_offset];
}

__global__ void ppl_cukernel_transpose_nhwc16_opt(
    int64_t num_elems,
    int num_dims,
    GArray<int> perm,
    GArray<DivModFast> output_strides_fast,
    GArray<int64_t> input_strides,
    const int8_t *input,
    int8_t *output)
{
    constexpr int MEM = 8192;
    __shared__ union {
        int8_t unpacked[MEM/sizeof(int8_t)];
        float4 packed[MEM/sizeof(float4)];
    } dbuf;
    int t_bid = blockIdx.x * blockDim.x;
    int t_tid = threadIdx.x;
    int block_offset = t_tid * 16;
    int block_output = t_bid * 16;
    int64_t output_offset = block_output + block_offset;
    if (output_offset >= num_elems)
        return;
    for (int i = 0; i < 16; i++){
        int64_t input_offset  = 0;
        int idx, remain = output_offset + i;
        for (int it = 0; it < num_dims; ++it) {
            output_strides_fast[it].divmod(remain, idx, remain);
            int i = perm[it];
            input_offset += idx * input_strides[i];
        }
        dbuf.unpacked[block_offset + i] = input[input_offset];
    }
    __syncthreads();
    float4* output_16x = (float4*) (output + block_output);
    output_16x[t_tid] = dbuf.packed[t_tid];
}

template <typename T>
__global__ void ppl_cukernel_transpose_nhwc_0321_opt(    // 0321 in nhwc equal nhw(c+p) --> nhcw
    const T *input, T *output, int width, int height, int channel, 
    int p_channels, int times, DivModFast fastblock, DivModFast fasttimes)
{
    constexpr int MEM = 8192;
    __shared__ union {
        T unpacked[MEM/sizeof(T)];
        float4 packed[MEM/sizeof(float4)];
    } dbuf;
    int bh_idx = blockIdx.z;
    int w_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int c_start = blockIdx.x * times;
    if (c_start >= p_channels) return;

    int bh_start = bh_idx * p_channels * width;     // input_offset
    int input_offset = w_idx * p_channels + c_start;
    float4 input_mx  = *((float4*) (input + bh_start + input_offset));
    T *ptr_data = (T*)&input_mx;
    
    for (int i = 0 ; i < times; i++){
        int c_idx = c_start + i;
        if (c_idx < channel) 
            dbuf.unpacked[i * width + w_idx] = ptr_data[i];
    }
    __syncthreads();
    int share_start = (bh_idx * channel + c_start) * width;   // output_offset
    float4* output_mx = (float4*) (output + share_start);
    int remain, quo;
    fastblock.divmod(w_idx, quo, w_idx);
    fasttimes.divmod(w_idx, quo, remain);
    if (c_start + remain < channel) {
        int float4_offset = quo + remain * (blockDim.x / times);
        output_mx[float4_offset] = dbuf.packed[float4_offset];
    }

}


bool transpose5d_opt_support_int8(TransposeKernelParam param,
    const ppl::common::TensorShape *input_shape,
    const ppl::common::TensorShape *output_shape
    )
{
    if (input_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY ||
        output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        return false;
    }
    if (input_shape->GetDataType() != ppl::common::DATATYPE_INT8) return false;

    int num_dims = output_shape->GetDimCount();
    if (num_dims != 5)
    {
        return false;
    }

    if (param.perm[0] != 0)
    {
        return false;
    }
    if (param.perm[1] != 3)
    {
        return false;
    }
    if (param.perm[2] != 4)
    {
        return false;
    }
    if (param.perm[3] != 1)
    {
        return false;
    }
    if (param.perm[4] != 2)
    {
        return false;
    }

    return true;
}

bool transpose5d_opt_support_fp16(TransposeKernelParam param,
    const ppl::common::TensorShape *input_shape,
    const ppl::common::TensorShape *output_shape
    )
{
    if (input_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY ||
        output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY)
    {
        return false;
    }

    if (input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) return false;
    if (output_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) return false;

    int num_dims = output_shape->GetDimCount();
    if (num_dims != 5)
    {
        return false;
    }

    if (param.perm[0] != 0)
    {
        return false;
    }
    if (param.perm[1] != 3)
    {
        return false;
    }
    if (param.perm[2] != 4)
    {
        return false;
    }
    if (param.perm[3] != 1)
    {
        return false;
    }
    if (param.perm[4] != 2)
    {
        return false;
    }

    return true;
}

bool transpose3d_opt_support_fp16(TransposeKernelParam param,
    const ppl::common::TensorShape *input_shape,
    const ppl::common::TensorShape *output_shape
    )
{
    if (input_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY ||
        output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        return false;
    }
    if (input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        return false;
    }
    int num_dims = output_shape->GetDimCount();
    if (num_dims != 3)
    {
        return false;
    }

    if (param.perm[0] != 0)
    {
        return false;
    }
    if (param.perm[1] != 2)
    {
        return false;
    }
    if (param.perm[2] != 1)
    {
        return false;
    }

    return true;
}

#define TILE  32

__global__ void transpose5d_opt_int8(const int8_t *input, int8_t *output, int batch, int height, int width)
{
    __shared__ int8_t sm[TILE][TILE + 1];

    for (int n = blockIdx.z; n <batch; n += gridDim.z) {
        const int8_t *input_start = (int8_t *)(input + n *height * width);
        int8_t *output_start      = (int8_t *)(output + n *height * width);
        int64_t w_index = blockIdx.x * blockDim.x + threadIdx.x;
        int64_t h_index = blockIdx.y * blockDim.y + threadIdx.y;
        if (w_index < width && h_index < height)
        {
            int64_t input_offset = h_index * width + w_index;
            sm[threadIdx.y][threadIdx.x] = input_start[input_offset];
        }
        __syncthreads();

        w_index = blockIdx.y * blockDim.y + threadIdx.x ;
        h_index = blockIdx.x * blockDim.x + threadIdx.y;
        if (w_index < height && h_index < width)
        {
            int64_t out_offset = h_index * height + w_index;
            output_start[out_offset] = sm[threadIdx.x][threadIdx.y];
        }
    }
}

__global__ void transpose5d_opt_fp16(const half *input, half *output, int batch, int height, int width)
{
    __shared__ half sm[TILE][TILE + 1];

    for (int n = blockIdx.z; n <batch; n += gridDim.z) {
        const half *input_start = (half *)(input + n *height * width);
        half *output_start      = (half *)(output + n *height * width);

        for (int64_t t = blockIdx.y; t < DivUp(height, TILE); t += gridDim.y) {
            int64_t w_index = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t h_index = t * blockDim.y + threadIdx.y;
            if (w_index < width && h_index < height)
            {
                int64_t input_offset = h_index * width + w_index;
                sm[threadIdx.y][threadIdx.x] = input_start[input_offset];
            }

            __syncthreads();

            w_index = t * blockDim.y + threadIdx.x ;
            h_index = blockIdx.x * blockDim.x + threadIdx.y;
            if (w_index < height && h_index < width)
            {
                int64_t out_offset = h_index * height + w_index;
                output_start[out_offset] = sm[threadIdx.x][threadIdx.y];
            }
        }
    }
}

__global__ void transpose3d_opt_fp16(const half *input, half *output, int batch, int height, int width, int HW, DivModFast fastheight, DivModFast fastwidth)
{
    __shared__ union {
        half unpacked[4096];
        float4 packed[512];
    } obuf;

    int b_idx = blockIdx.x;
    int start_offset = b_idx * height * width;
    int wt_idx = blockIdx.y;
    int wt_offset = wt_idx * 4096;
    half *input_start  = (half *)(input + start_offset);
    float4 *output_start = (float4 *)(output + start_offset + wt_offset);

    int tid = threadIdx.x;
    int inner_hw_idx = tid * 8;
    for (int i = 0; i < 8; i++){
        int hw_idx = wt_offset + inner_hw_idx + i;
        if (hw_idx < HW){
            int h_idx, w_idx;
            fastheight.divmod(hw_idx, w_idx, h_idx);
            int input_offset = h_idx * width + w_idx;
            obuf.unpacked[inner_hw_idx + i] = input_start[input_offset];
        }
    }

    __syncthreads();
    if (wt_offset + inner_hw_idx < HW)
        output_start[tid] = obuf.packed[tid];
}

bool transpose_0321_support_fp16(TransposeKernelParam param,   const ppl::common::TensorShape *input_shape,   const ppl::common::TensorShape *output_shape)
{

    if (input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) return false;
    if (output_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) return false;

    int num_dims = output_shape->GetDimCount();
    if (num_dims != 4)
    {
        return false;
    }

    if (param.perm[0] != 0)
    {
        return false;
    }
    if (param.perm[1] != 3)
    {
        return false;
    }
    if (param.perm[2] != 2)
    {
        return false;
    }
    if (param.perm[3] != 1)
    {
        return false;
    }

    return true;
}

bool FastTransposeSupport3(
    const ppl::common::TensorShape *input_shape,
    TransposeKernelParam param,
    const ppl::common::TensorShape *output_shape)
{
    int num_dims    = input_shape->GetDimCount();
    if (num_dims != 4) return false;
    if (param.perm[1] != 3) return false;
    if (param.perm[2] != 2) return false;
    if (param.perm[3] != 1) return false;
    if (param.perm[0] != 0) return false;
    return true;
}

template<typename T, typename VT, int N, int HEIGHT, int FAST_CHOSE, int CACHE_SIZE>
__global__ void transpose_0321_kernel_opt(
    const T* input,
    T* output,
    int width,
    int height,
    int channel,
    int blockDim_x,
    int blockDim_y,
    int blockDim_z,
    DivModFast grid_width_fast
    )
{
    int b_idx = blockIdx.z;
    int image_size = width*height;
    int channel_size = image_size * channel;
    int batch_offset = b_idx * channel_size;
    int dst_image_size = height*channel;
    __shared__ T sm_buffer[CACHE_SIZE][HEIGHT][CACHE_SIZE + 1];
    __shared__ T sm_buffer_t[CACHE_SIZE][HEIGHT][CACHE_SIZE + 1];
    const T* ptr_input = input + batch_offset;
    T* ptr_output = output + batch_offset;
    int w_offset, h_offset, c_offset;
    int w_idx, h_idx, mblock_width, mblock_channel;
    int block_width, block_height, block_channel;
    grid_width_fast.divmod(blockIdx.x, h_idx, w_idx);
    mblock_width = blockDim_x * N;
    mblock_channel = blockDim_z;
    w_offset = w_idx * mblock_width;
    h_offset = h_idx * blockDim_y;
    c_offset = blockIdx.y * mblock_channel;
    block_width = min(width - w_offset, mblock_width);
    block_height = min(height - h_offset, blockDim_y);
    block_channel = min(channel - c_offset, mblock_channel);
    if(block_width <=0 || block_height <= 0 || block_channel <= 0) return;

    const T* ptr_threadinput = ptr_input + c_offset * image_size + h_offset * width + w_offset;
    T* ptr_threadoutput = ptr_output + w_offset * dst_image_size + h_offset * channel + c_offset;
    if(threadIdx.z < block_channel && threadIdx.y < block_height) {
        const T* ptr_local = ptr_threadinput + threadIdx.z * image_size + threadIdx.y * width;
        for(int i = threadIdx.x; i < block_width; i += blockDim_x) {
            sm_buffer[threadIdx.z][threadIdx.y][i] = *(ptr_local + i);
        }
    }
    __syncthreads();
    if(threadIdx.z < block_channel && threadIdx.y < block_height) {
        for(int i = threadIdx.x; i < block_width; i += blockDim_x) {
            sm_buffer_t[i][threadIdx.y][threadIdx.z] = sm_buffer[threadIdx.z][threadIdx.y][i];
        }
    }
    __syncthreads();

    
    if(threadIdx.z < block_width && threadIdx.y < block_height) {
        if constexpr(FAST_CHOSE) {
            VT reg_dst;
            T* ptr_reg_dst = (T*)&reg_dst;
            int c = threadIdx.x * N;
            if(c >= block_channel) return;
            #pragma unroll N
            for(int i = 0; i < N; i++) {
                ptr_reg_dst[i] = sm_buffer_t[threadIdx.z][threadIdx.y][c + i];
            }
            *(VT*)(ptr_threadoutput + threadIdx.z * dst_image_size + threadIdx.y * channel + c) = reg_dst;
        } else {
            T* ptr_local_output = ptr_threadoutput + threadIdx.z * dst_image_size + threadIdx.y * channel;
            for(int c = threadIdx.x; c < block_channel; c += blockDim_x) {
                *(T*)(ptr_local_output + c) = sm_buffer_t[threadIdx.z][threadIdx.y][c];
            }
        }
    }
}

__global__ void transpose_0321_kernel_fp16(const half *input, half *output, int width, int height, int channel)
{
    int batchIndex = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channel; c++) {
            int inputIndex = ((batchIndex * channel + c) * height + y) * width + x;
            int outputIndex = ((batchIndex * width + x) * height + y) * channel + c;
            output[outputIndex] = input[inputIndex];
        }
    }
}


ppl::common::RetCode PPLCUDATransposeForwardImp(
    cudaStream_t stream,
    TransposeKernelParam param,
    const ppl::common::TensorShape *input_shape,
    const void *input,
    const ppl::common::TensorShape *output_shape,
    void *output)
{

    if (transpose5d_opt_support_int8(param, input_shape, output_shape))
    {
        dim3 dim_block(TILE, TILE, 1);
        int batch = input_shape->GetDim(0);
        int height = input_shape->GetDim(1)*input_shape->GetDim(2);
        int width = input_shape->GetDim(3)*input_shape->GetDim(4);
        dim3 dim_grid(DivUp(width, TILE), DivUp(height, TILE), batch);
        transpose5d_opt_int8<<<dim_grid, dim_block, 0, stream>>>((const int8_t *)input, (int8_t *)output, batch, height, width);
        return ppl::common::RC_SUCCESS;

    }

    if (transpose5d_opt_support_fp16(param, input_shape, output_shape))
    {
        dim3 dim_block(TILE, TILE, 1);
        int batch = input_shape->GetDim(0);
        int height = input_shape->GetDim(1)*input_shape->GetDim(2);
        int width = input_shape->GetDim(3)*input_shape->GetDim(4);
        dim3 dim_grid(DivUp(width, TILE), DivUp(height, TILE), batch);
        transpose5d_opt_fp16<<<dim_grid, dim_block, 0, stream>>>((const half *)input, (half *)output, batch, height, width);
        return ppl::common::RC_SUCCESS;
    }

    int num_dims      = output_shape->GetDimCount();
    FastTransposeParam fast_param;
    if (FastTransposeSupport(&fast_param, input_shape, param, output_shape)) {
        return PPLCUDATransposeFastForwardImp(stream, fast_param, input_shape, input, output_shape, output);
    } else if (FastTransposeSupport2(&fast_param, input_shape, param, output_shape)) {
        return PPLCUDATransposeFastForwardImp(stream, fast_param, input_shape, input, output_shape, output);
    } else if (MiddleFastTransposeSupport(&fast_param, input_shape, param, output_shape)) {
        return PPLCUDATransposeMiddleFastForwardImp(stream, fast_param, input_shape, input, output_shape, output);
    } else if (FastTransposeSupport3(input_shape, param, output_shape)) {
        int block_size = 512;
        int batch = input_shape->GetDim(0);
        int channel = input_shape->GetDim(1);
        int height = input_shape->GetDim(2);
        int width = input_shape->GetDim(3);
        if (input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY){
            dim3 blockSize(8,2,32);
            int block_x = DivUp(width, 32);
            int block_y = DivUp(height,2); 
            dim3 gridSize(block_x * block_y,DivUp(channel, 32),batch);
            if (input_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
                if((channel & 3) == 0) {
                    transpose_0321_kernel_opt<int8_t, float, 4, 2, 1, 32><<<gridSize, blockSize, 0, stream>>>((const int8_t*)input, (int8_t*)output,width,height,channel, blockSize.x, blockSize.y, blockSize.z, DivModFast(block_x));
                } else {
                    transpose_0321_kernel_opt<int8_t, float, 4, 2, 0, 32><<<gridSize, blockSize, 0, stream>>>((const int8_t*)input, (int8_t*)output,width,height,channel, blockSize.x, blockSize.y, blockSize.z, DivModFast(block_x));
                }
                return ppl::common::RC_SUCCESS;
            } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16){
                if((channel & 3) == 0) {
                    transpose_0321_kernel_opt<half, float2, 4, 2, 1, 32><<<gridSize, blockSize, 0, stream>>>((const half*)input, (half*)output,width,height,channel, blockSize.x, blockSize.y, blockSize.z, DivModFast(block_x));
                } else {
                    transpose_0321_kernel_opt<half, float2, 4, 2, 0, 32><<<gridSize, blockSize, 0, stream>>>((const half*)input, (half*)output,width,height,channel, blockSize.x, blockSize.y, blockSize.z, DivModFast(block_x));
                }
                return ppl::common::RC_SUCCESS;
            } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32){
                if((channel & 3) == 0) {
                    transpose_0321_kernel_opt<float, float4, 4, 2, 1, 32><<<gridSize, blockSize, 0, stream>>>((const float*)input, (float*)output,width,height,channel, blockSize.x, blockSize.y, blockSize.z, DivModFast(block_x));
                } else {
                    transpose_0321_kernel_opt<float, float4, 4, 2, 0, 32><<<gridSize, blockSize, 0, stream>>>((const float*)input, (float*)output,width,height,channel, blockSize.x, blockSize.y, blockSize.z, DivModFast(block_x));
                }
                return ppl::common::RC_SUCCESS;
            }
        } else if (input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8){
            int p_channels = input_shape->GetPadding0(1) + input_shape->GetPadding1(1) + channel;
            if(p_channels > channel){
                int bh_capity = batch * height;
                DivModFast fastblock(block_size);
                DivModFast fasttimes(8);
                dim3 grid_size(DivUp(channel, 8), DivUp(width, block_size), bh_capity);
                ppl_cukernel_transpose_nhwc_0321_opt<<<grid_size, block_size>>>((const half *)input, (half *)output, 
                    width, height, channel, p_channels, 8, fastblock, fasttimes);
                return ppl::common::RC_SUCCESS;
            }else{
                int WC = channel * width;
                dim3 dim_grid(batch * height, DivUp(WC, 4096), 1);
                dim3 dim_block(512, 1, 1);
                transpose3d_opt_fp16<<<dim_grid, dim_block, 0, stream>>>((const half *)input, (half *)output, batch, width, channel, WC, DivModFast(width), DivModFast(channel));
                return ppl::common::RC_SUCCESS;
            }
        } else if (input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16){
            int bh_capity = batch * height;
            DivModFast fastblock(block_size);
            DivModFast fasttimes(8);
            int p_channels = input_shape->GetPadding0(1) + input_shape->GetPadding1(1) + channel;
            dim3 grid_size(DivUp(channel, 16), DivUp(width, block_size), bh_capity);
            ppl_cukernel_transpose_nhwc_0321_opt<<<grid_size, block_size>>>((const int8_t *)input, (int8_t *)output, 
                width, height, channel, p_channels, 16, fastblock, fasttimes);
            return ppl::common::RC_SUCCESS;
        }
    } else if(output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 && num_dims == 4) {
        int64_t acc_output_stride = 1;
        int64_t acc_input_stride  = 1;
        GArray<DivModFast> output_strides_fast(num_dims);
        GArray<int64_t> input_strides(num_dims);
        GArray<int64_t> output_strides(num_dims);
    
        GArray<int> perm(num_dims);
        for (int it = num_dims - 1; it >= 0; --it) {
            if (it == num_dims - 1) {
                input_strides[it]  = acc_input_stride;
                output_strides[it] = acc_output_stride;
                output_strides_fast[it] = DivModFast(acc_output_stride);
                acc_input_stride *= input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
                acc_output_stride *= output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
            } else {
                input_strides[it]  = acc_input_stride;
                output_strides[it] = acc_output_stride;
                output_strides_fast[it] = DivModFast(acc_output_stride);
                acc_input_stride *= input_shape->GetDim(it + 1);
                acc_output_stride *= output_shape->GetDim(it + 1);
            }
            perm[it] = param.perm[it];
        }
        int block_size = 512;
        int hc_per_block = block_size * 16;
        int64_t num_elems = output_strides[0] * input_shape->GetDim(0);
        int grid_size = DivUp(num_elems, hc_per_block); 
        ppl_cukernel_transpose_nhwc16_opt<<<grid_size, block_size, 0, stream>>>(
            num_elems, num_dims, perm, output_strides_fast, input_strides, (const int8_t *)input, (int8_t *)output);
        return ppl::common::RC_SUCCESS;  
    } else if (transpose3d_opt_support_fp16(param, input_shape, output_shape)) {
        int batch = input_shape->GetDim(0);
        int height = input_shape->GetDim(1);
        int width = input_shape->GetDim(2);

        int HW = height * width;
        dim3 dim_grid(batch, DivUp(HW, 4096), 1);
        dim3 dim_block(512, 1, 1);
        transpose3d_opt_fp16<<<dim_grid, dim_block, 0, stream>>>((const half *)input, (half *)output, batch, height, width, HW, DivModFast(height), DivModFast(width));
        return ppl::common::RC_SUCCESS;
    }

    int64_t num_elems = output_shape->CalcElementsExcludingPadding();

    GArray<DivModFast> input_strides_fast(num_dims);
    GArray<int64_t> input_strides(num_dims);
    GArray<int64_t> output_strides(num_dims);
    int64_t acc_output_stride = 1;
    int64_t acc_input_stride  = 1;
    for (int it = num_dims - 1; it >= 0; --it) {
        input_strides_fast[it] = DivModFast(acc_input_stride);
        output_strides[it]     = acc_output_stride;
        acc_input_stride *= input_shape->GetDim(it);
        acc_output_stride *= output_shape->GetDim(it);
    }
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
        acc_input_stride  = 1;
        acc_output_stride = 1;
        for (int it = num_dims - 1; it >= 0; --it) {
            if (it == num_dims - 1) {
                input_strides[1]  = acc_input_stride;
                output_strides[1] = acc_output_stride;
                acc_input_stride *= input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
                acc_output_stride *= output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
            } else if (it == 0) {
                input_strides[it]  = acc_input_stride;
                output_strides[it] = acc_output_stride;
                acc_input_stride *= input_shape->GetDim(it);
                acc_output_stride *= output_shape->GetDim(it);
            } else {
                input_strides[it + 1]  = acc_input_stride;
                output_strides[it + 1] = acc_output_stride;
                acc_input_stride *= input_shape->GetDim(it + 1);
                acc_output_stride *= output_shape->GetDim(it + 1);
            }
        }
    }
    GArray<int64_t> output_flip_strides(num_dims);
    for (int i = 0; i < num_dims; ++i) {
        for (int j = 0; j < num_dims; ++j) {
            if (param.perm[j] == i) {
                output_flip_strides[i] = output_strides[j];
            }
        }
    }

    int block_size = 256;
    int grid_size  = (num_elems + block_size - 1) / block_size;

#define SWITCH_CASE(TYPE)                                                                                                          \
    case sizeof(TYPE): {                                                                                                           \
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {                                                      \
            ppl_cukernel_transpose_nhwc<<<grid_size, block_size, 0, stream>>>(                                                     \
                num_elems, num_dims, input_strides_fast, input_strides, output_flip_strides, (const TYPE *)input, (TYPE *)output); \
        } else {                                                                                                                   \
            ppl_cukernel_transpose<<<grid_size, block_size, 0, stream>>>(                                                          \
                num_elems, num_dims, input_strides_fast, output_flip_strides, (const TYPE *)input, (TYPE *)output);                \
        }                                                                                                                          \
        return ppl::common::RC_SUCCESS;                                                                                            \
    }

    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        SWITCH_CASE(int8_t);
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#undef SWITCH_CASE
}
