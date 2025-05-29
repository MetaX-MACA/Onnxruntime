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

#include "cuda_fp16.h"
using namespace PPLCUDA;
using namespace ppl::common;

#ifdef __MACACC__
#define OPT_CVT
#endif//__MACACC__

#ifdef PPLNN_USE_MACA
#define DIM 16
#else
#define DIM 32
#endif

#define LEASTCHANNEL 16
template <typename T, CVTFormatMode mode>
__global__ void cuda_kernel_cvtformat(
    T* input,
    T* output,
    ReFormatParam param)
{
}

#define cvtC16TOC8(type)                                                                               \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat<type, NHWC16_NHWC8>(                                              \
    type* input,                                                                                        \
    type* output,                                                                                       \
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
            output[dst_offset] = input[src_offset];                   \
        }                                                                                               \
    }                                                                                                   \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtC16TOC8(half)
#endif
    cvtC16TOC8(float)
    cvtC16TOC8(char)
    cvtC16TOC8(double)
    cvtC16TOC8(int8_t)

#define cvtC8TOC16(type)                                                                               \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat<type, NHWC8_NHWC16>(                                              \
    type* input,                                                                                        \
    type* output,                                                                                       \
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
            output[dst_offset] = idx_w < param.src_pad ? input[src_offset] : type(0);                   \
        }                                                                                               \
    }                                                                                                   \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtC8TOC16(half)
#endif
    cvtC8TOC16(float)
    cvtC8TOC16(char)
    cvtC8TOC16(double)
    cvtC8TOC16(int8_t)

#ifdef OPT_CVT
#define MIN(a,b) ((a) < (b) ? (a) :(b))
//support NDARRAY_NHWC
template<typename T1,typename T2>
__global__ void cuda_kernel_cvtformat_opt_ndarray_nhwc(
    T1* input,
    T1* output,
    ReFormatParam param)
{
    constexpr int times = sizeof(T2) / sizeof(T1);
    __shared__ union {
        T1 m1[64][64 + times];
        T2 m2[64][64 / times + 1];
    } sm_buffer;
    int64_t num = blockIdx.z;
    for(int n = num; n < param.n_outer; n += gridDim.z) {
        int64_t idx_w = (blockIdx.x * blockDim.x + threadIdx.x) * times;
        int64_t idx_h = blockIdx.y * blockDim.y + threadIdx.y;
        T1* ptr_input = (T1 *)(input + n * param.src_pad * param.n_inner);
        T1* ptr_output = (T1 *)(output + n * param.dst_pad * param.n_inner);
        if(idx_w < param.n_inner && idx_h < param.src_pad) {
            int64_t offset = idx_h * param.n_inner + idx_w;
            T2 *ptr_input_v = (T2*)(ptr_input + offset);
            sm_buffer.m2[threadIdx.y][threadIdx.x] = *ptr_input_v;
        }
        else{
            T2 tmp1;
            T1* ptr_tmp1 = (T1*)&tmp1;
            #pragma unroll times
            for(int i = 0; i < times; i++)
            {
                ptr_tmp1[i] = (T1)0;
            }
            sm_buffer.m2[threadIdx.y][threadIdx.x] = tmp1;
        }
        __syncthreads();

        idx_w = blockIdx.y * blockDim.y + threadIdx.x * times;
        idx_h = (blockIdx.x * blockDim.x) * times + threadIdx.y;
        
        if(idx_w < param.dst_pad && idx_h < param.n_inner)
        {
            union {
                T2 m2;
                T1 m1[times];
            }tmp_storage;
            int64_t offset = idx_h * param.dst_pad + idx_w;
            #pragma unroll times
            for(int i = 0; i < times; i++)
            {
                tmp_storage.m1[i] = sm_buffer.m1[threadIdx.x*times + i][threadIdx.y];
            }
            *(T2*)(ptr_output + offset) = tmp_storage.m2;
        }
    }
}

//support NDARRAY_NHWC
template<typename T1, typename T2, int N, int SHIFT>
__global__ void cuda_kernel_cvtformat_opt_ndarray_nhwc_unalign(
    void* input,
    void* output,
    ReFormatParam param)
{
    __shared__ union {
        T1 m1[64][64 + (N*3)];
        T2 m2[64][(64>>SHIFT) + 3];
    } sm_buffer;

    int n = blockIdx.z;
    int64_t src_offset = n * param.src_pad * param.n_inner;
    int64_t dst_offset = n * param.dst_pad * param.n_inner;
    int64_t idx_w_0 = (blockIdx.x * blockDim.x) << SHIFT;
    int64_t width = min(param.n_inner - idx_w_0, blockDim.x << SHIFT);
    if(width <= 0) return;
    int64_t idx_h = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t iIndex = src_offset + idx_h * param.n_inner + idx_w_0;
    int64_t idx_w;
    float4 reg_zero = make_float4(0,0,0,0);
    T2 reg_0 = *(T2*)&reg_zero;
    T1* ptr_output = (T1 *)(output) + dst_offset;

    if(blockIdx.x == gridDim.x - 1) 
    {
        T1* ptr_input = (T1*)input + iIndex;
        if(idx_h < param.src_pad) {
            for(int i = threadIdx.x; i < width; i += blockDim.x) {
                T1 reg = *(ptr_input + i);
                sm_buffer.m1[threadIdx.y][i] = reg;
            }
        }else{
            for(int i = threadIdx.x; i < width; i += blockDim.x) {
                sm_buffer.m1[threadIdx.y][i] = (T1)0;
            }
        }
        __syncthreads();

        idx_h = ((blockIdx.x * blockDim.x) << SHIFT) + threadIdx.y;
        idx_w = blockIdx.y * blockDim.y + (threadIdx.x << SHIFT);
        if(idx_w < param.dst_pad && idx_h < param.n_inner)
        {
            union {
                T2 m2;
                T1 m1[N];
            }tmp_storage;
            int64_t offset = idx_h * param.dst_pad + idx_w;
            #pragma unroll N
            for(int i = 0; i < N; i++)
            {
                tmp_storage.m1[i] = sm_buffer.m1[(threadIdx.x << SHIFT) + i][threadIdx.y];
            }
            *(T2*)(ptr_output + offset) = tmp_storage.m2;
        }
    } 
    else 
    {
        int64_t addr = iIndex & (~(N - 1));
        T1 * ptr_input = (T1*)input + addr;
        int diff = iIndex - addr;
        int blockwidth = width  + diff + 1;
        blockwidth = (blockwidth + N - 1) >> SHIFT;

        __shared__ int sm_offset[64][8];
        sm_offset[threadIdx.y][threadIdx.x] = diff;

        if(idx_h < param.src_pad) {
            for(int i = threadIdx.x; i < blockwidth; i += blockDim.x) {
                T2 reg_input = *(T2*)(ptr_input + (i << SHIFT));
                sm_buffer.m2[threadIdx.y][i] = reg_input;
            }
        } else {
            for(int i = threadIdx.x; i < blockwidth; i += blockDim.x) {
                sm_buffer.m2[threadIdx.y][i] = reg_0;
            }
        }
        __syncthreads();
        idx_w = blockIdx.y * blockDim.y + (threadIdx.x << SHIFT);
        idx_h = (blockIdx.x * blockDim.x << SHIFT) + threadIdx.y;
        if(idx_w < param.dst_pad && idx_h < param.n_inner) {
            union {
                T2 m2;
                T1 m1[N];
            } tmp_storage;
            int n0 = threadIdx.x << SHIFT;
            int64_t offset = idx_h * param.dst_pad + idx_w;
            #pragma unroll N
            for(int i = 0; i < N; i++)
            {
                int ld = sm_offset[n0 + i][0];
                tmp_storage.m1[i] = sm_buffer.m1[n0 + i][threadIdx.y + ld];
            }
            *(T2*)(ptr_output + offset) = tmp_storage.m2;
        }
    }
}

//only support nhwc->nchw
template<typename T1, typename T2>
__global__ void cuda_kernel_cvtformat_opt_nhwc_ndarray(
    T1* input,
    T1* output,
    int64_t block_num,
    ReFormatParam param)
{
    constexpr int times = sizeof(T2) / sizeof(T1);
    __shared__ union {
        T1 m1[64][64 + times];
        T2 m2[64][64 / times + 1];
    } sm_buffer;
    int64_t num = blockIdx.z;
    for(int n = num; n < param.n_outer; n += gridDim.z) {
        for(int t = blockIdx.y; t < block_num ; t += gridDim.y) {
            int64_t idx_w = (blockIdx.x * blockDim.x + threadIdx.x)*times;
            int64_t idx_h = t * blockDim.y + threadIdx.y;
            T1* ptr_input = (T1 *)(input + n * param.n_inner * param.src_pad);
            T1* ptr_output = (T1 *)(output + n * param.dst_pad * param.n_inner);
            if(idx_w < param.src_pad && idx_h < param.n_inner) {
                int offset = idx_h * param.src_pad + idx_w;
                sm_buffer.m2[threadIdx.y][threadIdx.x] = *(T2*)(ptr_input + offset);
            } else {
                T2 tmp1;
                T1* ptr_tmp1 = (T1*)&tmp1;
                #pragma unroll times
                for(int i = 0; i < times; i++)
                {
                    ptr_tmp1[i] = (T1)0;
                }
                sm_buffer.m2[threadIdx.y][threadIdx.x] = tmp1;
            }
            __syncthreads();
            idx_w = t * blockDim.y + threadIdx.x * times;
            idx_h = blockIdx.x * blockDim.x * times + threadIdx.y;
            if (idx_w < param.n_inner && idx_h < param.dst_pad) {
                union {
                    T2 m2;
                    T1 m1[times];
                }tmp_storage;
                int64_t offset = idx_h * param.n_inner + idx_w;
                #pragma unroll times
                for(int i = 0; i < times; i++)
                {
                    tmp_storage.m1[i] = sm_buffer.m1[threadIdx.x*times + i][threadIdx.y];
                }
                *(T2*)(ptr_output + offset) = tmp_storage.m2;
            }
        }
    }
}

__global__ void cuda_kernel_cvtformat_opt_nhwc_ndarray_spec(
    int8_t* input,
    int8_t* output,
    int64_t block_num,
    ReFormatParam param)
{
    __shared__ union {
        int8_t m1[128][128];
        int32_t m2[128][32];
    }sm_buffer;

    int64_t num = blockIdx.z;
    for(int n = num; n < param.n_outer; n += gridDim.z){
        for(int t = blockIdx.y; t < block_num; t += gridDim.y){
            int8_t *ptr_input = input + n * param.n_inner * param.src_pad;
            int8_t *ptr_output = output + n * param.dst_pad * param.n_inner;
            int tx = threadIdx.x << 2;
            int ty = threadIdx.y;
            int64_t idx_w = (blockIdx.x * blockDim.x + threadIdx.x) << 4;
            int64_t idx_h = (t * blockDim.y) + ty;
            if(idx_w < param.src_pad && idx_h < param.n_inner) {
                int offset = idx_h * param.src_pad + idx_w;
                float4 reg = *(float4*)(ptr_input + offset);
                int32_t *ptr_reg = (int32_t*)&reg;
                sm_buffer.m2[ty][tx] = ptr_reg[0];
                sm_buffer.m2[ty][tx + 1] = ptr_reg[1];
                sm_buffer.m2[ty][tx + 2] = ptr_reg[2];
                sm_buffer.m2[ty][tx + 3] = ptr_reg[3];
            } else{
                sm_buffer.m2[ty][tx] = 0;
                sm_buffer.m2[ty][tx + 1] = 0;
                sm_buffer.m2[ty][tx + 2] = 0;
                sm_buffer.m2[ty][tx + 3] = 0;
            }
            __syncthreads();
            idx_w = (t*blockDim.y);
            idx_h = blockIdx.x * (blockDim.x << 4) + (threadIdx.y);

            if(idx_w < param.n_inner && idx_h < param.dst_pad){
                int64_t offset = idx_h * param.n_inner;
                for(int i = threadIdx.x; i < 32; i += blockDim.x){
                        int x_offset = i << 2;
                        if(x_offset + idx_w < param.n_inner){
                            *((int32_t*)(ptr_output + offset + idx_w) + i) = (sm_buffer.m1[x_offset][threadIdx.y] & 0xFF) |
                                            ((sm_buffer.m1[x_offset + 1][threadIdx.y] & 0xFF) << 8) |
                                            ((sm_buffer.m1[x_offset + 2][threadIdx.y] & 0xFF) << 16) |
                                            ((sm_buffer.m1[x_offset + 3][threadIdx.y] & 0xFF) << 24);
                        }
                }
            }
        }
    }
}

template<typename T1, typename T2>
__global__ void cuda_kernel_cvtformat_opt_fast_nhwc_ndarray_unalign(
    T1* input,
    T1* output,
    int64_t block_num,
    int gridDim_z,
    int gridDim_y,
    int blockDim_x,
    int blockDim_y,
    ReFormatParam param)
{
    constexpr int times = sizeof(T2) / sizeof(T1);
    __shared__ union {
        T1 m1[65][64 + times];
        T2 m2[65][64 / times + 1];
    } sm_buffer;
    int num = blockIdx.z;
    for(int n = num; n < param.n_outer; n += gridDim_z) {
        for(int t = blockIdx.y; t < block_num ; t += gridDim_y) {
            int idx_w = (blockIdx.x * blockDim.x + threadIdx.x)*times;
            int idx_h = t * blockDim.y + threadIdx.y;
            T1* ptr_input = (T1 *)(input + n * param.n_inner * param.src_pad);
            T1* ptr_output = (T1 *)(output + n * param.dst_pad * param.n_inner);
            if(idx_w < param.src_pad && idx_h < param.n_inner) {
                int offset = idx_h * param.src_pad + idx_w;
                sm_buffer.m2[threadIdx.y][threadIdx.x] = *(T2*)(ptr_input + offset);
            } else {
                T2 tmp1;
                T1* ptr_tmp1 = (T1*)&tmp1;
                #pragma unroll times
                for(int i = 0; i < times; i++)
                {
                    ptr_tmp1[i] = (T1)0;
                }
                sm_buffer.m2[threadIdx.y][threadIdx.x] = tmp1;
            }
            __syncthreads();
            
            for(int i = threadIdx.x; i < blockDim_y; i += blockDim_x) {
                if(i > threadIdx.y) {
                    T1 value1 = sm_buffer.m1[threadIdx.y][i];
                    T1 value2 = sm_buffer.m1[i][threadIdx.y];
                    sm_buffer.m1[threadIdx.y][i] = value2;
                    sm_buffer.m1[i][threadIdx.y] = value1;
                }
            }
            __syncthreads();
            idx_w = t * blockDim_y;
            idx_h = blockIdx.x * blockDim_x * times + threadIdx.y;
            ptr_output += idx_h * param.n_inner + idx_w;
            int width = min(blockDim_y, param.n_inner - idx_w);
            if(idx_h < param.dst_pad) {
                for(int i = threadIdx.x; i < width; i += blockDim_x) {
                    *(ptr_output + i) = sm_buffer.m1[threadIdx.y][i];
                }
            }
        }
    }
}

template<typename T>
__global__ void cuda_kernel_cvtformat_opt_nc_nd(const T* input, T* output, ReFormatParam param, int num_elems, int N, DivModFast dst_pad_fast,int num_threads){
    int blocksize = num_threads * N;
    int index0 = blockIdx.x * blocksize;
    blocksize = min(num_elems - index0, blocksize);
    T *ptr_output = output + index0;
    
    for(int i = threadIdx.x; i < blocksize; i += num_threads) {
        int h, w;
        dst_pad_fast.divmod(index0 + i, h, w);
        int in_index = h * param.src_pad + w;
        ptr_output[i] = input[in_index];
    }
}

//src_pad <= 32 && dst_pad <= 32 blockSize(512)
__global__ void cuda_kernel_cvtformat_opt_nhwc_ndarray_unalign_sc(
    half* input,
    half* output,
    int32_t block_offset,
    DivModFast block_offset_fast,
    DivModFast remain_fast,
    ReFormatParam param)
{
    __shared__ union {
        half m1[4096];
        float4 m2[512];
    } sm_buffer;

    int64_t num = blockIdx.z;
    for(int n = num; n < param.n_outer; n += gridDim.z) {
        half* ptr_input = (half *)(input + n * param.n_inner * param.src_pad);
        half* ptr_output = (half *)(output + n * param.dst_pad * param.n_inner);
        int64_t start_h = blockIdx.x * block_offset;
        int64_t block_height = min(param.n_inner - start_h, block_offset);
        if(block_height <= 0) continue;
        ptr_input += start_h * param.src_pad;
        ptr_output += start_h;
        int64_t block_size = block_height * param.src_pad;
        int64_t block_size1 = block_size >> 3;
        int64_t dst_size = block_height * param.dst_pad;
        if(threadIdx.x < block_size1)
        {
            sm_buffer.m2[threadIdx.x] = *((float4*)ptr_input + threadIdx.x);
        }
        __syncthreads();
        
        if(blockIdx.x == gridDim.x - 1)
        {
            for(int i = threadIdx.x; i < dst_size; i += blockDim.x)
            {
                int h , w;
                remain_fast.divmod(i, h, w);
                ptr_output[h * param.n_inner + w] = sm_buffer.m1[w * param.src_pad + h];
            }
        } else {
            for(int i = threadIdx.x; i < dst_size; i += blockDim.x)
            {
                int h , w;
                block_offset_fast.divmod(i, h,w);
                ptr_output[h * param.n_inner + w] = sm_buffer.m1[w * param.src_pad + h];
            }
        }
        
    }
}

template<typename DST_T,int N, int SHIFT>
__global__ void cuda_kernel_cvtformat_type_opt_nhwc8_16_half(const half* input, half* output,DivModFast dst_pad_fast,ReFormatParam param)
{
    int64_t num = blockIdx.z;
    int iSize = param.src_pad * param.n_inner;
    int oSize = param.dst_pad * param.n_inner;
    DST_T reg_zero;
    half* ptr_reg = (half*)&reg_zero;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        ptr_reg[i] = (half)0;
    }
    for(int64_t n = num; n < param.n_outer; n += gridDim.z){
        int offset = blockIdx.x * blockDim.x << 3;
        int oblock_size = min(oSize - offset, blockDim.x << 3);
        const half* ptr_block_input = input + n*iSize;
        half* ptr_block_output = output + n*oSize + offset;
        oblock_size = oblock_size >> SHIFT;
        for(int i = threadIdx.x; i < oblock_size; i += blockDim.x) {
            int dst_w,dst_h;
            int index = i << SHIFT;
            dst_pad_fast.divmod(offset + index,dst_h,dst_w);
            DST_T reg_dst;
            reg_dst = reg_zero;
            if(dst_w < param.src_pad) {
                reg_dst = *(DST_T*)(ptr_block_input + dst_h * param.src_pad + dst_w);
            }
            *(DST_T*)(ptr_block_output + index) = reg_dst;
        }
    }
}

template<int type, CVTFormatMode mode>
void call_cvtkernel(void* input,void* output,ReFormatParam param, dim3 dimBlock, dim3 dimGrid, cudaStream_t stream)
{
    int use_opt = 0;
    do {
        if constexpr(mode == NDARRAY_NHWC){
            if constexpr(type == 0){
                if((param.n_inner % 16) == 0 && (param.dst_pad % 16) == 0){
                    use_opt = 1;
                    dimBlock.x = 4; dimBlock.y = 64;
                    dimGrid.x = DivUp(param.n_inner, 64);
                    dimGrid.y = DivUp(param.dst_pad, 64);
                    cuda_kernel_cvtformat_opt_ndarray_nhwc<int8_t, float4><<<dimGrid,dimBlock, 0, stream>>>(
                            (int8_t *)input,(int8_t *)output,param);
                    break;
                }
            }
            if((param.n_inner % 8)==0 && (param.dst_pad % 8) == 0){
                use_opt = 1;
                dimBlock.x = 8; dimBlock.y = 64;
                dimGrid.x  = DivUp(param.n_inner, 64);
                dimGrid.y  = DivUp(param.dst_pad, 64);
                if constexpr(type==1){
                    cuda_kernel_cvtformat_opt_ndarray_nhwc<half, float4><<<dimGrid,dimBlock, 0, stream>>>(
                        (half *)input,(half *)output,param);
                }else{
                    cuda_kernel_cvtformat_opt_ndarray_nhwc<int8_t, int64_t><<<dimGrid,dimBlock, 0, stream>>>(
                        (int8_t *)input,(int8_t *)output,param);
                }
                break;
            } else if((param.n_inner % 4)==0 && (param.dst_pad % 4) == 0){
                use_opt = 1;
                dimBlock.x = 8; dimBlock.y = 32;
                dimGrid.x  = DivUp(param.n_inner, 32);
                dimGrid.y  = DivUp(param.dst_pad, 32);
                if constexpr(type==1){
                    cuda_kernel_cvtformat_opt_ndarray_nhwc<half, float2><<<dimGrid,dimBlock, 0, stream>>>(
                        (half *)input,(half *)output,param);
                }else{
                    cuda_kernel_cvtformat_opt_ndarray_nhwc<int8_t, int32_t><<<dimGrid,dimBlock, 0, stream>>>(
                        (int8_t *)input,(int8_t *)output,param);
                }
                break;
            }else if((param.dst_pad % 8)==0 && param.dst_pad > 16){
                use_opt  = 1;
                dimBlock.x = 8; dimBlock.y = 64;
                dimGrid.x  = DivUp(param.n_inner, 64);
                dimGrid.y  = DivUp(param.dst_pad, 64);
                dimGrid.z = param.n_outer;
                if constexpr(type == 1){
                    cuda_kernel_cvtformat_opt_ndarray_nhwc_unalign<half, float4, 8, 3><<<dimGrid, dimBlock, 0, stream>>>(
                        input, output, param);
                }else{
                    cuda_kernel_cvtformat_opt_ndarray_nhwc_unalign<int8_t, int64_t, 8, 3><<<dimGrid, dimBlock, 0, stream>>>(
                        input, output, param);
                }
                break;
            }
        } else if constexpr(mode == NHWC_NDARRAY){
            if((param.n_inner % 4) == 0 && (param.src_pad % 4) == 0){
                use_opt = 1;
                if constexpr(type == 1){
                    if((param.n_inner % 8) == 0 && (param.src_pad % 8) == 0) {
                        int64_t block_num = DivUp(param.n_inner, 64);
                        dimBlock.x = 8; dimBlock.y = 64;
                        dimGrid.x  = DivUp(param.src_pad, 64);
                        dimGrid.y  = block_num > 65533? 65533 : block_num;
                        cuda_kernel_cvtformat_opt_nhwc_ndarray<half,float4><<<dimGrid, dimBlock, 0, stream>>>(
                            (half*)input, (half*)output, block_num, param);
                    } else {
                        int64_t block_num = DivUp(param.n_inner, 32);
                        dimBlock.x = 8; dimBlock.y = 32;
                        dimGrid.x  = DivUp(param.src_pad, 32);
                        dimGrid.y  = block_num > 65533? 65533 : block_num;
                        cuda_kernel_cvtformat_opt_nhwc_ndarray<half,float2><<<dimGrid, dimBlock, 0, stream>>>(
                            (half*)input, (half*)output, block_num, param);
                        }
                }else{
                    int64_t block_num = DivUp(param.n_inner, 128);
                    dimBlock.x = 8; dimBlock.y = 128;
                    dimGrid.x  = DivUp(param.src_pad, 128);
                    dimGrid.y  = block_num > 65533? 65533 : block_num;
                    cuda_kernel_cvtformat_opt_nhwc_ndarray_spec<<<dimGrid, dimBlock, 0, stream>>>(
                        (int8_t*)input, (int8_t*)output, block_num, param);
                }
                break;
            } else if((param.src_pad % 8) == 0){
                use_opt = 1;
                int64_t block_num = DivUp(param.n_inner, 64);
                dimBlock.x = 8; dimBlock.y = 64;
                dimGrid.x  = DivUp(param.src_pad, 64);
                dimGrid.y  = block_num > 65533? 65533 : block_num;
                if constexpr(type == 1){
                    if(param.src_pad <= 32)
                    {
                        int32_t block_offset = MIN(4096 / param.src_pad,param.n_inner);
                        dimBlock.x = 512; dimBlock.y = 1;
                        dimGrid.x = (param.n_inner + block_offset - 1) / block_offset;
                        dimGrid.y = 1;
                        int32_t remain = MIN(param.n_inner - (dimGrid.x - 1) * block_offset, block_offset);
                        cuda_kernel_cvtformat_opt_nhwc_ndarray_unalign_sc<<<dimGrid, dimBlock, 0, stream>>>(
                            (half*)input,(half*)output, block_offset, DivModFast(block_offset),DivModFast(remain), param
                        );
                    } else if(param.n_inner == 1) {
                        int out_elems = param.n_outer * param.n_inner * param.dst_pad;
                        int blocksize = 256;
                        int n = 8;
                        int gridsize = (out_elems + blocksize * n - 1) / (blocksize * n);
                        cuda_kernel_cvtformat_opt_nc_nd<half><<<gridsize, blocksize, 0, stream>>>((const half*)input, (half*)output,param,out_elems, n, DivModFast(param.dst_pad),blocksize);
                    } else {
                        cuda_kernel_cvtformat_opt_fast_nhwc_ndarray_unalign<half,float4><<<dimGrid, dimBlock, 0, stream>>>(
                            (half*)input, (half*)output, block_num, dimGrid.z, dimGrid.y, dimBlock.x, dimBlock.y, param);
                    }
                }else{
                    if(param.n_inner == 1) {
                        int out_elems = param.n_outer * param.n_inner * param.dst_pad;
                        int blocksize = 256;
                        int n = 8;
                        int gridsize = (out_elems + blocksize * n - 1) / (blocksize * n);
                        cuda_kernel_cvtformat_opt_nc_nd<int8_t><<<gridsize, blocksize, 0, stream>>>((const int8_t*)input, (int8_t*)output,param,out_elems, n, DivModFast(param.dst_pad),blocksize);
                    } else {
                        cuda_kernel_cvtformat_opt_fast_nhwc_ndarray_unalign<int8_t, int64_t><<<dimGrid, dimBlock, 0, stream>>>(
                            (int8_t*)input, (int8_t*)output, block_num, dimGrid.z, dimGrid.y, dimBlock.x, dimBlock.y, param);
                    }
                }
                break;
            }
        } else if constexpr(mode == NHWC8_NHWC16) {    
            if(param.src_pad == param.dst_pad){
                cudaMemcpyAsync(output, input, param.n_inner*param.src_pad*param.n_outer*GetSizeOfDataType(param.out_type), cudaMemcpyDeviceToDevice, stream);
                return;
            }

            if constexpr(type==1) {
                if((param.src_pad&7)==0) {
                    use_opt = 1;
                    dimBlock.x = 512; dimBlock.y = 1;
                    dimGrid.x = (param.dst_pad * param.n_inner + 4095) >> 12; dimGrid.y = 1;
                    if((param.dst_pad&7) == 0) {
                        cuda_kernel_cvtformat_type_opt_nhwc8_16_half<float4, 8 , 3><<<dimGrid, dimBlock, 0, stream>>>((const half*)input,(half*)output,DivModFast(param.dst_pad),param);
                    } else if((param.dst_pad&3) == 0) {
                        cuda_kernel_cvtformat_type_opt_nhwc8_16_half<float2, 4 , 2><<<dimGrid, dimBlock, 0, stream>>>((const half*)input,(half*)output,DivModFast(param.dst_pad),param);
                    } else if((param.dst_pad & 1) == 0) {
                        cuda_kernel_cvtformat_type_opt_nhwc8_16_half<float, 2 , 1><<<dimGrid, dimBlock, 0, stream>>>((const half*)input,(half*)output,DivModFast(param.dst_pad),param);
                    } else {
                        cuda_kernel_cvtformat_type_opt_nhwc8_16_half<half, 1 , 0><<<dimGrid, dimBlock, 0, stream>>>((const half*)input,(half*)output,DivModFast(param.dst_pad),param);
                    }
                }
            }
            break;
        }
    } while(0);
    
    if(!use_opt){
        if constexpr(type == 1){
        cuda_kernel_cvtformat<half, mode><<<dimGrid, dimBlock, 0, stream>>>(
                (half *)input, (half *)output, param);
        }else{
            cuda_kernel_cvtformat<int8_t, mode><<<dimGrid, dimBlock, 0, stream>>>(
                    (int8_t *)input, (int8_t *)output, param);
        }
    }
    return;
}

__global__ void cuda_kernel_packed_cvtformat_opt(
    int8_t *input,
    int8_t *output,
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
    for(int i = 0; i < LEASTCHANNEL; i++)
    {
        if(i < param.src_pad)
        {
            val[i] = input[offset];
            offset += param.n_inner;
        }
    }    
    float4* dst = (float4*)val;
    float4* dst_out = (float4*)output;
    dst_out[tid] = dst[0];
}

template<typename src_T, typename dst_T,int shift, int N>
__global__ void cuda_kernel_small_channel_cvtformat_nd_nhwc_opt(
    const void *input,
    void* output,
    float divPadChannel,
    ReFormatParam param
)
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
    half *ptr_input = (half*)input + src_offset + tidy * param.n_inner;
    half* ptr_block_output = (half*)output + dst_offset;
    int height = min(512,param.n_inner - iBlockOffset);
    int blockLength = height>>shift;
    if(tidy < param.src_pad) {
        for(int i = threadIdx.x; i < blockLength; i += blockDim.x)  
        {
            src_T vInput = *((src_T*)ptr_input + i);
            dst_T vOutput;
            half * ptr_reg_src = (half *)&vInput;
            half * ptr_reg_dst = (half*)&vOutput;
            #pragma unroll N
            for(int j = 0; j < N; j++)
            {
                ptr_reg_dst[j] = ptr_reg_src[j];
            }
            sm_buffer.m2[tidy][i] = vOutput;
        }
    } else {
        float4 reg_zero = make_float4(0,0,0,0);
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

template <typename type, typename T1,int N, int SHIFT,int SM_SIZE>
__global__ void cuda_kernel_cvtformat_nc_opt(
    int height,
    const type* input,
    type* output,
    DivModFast dst_pad_fast,
    ReFormatParam param)
{
    __shared__ union {
        type m1[SM_SIZE];
        T1 m2[SM_SIZE>>SHIFT];
    } buffer;

    if(blockIdx.x == gridDim.x - 1) 
    {
        int heightId = blockIdx.x * height;
        int real_height = min(param.n_outer - heightId, height);
        if(real_height <= 0) return;
        int iblockSize = real_height * param.src_pad;
        const type * ptr_input = input + heightId * param.src_pad;

        for(int i = threadIdx.x; i < iblockSize; i += blockDim.x) {
            buffer.m1[i] = *(ptr_input + i);
        }
        int oblockSize = real_height * param.dst_pad;
        int offset = heightId * param.dst_pad;
        oblockSize = oblockSize >> SHIFT;
        type * ptr_output = output + offset;
        __syncthreads();
        for(int i = threadIdx.x; i < oblockSize; i += blockDim.x) {
            T1 reg_dst; int outer, chl;
            type* ptr_reg_dst = (type*)&reg_dst;
            dst_pad_fast.divmod(offset + (i << SHIFT), outer, chl);
            type *ptr_sm_input = buffer.m1 + (outer - heightId) * param.src_pad + chl;
            #pragma unroll N
            for(int j = 0; j < N; j++) {
                if(chl + j < param.src_pad) {
                    ptr_reg_dst[j] = ptr_sm_input[j];
                }else {
                    ptr_reg_dst[j] = (type)0;
                }
            }
            *(T1*)(ptr_output + (i << SHIFT)) = reg_dst;
        }
    } 
    else {
        int heightId = blockIdx.x * height;
        int iblockSize = height * param.src_pad;
        iblockSize = iblockSize >> SHIFT;
        const type * ptr_input = input + heightId * param.src_pad;

        for(int i = threadIdx.x; i < iblockSize; i += blockDim.x) {
            buffer.m2[i] = *(T1 *)(ptr_input + (i<<SHIFT));
        }
        int oblockSize = height * param.dst_pad;
        int offset = blockIdx.x * oblockSize;
        oblockSize = oblockSize >> SHIFT;
        type * ptr_output = output + offset;
        __syncthreads();
        for(int i = threadIdx.x; i < oblockSize; i += blockDim.x) {
            T1 reg_dst; int outer, chl;
            type* ptr_reg_dst = (type*)&reg_dst;
            dst_pad_fast.divmod(offset + (i << SHIFT), outer, chl);
            type *ptr_sm_input = buffer.m1 + (outer - heightId) * param.src_pad + chl;
            #pragma unroll N
            for(int j = 0; j < N; j++) {
                if(chl + j < param.src_pad) {
                    ptr_reg_dst[j] = ptr_sm_input[j];
                }else {
                    ptr_reg_dst[j] = (type)0;
                }
            }
            *(T1*)(ptr_output + (i << SHIFT)) = reg_dst;
        }
    }
}

#endif//OPT_CVT

#ifndef PPLNN_USE_MACA
#define cvtNCTONHWC(type)                                                                               \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat<type, NDARRAY_NHWC>(                                              \
    type* input,                                                                                        \
    type* output,                                                                                       \
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
            share_val[threadIdx.y][threadIdx.x] = input[offset];                                        \
        } else {                                                                                        \
            share_val[threadIdx.y][threadIdx.x] = (type)0;                                              \
        }                                                                                               \
        __syncthreads();                                                                                \
                                                                                                        \
        idx_w = blockIdx.y * blockDim.y + threadIdx.x;                                                  \
        idx_h = blockIdx.x * blockDim.x + threadIdx.y;                                                  \
                                                                                                        \
        if (idx_w < param.dst_pad && idx_h < param.n_inner) {                                           \
            int64_t offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;         \
            output[offset] = share_val[threadIdx.x][threadIdx.y];                                       \
        }                                                                                               \
    }                                                                                                   \
}
#else
#define cvtNCTONHWC(type)                                                                               \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat<type, NDARRAY_NHWC>(                                              \
    type* input,                                                                                        \
    type* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    __shared__ type share_val[DIM][DIM + 1];                                                            \
                                                                                                        \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n+= gridDim.z) {                                              \
        int64_t idx_w = blockIdx.y * blockDim.y + threadIdx.x;                                          \
        int64_t idx_h = blockIdx.x * blockDim.x + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.n_inner && idx_h < param.src_pad) {                                           \
            int64_t offset = n * param.src_pad * param.n_inner + idx_h * param.n_inner + idx_w;         \
            share_val[threadIdx.x][threadIdx.y] = input[offset];                                        \
        } else {                                                                                        \
            share_val[threadIdx.x][threadIdx.y] = (type)0;                                              \
        }                                                                                               \
        __syncthreads();                                                                                \
                                                                                                        \
        idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                                  \
        idx_h = blockIdx.y * blockDim.y + threadIdx.y;                                                  \
                                                                                                        \
        if (idx_w < param.dst_pad && idx_h < param.n_inner) {                                           \
            int64_t offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;         \
            output[offset] = share_val[threadIdx.y][threadIdx.x];                                       \
        }                                                                                               \
    }                                                                                                   \
}
#endif
#if __CUDACC_VER_MAJOR__ >= 9
    cvtNCTONHWC(half)
#endif
    cvtNCTONHWC(float)
    cvtNCTONHWC(char)
    cvtNCTONHWC(double)
    cvtNCTONHWC(int8_t)

#define cvtNHWC8TONC(type)                                                                               \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat<type, NHWC_NDARRAY>(                                              \
    type* input,                                                                                        \
    type* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    __shared__ type share_val[DIM][DIM + 1];                                                            \
                                                                                                        \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n += gridDim.z) {                                              \
        for (int t = blockIdx.y; t < DivUp(param.n_inner, DIM) ; t+= gridDim.y) { \
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        int64_t idx_h = t * blockDim.y + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.src_pad && idx_h < param.n_inner) {                                           \
            int64_t offset = n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;         \
            share_val[threadIdx.y][threadIdx.x] = input[offset];                                        \
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
            output[offset] = share_val[threadIdx.x][threadIdx.y];                                       \
        }                                                                                               \
        }\
    }                                                                                                   \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtNHWC8TONC(half)
#endif
    cvtNHWC8TONC(float)
    cvtNHWC8TONC(char)
    cvtNHWC8TONC(double)
    cvtNHWC8TONC(int8_t)

#define cvtN4CXTONC(type)                                                                                              \
template <>                                                                                                            \
__global__ void cuda_kernel_cvtformat<type, N4CX_NDARRAY>(                                                             \
    type * input,                                                                                                      \
    type * output,                                                                                                     \
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
            output[outOffset]        = input[offset];                                                                  \
        }                                                                                                              \
    }                                                                                                                  \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtN4CXTONC(half)
#endif
    cvtN4CXTONC(float)
    cvtN4CXTONC(char)
    cvtN4CXTONC(double)
    cvtN4CXTONC(int8_t)

#define cvtNCTON4CX(type)                                                                                             \
template <>                                                                                                           \
__global__ void cuda_kernel_cvtformat<type, NDARRAY_N4CX>(                                                            \
    type * input,                                                                                                     \
    type * output,                                                                                                    \
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
            output[offset]          = input[inOffset];                                                                \
        }                                                                                                             \
    }                                                                                                                 \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtNCTON4CX(half)
#endif
    cvtNCTON4CX(float)
    cvtNCTON4CX(char)
    cvtNCTON4CX(double)
    cvtNCTON4CX(int8_t)

#define cvtNC1HWC0TONC(type)                                                                                           \
template <>                                                                                                            \
__global__ void cuda_kernel_cvtformat<type, NC1HWC0_NDARRAY>(                                                          \
    type * input,                                                                                                      \
    type * output,                                                                                                     \
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
            output[outOffset]        = input[offset];                                                                  \
        }                                                                                                              \
    }                                                                                                                  \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtNC1HWC0TONC(half)
#endif
    cvtNC1HWC0TONC(float)
    cvtNC1HWC0TONC(char)
    cvtNC1HWC0TONC(double)
    cvtNC1HWC0TONC(int8_t)

#define cvtNCTONC1HWC0(type)                                                                                          \
template <>                                                                                                           \
__global__ void cuda_kernel_cvtformat<type, NDARRAY_NC1HWC0>(                                                         \
    type * input,                                                                                                     \
    type * output,                                                                                                    \
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
            output[offset]          = input[inOffset];                                                                \
        }                                                                                                             \
    }                                                                                                                 \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtNCTONC1HWC0(half)
#endif
    cvtNCTONC1HWC0(float)
    cvtNCTONC1HWC0(char)
    cvtNCTONC1HWC0(double)
    cvtNCTONC1HWC0(int8_t)

template <typename T, CVTFormatMode mode>
__global__ void cuda_kernel_small_channel_cvtformat(
    T* input,
    int num_elems,
    DivModFast inner_fast,
    DivModFast src_pad_fast,
    DivModFast dst_pad_fast,
    T* output,
    ReFormatParam param)
{
}

#define cvtSMCHANNELNCTONHWC8(type)                                                                      \
template<>                                                                                              \
__global__ void cuda_kernel_small_channel_cvtformat<type, NDARRAY_NHWC>(                                \
    type* input,                                                                                        \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    type* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    dst_pad_fast.divmod(tid, num_inner, c_idx);                                                         \
    inner_idx = inner_fast.mod(num_inner);                                                              \
    int outer_idx = inner_fast.div(num_inner);                                                              \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx * param.n_inner + inner_idx;         \
    output[tid] =  c_idx < param.src_pad ? input[offset] : (type)0;                                     \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtSMCHANNELNCTONHWC8(half)
#endif
    cvtSMCHANNELNCTONHWC8(float)
    cvtSMCHANNELNCTONHWC8(char)
    cvtSMCHANNELNCTONHWC8(double)
    cvtSMCHANNELNCTONHWC8(int8_t)

#define cvtSMCHANNELNHWC8TONC(type)                                                                      \
template<>                                                                                              \
__global__ void cuda_kernel_small_channel_cvtformat<type, NHWC_NDARRAY>(                                \
    type* input,                                                                                        \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    type* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    inner_fast.divmod(tid, num_inner, inner_idx);                                                       \
    c_idx = dst_pad_fast.mod(num_inner);                                                                \
    int outer_idx = tid / (param.dst_pad * param.n_inner);                                              \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx + inner_idx * param.src_pad;         \
    output[tid] = input[offset];                                                                        \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtSMCHANNELNHWC8TONC(half)
#endif
    cvtSMCHANNELNHWC8TONC(float)
    cvtSMCHANNELNHWC8TONC(char)
    cvtSMCHANNELNHWC8TONC(double)
    cvtSMCHANNELNHWC8TONC(int8_t)

#define cvtSMCHANNELN4CXTONC(type)                                                                               \
template <>                                                                                                      \
__global__ void cuda_kernel_small_channel_cvtformat<type, N4CX_NDARRAY>(                                         \
    type * input,                                                                                                \
    int num_elems,                                                                                               \
    DivModFast inner_fast,                                                                                       \
    DivModFast src_pad_fast,                                                                                     \
    DivModFast dst_pad_fast,                                                                                     \
    type* output,                                                                                                \
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
    output[outOffset]          = input[offset];                                                                  \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtSMCHANNELN4CXTONC(half)
#endif
    cvtSMCHANNELN4CXTONC(float)
    cvtSMCHANNELN4CXTONC(char)
    cvtSMCHANNELN4CXTONC(double)
    cvtSMCHANNELN4CXTONC(int8_t)

#define cvtSMCHANNELNCTON4CX(type)                                                                               \
template <>                                                                                                      \
__global__ void cuda_kernel_small_channel_cvtformat<type, NDARRAY_N4CX>(                                         \
    type * input,                                                                                                \
    int num_elems,                                                                                               \
    DivModFast inner_fast,                                                                                       \
    DivModFast src_pad_fast,                                                                                     \
    DivModFast dst_pad_fast,                                                                                     \
    type* output,                                                                                                \
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
    output[offset]             = input[inOffset];                                                                \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtSMCHANNELNCTON4CX(half)
#endif
    cvtSMCHANNELNCTON4CX(float)
    cvtSMCHANNELNCTON4CX(char)
    cvtSMCHANNELNCTON4CX(double)
    cvtSMCHANNELNCTON4CX(int8_t)

#define cvtSMCHANNELNC1HWC0TONC(type)                                                                   \
template <>                                                                                             \
__global__ void cuda_kernel_small_channel_cvtformat<type, NC1HWC0_NDARRAY>(                             \
    type* input,                                                                                        \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    type* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    inner_fast.divmod(tid, num_inner, inner_idx);                                                       \
    c_idx = dst_pad_fast.mod(num_inner);                                                                \
    int outer_idx = tid / (param.dst_pad * param.n_inner);                                              \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx + inner_idx * param.src_pad;         \
    output[tid] = input[offset];                                                                        \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtSMCHANNELNC1HWC0TONC(half)
#endif
    cvtSMCHANNELNC1HWC0TONC(float)
    cvtSMCHANNELNC1HWC0TONC(char)
    cvtSMCHANNELNC1HWC0TONC(double)
    cvtSMCHANNELNC1HWC0TONC(int8_t)

#define cvtSMCHANNELNCTONC1HWC0(type)                                                                   \
template <>                                                                                             \
__global__ void cuda_kernel_small_channel_cvtformat<type, NDARRAY_NC1HWC0>(                             \
    type* input,                                                                                        \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    type* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    dst_pad_fast.divmod(tid, num_inner, c_idx);                                                         \
    inner_idx = inner_fast.mod(num_inner);                                                              \
    int outer_idx = inner_fast.div(num_inner);                                                          \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx * param.n_inner + inner_idx;         \
    output[tid] =  c_idx < param.src_pad ? input[offset] : (type)0;                                     \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtSMCHANNELNCTONC1HWC0(half)
#endif
    cvtSMCHANNELNCTONC1HWC0(float)
    cvtSMCHANNELNCTONC1HWC0(char)
    cvtSMCHANNELNCTONC1HWC0(double)
    cvtSMCHANNELNCTONC1HWC0(int8_t)

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
   #ifndef PPLNN_USE_MACA
        dimGrid.x  = DivUp(param.n_inner, DIM);
        dimGrid.y  = DivUp(param.dst_pad, DIM);
    #else
        dimGrid.y  = DivUp(param.n_inner, DIM);
        dimGrid.x  = DivUp(param.dst_pad, DIM);
    #endif
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

void PPLCUDANormalCVTFormat(cudaStream_t stream, const void *input, void *output, ReFormatParam param)
{
#ifdef OPT_CVT
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 dimBlock(DIM, 1, 1);                                                      \
        dim3 dimGrid(DIM, 1, 1);                                                       \
        GenDimParam<mode>(param, dimBlock, dimGrid);                                  \
        switch (GetSizeOfDataType(param.out_type)) {                                    \
            case 1:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormat for data size 1" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormat for data size 1" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormat for data size 1" ;           \
                        return;\
                    default:\
                        break;\
                }\
                call_cvtkernel<0,mode>((void*)input,(void*)output,param,dimBlock,dimGrid,stream);\
                break;                                                                \
            case 2:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormat for data size 2" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormat for data size 2" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormat for data size 2" ;           \
                        return;\
                    default:\
                        break;\
                }\
                call_cvtkernel<1,mode>((void*)input,(void*)output,param,dimBlock,dimGrid,stream);\
                break;                                                                \
            case 4:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormat for data size 4" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func OPT_CVT PPLCUDANormalCVTFormat for data size 4" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func OPT_CVT PPLCUDANormalCVTFormat for data size 4" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat<float, mode><<<dimGrid, dimBlock, 0, stream>>>( \
                    (float*)input, (float *)output, param);                          \
                break;                                                               \
            case 8:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func PPLCUDANormalCVTFormat for data size 8" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func PPLCUDANormalCVTFormat for data size 8" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func PPLCUDANormalCVTFormat for data size 8" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat<double, mode><<<dimGrid, dimBlock, 0, stream>>>(\
                    (double *)input, (double *)output, param);                        \
                break;                                                                \
            default:                                                                  \
                LOG(ERROR) << "Unsupport  for OPT_CVT PPLCUDANormalCVTFormat for data size" << (int)GetSizeOfDataType(param.out_type);           \
                break;                                                                \
        }                                                                             \
        return;                                                                       \
    } while (0)
#else //!OPT_CVT
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 dimBlock(DIM, 1, 1);                                                      \
        dim3 dimGrid(DIM, 1, 1);                                                       \
        GenDimParam<mode>(param, dimBlock, dimGrid);                                  \
        switch (GetSizeOfDataType(param.out_type)) {                                    \
            case 1:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormat for data size 1" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormat for data size 1" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormat for data size 1" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat<int8_t, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (int8_t *)input, (int8_t *)output, param);                            \
                break;                                                                \
            case 2:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormat for data size 2" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormat for data size 2" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormat for data size 2" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat<half, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (half *)input, (half *)output, param);                            \
                break;                                                                \
            case 4:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormat for data size 4" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormat for data size 4" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormat for data size 4" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat<float, mode><<<dimGrid, dimBlock, 0, stream>>>( \
                    (float *)input, (float *)output, param);                          \
                break;                                                                \
            case 8:                                                                   \
                switch(mode){                                                         \
                    case NHWC_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NHWC_NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormat for data size 8" ;           \
                        return;\
                    case NC1HWC0_NHWC:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0_NHWC in func !OPT_CVT PPLCUDANormalCVTFormat for data size 8" ;           \
                        return;\
                    case NC1HWC0_NC1HWC0:                                                \
                        LOG(ERROR) << "Unsupport NC1HWC0 in func !OPT_CVT PPLCUDANormalCVTFormat for data size 8" ;           \
                        return;\
                    default:\
                        break;\
                }\
                cuda_kernel_cvtformat<double, mode><<<dimGrid, dimBlock, 0, stream>>>(\
                    (double *)input, (double *)output, param);                        \
                break;                                                                \
            default:                                                                  \
                LOG(ERROR) << "Unsupport  for !OPT_CVT PPLCUDANormalCVTFormat for data size" << (int)GetSizeOfDataType(param.out_type);           \
                break;                                                                \
        }                                                                             \
        LOG(ERROR) << "Unsupport  for PPLCUDANormalCVTFormat for format converter:" << (int)mode;           \
        return;                                                                       \
    } while (0)
#endif//OPT_CVT

    switch (GetCVTFormatMode(param)) {
        RFC8C16
        RFNHWC
        RFN4CX
        RFNC1HWC0
        default:
            LOG(ERROR) << "Unsupport  for PPLCUDANormalCVTFormat data format converter: " << (int)GetCVTFormatMode(param);           \
            return;
    }
#undef RUN
}

__global__ void cuda_kernel_packed_cvtformat(
    int8_t *input,
    int8_t *output,
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
        val[i] = input[offset];
        offset += param.n_inner;
    }
    float4* dst = (float4*)val;
    float4* dst_out = (float4*)output;
    dst_out[tid] = dst[0];
}
void PPLCUDASmallChannelCVTPackedFormat(cudaStream_t stream, const void *input, void *output, ReFormatParam param)
{
    dim3 dimBlock(256, 1, 1);
    int num_elems = param.out_elems / param.dst_pad;
    dim3 dimGrid(DivUp(num_elems, 256), 1, 1);
    DivModFast inner_fast(param.n_inner);
#ifdef OPT_CVT
    cuda_kernel_packed_cvtformat_opt<<<dimGrid, dimBlock, 0, stream>>>((int8_t*)input, (int8_t*)output, inner_fast, num_elems, param);
#else//!OPT_CVT
    cuda_kernel_packed_cvtformat<<<dimGrid, dimBlock, 0, stream>>>((int8_t*)input, (int8_t*)output, inner_fast, num_elems, param);
#endif//OPT_CVT
}
void PPLCUDASmallChannelCVTFormat(cudaStream_t stream, const void *input, void *output, ReFormatParam param)
{
    if (param.out_type == ppl::common::DATATYPE_INT8 && (param.out_format == ppl::common::DATAFORMAT_NHWC16 || param.out_format == ppl::common::DATAFORMAT_NCHW16)
        && param.in_format == ppl::common::DATAFORMAT_NDARRAY) {
        #if 0
            if(param.out_format == ppl::common::DATAFORMAT_NCHW16){
                LOG(ERROR) << "Need support func like PPLCUDASmallChannelCVTPackedFormat for NCHW16";
                return;
            }
            PPLCUDASmallChannelCVTPackedFormat(stream, input, output, param);
            return;
        #endif
        }
#ifdef OPT_CVT
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 dimBlock(256, 1, 1);                                                     \
        int num_elems = param.out_elems;                                              \
        dim3 dimGrid(DivUp(num_elems, 256), 1, 1);                                    \
        DivModFast inner_fast(param.n_inner);                                         \
        DivModFast src_pad_fast(param.src_pad);                                       \
        DivModFast dst_pad_fast(param.dst_pad);                                       \
        switch (GetSizeOfDataType(param.out_type)) {                                    \
            case 1:                                                                   \
                cuda_kernel_small_channel_cvtformat<char, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (char *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast, \
                                    (char *)output, param);                           \
                break;                                                                \
            case 2:                                                                   \
                switch(mode){                                                         \
                    case NDARRAY_NHWC:                                                \
                        if((param.dst_pad & 7)==0)                                    \
                        {                                                             \
                            float divPadChannel = 1.0 / param.dst_pad;                \
                            if((param.n_inner & 7) == 0)                              \
                            {                                                         \
                                cuda_kernel_small_channel_cvtformat_nd_nhwc_opt<float4,float4,3,8>                        \
                                <<<dim3((param.n_inner + 511)>>9, param.n_outer,1), dim3(64,param.dst_pad,1),0,stream>>>  \
                                (input, output, divPadChannel,param);                                                     \
                            } else if((param.n_inner & 3) == 0)                                                           \
                            {                                                                                             \
                                cuda_kernel_small_channel_cvtformat_nd_nhwc_opt<float2,float2,2,4>                        \
                                <<<dim3((param.n_inner + 511)>>9, param.n_outer,1), dim3(64,param.dst_pad,1),0,stream>>>  \
                                (input, output, divPadChannel,param);                                                     \
                            } else {                                                                                      \
                                cuda_kernel_small_channel_cvtformat_nd_nhwc_opt<half,half,0,1>                            \
                                <<<dim3((param.n_inner + 511)>>9, param.n_outer,1), dim3(64,param.dst_pad,1),0,stream>>>  \
                                (input, output, divPadChannel,param);                                                     \
                            }                                                                                             \
                            return;                                                                                       \
                        }                                                                                   \
                    default:                                                                                \
                        cuda_kernel_small_channel_cvtformat<half, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                            (half *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast,               \
                            (half *)output, param);                                   \
                        break;                                                        \
                }                                                                     \
                break;                                                                \
            case 4:                                                                   \
                cuda_kernel_small_channel_cvtformat<float, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (float *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast, \
                                (float *)output, param);                               \
                break;                                                                \
            case 8:                                                                   \
                cuda_kernel_small_channel_cvtformat<double, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (double *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast, \
                                (double *)output, param);                               \
                break;                                                                \
            default:                                                                  \
                LOG(ERROR) << "Unsupport  for data size: " << GetSizeOfDataType(param.out_type);           \
                break;                                                                \
        }                                                                             \
        return;                                                                       \
    } while (0)
#else//!OPT_CVT
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 dimBlock(256, 1, 1);                                                     \
        int num_elems = param.out_elems;                                              \
        dim3 dimGrid(DivUp(num_elems, 256), 1, 1);                                    \
        DivModFast inner_fast(param.n_inner);                                         \
        DivModFast src_pad_fast(param.src_pad);                                       \
        DivModFast dst_pad_fast(param.dst_pad);                                       \
        switch (GetSizeOfDataType(param.out_type)) {                                    \
            case 1:                                                                   \
                cuda_kernel_small_channel_cvtformat<char, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (char *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast, \
                                    (char *)output, param);                           \
                break;                                                                \
            case 2:                                                                   \
                cuda_kernel_small_channel_cvtformat<half, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (half *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast, \
                                (half *)output, param);                               \
                break;                                                                \
            case 4:                                                                   \
                cuda_kernel_small_channel_cvtformat<float, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (float *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast, \
                                (float *)output, param);                               \
                break;                                                                \
            case 8:                                                                   \
                cuda_kernel_small_channel_cvtformat<double, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (double *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast, \
                                (double *)output, param);                               \
                break;                                                                \
            default:                                                                  \
                LOG(ERROR) << "Unsupport  for data size:" << GetSizeOfDataType(param.out_type);  \
                break;                                                                \
        }                                                                             \
        return;                                                                       \
    } while (0)
#endif//OPT_CVT

    switch (GetCVTFormatMode(param)) {
        RFNHWC
        RFN4CX
        RFNC1HWC0
        default:
            LOG(ERROR) << "PPLCUDASmallChannelCVTFormat Unsupport  for GetCVTFormatMode:" << (int)GetCVTFormatMode(param);  \
            return;
    }
#undef RUN
}

void PPLCUDACVTFormat(
    cudaStream_t stream,
    const void* input,
    void* output,
    ReFormatParam param)
{
    if (param.channel <= LEASTCHANNEL && !(GetCVTFormatMode(param) == NHWC8_NHWC16 || GetCVTFormatMode(param) == NHWC16_NHWC8
                                          || GetCVTFormatMode(param) == NC1HWC0_NHWC || GetCVTFormatMode(param) == NHWC_NC1HWC0
                                          || GetCVTFormatMode(param) == NC1HWC0_NC1HWC0 )) {
        if (param.out_type == DATATYPE_INT8) {
            PPLCUDASmallChannelCVTFormat(stream, input, output, param);
        } else if (param.out_type == DATATYPE_FLOAT32) {
            PPLCUDASmallChannelCVTFormat(stream, input, output, param);
        } else {
            PPLCUDASmallChannelCVTFormat(stream, input, output, param);
        }
    } else
    {
        PPLCUDANormalCVTFormat(stream, input, output, param);
    }
}

template <typename type>
__global__ void cuda_kernel_cvtformat_nc(
    type* input,
    type* output,
    ReFormatParam param,
    bool ndarray_nhwc)
{
    int64_t idx_chl = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t idx_outer = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_chl >= param.dst_pad || idx_outer >= param.n_outer) return;
    int64_t out_offset = idx_outer * param.dst_pad + idx_chl;
    if (!ndarray_nhwc || idx_chl < param.src_pad) {
        int64_t in_offset = idx_outer * param.src_pad + idx_chl;
        output[out_offset] = input[in_offset];
    } else {
        output[out_offset] = (type)0;
    }
}

void PPLCUDACVTFormatNC(
    cudaStream_t stream,
    const void* input,
    void* output,
    ReFormatParam param)
{
    // only for ndarray_nhwc when param.inner == 1, which means just padded
    dim3 dimBlock, dimGrid;
    dimBlock.x = DIM;
    dimBlock.y = DIM;
    dimGrid.x  = DivUp(param.dst_pad, DIM);
    dimGrid.y  = DivUp(param.n_outer, DIM);
    bool ndarray_nhwc = (GetCVTFormatMode(param) == NDARRAY_NHWC);
#ifdef OPT_CVT
    if(param.src_pad == param.dst_pad){
        cudaMemcpyAsync(output, input, param.n_inner*param.src_pad*param.n_outer*GetSizeOfDataType(param.out_type), cudaMemcpyDeviceToDevice, stream);
        return;
    }
#endif//OPT_CVT 
    if(!ndarray_nhwc){ //ndarray_nc1hwc0
        switch (GetSizeOfDataType(param.out_type)) {
        case 1:
            LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func PPLCUDACVTFormatNC for data size 1" ;
            break;
        case 2:
            LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func PPLCUDACVTFormatNC for data size 2" ;
            break;
        case 4:
            LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func PPLCUDACVTFormatNC for data size 4" ;
            break;
        case 8:
            LOG(ERROR) << "Unsupport  ndarray_nc1hwc0 in func PPLCUDACVTFormatNC for data size 8" ;
            break;
        default:
            break;
        }
        return;
    }
#ifdef OPT_CVT
     int use_opt = 0;
     int outSizeType = GetSizeOfDataType(param.out_type);
    if(outSizeType == 1) {
        if((param.dst_pad & 7) == 0) {
            if((int(8192 / param.dst_pad) >> 3) != 0) {
                use_opt = 1;
                int height = (8192 / param.dst_pad) >> 3 << 3;
                int blockSize = 512;
                int gridSize = (param.n_outer + height - 1) / height;
                cuda_kernel_cvtformat_nc_opt<int8_t, float2, 8, 3, 8192><<<gridSize, blockSize,0, stream>>>(height,(const int8_t*)input,(int8_t*)output,DivModFast(param.dst_pad),param);
            } else if((int(8192 / param.dst_pad) >> 2)!=0) {
                use_opt = 1;
                int height = (8192 / param.dst_pad) >> 2 << 2;
                int blockSize = 512;
                int gridSize = (param.n_outer + height - 1) / height;
                cuda_kernel_cvtformat_nc_opt<int8_t, float, 4, 2, 8192><<<gridSize, blockSize,0, stream>>>(height,(const int8_t*)input,(int8_t*)output,DivModFast(param.dst_pad),param);
            } else if((int(8192 / param.dst_pad))!=0) {
                use_opt = 1;
                int height = (8192 / param.dst_pad);
                int blockSize = 512;
                int gridSize = (param.n_outer + height - 1) / height;
                cuda_kernel_cvtformat_nc_opt<int8_t, int8_t, 1, 0, 8192><<<gridSize, blockSize,0, stream>>>(height,(const int8_t*)input,(int8_t*)output,DivModFast(param.dst_pad),param);
            }
        }
    } else if(outSizeType == 2) {
        if((param.dst_pad & 7) == 0) {
            if((int(8192 / param.dst_pad) >> 3) != 0) {
                use_opt = 1;
                int height = (8192 / param.dst_pad) >> 3 << 3;
                int blockSize = 512;
                int gridSize = (param.n_outer + height - 1) / height;
                cuda_kernel_cvtformat_nc_opt<half, float4, 8, 3, 8192><<<gridSize, blockSize,0, stream>>>(height,(const half*)input,(half*)output,DivModFast(param.dst_pad),param);
            } else if((int(8192 / param.dst_pad) >> 2)!=0) {
                use_opt = 1;
                int height = (8192 / param.dst_pad) >> 2 << 2;
                int blockSize = 512;
                int gridSize = (param.n_outer + height - 1) / height;
                cuda_kernel_cvtformat_nc_opt<half, float2, 4, 2, 8192><<<gridSize, blockSize,0, stream>>>(height,(const half*)input,(half*)output,DivModFast(param.dst_pad),param);
            } else if((int(8192 / param.dst_pad))!=0) {
                use_opt = 1;
                int height = (8192 / param.dst_pad);
                int blockSize = 512;
                int gridSize = (param.n_outer + height - 1) / height;
                cuda_kernel_cvtformat_nc_opt<half, half, 1, 0, 8192><<<gridSize, blockSize,0, stream>>>(height,(const half*)input,(half*)output,DivModFast(param.dst_pad),param);
            }
        }
    } else if(outSizeType == 4) {
        if(int(4096 / param.dst_pad) != 0) {
            use_opt = 1;
            int height = (4096 / param.dst_pad);
            int blockSize = 512;
            int gridSize = (param.n_outer + height - 1) / height;
            cuda_kernel_cvtformat_nc_opt<float, float, 1, 0, 4096><<<gridSize, blockSize,0, stream>>>(height,(const float*)input,(float*)output,DivModFast(param.dst_pad),param);
        }
    } else if(outSizeType == 8) {
        if(int(2048 / param.dst_pad) != 0) {
            use_opt = 1;
            int height = (2048 / param.dst_pad);
            int blockSize = 512;
            int gridSize = (param.n_outer + height - 1) / height;
            cuda_kernel_cvtformat_nc_opt<double, double, 1, 0, 2048><<<gridSize, blockSize,0, stream>>>(height,(const double*)input,(double*)output,DivModFast(param.dst_pad),param);
        }
    }
    if(use_opt) return;
#endif//OPT_CVT
    switch (GetSizeOfDataType(param.out_type)) {
        case 1:
            cuda_kernel_cvtformat_nc<int8_t><<<dimGrid, dimBlock, 0, stream>>>(
                (int8_t *)input, (int8_t *)output, param, ndarray_nhwc);
            break;
        case 2:
            cuda_kernel_cvtformat_nc<half><<<dimGrid, dimBlock, 0, stream>>>(
                (half *)input, (half *)output, param, ndarray_nhwc);
            break;
        case 4:
            cuda_kernel_cvtformat_nc<float><<<dimGrid, dimBlock, 0, stream>>>(
                (float *)input, (float *)output, param, ndarray_nhwc);
            break;
        case 8:
            cuda_kernel_cvtformat_nc<double><<<dimGrid, dimBlock, 0, stream>>>(
                (double *)input, (double *)output, param, ndarray_nhwc);
            break;
        default:
            break;
    }
    return;
}

CVTFormatMode GetCVTFormatMode(ReFormatParam param)
{
    if (param.in_format == DATAFORMAT_NDARRAY) {
        switch (param.out_format) {
            case DATAFORMAT_NHWC8:
                return NDARRAY_NHWC;
            case DATAFORMAT_NHWC16:
                return NDARRAY_NHWC;
            case DATAFORMAT_NHWC:
                return NDARRAY_NHWC;
            case DATAFORMAT_N4CX:
                return NDARRAY_N4CX;
            case DATAFORMAT_NCHW8:
                return NDARRAY_NC1HWC0;
            case DATAFORMAT_NCHW16:
                return NDARRAY_NC1HWC0;
            case DATAFORMAT_NHWC4:
                return NDARRAY_NHWC;
            case DATAFORMAT_NDARRAY:
                return CVTFormatUnknown;
            default:
                LOG(ERROR) << "Unsupport in format(" << GetDataFormatStr(param.in_format) << ") to out format(" << GetDataFormatStr(param.out_format) << ")";
                return CVTFormatUnknown;
        }
    } else if (param.in_format == DATAFORMAT_N4CX) {
        switch (param.out_format) {
            case DATAFORMAT_NDARRAY:
                return N4CX_NDARRAY;
            case DATAFORMAT_N4CX:
                return CVTFormatUnknown;
            default:
                LOG(ERROR) << "Unsupport in format(" << GetDataFormatStr(param.in_format) << ") to out format(" << GetDataFormatStr(param.out_format) << ")";
                return CVTFormatUnknown;
        }
    } else if (param.in_format == DATAFORMAT_NHWC) {
        switch (param.out_format) {
            case DATAFORMAT_NDARRAY:
                return NHWC_NDARRAY;
            case DATAFORMAT_NHWC16:
                return NHWC8_NHWC16;
            case DATAFORMAT_NHWC8:
                return NHWC8_NHWC16;
            case DATAFORMAT_NCHW8:
                return NHWC_NC1HWC0;
            case DATAFORMAT_NCHW16:
                return NHWC_NC1HWC0;
            case DATAFORMAT_NHWC:
                return CVTFormatUnknown;
            default:
                LOG(ERROR) << "Unsupport in format(" << GetDataFormatStr(param.in_format) << ") to out format(" << GetDataFormatStr(param.out_format) << ")";
                return CVTFormatUnknown;
        }
    } else if (param.in_format == DATAFORMAT_NHWC8) {
        switch (param.out_format) {
            case DATAFORMAT_NDARRAY:
                return NHWC_NDARRAY;
            case DATAFORMAT_NHWC16:
                return NHWC8_NHWC16;
            case DATAFORMAT_NHWC:
                return NHWC8_NHWC16;
            case DATAFORMAT_NCHW8:
                return NHWC_NC1HWC0;
            case DATAFORMAT_NCHW16:
                return NHWC_NC1HWC0;
            case DATAFORMAT_NHWC8:
                return CVTFormatUnknown;
            default:
                LOG(ERROR) << "Unsupport in format(" << GetDataFormatStr(param.in_format) << ") to out format(" << GetDataFormatStr(param.out_format) << ")";
                return CVTFormatUnknown;
        }
    } else if (param.in_format == DATAFORMAT_NHWC16) {
        switch (param.out_format) {
            case DATAFORMAT_NDARRAY:
                return NHWC_NDARRAY;
            case DATAFORMAT_NHWC8:
                return NHWC16_NHWC8;
            case DATAFORMAT_NHWC:
                return NHWC8_NHWC16;
            case DATAFORMAT_NCHW8:
                return NHWC_NC1HWC0;
            case DATAFORMAT_NCHW16:
                return NHWC_NC1HWC0;
            case DATAFORMAT_NHWC16:
                return CVTFormatUnknown;
            default:
                LOG(ERROR) << "Unsupport in format(" << GetDataFormatStr(param.in_format) << ") to out format(" << GetDataFormatStr(param.out_format) << ")";
                return CVTFormatUnknown;
        }
    } else if (param.in_format == DATAFORMAT_NCHW8) {
        switch (param.out_format) {
            case DATAFORMAT_NDARRAY:
                return NC1HWC0_NDARRAY;
            case DATAFORMAT_NHWC:
                return NC1HWC0_NHWC;
            case DATAFORMAT_NHWC16:
                return NC1HWC0_NHWC;
            case DATAFORMAT_NHWC8:
                return NC1HWC0_NHWC;
            case DATAFORMAT_NCHW16:
                return NC1HWC0_NC1HWC0;
            case DATAFORMAT_NCHW8:
                return CVTFormatUnknown;
            default:
                LOG(ERROR) << "Unsupport in format(" << GetDataFormatStr(param.in_format) << ") to out format(" << GetDataFormatStr(param.out_format) << ")";
                return CVTFormatUnknown;
        }
    } else if (param.in_format == DATAFORMAT_NCHW16) {
        switch (param.out_format) {
            case DATAFORMAT_NDARRAY:
                return NC1HWC0_NDARRAY;
            case DATAFORMAT_NHWC:
                return NC1HWC0_NHWC;
            case DATAFORMAT_NHWC16:
                return NC1HWC0_NHWC;
            case DATAFORMAT_NHWC8:
                return NC1HWC0_NHWC;
            case DATAFORMAT_NCHW8:
                return NC1HWC0_NC1HWC0;
            case DATAFORMAT_NCHW16:
                return CVTFormatUnknown;
            default:
                LOG(ERROR) << "Unsupport in format(" << GetDataFormatStr(param.in_format) << ") to out format(" << GetDataFormatStr(param.out_format) << ")";
                return CVTFormatUnknown;
        }
    } else {
        LOG(ERROR) << "Unsupport in format(" << GetDataFormatStr(param.in_format) << ") to out format(" << GetDataFormatStr(param.out_format) << ")";
        return CVTFormatUnknown;
    }
}

CVTTypeMode GetCVTTypeMode(ReFormatParam param)
{
    if (param.in_type == DATATYPE_FLOAT32) {
        switch (param.out_type) {
            case DATATYPE_FLOAT16:
                return FLOAT32_FLOAT16;
            case DATATYPE_FLOAT64:
                return FLOAT32_FLOAT64;
            case DATATYPE_INT8:
                return FLOAT32_INT8;
            case DATATYPE_INT4B:
                return FLOAT32_INT4B;
            case DATATYPE_FLOAT32:
                return CVTTypeUnknown;
            default:
                LOG(ERROR) << "Unsupport in datatype(" << GetDataTypeStr(param.in_type) << ") to out datatype(" << GetDataTypeStr(param.out_type) << ")";
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_FLOAT64) {
        switch (param.out_type) {
            case DATATYPE_FLOAT32:
                return FLOAT64_FLOAT32;
            case DATATYPE_FLOAT64:
                return CVTTypeUnknown;
            default:
                LOG(ERROR) << "Unsupport in datatype(" << GetDataTypeStr(param.in_type) << ") to out datatype(" << GetDataTypeStr(param.out_type) << ")";
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_FLOAT16) {
        switch (param.out_type) {
            case DATATYPE_FLOAT32:
                return FLOAT16_FLOAT32;
            case DATATYPE_INT8:
                return FLOAT16_INT8;
            case DATATYPE_INT4B:
                return FLOAT16_INT4B;
            case DATATYPE_FLOAT16:
                return CVTTypeUnknown;
            case DATATYPE_INT64:
                return FLOAT16_INT64;
            default:
                LOG(ERROR) << "Unsupport in datatype(" << GetDataTypeStr(param.in_type) << ") to out datatype(" << GetDataTypeStr(param.out_type) << ")";
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_INT8) {
        switch (param.out_type) {
            case DATATYPE_FLOAT16:
                return INT8_FLOAT16;
            case DATATYPE_FLOAT32:
                return INT8_FLOAT32;
            case DATATYPE_INT4B:
                return INT8_INT4B;
            case DATATYPE_INT8:
                return INT8_INT8;
            default:
                LOG(ERROR) << "Unsupport in datatype(" << GetDataTypeStr(param.in_type) << ") to out datatype(" << GetDataTypeStr(param.out_type) << ")";
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_UINT8) {
        switch (param.out_type) {
            case DATATYPE_FLOAT32:
                return UINT8_FLOAT32;
            case DATATYPE_FLOAT16:
                return UINT8_FLOAT16;
            case DATATYPE_UINT8:
                return CVTTypeUnknown;
            default:
                LOG(ERROR) << "Unsupport in datatype(" << GetDataTypeStr(param.in_type) << ") to out datatype(" << GetDataTypeStr(param.out_type) << ")";
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_INT4B) {
        switch (param.out_type) {
            case DATATYPE_FLOAT16:
                return INT4B_FLOAT16;
            case DATATYPE_FLOAT32:
                return INT4B_FLOAT32;
            case DATATYPE_INT8:
                return INT4B_INT8;
            case DATATYPE_INT4B:
                return INT4B_INT4B;
            default:
                LOG(ERROR) << "Unsupport in datatype(" << GetDataTypeStr(param.in_type) << ") to out datatype(" << GetDataTypeStr(param.out_type) << ")";
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_INT32) {
        switch (param.out_type) {
            case DATATYPE_INT64:
                return INT32_INT64;
            case DATATYPE_INT32:
                return CVTTypeUnknown;
            default:
                LOG(ERROR) << "Unsupport in datatype(" << GetDataTypeStr(param.in_type) << ") to out datatype(" << GetDataTypeStr(param.out_type) << ")";
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_INT64) {
        switch (param.out_type) {
            case DATATYPE_INT32:
                return INT64_INT32;
            case DATATYPE_FLOAT32:
                return INT64_FLOAT32;
            case DATATYPE_BOOL:
                return INT64_BOOL;
            case DATATYPE_INT64:
                return CVTTypeUnknown;
            default:
                LOG(ERROR) << "Unsupport in datatype(" << GetDataTypeStr(param.in_type) << ") to out datatype(" << GetDataTypeStr(param.out_type) << ")";
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_BOOL) {
        switch (param.out_type) {
            case DATATYPE_INT64:
                return BOOL_INT64;
            case DATATYPE_BOOL:
                return CVTTypeUnknown;
            default:
                LOG(ERROR) << "Unsupport in datatype(" << GetDataTypeStr(param.in_type) << ") to out datatype(" << GetDataTypeStr(param.out_type) << ")";
                return CVTTypeUnknown;
        }
    }
    LOG(ERROR) << "Unsupport in datatype(" << GetDataTypeStr(param.in_type) << ") to out datatype(" << GetDataTypeStr(param.out_type) << ")";
    return CVTTypeUnknown;
}

bool IsFloatEqual(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (uint32_t i = 0; i < a.size(); i++) {
        if (fabs(a[i] - b[i]) > FLT_EPSILON) {
            return false;
        }
    }
    return true;
}

bool EqualQuant(const CudaTensorKernelQuant& quant_a, const CudaTensorKernelQuant& quant_b) {
    return quant_a.bit_width == quant_b.bit_width &&
           IsFloatEqual(quant_a.scale, quant_b.scale) &&
           IsFloatEqual(quant_a.zero_point, quant_b.zero_point);
}

ppl::common::RetCode SetReLayoutParam(
    ReFormatParam *param,
    const ppl::common::TensorShape& input,
    const ppl::common::TensorShape& output)
{
    if (input.GetDimCount() <= 1 &&
        ((input.GetDataFormat() == DATAFORMAT_NHWC8) ||
        (output.GetDataFormat() == DATAFORMAT_NHWC8) ||
        (input.GetDataFormat() == DATAFORMAT_NHWC16) ||
        (output.GetDataFormat() == DATAFORMAT_NHWC16) ||
        (input.GetDataFormat() == DATAFORMAT_NCHW8) || 
        (output.GetDataFormat() == DATAFORMAT_NCHW8) || 
        (input.GetDataFormat() == DATAFORMAT_NCHW16) || 
        (output.GetDataFormat() == DATAFORMAT_NCHW16) || 
        (input.GetDataFormat() == DATAFORMAT_NHWC4) || 
        (output.GetDataFormat() == DATAFORMAT_NHWC4)))
        return RC_INVALID_VALUE;
    param->n_outer = input.GetDim(0);
    param->channel = input.GetDimCount() > 1 ? input.GetDim(1) : 1;
    param->n_inner = input.GetDimCount() > 2 ? input.CalcElementsFromDimensionIncludingPadding(2) : 1;
    param->in_format = input.GetDataFormat();
    param->out_format = output.GetDataFormat();
    param->in_type = input.GetDataType();
    param->out_type = output.GetDataType();
    param->mix_type   = (param->in_type != param->out_type);
    param->mix_format = (param->in_format != param->out_format);

    param->src_pad = Align(param->channel, AlignDataFormat(param->in_format));
    param->dst_pad = Align(param->channel, AlignDataFormat(param->out_format));

    param->out_elems = output.CalcElementsIncludingPadding();
    param->in_elems = input.CalcElementsIncludingPadding();
    return RC_SUCCESS;

}

ppl::common::RetCode SetReLayoutParam(
    ReFormatParam *param,
    const ppl::common::TensorShape& input,
    const CudaTensorKernelQuant& input_quant,
    const ppl::common::TensorShape& output,
    const CudaTensorKernelQuant& output_quant)
{
    SetReLayoutParam(param, input, output);
    param->same_scale = IsFloatEqual(input_quant.scale, output_quant.scale);
    if (input_quant.per_channel) {
        param->per_channel = true;
        param->quant_stride = input.GetDataFormat() == DATAFORMAT_NDARRAY? param->n_inner : 1;
        param->quant_dim_size = param->n_outer;
        param->quant_stride *= param->channel;
    } else {
        param->i_step = input_quant.scale[0];
        param->o_step = output_quant.scale[0];
    }
    param->i_zero_point = input_quant.zero_point[0];
    param->o_zero_point = output_quant.zero_point[0];
    if (param->in_type == param->out_type && param->in_type == DATATYPE_INT8) {
        param->mix_type = !EqualQuant(input_quant, output_quant);
    }
    return RC_SUCCESS;
}

void PPLCUDADataConvert(
    cudaStream_t stream,
    const void* input,
    void* output,
    void* tempBuf,
    ReFormatParam& param)
{
    bool only_nc = param.n_inner == 1 && (GetCVTFormatMode(param) == NDARRAY_NHWC || GetCVTFormatMode(param) == NDARRAY_NC1HWC0);
    bool if_padded = param.dst_pad != param.src_pad;
    if (param.in_format != param.out_format && (param.in_type != param.out_type || !param.same_scale)) { // mix-type and mix-format
        if (param.per_channel) {
            PPLCUDACVTTypePerChannel(stream, input, tempBuf, param);
            PPLCUDACVTFormat(stream, tempBuf, output, param);
        } else if (only_nc) { // ndarray<->nhwc/ndarray<->nc1hwc0, in-shape is N*C, out-shape is N*C_pad
            if (!if_padded) { PPLCUDACVTTypePerTensor(stream, input, output, param);
            } else { PPLCUDACVTFormatTypeNC(stream, input, output, param); }
        } else {
            PPLCUDACVTFormatType(stream, input, output, param);
        }
        return;
    } else if (param.in_format != param.out_format && (param.in_type = param.out_type && param.same_scale)) { // only mix-format
        if (only_nc) { // ndarray<->nhwc/ndarray<->nc1hwc0, in-shape is N*C, out-shape is N*C_pad
            PPLCUDACVTFormatNC(stream, input, output, param);
        } else {
            PPLCUDACVTFormat(stream, input, output, param);
        }
        return;
    } else if (param.in_type != param.out_type || !param.same_scale) { // only mix-type
        if (param.per_channel) {
            PPLCUDACVTTypePerChannel(stream, input, output, param);
        } else {
            PPLCUDACVTTypePerTensor(stream, input, output, param);
        }
        return;
    } else {
        return;
    }
}
