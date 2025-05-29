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

#include "cudakernel/nn/softmax.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/reduce/reduce.h"
#include "cudakernel/arithmetic/arithmetic.h"
#include "cudakernel/unary/exp.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "cudakernel/common/common.cuh"
#include "cudakernel/common/common.h"
#include "../reformat/cvt_int8_float.cuh"
#include "ppl/common/log.h"
#include "cudakernel/common/atomic.h"

#define _HLAF_MIN -65504
#define _FLT_MIN  -3.40282346638528859811704183484516925e+38F

#ifdef __MACACC__
#define __SOFTMAX_OPT__
#endif//__MACACC__

template <typename T>
inline __host__ __device__ T get_min();

template <>
inline __host__ __device__ float get_min<float>()
{
    return _FLT_MIN;
}
template <>
inline __host__ __device__ half get_min<half>()
{
    return _HLAF_MIN;
}

template <typename T>
__device__ inline T __ldg_ver_ctrl(T* ptr) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

uint64_t PPLSoftmaxGetTempBufferSize(
    const ppl::common::TensorShape* input_shape,
    int axis)
{
    int N = input_shape->CalcElementsIncludingPadding() / input_shape->GetDim(axis);
    return N * ppl::common::GetSizeOfDataType(input_shape->GetDataType());
}

ppl::common::RetCode PPLCUDASoftmaxForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    int axis)
{
    int N = input_shape->CalcElementsToDimensionIncludingPadding(axis);
    int R = input_shape->GetDim(axis);
    int D = input_shape->CalcElementsFromDimensionIncludingPadding(axis + 1);
    if(D==1){
        bool key_padding_mask = false;
        auto status = PPLCUDAFastSoftmax(stream, input_shape, input, output_shape, output, &key_padding_mask);
        return status;
    }
    // reduce max
    PPLReduceDimDes reduce_desc(D, R, N);
    ReduceParam reduce_max = ReduceMax;
    void* max_sum_output   = temp_buffer;
    ppl::common::TensorShape max_sum_shape(*input_shape);
    max_sum_shape.SetDimCount(3);
    max_sum_shape.SetDim(0, N);
    max_sum_shape.SetDim(1, 1);
    max_sum_shape.SetDim(2, D);

    auto status = PPLCUDAReduceForwardImp(stream, reduce_max, reduce_desc, input_shape, input, &max_sum_shape, max_sum_output);
    // sub
    ppl::common::TensorShape nd_shape(*input_shape);
    nd_shape.SetDimCount(3);
    nd_shape.SetDim(0, N);
    nd_shape.SetDim(1, R);
    nd_shape.SetDim(2, D);
    status = PPLCUDAArithMeticSubForwardImp(stream, &nd_shape, input, &max_sum_shape, max_sum_output, &nd_shape, output);
    // exp
    status                 = PPLCUDAExpForwardImp(stream, &nd_shape, output, &nd_shape, output);
    // reduce sum
    ReduceParam reduce_sum = ReduceSum;
    status = PPLCUDAReduceForwardImp(stream, reduce_sum, reduce_desc, &nd_shape, output, &max_sum_shape, max_sum_output);
    //div
    status = PPLCUDAArithMeticDivForwardImp(stream, &nd_shape, output, &max_sum_shape, max_sum_output, &nd_shape, output);
    return status;
}

__global__ void __launch_bounds__(256) ppl_cukernel_softmax_int8(
    const int8_t* input, int8_t* output, int max_int8,
    int outer, int axis_width, int inner,
    QuantKernelParamCuda qparam) {
    int tid = threadIdx.x;
    int inner_idx = blockIdx.x;
    int out_idx = blockIdx.y;
    __shared__ float shared[256];
    shared[tid] = 0.f;
    float max_val = _int82float(max_int8, qparam.i_step, qparam.i_zero_point);
    for(int id = tid; id < axis_width; id += blockDim.x) {
        if(id < axis_width) {
            uint64_t in_index = out_idx * axis_width * inner +
                id * inner + inner_idx;
            float in_val  = _int82float(input[in_index], qparam.i_step, qparam.i_zero_point);
            //calculate each c exp sum
            shared[tid] += expf(in_val - max_val);

        }
    }
    //accumulate all c exp sum
    __syncwarp();
    float exp_sum = BlockReduceSum(shared[tid]);

    for(int id = tid; id < axis_width; id += blockDim.x) {
        if(id < axis_width) {
            uint64_t in_index = out_idx * axis_width * inner +
                id * inner + inner_idx;
            //calculate output
            float in_val  = _int82float(input[in_index], qparam.i_step, qparam.i_zero_point);
            float out_val = expf(in_val - max_val) / exp_sum;
            output[in_index] = _float2int8(out_val, qparam.o_step, qparam.o_zero_point);
        }
    }
    __syncthreads();
}

ppl::common::RetCode PPLCUDASoftmaxForwardImpInt8(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    int axis,
    const QuantKernelParamCuda* qparam)
{
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    int outer = input_shape->CalcElementsToDimensionIncludingPadding(axis);
    int axis_width = input_shape->GetDim(axis);
    int inner = input_shape->CalcElementsFromDimensionIncludingPadding(axis + 1);
    // for int8 case, use 127 as the max_val
    int max_int8 = 127;
    int block_size = 256;
    dim3 grid_size(inner, outer, 1);
    ppl_cukernel_softmax_int8<<<grid_size, block_size, 0, stream>>>((const int8_t*)input,
            (int8_t*)output, max_int8, outer, axis_width, inner, *qparam);

    return status;
}

template <typename T, typename acc_t, int log2_ceil>
__global__ void SoftmaxWarpImpl(const T* X, T* Y, int o_dim, int i_dim) {
    constexpr int next_power_of_two = 1 << log2_ceil;
    constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    int local_batches = o_dim - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    int tid = threadIdx.x;
    
    uint offset = first_batch * i_dim + tid;
    X += offset;
    Y += offset;

    acc_t x[WARP_BATCH][WARP_ITERATIONS];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : i_dim;
        #pragma unroll
        for (int j = 0;  j < WARP_ITERATIONS;  ++j) {
            int element_index = tid + j * WARP_SIZE;
            if (element_index < batch_element_count) {
                x[i][j] = X[i * i_dim + j * WARP_SIZE];
            } else {
                x[i][j] = get_min<acc_t>();
            }
        }
    }

    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        max_value[i] = x[i][0];
        #pragma unroll
        for (int j = 1;  j < WARP_ITERATIONS;  ++j) {
            max_value[i] = (max_value[i] > x[i][j]) ? max_value[i] : x[i][j];
        }
    }
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    #pragma unroll
        for (int i = 0; i < WARP_BATCH; ++i) {
            acc_t b = __shfl_xor_sync(0xffffffff, max_value[i], offset);
            max_value[i] = (max_value[i] > b) ? max_value[i] : b;
        }
    }

    acc_t sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        #pragma unroll
        for (int j = 0;  j < WARP_ITERATIONS;  ++j) {
            x[i][j] = std::exp(x[i][j] - max_value[i]);
            sum[i] += x[i][j];
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    #pragma unroll
        for (int i = 0; i < WARP_BATCH; ++i) {
            sum[i] += __shfl_xor_sync(0xffffffff, sum[i], offset);
        }
    }
    
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        sum[i] = 1.f / sum[i];
    }

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int j = 0;  j < WARP_ITERATIONS;  ++j) {
            int element_index = tid + j * WARP_SIZE;
            if (element_index < i_dim) {
                Y[i*i_dim+j*WARP_SIZE] =  x[i][j] * sum[i];
            } else {
                break;
            }
        }
    }
}

#ifdef __SOFTMAX_OPT__
#define MACA_WARP_SIZE 64
//inner_dim [961,1024]
/**
 * @brief Unroll thread operation within a single warp.
 *
 * @tparam block_size threads number per block
 * @param sdata shared memory data
 * @param tid thread index
 * @return __device__
 */
template <uint32_t block_size, typename T = float>
__device__ __forceinline__ void WarpReduceAdd(volatile T *sdata, uint32_t tid) {
    if (block_size >= 128) sdata[tid] += sdata[tid + 64];
    if (block_size >= 64) sdata[tid] += sdata[tid + 32];
    if (block_size >= 32) sdata[tid] += sdata[tid + 16];
    if (block_size >= 16) sdata[tid] += sdata[tid + 8];
    if (block_size >= 8) sdata[tid] += sdata[tid + 4];
    if (block_size >= 4) sdata[tid] += sdata[tid + 2];
    if (block_size >= 2) sdata[tid] += sdata[tid + 1];
}

/**
 * @brief Unroll thread operation within a single warp.
 *
 * @tparam block_size threads number per block
 * @param sdata shared memory data
 * @param tid thread index
 * @return __device__
 */
template <uint32_t block_size, typename T = float>
__device__ __forceinline__ void WarpReduceMax(volatile T *sdata, uint32_t tid) {
    if (block_size >= 256) sdata[tid] = max(sdata[tid],sdata[tid + 128]);
    if (block_size >= 128) sdata[tid] = max(sdata[tid], sdata[tid + 64]);
    if (block_size >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    if (block_size >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    if (block_size >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
    if (block_size >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
    if (block_size >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
    if (block_size >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
}

template <typename T>
__device__ __forceinline__ T max_fp16(T v0,T v1)
{
    return v0 < v1? v1: v0;
}

template <typename T, typename acc_t>
__global__ void
SoftmaxWarpImpl_Opt_Large(T* output, T* input, int i_dim, int o_dim,int blockSize,acc_t max_default)
{
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[]; //size: 1024*sizeof(half) + 64*sizeof(float)*2
    acc_t* sdata = reinterpret_cast<acc_t*>(shared_buf);
    acc_t* sbuffer = (acc_t*)(sdata + 1024);
    acc_t thread_max = max_default;
    int tid = threadIdx.x;
    input = input + blockIdx.x*i_dim;
    output = output + blockIdx.x*i_dim;
    
    for(int i = tid; i < i_dim; i += blockSize)
    {
        sdata[i] = *(input + i);
        thread_max = max(thread_max,sdata[i]);
    }

    sbuffer[tid] = thread_max;

    __syncthreads();
    
    if(tid < 128) WarpReduceMax<256, acc_t>(sbuffer, tid);
     __syncthreads();
  
    acc_t thread_sum = 0.0f;
    acc_t row_max = sbuffer[0];
    
    for(int i = tid; i < i_dim; i += blockSize)
    {
        acc_t exp_x = __builtin_expf(sdata[i] - row_max);
        sdata[i] = exp_x;
        thread_sum += exp_x;
    }
    
    sbuffer[tid] = thread_sum;
    __syncthreads();

    if(tid < 128){
        sbuffer[tid] += sbuffer[tid+128];
    }
    __syncthreads();

    if(tid < 64) WarpReduceAdd<256,float>(sbuffer,tid);
    __syncthreads();
    acc_t sum = sbuffer[0];
    sum = __builtin_mxc_rcpf(sum);
    
     for(int i = tid; i < i_dim; i += blockSize)
     {
        T val = sdata[i] * sum;
        output[i] = val;
     }
}

// template <typename T, typename acc_t>
// __global__ void
// SoftmaxWarpImpl_Opt_Large(T* output, T* input, int i_dim, int o_dim,int blockSize,T max_default)
// {
//     extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[]; //size: 1024*sizeof(float) + 64*sizeof(float)*2
//     acc_t* sdata = reinterpret_cast<acc_t*>(shared_buf);
//     acc_t* sbuffer = (acc_t*)(sdata + 1024);
//     T thread_max = max_default;
//     int tid = threadIdx.x;
//     int i;
//     int blockSize4 = blockSize<<2;
//     input = input + blockIdx.x*i_dim;
//     output = output + blockIdx.x*i_dim;
    
//     if((blockIdx.x&3)==0)
//     {
//       for(i = (tid<<2); i < i_dim - 3; i += blockSize4)
//       {
//           float2 reg;
//           half*buffer=(half*)&reg;
//           *(float2*)buffer = *((float2*)(input + i));
//           sdata[i] = buffer[0];
//           sdata[i + 1] = buffer[1];
//           sdata[i + 2] = buffer[2];
//           sdata[i + 3] = buffer[3];
//           thread_max = max_fp16<T>(thread_max,buffer[0]);
//           thread_max = max_fp16<T>(thread_max,buffer[1]);
//           thread_max = max_fp16<T>(thread_max,buffer[2]);
//           thread_max = max_fp16<T>(thread_max,buffer[3]);
//       }
//       for(; i < i_dim; i += blockSize)
//       {
//         T tmp = *(input + i);
//         sdata[i] = tmp;
//         thread_max = max_fp16<T>(thread_max,tmp);
//       }
//     }else{
//       for(i = tid; i < i_dim; i += blockSize)
//       {
//         T tmp = *(input + i);
//         sdata[i] = tmp;
//         thread_max = max_fp16<T>(thread_max,tmp);
//       }
//     }
  
//     sbuffer[tid] = (float)thread_max;

//     __syncthreads();
    
//     if(tid < 128) WarpReduceMax<256, acc_t>(sbuffer, tid);
//      __syncthreads();
  
//     acc_t thread_sum = 0.0f;
//     acc_t row_max = sbuffer[0];
//     for(i = tid; i < i_dim; i += blockSize)
//     {
//         acc_t exp_x = __builtin_expf(sdata[i] - row_max);
//         sdata[i] = exp_x;
//         thread_sum += exp_x;
//     }
    
//     sbuffer[tid] = thread_sum;
//     __syncthreads();

//     if(tid < 128){
//         sbuffer[tid] += sbuffer[tid+128];
//     }
//     __syncthreads();

//     if(tid < 64) WarpReduceAdd<256,float>(sbuffer,tid);
//     __syncthreads();
//     acc_t sum = sbuffer[0];
//     sum = __builtin_mxc_rcpf(sum);
//     if((blockIdx.x&3)==0)
//     {
//         for(i = (tid<<2); i < i_dim - 3; i += blockSize4)
//         {
//             float2 reg;
//             half*buffer=(half*)&reg;
//             buffer[0] = sdata[i] * sum;
//             buffer[1] = sdata[i + 1] * sum;
//             buffer[2] = sdata[i + 2] * sum;
//             buffer[3] = sdata[i + 3] * sum;
//             *((float2*)(output + i)) = reg;
//         }
//         for(; i < i_dim; i += blockSize)
//         {
//             T val = sdata[i] * sum;
//             output[i] = val;
//         }
//     }else{
//         for(i = tid; i < i_dim; i += blockSize)
//         {
//             T val = sdata[i] * sum;
//             output[i] = val;
//         }
//     }
// }

template <typename T,typename acc_t>
__global__ void SoftmaxWarpImpl_Opt_Small(const T* X,T* Y,const int o_dim,const int i_dim,const int tile_row,acc_t max_default){
    extern __shared__ acc_t sm_x[];
    acc_t* sm_max = sm_x + 256;
    acc_t* sm_sum = sm_max + 256;
    
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int start_row = blockIdx.x*blockDim.y*tile_row;
    int height = min(o_dim - start_row,tile_row*blockDim.y);
    int count = height * i_dim;
    int inner_count = tile_row * i_dim;
    inner_count = min(inner_count,count);
    int start_offset = start_row * i_dim;
    int inner_offset = tidy * tile_row*i_dim;
    int tid = tidy * blockDim.x + tidx;
    unsigned long mask;
    X += start_offset;
    Y += start_offset;
    
    for(int i = tid; i < count; i += 256){
        sm_x[i] = X[i];
    }
    __syncthreads();
    acc_t *pSm_x = sm_x + inner_offset;
    acc_t *pSm_max = sm_max + inner_offset;
    acc_t *pSm_sum = sm_sum + inner_offset;
    int phase0 = (tidx / i_dim)*i_dim;
    int phase1 = phase0 + i_dim; 
    mask = (1UL<<phase1) - (1UL<<phase0);
    
    for(int i = tidx,index = 0; i < inner_count; i += blockDim.x,index++)
    {
        pSm_max[i] = __reduce_max_sync(mask,pSm_x[i]);
    }
    __syncthreads();
    
    for(int i = tidx, index = 0; i < inner_count; i += blockDim.x,index++)
    {
        pSm_x[i] = __builtin_expf(pSm_x[i] - pSm_max[i]);
        pSm_sum[i] = __reduce_add_sync(mask, pSm_x[i]);
    }
    __syncthreads();
    
    for(int i = tidx, index = 0; i < inner_count; i += blockDim.x,index++){
        pSm_x[i] = pSm_x[i]* __builtin_mxc_rcpf(pSm_sum[i]);
    }
    
    for(int i = tid; i < count; i += 256){
        Y[i] = sm_x[i];
    }
}

// template <typename T,typename acc_t>
// __global__ void SoftmaxWarpImpl_Opt_Small_batch(const T* X,T* Y,const int o_dim,const int i_dim,const int tile_row,acc_t max_default){
//     extern __shared__ acc_t sm_x[];
//     acc_t* sm_max = sm_x + 2048;
//     acc_t buffer_sum[4];
//     int size = tile_row*blockDim.y;
//     int tidx = threadIdx.x, tidy = threadIdx.y;
//     int start_row = blockIdx.x*size<<3;
//     int height = min(o_dim - start_row,size<<3);
//     int loopcount = (height + tile_row*blockDim.y - 1) / (tile_row*blockDim.y);
//     int count = height * i_dim;
//     int size1 = size*i_dim;
//     int start_offset = start_row * i_dim;
//     int inner_offset = tidy * tile_row*i_dim;
//     int tid = tidy * blockDim.x + tidx;
//     int height_offset = tidy*tile_row;
//     unsigned long mask;
//     X += start_offset;
//     Y += start_offset;

//     for(int i = tid; i < count; i += 256){
//         sm_x[i] = X[i];
//     }
//     __syncthreads();
    
//     acc_t *ptr_sm = sm_x;

//     for(int j = 0; j < loopcount; j++)
//     {
//         acc_t *pSm_x = ptr_sm + inner_offset;
//         ptr_sm += size1;
//         acc_t *pSm_max = sm_max + inner_offset;
        
//         int inner_count = min(tile_row,height - j*size - height_offset) * i_dim;
//         int wid = tidx / i_dim;
//         int phase0 = wid*i_dim;
//         int phase1 = phase0 + i_dim; 
//         mask = (1UL<<phase1) - (1UL<<phase0);
//         for(int i = tidx; i < inner_count; i += blockDim.x)
//         {
//             pSm_max[i] = __reduce_max_sync(mask,pSm_x[i]);
//         }
//         __syncthreads();
//         for(int i = tidx; i < inner_count; i += blockDim.x)
//         {
//             pSm_x[i] = __builtin_expf(pSm_x[i] - pSm_max[i]);
//         }
//         __syncthreads();
//         for(int i = i_dim - 1,j = 0; i < 64; i += i_dim, j++)
//         {
//             acc_t tmp = 0.0;
//             for(int k = 0; k < i_dim; k++)
//             {
//                  tmp += pSm_x[i - k]; 
//             }
//             buffer_sum[j] = tmp;
//         }
//         // for(int i = tidx; i < inner_count; i += blockDim.x)
//         // {
//         //     //   pSm_sum[i] = __reduce_add_sync(mask,pSm_x[i]);
//         // }   
//         // __syncthreads();
//         for(int i = tidx; i < inner_count; i += blockDim.x){
//             // pSm_x[i] = pSm_x[i]* __builtin_mxc_rcpf(pSm_sum[i]);
//             pSm_x[i] = pSm_x[i]* __builtin_mxc_rcpf(buffer_sum[wid]);
//         }
//     }

//     for(int i = tid; i < count; i += 256){
//         Y[i] = sm_x[i];
//     }
// }

template <typename T,typename acc_t>
__global__ void SoftmaxWarpImpl_Opt_Small_batch(const T* X,T* Y,const int o_dim,const int i_dim,const int tile_row,acc_t max_default){
    extern __shared__ acc_t sm_x[];
    acc_t* sm_max = sm_x + 2048;
    acc_t* sm_sum = sm_max + 256;

    acc_t buffer_sum[4];
    int size = tile_row*blockDim.y;
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int start_row = blockIdx.x*size<<3;
    int height = min(o_dim - start_row,size<<3);
    int loopcount = (height + tile_row*blockDim.y - 1) / (tile_row*blockDim.y);
    int count = height * i_dim;
    int size1 = size*i_dim;
    int start_offset = start_row * i_dim;
    int inner_offset = tidy * tile_row*i_dim;
    int inner_sum_offset = tidy*tile_row*64;
    int tid = tidy * blockDim.x + tidx;
    int height_offset = tidy*tile_row;
    unsigned long mask;
    X += start_offset;
    Y += start_offset;
    
    for(int i = tid; i < count; i += 256){
        sm_x[i] = X[i];
    }
    
    sm_sum[tid<<2] = 0.0f;
    sm_sum[(tid<<2) + 1] = 0.0f;
    sm_sum[(tid<<2) + 2] = 0.0f;
    sm_sum[(tid<<2) + 3] = 0.0f;
    __syncthreads();

    acc_t *ptr_sm = sm_x;

    for(int j = 0; j < loopcount; j++)
    {
        acc_t *pSm_x = ptr_sm + inner_offset;
        ptr_sm += size1;
        acc_t *pSm_max = sm_max + inner_offset;
        acc_t *pSm_sum = sm_sum + inner_sum_offset;

        int inner_tile = min(tile_row,height - j*size - height_offset);
        int inner_count = inner_tile * i_dim;
        int wid = tidx / i_dim;
        int phase0 = wid * i_dim;
        int phase1 = phase0 + i_dim; 
        mask = (1UL<<phase1) - (1UL<<phase0);

        acc_t *pSm_sum1 = pSm_sum + (wid<<6);
        
        for(int i = tidx; i < inner_count; i += blockDim.x)
        {
            pSm_max[i] = __reduce_max_sync(mask,pSm_x[i]);
        }
        __syncthreads();
        
        for(int i = tidx; i < inner_count; i += blockDim.x)
        {
            pSm_x[i] = __builtin_expf(pSm_x[i] - pSm_max[i]);
            pSm_sum1[i - phase0] = pSm_x[i];
        }
        __syncthreads();
        
        for(int i = 0, index = 0; i < inner_tile; i++,index+=64)
        {
            acc_t tmp = pSm_sum[index + tidx];
            #pragma unroll
            for (int offset = 32; offset > 0; offset>>=1) {
                tmp += __shfl_xor_sync(0xffffffffffffffff, tmp, offset,64);
            }
            __syncthreads();
            buffer_sum[i] = tmp;
        }
        
        for(int i = tidx, k = wid; i < inner_count; i += blockDim.x, k++){
            pSm_x[i] = pSm_x[i] * __builtin_mxc_rcpf(buffer_sum[k]);
        }
    }

    for(int i = tid; i < count; i += 256){
        Y[i] = sm_x[i];
    }
}

#endif//__SOFTMAX_OPT__

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void SoftmaxBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                     const int64_t cols) {

  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  __shared__ ComputeType row_sum_r;
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_max = -80.0f;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        thread_max = max(thread_max, pack[i]);
      }
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    #else
        const ComputeType row_max = blockReduceMax<ComputeType>(thread_max);
    #endif
    ComputeType thread_sum = 0;
    for (int col = tid; col < cols; col += block_size) {
        const ComputeType exp_x = std::exp(buf[col] - row_max);
        buf[col] = exp_x;
        thread_sum += exp_x;
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    #else
        const ComputeType row_sum = blockReduceSum<ComputeType>(thread_sum);
    #endif
    if(threadIdx.x == 0) row_sum_r = 1.f / row_sum;
    __syncthreads();

    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
          pack[i] = buf[i * num_packs + pack_id] * row_sum_r;
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void SoftmaxBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                         const int64_t cols) {
  const int tid = threadIdx.x;
  const int num_packs = cols / pack_size;
  __shared__ ComputeType row_sum_r;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_max = -80.0f;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_max = max(thread_max, pack[i]); }
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    #else
        const ComputeType row_max = blockReduceMax<ComputeType>(thread_max);
    #endif
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_sum += std::exp(pack[i] - row_max); }
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    #else
        const ComputeType row_sum = blockReduceSum<ComputeType>(thread_sum);
    #endif
    if(threadIdx.x == 0) row_sum_r = 1.f / row_sum;
    __syncthreads();
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        pack[i] = std::exp(pack[i] - row_max) * row_sum_r;
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

/*
par:
    in & out : [BHTT]
    key_padding_mask : [B, H, T, T] or [B, 1, T, T] or [B, 1, 1, T], or [1, 1, T, T]
*/
template<typename T, int pack_size>
ppl::common::RetCode PPLCUDAFastSoftmaxForwardImp(
    cudaStream_t stream,
    const T* input,
    T* output,
    const bool* key_padding_mask,
    const int mask_scale,
    const int outer_dim,
    const int inner_dim)
{
    using ComputeType = typename DefaultComputeType<T>::type;
    DirectLoad<T, ComputeType> load(input, inner_dim);
    DirectStore<ComputeType, T> store(output, inner_dim);
    if(inner_dim <= 1024)
	{
#ifdef __SOFTMAX_OPT__
        if(inner_dim<=64&&inner_dim >=16)
        {
            dim3 blockSize(64,4, 1);
            const int share_size = sizeof(ComputeType)*(3328);
            int row_num = 64 / inner_dim;
            int grid_size = (outer_dim + (row_num<<2) - 1)/ (row_num<<2);
            ComputeType max_default = get_min<ComputeType>();    
            if(grid_size>=100000){
                grid_size = (outer_dim + (row_num<<5) - 1)/ (row_num<<5);
                SoftmaxWarpImpl_Opt_Small_batch<T,ComputeType><<<grid_size,blockSize,share_size,stream>>>(
                        (T*)input, (T*)output, outer_dim, inner_dim,row_num,max_default
                    );
            }else{
                SoftmaxWarpImpl_Opt_Small<T,ComputeType><<<grid_size,blockSize,share_size,stream>>>(
                        (T*)input, (T*)output, outer_dim, inner_dim,row_num,max_default
                    );
            }
            return ppl::common::RC_SUCCESS;
        }else if(inner_dim >= 961 && inner_dim <= 1024){
            int grid = outer_dim;
            int share_size = (1024 + 256)*sizeof(ComputeType);
            ComputeType max_default = get_min<ComputeType>();    
            SoftmaxWarpImpl_Opt_Large<T, ComputeType>
            <<<grid, 256, share_size, stream>>>(
            (T*)output, (T*)input, inner_dim, outer_dim, 256,max_default);
            return ppl::common::RC_SUCCESS;
        }else if(inner_dim == 1){
            //[B, T, 1, 1]
            int grid = outer_dim / mask_scale;
            int block_size = 256;
            int share_size = (1024 + 256)*sizeof(ComputeType);
            ComputeType max_default = get_min<ComputeType>();    
            SoftmaxWarpImpl_Opt_Large<T, ComputeType><<<grid, block_size, share_size, stream>>>(
            (T*)output, (T*)input, mask_scale, outer_dim, block_size, max_default);
            return ppl::common::RC_SUCCESS;
        }
#endif//__SOFTMAX_OPT__
        int log2_ceil = 0;
        while((1 << log2_ceil) < inner_dim) log2_ceil++;
        const int next_power_of_two = 1 << log2_ceil;

        int warp_size = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;

        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int gridSize = (outer_dim + batches_per_block - 1) / batches_per_block;
		dim3 blockSize(warp_size, warps_per_block, 1);
        
        switch(log2_ceil) {
            case 0:
                SoftmaxWarpImpl<T,ComputeType,0><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 1:
                SoftmaxWarpImpl<T,ComputeType,1><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 2:
                SoftmaxWarpImpl<T,ComputeType,2><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 3:
                SoftmaxWarpImpl<T,ComputeType,3><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 4:
                SoftmaxWarpImpl<T,ComputeType,4><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 5:
                SoftmaxWarpImpl<T,ComputeType,5><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 6:
                SoftmaxWarpImpl<T,ComputeType,6><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 7:
                SoftmaxWarpImpl<T,ComputeType,7><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 8:
                SoftmaxWarpImpl<T,ComputeType,8><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 9:
                SoftmaxWarpImpl<T,ComputeType,9><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 10:
                SoftmaxWarpImpl<T,ComputeType,10><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            default:
                break;
        }
	} else {
        int grid = outer_dim;
        constexpr int block_size_conf = 256;
        const size_t smem = inner_dim * sizeof(ComputeType);
        int max_active_blocks_conf_1;
        {
            cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_active_blocks_conf_1,
                SoftmaxBlockSMemImpl<decltype(load), decltype(store), ComputeType, pack_size, block_size_conf>,
                block_size_conf, smem);
            if (err != cudaSuccess) { LOG(ERROR) << "cudaOccupancyMaxActiveBlocksPerMultiprocessor error"; }
        }
        if (max_active_blocks_conf_1 <= 0) {
            SoftmaxBlockUncachedImpl<decltype(load), decltype(store), ComputeType, pack_size, 1024><<<grid, 1024, 0, stream>>>(load, store, outer_dim, inner_dim);
        }

        SoftmaxBlockSMemImpl<decltype(load),decltype(store),ComputeType,pack_size,block_size_conf><<<grid, block_size_conf, smem, stream>>>(load, store, outer_dim, inner_dim);
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAFastSoftmax(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const void* key_padding_mask) 
{
    int dim_cnt = input_shape->GetDimCount();
    int outer_dim = 1;
    int inner_dim = input_shape->GetDim(dim_cnt - 1);
    for(int i = 0; i < dim_cnt - 1; i++) {
        outer_dim *= input_shape->GetDim(i);
    }
    int mask_scale = outer_dim / input_shape->GetDim(0);
    switch(output_shape->GetDataType()) {
        case ppl::common::DATATYPE_FLOAT32: {
            if (inner_dim % 2 == 0)
                PPLCUDAFastSoftmaxForwardImp<float, 2>(stream, (const float*)input, (float*)output, (const bool*)key_padding_mask, mask_scale, outer_dim, inner_dim);
            else
                PPLCUDAFastSoftmaxForwardImp<float, 1>(stream, (const float*)input, (float*)output, (const bool*)key_padding_mask, mask_scale, outer_dim, inner_dim);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16: {
            if (inner_dim % 2 == 0)
                PPLCUDAFastSoftmaxForwardImp<half, 2>(stream, (const half*)input, (half*)output, (const bool*)key_padding_mask, mask_scale, outer_dim, inner_dim);
            else
                PPLCUDAFastSoftmaxForwardImp<half, 1>(stream, (const half*)input, (half*)output, (const bool*)key_padding_mask, mask_scale, outer_dim, inner_dim);
            break;
        }
        default:
            break;
    }
    return ppl::common::RC_SUCCESS;
}
