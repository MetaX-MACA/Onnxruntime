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

#include "cudakernel/unary/hardsigmoid.h"
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <typename DataT>
__device__ __inline__ DataT ppl_scalar_hardsigmoid(const DataT& in_val, float alpha, float beta);

template <>
__device__ __inline__ float ppl_scalar_hardsigmoid<float>(const float& in_val, float alpha, float beta)
{
    float res;
    res = max(0.0, min(1.0, alpha * in_val + beta));
    return res;
}

__device__ __inline__ int8_t ppl_scalar_hardsigmoid_int8(const int8_t& in_val, float alpha, float beta, float in_scale, float out_scale)
{
    int8_t res;
    float res_f = max(0.0, min(1.0, alpha * in_scale * in_val + beta));
    res = max(min(round(res_f / out_scale), 127.0), -127.0);
    return res;
}

template <>
__device__ __inline__ half ppl_scalar_hardsigmoid<half>(const half& in_val, float alpha, float beta)
{
    float res = __half2float(in_val);
    res = max(0.0, min(1.0, alpha * res + beta));
    return __float2half(res);
}
#endif

template <typename DataT>
__global__ void ppl_cukernel_unary_hardsigmoid(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output,
    float alpha,
    float beta)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    output[index] = ppl_scalar_hardsigmoid<DataT>(in_val, alpha, beta);
#endif
}

__global__ void ppl_cukernel_unary_hardsigmoid(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    float alpha,
    float beta,
    float in_scale,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int8_t in_val  = input[index];
    output[index] = ppl_scalar_hardsigmoid_int8(in_val, alpha, beta, in_scale, out_scale);
#endif
}

__global__ void ppl_cukernel_unary_hardsigmoid_float(const uint64_t num_elems, const float *input,
                                                  float *ouput, const float alpha, const float beta)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    for (uint64_t index = threadIdx.x + blockIdx.x * blockDim.x; index < num_elems; index += blockDim.x * gridDim.x)
    {
        float in_val = input[index];
        ouput[index] = ppl_scalar_hardsigmoid(in_val, alpha, beta);
    }
#endif
}

__global__ void ppl_cukernel_unary_hardsigmoid_opt(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    float alpha,
    float beta,
    float in_scale,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_elems / 8) {
        int64_t in_val64   = ((int64_t*)input)[index];
        int64_t out_val    = 0;
        int8_t* out_val_ptr = reinterpret_cast<int8_t*>(&out_val);
        int8_t* in_val_ptr  = reinterpret_cast<int8_t*>(&in_val64);
        #pragma unroll
        for (int it = 0; it < 8; it++) {
            int8_t in_val = in_val_ptr[it];
            out_val_ptr[it] = ppl_scalar_hardsigmoid_int8(in_val, alpha, beta, in_scale, out_scale);
        }

        ((int64_t*)output)[index] = out_val;


    } else if (num_elems % 8 > 0 && index < num_elems / 8 + 1) {
         int64_t start_pos = index * 8;
         for (int64_t it = start_pos; it < num_elems; it++) {
            int8_t in_val = input[it];
            output[it] = ppl_scalar_hardsigmoid_int8(in_val, alpha, beta, in_scale, out_scale);
         }
    }
#endif
}

__global__ void ppl_cukernel_unary_hardsigmoid_fp16_opt(
    const uint64_t num_elems,
    const __half* input,
    __half* output,
    const float alpha,
    const float beta)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < num_elems / 8)
        {
            float4 in_val4 = ((float4*)input)[index];
            //float4 out_val4 = 0;
            float4 out_val4 = make_float4(0, 0, 0, 0);
            __half* out_val_ptr = reinterpret_cast<__half*>(&out_val4);
            __half* in_val_ptr = reinterpret_cast<__half*>(&in_val4);
            #pragma unroll
            for (int it = 0; it < 8; it++)
            {
                __half in_val = in_val_ptr[it];
                out_val_ptr[it] = ppl_scalar_hardsigmoid<__half>(in_val, alpha, beta);
            }
            ((float4*)output)[index] = out_val4;
             }
        else if(index < num_elems / 8 +1 && num_elems % 8 > 0)
        {
            int64_t start_pos = index * 8;
            for (int64_t it = start_pos; it < num_elems; it++)
            {
               __half in_val = input[it];
                output[it] = ppl_scalar_hardsigmoid<__half>(in_val, alpha, beta);
            }
           }
#endif
}

ppl::common::RetCode PPLCUDAUnaryHardSigmoidForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    float alpha,
    float beta,
    float in_scale,
    float out_scale)
{
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int block_size     = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        // printf("alpha: %f, beta: %f\n", alpha, beta);
        //  ppl_cukernel_unary_hardsigmoid<float><<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (float*)output, alpha, beta);
        int capacity_per_block = block_size * 4;
        grid_size = (num_elems + capacity_per_block - 1) / capacity_per_block;
        ppl_cukernel_unary_hardsigmoid_float<<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (float*)output, alpha, beta);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        // ppl_cukernel_unary_hardsigmoid<half><<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, (half*)output, alpha, beta);
        int capacity_per_block = block_size * 8;
        grid_size = (num_elems + capacity_per_block - 1) / capacity_per_block;
        ppl_cukernel_unary_hardsigmoid_fp16_opt<<<grid_size, block_size, 0, stream>>>(num_elems, (const __half*)input, (__half*)output, alpha, beta);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        int capacity_per_block = block_size * 8;
        grid_size = (num_elems + capacity_per_block - 1) / capacity_per_block;
        ppl_cukernel_unary_hardsigmoid_opt<<<grid_size, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, alpha, beta, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}
