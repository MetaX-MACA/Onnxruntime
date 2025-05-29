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

#include "cudakernel/unary/leakyrelu.h"
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <typename DataT>
__device__ __inline__ DataT ppl_scalar_leakyrelu(const DataT& in_val, float alpha);

template <>
__device__ __inline__ float ppl_scalar_leakyrelu<float>(const float& in_val, float alpha)
{
    float res;
    res = (in_val > 0) ? in_val : alpha * in_val;
    return res;
}

__device__ __inline__ int8_t ppl_scalar_leakyrelu_int8(const int8_t& in_val, float alpha, float in_scale, float out_scale)
{
    int8_t res;
    float res_f = (in_val > 0) ? in_val : alpha * in_val;
    res = round(res_f * in_scale / out_scale);
    res = min(127, max(-128, res));
    return res;
}

#ifdef __MACACC__
__device__ __inline__ uint64_t ppl_scalar_leakyrelu_int8_opt(uint64_t input, float alpha, float in_scale, float out_scale)
{
    uint64_t out_val = 0;
    int8_t* input_pt = (int8_t*)&input;
    int8_t* output_pt = (int8_t*)&out_val;
    for(int i = 0;i<8;i++){
        int8_t in_val  = input_pt[i];
        float res_f = (in_val > 0) ? in_val : alpha * in_val;
        //output_pt[i] = __builtin_roundf(res_f * in_scale * __builtin_mxc_rcpf(out_scale));
        int8_t out_val = __builtin_roundf(res_f * in_scale * __builtin_mxc_rcpf(out_scale));
        output_pt[i] = min(127, max(-128, out_val));
    }

    return out_val;
}
#endif //__MACACC__

template <>
__device__ __inline__ half ppl_scalar_leakyrelu<half>(const half& in_val, float alpha)
{
    half res;
    res = __hgt(in_val, 0) ? in_val : __hmul((half)alpha, in_val);
    return res;
}
#endif

template <typename DataT>
__global__ void ppl_cukernel_unary_leakyrelu(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output,
    float alpha)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    output[index] = ppl_scalar_leakyrelu<DataT>(in_val, alpha);
#endif
}

__global__ void ppl_cukernel_unary_leakyrelu(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    float alpha,
    float in_scale,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int8_t in_val  = input[index];
    output[index] = ppl_scalar_leakyrelu_int8(in_val, alpha, in_scale, out_scale);
#endif
}

#ifdef __MACACC__
#define INT8_COMPAT 8
#define FLOAT_COMPAT 4
#define HALF_COMPAT 8

__global__ void ppl_cukernel_unary_leakyrelu_opt(
    const uint64_t num_elems,
    const uint64_t disc,
    const int remainder,
    const float* input,
    float* output,
    float alpha)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index > disc) return;
    if (index < disc) {
        float4 in_val = *((float4*)input + index);
        float4 *out_val = (float4*)output + index;
        float4 reg;
        float *ptr_data = (float*)&in_val;
        float *ptr_reg = (float*)&reg;
        for (int i = 0; i < FLOAT_COMPAT; i++) {
            ptr_reg[i] = ppl_scalar_leakyrelu<float>(ptr_data[i], alpha);
        }
        *out_val = reg;
        return;
    } else if (remainder > 0){
        uint64_t start_pos = index * FLOAT_COMPAT;
        for (uint64_t i = start_pos; i < num_elems; i++) {
            output[i] = ppl_scalar_leakyrelu<float>(input[i], alpha);
        }
    } 
#endif
}
__global__ void ppl_cukernel_unary_leakyrelu_opt(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    float alpha,
    float in_scale,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index < num_elems / INT8_COMPAT) {
        uint64_t in_val = ((uint64_t*)input)[index];
        uint64_t out_val = ppl_scalar_leakyrelu_int8_opt(in_val, alpha, in_scale, out_scale);
        ((uint64_t*)output)[index] = out_val;
        return;
    } else if (num_elems % INT8_COMPAT > 0 && index < num_elems / INT8_COMPAT + 1) {
        uint64_t start_pos = index * INT8_COMPAT;
        for (uint64_t i = start_pos; i < num_elems; i++) {
            int8_t in_val  = input[i];
            output[i] = ppl_scalar_leakyrelu_int8(in_val, alpha, in_scale, out_scale);
        }
    } else {
        return;
    }
#endif
}


__global__ void ppl_cukernel_unary_leakyrelu_opt_fp16(
    const uint64_t num_elems,
    const uint64_t num_float4,
    const int remainder,
    const half* input,
    half* output,
    float alpha)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > num_float4) return;
    if (index < num_float4)
    {
        float4 in_val = *((float4*)input + index);
        float4 *out_val = (float4*)output + index;
        float4 reg;
        half *ptr_data = (half*)&in_val;
        half *ptr_reg = (half*)&reg;
        for (int i = 0; i < HALF_COMPAT; i++) {
            ptr_reg[i] = ppl_scalar_leakyrelu<half>(ptr_data[i], alpha);
        }
        *out_val = reg;
        return;
    }
    else if (remainder > 0)
    {
        uint64_t start_pos = index * HALF_COMPAT;
        for (uint64_t i = start_pos; i < num_elems; i++) {
            output[i] = ppl_scalar_leakyrelu<half>(input[i], alpha);
        }
    }
}


ppl::common::RetCode PPLCUDAUnaryLeakyReluForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    float alpha,
    float in_scale,
    float out_scale)
{
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int block_size     = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        int capacity_per_block = block_size * FLOAT_COMPAT;
        uint64_t gs = (num_elems + capacity_per_block - 1) / capacity_per_block;
        ppl_cukernel_unary_leakyrelu_opt<<<gs, block_size, 0, stream>>>(num_elems, num_elems / FLOAT_COMPAT, num_elems % FLOAT_COMPAT, (const float*)input, (float*)output, alpha);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        int capacity_per_block = block_size * HALF_COMPAT;
        uint64_t gs = (num_elems + capacity_per_block - 1) / capacity_per_block;
        ppl_cukernel_unary_leakyrelu_opt_fp16<<<gs, block_size, 0, stream>>>(num_elems, num_elems / HALF_COMPAT, num_elems % HALF_COMPAT, (const half*)input, (half*)output, alpha);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        int capacity_per_block = block_size * INT8_COMPAT;
        uint64_t gs = (num_elems + capacity_per_block - 1) / capacity_per_block;
        ppl_cukernel_unary_leakyrelu_opt<<<gs, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, alpha, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}
#else //!__MACACC__

ppl::common::RetCode PPLCUDAUnaryLeakyReluForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    float alpha,
    float in_scale,
    float out_scale)
{
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int block_size     = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        ppl_cukernel_unary_leakyrelu<float><<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (float*)output, alpha);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        ppl_cukernel_unary_leakyrelu<half><<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, (half*)output, alpha);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        ppl_cukernel_unary_leakyrelu<<<grid_size, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, alpha, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

#endif //__MACACC__


