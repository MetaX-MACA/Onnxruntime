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

#include "cudakernel/unary/selu.h"
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <typename DataT>
__device__ __inline__ DataT ppl_scalar_selu(const DataT& in_val, float alpha, float gamma);

template <>
__device__ __inline__ float ppl_scalar_selu<float>(const float& in_val, float alpha, float gamma)
{
    float res = in_val;
    if(res < 0)
        res = gamma *(alpha * exp(res) - alpha);
    else
        res = gamma * res;
    return res;
}

__device__ __inline__ int8_t ppl_scalar_selu_int8(const int8_t& in_val, float alpha, float gamma, float in_scale, float out_scale)
{
    int8_t res;
    float res_f = in_scale * in_val;
    if(res_f < 0)
        res_f = gamma *(alpha * exp(res_f) - alpha);
    else
        res_f = gamma * res_f;
    res = max(min(round(res_f / out_scale), 127.0), -127.0);
    return res;
}

template <>
__device__ __inline__ half ppl_scalar_selu<half>(const half& in_val, float alpha, float gamma)
{
    float res = __half2float(in_val);
    if(res < 0)
        res = gamma *(alpha * exp(res) - alpha);
    else
        res = gamma * res;
    return __float2half(res);
}
#endif

template <typename DataT>
__global__ void ppl_cukernel_unary_selu(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output,
    float alpha,
    float gamma)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    output[index] = ppl_scalar_selu<DataT>(in_val, alpha, gamma);
#endif
}

__global__ void ppl_cukernel_unary_selu(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    float alpha,
    float gamma,
    float in_scale,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int8_t in_val  = input[index];
    output[index] = ppl_scalar_selu_int8(in_val, alpha, gamma, in_scale, out_scale);
#endif
}
#include<stdio.h>
ppl::common::RetCode PPLCUDAUnarySeluForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    float alpha,
    float gamma,
    float in_scale,
    float out_scale)
{
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int block_size     = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        //printf("alpha: %f, beta: %f\n", alpha, beta);
        ppl_cukernel_unary_selu<float><<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (float*)output, alpha, gamma);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        ppl_cukernel_unary_selu<half><<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, (half*)output, alpha, gamma);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        ppl_cukernel_unary_selu<<<grid_size, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, alpha, gamma, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}
