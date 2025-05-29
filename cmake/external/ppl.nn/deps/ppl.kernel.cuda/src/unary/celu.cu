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

#include "cudakernel/unary/celu.h"
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <typename DataT>
__device__ __inline__ DataT ppl_scalar_celu(const DataT& in_val, float alpha);

template <>
__device__ __inline__ float ppl_scalar_celu<float>(const float& in_val, float alpha)
{
    float res;
    res = max(in_val, 0.0f) + min(0.0f, alpha * (exp(in_val / alpha) - 1));
    return res;
}

__device__ __inline__ int8_t ppl_scalar_celu_int8(const int8_t& in_val, float alpha, float in_scale, float out_scale)
{
    int8_t res;
    float res_f = in_scale * in_val;
    res_f = max(res_f, 0.0f) + min(0.0f, alpha * (exp(res_f / alpha) - 1));
    res = max(min(round(res_f / out_scale), 127.0), -127.0);
    return res;
}

template <>
__device__ __inline__ half ppl_scalar_celu<half>(const half& in_val, float alpha)
{
    float res = __half2float(in_val);
    res = max(res, 0.0f) + min(0.0f, alpha * (exp(res / alpha) - 1));
    return __float2half(res);
}
#endif

template <typename DataT>
__global__ void ppl_cukernel_unary_celu(
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
    output[index] = ppl_scalar_celu<DataT>(in_val, alpha);
#endif
}

__global__ void ppl_cukernel_unary_celu(
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
    output[index] = ppl_scalar_celu_int8(in_val, alpha, in_scale, out_scale);
#endif
}
#include<stdio.h>
ppl::common::RetCode PPLCUDAUnaryCeluForwardImp(
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
        //printf("alpha: %f, beta: %f\n", alpha, beta);
        ppl_cukernel_unary_celu<float><<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (float*)output, alpha);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        ppl_cukernel_unary_celu<half><<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, (half*)output, alpha);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        ppl_cukernel_unary_celu<<<grid_size, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, alpha, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}
