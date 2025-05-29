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

#include "cudakernel/nn/quickgelu.h"
#include "../reformat/cvt_int8_float.cuh"
#include "cudakernel/common/common.h"
#include <cuda_fp16.h>

#ifdef __MACACC__
#include "cudakernel/common/divmod_fast.h"
#define __USEOPT__
#endif

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <typename T>
__device__ __inline__ T ppl_scalar_quickgelu(const T& in_val, const float alpha);

template <>
__device__ __inline__ float ppl_scalar_quickgelu<float>(const float& in_val, const float alpha)
{
    return __builtin_mxc_rcpf(1.f + __builtin_expf(-in_val * alpha)) * in_val;
}

template <>
__device__ __inline__ half ppl_scalar_quickgelu<half>(const half& in_val, const float alpha)
{
    float in_valf = __half2float(in_val);
    float resf = __builtin_mxc_rcpf(1.f + __builtin_expf(-in_valf * alpha)) * in_valf;
    return __float2half(resf);
}
#endif

template <typename T>
__global__ void ppl_cukernel_quickgelu(
    const uint64_t num_elems,
    const T* input,
    T* output,
    const float alpha)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    T in_val = *(input + index);
    output[index] = ppl_scalar_quickgelu<T>(in_val, alpha);
#endif
}

__global__ void ppl_cukernel_quickgelu_int8(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    const float alpha,
    QuantKernelParamCuda qparam)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int8_t in_val = *(input + index);
    float in_valf = _int82float(in_val, qparam.i_step, qparam.i_zero_point);
    float resf = __builtin_mxc_rcpf(1.f + __builtin_expf(-in_valf * alpha)) * in_valf;
    output[index] = _float2int8(resf, qparam.o_step, qparam.o_zero_point);
#endif
}

template <typename T, typename PackT, int Times>
__global__ void ppl_cukernel_quickgelu_opt(
    const uint64_t num_elems,
    const T* input,
    T* output,
    const float alpha)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    PackT in_val = *((PackT*)input + index);
    T *ptr_data = (T*)&in_val;
    PackT reg;
    T* ptr_reg = (T*)&reg;
    for (int i = 0; i < Times; i++) {
        ptr_reg[i]=ppl_scalar_quickgelu<T>(ptr_data[i], alpha);
    }
    PackT *out_val = (PackT*)output + index;
    *out_val = reg;
#endif
}

template <typename T, typename PackT, int Times>
__global__ void ppl_cukernel_quickgelu_int8_opt(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    const float alpha,
    QuantKernelParamCuda qparam)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    PackT in_val = *((PackT*)input + index);
    T *ptr_data = (T*)&in_val;
    PackT reg;
    T* ptr_reg = (T*)&reg;
    for (int i = 0; i < Times; i++) {
        float in_valf = _int82float(ptr_data[i], qparam.i_step, qparam.i_zero_point);
        float resf = __builtin_mxc_rcpf(1.f + __builtin_expf(-in_valf * alpha)) * in_valf;
        ptr_reg[i] = _float2int8(resf, qparam.o_step, qparam.o_zero_point);
    }
    PackT *out_val = (PackT*)output + index;
    *out_val = reg;
#endif
}

ppl::common::RetCode PPLCUDAQuickGeluForwardImp(
    cudaStream_t stream, 
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const float alpha,
    const QuantKernelParamCuda* qparam)
{
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        int block_size = 256;
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            int capacity_per_block = block_size * 4;
            int grid_size = (num_elems + capacity_per_block - 1) / capacity_per_block;
            ppl_cukernel_quickgelu_opt<float, float4, 4><<<grid_size, block_size, 0, stream>>>
                ((num_elems + 3) / 4, (const float *)input, (float *)output, alpha);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            int capacity_per_block = block_size * 8;
            int grid_size = (num_elems + capacity_per_block - 1) / capacity_per_block;
            ppl_cukernel_quickgelu_opt<half, float4, 8><<<grid_size, block_size, 0, stream>>>
                ((num_elems + 7) / 8, (const half *)input, (half *)output, alpha);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
            int capacity_per_block = block_size * 16;
            int grid_size = (num_elems + capacity_per_block - 1) / capacity_per_block;
            ppl_cukernel_quickgelu_int8_opt<int8_t, float4, 16><<<grid_size, block_size, 0, stream>>>
                ((num_elems + 15) / 16, (const int8_t *)input, (int8_t *)output, alpha, *qparam);
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }

    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
        int block_size = 512;
        int grid_size = (num_elems + block_size - 1) / block_size;
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_quickgelu<float><<<grid_size, block_size, 0, stream>>>
                (num_elems, (const float *)input, (float *)output, alpha);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_quickgelu<half><<<grid_size, block_size, 0, stream>>>
                (num_elems, (const half *)input, (half *)output, alpha);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_quickgelu_int8<<<grid_size, block_size, 0, stream>>>
                (num_elems, (const int8_t *)input, (int8_t *)output, alpha, *qparam);
        } else {
            return ppl::common::RC_UNSUPPORTED;
        } 
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS; 
}