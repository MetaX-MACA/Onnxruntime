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

#include "cudakernel/unary/isinf.h"
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <typename DataT>
__device__ __inline__ bool ppl_scalar_isinf(const DataT& in_val, int64_t detect_negative, int64_t detect_positive);

template <>
__device__ __inline__ bool ppl_scalar_isinf<float>(const float& in_val, int64_t detect_negative, int64_t detect_positive)
{
    uint32_t in_val_u32 = *(uint32_t*)(&in_val);
    if(((in_val_u32 << 1) >> 24) == 255 && (in_val_u32 << 9) == 0)
    {
        if(detect_negative > 0 && (in_val_u32 >> 31) > 0)
            return true;
        if(detect_positive > 0 && (in_val_u32 >> 31) == 0)
            return true;
        return false;
    }
    else
        return false;
}

template <>
__device__ __inline__ bool ppl_scalar_isinf<double>(const double& in_val, int64_t detect_negative, int64_t detect_positive)
{
    uint64_t in_val_u64 = *(uint64_t*)(&in_val);
    if(((in_val_u64 << 1) >> 53) == 4095 && (in_val_u64 << 12) == 0)
    {
        if(detect_negative > 0 && (in_val_u64 >> 63) > 0)
            return true;
        if(detect_positive > 0 && (in_val_u64 >> 63) == 0)
            return true;
        return false;
    }
    else
        return false;
}


template <>
__device__ __inline__ bool ppl_scalar_isinf<half>(const half& in_val, int64_t detect_negative, int64_t detect_positive)
{
    uint16_t in_val_u16 = *(uint16_t*)(&in_val);
    if(((in_val_u16 << 1) >> 11) == 31 && (in_val_u16 << 6) == 0)
    {
        if(detect_negative > 0 && (in_val_u16 >> 15) > 0)
            return true;
        if(detect_positive > 0 && (in_val_u16 >> 15) == 0)
            return true;
        return false;
    }
    else
        return false;
}
#endif

template <typename DataT>
__global__ void ppl_cukernel_unary_isinf(
    const uint64_t num_elems,
    const DataT* input,
    bool* output,
    int64_t detect_negative,
    int64_t detect_positive)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    output[index] = ppl_scalar_isinf<DataT>(in_val, detect_negative, detect_positive);
#endif
}


#include<stdio.h>
ppl::common::RetCode PPLCUDAUnaryIsinfForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    int64_t detect_negative,
    int64_t detect_positive)
{
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int block_size     = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        ppl_cukernel_unary_isinf<float><<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (bool*)output, detect_negative, detect_positive);
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        ppl_cukernel_unary_isinf<half><<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, (bool*)output, detect_negative, detect_positive);
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT64) {
        ppl_cukernel_unary_isinf<<<grid_size, block_size, 0, stream>>>(num_elems, (const double*)input, (bool*)output, detect_negative, detect_positive);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}
