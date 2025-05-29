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
#include "cudakernel/arithmetic/bitwise.h"
#include <cuda_fp16.h>
template<typename T>
__global__ void ppl_cukernel_and(const T* input0,const T* input1,T* output,uint64_t num_elems)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    __shared__ T s_m[2][512];
    s_m[0][threadIdx.x] = input0[index]; 
    s_m[1][threadIdx.x] = input1[index];
    __syncthreads();
    T tmp1 = s_m[0][threadIdx.x];
    T tmp2 = s_m[1][threadIdx.x];
    T tmp = tmp1 & tmp2;
    output[index] = tmp;
}

template<typename T>
__global__ void ppl_cukernel_not(const T* input,T* output,uint64_t num_elems)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= num_elems) return;
    T tmp1 = input[index];
    T tmp = ~tmp1;
    output[index] = tmp;
}

template<typename T>
__global__ void ppl_cukernel_or(const T* input0,const T* input1,T* output,uint64_t num_elems)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    //may can be used shared memory
    T tmp1 = input0[index];
    T tmp2 = input1[index];
    T tmp = tmp1 | tmp2;
    output[index] = tmp;
}

template<typename T>
__global__ void ppl_cukernel_xor(const T* input0,const T* input1,T* output,uint64_t num_elems)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    __shared__ T s_m[2][512];
    s_m[0][threadIdx.x] = input0[index]; 
    s_m[1][threadIdx.x] = input1[index];
    __syncthreads();
    T tmp1 = s_m[0][threadIdx.x];
    T tmp2 = s_m[1][threadIdx.x];
    
    T tmp = tmp1 ^ tmp2;
    output[index] = tmp;
}

template<typename T>
__global__ void ppl_cukernel_shiftL(const T* input0,const T* input1,T* output,uint64_t num_elems)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= num_elems) return;
    __shared__ T s_m[2][512];
    s_m[0][threadIdx.x] = input0[index]; 
    s_m[1][threadIdx.x] = input1[index];
    __syncthreads();
    T tmp1 = s_m[0][threadIdx.x];
    T tmp2 = s_m[1][threadIdx.x];
    tmp1 = tmp1 << tmp2;
    output[index] = tmp1;
}

template<typename T>
__global__ void ppl_cukernel_shiftR(const T* input0,const T* input1,T* output,uint64_t num_elems)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= num_elems) return;
    __shared__ T s_m[2][512];
    s_m[0][threadIdx.x] = input0[index]; 
    s_m[1][threadIdx.x] = input1[index];
    __syncthreads();
    T tmp1 = s_m[0][threadIdx.x];
    T tmp2 = s_m[1][threadIdx.x];
    tmp1 = tmp1 >> tmp2;
    output[index] = tmp1;
}

ppl::common::RetCode PPLCUDABitWiseAndForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input0_shape,
    const void* input0,
    const ppl::common::TensorShape* input1_shape,
    const void* input1,
    const ppl::common::TensorShape* output_shape,
    void* output
)
{
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    
    auto datatype = output_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    uint64_t block_size     = 512;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;

    switch(datatype){
        case ppl::common::DATATYPE_INT64:{
            ppl_cukernel_and<long><<<grid_size, block_size, 0, stream>>>((const long*)input0,
                (const long*)input1,(long*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_INT32:{
            ppl_cukernel_and<int><<<grid_size, block_size, 0, stream>>>((const int*)input0,
                (const int*)input1,(int*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_INT16:{
            ppl_cukernel_and<short><<<grid_size, block_size, 0, stream>>>((const short*)input0,
                (const short*)input1,(short*)output,num_elems);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
    return status;
}

ppl::common::RetCode PPLCUDABitWiseNotForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output
)
{
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    
    auto datatype = output_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    uint64_t block_size     = 512;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;

    switch(datatype){
        case ppl::common::DATATYPE_INT64:{
            ppl_cukernel_not<long><<<grid_size, block_size, 0, stream>>>((const long*)input,
                (long*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_INT32:{
            ppl_cukernel_not<int><<<grid_size, block_size, 0, stream>>>((const int*)input,
                (int*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_INT16:{
            ppl_cukernel_not<short><<<grid_size, block_size, 0, stream>>>((const short*)input,
                (short*)output,num_elems);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
    return status;
}

ppl::common::RetCode PPLCUDABitWiseOrForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input0_shape,
    const void* input0,
    const ppl::common::TensorShape* input1_shape,
    const void* input1,
    const ppl::common::TensorShape* output_shape,
    void* output
)
{
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    
    auto datatype = output_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    uint64_t block_size     = 512;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;

    switch(datatype){
        case ppl::common::DATATYPE_INT64:{
            ppl_cukernel_or<long><<<grid_size, block_size, 0, stream>>>((const long*)input0,
                (const long*)input1,(long*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_INT32:{
            ppl_cukernel_or<int><<<grid_size, block_size, 0, stream>>>((const int*)input0,
                (const int*)input1,(int*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_INT16:{
            ppl_cukernel_or<short><<<grid_size, block_size, 0, stream>>>((const short*)input0,
                (const short*)input1,(short*)output,num_elems);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
    return status;
}

ppl::common::RetCode PPLCUDABitWiseXorForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input0_shape,
    const void* input0,
    const ppl::common::TensorShape* input1_shape,
    const void* input1,
    const ppl::common::TensorShape* output_shape,
    void* output
)
{
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    
    auto datatype = output_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    uint64_t block_size     = 512;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;

    switch(datatype){
        case ppl::common::DATATYPE_INT64:{
            ppl_cukernel_xor<long><<<grid_size, block_size, 0, stream>>>((const long*)input0,
                (const long*)input1,(long*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_INT32:{
            ppl_cukernel_xor<int><<<grid_size, block_size, 0, stream>>>((const int*)input0,
                (const int*)input1,(int*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_INT16:{
            ppl_cukernel_xor<short><<<grid_size, block_size, 0, stream>>>((const short*)input0,
                (const short*)input1,(short*)output,num_elems);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
    return status;
}

ppl::common::RetCode PPLCUDABitShiftForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input0_shape,
    const void* input0,
    const ppl::common::TensorShape* input1_shape,
    const void* input1,
    const ppl::common::TensorShape* output_shape,
    void* output,
    int type
)
{
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    auto datatype = output_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    uint64_t block_size     = 512;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;

    if(type == 0)
    {
        switch(datatype){
        case ppl::common::DATATYPE_UINT64:{
            ppl_cukernel_shiftL<uint64_t><<<grid_size, block_size, 0, stream>>>((const uint64_t*)input0,
                (const uint64_t*)input1,(uint64_t*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_UINT32:{
            ppl_cukernel_shiftL<uint32_t><<<grid_size, block_size, 0, stream>>>((const uint32_t*)input0,
                (const uint32_t*)input1,(uint32_t*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_UINT16:{
            ppl_cukernel_shiftL<uint16_t><<<grid_size, block_size, 0, stream>>>((const uint16_t*)input0,
                (const uint16_t*)input1,(uint16_t*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_UINT8:{
            ppl_cukernel_shiftL<uint8_t><<<grid_size, block_size, 0, stream>>>((const uint8_t*)input0,
                (const uint8_t*)input1,(uint8_t*)output,num_elems);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
        }
    } 
    else if(type == 1)
    {
        switch(datatype){
        case ppl::common::DATATYPE_UINT64:{
            ppl_cukernel_shiftR<uint64_t><<<grid_size, block_size, 0, stream>>>((const uint64_t*)input0,
                (const uint64_t*)input1,(uint64_t*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_UINT32:{
            ppl_cukernel_shiftR<uint32_t><<<grid_size, block_size, 0, stream>>>((const uint32_t*)input0,
                (const uint32_t*)input1,(uint32_t*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_UINT16:{
            ppl_cukernel_shiftR<uint16_t><<<grid_size, block_size, 0, stream>>>((const uint16_t*)input0,
                (const uint16_t*)input1,(uint16_t*)output,num_elems);
            break;
        }
        case ppl::common::DATATYPE_UINT8:{
            ppl_cukernel_shiftR<uint8_t><<<grid_size, block_size, 0, stream>>>((const uint8_t*)input0,
                (const uint8_t*)input1,(uint8_t*)output,num_elems);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
        }
    } else {
        status = ppl::common::RC_UNSUPPORTED;
    }
    
    return status;
}