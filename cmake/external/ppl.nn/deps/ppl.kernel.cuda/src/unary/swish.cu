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

#include "cudakernel/unary/swish.h"
#include "cudakernel/common/common.h"
#include "../reformat/cvt_int8_float.cuh"
#include <cuda_fp16.h>

template <typename T>
__global__ void SwishKernel(uint64_t size, const T* input, T* output) {
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){
    return;
  }
  T value = input[index];
  T div_value = 1.0 + expf(-input[index]);
  output[index] = value / div_value ;
}

template <typename DataT>
__global__ void SwishQuant_int8(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output,
    float inScale,
    float outScale)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    float in_val_f = _int82float(in_val, inScale, 0.0);
    float out_val_f = in_val_f / (1.0 + expf(-in_val_f));
    output[index] = _float2int8(out_val_f, outScale, 0.0);
}

#ifdef __MACACC__
__global__ void SwishQuant_int8_opt_16(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    float inScale,
    float outScale)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index *= 16;
    if (index >= num_elems) return;
    const uint64_t *ip = (uint64_t*)(input+index);
    uint64_t in_val0 = ip[0], in_val1 = ip[1];

    uint64_t out0 = 0;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        int8_t in_val_int8 = (int8_t)(in_val0&0xFF);
        float in_val_f = _int82float(in_val_int8, inScale, 0.0);
        float out_val_f = __builtin_mxc_rcpf(__builtin_expf(-in_val_f) + 1.0) * in_val_f;
        int8_t out_val_int8 = _float2int8(out_val_f, outScale, 0.0);
        out0 |= (((uint64_t)(out_val_int8) & 0xFF) << i*8);
        in_val0 >>= 8;
    }

    uint64_t out1 = 0;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        int8_t in_val_int8 = (int8_t)(in_val1&0xFF);
        float in_val_f = _int82float(in_val_int8, inScale, 0.0);
        float out_val_f = __builtin_mxc_rcpf(__builtin_expf(-in_val_f) + 1.0) * in_val_f;
        int8_t out_val_int8 = _float2int8(out_val_f, outScale, 0.0);
        out1 |= (((uint64_t)(out_val_int8) & 0xFF) << i*8);
	    in_val1 >>= 8;
    }

    uint64_t *op = (uint64_t*)(output+index);
    op[0] = out0;
    op[1] = out1;
}

__global__ void SwishQuant_int8_opt_8(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    float inScale,
    float outScale)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index *= 8;
    if (index >= num_elems) return;
    const uint64_t *ip = (uint64_t*)(input+index);
    uint64_t in_val0 = ip[0];

    uint64_t out0 = 0;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        int8_t in_val_int8 = (int8_t)(in_val0&0xFF);
        float in_val_f = _int82float(in_val_int8, inScale, 0.0);
        float out_val_f = __builtin_mxc_rcpf(__builtin_expf(-in_val_f) + 1.0) * in_val_f;
        int8_t out_val_int8 = _float2int8(out_val_f, outScale, 0.0);
        out0 |= ((uint64_t)(out_val_int8) & 0xFF) << i*8;;
        in_val0 >>= 8;
    }


    uint64_t *op = (uint64_t*)(output+index);
    op[0] = out0;
}


__global__ void SwishQuant_int8_opt_1(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    float inScale,
    float outScale)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int8_t in_val  = input[index];
    float in_val_f = _int82float(in_val, inScale, 0.0);
    float out_val_f = __builtin_mxc_rcpf(__builtin_expf(-in_val_f)+1.0) * in_val_f;//in_val_f / (1.0 + expf(-in_val_f));
    output[index] = _float2int8(out_val_f, outScale, 0.0);
}
#endif


__global__ void SwishKernel_opt_fp16(const float4 * input, float4* output, uint64_t size) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= size){    return;  }

    float4 in_val = *((float4*)input + index);
    half *ptr_data = (half*)&in_val;
    float4 reg;
    half* ptr_reg = (half*)&reg;
    for (int i = 0; i < 8; i++) {
        ptr_reg[i]= ptr_data[i]/__float2half(1.0 + expf(-ptr_data[i]));
    }
    float4 *out_val = (float4*)output + index;
    *out_val = reg;
}


ppl::common::RetCode PPLCUDAUnarySwishForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    float beta,
    float in_scale,
    float out_scale)
{
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    uint64_t block_size     = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;

    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        SwishKernel<float><<<grid_size, block_size, 0, stream>>>(num_elems,(const float *)input,(float *)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        if (num_elems % 8 == 0)
        {
            uint64_t threads = 256;
            uint64_t num = num_elems/8;
            uint64_t blocks = (num + threads - 1) / threads;
            SwishKernel_opt_fp16<<<blocks, threads, 0, stream>>>((const float4 *)input, (float4*)output, num);
        }
        else
        {
            SwishKernel<half><<<grid_size, block_size, 0, stream>>>(num_elems,(const half *)input,(half*)output);
        }

    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8){
#ifndef __MACACC__
        SwishQuant_int8<char><<<grid_size, block_size, 0, stream>>>(num_elems,(const char *)input,(char *)output,in_scale,out_scale);
#else
        if (num_elems % 16 == 0 && grid_size/16 >= 128)
            SwishQuant_int8_opt_16<<<(grid_size+16-1)/16, block_size, 0, stream>>>(num_elems, (const int8_t *)input,(int8_t *)output,in_scale,out_scale);
        else if (num_elems % 8 == 0 && grid_size/8 >= 128)
            SwishQuant_int8_opt_8<<<(grid_size+8-1)/8, block_size, 0, stream>>>(num_elems, (const int8_t *)input,(int8_t *)output,in_scale,out_scale);
        else
            SwishQuant_int8_opt_1<<<grid_size, block_size, 0, stream>>>(num_elems, (const int8_t *)input,(int8_t *)output,in_scale,out_scale);
#endif
        /*cudaDeviceSynchronize();
        int8_t *out_128 = (int8_t*)malloc(128*sizeof(int8_t));
        cudaMemcpy(out_128, output, 128*sizeof(int8_t), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 128; i++) {
            printf("out index = %d, value = %d\n", i, (int)out_128[i]);
        }*/
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}
