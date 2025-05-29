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

#include "cudakernel/reformat/reformat.h"
#include <cuda_fp16.h>
#ifdef __MACACC__
#define OPT_CVT_TENSOR
#endif//__MACACC__
#define JUDGE(elems) uint64_t id = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x; \
                    if (id >= elems) return;
static __device__ inline signed char _float2int8(
    float data_in,
    float step,
    signed char zeroPoint)
{
    float tmp = (float)data_in / step + zeroPoint;

    return tmp > 127 ? 127 : tmp < -128 ? -128
                                        : (signed char)(__float2int_rn(tmp)); //saturate
}

static __device__ inline float _int82float(
    signed char data_in,
    float step,
    signed char zeroPoint)
{
    float tmp = (float)(data_in - zeroPoint) * step;

    return tmp;
}

static __device__ inline float _uint82float(
    unsigned char data_in,
    float step,
    unsigned char zeroPoint)
{
    float tmp = ((float)data_in - (float)zeroPoint) * step;
    return tmp;
}

static __device__ inline signed char _float2int4B(
    float data_in,
    float step,
    signed char zeroPoint)
{
    float tmp = (float)data_in / step + zeroPoint;

    return tmp > 7 ? 7 : tmp < -8 ? -8
                                  : (signed char)(__float2int_rn(tmp)); //saturate
}

static __device__ inline float _int4B2float(
    signed char data_in,
    float step,
    signed char zeroPoint)
{
    float tmp = (float)(data_in - zeroPoint) * step;
    return tmp;
}
#ifdef OPT_CVT_TENSOR
__global__ void cuda_kernel_cvt_opt_int8_float(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    if(blockIdx.x == gridDim.x - 1){
        uint64_t id = (blockDim.x * blockIdx.x << 2) + threadIdx.x;
        for(uint64_t i = id; i < num_elems; i += blockDim.x)
        {
            int8_t tmp = ((int8_t*)input)[i];
            ((float*)output)[i] = _int82float(tmp, param.i_step, param.i_zero_point);
        }
    }else{
        uint64_t id = (blockDim.x * blockIdx.x + threadIdx.x);
        int32_t data = *((int32_t*)input + id);
        int8_t *ptr_data = (int8_t*)&data;
        float4 reg;
        float* ptr_reg = (float*)&reg;
        #pragma unroll 4
        for(int i = 0; i < 4; i++)
        {
            ptr_reg[i] = _int82float(ptr_data[i],param.i_step, param.i_zero_point);
        }
        float4 *ptr_out = (float4*)output + id;
        *ptr_out = reg;
    }
}

__global__ void cuda_kernel_cvt_opt_int8_half(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    if(blockIdx.x == gridDim.x - 1){
        uint64_t id = (blockDim.x * blockIdx.x << 3) + threadIdx.x;
        for(uint64_t i = id; i < num_elems; i += blockDim.x)
        {
            int8_t tmp = ((int8_t*)input)[i];
            ((half*)output)[i] = __float2half(_int82float(tmp, param.i_step, param.i_zero_point));
        }
    } else {
        uint64_t id = (blockDim.x * blockIdx.x + threadIdx.x)<<3;
        float2 data = *(float2*)((int8_t*)input + id);
        int8_t *ptr_data = (int8_t*)&data;
        float4 reg;
        half* ptr_reg = (half*)&reg;
        #pragma unroll 8
        for(int i = 0; i < 8; i++)
        {
            ptr_reg[i] = __float2half(_int82float(ptr_data[i],param.i_step, param.i_zero_point));
        }
        float4 *ptr_out = (float4*)((half*)output + id);
        *ptr_out = reg;
    }
}

__global__ void cuda_kernel_cvt_opt_half_float(size_t num_elems, const half* input, ReFormatParam param, float* output)
{
    if(blockIdx.x == gridDim.x - 1)
    {
        const half*ptr_input = input + (blockIdx.x << 2)*blockDim.x;
        int64_t offset = (blockIdx.x<<2)*blockDim.x;
        float* ptr_output = output + offset;
        int64_t blockSize = min(num_elems - offset, (blockDim.x << 2));
        for(int64_t i = threadIdx.x; i < blockSize; i += blockDim.x)
        {
            half tmp = ptr_input[i];
            ptr_output[i] = __half2float(tmp);
        }
    }else{
        uint64_t id = (blockDim.x * blockIdx.x + threadIdx.x);
        float2 data = *((float2*)input + id);
        half *ptr_data = (half*)&data;
        float4 reg;
        float* ptr_reg = (float*)&reg;
        #pragma unroll 4
        for(int i = 0; i < 4; i++)
        {
            ptr_reg[i] = __half2float(ptr_data[i]);
        }
        float4 *ptr_out = (float4*)output + id;
        *ptr_out = reg;
    }
}

__global__ void cuda_kernel_cvt_opt_float_half(size_t num_elems, const float* input, ReFormatParam param, half* output)
{
    if(blockIdx.x == gridDim.x - 1)
    {
        int64_t offset = (blockIdx.x << 2)*blockDim.x;
        const float*ptr_input = input + offset;
        half* ptr_output = output + offset;
        int64_t blockSize = min(num_elems - offset, (blockDim.x << 2));
        for(int64_t i = threadIdx.x; i < blockSize; i += blockDim.x)
        {
            float tmp = ptr_input[i];
            ptr_output[i] = __float2half(tmp);
        }
    }else{
        uint64_t id = (blockDim.x * blockIdx.x + threadIdx.x);
        float4 data = *((float4*)input + id);
        float *ptr_data = (float*)&data;
        float2 reg;
        half* ptr_reg = (half*)&reg;
        #pragma unroll 4
        for(int i = 0; i < 4; i++)
        {
            ptr_reg[i] = __float2half(ptr_data[i]);
        }
        float2 *ptr_out = (float2*)output + id;
        *ptr_out = reg;
    }
}

__global__ void cuda_kernel_cvt_packed_f32_8_opt(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    int64_t id = blockIdx.x * blockDim.x << 4;
    float scale = 1.0 / param.o_step;
    float zp = param.o_zero_point;
    if(blockIdx.x == gridDim.x - 1) {
        int blockLength = min(num_elems - id, blockDim.x << 4);
        float *ptr_input = (float*)input + id;
        int8_t * ptr_output = (int8_t*)output + id;
        for(int i = threadIdx.x; i < blockLength; i += blockDim.x)
        {
            int reg_i;
            float reg = ptr_input[i];
            reg = reg * scale + zp;
            reg_i = reg > 127 ? 127 : reg < -128 ? -128
                                         : (__float2int_rn(reg));
            *(ptr_output + i) = reg_i;
        }
    } else {
        id = id + (threadIdx.x << 4);
        float * ptr_input = (float*)input + id;
        float reg_input[16];
        float4 reg_dst;
        *(float4*)(reg_input) = *(float4*)(ptr_input);
        *(float4*)(reg_input + 4) = *(float4*)(ptr_input + 4);
        *(float4*)(reg_input + 8) = *(float4*)(ptr_input + 8);
        *(float4*)(reg_input + 12) = *(float4*)(ptr_input + 12);
        int8_t * ptr_reg_dst = (int8_t*)&reg_dst;
        #pragma unroll 16
        for(int i = 0; i < 16; i++) {
            float reg = reg_input[i];
            int reg_i;
            reg = reg * scale + zp;
            reg_i = reg > 127 ? 127 : reg < -128 ? -128
                                         : (__float2int_rn(reg));
            ptr_reg_dst[i] = reg_i;
        }
        int8_t * ptr_output = (int8_t*)output;
        *(float4*)(ptr_output + id) = reg_dst;
    }
}

#endif//OPT_CVT_TENSOR
template <CVTTypeMode mode>
__global__ void cuda_kernel_cvt(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
}

template <>
__global__ void cuda_kernel_cvt<INT8_FLOAT16>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((half*)output)[id] = __float2half(_int82float(((int8_t*)input)[id], param.i_step, param.i_zero_point));
}
template <>
__global__ void cuda_kernel_cvt<FLOAT16_INT8>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((int8_t*)output)[id] = _float2int8((float)*((__half*)input + id), param.o_step, param.o_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<INT8_FLOAT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = _int82float(((int8_t*)input)[id], param.i_step, param.i_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<UINT8_FLOAT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = _uint82float(((uint8_t*)input)[id], param.i_step, param.i_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<UINT8_FLOAT16>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((half*)output)[id] = ((uint8_t*)input)[id];
}


template <>
__global__ void cuda_kernel_cvt<INT8_INT8>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    float tmp             = _int82float(((int8_t*)input)[id], param.i_step, param.i_zero_point);
    ((int8_t*)output)[id] = _float2int8(tmp, param.o_step, param.o_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<FLOAT32_INT8>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    *((int8_t*)output + id) = _float2int8(*((float*)input + id), param.o_step, param.o_zero_point);
}
template <>
__global__ void cuda_kernel_cvt<FLOAT32_INT4B>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    *((int8_t*)output + id) = _float2int4B(*((float*)input + id), param.o_step, param.o_zero_point);
}
template <>
__global__ void cuda_kernel_cvt<INT4B_FLOAT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = _int4B2float(((int8_t*)input)[id], param.i_step, param.i_zero_point);
}
template <>
__global__ void cuda_kernel_cvt<INT4B_INT4B>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    float tmp             = _int4B2float(((int8_t*)input)[id], param.i_step, param.i_zero_point);
    ((int8_t*)output)[id] = _float2int4B(tmp, param.o_step, param.o_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<FLOAT16_FLOAT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = __half2float(((half*)input)[id]);
}

template <>
__global__ void cuda_kernel_cvt<FLOAT32_FLOAT16>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((half*)output)[id] = __float2half(((float*)input)[id]);
}

template <>
__global__ void cuda_kernel_cvt<FLOAT32_FLOAT64>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((double*)output)[id] = ((float*)input)[id];
}

template <>
__global__ void cuda_kernel_cvt<FLOAT64_FLOAT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = ((double*)input)[id];
}

template <>
__global__ void cuda_kernel_cvt<INT8_INT4B>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    signed char tmp     = ((signed char*)input)[id];
    ((char*)output)[id] = tmp > 7 ? 7 : tmp < -8 ? -8 : tmp;
}

template <>
__global__ void cuda_kernel_cvt<INT32_INT64>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((int64_t*)output)[id] = ((int32_t*)input)[id];
}

template <>
__global__ void cuda_kernel_cvt<INT64_INT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((int32_t*)output)[id] = ((int64_t*)input)[id];
}


template <>
__global__ void cuda_kernel_cvt<INT64_FLOAT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = ((int64_t*)input)[id];
}

template <>
__global__ void cuda_kernel_cvt<INT64_BOOL>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((bool*)output)[id] = ((int64_t*)input)[id] > 0 ? true : false;
}

template <>
__global__ void cuda_kernel_cvt<BOOL_INT64>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((int64_t*)output)[id] = ((bool*)input)[id] == true ? 1 : 0;
}

template <>
__global__ void cuda_kernel_cvt<FLOAT32_INT64>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((int64_t*)output)[id] = ((float*)input)[id];
}

static __device__ inline char4 _float42int8(
    float4 data_in,
    float step,
    signed char zeroPoint)
{
    float4 tmp;
    tmp.x = data_in.x / step + zeroPoint;
    tmp.y = data_in.y / step + zeroPoint;
    tmp.z = data_in.z / step + zeroPoint;
    tmp.w = data_in.w / step + zeroPoint;
    char4 dst;
    dst.x = tmp.x > 127 ? 127 : tmp.x < -128 ? -128 : (signed char)(__float2int_rn(tmp.x));
    dst.y = tmp.y > 127 ? 127 : tmp.y < -128 ? -128 : (signed char)(__float2int_rn(tmp.y));
    dst.z = tmp.z > 127 ? 127 : tmp.z < -128 ? -128 : (signed char)(__float2int_rn(tmp.z));
    dst.w = tmp.w > 127 ? 127 : tmp.w < -128 ? -128 : (signed char)(__float2int_rn(tmp.w));
    return dst;
}

template <CVTTypeMode mode>
__global__ void cuda_kernel_cvt_packed(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
}
template <>
__global__ void cuda_kernel_cvt_packed<FLOAT32_INT8>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    *((char4*)output + id) = _float42int8(*((float4*)input + id), param.o_step, param.o_zero_point);
}

void PPLCUDACVTTypePerTensor(
    cudaStream_t stream,
    const void* input,
    void* output,
    ReFormatParam param)
{
    int block_size   = 512;
    uint64_t num_elems = param.n_outer * param.src_pad * param.n_inner;

    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    switch (GetCVTTypeMode(param)) {
        case FLOAT32_INT8:
#ifdef OPT_CVT_TENSOR
            {
                int blockSize = 512;
                uint64_t gridSize = (num_elems + 8191)>>13;
                cuda_kernel_cvt_packed_f32_8_opt<<<gridSize, blockSize,0,stream>>>(num_elems, input, param, output);
            }
#else//!OPT_CVT_TENSOR
            if(num_elems % 4 == 0) {
                block_size = 128;
                num_elems = num_elems >> 2;
                grid_size = (num_elems + block_size - 1) / block_size;
                cuda_kernel_cvt_packed<FLOAT32_INT8><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            } else{
                cuda_kernel_cvt<FLOAT32_INT8><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            }
#endif//OPT_CVT_TENSOR
            break;
        case INT8_FLOAT32:
#ifdef OPT_CVT_TENSOR
            grid_size = (num_elems + (block_size<<2) - 1) / (block_size<<2);
            cuda_kernel_cvt_opt_int8_float<<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
#else//!OPT_CVT_TENSOR
            cuda_kernel_cvt<INT8_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
#endif//OPT_CVT_TENSOR
            break;
        case UINT8_FLOAT32:
            cuda_kernel_cvt<UINT8_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case UINT8_FLOAT16:
            cuda_kernel_cvt<UINT8_FLOAT16><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case FLOAT32_FLOAT16:
#ifdef OPT_CVT_TENSOR
            grid_size = (num_elems + (block_size<<2) - 1) / (block_size<<2);
            //printf("################## cuda_kernel_cvt_opt_float_half input ptr(%p), output ptr(%p)\n", input, output);
            cuda_kernel_cvt_opt_float_half<<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, param, (half*)output);
#else//!OPT_CVT_TENSOR
            cuda_kernel_cvt<FLOAT32_FLOAT16><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
#endif//OPT_CVT_TENSOR
            break;
        case FLOAT16_FLOAT32:
#ifdef OPT_CVT_TENSOR
            grid_size = (num_elems + (block_size<<2) - 1) / (block_size<<2);
            cuda_kernel_cvt_opt_half_float<<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, param, (float*)output);
#else//!OPT_CVT_TENSOR
            cuda_kernel_cvt<FLOAT16_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
#endif//OPT_CVT_TENSOR
            break;
        case FLOAT32_INT4B:
            cuda_kernel_cvt<FLOAT32_INT4B><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT4B_FLOAT32:
            cuda_kernel_cvt<INT4B_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT8_FLOAT16:
#ifdef OPT_CVT_TENSOR
            grid_size = (num_elems + (block_size << 3) - 1) / (block_size << 3);
            cuda_kernel_cvt_opt_int8_half<<<grid_size, block_size, 0, stream>>>(num_elems, input, param,output);
#else //!OPT_CVT_TENSOR
            cuda_kernel_cvt<INT8_FLOAT16><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
#endif//OPT_CVT_TENSOR
            break;
        case FLOAT16_INT8:
            cuda_kernel_cvt<FLOAT16_INT8><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT8_INT4B:
            cuda_kernel_cvt<INT8_INT4B><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT8_INT8:
            cuda_kernel_cvt<INT8_INT8><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT4B_INT4B:
            cuda_kernel_cvt<INT4B_INT4B><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT32_INT64:
            cuda_kernel_cvt<INT32_INT64><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT64_INT32:
            cuda_kernel_cvt<INT64_INT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT64_FLOAT32:
            cuda_kernel_cvt<INT64_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case FLOAT32_INT64:
            cuda_kernel_cvt<FLOAT32_INT64><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case FLOAT32_FLOAT64:
            cuda_kernel_cvt<FLOAT32_FLOAT64><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case FLOAT64_FLOAT32:
            cuda_kernel_cvt<FLOAT64_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT64_BOOL:
            cuda_kernel_cvt<INT64_BOOL><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case BOOL_INT64:
            cuda_kernel_cvt<BOOL_INT64><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        default:
            LOG(ERROR) << "Unsupport  PPLCUDACVTTypePerTensor for data type converter: " << (int)GetCVTTypeMode(param);
            break;
    }
}
