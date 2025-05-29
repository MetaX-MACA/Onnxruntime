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

#include "cudakernel/unary/unary.h"
#include "../reformat/cvt_int8_float.cuh"
#include "cudakernel/common/common.h"
#include <cuda_fp16.h>

#ifdef __MACACC__
#define HardSwish_OPT 0.1666666667
#endif

enum UnaryOpType {
    Unary_Unknown = 0,
    Unary_Abs,
    Unary_Relu,
    Unary_Sigmoid,
    Unary_Sqrt,
    Unary_Square,
    Unary_TanH,
    Unary_Floor,
    Unary_Ceil,
    Unary_OpNum,
    Unary_Erf,
    Unary_Sin,
    Unary_Cos,
    Unary_Round,
    Unary_Sign,
    Unary_Sinh,
    Unary_Cosh,
    Unary_Asin,
    Unary_Acos,
    Unary_Atan,
    Unary_Asinh,
    Unary_Acosh,
    Unary_Atanh,
    Unary_HardSwish,
    Unary_HardMish,
    Unary_Mish,
    Unary_Softsign,
    Unary_ForceWord = INT_MAX,
};

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <UnaryOpType OpT, typename DataT>
__device__ __inline__ DataT ppl_scalar_unary(const DataT& in_val);

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Abs, float>(const float& in_val)
{
    return fabsf(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Abs, half>(const half& in_val)
{
    return __float2half(fabsf(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Abs, int8_t>(const int8_t& in_val)
{
    return in_val >= 0 ? in_val : -in_val; 
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Relu, float>(const float& in_val)
{
    float res;
    res = (in_val > 0) ? in_val : 0;
    return res;
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Relu, half>(const half& in_val)
{
    half res;
    res = __hgt(in_val, 0) ? in_val : half(0);
    return res;
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Relu, int8_t>(const int8_t& in_val)
{
    int8_t res;
    res = (in_val > 0) ? in_val : 0;
    return res;
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Sigmoid, float>(const float& in_val)
{
    return 1.f / (1.f + expf(-in_val));
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Sigmoid, half>(const half& in_val)
{
    float in_valf = __half2float(in_val);
    // float resf    = 1.f / (1.f + expf(-in_valf));
    float resf = __builtin_mxc_rcpf(1.f + __builtin_expf(-in_valf));
    return __float2half(resf);
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Sigmoid, int8_t>(const int8_t& in_val)
{
    return 1 / (1 + int8_t(expf(float(-in_val))));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Sqrt, float>(const float& in_val)
{
    return sqrt(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Sqrt, half>(const half& in_val)
{
    return __float2half(sqrt(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Sqrt, int8_t>(const int8_t& in_val)
{
    return int8_t(sqrt(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Square, float>(const float& in_val)
{
    return in_val * in_val;
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Square, half>(const half& in_val)
{
    return in_val * in_val;
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Square, int8_t>(const int8_t& in_val)
{
    return in_val * in_val;
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_TanH, float>(const float& in_val)
{
// out of float range when large in_val
// #ifdef __MACACC__
//     float e = expf(2 * in_val);
//     return (e - 1) * __builtin_mxc_rcpf(e + 1);
// #endif//__MACACC__
    return tanh(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_TanH, half>(const half& in_val)
{
    #ifdef __MACACC__
        // float in_valf = __half2float(in_val);
        // float e = expf(2 * in_valf);
        // float resf = (e - 1) * __builtin_mxc_rcpf(e + 1);
        // return __float2half(resf);
        return __float2half(tanhf(__half2float(in_val)));
    #else//!__MACACC__
        return __float2half(tanh(__half2float(in_val)));
    #endif//__MACACC__
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_TanH, int8_t>(const int8_t& in_val)
{
    return int8_t(tanh(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Floor, float>(const float& in_val)
{
    return floor(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Floor, half>(const half& in_val)
{
    return hfloor(in_val);
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Floor, int8_t>(const int8_t& in_val)
{
    return int8_t(floor(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Ceil, float>(const float& in_val)
{
    return ceil(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Ceil, half>(const half& in_val)
{
    return hceil(in_val);
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Ceil, int8_t>(const int8_t& in_val)
{
    return int8_t(ceil(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Erf, float>(const float& in_val)
{
    return erf(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Erf, half>(const half& in_val)
{
    return __float2half(erf(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Erf, int8_t>(const int8_t& in_val)
{
    return int8_t(erf(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Sin, float>(const float& in_val)
{
    return sin(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Sin, half>(const half& in_val)
{
    return __float2half(sin(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Sin, int8_t>(const int8_t& in_val)
{
    return int8_t(sin(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Cos, float>(const float& in_val)
{
    return cos(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Cos, half>(const half& in_val)
{
    return __float2half(cos(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Cos, int8_t>(const int8_t& in_val)
{
    return int8_t(cos(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Sinh,float>(const float& in_val)
{
    return sinh(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Sinh,half>(const half& in_val)
{
    return __float2half(sinh(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Sinh,int8_t>(const int8_t& in_val)
{
    return int8_t(sinh(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Cosh,float>(const float& in_val)
{
    return cosh(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Cosh,half>(const half& in_val)
{
    return __float2half(cosh(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Cosh,int8_t>(const int8_t& in_val)
{
    return int8_t(cosh(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Asin,float>(const float& in_val)
{
    return asin(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Asin,half>(const half& in_val)
{
    return __float2half(asin(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Asin,int8_t>(const int8_t& in_val)
{
    return int8_t(asin(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Acos,float>(const float& in_val)
{
    return acos(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Acos,half>(const half& in_val)
{
    return __float2half(acos(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Acos,int8_t>(const int8_t& in_val)
{
    return int8_t(acos(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Atan,float>(const float& in_val)
{
    return atan(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Atan,half>(const half& in_val)
{
    return __float2half(atan(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Atan,int8_t>(const int8_t& in_val)
{
    return int8_t(atan(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Asinh,float>(const float& in_val)
{
    return asinh(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Asinh,half>(const half& in_val)
{
    return __float2half(asinh(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Asinh,int8_t>(const int8_t& in_val)
{
    return int8_t(asinh(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Acosh,float>(const float& in_val)
{
    return acosh(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Acosh,half>(const half& in_val)
{
    return __float2half(acosh(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Acosh,int8_t>(const int8_t& in_val)
{
    return int8_t(acosh(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Atanh,float>(const float& in_val)
{
    return atanh(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Atanh,half>(const half& in_val)
{
    return __float2half(atanh(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Atanh,int8_t>(const int8_t& in_val)
{
    return int8_t(atanh(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Round, float>(const float& in_val)
{
    return rint(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Round, half>(const half& in_val)
{
    return __float2half(rint(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Round, int8_t>(const int8_t& in_val)
{
    return int8_t(rint(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Sign, float>(const float& in_val)
{
    return in_val > 0 ? 1 : (in_val == 0 ? 0 : -1);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Sign, half>(const half& in_val)
{
    float temp_in_val = __half2float(in_val);
    return __float2half(temp_in_val > 0 ? 1 : (temp_in_val == 0 ? 0 : -1));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Sign, int8_t>(const int8_t& in_val)
{
    return in_val > 0 ? 1 : (in_val == 0 ? 0 : -1);
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_HardSwish, float>(const float& in_val)
{  
#ifdef __MACACC__
    float res = in_val * max(0.0f, min(1.0f, float(HardSwish_OPT) * in_val + 0.5f));
#else//!__MACACC__
    float res = in_val * max(0.0, min(1.0, 1.0 / 6.0 * in_val + 0.5));
#endif//__MACACC__
    return res;
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_HardSwish, half>(const half& in_val)
{
#ifdef __MACACC__
    float in_valf = __half2float(in_val);
    float resf    = in_valf * max(0.0f, min(1.0f, float(HardSwish_OPT) * in_valf + 0.5f));
    return __float2half(resf);
#else//!__MACACC__
    float in_valf = __half2float(in_val);
    float resf    = in_valf * max(0.0, min(1.0, 1.0 / 6.0 * in_valf + 0.5));
    return __float2half(resf);
#endif//__MACACC__
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_HardMish, float>(const float& in_val)
{
    if(in_val > 0)
        return in_val;
    else if (in_val <= 0 && in_val > -2)
        return in_val * in_val * 0.5f + in_val;
    else
        return 0;
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_HardMish, half>(const half& in_val)
{
    float in_valf = __half2float(in_val);
    if(in_valf > 0)
        return __float2half(in_valf);
    else if (in_valf <= 0 && in_valf> -2)
        return __float2half(in_valf * in_valf * 0.5f + in_valf);
    else
        return __float2half(0.0);
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Mish, float>(const float& in_val){
    if (in_val > 20.0f) return in_val;
    float value;
#ifdef __MACACC__
    value = 1.f + exp(in_val);
    value = value * value;
    value = (value - 1.f) * __builtin_mxc_rcpf(value + 1.f);
#else//!__MACACC__
    value = 1 + exp(in_val);
    value = value * value;
    value = (value - 1) / (value + 1);
#endif//__MACACC__
    return in_val * value;
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Mish, half>(const half& in_val)
{
    float in_valf = __half2float(in_val);
    if (in_valf > 20.0f) return in_val;
    float value;
#ifdef __MACACC__
    value = 1.f + exp(in_valf);
    value = value * value;
    value = (value - 1.f) * __builtin_mxc_rcpf(value + 1.f);
#else//!__MACACC__
    value = 1 + exp(in_valf);
    value = value * value;
    value = (value - 1) / (value + 1);
#endif//__MACACC__
    return __float2half(in_valf * value);
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Softsign, float>(const float& in_val)
{
    return in_val / (abs(in_val) + 1);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Softsign, half>(const half& in_val)
{
    float in_valf = __half2float(in_val);
    return __float2half(in_valf / (abs(in_valf) + 1));
}

#endif

template <UnaryOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_any(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    output[index] = ppl_scalar_unary<OpT, DataT>(in_val);
#endif
}

template <UnaryOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_any_int8(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output,
    QuantKernelParamCuda qparam)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    float in_val_f = _int82float(in_val, qparam.i_step, qparam.i_zero_point);
    float out_val_f = ppl_scalar_unary<OpT, float>(in_val_f);
    output[index] = _float2int8(out_val_f, qparam.o_step, qparam.o_zero_point);
#endif
}

#ifdef __MACACC__
#define INT8_COMPAT 8
#define HALF_COMPAT 8
#define FLOAT_COMPAT 4

template <UnaryOpType OpT>
__global__ void ppl_cukernel_unary_any_opt(
    const uint64_t num_elems,
    const float* input,
    float* output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (num_elems + FLOAT_COMPAT - 1) / FLOAT_COMPAT)
        return;
    float4 in_val = *((float4*)input + index);
    float *ptr_data = (float*)&in_val;
    float4 reg;
    float* ptr_reg = (float*)&reg;
    for (int i = 0; i < FLOAT_COMPAT; i++) {
        ptr_reg[i]=ppl_scalar_unary<OpT, float>(ptr_data[i]);
    }
    float4 *out_val = (float4*)output + index;
    *out_val = reg;
#endif
}

template <UnaryOpType OpT>
__global__ void ppl_cukernel_unary_any_opt(
    const uint64_t num_elems,
    const half* input,
    half* output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (num_elems + HALF_COMPAT - 1) / HALF_COMPAT)
        return;
    float4 in_val = *((float4*)input + index);
    half *ptr_data = (half*)&in_val;
    float4 reg;
    half* ptr_reg = (half*)&reg;
    for (int i = 0; i < HALF_COMPAT; i++) {
        ptr_reg[i]=ppl_scalar_unary<OpT, half>(ptr_data[i]);
    }
    float4 *out_val = (float4*)output + index;
    *out_val = reg;
#endif
}

template <UnaryOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_any_int8_opt(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output,
    QuantKernelParamCuda qparam)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems / INT8_COMPAT) return;
    uint64_t in_val = ((uint64_t*)input)[index];
    uint64_t out_val = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float in_val_f = _int82float((DataT)(in_val&0xFF), qparam.i_step, qparam.i_zero_point);
        float out_val_f = ppl_scalar_unary<OpT, float>(in_val_f);
        DataT oc = _float2int8(out_val_f, qparam.o_step, qparam.o_zero_point);
        out_val |= (((uint64_t(oc)) & 0xFF) << (i*8));
        in_val >>= 8;
    }
    ((uint64_t*)output)[index] = out_val;
#endif
}

template<bool SAME_STEP = false, bool SAME_ZERO_POINT = false, bool ZERO_POINT=false>
__global__ void ppl_cukernel_unary_relu_int8_opt(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    QuantKernelParamCuda qparam)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems / INT8_COMPAT) return;
    uint64_t in_val = ((uint64_t*)input)[index];
    uint64_t out_val = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float in_val_f = _int82float((int8_t)(in_val&0xff), qparam.i_step, qparam.i_zero_point);
        float out_val_f = ppl_scalar_unary<Unary_Relu, float>(in_val_f);
        int8_t oc = _float2int8(out_val_f, qparam.o_step, qparam.o_zero_point);
        out_val |= (((uint64_t(oc)) & 0xff) << (i*8));
        in_val >>= 8;
    }
    ((uint64_t*)output)[index] = out_val;
#endif
}

/*
When i_step == o_step and i_zero_point == o_zero_point == 0
when i_step > 0, do a mask on input for current in_val, and let in_val & 0x80 to detect the flag
of this int8_t value, when in_val & 0x80 == 0, means the in_val is positive, otherwise in_val is negative
only in_val have same flag with i_step, the value will be saved, otherwise 0 is filled
*/
template<>
__global__ void ppl_cukernel_unary_relu_int8_opt<true,true,true>(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    QuantKernelParamCuda qparam)
{
    #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems / INT8_COMPAT) return;
    uint64_t in_val = ((uint64_t*)input)[index];
    uint64_t mask1 = 0x80;  //Detect the flag 
    uint64_t mask2 = 0xFF;
    uint64_t out_val = 0;
    
    if (qparam.i_step > 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if ((in_val & mask1) == 0) out_val |= (in_val & mask2);
            mask1 <<= 8;
            mask2 <<= 8;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if ((in_val & mask1) > 0) out_val |= (in_val & mask2);
            mask1 <<= 8;
            mask2 <<= 8;
        }
    }
    ((uint64_t*)output)[index] = out_val;
    #endif
}

/*
When i_step == o_step and i_zero_point == o_zero_point != 0
when i_step > 0, do a mask on input for current in_val, and check if (int)in_val - i_zero_point > 0
of this int8_t value, when in_val & 0x80 == 0, means the in_val is positive, otherwise in_val is negative
only in_val-i_zero_point have same flag with i_step, the value will be saved, otherwise 0 is filled
*/
template<>
__global__ void ppl_cukernel_unary_relu_int8_opt<true, true, false>(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    QuantKernelParamCuda qparam)
{
    #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems / INT8_COMPAT) return;
    uint64_t in_val_uint64 = ((uint64_t*)input)[index];
    uint64_t mask2 = 0xFF;
    uint64_t out_val = 0;
    if (qparam.i_step > 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int8_t in_val = (int8_t)((in_val_uint64 >> i * 8) & 0xFF);
            if ((int)in_val - qparam.i_zero_point > 0) out_val |= (in_val_uint64 & mask2);
            mask2 <<= 8;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int8_t in_val = (int8_t)((in_val_uint64 >> i * 8) & 0xFF);
            if ((int)in_val - qparam.i_zero_point < 0) out_val |= (in_val_uint64 & mask2);
            mask2 <<= 8;
        }
    }
    ((uint64_t*)output)[index] = out_val;
    #endif
}

#endif
#ifdef __MACACC__
#define UNARY_RELU_INT8_OPT \
if (qparam->i_step == qparam->o_step) { \
    if (qparam->i_zero_point == 0 && qparam->o_zero_point == 0) { \
        ppl_cukernel_unary_relu_int8_opt<true,true,true><<<gs, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, *qparam); \
    } else if (qparam->i_zero_point == qparam->o_zero_point) { \
        ppl_cukernel_unary_relu_int8_opt<true,true,false><<<gs, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, *qparam); \
    } else { \
        ppl_cukernel_unary_relu_int8_opt<true,false,false><<<gs, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, *qparam);    \
    } \
} else { \
    ppl_cukernel_unary_relu_int8_opt<false,false,false><<<gs, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, *qparam);    \
}

#define UNARY_INSTANT(TYPE)                                                                                                                    \
    ppl::common::RetCode PPLCUDAUnary##TYPE##ForwardImp(                                                                                       \
        cudaStream_t stream,                                                                                                                   \
        const ppl::common::TensorShape* input_shape,                                                                                               \
        const void* input,                                                                                                                     \
        const ppl::common::TensorShape* output_shape,                                                                                              \
        void* output,                                                                                                                          \
        const QuantKernelParamCuda* qparam)                                                                                                 \
    {                                                                                                                                          \
        uint64_t num_elems = output_shape->CalcElementsIncludingPadding();                                                                     \
        int block_size     = 256;                                                                                                              \
        uint64_t grid_size = (num_elems + block_size - 1) / block_size;                                                                        \
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {                                                                    \
            int capacity_per_block = block_size * FLOAT_COMPAT;                                                       \
            int gs = (num_elems + capacity_per_block - 1) / capacity_per_block;                                     \
            ppl_cukernel_unary_any_opt<Unary_##TYPE><<<gs, block_size, 0, stream>>>(num_elems, (const float*)input, (float*)output); \
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {                                                             \
            if (Unary_##TYPE == Unary_HardSwish or Unary_##TYPE == Unary_Relu or  Unary_##TYPE == Unary_TanH or Unary_##TYPE == Unary_Mish or Unary_##TYPE == Unary_Sigmoid) {    \
                int capacity_per_block = block_size * HALF_COMPAT;                                                       \
                int gs = (num_elems + capacity_per_block - 1) / capacity_per_block;                                     \
                ppl_cukernel_unary_any_opt<Unary_##TYPE><<<gs, block_size, 0, stream>>>(num_elems, (const half*)input, (half*)output);    \
            } else {                                                                                                                           \
                ppl_cukernel_unary_any<Unary_##TYPE, half><<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, (half*)output);    \
            }                                                                                                                                   \
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {                                                                \
            if (num_elems >= 256*INT8_COMPAT*1024 && num_elems % INT8_COMPAT == 0) {                                                           \
                int capacity_per_block = block_size * INT8_COMPAT;                                                                                 \
                int gs = (num_elems + capacity_per_block - 1) / capacity_per_block;                                                                \
                if (Unary_##TYPE == Unary_Relu) {                                                                                                  \
                    UNARY_RELU_INT8_OPT                                                                                                            \
                } else {                                                                                                                           \
                    ppl_cukernel_unary_any_int8_opt<Unary_##TYPE, int8_t><<<gs, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, *qparam);    \
                }                                                                                                                                                       \
            } else {                                                                                                                                                    \
                ppl_cukernel_unary_any_int8<Unary_##TYPE, int8_t><<<grid_size, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, *qparam);     \
            }                                                                                                                                                           \
        } else {                                                                                                                               \
            return ppl::common::RC_UNSUPPORTED;                                                                                                \
        }                                                                                                                                      \
        return ppl::common::RC_SUCCESS;                                                                                                        \
    }
#else
#define UNARY_INSTANT(TYPE)                                                                                                                    \
    ppl::common::RetCode PPLCUDAUnary##TYPE##ForwardImp(                                                                                       \
        cudaStream_t stream,                                                                                                                   \
        const ppl::common::TensorShape* input_shape,                                                                                               \
        const void* input,                                                                                                                     \
        const ppl::common::TensorShape* output_shape,                                                                                              \
        void* output,                                                                                                                          \
        const QuantKernelParamCuda* qparam)                                                                                                 \
    {                                                                                                                                          \
        uint64_t num_elems = output_shape->CalcElementsIncludingPadding();                                                                     \
        int block_size     = 256;                                                                                                              \
        uint64_t grid_size = (num_elems + block_size - 1) / block_size;                                                                        \
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {                                                                    \
            ppl_cukernel_unary_any<Unary_##TYPE, float><<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (float*)output); \
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {                                                             \
            ppl_cukernel_unary_any<Unary_##TYPE, half><<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, (half*)output);    \
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {                                                                \
            ppl_cukernel_unary_any_int8<Unary_##TYPE, int8_t><<<grid_size, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, *qparam);    \
        } else {                                                                                                                               \
            return ppl::common::RC_UNSUPPORTED;                                                                                                \
        }                                                                                                                                      \
        return ppl::common::RC_SUCCESS;                                                                                                        \
    }
#endif

UNARY_INSTANT(Abs);
UNARY_INSTANT(Relu);
UNARY_INSTANT(TanH);
UNARY_INSTANT(Sigmoid);
UNARY_INSTANT(Sqrt);
UNARY_INSTANT(Square);
UNARY_INSTANT(Floor);
UNARY_INSTANT(Ceil);
UNARY_INSTANT(Erf);
UNARY_INSTANT(Sin);
UNARY_INSTANT(Cos);
UNARY_INSTANT(Round);
UNARY_INSTANT(Sign);
UNARY_INSTANT(Sinh);
UNARY_INSTANT(Cosh);
UNARY_INSTANT(Asin);
UNARY_INSTANT(Acos);
UNARY_INSTANT(Atan);
UNARY_INSTANT(Asinh);
UNARY_INSTANT(Acosh);
UNARY_INSTANT(Atanh);
UNARY_INSTANT(HardSwish);
UNARY_INSTANT(HardMish);
UNARY_INSTANT(Mish);
UNARY_INSTANT(Softsign);

#undef UNARY_INSTANT
