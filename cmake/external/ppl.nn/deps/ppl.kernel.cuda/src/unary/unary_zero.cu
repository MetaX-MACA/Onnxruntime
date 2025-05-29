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

#include "cudakernel/unary/unary_zero.h"
#include "../reformat/cvt_int8_float.cuh"
#include "cudakernel/common/common.h"
#include <cuda_fp16.h>

#ifdef __MACACC__
#include "cudakernel/common/divmod_fast.h"
#define __USEOPT__
#endif

enum UnaryZeroOpType {
    UnaryZero_Unknown = 0,
    UnaryZero_Cos,
    UnaryZero_Exp,
    UnaryZero_Log,
    UnaryZero_Sigmoid,
    UnaryZero_Softplus,
    UnaryZero_Reciprocal,
    UnaryZero_ForceWord = INT_MAX,
};

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <UnaryZeroOpType OpT, typename DataT>
__device__ __inline__ DataT ppl_scalar_unary_zero(const DataT& in_val);

template <>
__device__ __inline__ float ppl_scalar_unary_zero<UnaryZero_Cos, float>(const float& in_val)
{
    return cos(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary_zero<UnaryZero_Cos, half>(const half& in_val)
{
    return __float2half(cos(__half2float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary_zero<UnaryZero_Exp, float>(const float& in_val)
{
    #ifdef __USEOPT__
        return __builtin_expf(in_val);
    #else//!__USEOPT__
        return exp(in_val);
    #endif//__USEOPT__
}

template <>
__device__ __inline__ half ppl_scalar_unary_zero<UnaryZero_Exp, half>(const half& in_val)
{
    return __float2half(exp(__half2float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary_zero<UnaryZero_Log, float>(const float& in_val)
{
    #ifdef __USEOPT__
        return __builtin_logf(in_val);
    #else//!__USEOPT__
        return log(in_val);
    #endif//__USEOPT__
}

template <>
__device__ __inline__ half ppl_scalar_unary_zero<UnaryZero_Log, half>(const half& in_val)
{
    return __float2half(log(__half2float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary_zero<UnaryZero_Sigmoid, float>(const float& in_val)
{
    // #ifdef __USEOPT__
    //     return __builtin_mxc_rcpf(1.f + __builtin_expf(-in_val));
    // #else//!__USEOPT__
    //     return 1.f / (1.f + expf(-in_val));
    // #endif//__USEOPT__
    return 1.f / (1.f + expf(-in_val));
}

template <>
__device__ __inline__ half ppl_scalar_unary_zero<UnaryZero_Sigmoid, half>(const half& in_val)
{
    float in_valf = __half2float(in_val);
    #ifdef __USEOPT__  
        float resf = __builtin_mxc_rcpf(1.f + __builtin_expf(-in_valf));
    #else//!__USEOPT__
        float resf = 1.f / (1.f + expf(-in_valf));
    #endif//__USEOPT__
    return __float2half(resf);
}
template <>
__device__ __inline__ float ppl_scalar_unary_zero<UnaryZero_Softplus, float>(const float& in_val)
{  
    #ifdef __USEOPT__
        return __builtin_logf(1.f + __builtin_expf(in_val));
    #else//!__USEOPT__
        return log(1.f + exp(in_val));
    #endif//__USEOPT__
}
template <>
__device__ __inline__ half ppl_scalar_unary_zero<UnaryZero_Softplus, half>(const half& in_val)
{
    float in_valf = __half2float(in_val);
    float resf    = log(1.f + expf(in_valf));
    return __float2half(resf);
}
template <>
__device__ __inline__ float ppl_scalar_unary_zero<UnaryZero_Reciprocal, float>(const float& in_val)
{
    return 1.f / in_val;
}
template <>
__device__ __inline__ half ppl_scalar_unary_zero<UnaryZero_Reciprocal, half>(const half& in_val)
{
    float in_valf = __half2float(in_val);
    float resf    = 1.f / in_valf;
    return __float2half(resf);
}
#endif

template <UnaryZeroOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_zero_any(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    output[index] = ppl_scalar_unary_zero<OpT, DataT>(in_val);
#endif
}

template <UnaryZeroOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_zero_any_int8(
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
    float out_val_f = ppl_scalar_unary_zero<OpT, float>(in_val_f);
    output[index] = _float2int8(out_val_f, qparam.o_step, qparam.o_zero_point);
#endif
}


#ifdef __USEOPT__
#define INT8_COMPAT 8

template <UnaryZeroOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_zero_any_opt(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    float4 in_val = *((float4*)input + index);
    half *ptr_data = (half*)&in_val;
    float4 reg;
    half* ptr_reg = (half*)&reg;
    for (int i = 0; i < INT8_COMPAT; i++) {
        ptr_reg[i]=ppl_scalar_unary_zero<OpT, DataT>(ptr_data[i]);
    }
    float4 *out_val = (float4*)output + index;
    *out_val = reg;
#endif
}

template <UnaryZeroOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_zero_any_int8_opt(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output,
    QuantKernelParamCuda qparam)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elems / INT8_COMPAT) {
        uint64_t in_val = ((uint64_t*)input)[index];
        uint64_t out_val = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float in_val_f = _int82float((DataT)(in_val&0xff), qparam.i_step, qparam.i_zero_point);
            float out_val_f = ppl_scalar_unary_zero<OpT, float>(in_val_f);
            DataT oc = _float2int8(out_val_f, qparam.o_step, qparam.o_zero_point);
            uint64_t oc64 = 0;
            oc64 |= oc;
            oc64 <<= (i*8);
            out_val |= oc64;
            in_val >>= 8;
        }
        ((uint64_t*)output)[index] = out_val;
        return;
    } else if (num_elems % INT8_COMPAT > 0 && index < num_elems / INT8_COMPAT + 1) {
        uint64_t start_pos = index * INT8_COMPAT;
        for (uint64_t i = start_pos; i < num_elems; i++) {
            DataT in_val  = input[i];
            float in_val_f = _int82float(in_val, qparam.i_step, qparam.i_zero_point);
            float out_val_f = ppl_scalar_unary_zero<OpT, float>(in_val_f);
            output[i] = _float2int8(out_val_f, qparam.o_step, qparam.o_zero_point);
        }
        return;
    } else {
        return;
    }
#endif
}

template <UnaryZeroOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_zero_nhwc_opt(const uint64_t num_elems,
                                      int channels,
                                      int pad_channels,
                                      int chw,
                                      int hw,
                                      const DataT *input,
                                      DataT *output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int chw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chw_idx >= chw)
        return;
    int b_idx     = blockIdx.z;
    int64_t start_pos = b_idx * hw + chw_idx;
    float4 in_val = *((float4*)input + start_pos);
    float4 *out_val = (float4*)output + start_pos;
    half *ptr_data = (half*)&in_val;
    
    float4 reg = make_float4(1,2,3,4);
    half* ptr_reg = (half*)&reg;
    ptr_reg[0]=ppl_scalar_unary_zero<OpT, DataT>(ptr_data[0]);
    *out_val = reg;

#endif
}
template <UnaryZeroOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_zero_nhwc_opt2(const uint64_t num_elems,
                                      int channels,
                                      int pad_channels,
                                      int chw,
                                      int hw,
                                      const DataT *input,
                                      DataT *output,
                                      DivModFast channels_fast)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int chw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chw_idx >= chw) 
        return;
    int c_idx, hw_idx;
    channels_fast.divmod(chw_idx, hw_idx, c_idx);
    int b_idx     = blockIdx.z;
    // pad_channels == channels
    int64_t start_pos = b_idx * chw + chw_idx;
    float4 in_val = *((float4*)input + start_pos);
    float4 *out_val = (float4*)output + start_pos;
    half *ptr_data = (half*)&in_val;
    
    float4 reg = make_float4(1,2,3,4);
    half* ptr_reg = (half*)&reg;
    for(int i = 0; i < INT8_COMPAT; i++){
        ptr_reg[i]=ppl_scalar_unary_zero<OpT, DataT>(ptr_data[i]);
    }
    *out_val = reg;

#endif
}


#endif


template <UnaryZeroOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_zero_nhwc(const uint64_t num_elems,
                                      int channels,
                                      int pad_channels,
                                      int chw,
                                      int hw,
                                      const DataT *input,
                                      DataT *output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int chw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chw_idx >= chw)
        return;
    int c_idx     = chw_idx % channels;
    int hw_idx    = chw_idx / channels;
    int b_idx     = blockIdx.z;
    int64_t index = (b_idx * hw + hw_idx) * pad_channels + c_idx;
    output[index] = ppl_scalar_unary_zero<OpT, DataT>(input[index]);
#endif
}

template <UnaryZeroOpType OpT>
__global__ void ppl_cukernel_unary_zero_nhwc_int8(const uint64_t num_elems,
                                      int channels,
                                      int pad_channels,
                                      int chw,
                                      int hw,
                                      const int8_t *input,
                                      int8_t *output,
                                      QuantKernelParamCuda qparam)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int chw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chw_idx >= chw)
        return;
    int c_idx     = chw_idx % channels;
    int hw_idx    = chw_idx / channels;
    int b_idx     = blockIdx.z;
    int64_t index = (b_idx * hw + hw_idx) * pad_channels + c_idx;
    float in_val_f = _int82float(input[index], qparam.i_step, qparam.i_zero_point);
    float out_val_f = ppl_scalar_unary_zero<OpT, float>(in_val_f);
    output[index] = _float2int8(out_val_f, qparam.o_step, qparam.o_zero_point);
#endif
}

#ifdef __USEOPT__
#define UNARYZERO_INSTANT(TYPE)                                                                                        \
    ppl::common::RetCode PPLCUDAUnaryZero##TYPE##ForwardImp(                                                           \
        cudaStream_t stream,                                                                                           \
        const ppl::common::TensorShape* input_shape,                                                                       \
        const void* input,                                                                                             \
        const ppl::common::TensorShape* output_shape,                                                                      \
        void* output,                                                                                                  \
        const QuantKernelParamCuda* qparam)                                                                         \
    {                                                                                                                  \
        uint64_t num_elems = output_shape->CalcElementsIncludingPadding();                                             \
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {                                        \
            int block_size = 256;                                                                                      \
            int grid_size  = (num_elems + block_size - 1) / block_size;                                                \
            if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {                                        \
                ppl_cukernel_unary_zero_any<UnaryZero_##TYPE, float><<<grid_size, block_size, 0, stream>>>(num_elems,      \
                                                                                    (const float *)input,              \
                                                                                    (float *)output);                  \
            } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {                                 \
                if(UnaryZero_##TYPE == UnaryZero_Sigmoid or UnaryZero_##TYPE == UnaryZero_Exp){                                                                   \
                    int capacity_per_block = block_size * INT8_COMPAT;                                                       \
                    int gs = (num_elems + capacity_per_block - 1) / capacity_per_block;                                     \
                    ppl_cukernel_unary_zero_any_opt<UnaryZero_##TYPE, half><<<gs, block_size, 0, stream>>>((num_elems + INT8_COMPAT - 1) / INT8_COMPAT,       \
                                                                                    (const half *)input,               \
                                                                                    (half *)output);                   \
                } else {                                                                                                     \
                    ppl_cukernel_unary_zero_any<UnaryZero_##TYPE, half><<<grid_size, block_size, 0, stream>>>(num_elems,       \
                                                                                    (const half *)input,               \
                                                                                    (half *)output);                              \
                }                                                                                                                           \
            } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {                                    \
                if(UnaryZero_##TYPE == UnaryZero_Softplus){                                                                   \
                    int capacity_per_block = block_size * INT8_COMPAT;                                                       \
                    int gs = (num_elems + capacity_per_block - 1) / capacity_per_block;                                     \
                    ppl_cukernel_unary_zero_any_int8_opt<UnaryZero_##TYPE><<<gs, block_size, 0, stream>>>(num_elems,        \
                                                                                        (const int8_t *)input,             \
                                                                                        (int8_t *)output,                  \
                                                                                        *qparam);                          \
                } else {                                                                                                     \
                    ppl_cukernel_unary_zero_any_int8<UnaryZero_##TYPE><<<grid_size, block_size, 0, stream>>>(num_elems,        \
                                                                                    (const int8_t *)input,             \
                                                                                    (int8_t *)output,                  \
                                                                                    *qparam);                              \
                }                                                                                                       \
            } else {                                                                                                   \
                return ppl::common::RC_UNSUPPORTED;                                                                    \
            }                                                                                                          \
        } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||                                   \
                    output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||                                 \
                    output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {                                   \
            int block_size = 256;                                                                                      \
            dim3 grid_size;                                                                                            \
            int batch      = output_shape->GetDim(0);                                                                  \
            int channels   = output_shape->GetDim(1);                                                                  \
            int pad_channels   = output_shape->GetDim(1) + output_shape->GetPadding1(1);                               \
            int hw         = 1;                                                                                        \
            int dim_count = output_shape->GetDimCount();                                                               \
            for (int i=2; i < dim_count; i++)   hw *= output_shape->GetDim(i);                                         \
            int chw        = channels * hw;                                                                            \
            grid_size.x = (chw + block_size - 1) / block_size;                                                         \
            grid_size.z = batch;                                                                                       \
            if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {                                        \
                ppl_cukernel_unary_zero_nhwc<UnaryZero_##TYPE, float><<<grid_size, block_size, 0, stream>>>(           \
                    num_elems, channels, pad_channels, chw, hw,                      \
                    (const float *)input, (float *)output);                                                            \
            } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {                                 \
                if(pad_channels > channels){                                                                           \
                    if(channels == 1 && pad_channels == 8) {                                                            \
                        ppl_cukernel_unary_zero_nhwc_opt<UnaryZero_##TYPE, half><<<grid_size, block_size, 0, stream>>>(    \
                            num_elems, channels, pad_channels, chw, hw,                                        \
                            (const half *)input, (half *)output);                                                      \
                    } else {                                                                                           \
                        ppl_cukernel_unary_zero_nhwc<UnaryZero_##TYPE, half><<<grid_size, block_size, 0, stream>>>(            \
                            num_elems, channels, pad_channels, chw, hw,                      \
                            (const half *)input, (half *)output);                                                              \
                    }                                                                                                  \
                } else if(channels % 8 == 0 && grid_size.x > 8){                                                                                               \
                    int capacity_per_block = block_size * INT8_COMPAT;                                                 \
                    dim3 gs;                                                                                           \
                    gs.x = (chw + capacity_per_block - 1) / capacity_per_block;                                        \
                    gs.z = batch;                                                                                      \
                    ppl_cukernel_unary_zero_nhwc_opt2<UnaryZero_##TYPE, half><<<gs, block_size, 0, stream>>>(          \
                        num_elems, channels, pad_channels, chw / INT8_COMPAT, hw,                          \
                        (const half *)input, (half *)output, DivModFast(INT8_COMPAT));                                \
                } else {                                                                                                      \
                    ppl_cukernel_unary_zero_nhwc<UnaryZero_##TYPE, half><<<grid_size, block_size, 0, stream>>>(            \
                            num_elems, channels, pad_channels, chw, hw,                      \
                            (const half *)input, (half *)output);                                                              \
                }                                                                                                          \
            } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {                                    \
                ppl_cukernel_unary_zero_nhwc_int8<UnaryZero_##TYPE><<<grid_size, block_size, 0, stream>>>(             \
                    num_elems, channels, pad_channels, chw, hw,                      \
                    (const int8_t *)input, (int8_t *)output, *qparam);                                                 \
            } else {                                                                                                   \
                return ppl::common::RC_UNSUPPORTED;                                                                    \
            }                                                                                                          \
        } else {                                                                                                       \
            return ppl::common::RC_UNSUPPORTED;                                                                        \
        }                                                                                                              \
        return ppl::common::RC_SUCCESS;                                                                                \
    }
#else //!__USEOPT__
#define UNARYZERO_INSTANT(TYPE)                                                                                        \
    ppl::common::RetCode PPLCUDAUnaryZero##TYPE##ForwardImp(                                                           \
        cudaStream_t stream,                                                                                           \
        const ppl::common::TensorShape* input_shape,                                                                       \
        const void* input,                                                                                             \
        const ppl::common::TensorShape* output_shape,                                                                      \
        void* output,                                                                                                  \
        const QuantKernelParamCuda* qparam)                                                                         \
    {                                                                                                                  \
        uint64_t num_elems = output_shape->CalcElementsIncludingPadding();                                             \
        int batch          = output_shape->GetDim(0);                                                                  \
        int channels       = output_shape->GetDim(1);                                                                  \
        int pad_channels   = output_shape->GetDim(1) + output_shape->GetPadding1(1);                                   \
        int height         = output_shape->GetDim(2);                                                                  \
        int width          = output_shape->CalcElementsFromDimensionIncludingPadding(3);                                                                  \
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {                                        \
            int block_size = 256;                                                                                      \
            int grid_size  = (num_elems + block_size - 1) / block_size;                                                \
            if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {                                        \
                ppl_cukernel_unary_zero_any<UnaryZero_##TYPE, float><<<grid_size, block_size, 0, stream>>>(num_elems,      \
                                                                                    (const float *)input,              \
                                                                                    (float *)output);                  \
            } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {                                 \
                ppl_cukernel_unary_zero_any<UnaryZero_##TYPE, half><<<grid_size, block_size, 0, stream>>>(num_elems,       \
                                                                                    (const half *)input,               \
                                                                                    (half *)output);                   \
            } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {                                    \
                ppl_cukernel_unary_zero_any_int8<UnaryZero_##TYPE><<<grid_size, block_size, 0, stream>>>(num_elems,        \
                                                                                    (const int8_t *)input,             \
                                                                                    (int8_t *)output,                  \
                                                                                    *qparam);                          \
            } else {                                                                                                   \
                return ppl::common::RC_UNSUPPORTED;                                                                    \
            }                                                                                                          \
        } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||                                   \
                    output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||                                 \
                    output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {                                   \
            int block_size = 256;                                                                                      \
            dim3 grid_size;                                                                                            \
            int chw     = channels * height * width;                                                                   \
            grid_size.x = (chw + block_size - 1) / block_size;                                                         \
            grid_size.z = batch;                                                                                       \
            if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {                                        \
                ppl_cukernel_unary_zero_nhwc<UnaryZero_##TYPE, float><<<grid_size, block_size, 0, stream>>>(           \
                    num_elems, channels, pad_channels, channels * height * width, height * width,                      \
                    (const float *)input, (float *)output);                                                            \
            } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {                                 \
                ppl_cukernel_unary_zero_nhwc<UnaryZero_##TYPE, half><<<grid_size, block_size, 0, stream>>>(            \
                    num_elems, channels, pad_channels, channels * height * width, height * width,                      \
                    (const half *)input, (half *)output);                                                              \
            } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {                                    \
                ppl_cukernel_unary_zero_nhwc_int8<UnaryZero_##TYPE><<<grid_size, block_size, 0, stream>>>(             \
                    num_elems, channels, pad_channels, channels * height * width, height * width,                      \
                    (const int8_t *)input, (int8_t *)output, *qparam);                                                 \
            } else {                                                                                                   \
                return ppl::common::RC_UNSUPPORTED;                                                                    \
            }                                                                                                          \
        } else {                                                                                                       \
            return ppl::common::RC_UNSUPPORTED;                                                                        \
        }                                                                                                              \
        return ppl::common::RC_SUCCESS;                                                                                \
    }
#endif//__USEOPT__

UNARYZERO_INSTANT(Cos);
UNARYZERO_INSTANT(Exp);
UNARYZERO_INSTANT(Log);
UNARYZERO_INSTANT(Sigmoid);
UNARYZERO_INSTANT(Softplus);
UNARYZERO_INSTANT(Reciprocal);

#undef UNARYZERO_INSTANT
