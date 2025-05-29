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

#include "cudakernel/unary/cast.h"
#include <cuda_fp16.h>
#include <utility>
#include <map>

#ifdef __MACACC__
#define OPT_CAST
#endif//__MACACC__

template <typename T>
struct ViaTypeMap {
    typedef T ViaT;
};

template <>
struct ViaTypeMap<half> {
    typedef float ViaT;
};

template <typename InT, typename OutT>
__device__ __inline__ OutT ppl_scalar_cast(const InT &a)
{
    const bool any_float16 = std::is_same<half, InT>::value || std::is_same<half, OutT>::value;
    typedef typename std::conditional<any_float16, half, OutT>::type T;
    typedef typename ViaTypeMap<T>::ViaT ViaT;
    return (OutT)((ViaT)a);
}

template <typename InT, typename OutT>
__global__ void ppl_cukernel_cast_any(
    const uint64_t num_elems,
    const void *input,
    void *output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    const InT *input_ptr = static_cast<const InT *>(input);
    InT in_val           = input_ptr[index];
    OutT *output_ptr     = static_cast<OutT *>(output);
    output_ptr[index]    = ppl_scalar_cast<InT, OutT>(in_val);
#endif
}

#ifdef OPT_CAST
template <typename InT, typename OutT,typename T1,typename T2>
__global__ void ppl_cukernel_cast_any_opt(
    const uint64_t num_elems,
    const void *input,
    void *output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    constexpr int times = sizeof(T1) / sizeof(InT);
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x)*times;
    if(blockIdx.x == gridDim.x - 1){
        const InT *input_ptr = static_cast<const InT *>(input);
        OutT *output_ptr     = static_cast<OutT *>(output);
        #pragma unroll times
        for(int i = 0; i < times; i++){
            if(index + i < num_elems){
                InT in_val           = input_ptr[index + i];
                OutT *output_ptr     = static_cast<OutT *>(output);
                output_ptr[index + i]    = ppl_scalar_cast<InT, OutT>(in_val);
            }
        }
    }else{
        T1 tmp = *(T1*)((InT*)input + index);
        InT* ptr_tmp = (InT*)&tmp;
        T2 tmp_out;
        OutT* ptr_tmp_out = (OutT*)&tmp_out;
        #pragma unroll times
        for(int i = 0; i < times; i++)
        {
            ptr_tmp_out[i] = ppl_scalar_cast<InT, OutT>(ptr_tmp[i]);
        }
        *(T2*)((OutT*)output + index) = tmp_out;
    }
#endif
}
#endif//OPT_CAST

#define INSERT_CAST_FUNC(SrcTyName, DstTyName, SrcT, DstT) \
    func_map.insert({DataTypePair(SrcTyName, DstTyName), ppl_cukernel_cast_any<SrcT, DstT>});

#define INSERT_CAST_FUNC2(SrcTyName, SrcT)                                    \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_FLOAT16, SrcT, half)    \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_FLOAT32, SrcT, float)   \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_FLOAT64, SrcT, double)  \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_INT8, SrcT, int8_t)     \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_INT16, SrcT, int16_t)   \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_INT32, SrcT, int32_t)   \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_INT64, SrcT, int64_t)   \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_UINT8, SrcT, uint8_t)   \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_UINT16, SrcT, uint16_t) \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_UINT32, SrcT, uint32_t) \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_UINT64, SrcT, uint64_t) \
    INSERT_CAST_FUNC(SrcTyName, ppl::common::DATATYPE_BOOL, SrcT, bool)

ppl::common::RetCode PPLCUDACastForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape *input_shape,
    const void *input,
    const ppl::common::TensorShape *output_shape,
    void *output,
    int to_)
{
    uint64_t num_elems                  = output_shape->CalcElementsIncludingPadding();
    int channels, pad_channels,height,width,block_size;
    uint64_t grid_size;
    if(output_shape->GetDimCount()==3){
        channels                        = output_shape->GetDim(0);
        pad_channels                    = output_shape->GetDim(0) + output_shape->GetPadding1(0);
        height                          = output_shape->GetDim(1);
        width                           = output_shape->GetDim(2);
    }else if(output_shape->GetDimCount() == 4){
        channels                        = output_shape->GetDim(1);
        pad_channels                    = output_shape->GetDim(1) + output_shape->GetPadding1(1);
        height                          = output_shape->GetDim(2);
        width                           = output_shape->GetDim(3);
    }
    
    block_size                      = 256;
    grid_size                  = (num_elems + block_size - 1) / block_size;
    const ppl::common::datatype_t in_t  = input_shape->GetDataType();
    const ppl::common::datatype_t out_t = output_shape->GetDataType();
    typedef void (*FuncType)(const uint64_t, const void *, void *);
    typedef std::pair<ppl::common::datatype_t, ppl::common::datatype_t> DataTypePair;
    std::map<DataTypePair, FuncType> func_map;
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_FLOAT16, half)
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_FLOAT32, float)
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_FLOAT64, double)
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_INT8, int8_t)
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_INT16, int16_t)
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_INT32, int32_t)
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_INT64, int64_t)
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_UINT8, uint8_t)
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_UINT16, uint16_t)
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_UINT32, uint32_t)
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_UINT64, uint64_t)
    INSERT_CAST_FUNC2(ppl::common::DATATYPE_BOOL, bool)

#ifdef OPT_CAST
    if(in_t == 5 && out_t == 6){
        grid_size                  = (num_elems + (block_size<<2) - 1) / (block_size<<2);
        ppl_cukernel_cast_any_opt<half,float,float2,float4><<<grid_size,block_size,0,stream>>>(
            num_elems, (const void *)input, (void *)output);
        return ppl::common::RC_SUCCESS;
    }else if(in_t == 6 && out_t == 5){
        grid_size                  = (num_elems + (block_size<<2) - 1) / (block_size<<2);
        ppl_cukernel_cast_any_opt<float,half,float4,float2><<<grid_size,block_size,0,stream>>>(
            num_elems, (const void *)input, (void *)output);
        return ppl::common::RC_SUCCESS;
    }
#endif//OPT_CAST
    func_map[DataTypePair(in_t, out_t)]<<<grid_size, block_size, 0, stream>>>(
        num_elems, (const void *)input, (void *)output);

    return ppl::common::RC_SUCCESS;
}