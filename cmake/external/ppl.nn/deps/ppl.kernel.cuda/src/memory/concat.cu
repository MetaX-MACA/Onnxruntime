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

#include "cudakernel/memory/concat.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "cudakernel/common/common.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#define NHWC8_ALIGNED_AXIS (8)
#define NHWC16_ALIGNED_AXIS (16)
#ifdef __MACACC__
#define __CONCAT_OPT__
#endif//__MACACC__

bool is_aligned_axis(int axis,
                    int num_inputs,
                    int* input_dims[],
                    int* input_padded_dims[],
                    ppl::common::TensorShape* output_shape,
                    int n) {
    bool align = true;
    int64_t concat_size = 1;
    int num_dims     = output_shape->GetDimCount();
    int data_size = ppl::common::GetSizeOfDataType(output_shape->GetDataType());

    for (int i = 0; i < num_inputs; ++i) {
        for (int j = axis; j < num_dims; ++j)
            concat_size *= input_dims[i][j];

        if ((concat_size * data_size) % n != 0) {
            align = false;
            break;
        }
        concat_size = 1;
    }
  return align;
}

template <typename T>
__global__ void ppl_cukernel_concat(
    int64_t num_elems,
    const T* inputs,
    int64_t concat_size,
    int64_t top_axis_width,
    DivModFast num_elems_inner_fast,
    int axis_offset,
    T* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int outer_idx, inner_idx;
        num_elems_inner_fast.divmod(i, outer_idx, inner_idx);
        int64_t top_idx = inner_idx +
                          (outer_idx * top_axis_width + axis_offset) * concat_size;
        output[top_idx] = inputs[i];
    }
}

#ifdef __MACACC__
/*
    The input data will be strictly restrict to:
    1. num_elems % (16 / sizeof(TYPE)) == 0
    2. input_concat_size % (16 / sizeof(TYPE)) == 0
    3. axis_offset * concat_size should be aligned to 16 / sizeof(TYPE)
*/
template <typename T>
__global__ void ppl_cukernel_concat_opt_16(
    int64_t num_elems,
    const T* inputs,
    int64_t concat_size,
    int64_t top_axis_width,
    DivModFast num_elems_inner_fast,
    int axis_offset,
    T* output)
{
    const uint64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t thread_capacity = 16 / sizeof(T);
    const uint64_t gstart = gid * thread_capacity;
    T tmp[16];
    if (gstart < num_elems) {
        int outer_idx, inner_idx;
        num_elems_inner_fast.divmod(gstart, outer_idx, inner_idx);
        int64_t top_idx = inner_idx + (outer_idx * top_axis_width + axis_offset) * concat_size;
        // uint64_t* i_addr = (uint64_t*)(inputs + gid * thread_capacity);
        // uint64_t i1 = i_addr[0];
        // uint64_t i2 = i_addr[1];
        // uint64_t* o_addr = (uint64_t*)(output + top_idx);
        // o_addr[0] = i1;
        // o_addr[1] = i2;
        T* i_addr = (T*)(inputs + gstart);
        T* o_addr = (T*)(output + top_idx);
        #pragma unroll
        for (size_t i=0; i<thread_capacity; i++) {
            tmp[i] = i_addr[i];
        }
        #pragma unroll
        for (size_t i=0; i<thread_capacity; i++) {
            o_addr[i] = tmp[i];
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_concat_nd_padding_two_input_opt(
    int64_t out_num_elems,
    int scale,
    const T* input0,
    const T* input1,
    int64_t input0_axis_width,
    int64_t input1_axis_width,
    DivModFast out_inner_dims_fast,
    T* output)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t idx = tid * scale;
    if (idx >= out_num_elems) return;
    int bs_idx, inner_idx;
    if (idx + scale <= out_num_elems){
        float4* out_val = (float4*)(output + idx);
        float4 out_data = make_float4(1,2,3,4);
        out_inner_dims_fast.divmod(idx, bs_idx, inner_idx);
        T* ptr_out = (T*)&out_data;
        for(int i = 0; i < scale; i++){
            out_inner_dims_fast.divmod(idx + i, bs_idx, inner_idx);
            if (inner_idx >= input0_axis_width) {
                ptr_out[i] = input1[bs_idx * input1_axis_width + inner_idx - input0_axis_width];
            } else {
                ptr_out[i] = input0[bs_idx * input0_axis_width + inner_idx];
            }
        }
        out_val[0] = out_data; 
    } else {
        for(int i = 0; idx + i < out_num_elems; i++){
            out_inner_dims_fast.divmod(idx + i, bs_idx, inner_idx);
            if (inner_idx >= input0_axis_width) {
                output[idx + i] = input1[bs_idx * input1_axis_width + inner_idx - input0_axis_width];
            } else {
                output[idx + i] = input0[bs_idx * input0_axis_width + inner_idx];
            }
        }
    } 
}

/*
    Only accept input data type that sizeof(TYPE) <= 8
    when sizeof(TYPE) == 8, There is no optimization
*/
template <typename T>
__global__ void ppl_cukernel_concat_opt_8(
    int64_t num_elems,
    const T* inputs,
    int64_t concat_size,
    int64_t top_axis_width,
    DivModFast num_elems_inner_fast,
    int axis_offset,
    T* output)
{
    const uint64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t thread_capacity = 8 / sizeof(T);
    const uint64_t gstart = gid * thread_capacity;
    if (gid < num_elems / thread_capacity) {
        int outer_idx, inner_idx;
        num_elems_inner_fast.divmod(gstart, outer_idx, inner_idx);
        int outer_idx_end, inner_idx_end;
        num_elems_inner_fast.divmod(gstart+thread_capacity-1, outer_idx_end, inner_idx_end);
        bool continuous = true;
        if (outer_idx_end != outer_idx || inner_idx_end - inner_idx != thread_capacity - 1) continuous = false;
        if (continuous) {
            int64_t top_idx = inner_idx +
                                (outer_idx * top_axis_width + axis_offset) * concat_size;
            uint64_t* i_addr = (uint64_t*)(inputs + gstart);
            uint64_t i1 = i_addr[0];
            uint64_t* o_addr = (uint64_t*)(output + top_idx);
            o_addr[0] = i1;
        } else {
            #pragma unroll
            for (uint64_t i = 0; i < thread_capacity; i++) {
                int outer_idx, inner_idx;
                num_elems_inner_fast.divmod(gstart + i, outer_idx, inner_idx);
                int64_t top_idx = inner_idx +
                                (outer_idx * top_axis_width + axis_offset) * concat_size;
                output[top_idx] = inputs[gstart+i];
            }
        }
    } else if (gid == num_elems / thread_capacity && num_elems % thread_capacity > 0) {
        #pragma unroll
        for (uint64_t i = gstart; i < num_elems; i++) {
            int outer_idx, inner_idx;
            num_elems_inner_fast.divmod(i, outer_idx, inner_idx);
            int64_t top_idx = inner_idx +
                            (outer_idx * top_axis_width + axis_offset) * concat_size;
            output[top_idx] = inputs[i];
        }
    } else {
        return;
    }
}

/*
  ppl_cukernel_concat_opt_1 have no optimization
  but we need to re-write it for another name
  so we can easily figure out if we have do optimization
  Most of the time, this kernel will not be called.
*/
template <typename T>
__global__ void ppl_cukernel_concat_opt_1(
    int64_t num_elems,
    const T* inputs,
    int64_t concat_size,
    int64_t top_axis_width,
    DivModFast num_elems_inner_fast,
    int axis_offset,
    T* output)
{
    uint64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elems) return;
    int outer_idx, inner_idx;
    num_elems_inner_fast.divmod(i, outer_idx, inner_idx);
    int64_t top_idx = inner_idx +
                    (outer_idx * top_axis_width + axis_offset) * concat_size;
    output[top_idx] = inputs[i];
}

template<typename T, int N,bool fastmode>
__global__ void ppl_cukernel_concat_nhwc_nopadding_optN(
    int64_t num_elems,
    const T* inputs,
    int64_t input_axis_width,
    int64_t top_axis_width,
    DivModFast num_elems_inner_fast,
    int axis_offset,
    T* output)
{
    int64_t start_pos = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    const T* ptr_inputs = inputs + start_pos;
    int outer_idx, inner_idx;
    num_elems_inner_fast.divmod(start_pos, outer_idx, inner_idx);
    if constexpr(fastmode)
    {
        if(start_pos < num_elems)
        {
            uint64_t tmp = *((uint64_t*)ptr_inputs);
            int64_t top_idx = outer_idx * top_axis_width + axis_offset + inner_idx;

            *(uint64_t*)(output + top_idx) = tmp;
        }else{
            return;
        }
    }
    else
    {
        if(blockIdx.x == gridDim.x - 1)
        {
            uint64_t tmp = *((uint64_t*)ptr_inputs);
            int thread_capacity = min(N, num_elems - start_pos);
            int64_t top_idx = outer_idx * top_axis_width + axis_offset;
            for(int j = 0, thread_in_idx = inner_idx, shift = 0; j < thread_capacity; j++, thread_in_idx++, shift += N)
            {
                if constexpr(N==8)
                {
                    int8_t out = (tmp >> shift)&0XFF;
                    output[top_idx + thread_in_idx] = out;
                }
                else if constexpr(N==4)
                {
                    int16_t out = (tmp>>shift)&0xFFFF;
                    output[top_idx + thread_in_idx] = out;
                }
                else if constexpr(N==2)
                {
                    int32_t out = (tmp>>shift)&0xFFFFFFFF;
                    output[top_idx + thread_in_idx] = out;
                }
                else
                {
                    int64_t out = (tmp>>shift)&0xFFFFFFFFFFFFFFFF;
                    output[top_idx + thread_in_idx] = out;
                }

                if(thread_in_idx >= input_axis_width){
                        thread_in_idx = 0;
                        top_idx += top_axis_width;
                }
            }
        }
        else
        {
            uint64_t tmp = *((uint64_t*)ptr_inputs);
            int64_t top_idx = outer_idx * top_axis_width + axis_offset;
            #pragma unroll
            for(int j = 0, thread_in_idx = inner_idx, shift = 0; j < N; j++, thread_in_idx++, shift += N)
            {
                if constexpr(N==8)
                {
                    int8_t out = (tmp >> shift)&0XFF;
                    output[top_idx + thread_in_idx] = out;
                }
                else if constexpr(N==4)
                {
                    int16_t out = (tmp>>shift)&0xFFFF;
                    output[top_idx + thread_in_idx] = out;
                }
                else if constexpr(N==2)
                {
                    int32_t out = (tmp>>shift)&0xFFFFFFFF;
                    output[top_idx + thread_in_idx] = out;
                }
                else
                {
                    int64_t out = (tmp>>shift)&0xFFFFFFFFFFFFFFFF;
                    output[top_idx + thread_in_idx] = out;
                }

                if(thread_in_idx >= input_axis_width){
                    thread_in_idx = 0;
                    top_idx += top_axis_width;
                }
            }
        }
    }
}

template <typename T1, typename T2, int N=8>
__global__ void __launch_bounds__(256) ppl_cukernel_concat_nhwc_two_inputs_optN(
    int64_t num_elems,
    int inner_dims,
    int pad_inner_dims,
    int axis_width0,
    int pad_axis_width0,
    int axis_width1,
    int pad_axis_width1,
    DivModFast pad_inner_dims_fast,
    const T1* input0,
    const T1* input1,
    T2* output)
{
    int64_t start_pos = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    T2* ptr_output = output + start_pos;
    int outer_idx, inner_idx;
    pad_inner_dims_fast.divmod(start_pos, outer_idx, inner_idx);
    const T1* ptr_input0 = input0 + outer_idx*pad_axis_width0;
    const T1* ptr_input1 = input1 + outer_idx*pad_axis_width1;


    if(blockIdx.x == gridDim.x - 1)
    {
        if(start_pos < num_elems)
        {
            uint64_t result = 0;
            #pragma unroll
            for(int i =0 ,shift = 0; i < N; i++, shift += N)
            {
                int index = inner_idx + i;
                if(index >= axis_width0)
                {
                    uint64_t tmp = ptr_input1[index - axis_width0]&0xFF;
                    result |= (tmp<<shift);
                } else {
                    uint64_t tmp = (ptr_input0[index]&0xFF);
                    result |= (tmp<<shift);
                }
            }
            *(uint64_t*)ptr_output = result;
        }
        else
        {
            return;
        }
    }
    else
    {
        uint64_t result = 0;
        #pragma unroll
        for(int i =0 ,shift = 0; i < N; i++, shift += N)
        {
            int index = inner_idx + i;
            if(index >= axis_width0)
            {
                uint64_t tmp = ptr_input1[index - axis_width0]&0xFF;
                result |= (tmp<<shift);
            } else {
                uint64_t tmp = ptr_input0[index]&0xFF;
                result |= (tmp<<shift);
            }
        }
        *(uint64_t*)ptr_output = result;
    }
}

template <typename T,int N,int size_T>
__global__ void ppl_cukernel_concat_nhwc_axis1_optN(
    int64_t num_elems,
    int axis_offset,
    int input_padded_stride,
    int output_padded_stride,
    DivModFast input_stride_fast,
    const T* input,
    T* output)
{
    int64_t start_pos = (blockIdx.x * blockDim.x + threadIdx.x)*N;
    if(start_pos < num_elems)
    {
        int outer_idx, inner_idx;
        input_stride_fast.divmod(start_pos,outer_idx,inner_idx);
        const T* ptr_input = input + outer_idx * input_padded_stride + inner_idx;
        T* ptr_output = output + outer_idx*output_padded_stride + axis_offset + inner_idx;
        if constexpr(N == 16) {
            float4 tmp = *(float4*)ptr_input;
            *(float4*)ptr_output = tmp;
        }
        else 
        if constexpr(N==8) {
            uint64_t tmp = *(uint64_t*)ptr_input;
            *(uint64_t*)ptr_output = tmp;
        }
        else
        if constexpr(N==4)
        {
            if constexpr(size_T==1)
            {
                uint32_t tmp = *(uint32_t*)ptr_input;
                *(uint32_t*)ptr_output = tmp;
            }
            else
            if constexpr(size_T==2)
            {
                uint64_t tmp = *(uint64_t*)ptr_input;
                *(uint64_t*)ptr_output = tmp;
            }
        }
        else
        if constexpr(N==2)
        {
            if constexpr(size_T==1)
            {
                uint16_t tmp = *(uint16_t*)ptr_input;
                *(uint16_t*)ptr_output = tmp;
            }
            else
            if constexpr(size_T==2)
            {
                uint32_t tmp = *(uint32_t*)ptr_input;
                *(uint32_t*)ptr_output = tmp;
            }
            else
            if constexpr(size_T==4)
            {
                uint64_t tmp = *(uint64_t*)ptr_input;
                *(uint64_t*)ptr_output = tmp;
            }
        }else{
            T tmp = ptr_input[0];
            ptr_output[0] = tmp;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_concat_nhwc16_nopadding_two_input_int8_opt(
    int64_t num_elems,
    const T* input0,
    const T* input1,
    int64_t input0_axis_width,
    int64_t input1_axis_width,
    DivModFast num_elems_inner_fast,
    T* output)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    int outer_idx, inner_idx;
    num_elems_inner_fast.divmod(idx, outer_idx, inner_idx);

    if (inner_idx >= input0_axis_width) {
        output[idx] = input1[outer_idx * input1_axis_width + inner_idx - input0_axis_width];
    } else {
        output[idx] = input0[outer_idx * input0_axis_width + inner_idx];
    }
}

template <typename T>
__global__ void ppl_cukernel_concat_nhwc16_nopadding_three_input_int8_opt(
    int64_t num_elems,
    const T* input0,
    const T* input1,
    const T* input2,
    int64_t input0_axis_width,
    int64_t input1_axis_width,
    int64_t input2_axis_width,
    int64_t output_axis_width,
    DivModFast num_elems_inner_fast,
    T* output)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    int outer_idx, inner_idx;
    num_elems_inner_fast.divmod(idx, outer_idx, inner_idx);

    if (inner_idx >= (input0_axis_width + input1_axis_width)) {
        output[idx] = input2[outer_idx * input2_axis_width + inner_idx - (input0_axis_width + input1_axis_width)];
    } else if (inner_idx >= (input0_axis_width)) {
        output[idx] = input1[outer_idx * input1_axis_width + inner_idx - (input0_axis_width)];
    } else {
        output[idx] = input0[outer_idx * input0_axis_width + inner_idx];
    }
}

template <typename T>
__global__ void ppl_cukernel_concat_nhwc16_nopadding_four_input_int8_opt(
    int64_t num_elems,
    const T* input0,
    const T* input1,
    const T* input2,
    const T* input3,
    int64_t input0_axis_width,
    int64_t input1_axis_width,
    int64_t input2_axis_width,
    int64_t input3_axis_width,
    int64_t output_axis_width,
    DivModFast num_elems_inner_fast,
    T* output)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    int outer_idx, inner_idx;
    num_elems_inner_fast.divmod(idx, outer_idx, inner_idx);

    float4 val;
    if (inner_idx >= (input0_axis_width + input1_axis_width + input2_axis_width)) {
        val = input3[outer_idx * input3_axis_width + inner_idx - (input0_axis_width + input1_axis_width + input2_axis_width)];
    } else if (inner_idx >= (input0_axis_width + input1_axis_width)) {
        val = input2[outer_idx * input2_axis_width + inner_idx - (input0_axis_width + input1_axis_width)];
    } else if (inner_idx >= (input0_axis_width)) {
        val = input1[outer_idx * input1_axis_width + inner_idx - (input0_axis_width)];
    } else {
        val = input0[outer_idx * input0_axis_width + inner_idx];
    }
    output[idx] = val;
}

template <typename T>
__global__ void ppl_cukernel_concat_nd_nopadding_two_input_opt(
    int64_t num_elems,
    const T* input0,
    const T* input1,
    int64_t input0_axis_width,
    int64_t input1_axis_width,
    DivModFast num_elems_inner_fast,
    T* output)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    int outer_idx, inner_idx;
    num_elems_inner_fast.divmod(idx, outer_idx, inner_idx);

    if (inner_idx >= input0_axis_width) {
        output[idx] = input1[outer_idx * input1_axis_width + inner_idx - input0_axis_width];
    } else {
        output[idx] = input0[outer_idx * input0_axis_width + inner_idx];
    }
}

template <typename T>
__global__ void ppl_cukernel_concat_nd_nopadding_three_input_opt(
    int64_t num_elems,
    const T* input0,
    const T* input1,
    const T* input2,
    int64_t input0_axis_width,
    int64_t input1_axis_width,
    int64_t input2_axis_width,
    DivModFast num_elems_inner_fast,
    T* output)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    int outer_idx, inner_idx;
    num_elems_inner_fast.divmod(idx, outer_idx, inner_idx);

    if (inner_idx >= (input0_axis_width + input1_axis_width)) {
        output[idx] = input2[outer_idx * input2_axis_width + inner_idx - (input0_axis_width + input1_axis_width)];
    } else if (inner_idx >= (input0_axis_width)) {
        output[idx] = input1[outer_idx * input1_axis_width + inner_idx - (input0_axis_width)];
    } else {
        output[idx] = input0[outer_idx * input0_axis_width + inner_idx];
    }
}

template <typename T>
__global__ void ppl_cukernel_concat_nd_nopadding_four_input_opt(
    int64_t num_elems,
    const T* input0,
    const T* input1,
    const T* input2,
    const T* input3,
    int64_t input0_axis_width,
    int64_t input1_axis_width,
    int64_t input2_axis_width,
    int64_t input3_axis_width,
    DivModFast num_elems_inner_fast,
    T* output)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    int outer_idx, inner_idx;
    num_elems_inner_fast.divmod(idx, outer_idx, inner_idx);

    if (inner_idx >= (input0_axis_width + input1_axis_width + input2_axis_width)) {
        output[idx] = input3[outer_idx * input3_axis_width + inner_idx - (input0_axis_width + input1_axis_width + input2_axis_width)];
    } else if (inner_idx >= (input0_axis_width + input1_axis_width)) {
        output[idx] = input2[outer_idx * input2_axis_width + inner_idx - (input0_axis_width + input1_axis_width)];
    } else if (inner_idx >= (input0_axis_width)) {
        output[idx] = input1[outer_idx * input1_axis_width + inner_idx - (input0_axis_width)];
    } else {
        output[idx] = input0[outer_idx * input0_axis_width + inner_idx];
    }
}

constexpr int CONCAT_BATCH_SIZE = 128;
template <typename T, int n, int stride_size>
struct ConcatTensorMetadata {
    const T* input[n];
    int64_t axis_width[n];
    int64_t axis_stride[n];
    int64_t axis_width_reduce_sum[n];
    // int64_t nElements[n];
    // bool isContiguous[n];
    DivModFast modFast[n];
};

template <typename T>
__global__ void ppl_cukernel_concat_nhwc_nopadding_multi_input_opt(
    int64_t num_elems,
    ConcatTensorMetadata<T, CONCAT_BATCH_SIZE, 1> inputs,
    int num_input,
    DivModFast num_elems_inner_fast,
    T* output)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    int outer_idx, inner_idx;
    num_elems_inner_fast.divmod(idx, outer_idx, inner_idx);

    for (int i = 0; i < num_input; i++) {
        if (inner_idx >= inputs.axis_width_reduce_sum[i] && inner_idx < (inputs.axis_width_reduce_sum[i] + inputs.axis_width[i])) {
            int input_offset = outer_idx * inputs.axis_width[i] + inner_idx - inputs.axis_width_reduce_sum[i];
            output[idx] = inputs.input[i][input_offset];
            return;
        }
    }
}

template<typename T, int NUM_INPUTS, typename DST_T, int N, int SHIFT, int SM_SIZE>
__global__ void ppl_cukernel_concat_sm_multi_input_opt(ConcatTensorMetadata<void, NUM_INPUTS, 1> inputs, T* output,int64_t num_elems, int output_axis_stride,
                                                    int num_inputs,DivModFast output_stride_fast,int line, int num_concats)
{
    __shared__ T sm_input[SM_SIZE];
    __shared__ T sm_output[SM_SIZE];
    T* ptr_sm_input = sm_input;
    T* ptr_sm_output = sm_output;
    int offset = blockIdx.x * line;
    int height = min(num_concats - offset, line);
    if(height <= 0) return;
    T * ptr_block_output = output + offset * output_axis_stride;
    if(blockIdx.x == gridDim.x - 1) {
        for(int i = 0; i < num_inputs; i++)
        {
            int iblock_size = height* inputs.axis_width[i];
            int ioffset = offset * inputs.axis_width[i];
            const T* ptr_input = (const T*)inputs.input[i] + ioffset;
            for(int j = threadIdx.x; j < iblock_size; j += blockDim.x) {
                ptr_sm_input[j] = ptr_input[j];
            }
            __syncthreads();

            for(int j = threadIdx.x; j < iblock_size; j += blockDim.x) {
                int outer_idx, inner_idx;
                int top_idx;
                inputs.modFast[i].divmod(j, outer_idx, inner_idx);
                top_idx = inner_idx + outer_idx * output_axis_stride + inputs.axis_width_reduce_sum[i];
                ptr_sm_output[top_idx] = ptr_sm_input[j];
            }
            ptr_sm_input += iblock_size;
        }
        __syncthreads();
        int out_block_size = height * output_axis_stride;
        for(int i = threadIdx.x; i < out_block_size; i += blockDim.x) {
            ptr_block_output[i] = sm_output[i];
        }
    } else {
        for(int i = 0; i < num_inputs; i++)
        {
            int iblock_size = height* inputs.axis_width[i];
            int iblock_sizeN = iblock_size >> SHIFT;
            int ioffset = offset * inputs.axis_width[i];
            const T* ptr_input = (const T*)inputs.input[i] + ioffset;
            for(int j = threadIdx.x; j < iblock_sizeN; j += blockDim.x) {
                *(DST_T*)(ptr_sm_input + (j << SHIFT)) = *(DST_T*)(ptr_input + (j << SHIFT));
            }
            __syncthreads();

            for(int j = threadIdx.x; j < iblock_size; j += blockDim.x) {
                int outer_idx, inner_idx;
                int top_idx;
                inputs.modFast[i].divmod(j, outer_idx, inner_idx);
                top_idx = inner_idx + outer_idx * output_axis_stride + inputs.axis_width_reduce_sum[i];
                ptr_sm_output[top_idx] = ptr_sm_input[j];
            }
            ptr_sm_input += iblock_size;
        }
        __syncthreads();
        int out_block_size = height * output_axis_stride;
        int out_block_sizeN = out_block_size >> SHIFT;
        for(int i = threadIdx.x; i < out_block_sizeN; i += blockDim.x) {
            *(DST_T*)(ptr_block_output + (i << SHIFT)) = *(DST_T*)(sm_output + (i <<SHIFT));
        }
    }
}

template<typename T, int NUM_INPUTS, typename DST_T, int N, int SHIFT>
__global__ void ppl_cukernel_concat_multi_input_opt(ConcatTensorMetadata<void, NUM_INPUTS, 1> inputs, T* output,int64_t num_elems, int concat_size,int num_inputs,DivModFast output_stride_fast)
{
    int64_t block_offset = blockIdx.x * blockDim.x << SHIFT;
    int block_size = min(num_elems - block_offset, blockDim.x << SHIFT);
    if(block_size <= 0) return;
    T * ptr_output = output + block_offset;
    for(int i = threadIdx.x; i < block_size; i += blockDim.x) {
        int64_t offset = block_offset + i;
        int outer_idx, inner_idx;
        int j = 0;
        output_stride_fast.divmod(offset, outer_idx, inner_idx);
        for(j = 0; j < num_inputs; j++) {
            if(inner_idx >= inputs.axis_width_reduce_sum[j] && inner_idx < inputs.axis_width_reduce_sum[j] + inputs.axis_width[j]) {
                break;
            }
        }
        T *ptr_input = (T*)inputs.input[j];
        T tmp = ptr_input[outer_idx * inputs.axis_width[j] + inner_idx - inputs.axis_width_reduce_sum[j]];
        ptr_output[i] = tmp;
    }
}

template<typename T, int NUM_INPUTS, typename DST_T, int N, int SHIFT>
__global__ void ppl_cukernel_concat_sm_multi_input_padding_opt(ConcatTensorMetadata<void, NUM_INPUTS, 1> inputs, void* output, int output_axis_stride,
                                                    int num_inputs,DivModFast output_stride_fast,int line, int num_concats)
{
    __shared__ int8_t sm_buffer[16384];
    T* sm_input = (T*)sm_buffer;
    T* sm_output = (T *)(sm_buffer + 8192);
    T * ptr_sm_input = sm_input, *ptr_sm_output = sm_output;
    int offset = blockIdx.x * line;
    int height = min(num_concats - offset, line);
    if(height <= 0) return;
    T * ptr_block_output = (T*)output + offset * output_axis_stride;
    for(int i = 0; i < num_inputs; i++) {
        int iblock_size = height * inputs.axis_stride[i];
        int ioffset = offset * inputs.axis_stride[i];
        int iblock_sizeN = iblock_size >> SHIFT;
        const T* ptr_input = (const T*)inputs.input[i] + ioffset;
        for(int j = threadIdx.x; j < iblock_sizeN; j += blockDim.x) {
            *(DST_T*)(ptr_sm_input + (j << SHIFT)) = *(DST_T*)(ptr_input + (j << SHIFT));
        }
        __syncthreads();

        for(int j = threadIdx.x; j < iblock_size; j += blockDim.x) {
            int outer_idx, inner_idx;
            int top_idx;
            inputs.modFast[i].divmod(j, outer_idx, inner_idx);
            if(inner_idx < inputs.axis_width[i])
            {
                top_idx = inner_idx + outer_idx * output_axis_stride + inputs.axis_width_reduce_sum[i];
                ptr_sm_output[top_idx] = ptr_sm_input[j];
            }
        }
        ptr_sm_input += iblock_size;
    }

    __syncthreads();
    int out_block_size = height * output_axis_stride;
    int out_block_sizeN = out_block_size >> SHIFT;
    for(int i = threadIdx.x; i < out_block_sizeN; i += blockDim.x) {
        *(DST_T*)(ptr_block_output + (i << SHIFT)) = *(DST_T*)(sm_output + (i <<SHIFT));
    }
}

template<typename T, int NUM_INPUTS, int SHIFT>
__global__ void ppl_cukernel_concat_multi_input_padding_opt(ConcatTensorMetadata<void, NUM_INPUTS, 1> inputs, void* output,int64_t num_elems, int num_inputs,DivModFast output_stride_fast)
{
    int64_t block_offset = blockIdx.x * blockDim.x << SHIFT;
    int block_size = min(num_elems - block_offset, blockDim.x << SHIFT);
    if(block_size <= 0) return;
    T * ptr_output = (T*)output + block_offset;
    for(int i = threadIdx.x; i < block_size; i += blockDim.x) {
        int64_t offset = block_offset + i;
        int outer_idx, inner_idx;
        int j = 0;
        output_stride_fast.divmod(offset, outer_idx, inner_idx);
        for(j = 0; j < num_inputs; j++) {
            if(inner_idx >= inputs.axis_width_reduce_sum[j] && inner_idx < inputs.axis_width_reduce_sum[j] + inputs.axis_width[j]) {
                break;
            }
        }
        if(j < num_inputs) {
            T *ptr_input = (T*)inputs.input[j];
            T tmp = ptr_input[outer_idx * inputs.axis_stride[j] + inner_idx - inputs.axis_width_reduce_sum[j]];
            ptr_output[i] = tmp;
        }
    }
}

#endif//__MACACC__

template <typename T1, typename T2>
__global__ void __launch_bounds__(256) ppl_cukernel_concat_two_inputs(
    int64_t num_elems,
    const T1* input0,
    const T1* input1,
    T2* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int tid = threadIdx.x;
        __shared__ T1 buffer[2 * 256];
        buffer[2 * tid]     = input0[i];
        buffer[2 * tid + 1] = input1[i];
        T2* buffer_ptr      = reinterpret_cast<T2*>(buffer);
        output[i]           = buffer_ptr[tid];
    }
}

template <typename T>
__global__ void ppl_cukernel_concat_nhwc(
    int64_t num_elems,
    int num_dims,
    int nhwc_axis,
    int axis_offset,
    GArray<DivModFast> input_strides_fast,
    GArray<int64_t> input_padded_strides,
    GArray<int64_t> output_padded_strides,
    const T* input,
    T* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int64_t output_offset = 0, input_offset = 0;
    int idx, remain                         = index;
    for (int it = 0; it < num_dims; ++it) {
        input_strides_fast[it].divmod(remain, idx, remain);
        input_offset += idx * input_padded_strides[it];
        idx = (it == nhwc_axis) ? idx + axis_offset : idx;
        output_offset += idx * output_padded_strides[it];
    }
    output[output_offset] = input[input_offset];
}

template <typename T1, typename T2>
__global__ void __launch_bounds__(256) ppl_cukernel_concat_nhwc_two_inputs(
    int64_t num_elems,
    int inner_dims,
    int axis_width0,
    int axis_width1,
    const T1* input0,
    const T1* input1,
    T2* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int inner_idx = i % inner_dims;
        int outer_idx = i / inner_dims;
        if (inner_idx >= axis_width0) {
            int input_offset = outer_idx * axis_width1 + (inner_idx - axis_width0);
            output[i]        = input1[input_offset];
        } else {
            int input_offset = outer_idx * axis_width0 + inner_idx;
            output[i]        = input0[input_offset];
        }
    }
}

template <typename T1, typename T2>
__global__ void __launch_bounds__(256) ppl_cukernel_concat_nhwc_two_inputs(
    int64_t num_elems,
    int inner_dims,
    int pad_inner_dims,
    int axis_width0,
    int pad_axis_width0,
    int axis_width1,
    int pad_axis_width1,
    const T1* input0,
    const T1* input1,
    T2* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int inner_idx = i % pad_inner_dims;
        int outer_idx = i / pad_inner_dims;
        // int output_offset = outer_idx * pad_inner_dims + inner_idx;
        if (inner_idx >= axis_width0) {
            int axis_offset  = inner_idx - axis_width0;
            int input_offset = outer_idx * pad_axis_width1 + axis_offset;
            output[i]        = axis_offset >= axis_width1 ? 0 : input1[input_offset];
        } else {
            int axis_offset  = inner_idx;
            int input_offset = outer_idx * pad_axis_width0 + axis_offset;
            output[i]        = axis_offset >= axis_width0 ? 0 : input0[input_offset];
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_concat_nhwc_nopadding(
    int64_t num_elems,
    const T* inputs,
    int64_t concat_size,
    int64_t top_axis_width,
    DivModFast num_elems_inner_fast,
    int axis_offset,
    T* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int outer_idx, inner_idx;
        num_elems_inner_fast.divmod(i, outer_idx, inner_idx);
        int64_t top_idx = inner_idx + (outer_idx * top_axis_width + axis_offset);
        output[top_idx] = inputs[i];
    }
}

bool IsConcatNoPadding(
    int axis,
    int num_inputs,
    int* input_dims[],
    int* input_padded_dims[],
    ppl::common::TensorShape* output_shape,
    int mask)
{
    if ((output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NHWC8 && output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NHWC16 &&
         output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NHWC) || axis != 1)
        return false;
    for (int i = 0; i < num_inputs; i++) {
        if (input_padded_dims[i][axis] - input_dims[i][axis] != 0)
            return false;
    }
    return true;
}

ppl::common::RetCode PPLCUDAConcatNoPaddingForwardImp(
    cudaStream_t stream,
    int axis,
    int num_inputs,
    int* input_dims[],
    int* input_padded_dims[],
    const void* inputs[],
    ppl::common::TensorShape* output_shape,
    void* output,
    int mask)
{
    int64_t num_elems         = output_shape->CalcElementsIncludingPadding() / output_shape->GetDim(axis);
    int64_t output_axis_width = output_shape->GetDim(axis);
    int64_t axis_offset       = 0;
    if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8 && output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
#ifdef __MACACC__
        if (num_inputs == 2) {
            int64_t output_num_elems  = output_shape->CalcElementsIncludingPadding() >> 4;
            int64_t input0_axis_width = input_dims[0][axis] >> 4;
            int64_t input1_axis_width = input_dims[1][axis] >> 4;
            int64_t output_axis_width = input0_axis_width + input1_axis_width;

            DivModFast pad_inner_dims_fast = DivModFast(output_axis_width);

            int block_size = 256;
            int grid_size  = (output_num_elems + block_size - 1) / block_size;
            ppl_cukernel_concat_nhwc16_nopadding_two_input_int8_opt<<<grid_size, block_size, 0, stream>>>(output_num_elems,
                                                                                                (const float4*)inputs[0],
                                                                                                (const float4*)inputs[1],
                                                                                                input0_axis_width,
                                                                                                input1_axis_width,
                                                                                                pad_inner_dims_fast,
                                                                                                (float4*)output);
        } else if (num_inputs == 3) {
            int64_t output_num_elems  = output_shape->CalcElementsIncludingPadding() >> 4;
            int64_t input0_axis_width = input_dims[0][axis] >> 4;
            int64_t input1_axis_width = input_dims[1][axis] >> 4;
            int64_t input2_axis_width = input_dims[2][axis] >> 4;
            int64_t output_axis_width = input0_axis_width + input1_axis_width + input2_axis_width;

            DivModFast pad_inner_dims_fast = DivModFast(output_axis_width);

            int block_size = 256;
            int grid_size  = (output_num_elems + block_size - 1) / block_size;
            ppl_cukernel_concat_nhwc16_nopadding_three_input_int8_opt<<<grid_size, block_size, 0, stream>>>(output_num_elems,
                                                                                                (const float4*)inputs[0],
                                                                                                (const float4*)inputs[1],
                                                                                                (const float4*)inputs[2],
                                                                                                input0_axis_width,
                                                                                                input1_axis_width,
                                                                                                input2_axis_width,
                                                                                                output_axis_width,
                                                                                                pad_inner_dims_fast,
                                                                                                (float4*)output);
        } else if (num_inputs == 4) {
            int64_t output_num_elems  = output_shape->CalcElementsIncludingPadding() >> 4;
            int64_t input0_axis_width = input_dims[0][axis] >> 4;
            int64_t input1_axis_width = input_dims[1][axis] >> 4;
            int64_t input2_axis_width = input_dims[2][axis] >> 4;
            int64_t input3_axis_width = input_dims[3][axis] >> 4;
            int64_t output_axis_width = input0_axis_width + input1_axis_width + input2_axis_width + input3_axis_width;

            DivModFast pad_inner_dims_fast = DivModFast(output_axis_width);

            int block_size = 256;
            int grid_size  = (output_num_elems + block_size - 1) / block_size;
            ppl_cukernel_concat_nhwc16_nopadding_four_input_int8_opt<<<grid_size, block_size, 0, stream>>>(output_num_elems,
                                                                                                (const float4*)inputs[0],
                                                                                                (const float4*)inputs[1],
                                                                                                (const float4*)inputs[2],
                                                                                                (const float4*)inputs[3],
                                                                                                input0_axis_width,
                                                                                                input1_axis_width,
                                                                                                input2_axis_width,
                                                                                                input3_axis_width,
                                                                                                output_axis_width,
                                                                                                pad_inner_dims_fast,
                                                                                                (float4*)output);
        } else if (num_inputs < CONCAT_BATCH_SIZE) {
            int64_t output_num_elems  = output_shape->CalcElementsIncludingPadding() >> 4;
            int block_size = 256;
            int grid_size  = (output_num_elems + block_size - 1) / block_size;

            ConcatTensorMetadata<float4, CONCAT_BATCH_SIZE, 1> catMetaData;
            int64_t axis_reduce_sum = 0;
            for (unsigned i = 0; i < num_inputs; i++) {
                catMetaData.input[i] = (float4*)inputs[i];
                int64_t axis_width = input_dims[i][axis] >> 4;
                catMetaData.axis_width[i] = axis_width;
                catMetaData.axis_width_reduce_sum[i] = axis_reduce_sum;
                axis_reduce_sum += axis_width;
                // catMetaData.nElements[i] = num_elems * axis_width;
            }
            int64_t output_axis_width = axis_reduce_sum;
            DivModFast pad_inner_dims_fast = DivModFast(output_axis_width);
            ppl_cukernel_concat_nhwc_nopadding_multi_input_opt<<<grid_size, block_size, 0, stream>>>(output_num_elems,
                                                                                            catMetaData,
                                                                                            num_inputs,
                                                                                            pad_inner_dims_fast,
                                                                                            (float4*)output);
        } else {
#endif//__MACACC__
            output_axis_width = output_axis_width >> 4;
            for (int j = 0; j < num_inputs; ++j) {
                int input_axis_width = (input_dims[j][axis] >> 4);
                int num_in_elems     = num_elems * input_axis_width;
                if (!(mask & (1 << j))) {
                    if (num_in_elems > 0) {
                        DivModFast num_elems_inner_fast = DivModFast(input_axis_width);
                        int block_size                  = 256;
                        int grid_size                   = (num_in_elems + block_size - 1) / block_size;
                        ppl_cukernel_concat_nhwc_nopadding<<<grid_size, block_size, 0, stream>>>(num_in_elems,
                                                                                                (const float4*)inputs[j],
                                                                                                num_in_elems,
                                                                                                output_axis_width,
                                                                                                num_elems_inner_fast,
                                                                                                axis_offset,
                                                                                                (float4*)output);
                    }
                }
                axis_offset += (input_axis_width);
            }
#ifdef __MACACC__
        }
#endif//__MACACC__
        return ppl::common::RC_SUCCESS;
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16 && output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
#ifdef __MACACC__
    int64_t output_num_elems  = output_shape->CalcElementsIncludingPadding() >> 3;
    int block_size = 256;
    int grid_size  = (output_num_elems + block_size - 1) / block_size;

    ConcatTensorMetadata<float4, CONCAT_BATCH_SIZE, 1> catMetaData;
    int64_t axis_reduce_sum = 0;
    for (unsigned i = 0; i < num_inputs; i++) {
        catMetaData.input[i] = (float4*)inputs[i];
        int64_t axis_width = input_dims[i][axis] >> 3;
        catMetaData.axis_width[i] = axis_width;
        catMetaData.axis_width_reduce_sum[i] = axis_reduce_sum;
        axis_reduce_sum += axis_width;
        // catMetaData.nElements[i] = num_elems * axis_width;
    }
    int64_t output_axis_width = axis_reduce_sum;
    DivModFast pad_inner_dims_fast = DivModFast(output_axis_width);
    ppl_cukernel_concat_nhwc_nopadding_multi_input_opt<<<grid_size, block_size, 0, stream>>>(output_num_elems,
                                                                                    catMetaData,
                                                                                    num_inputs,
                                                                                    pad_inner_dims_fast,
                                                                                    (float4*)output);
#else//__MACACC__
        output_axis_width = output_axis_width >> 3;
        for (int j = 0; j < num_inputs; ++j) {
            int input_axis_width = (input_dims[j][axis] >> 3);
            int num_in_elems     = num_elems * input_axis_width;
            if (!(mask & (1 << j))) {
                if (num_in_elems > 0) {
                    DivModFast num_elems_inner_fast = DivModFast(input_axis_width);
                    int block_size                  = 256;
                    int grid_size                   = (num_in_elems + block_size - 1) / block_size;
                    ppl_cukernel_concat_nhwc_nopadding<<<grid_size, block_size, 0, stream>>>(num_in_elems,
                                                                                             (const float4*)inputs[j],
                                                                                             num_in_elems,
                                                                                             output_axis_width,
                                                                                             num_elems_inner_fast,
                                                                                             axis_offset,
                                                                                             (float4*)output);
                }
            }
            axis_offset += (input_axis_width);
        }
#endif//__MACACC__
        return ppl::common::RC_SUCCESS;
    }
#ifdef __CONCAT_OPT__
#define SWITCH_CASE(TYPE)                                                                                               \
    case sizeof(TYPE):{                                                                                                 \
        for(int j = 0; j < num_inputs; ++j) {                                                                           \
            int input_axis_width = input_dims[j][axis];                                                                 \
            int num_in_elems = num_elems * input_axis_width;                                                            \
            if (!(mask & (1 << j))) {                                                                                   \
                if(num_in_elems > 0) {                                                                                  \
                    DivModFast num_elems_inner_fast = DivModFast(input_axis_width);                                     \
                    const int N = 8 / sizeof(TYPE);                                                                     \
                    int block_size = 256;                                                                               \
                    int num_per_block = block_size*N;                                                                   \
                    int grid_size                   = (num_in_elems + num_per_block - 1) / num_per_block;               \
                    if((input_axis_width % N) == 0 && (output_axis_width % N) == 0)                                     \
                    {                                                                                                   \
                        ppl_cukernel_concat_nhwc_nopadding_optN<TYPE,N,1><<<grid_size, block_size, 0, stream>>>(num_in_elems, \
                                                                                             (const TYPE*)inputs[j],    \
                                                                                             input_axis_width,          \
                                                                                             output_axis_width,         \
                                                                                             num_elems_inner_fast,      \
                                                                                             axis_offset,               \
                                                                                             (TYPE*)output);            \
                    }                                                                                                   \
                    else                                                                                                \
                    {                                                                                                   \
                        ppl_cukernel_concat_nhwc_nopadding_optN<TYPE,N,0><<<grid_size, block_size, 0, stream>>>(num_in_elems, \
                                                                                             (const TYPE*)inputs[j],    \
                                                                                             input_axis_width,          \
                                                                                             output_axis_width,         \
                                                                                             num_elems_inner_fast,      \
                                                                                             axis_offset,               \
                                                                                             (TYPE*)output);            \
                    }                                                                                                   \
                    axis_offset += input_axis_width;                                                                    \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
        return ppl::common::RC_SUCCESS;                                                                                 \
    }
#else//!__CONCAT_OPT__
#define SWITCH_CASE(TYPE)                                                                                            \
    case sizeof(TYPE): {                                                                                             \
        for (int j = 0; j < num_inputs; ++j) {                                                                       \
            int input_axis_width = input_dims[j][axis];                                                              \
            int num_in_elems     = num_elems * input_axis_width;                                                     \
            if (!(mask & (1 << j))) {                                                                                \
                if (num_in_elems > 0) {                                                                              \
                    DivModFast num_elems_inner_fast = DivModFast(input_axis_width);                                  \
                    int block_size                  = 256;                                                           \
                    int grid_size                   = (num_in_elems + block_size - 1) / block_size;                  \
                    ppl_cukernel_concat_nhwc_nopadding<<<grid_size, block_size, 0, stream>>>(num_in_elems,           \
                                                                                             (const TYPE*)inputs[j], \
                                                                                             num_in_elems,           \
                                                                                             output_axis_width,      \
                                                                                             num_elems_inner_fast,   \
                                                                                             axis_offset,            \
                                                                                             (TYPE*)output);         \
                }                                                                                                    \
            }                                                                                                        \
            axis_offset += input_axis_width;                                                                         \
        }                                                                                                            \
        return ppl::common::RC_SUCCESS;                                                                              \
    }
#endif//__CONCAT_OPT__
    switch (ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
        SWITCH_CASE(int8_t);
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#undef SWITCH_CASE
}

bool IsFusedBridge(
    int axis,
    int num_inputs,
    ppl::common::TensorShape* output_shape)
{
    if ((output_shape->GetPadding0(axis) + output_shape->GetPadding1(axis) > 0) 
        && output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 
        && output_shape->GetDimCount() == 2)
        return true;
    else
        return false;
}   



ppl::common::RetCode PPLCUDAConcatBridgeNDToNHWC8ForwardImp(
    cudaStream_t stream,
    int axis,
    int num_inputs,
    int* input_dims[],
    int* input_padded_dims[],
    const void* inputs[],
    ppl::common::TensorShape* output_shape,
    void* output,
    int mask)
{

    int64_t concat_size = 1;
    int64_t num_concats = 1;
    int concat_num_dims = output_shape->GetDimCount();
    for (int i = concat_num_dims - 1; i > axis; --i)
        concat_size *= output_shape->GetDim(i);
    for (int i = 0; i < axis; ++i)
        num_concats *= output_shape->GetDim(i);
        
    int output_axis_width = output_shape->GetDim(axis) + output_shape->GetPadding0(axis) + output_shape->GetPadding1(axis);

    int out_stride = output_axis_width * concat_size;   // 848
    int64_t out_size = out_stride * num_concats;

    ConcatTensorMetadata<void, 32, 1> inputs_meta;
    int64_t axis_reduce_sum = 0;
    for(int i = 0 ; i < num_inputs; i++) {
        inputs_meta.input[i] = inputs[i];
        inputs_meta.axis_width[i] = input_dims[i][axis] * concat_size;
        inputs_meta.axis_width_reduce_sum[i] = axis_reduce_sum;
        inputs_meta.modFast[i] = DivModFast(inputs_meta.axis_width[i]);
        axis_reduce_sum += inputs_meta.axis_width[i];
    }

    switch(ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
        case 1:{
            int line = 8192 / out_stride;
            if(line >= 1) {
                int shift = 0;
                while((line >> shift)) {
                    shift++;
                    if(shift == 4){
                        break;
                    }
                }
                if((line >> shift) == 0) shift--;
                line = line >> shift << shift;
                int gridSize = (num_concats + line - 1) / line;
                int blockSize = 512;
                if(shift == 4) {
                    ppl_cukernel_concat_sm_multi_input_opt<int8_t, 32, float4, 16, 4, 8192><<<gridSize, blockSize, 0, stream>>>(
                        inputs_meta,(int8_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else if(shift == 3) {
                    ppl_cukernel_concat_sm_multi_input_opt<int8_t, 32, float2, 8, 3, 8192><<<gridSize, blockSize, 0, stream>>>(
                        inputs_meta,(int8_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else if(shift == 2) {
                    ppl_cukernel_concat_sm_multi_input_opt<int8_t, 32, float, 4, 2, 8192><<<gridSize, blockSize, 0, stream>>>(
                        inputs_meta,(int8_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else if(shift == 1) {
                    ppl_cukernel_concat_sm_multi_input_opt<int8_t, 32, short, 2, 1, 8192><<<gridSize, blockSize, 0, stream>>>(
                        inputs_meta,(int8_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else {
                    ppl_cukernel_concat_sm_multi_input_opt<int8_t, 32, int8_t, 1, 0, 8192><<<gridSize, blockSize, 0, stream>>>(
                        inputs_meta,(int8_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                }
            } else {
                int blockSize = 512;
                int gridSize = (out_size + 4095) >> 12;
                ppl_cukernel_concat_multi_input_opt<int8_t, 32, float2, 8, 3><<<gridSize, blockSize, 0, stream>>>(
                    inputs_meta, (int8_t*)output, out_size, concat_size, num_inputs, DivModFast(out_stride));
            }  
            break;
        } 
        case 2:{
            int line = 4096 / (out_stride);
            if(line >= 1) {
                int shift = 0;
                while((line >> shift)) {
                    shift++;
                    if(shift == 3){
                        break;
                    }
                }
                if((line >> shift) == 0) shift--;
                line = line >> shift << shift;
                int gridSize = (num_concats + line - 1) / line;
                int blockSize = 512;
                if(shift == 3) {
                    ppl_cukernel_concat_sm_multi_input_opt<short, 32, float4, 8, 3, 4096><<<gridSize, blockSize, 0, stream>>>(
                            inputs_meta, (short*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else if(shift == 2) {
                    ppl_cukernel_concat_sm_multi_input_opt<short, 32, float2, 4, 2, 4096><<<gridSize, blockSize, 0, stream>>>(
                            inputs_meta, (short*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else if(shift == 1) {
                    ppl_cukernel_concat_sm_multi_input_opt<short, 32, float, 2, 1, 4096><<<gridSize, blockSize, 0, stream>>>(
                            inputs_meta, (short*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else {
                    ppl_cukernel_concat_sm_multi_input_opt<short, 32, short, 1, 0, 4096><<<gridSize, blockSize, 0, stream>>>(
                            inputs_meta, (short*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                }
            } else {
                return ppl::common::RC_UNSUPPORTED;
            }
            break;
        }
        case 4:{
            int line = 4096 / (out_stride);
            if(line >= 1) {
                int gridSize = (num_concats + line - 1) / line;
                int blockSize = 512;
                ppl_cukernel_concat_sm_multi_input_opt<int32_t, 32, int32_t, 1, 0, 4096><<<gridSize, blockSize, 0, stream>>>(
                    inputs_meta, (int32_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
            } else {
                int blockSize = 512;
                int gridSize = (out_size + 2047) >> 11;
                ppl_cukernel_concat_multi_input_opt<int32_t, 32, float4, 4, 2><<<gridSize, blockSize, 0, stream>>>(
                    inputs_meta, (int32_t*)output, out_size, concat_size, num_inputs, DivModFast(out_stride));
            }
            break; 
        }
        case 8:{
            int line = 2048 / (out_stride);
            if(line >= 1) {
                int gridSize = (num_concats + line - 1) / line;
                int blockSize = 512;
                ppl_cukernel_concat_sm_multi_input_opt<int64_t, 32, int64_t, 1, 0, 2048><<<gridSize, blockSize, 0, stream>>>(
                    inputs_meta, (int64_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
            } else {
                int blockSize = 512;
                int gridSize = (out_size + 1023) >> 10;
                ppl_cukernel_concat_multi_input_opt<int64_t, 32, float4, 2, 1><<<gridSize, blockSize, 0, stream>>>(
                    inputs_meta, (int64_t*)output, out_size, concat_size,num_inputs, DivModFast(out_stride));
            }
            break; 
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAConcatForwardImp(
    cudaStream_t stream,
    int axis,
    int num_inputs,
    int* input_dims[],
    int* input_padded_dims[],
    const void* inputs[],
    ppl::common::TensorShape* output_shape,
    void* output,
    int mask)
{
    if (IsFusedBridge(axis, num_inputs,output_shape)) {
        return PPLCUDAConcatBridgeNDToNHWC8ForwardImp(stream, axis, num_inputs, input_dims, input_padded_dims, inputs, output_shape, output, mask);
    }
    if (IsConcatNoPadding(axis, num_inputs, input_dims, input_padded_dims, output_shape, mask)) {
        return PPLCUDAConcatNoPaddingForwardImp(stream, axis, num_inputs, input_dims, input_padded_dims, inputs, output_shape, output, mask);
    }
    int num_dims     = output_shape->GetDimCount();
    int output_elems = output_shape->CalcElementsIncludingPadding();
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        if (num_inputs == 2 && axis == (num_dims - 1) && input_dims[0][axis] == 1 && input_dims[1][axis] == 1) {
            int num_elems = 1;
            for (int it = 0; it < num_dims; num_elems *= input_dims[0][it], ++it)
                ;

#define SWITCH_CASE(TYPE1, TYPE2)                                                                     \
    case sizeof(TYPE1): {                                                                             \
        int block_size = 256;                                                                         \
        int grid_size  = (num_elems + block_size - 1) / block_size;                                   \
        ppl_cukernel_concat_two_inputs<<<grid_size, block_size, 0, stream>>>(num_elems,               \
                                                                             (const TYPE1*)inputs[0], \
                                                                             (const TYPE1*)inputs[1], \
                                                                             (TYPE2*)output);         \
        return ppl::common::RC_SUCCESS;                                                               \
    }
            switch (ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
                SWITCH_CASE(int8_t, int16_t);
                SWITCH_CASE(int16_t, int32_t);
                SWITCH_CASE(int32_t, int64_t);
                SWITCH_CASE(int64_t, float4);
                default:
                    return ppl::common::RC_UNSUPPORTED;
            }
#undef SWITCH_CASE
        } else {
            int64_t concat_size = 1;
            int64_t num_concats = 1;
            for (int i = num_dims - 1; i > axis; --i)
                concat_size *= input_dims[0][i];
            for (int i = 0; i < axis; ++i)
                num_concats *= input_dims[0][i];
            int axis_offset       = 0;
            int output_axis_width = output_shape->GetDim(axis);

#ifdef __MACACC__
        {
#define ND_OPT_TWO_INPUT(TYPE, N)                                                                                   \
    scale = N / data_size;                                                                                          \
    DivModFast pad_inner_dims_fast = DivModFast(output0_axis_width / scale);                                         \
    int grid_size  = (output_num_elems / scale + block_size - 1) / block_size;                                      \
    ppl_cukernel_concat_nd_nopadding_two_input_opt<<<grid_size, block_size, 0, stream>>>(output_num_elems / scale,  \
                                                                                        (const TYPE*)inputs[0],     \
                                                                                        (const TYPE*)inputs[1],     \
                                                                                        input0_axis_width / scale,  \
                                                                                        input1_axis_width / scale,  \
                                                                                        pad_inner_dims_fast,        \
                                                                                        (TYPE*)output);             \
    return ppl::common::RC_SUCCESS;                                                                                 \

#define ND_OPT_THREE_INPUT(TYPE, N)                                                                                 \
    scale = N / data_size;                                                                                          \
    DivModFast pad_inner_dims_fast = DivModFast(output0_axis_width / scale);                                         \
    int grid_size  = (output_num_elems / scale + block_size - 1) / block_size;                                      \
    ppl_cukernel_concat_nd_nopadding_three_input_opt<<<grid_size, block_size, 0, stream>>>(output_num_elems / scale,\
                                                                                        (const TYPE*)inputs[0],     \
                                                                                        (const TYPE*)inputs[1],     \
                                                                                        (const TYPE*)inputs[2],     \
                                                                                        input0_axis_width / scale,  \
                                                                                        input1_axis_width / scale,  \
                                                                                        input2_axis_width / scale,  \
                                                                                        pad_inner_dims_fast,        \
                                                                                        (TYPE*)output);             \
    return ppl::common::RC_SUCCESS;                                                                                 \

#define ND_OPT_FOUR_INPUT(TYPE, N)                                                                                  \
    scale = N / data_size;                                                                                          \
    DivModFast pad_inner_dims_fast = DivModFast(output0_axis_width / scale);                                         \
    int grid_size  = (output_num_elems / scale + block_size - 1) / block_size;                                      \
    ppl_cukernel_concat_nd_nopadding_four_input_opt<<<grid_size, block_size, 0, stream>>>(output_num_elems / scale, \
                                                                                        (const TYPE*)inputs[0],     \
                                                                                        (const TYPE*)inputs[1],     \
                                                                                        (const TYPE*)inputs[2],     \
                                                                                        (const TYPE*)inputs[3],     \
                                                                                        input0_axis_width / scale,  \
                                                                                        input1_axis_width / scale,  \
                                                                                        input2_axis_width / scale,  \
                                                                                        input3_axis_width / scale,  \
                                                                                        pad_inner_dims_fast,        \
                                                                                        (TYPE*)output);             \
    return ppl::common::RC_SUCCESS;                                                                                 \

            int block_size = 512;
            int scale = 0;
            int data_size = ppl::common::GetSizeOfDataType(output_shape->GetDataType());
            int64_t output0_axis_width = output_axis_width * concat_size;
            int64_t output_num_elems  = output_shape->CalcElementsIncludingPadding();
            int64_t input0_axis_width = input_dims[0][axis] * concat_size;
            int64_t input1_axis_width = input_dims[1][axis] * concat_size;

            if (is_aligned_axis(axis, num_inputs, input_dims, input_padded_dims, output_shape, 16)) {
                if (num_inputs == 2) {
                    ND_OPT_TWO_INPUT(float4, 16);
                } else if (num_inputs == 3) {
                    int64_t input2_axis_width = input_dims[2][axis] * concat_size;
                    ND_OPT_THREE_INPUT(float4, 16);
                } else if (num_inputs == 4) {
                    int64_t input2_axis_width = input_dims[2][axis] * concat_size;
                    int64_t input3_axis_width = input_dims[3][axis] * concat_size;
                    ND_OPT_FOUR_INPUT(float4, 16);
                }
            } else if (is_aligned_axis(axis, num_inputs, input_dims, input_padded_dims, output_shape, 8)) {
                if (num_inputs == 2) {
                    ND_OPT_TWO_INPUT(int64_t, 8);
                } else if (num_inputs == 3) {
                    int64_t input2_axis_width = input_dims[2][axis] * concat_size;
                    ND_OPT_THREE_INPUT(int64_t, 8);
                } else if (num_inputs == 4) {
                    int64_t input2_axis_width = input_dims[2][axis] * concat_size;
                    int64_t input3_axis_width = input_dims[3][axis] * concat_size;
                    ND_OPT_FOUR_INPUT(int64_t, 8);
                }
            } else if (is_aligned_axis(axis, num_inputs, input_dims, input_padded_dims, output_shape, 4)) {
                if (num_inputs == 2) {
                    ND_OPT_TWO_INPUT(int32_t, 4);
                } else if (num_inputs == 3) {
                    int64_t input2_axis_width = input_dims[2][axis] * concat_size;
                    ND_OPT_THREE_INPUT(int32_t, 4);
                } else if (num_inputs == 4) {
                    int64_t input2_axis_width = input_dims[2][axis] * concat_size;
                    int64_t input3_axis_width = input_dims[3][axis] * concat_size;
                    ND_OPT_FOUR_INPUT(int32_t, 4);
                }
            }
        }
#define CALL_CONCAT_OPT(TYPE,N) \
    ppl_cukernel_concat_opt_##N<<<grid_size, block_size, 0, stream>>>(num_elems,                \
                                                                        (const TYPE*)inputs[j], \
                                                                        concat_size,            \
                                                                        output_axis_width,      \
                                                                        num_elems_inner_fast,   \
                                                                        axis_offset,            \
                                                                        (TYPE*)output);         \

#define CONTACT_OPT(TYPE, input_concat_size) \
        if (num_elems % (16/sizeof(TYPE)) == 0                                                  \
		&& input_concat_size % (16/sizeof(TYPE)) == 0                                   \
		&& axis_offset*concat_size % (16/sizeof(TYPE)) == 0) {                          \
            int block_capacity              = block_size * 16 / sizeof(TYPE);                   \
            int grid_size                   = (num_elems + block_capacity - 1) / block_capacity;\
            CALL_CONCAT_OPT(TYPE,16)                                                            \
        } else if (sizeof(TYPE) <= 4                                                            \
		&& input_concat_size % (8/sizeof(TYPE)) == 0                                    \
		&& axis_offset*concat_size % (8/sizeof(TYPE)) == 0                              \
        && output_axis_width % (8/sizeof(TYPE)) == 0) {                           \
            int block_capacity              = block_size * 8 / sizeof(TYPE);                    \
            int grid_size                   = (num_elems + block_capacity - 1) / block_capacity;\
            CALL_CONCAT_OPT(TYPE,8)                                                             \
        } else {                                                                                \
            int grid_size                   = (num_elems + block_size - 1) / block_size;        \
            CALL_CONCAT_OPT(TYPE,1)                                                             \
        }

#define SWITCH_CASE(TYPE)                                                                             \
    case sizeof(TYPE): {                                                                              \
        if (num_inputs == 2){                                                                                               \
            int block_size = 256;                                                                                           \
            int scale = sizeof(float4) / sizeof(TYPE);                                                                   \
            int64_t output0_axis_width = output_axis_width * concat_size;                                                   \
            int64_t input0_axis_width = input_dims[0][axis] * concat_size;                                                  \
            int64_t input1_axis_width = input_dims[1][axis] * concat_size;                                                  \
            DivModFast out_inner_dims_fast = DivModFast(output0_axis_width);                                                \
            int64_t output_num_elems  = output_shape->CalcElementsIncludingPadding();                                       \
            int grid_size  = DivUp(output_num_elems, block_size * scale);                                                   \
            ppl_cukernel_concat_nd_padding_two_input_opt<<<grid_size, block_size, 0, stream>>>(output_num_elems, scale,     \
                                                                                                (const TYPE*)inputs[0],  \
                                                                                                (const TYPE*)inputs[1],  \
                                                                                                input0_axis_width,          \
                                                                                                input1_axis_width,          \
                                                                                                out_inner_dims_fast,        \
                                                                                                (TYPE*)output);          \
        } else{                                                                                                             \
            for (int j = 0; j < num_inputs; ++j) {                                                        \
                int input_axis_width = input_dims[j][axis];                                               \
                if (!(mask & (1 << j))) {                                                                 \
                    int64_t input_concat_size = input_axis_width * concat_size;                           \
                    int64_t num_elems         = input_concat_size * num_concats;                          \
                    if (num_elems > 0) {                                                                  \
                        DivModFast num_elems_inner_fast = DivModFast(input_concat_size);                  \
                        int block_size                  = 256;                                            \
                        CONTACT_OPT(TYPE, input_concat_size)                                              \
                    }                                                                                     \
                }                                                                                         \
                axis_offset += input_axis_width;                                                          \
            }                                                                                             \
        }                                                                                                \
        return ppl::common::RC_SUCCESS;                                                               \
    }
#else
#define SWITCH_CASE(TYPE)                                                                             \
    case sizeof(TYPE): {                                                                              \
        for (int j = 0; j < num_inputs; ++j) {                                                        \
            int input_axis_width = input_dims[j][axis];                                               \
            if (!(mask & (1 << j))) {                                                                 \
                int64_t input_concat_size = input_axis_width * concat_size;                           \
                int64_t num_elems         = input_concat_size * num_concats;                          \
                if (num_elems > 0) {                                                                  \
                    DivModFast num_elems_inner_fast = DivModFast(input_concat_size);                  \
                    int block_size                  = 256;                                            \
                    int grid_size                   = (num_elems + block_size - 1) / block_size;      \
                    ppl_cukernel_concat<<<grid_size, block_size, 0, stream>>>(num_elems,              \
                                                                              (const TYPE*)inputs[j], \
                                                                              concat_size,            \
                                                                              output_axis_width,      \
                                                                              num_elems_inner_fast,   \
                                                                              axis_offset,            \
                                                                              (TYPE*)output);         \
                }                                                                                     \
            }                                                                                         \
            axis_offset += input_axis_width;                                                          \
        }                                                                                             \
        return ppl::common::RC_SUCCESS;                                                               \
    }
#endif

#ifdef __CONCAT_OPT__
    ppl::common::GetSizeOfDataType(output_shape->GetDataType());
    if(num_inputs <= 32 && mask == 0) {
        ConcatTensorMetadata<void, 32, 1> inputs_meta;
        int64_t axis_reduce_sum = 0;
        for(int i = 0 ; i < num_inputs; i++) {
            inputs_meta.input[i] = inputs[i];
            inputs_meta.axis_width[i] = input_dims[i][axis] * concat_size;
            inputs_meta.axis_width_reduce_sum[i] = axis_reduce_sum;
            inputs_meta.modFast[i] = DivModFast(inputs_meta.axis_width[i]);
            axis_reduce_sum += inputs_meta.axis_width[i];
        }
        int64_t out_size = output_axis_width * concat_size * num_concats;
        int out_stride = output_axis_width * concat_size;
        switch(ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
        case 1: {
            int line = 8192 / out_stride;
            if(line >= 1) {
                int shift = 0;
                while((line >> shift)) {
                    shift++;
                    if(shift == 4){
                        break;
                    }
                }
                if((line >> shift) == 0) shift--;
                line = line >> shift << shift;
                int gridSize = (num_concats + line - 1) / line;
                int blockSize = 512;
                if(shift == 4) {
                    ppl_cukernel_concat_sm_multi_input_opt<int8_t, 32, float4, 16, 4, 8192><<<gridSize, blockSize, 0, stream>>>(inputs_meta,(int8_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else if(shift == 3) {
                    ppl_cukernel_concat_sm_multi_input_opt<int8_t, 32, float2, 8, 3, 8192><<<gridSize, blockSize, 0, stream>>>(inputs_meta,(int8_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else if(shift == 2) {
                    ppl_cukernel_concat_sm_multi_input_opt<int8_t, 32, float, 4, 2, 8192><<<gridSize, blockSize, 0, stream>>>(inputs_meta,(int8_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else if(shift == 1) {
                    ppl_cukernel_concat_sm_multi_input_opt<int8_t, 32, short, 2, 1, 8192><<<gridSize, blockSize, 0, stream>>>(inputs_meta,(int8_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else {
                    ppl_cukernel_concat_sm_multi_input_opt<int8_t, 32, int8_t, 1, 0, 8192><<<gridSize, blockSize, 0, stream>>>(inputs_meta,(int8_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                }
            } else {
                int blockSize = 512;
                int gridSize = (out_size + 4095) >> 12;
                ppl_cukernel_concat_multi_input_opt<int8_t, 32, float2, 8, 3><<<gridSize, blockSize, 0, stream>>>(inputs_meta, (int8_t*)output, out_size, concat_size,num_inputs, DivModFast(out_stride));
            }    
            break;
        }
        case 2: {
            int line = 4096 / (output_axis_width * concat_size);
            if(line >= 1) {
                int shift = 0;
                while((line >> shift)) {
                    shift++;
                    if(shift == 3){
                        break;
                    }
                }
                if((line >> shift) == 0) shift--;
                line = line >> shift << shift;
                int gridSize = (num_concats + line - 1) / line;
                int blockSize = 512;
                int out_stride = output_axis_width * concat_size;
                if(shift == 3) {
                    ppl_cukernel_concat_sm_multi_input_opt<short, 32, float4, 8, 3, 4096><<<gridSize, blockSize, 0, stream>>>(inputs_meta,(short*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else if(shift == 2) {
                    ppl_cukernel_concat_sm_multi_input_opt<short, 32, float2, 4, 2, 4096><<<gridSize, blockSize, 0, stream>>>(inputs_meta,(short*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else if(shift == 1) {
                    ppl_cukernel_concat_sm_multi_input_opt<short, 32, float, 2, 1, 4096><<<gridSize, blockSize, 0, stream>>>(inputs_meta,(short*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                } else {
                    ppl_cukernel_concat_sm_multi_input_opt<short, 32, short, 1, 0, 4096><<<gridSize, blockSize, 0, stream>>>(inputs_meta,(short*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                }
            } else {
                int blockSize = 512;
                int gridSize = (out_size + 4095) >> 12;
                ppl_cukernel_concat_multi_input_opt<short, 32, float4, 8, 3><<<gridSize, blockSize, 0, stream>>>(inputs_meta, (short*)output, out_size, concat_size,num_inputs, DivModFast(out_stride));
            }
            break;
        }
        case 4:{
            int line = 4096 / (output_axis_width * concat_size);
            if(line >= 1) {
                int gridSize = (num_concats + line - 1) / line;
                int blockSize = 512;
                ppl_cukernel_concat_sm_multi_input_opt<int32_t, 32, int32_t, 1, 0, 4096><<<gridSize, blockSize, 0, stream>>>(inputs_meta, (int32_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
            } else {
                int blockSize = 512;
                int gridSize = (out_size + 2047) >> 11;
                ppl_cukernel_concat_multi_input_opt<int32_t, 32, float4, 4, 2><<<gridSize, blockSize, 0, stream>>>(inputs_meta, (int32_t*)output, out_size, concat_size,num_inputs, DivModFast(out_stride));
            }
            break;
        }
        case 8:{
            int line = 2048 / (output_axis_width * concat_size);
            if(line >= 1) {
                int gridSize = (num_concats + line - 1) / line;
                int blockSize = 512;
                ppl_cukernel_concat_sm_multi_input_opt<int64_t, 32, int64_t, 1, 0, 2048><<<gridSize, blockSize, 0, stream>>>(inputs_meta, (int64_t*)output, out_size, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
            } else {
                int blockSize = 512;
                int gridSize = (out_size + 1023) >> 10;
                ppl_cukernel_concat_multi_input_opt<int64_t, 32, float4, 2, 1><<<gridSize, blockSize, 0, stream>>>(inputs_meta, (int64_t*)output, out_size, concat_size,num_inputs, DivModFast(out_stride));
            }
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
            break;
        }
        return ppl::common::RC_SUCCESS;
    }
#endif //__CONCAT_OPT__

            switch (ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
                SWITCH_CASE(int8_t);
                SWITCH_CASE(int16_t);
                SWITCH_CASE(int32_t);
                SWITCH_CASE(int64_t);
                default:
                    return ppl::common::RC_UNSUPPORTED;
            }
#undef SWITCH_CASE
        }
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
        // nhwc, axis == 1 means last dim
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16 && num_inputs == 2 &&
            axis == 1 && !(input_dims[0][axis] & 0x7) && !(input_dims[1][axis] & 0x7)) {
            if (!(input_dims[0][axis] & 0x7) && !(input_dims[1][axis] & 0x7)) {
                int block_size    = 256;
                int channel_shift = 3;
                int grid_size     = ((output_elems >> channel_shift) + block_size - 1) / block_size;
                int axis_width0   = input_dims[0][axis] >> channel_shift;
                int axis_width1   = input_dims[1][axis] >> channel_shift;
                int inner_dims    = axis_width0 + axis_width1;
                ppl_cukernel_concat_nhwc_two_inputs<<<grid_size, block_size, 0, stream>>>(output_elems >> channel_shift,
                                                                                          inner_dims,
                                                                                          axis_width0,
                                                                                          axis_width1,
                                                                                          (const float4*)inputs[0],
                                                                                          (const float4*)inputs[1],
                                                                                          (float4*)output);
            } else {
                int block_size      = 256;
                int grid_size       = (output_elems + block_size - 1) / block_size;
                int axis_width0     = input_dims[0][axis];
                int pad_axis_width0 = Align(axis_width0, NHWC8_ALIGNED_AXIS);
                int axis_width1     = input_dims[1][axis];
                int pad_axis_width1 = Align(axis_width1, NHWC8_ALIGNED_AXIS);
                int inner_dims      = axis_width0 + axis_width1;
                int pad_inner_dims  = Align(inner_dims, NHWC8_ALIGNED_AXIS);
                ppl_cukernel_concat_nhwc_two_inputs<<<grid_size, block_size, 0, stream>>>(output_elems,
                                                                                          inner_dims,
                                                                                          pad_inner_dims,
                                                                                          axis_width0,
                                                                                          pad_axis_width0,
                                                                                          axis_width1,
                                                                                          pad_axis_width1,
                                                                                          (const int16_t*)inputs[0],
                                                                                          (const int16_t*)inputs[1],
                                                                                          (int16_t*)output);
            }
            return ppl::common::RC_SUCCESS;
        }

                // nhwc, axis == 1 means last dim
        if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8 && num_inputs == 2 &&
            axis == 1) {
            if (!(input_dims[0][axis] & 0xFFFFFFFF) && !(input_dims[1][axis] & 0xFFFFFFFF)) {
                int block_size    = 256;
                int channel_shift = 4;
                int grid_size     = ((output_elems >> channel_shift) + block_size - 1) / block_size;
                int axis_width0   = input_dims[0][axis] >> channel_shift;
                int axis_width1   = input_dims[1][axis] >> channel_shift;
                int inner_dims    = axis_width0 + axis_width1;
                ppl_cukernel_concat_nhwc_two_inputs<<<grid_size, block_size, 0, stream>>>(output_elems >> channel_shift,
                                                                                          inner_dims,
                                                                                          axis_width0,
                                                                                          axis_width1,
                                                                                          (const float4*)inputs[0],
                                                                                          (const float4*)inputs[1],
                                                                                          (float4*)output);
            } else {
                int block_size      = 256;
                int grid_size       = (output_elems + block_size - 1) / block_size;
                int axis_width0     = input_dims[0][axis];
                int pad_axis_width0 = Align(axis_width0, NHWC16_ALIGNED_AXIS);
                int axis_width1     = input_dims[1][axis];
                int pad_axis_width1 = Align(axis_width1, NHWC16_ALIGNED_AXIS);
                int inner_dims      = axis_width0 + axis_width1;
                int pad_inner_dims  = Align(inner_dims, NHWC16_ALIGNED_AXIS);
#ifdef __CONCAT_OPT__
                DivModFast pad_inner_dims_fast = DivModFast(pad_inner_dims);
                grid_size = (output_elems + block_size*8 - 1) / (block_size*8);
                ppl_cukernel_concat_nhwc_two_inputs_optN<<<grid_size, block_size, 0, stream>>>(output_elems,
                                                                                          inner_dims,
                                                                                          pad_inner_dims,
                                                                                          axis_width0,
                                                                                          pad_axis_width0,
                                                                                          axis_width1,
                                                                                          pad_axis_width1,
                                                                                          pad_inner_dims_fast,
                                                                                          (const int8_t*)inputs[0],
                                                                                          (const int8_t*)inputs[1],
                                                                                          (int8_t*)output);
#else//!__CONCAT_OPT__
                ppl_cukernel_concat_nhwc_two_inputs<<<grid_size, block_size, 0, stream>>>(output_elems,
                                                                                          inner_dims,
                                                                                          pad_inner_dims,
                                                                                          axis_width0,
                                                                                          pad_axis_width0,
                                                                                          axis_width1,
                                                                                          pad_axis_width1,
                                                                                          (const int8_t*)inputs[0],
                                                                                          (const int8_t*)inputs[1],
                                                                                          (int8_t*)output);
#endif//__CONCAT_OPT__
            }
            return ppl::common::RC_SUCCESS;
        }
        int axis_offset = 0;
        std::vector<int32_t> nhwc_output_padded_dims(num_dims);
        nhwc_output_padded_dims[num_dims - 1] = output_shape->GetDim(1) +
                                                output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
        int jump_step = 0;
        for (int it = 0; it < num_dims - 1; ++it) {
            if (it == 1)
                jump_step = 1;
            nhwc_output_padded_dims[it] = output_shape->GetDim(it + jump_step);
        }
        GArray<int64_t> output_padded_strides(num_dims);
        int64_t acc_output_stride = 1;
        for (int it = num_dims - 1; it >= 0; --it) {
            output_padded_strides[it] = acc_output_stride;
            acc_output_stride *= nhwc_output_padded_dims[it];
        }

#ifdef __CONCAT_OPT__
#define OPT_AXIS_ONE(TYPE)                                                                                                                                                              \
    const int SIZE_T = sizeof(TYPE);                                                                                                                                                    \
    if(SIZE_T==1)                                                                                                                                                                       \
    {                                                                                                                                                                                   \
        if((nhwc_input_dims[num_dims - 1] % 16)==0&&(axis_offset%16)==0)                                                                                                                \
        {                                                                                                                                                                               \
            grid_size = (num_elems + block_size*16 - 1) / (block_size*16);                                                                                                              \
            DivModFast input_stride_fast = DivModFast(nhwc_input_dims[num_dims - 1]);                                                                                                   \
            ppl_cukernel_concat_nhwc_axis1_optN<TYPE,16,SIZE_T><<<grid_size, block_size, 0, stream>>>(                                                                                  \
                    num_elems, axis_offset, nhwc_input_padded_dims[num_dims - 1], nhwc_output_padded_dims[num_dims - 1], input_stride_fast, (const TYPE*)inputs[j], (TYPE*)output);     \
        } else if((nhwc_input_dims[num_dims - 1] % 8)==0&&(axis_offset%8)==0) {                                                                                                         \
            grid_size = (num_elems + block_size*8 - 1) / (block_size*8);                                                                                                                \
            DivModFast input_stride_fast = DivModFast(nhwc_input_dims[num_dims - 1]);                                                                                                   \
            ppl_cukernel_concat_nhwc_axis1_optN<TYPE,8,SIZE_T><<<grid_size, block_size, 0, stream>>>(                                                                                   \
                    num_elems, axis_offset, nhwc_input_padded_dims[num_dims - 1], nhwc_output_padded_dims[num_dims - 1], input_stride_fast, (const TYPE*)inputs[j], (TYPE*)output);     \
        }                                                                                                                                                                               \
        else if((nhwc_input_dims[num_dims - 1] % 4)==0&&(axis_offset%4)==0)                                                                                                             \
        {                                                                                                                                                                               \
            grid_size = (num_elems + block_size*4 - 1) / (block_size*4);                                                                                                                \
            DivModFast input_stride_fast = DivModFast(nhwc_input_dims[num_dims - 1]);                                                                                                   \
            ppl_cukernel_concat_nhwc_axis1_optN<TYPE,4,SIZE_T><<<grid_size, block_size, 0, stream>>>(                                                                                   \
                    num_elems, axis_offset, nhwc_input_padded_dims[num_dims - 1], nhwc_output_padded_dims[num_dims - 1], input_stride_fast, (const TYPE*)inputs[j], (TYPE*)output);     \
        }                                                                                                                                                                               \
        else if((nhwc_input_dims[num_dims - 1] % 2)==0&&(axis_offset%2)==0)                                                                                                             \
        {                                                                                                                                                                               \
            grid_size = (num_elems + block_size*2 - 1) / (block_size*2);                                                                                                                \
            DivModFast input_stride_fast = DivModFast(nhwc_input_dims[num_dims - 1]);                                                                                                   \
            ppl_cukernel_concat_nhwc_axis1_optN<TYPE,2,SIZE_T><<<grid_size, block_size, 0, stream>>>(                                                                                   \
                    num_elems, axis_offset, nhwc_input_padded_dims[num_dims - 1], nhwc_output_padded_dims[num_dims - 1], input_stride_fast, (const TYPE*)inputs[j], (TYPE*)output);     \
        }                                                                                                                                                                               \
        else{                                                                                                                                                                           \
            grid_size = (num_elems + block_size - 1) / (block_size);                                                                                                                    \
            DivModFast input_stride_fast = DivModFast(nhwc_input_dims[num_dims - 1]);                                                                                                   \
            ppl_cukernel_concat_nhwc_axis1_optN<TYPE,1,SIZE_T><<<grid_size, block_size, 0, stream>>>(                                                                                   \
                    num_elems, axis_offset, nhwc_input_padded_dims[num_dims - 1], nhwc_output_padded_dims[num_dims - 1], input_stride_fast, (const TYPE*)inputs[j], (TYPE*)output);     \
        }                                                                                                                                                                               \
    }                                                                                                                                                                                   \
    else if(SIZE_T==2){                                                                                                                                                                 \
        if((nhwc_input_dims[num_dims - 1] % 4)==0&&(axis_offset%4)==0)                                                                                                                  \
        {                                                                                                                                                                               \
            grid_size = (num_elems + block_size*4 - 1) / (block_size*4);                                                                                                                \
            DivModFast input_stride_fast = DivModFast(nhwc_input_dims[num_dims - 1]);                                                                                                   \
            ppl_cukernel_concat_nhwc_axis1_optN<TYPE,4,SIZE_T><<<grid_size, block_size, 0, stream>>>(                                                                                   \
                    num_elems, axis_offset, nhwc_input_padded_dims[num_dims - 1], nhwc_output_padded_dims[num_dims - 1], input_stride_fast, (const TYPE*)inputs[j], (TYPE*)output);     \
        }                                                                                                                                                                               \
        else if((nhwc_input_dims[num_dims - 1] % 2)==0&&(axis_offset%2)==0)                                                                                                             \
        {                                                                                                                                                                               \
            grid_size = (num_elems + block_size*2 - 1) / (block_size*2);                                                                                                                \
            DivModFast input_stride_fast = DivModFast(nhwc_input_dims[num_dims - 1]);                                                                                                   \
            ppl_cukernel_concat_nhwc_axis1_optN<TYPE,2,SIZE_T><<<grid_size, block_size, 0, stream>>>(                                                                                   \
                    num_elems, axis_offset, nhwc_input_padded_dims[num_dims - 1], nhwc_output_padded_dims[num_dims - 1], input_stride_fast, (const TYPE*)inputs[j], (TYPE*)output);     \
        }                                                                                                                                                                               \
        else{                                                                                                                                                                           \
            grid_size = (num_elems + block_size - 1) / (block_size);                                                                                                                    \
            DivModFast input_stride_fast = DivModFast(nhwc_input_dims[num_dims - 1]);                                                                                                   \
            ppl_cukernel_concat_nhwc_axis1_optN<TYPE,1,SIZE_T><<<grid_size, block_size, 0, stream>>>(                                                                                   \
                    num_elems, axis_offset, nhwc_input_padded_dims[num_dims - 1], nhwc_output_padded_dims[num_dims - 1], input_stride_fast, (const TYPE*)inputs[j], (TYPE*)output);     \
        }                                                                                                                                                                               \
    }                                                                                                                                                                                   \
    else if(SIZE_T==4){                                                                                                                                                                 \
        if((nhwc_input_dims[num_dims - 1] % 2)==0&&(axis_offset%2)==0)                                                                                                                  \
        {                                                                                                                                                                               \
            grid_size = (num_elems + block_size*2 - 1) / (block_size*2);                                                                                                                \
            DivModFast input_stride_fast = DivModFast(nhwc_input_dims[num_dims - 1]);                                                                                                   \
            ppl_cukernel_concat_nhwc_axis1_optN<TYPE,2,SIZE_T><<<grid_size, block_size, 0, stream>>>(                                                                                   \
                    num_elems, axis_offset, nhwc_input_padded_dims[num_dims - 1], nhwc_output_padded_dims[num_dims - 1], input_stride_fast, (const TYPE*)inputs[j], (TYPE*)output);     \
        }                                                                                                                                                                               \
        else{                                                                                                                                                                           \
            grid_size = (num_elems + block_size - 1) / (block_size);                                                                                                                    \
            DivModFast input_stride_fast = DivModFast(nhwc_input_dims[num_dims - 1]);                                                                                                   \
            ppl_cukernel_concat_nhwc_axis1_optN<TYPE,1,SIZE_T><<<grid_size, block_size, 0, stream>>>(                                                                                   \
                    num_elems, axis_offset, nhwc_input_padded_dims[num_dims - 1], nhwc_output_padded_dims[num_dims - 1], input_stride_fast, (const TYPE*)inputs[j], (TYPE*)output);     \
        }                                                                                                                                                                               \
    }                                                                                                                                                                                   \
    else{                                                                                                                                                                               \
            grid_size = (num_elems + block_size - 1) / (block_size);                                                                                                                    \
            DivModFast input_stride_fast = DivModFast(nhwc_input_dims[num_dims - 1]);                                                                                                   \
            ppl_cukernel_concat_nhwc_axis1_optN<TYPE,1,SIZE_T><<<grid_size, block_size, 0, stream>>>(                                                                                   \
                    num_elems, axis_offset, nhwc_input_padded_dims[num_dims - 1], nhwc_output_padded_dims[num_dims - 1], input_stride_fast, (const TYPE*)inputs[j], (TYPE*)output);     \
    }

#define SWITCH_CASE(TYPE)                                                                                                                                             \
    case sizeof(TYPE): {                                                                                                                                              \
        for (int j = 0; j < num_inputs; ++j) {                                                                                                                        \
            int nhwc_axis = (axis == 1) ? num_dims - 1 : axis - 1;                                                                                                    \
            nhwc_axis     = (axis == 0) ? 0 : nhwc_axis;                                                                                                              \
            std::vector<int32_t> nhwc_input_dims(num_dims);                                                                                                           \
            std::vector<int32_t> nhwc_input_padded_dims(num_dims);                                                                                                    \
            nhwc_input_dims[num_dims - 1]        = input_dims[j][1];                                                                                                  \
            nhwc_input_padded_dims[num_dims - 1] = input_padded_dims[j][1];                                                                                           \
            jump_step                            = 0;                                                                                                                 \
            for (int it = 0; it < num_dims - 1; ++it) {                                                                                                               \
                if (it == 1)                                                                                                                                          \
                    jump_step = 1;                                                                                                                                    \
                nhwc_input_dims[it]        = input_dims[j][it + jump_step];                                                                                           \
                nhwc_input_padded_dims[it] = input_padded_dims[j][it + jump_step];                                                                                    \
            }                                                                                                                                                         \
            GArray<DivModFast> input_strides_fast(num_dims);                                                                                                          \
            GArray<int64_t> input_padded_strides(num_dims);                                                                                                           \
            int64_t acc_input_stride = 1, acc_input_padded_stride = 1;                                                                                                \
            for (int it = num_dims - 1; it >= 0; --it) {                                                                                                              \
                input_strides_fast[it]   = DivModFast(acc_input_stride);                                                                                              \
                input_padded_strides[it] = acc_input_padded_stride;                                                                                                   \
                acc_input_stride *= nhwc_input_dims[it];                                                                                                              \
                acc_input_padded_stride *= nhwc_input_padded_dims[it];                                                                                                \
            }                                                                                                                                                         \
            int input_axis_width = nhwc_input_dims[nhwc_axis];                                                                                                        \
            if (!(mask & (1 << j))) {                                                                                                                                 \
                int64_t num_elems    = 1;                                                                                                                             \
                for (int it = 0; it < num_dims; ++it)                                                                                                                 \
                    num_elems *= nhwc_input_dims[it];                                                                                                                 \
                int block_size = 256;                                                                                                                                 \
                int grid_size  = (num_elems + block_size - 1) / block_size;                                                                                           \
                if(axis == 1)                                                                                                                                         \
                {                                                                                                                                                     \
                    OPT_AXIS_ONE(TYPE);                                                                                                                               \
                }else{                                                                                                                                                \
                    ppl_cukernel_concat_nhwc<<<grid_size, block_size, 0, stream>>>(                                                                                   \
                num_elems, num_dims, nhwc_axis, axis_offset, input_strides_fast, input_padded_strides, output_padded_strides, (const TYPE*)inputs[j], (TYPE*)output); \
                }                                                                                                                                                         \
            }                                                                                                                                                             \
            axis_offset += input_axis_width;                                                                                                                              \
        }                                                                                                                                                                 \
        return ppl::common::RC_SUCCESS;                                                                                                                                   \
    }
#else//!__CONCAT_OPT__
#define SWITCH_CASE(TYPE)                                                                                                                                             \
    case sizeof(TYPE): {                                                                                                                                              \
        for (int j = 0; j < num_inputs; ++j) {                                                                                                                        \
            int nhwc_axis = (axis == 1) ? num_dims - 1 : axis - 1;                                                                                                    \
            nhwc_axis     = (axis == 0) ? 0 : nhwc_axis;                                                                                                              \
            std::vector<int32_t> nhwc_input_dims(num_dims);                                                                                                           \
            std::vector<int32_t> nhwc_input_padded_dims(num_dims);                                                                                                    \
            nhwc_input_dims[num_dims - 1]        = input_dims[j][1];                                                                                                  \
            nhwc_input_padded_dims[num_dims - 1] = input_padded_dims[j][1];                                                                                           \
            jump_step                            = 0;                                                                                                                 \
            for (int it = 0; it < num_dims - 1; ++it) {                                                                                                               \
                if (it == 1)                                                                                                                                          \
                    jump_step = 1;                                                                                                                                    \
                nhwc_input_dims[it]        = input_dims[j][it + jump_step];                                                                                           \
                nhwc_input_padded_dims[it] = input_padded_dims[j][it + jump_step];                                                                                    \
            }                                                                                                                                                         \
            GArray<DivModFast> input_strides_fast(num_dims);                                                                                                          \
            GArray<int64_t> input_padded_strides(num_dims);                                                                                                           \
            int64_t acc_input_stride = 1, acc_input_padded_stride = 1;                                                                                                \
            for (int it = num_dims - 1; it >= 0; --it) {                                                                                                              \
                input_strides_fast[it]   = DivModFast(acc_input_stride);                                                                                              \
                input_padded_strides[it] = acc_input_padded_stride;                                                                                                   \
                acc_input_stride *= nhwc_input_dims[it];                                                                                                              \
                acc_input_padded_stride *= nhwc_input_padded_dims[it];                                                                                                \
            }                                                                                                                                                         \
            int input_axis_width = nhwc_input_dims[nhwc_axis];                                                                                                        \
            if (!(mask & (1 << j))) {                                                                                                                                 \
                int64_t num_elems    = 1;                                                                                                                             \
                for (int it = 0; it < num_dims; ++it)                                                                                                                 \
                    num_elems *= nhwc_input_dims[it];                                                                                                                 \
                int block_size = 256;                                                                                                                                 \
                int grid_size  = (num_elems + block_size - 1) / block_size;                                                                                           \
                ppl_cukernel_concat_nhwc<<<grid_size, block_size, 0, stream>>>(                                                                                       \
                    num_elems, num_dims, nhwc_axis, axis_offset, input_strides_fast, input_padded_strides, output_padded_strides, (const TYPE*)inputs[j], (TYPE*)output); \
            }                                                                                                                                                             \
            axis_offset += input_axis_width;                                                                                                                          \
        }                                                                                                                                                             \
        return ppl::common::RC_SUCCESS;                                                                                                                               \
    }
#endif//__CONCAT_OPT__

#ifdef __CONCAT_OPT__
        if(axis == 1 && num_inputs < 32) {
            ConcatTensorMetadata<void, 32, 1> inputs_meta;
            int64_t axis_reduce_sum = 0;
            int input_stride = 0;
            int64_t num_concats = 1;
            for(int i = 0; i < num_dims - 1; i++) {
                num_concats *= nhwc_output_padded_dims[i];
            }
            for(int i = 0 ; i < num_inputs; i++)
            {
                inputs_meta.input[i] = inputs[i];
                inputs_meta.axis_width[i] = input_dims[i][1];
                inputs_meta.axis_stride[i] = input_padded_dims[i][1];
                inputs_meta.axis_width_reduce_sum[i] = axis_reduce_sum;
                inputs_meta.modFast[i] = DivModFast(inputs_meta.axis_stride[i]);
                axis_reduce_sum += inputs_meta.axis_width[i];
                input_stride += input_padded_dims[i][1];
            }
            int64_t out_size = nhwc_output_padded_dims[num_dims - 1] * num_concats;
            int out_stride = nhwc_output_padded_dims[num_dims - 1];
            switch(ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
                case 1: {
                    int line = 8192 / input_stride;
                    int is_align = 1;
                    for(int i = 0; i < num_inputs; i++) {
                        is_align = is_align & (!(input_padded_dims[i][1] & 15));
                    }
                    is_align = is_align & (!(nhwc_output_padded_dims[num_dims - 1] & 15));
                    if(line >= 1 && is_align) {
                        int gridSize = (num_concats + line - 1) / line;
                        int blockSize = 512;
                        ppl_cukernel_concat_sm_multi_input_padding_opt<int8_t, 32, float4, 16, 4><<<gridSize, blockSize, 0, stream>>>(inputs_meta,output, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                    } else {
                        int blockSize = 512;
                        int gridSize = (out_size + 4095) >> 12;
                        ppl_cukernel_concat_multi_input_padding_opt<int8_t, 32, 3><<<gridSize, blockSize, 0, stream>>>(inputs_meta, output,out_size, num_inputs, DivModFast(out_stride));
                    }    
                    return ppl::common::RC_SUCCESS;  
                }
                case 2: {
                    int line = 4096 / input_stride;
                    int is_align = 1;
                    for(int i = 0; i < num_inputs; i++) {
                        is_align = is_align & (!(input_padded_dims[i][1] & 7));
                    }
                    is_align = is_align & !(nhwc_output_padded_dims[num_dims - 1] & 7);
                    if(line >= 1 && is_align) {
                        int gridSize = (num_concats + line - 1) / line;
                        int blockSize = 512;
                        ppl_cukernel_concat_sm_multi_input_padding_opt<short, 32, float4, 8, 3><<<gridSize, blockSize, 0, stream>>>(inputs_meta,output, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                    } else {
                        int blockSize = 512;
                        int gridSize = (out_size + 4095) >> 12;
                        ppl_cukernel_concat_multi_input_padding_opt<short, 32, 3><<<gridSize, blockSize, 0, stream>>>(inputs_meta, output, out_size, num_inputs, DivModFast(out_stride));
                    }
                    return ppl::common::RC_SUCCESS;  
                }
                case 4:{
                    int line = 2048 / input_stride;
                    if(line >= 1) {
                        int gridSize = (num_concats + line - 1) / line;
                        int blockSize = 512;
                        ppl_cukernel_concat_sm_multi_input_padding_opt<int32_t, 32, int32_t, 1, 0><<<gridSize, blockSize, 0, stream>>>(inputs_meta,output, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                    } else {
                        int blockSize = 512;
                        int gridSize = (out_size + 4095) >> 12;
                        ppl_cukernel_concat_multi_input_padding_opt<int32_t, 32, 3><<<gridSize, blockSize, 0, stream>>>(inputs_meta, output, out_size, num_inputs, DivModFast(out_stride));
                    }
                    return ppl::common::RC_SUCCESS;  
                }
                case 8:{
                    int line = 1024 / input_stride;
                    if(line >= 1) {
                        int gridSize = (num_concats + line - 1) / line;
                        int blockSize = 512;
                        ppl_cukernel_concat_sm_multi_input_padding_opt<long, 32, long, 1, 0><<<gridSize, blockSize, 0, stream>>>(inputs_meta,output, out_stride, num_inputs, DivModFast(out_stride),line,num_concats);
                    } else {
                        int blockSize = 512;
                        int gridSize = (out_size + 4095) >> 12;
                        ppl_cukernel_concat_multi_input_padding_opt<long, 32, 3><<<gridSize, blockSize, 0, stream>>>(inputs_meta, output, out_size, num_inputs, DivModFast(out_stride));
                    }
                    return ppl::common::RC_SUCCESS;  
                }
                default:
                    return ppl::common::RC_UNSUPPORTED;
            }
        }
#endif//__CONCAT_OPT__
        switch (ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
            SWITCH_CASE(int8_t);
            SWITCH_CASE(int16_t);
            SWITCH_CASE(int32_t);
            SWITCH_CASE(int64_t);
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
#undef SWITCH_CASE
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}
