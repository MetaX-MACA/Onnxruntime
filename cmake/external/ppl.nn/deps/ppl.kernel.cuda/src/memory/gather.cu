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

#include "cudakernel/memory/gather.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/common.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <memory>

__host__ __device__ __inline__ int get_indices_val(
    int indices_element_size,
    int offset,
    const void* indices)
{
    int res = 0;
    switch (indices_element_size) {
        case sizeof(int32_t):
            res = static_cast<const int32_t*>(indices)[offset];
            break;
        case sizeof(int64_t):
            res = static_cast<const int64_t*>(indices)[offset];
            break;
        default:
            break;
    }
    return res;
}

template <typename T>
__global__ void ppl_cukernel_gather(
    int64_t num_elems,
    DivModFast output_outer_block_fast,
    int input_axis_size,
    DivModFast output_inner_block_fast,
    const T* input,
    T* output,
    int indices_element_size,
    const void* indices)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int outer_idx, block_offset;
    output_outer_block_fast.divmod(index, outer_idx, block_offset);
    int indices_offset, inner_idx;
    output_inner_block_fast.divmod(block_offset, indices_offset, inner_idx);
    int64_t indices_idx = get_indices_val(indices_element_size, indices_offset, indices);
    // -d means distance from last dimension
    indices_idx         = indices_idx < 0 ? indices_idx + input_axis_size : indices_idx;
    if (indices_idx < 0 || indices_idx >= input_axis_size) {
        output[index] = 0;
        return;
    }
    int64_t input_idx = (outer_idx * input_axis_size + indices_idx) *
                            output_inner_block_fast.d_ +
                        inner_idx;
    output[index] = input[input_idx];
}

template <typename T1, typename T2, typename T3>
__global__ void ppl_cukernel_gather_nhwc16_sm_opt(
    const T1* input,
    const T2* indices,
    T3* output,
    int input_num_elems,
    int input_axis_elem_exclude_padding,
    int input_axis_size_include_padding,
    int indices_elems,
    int output_axis_elem_include_padding,
    int elems_per_block,
    DivModFast input_axis_size_include_padding_fast,
    DivModFast channels_per_block_fast)
{
    int index = blockIdx.x * elems_per_block + threadIdx.x;
    __shared__ float4 sm_input[256];
    __shared__ int sm_indices[1024];

    if (index < input_num_elems) {
        sm_input[threadIdx.x] = input[index];
    } else {
        sm_input[threadIdx.x] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        int index_offset = (threadIdx.x << 2) + i;
        if (index_offset < indices_elems) {
            sm_indices[index_offset] = (int)indices[index_offset];
        } else {
            sm_indices[index_offset] = 0;
        }
    }

    __syncthreads();

    if (index >= input_num_elems || threadIdx.x >= elems_per_block) {
        return;
    }
    int8_t result[16] = {
            (int8_t)0, (int8_t)0, (int8_t)0, (int8_t)0 ,(int8_t)0, (int8_t)0, (int8_t)0, (int8_t)0,
            (int8_t)0, (int8_t)0, (int8_t)0, (int8_t)0 ,(int8_t)0, (int8_t)0, (int8_t)0, (int8_t)0};

    int outer_idx, inner_idx, inner_c, indices_offset;
    input_axis_size_include_padding_fast.divmod(index, outer_idx, inner_idx);
    if (inner_idx >= output_axis_elem_include_padding)
        return;
    inner_c = channels_per_block_fast.mod(outer_idx);
    indices_offset = inner_idx << 4;

    int output_offset = outer_idx * output_axis_elem_include_padding + inner_idx;
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        if ((indices_offset + i) < indices_elems) {
            int indices_idx = sm_indices[indices_offset + i];
            indices_idx     = indices_idx < 0 ? indices_idx + input_axis_elem_exclude_padding : indices_idx;
            if (indices_idx >= 0 && indices_idx < input_axis_elem_exclude_padding) {
                int input_idx = inner_c * (input_axis_size_include_padding << 4) + indices_idx;
                result[i] = ((int8_t*)sm_input)[input_idx];
            }
        }
    }
    output[output_offset] = *(float4*)result;
}

template <typename T1, typename T2, typename T3>
__global__ void ppl_cukernel_gather_nhwc8_sm_opt(
    const T1* input,
    const T2* indices,
    T3* output,
    int input_num_elems,
    int input_axis_elem_exclude_padding,
    int input_axis_size_include_padding,
    int indices_elems,
    int output_axis_elem_include_padding,
    int elems_per_block,
    DivModFast input_axis_size_include_padding_fast,
    DivModFast channels_per_block_fast)
{
    int index = blockIdx.x * elems_per_block + threadIdx.x;
    __shared__ float4 sm_input[256];
    __shared__ int sm_indices[1024];

    if (index < input_num_elems) {
        sm_input[threadIdx.x] = input[index];
    } else {
        sm_input[threadIdx.x] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        int index_offset = (threadIdx.x << 2) + i;
        if (index_offset < indices_elems) {
            sm_indices[index_offset] = (int)indices[index_offset];
        } else {
            sm_indices[index_offset] = 0;
        }
    }

    __syncthreads();

    if (index >= input_num_elems || threadIdx.x >= elems_per_block) {
        return;
    }
    half result[8] = {(half)0, (half)0, (half)0, (half)0 ,(half)0, (half)0, (half)0, (half)0};

    int outer_idx, inner_idx, inner_c, indices_offset;
    input_axis_size_include_padding_fast.divmod(index, outer_idx, inner_idx);
    if (inner_idx >= output_axis_elem_include_padding)
        return;
    inner_c = channels_per_block_fast.mod(outer_idx);
    indices_offset = inner_idx << 3;

    int output_offset = outer_idx * output_axis_elem_include_padding + inner_idx;
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        if ((indices_offset + i) < indices_elems) {
            int indices_idx = sm_indices[indices_offset + i];
            indices_idx     = indices_idx < 0 ? indices_idx + input_axis_elem_exclude_padding : indices_idx;
            if (indices_idx >= 0 && indices_idx < input_axis_elem_exclude_padding) {
                int input_idx = inner_c * (input_axis_size_include_padding << 3) + indices_idx;
                result[i] = ((half*)sm_input)[input_idx];
            }
        }
    }
    output[output_offset] = *(float4*)result;
}

template <typename T, int Times>
__global__ void ppl_cukernel_gather_opt(
    int64_t num_elems,
    DivModFast output_outer_block_fast,
    int input_axis_size,
    DivModFast output_inner_block_fast,
    const T* input,
    T* output,
    int indices_element_size,
    const void* indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = Times * idx;
    if (index >= num_elems)
        return;
    int outer_idx, block_offset;
    output_outer_block_fast.divmod(index, outer_idx, block_offset);
    int indices_offset, inner_idx;
    output_inner_block_fast.divmod(block_offset, indices_offset, inner_idx);
    int64_t indices_idx = get_indices_val(indices_element_size, indices_offset, indices);
    // -d means distance from last dimension
    indices_idx         = indices_idx < 0 ? indices_idx + input_axis_size : indices_idx;
    if (indices_idx < 0 || indices_idx >= input_axis_size) {
        output[index] = 0;
        return;
    }
    int64_t input_idx = (outer_idx * input_axis_size + indices_idx) *
                            output_inner_block_fast.d_ +
                        inner_idx;
    output[idx] = input[input_idx / Times];
}

ppl::common::RetCode PPLCUDAGatherForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* indices_shape,
    const void* indices,
    const ppl::common::TensorShape* output_shape,
    void* output,
    int axis)
{
    int indices_element_size = ppl::common::GetSizeOfDataType(indices_shape->GetDataType());
    // special case, need further evaluement (performance is not usually better)
    if (axis == 0 && indices_shape->GetDimCount() == 1 && indices_shape->GetDim(0) == 1) {
        int indices_data_size = indices_shape->CalcBytesIncludingPadding();
        std::vector<char> indices_data(indices_data_size);
        cudaMemcpyAsync(indices_data.data(), indices, indices_data_size, cudaMemcpyDeviceToHost, stream);
        int inner_size   = input_shape->CalcBytesIncludingPadding() / input_shape->GetDim(0);
        int input_offset = get_indices_val(indices_element_size, 0, indices_data.data());
        int input_axis_size = input_shape->GetDim(axis);
        input_offset        = input_offset < 0 ? input_offset + input_axis_size : input_offset;
        cudaMemcpyAsync(output, static_cast<const char*>(input) + input_offset * inner_size, output_shape->CalcBytesIncludingPadding(), cudaMemcpyDeviceToDevice, stream);
        return ppl::common::RC_SUCCESS;
    }

#ifdef __MACACC__
    if ((output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16)) {
        int block_size            = 256;
        int indices_elems_exclude_padding = indices_shape->CalcElementsExcludingPadding();
        int input_axis_elem_include_padding = input_shape->GetDim(axis) + input_shape->GetPadding0(axis) + input_shape->GetPadding1(axis);
        if (axis == 1 && indices_elems_exclude_padding <= 1024 && input_axis_elem_include_padding <= 4096) {
            int64_t num_per_thread    = 16;
            int64_t input_num_elems   = input_shape->CalcElementsIncludingPadding() / num_per_thread;
            int output_axis_elem_include_padding = (output_shape->GetDim(axis) + output_shape->GetPadding0(axis) + output_shape->GetPadding1(axis)) / num_per_thread;
            input_axis_elem_include_padding      = input_axis_elem_include_padding / num_per_thread;
            int input_axis_elem_exclude_padding  = input_shape->GetDim(axis);

            int channels_per_block = block_size / input_axis_elem_include_padding;
            int elems_per_block    = input_axis_elem_include_padding * channels_per_block;
            DivModFast input_axis_elem_include_padding_fast(input_axis_elem_include_padding);
            DivModFast channels_per_block_fast(channels_per_block);

            int grid_size = (input_num_elems + elems_per_block - 1) / elems_per_block;
            if (indices_element_size == 4) {
                ppl_cukernel_gather_nhwc16_sm_opt<<<grid_size, block_size, 0, stream>>>(
                                                    (const float4*)input,
                                                    (const int32_t*)indices,
                                                    (float4*)output,
                                                    input_num_elems,
                                                    input_axis_elem_exclude_padding,
                                                    input_axis_elem_include_padding,
                                                    indices_elems_exclude_padding,
                                                    output_axis_elem_include_padding,
                                                    elems_per_block,
                                                    input_axis_elem_include_padding_fast,
                                                    channels_per_block_fast);
            } else if (indices_element_size == 8) {
                ppl_cukernel_gather_nhwc16_sm_opt<<<grid_size, block_size, 0, stream>>>(
                                                    (const float4*)input,
                                                    (const int64_t*)indices,
                                                    (float4*)output,
                                                    input_num_elems,
                                                    input_axis_elem_exclude_padding,
                                                    input_axis_elem_include_padding,
                                                    indices_elems_exclude_padding,
                                                    output_axis_elem_include_padding,
                                                    elems_per_block,
                                                    input_axis_elem_include_padding_fast,
                                                    channels_per_block_fast);
            }
            return ppl::common::RC_SUCCESS;
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
        int block_size            = 256;
        int indices_elems_exclude_padding = indices_shape->CalcElementsExcludingPadding();
        int input_axis_elem_include_padding = input_shape->GetDim(axis) + input_shape->GetPadding0(axis) + input_shape->GetPadding1(axis);
        if (axis == 1 && indices_elems_exclude_padding <= 1024 && input_axis_elem_include_padding <= 2048) {
            int64_t num_per_thread    = 8;
            int64_t input_num_elems   = input_shape->CalcElementsIncludingPadding() / num_per_thread;
            int output_axis_elem_include_padding = (output_shape->GetDim(axis) + output_shape->GetPadding0(axis) + output_shape->GetPadding1(axis)) / num_per_thread;
            input_axis_elem_include_padding      = input_axis_elem_include_padding / num_per_thread;
            int input_axis_elem_exclude_padding  = input_shape->GetDim(axis);

            int channels_per_block = block_size / input_axis_elem_include_padding;
            int elems_per_block    = input_axis_elem_include_padding * channels_per_block;
            DivModFast input_axis_elem_include_padding_fast(input_axis_elem_include_padding);
            DivModFast channels_per_block_fast(channels_per_block);

            int grid_size = (input_num_elems + elems_per_block - 1) / elems_per_block;
            if (indices_element_size == 4) {
                ppl_cukernel_gather_nhwc8_sm_opt<<<grid_size, block_size, 0, stream>>>(
                                                    (const float4*)input,
                                                    (const int32_t*)indices,
                                                    (float4*)output,
                                                    input_num_elems,
                                                    input_axis_elem_exclude_padding,
                                                    input_axis_elem_include_padding,
                                                    indices_elems_exclude_padding,
                                                    output_axis_elem_include_padding,
                                                    elems_per_block,
                                                    input_axis_elem_include_padding_fast,
                                                    channels_per_block_fast);
            } else if (indices_element_size == 8) {
                ppl_cukernel_gather_nhwc8_sm_opt<<<grid_size, block_size, 0, stream>>>(
                                                    (const float4*)input,
                                                    (const int64_t*)indices,
                                                    (float4*)output,
                                                    input_num_elems,
                                                    input_axis_elem_exclude_padding,
                                                    input_axis_elem_include_padding,
                                                    indices_elems_exclude_padding,
                                                    output_axis_elem_include_padding,
                                                    elems_per_block,
                                                    input_axis_elem_include_padding_fast,
                                                    channels_per_block_fast);
            }
            return ppl::common::RC_SUCCESS;
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
#endif // __MACACC__

    int64_t num_elems      = output_shape->CalcElementsIncludingPadding();
    int block_size         = 256;
    int grid_size          = (num_elems + block_size - 1) / block_size;
    // output dimension can be partitioned as outer--indices--inner. (before axis, axis, after axis)
    int output_inner_block = input_shape->CalcElementsFromDimensionIncludingPadding(axis + 1);
    int input_axis_size    = input_shape->GetDim(axis);
    int indices_block_size = indices_shape->CalcElementsIncludingPadding();
    int output_outer_block = indices_block_size * output_inner_block;

    DivModFast output_outer_block_fast(output_outer_block);
    DivModFast output_inner_block_fast(output_inner_block);

    int coeff = 8 / ppl::common::GetSizeOfDataType(input_shape->GetDataType());
    for (;coeff >= 1; coeff = coeff >> 1)
    {
        if (output_inner_block % coeff == 0)
            break;
    }
#define SWITCH_CASE(TYPE)                                                                                                                                                                                                       \
    case sizeof(TYPE): {                                                                                                                                                                                                        \
        ppl_cukernel_gather<<<grid_size, block_size, 0, stream>>>(num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, (const TYPE*)input, (TYPE*)output, indices_element_size, (const void*)indices); \
        return ppl::common::RC_SUCCESS;                                                                                                                                                                                         \
    }

    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        case sizeof(int8_t):
            switch (coeff)
            {
            case 8:
                ppl_cukernel_gather_opt<int64_t, 8><<<(grid_size+7) / 8, block_size, 0, stream>>>(num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, (const int64_t*)input, (int64_t*)output, indices_element_size, (const void*)indices);
                break;
            case 4:
                ppl_cukernel_gather_opt<int32_t, 4><<<(grid_size+3) / 4, block_size, 0, stream>>>(num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, (const int32_t*)input, (int32_t*)output, indices_element_size, (const void*)indices);
                break;
            case 2:
                ppl_cukernel_gather_opt<int16_t, 2><<<(grid_size+1) / 2, block_size, 0, stream>>>(num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, (const int16_t*)input, (int16_t*)output, indices_element_size, (const void*)indices);
                break;
            case 1:
                ppl_cukernel_gather<<<grid_size, block_size, 0, stream>>>(num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, (const int8_t*)input, (int8_t*)output, indices_element_size, (const void*)indices);
                break;
            default:
                break;
            }
            return ppl::common::RC_SUCCESS;
        case sizeof(int16_t):
            switch (coeff)
            {
            case 4:
                ppl_cukernel_gather_opt<int64_t, 4><<<(grid_size+3) / 4, block_size, 0, stream>>>(num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, (const int64_t*)input, (int64_t*)output, indices_element_size, (const void*)indices);
                break;
            case 2:
                ppl_cukernel_gather_opt<int32_t, 2><<<(grid_size+1) / 2, block_size, 0, stream>>>(num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, (const int32_t*)input, (int32_t*)output, indices_element_size, (const void*)indices);
                break;
            case 1:
                ppl_cukernel_gather<<<grid_size, block_size, 0, stream>>>(num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, (const int16_t*)input, (int16_t*)output, indices_element_size, (const void*)indices);
                break;
            default:
                break;
            }
            return ppl::common::RC_SUCCESS;
        // SWITCH_CASE(int8_t);
        // SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }

#undef SWITCH_CASE
#ifdef __MACACC__
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
#endif // __MACACC__
}
