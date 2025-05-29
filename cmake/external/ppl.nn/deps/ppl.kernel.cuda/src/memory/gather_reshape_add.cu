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

#include "cudakernel/memory/gather_reshape_add.h"
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

template <typename T, typename PackT,int Times>
__global__ void ppl_cukernel_GatherAdd_opt(
    int64_t num_elems,
    DivModFast output_outer_block_fast,
    int input_axis_size,
    DivModFast output_inner_block_fast,
    DivModFast add_outer_block_fast,
    const PackT* input,
    const PackT* input_add,
    PackT* output,
    int indices_element_size,
    const void* indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = Times * idx;
    if (index >= num_elems)
        return;
    int outer_idx, block_offset;
    output_outer_block_fast.divmod(index, outer_idx, block_offset);
    int indices_offset, inner_idx, add_outer_idx;
    output_inner_block_fast.divmod(block_offset, indices_offset, inner_idx);
    int64_t indices_idx = get_indices_val(indices_element_size, indices_offset, indices);

    add_outer_idx = add_outer_block_fast.mod(indices_offset);
    int add_idx = (add_outer_idx * output_inner_block_fast.d_ + inner_idx) / Times;
    PackT add_val = input_add[add_idx];

    // -d means distance from last dimension
    indices_idx         = indices_idx < 0 ? indices_idx + input_axis_size : indices_idx;
    if (indices_idx < 0 || indices_idx >= input_axis_size) {
        output[idx] = add_val;
        return;
    }
    int64_t input_idx = (outer_idx * input_axis_size + indices_idx) *
                            output_inner_block_fast.d_ +
                        inner_idx;
    PackT in_val = input[input_idx / Times];
    PackT out_val;
    T* ptr_in_val = (T*)&in_val;
    T* ptr_add_val = (T*)&add_val;
    T* ptr_out_val = (T*)&out_val;
    for (int i = 0; i < Times; i++){
        ptr_out_val[i] = ptr_in_val[i] + ptr_add_val[i];
    }
    output[idx] = out_val;
}

ppl::common::RetCode PPLCUDAGatResAddForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* indices_shape,
    const void* indices,
    const ppl::common::TensorShape* input_add_shape,
    const void* input_add,
    const ppl::common::TensorShape* output_shape,
    void* output,
    int axis)
{
    int indices_element_size = ppl::common::GetSizeOfDataType(indices_shape->GetDataType());
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
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
        int add_dim_count = input_add_shape->GetDimCount();
        int add_outer_block = input_add_shape->CalcElementsToDimensionIncludingPadding(add_dim_count - 1);
        DivModFast add_outer_block_fast(add_outer_block);

        switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
            case sizeof(int8_t):
                ppl_cukernel_GatherAdd_opt<int8_t, int64_t, 8><<<(grid_size+7) / 8, block_size, 0, stream>>>(
                        num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, add_outer_block_fast, 
                        (const int64_t*)input, (const int64_t*)input_add, (int64_t*)output, 
                        indices_element_size, (const void*)indices);
                return ppl::common::RC_SUCCESS;
            case sizeof(half):
                ppl_cukernel_GatherAdd_opt<half, float4, 8><<<(grid_size+7) / 8, block_size, 0, stream>>>(
                        num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, add_outer_block_fast, 
                        (const float4*)input, (const float4*)input_add, (float4*)output, 
                        indices_element_size, (const void*)indices);
                return ppl::common::RC_SUCCESS;
            case sizeof(float):
                ppl_cukernel_GatherAdd_opt<float, float4, 4><<<(grid_size+3) / 4, block_size, 0, stream>>>(
                        num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, add_outer_block_fast, 
                        (const float4*)input, (const float4*)input_add, (float4*)output, 
                        indices_element_size, (const void*)indices);
                return ppl::common::RC_SUCCESS;
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
    } else {
        // output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC / NHWC8 / NHWC16
        return ppl::common::RC_UNSUPPORTED;
    }
}