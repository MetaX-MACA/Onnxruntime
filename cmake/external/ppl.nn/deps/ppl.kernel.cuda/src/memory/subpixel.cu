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

#include "cudakernel/memory/subpixel.h"
#include "cudakernel/common/memory_utils.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include "cudakernel/common/divmod_fast.h"

template <typename T>
__global__ void ppl_cukernel_subpixel_down(
    int64_t num_elems,
    int down_ratio,
    int num_output_dim,
    GArray<int64_t> output_strides,
    int out_chl, int out_hgt, int out_width,
    GArray<int64_t> input_strides,
    const T* input,
    T* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int output_idx[4] = {0, 0, 0, 0};
    output_idx[0] = index / output_strides[0];
    output_idx[1] = (index / output_strides[1]) % out_chl;
    output_idx[2] = (index / output_strides[2]) % out_hgt;
    output_idx[3] = index % out_width;

    int c_idx       = output_idx[1];
    int bottom_cidx = c_idx / (down_ratio * down_ratio);
    int hw_offset   = c_idx % (down_ratio * down_ratio);
    int sub_hidx    = hw_offset / down_ratio;
    int sub_widx    = hw_offset % down_ratio;

    int bottom_hidx = output_idx[2] * down_ratio + sub_hidx;
    int bottom_widx = output_idx[3] * down_ratio + sub_widx;

    int64_t input_index = output_idx[0] * input_strides[0] +
                          bottom_cidx * input_strides[1] + bottom_hidx * input_strides[2] +
                          bottom_widx * input_strides[3];

    output[index] = input[input_index];
}

template <typename T>
__global__ void ppl_cukernel_subpixel_up(
    int64_t num_elems,
    int up_ratio,
    DivModFast fast_up_ratio,
    int num_output_dim,
    GArray<int64_t> output_strides,
    GArray<DivModFast> fast_output_strides,
    int out_chl, int out_hgt, int out_width,
    DivModFast fast_out_chl,
    DivModFast fast_out_hgt,
    DivModFast fast_out_width,
    GArray<int64_t> input_strides,
    const T* input,
    T* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int output_idx[4] = {0, 0, 0, 0};
    //output_idx[0] = index / output_strides[0];
    //output_idx[1] = (index / output_strides[1]) % out_chl;
    //output_idx[2] = (index / output_strides[2]) % out_hgt;
    //output_idx[3] = index % out_width;
    output_idx[0] = fast_output_strides[0].div(index);
    output_idx[1] = fast_out_chl.mod(fast_output_strides[1].div(index));
    output_idx[2] = fast_out_hgt.mod(fast_output_strides[2].div(index));
    output_idx[3] = fast_out_width.mod(index);

    int c_idx = output_idx[1];
    int h_idx = output_idx[2], w_idx = output_idx[3];
    //int bottom_hidx = h_idx / up_ratio;
    //int bottom_widx = w_idx / up_ratio;
    int bottom_hidx = fast_up_ratio.div(h_idx);
    int bottom_widx = fast_up_ratio.div(w_idx);

    //int sub_hidx = h_idx % up_ratio;
    //int sub_widx = w_idx % up_ratio;
    int sub_hidx = fast_up_ratio.mod(h_idx);
    int sub_widx = fast_up_ratio.mod(w_idx);

    int bottom_cidx = (sub_hidx * up_ratio + sub_widx) +
                      c_idx * up_ratio * up_ratio;

    int64_t input_index = output_idx[0] * input_strides[0] +
                          bottom_cidx * input_strides[1] + bottom_hidx * input_strides[2] +
                          bottom_widx * input_strides[3];

    output[index] = input[input_index];
}

ppl::common::RetCode PPLCUDASubpixelDownForwardImp(
    cudaStream_t stream,
    int down_ratio,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    int64_t num_elems  = output_shape->CalcElementsIncludingPadding();
    int num_output_dim = output_shape->GetDimCount();
    GArray<int64_t> output_strides(num_output_dim);
    GArray<int64_t> input_strides(num_output_dim);

    int64_t acc_output_stride = 1;
    int64_t acc_input_stride  = 1;
    for (int it = num_output_dim - 1; it >= 0; --it) {
        if (input_shape->GetDim(it) == 1) {
            input_strides[it] = 0;
        } else {
            input_strides[it] = acc_input_stride;
        }
        output_strides[it] = acc_output_stride;
        acc_input_stride *= input_shape->GetDim(it);
        acc_output_stride *= output_shape->GetDim(it);
    }

    int block_size = 256;
    int grid_size  = (num_elems + block_size - 1) / block_size;
    int out_chl = output_shape->GetDim(1);
    int out_hgt = output_shape->GetDim(2);
    int out_width = output_shape->GetDim(3);

#define SWITCH_CASE(TYPE)                                                                                                  \
    case sizeof(TYPE): {                                                                                                   \
        ppl_cukernel_subpixel_down<<<grid_size, block_size, 0, stream>>>(                                                  \
            num_elems, down_ratio, num_output_dim, output_strides, out_chl, out_hgt,                                       \
            out_width, input_strides, (const TYPE*)input, (TYPE*)output);                                                  \
        return ppl::common::RC_SUCCESS;                                                                                    \
    }

    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        SWITCH_CASE(int8_t);
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }

#undef SWITCH_CASE
}

ppl::common::RetCode PPLCUDASubpixelUpForwardImp(
    cudaStream_t stream,
    int up_ratio,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    int64_t num_elems  = output_shape->CalcElementsIncludingPadding();
    int num_output_dim = output_shape->GetDimCount();
    GArray<int64_t> output_strides(num_output_dim);
    GArray<int64_t> input_strides(num_output_dim);
    GArray<DivModFast> fast_output_strides(num_output_dim);

    int64_t acc_output_stride = 1;
    int64_t acc_input_stride  = 1;
    for (int it = num_output_dim - 1; it >= 0; --it) {
        if (input_shape->GetDim(it) == 1) {
            input_strides[it] = 0;
        } else {
            input_strides[it] = acc_input_stride;
        }
        fast_output_strides[it] = DivModFast(acc_output_stride);
        output_strides[it] = acc_output_stride;
        acc_input_stride *= input_shape->GetDim(it);
        acc_output_stride *= output_shape->GetDim(it);
    }

    int block_size = 256;
    int grid_size  = (num_elems + block_size - 1) / block_size;
    int out_chl = output_shape->GetDim(1);
    int out_hgt = output_shape->GetDim(2);
    int out_width = output_shape->GetDim(3);

#define SWITCH_CASE(TYPE)                                                                                                \
    case sizeof(TYPE): {                                                                                                 \
        ppl_cukernel_subpixel_up<<<grid_size, block_size, 0, stream>>>(                                                  \
            num_elems, up_ratio, DivModFast(up_ratio), num_output_dim, output_strides,                                   \
            fast_output_strides, out_chl, out_hgt, out_width, DivModFast(out_chl),                                       \
            DivModFast(out_hgt), DivModFast(out_width), input_strides, (const TYPE*)input, (TYPE*)output);               \
        return ppl::common::RC_SUCCESS;                                                                                  \
    }

    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        SWITCH_CASE(int8_t);
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }

#undef SWITCH_CASE
}