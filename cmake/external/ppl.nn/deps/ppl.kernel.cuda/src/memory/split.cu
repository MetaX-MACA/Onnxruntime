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

#include "cudakernel/memory/split.h"
#include "cudakernel/memory/slice.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/common.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#ifdef __MACACC__
#define __SPLIT_OPT__
#endif//__MACACC__

#define NHWC8_ALIGNED_AXIS (8)

#ifdef __SPLIT_OPT__
    template<typename T1, typename IT, int IN, int IS,  typename OT, int ON, int OS>
__global__ void __launch_bounds__(512) ppl_cukernel_split_nhwc_two_inputs_opt(
    int64_t height,
    int inner_dims,
    int pad_inner_dims,
    int axis_width0,
    int pad_axis_width0,
    int axis_width1,
    int pad_axis_width1,
    int line_per_block,
    DivModFast inner_dims_fast,
    const void* input,
    void* output0,
    void* output1)
{
    __shared__ int8_t sm_buffer[16384];
    T1* ptr_sm_input = (T1*)sm_buffer;
    T1* ptr_sm_output = (T1*)(sm_buffer + 8192);
    int64_t heightId = blockIdx.x * line_per_block;
    int block_line = min(height - heightId,line_per_block);
    if(block_line <= 0) return;
    int obs0 = block_line * pad_axis_width0;
    int obs1 = block_line * pad_axis_width1;
    T1* ptr_sm_output0 = ptr_sm_output;
    T1* ptr_sm_output1 = ptr_sm_output + obs0;
    
    int ibs = block_line * inner_dims;
    int ibs_pad = block_line * pad_inner_dims;
    int64_t src_offset = heightId * pad_inner_dims;
    const T1* ptr_block_input = (const T1*)input + src_offset;
    ibs_pad = ibs_pad >> IS;
    for(int i = threadIdx.x; i < ibs_pad; i += blockDim.x) {
        *(IT *)(ptr_sm_input + (i << IS)) = *(IT*)(ptr_block_input + (i << IS));
    }
    T1* ptr_block_output0 = (T1*)output0 + heightId * pad_axis_width0;
    T1* ptr_block_output1 = (T1*)output1 + heightId * pad_axis_width1;
    __syncthreads();

    for(int i = threadIdx.x; i < ibs; i += blockDim.x) {
        int inner_idx, outer_idx;
        inner_dims_fast.divmod(i, outer_idx, inner_idx);
        int input_offset = outer_idx * pad_inner_dims + inner_idx;
        if (inner_idx >= axis_width0) {
            int output_offset = outer_idx * pad_axis_width1 + (inner_idx - axis_width0);
            ptr_sm_output1[output_offset] = ptr_sm_input[input_offset];
        } else {
            int output_offset      = outer_idx * pad_axis_width0 + inner_idx;
            ptr_sm_output0[output_offset] = ptr_sm_input[input_offset];
        }
    }
    __syncthreads();
    obs0 = obs0 >> OS;
    obs1 = obs1 >> OS;
    for(int i = threadIdx.x; i < obs0; i += blockDim.x) {
        *(OT*)(ptr_block_output0 + (i << OS)) = *(OT*)(ptr_sm_output0 + (i << OS));
    }
    for(int i = threadIdx.x; i < obs1; i += blockDim.x) {
        *(OT*)(ptr_block_output1 + (i << OS)) = *(OT*)(ptr_sm_output1 + (i << OS));
    }
}
#endif//__SPLIT_OPT__

template <typename T1, typename T2>
__global__ void __launch_bounds__(256) ppl_cukernel_split_nhwc_two_inputs(
    int64_t num_elems,
    int inner_dims,
    int pad_inner_dims,
    int axis_width0,
    int pad_axis_width0,
    int axis_width1,
    int pad_axis_width1,
    T1* input,
    T2* output0,
    T2* output1)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int inner_idx    = i % inner_dims;
        int outer_idx    = i / inner_dims;
        int input_offset = outer_idx * pad_inner_dims + inner_idx;
        if (inner_idx >= axis_width0) {
            int output_offset      = outer_idx * pad_axis_width1 + (inner_idx - axis_width0);
            output1[output_offset] = input[input_offset];
        } else {
            int output_offset      = outer_idx * pad_axis_width0 + inner_idx;
            output0[output_offset] = input[input_offset];
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(512) ppl_cukernel_split_ndarray(
    int64_t num_elems,
    DivModFast inner_dims_fast,
    int in_split_axis_size,
    DivModFast out_split_axis_size_fast,
    int offset_split_axis,
    const T* input,
    T* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int inner_idx, outer_split_idx;
        inner_dims_fast.divmod(i, outer_split_idx, inner_idx);
        int split_idx, outer_idx;
        out_split_axis_size_fast.divmod(outer_split_idx, outer_idx, split_idx);
        int inner_dims   = inner_dims_fast.d_;
        int input_offset = outer_idx * in_split_axis_size * inner_dims +
                           (split_idx + offset_split_axis) * inner_dims + inner_idx;
        output[i] = input[input_offset];
    }
}

template <typename T>
__global__ void __launch_bounds__(512) ppl_cukernel_split_ndarray_opt(
    int64_t num_elems,
    DivModFast inner_dims_fast,
    int in_split_axis_size,
    DivModFast out_split_axis_size_fast,
    int offset_split_axis,
    const T* input,
    T* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int index = i * (16 / sizeof(T));
        int inner_idx, outer_split_idx;
        inner_dims_fast.divmod(index, outer_split_idx, inner_idx);
        int split_idx, outer_idx;
        out_split_axis_size_fast.divmod(outer_split_idx, outer_idx, split_idx);
        int inner_dims   = inner_dims_fast.d_;
        int input_offset = outer_idx * in_split_axis_size * inner_dims +
                           (split_idx + offset_split_axis) * inner_dims + inner_idx;
        const float4* ptr_in = (const float4*)(input + input_offset);
        float4* ptr_out = (float4*)(output + index);
        ptr_out[0] = ptr_in[0];
    }
}

ppl::common::RetCode PPLCUDASplitForwardImp(
    cudaStream_t stream,
    int split_axis,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    int num_outputs,
    const int64_t* out_dims[],
    void* outputs[])
{
    int64_t num_byte_elem = ppl::common::GetSizeOfDataType(input_shape->GetDataType());
    if (input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        int num_dims              = input_shape->GetDimCount();
        int64_t split_size        = 1;
        int64_t split_count       = 1;
        int64_t offset_split_axis = 0;
        int64_t split_axis_length = input_shape->GetDim(split_axis);
        for (int i = 0; i < split_axis; i++)
            split_count *= input_shape->GetDim(i);
        for (int i = split_axis + 1; i < num_dims; i++)
            split_size *= input_shape->GetDim(i);
        int opt_size = 16 / num_byte_elem; 
        bool use_opt = false;
        if((split_size & (opt_size - 1)) == 0){
            use_opt = true;
        }
        
#define SWITCH_CASE(TYPE)                                                                                                                                                                         \
    case sizeof(TYPE): {                                                                                                                                                                          \
        for (int i = 0; i < num_outputs; i++) {                                                                                                                                                   \
            TYPE* out_ptr             = static_cast<TYPE*>(outputs[i]);                                                                                                                           \
            const TYPE* input_ptr     = static_cast<const TYPE*>(input);                                                                                                                          \
            int out_split_axis_length = out_dims[i][split_axis];                                                                                                                                  \
            int split_size_with_axis  = out_split_axis_length * split_size;                                                                                                                       \
            int memcpy_threshold      = 64;                                                                                                                                                       \
            if (split_count > memcpy_threshold || split_size_with_axis < memcpy_threshold) {                                                                                                      \
                int block_size = 512;                                                                                                                                                             \
                int out_elems  = split_size_with_axis * split_count;                                                                                                                              \
                DivModFast split_size_fast(split_size);                                                                                                                                           \
                DivModFast out_split_axis_size_fast(out_split_axis_length);                                                                                                                       \
                if (use_opt) {                                                                                                                                                                    \
                    int grid_size  = (out_elems + (block_size * opt_size) - 1) / (block_size * opt_size);                                                                                         \
                    ppl_cukernel_split_ndarray_opt<<<grid_size, block_size, 0, stream>>>(out_elems / opt_size, split_size_fast, split_axis_length, out_split_axis_size_fast, offset_split_axis, input_ptr, out_ptr); \
                } else {                                                                                                                                                                          \
                    int grid_size  = (out_elems + (block_size * 8) - 1) / (block_size * 8);                                                                                                       \
                    ppl_cukernel_split_ndarray<<<grid_size, block_size, 0, stream>>>(out_elems, split_size_fast, split_axis_length, out_split_axis_size_fast, offset_split_axis, input_ptr, out_ptr); \
                }                                                                                                                                                                                 \
            } else {                                                                                                                                                                              \
                for (int n = 0; n < split_count; n++) {                                                                                                                                           \
                    int64_t out_offset = n * split_size_with_axis;                                                                                                                                \
                    int64_t in_offset  = (n * split_axis_length + offset_split_axis) * split_size;                                                                                                \
                    cudaMemcpyAsync(out_ptr + out_offset, input_ptr + in_offset, split_size_with_axis * num_byte_elem, cudaMemcpyDeviceToDevice, stream);                                         \
                }                                                                                                                                                                                 \
            }                                                                                                                                                                                     \
            offset_split_axis += out_split_axis_length;                                                                                                                                           \
        }                                                                                                                                                                                         \
        return ppl::common::RC_SUCCESS;                                                                                                                                                           \
    }

        switch (num_byte_elem) {
            SWITCH_CASE(int8_t);
            SWITCH_CASE(int16_t);
            SWITCH_CASE(int32_t);
            SWITCH_CASE(int64_t);
            default:
                return ppl::common::RC_UNSUPPORTED;
        }

#undef SWITCH_CASE
    } else if (input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
               input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
        int num_dims = input_shape->GetDimCount();
        if (num_dims < 2)
            return ppl::common::RC_UNSUPPORTED;
        int input_elems = input_shape->CalcElementsExcludingPadding();
        if (num_outputs == 2 && split_axis == 1) {
            int align_size = NHWC8_ALIGNED_AXIS;
            if (input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
                align_size = NHWC8_ALIGNED_AXIS * 2;
            }else if (input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
                align_size = 1;
            }
#define SWITCH_CASE(TYPE)                                                                            \
    case sizeof(TYPE): {                                                                             \
        int block_size      = 512;                                                                   \
        int grid_size       = (input_elems + block_size - 1) / block_size;                           \
        int axis_width0     = out_dims[0][split_axis];                                               \
        int pad_axis_width0 = Align(axis_width0, align_size);                                \
        int axis_width1     = out_dims[1][split_axis];                                               \
        int pad_axis_width1 = Align(axis_width1, align_size);                                \
        int inner_dims      = axis_width0 + axis_width1;                                             \
        int pad_inner_dims  = Align(inner_dims, align_size);                                 \
        ppl_cukernel_split_nhwc_two_inputs<<<grid_size, block_size, 0, stream>>>(input_elems,        \
                                                                                 inner_dims,         \
                                                                                 pad_inner_dims,     \
                                                                                 axis_width0,        \
                                                                                 pad_axis_width0,    \
                                                                                 axis_width1,        \
                                                                                 pad_axis_width1,    \
                                                                                 (const TYPE*)input, \
                                                                                 (TYPE*)outputs[0],  \
                                                                                 (TYPE*)outputs[1]); \
        return ppl::common::RC_SUCCESS;                                                              \
    }

#ifdef __SPLIT_OPT__
    int axis_width0     = out_dims[0][split_axis]; 
    int pad_axis_width0 = Align(axis_width0, align_size); 
    int axis_width1     = out_dims[1][split_axis];
    int pad_axis_width1 = Align(axis_width1, align_size);

    int BlockSize = 512; 
    int block_line = 8192 / num_byte_elem / (pad_axis_width0 + pad_axis_width1);
    int inner_dims = axis_width0 + axis_width1;
    int pad_inner_dims  = Align(inner_dims, align_size);
    int64_t height = input_elems / inner_dims;
    int GridSize       = (height + block_line - 1) / block_line;
    int use_opt = 0;
    if(block_line >= 1) {
        if(num_byte_elem == 2) {
            if(!(align_size & 7) || (!(pad_axis_width0 & 7) && !(pad_axis_width1 & 7) && !(pad_inner_dims & 7))) {
                ppl_cukernel_split_nhwc_two_inputs_opt<short,float4,8,3,float4,8,3><<<GridSize,BlockSize,0,stream>>>(height, inner_dims,pad_inner_dims, axis_width0,pad_axis_width0,axis_width1,pad_axis_width1,
                    block_line,DivModFast(inner_dims),input, outputs[0],outputs[1]);
                use_opt = 1;
            }
            else if(!(align_size & 3) || (!(pad_axis_width0 & 3) && !(pad_axis_width1 & 3) && !(pad_inner_dims & 3))) {
                ppl_cukernel_split_nhwc_two_inputs_opt<short,float2,4,2,float2,4,2><<<GridSize,BlockSize,0,stream>>>(height, inner_dims,pad_inner_dims, axis_width0,pad_axis_width0,axis_width1,pad_axis_width1,
                    block_line,DivModFast(inner_dims),input, outputs[0],outputs[1]);
                use_opt = 1;
            } else {
                ppl_cukernel_split_nhwc_two_inputs_opt<short,short,1,0,short,1,0><<<GridSize,BlockSize,0,stream>>>(height, inner_dims,pad_inner_dims, axis_width0,pad_axis_width0,axis_width1,pad_axis_width1,
                    block_line,DivModFast(inner_dims),input, outputs[0],outputs[1]);
                use_opt = 1;
            }
        } else if(num_byte_elem == 1) {
            if(!(align_size & 15) || (!(pad_axis_width0 & 15) && !(pad_axis_width1 & 15) && !(pad_inner_dims & 15))) {
                ppl_cukernel_split_nhwc_two_inputs_opt<int8_t,float4,16,4,float4,16,4><<<GridSize,BlockSize,0,stream>>>(height, inner_dims,pad_inner_dims, axis_width0,pad_axis_width0,axis_width1,pad_axis_width1,
                    block_line,DivModFast(inner_dims),input, outputs[0],outputs[1]);
                use_opt = 1;
            } else if(!(align_size & 7) || (!(pad_axis_width0 & 7) && !(pad_axis_width1 & 7) && !(pad_inner_dims & 7))) {
                ppl_cukernel_split_nhwc_two_inputs_opt<int8_t,float2,8,3,float2,8,3><<<GridSize,BlockSize,0,stream>>>(height, inner_dims,pad_inner_dims, axis_width0,pad_axis_width0,axis_width1,pad_axis_width1,
                    block_line,DivModFast(inner_dims),input, outputs[0],outputs[1]);
                use_opt = 1;
            }
            else if(!(align_size & 3) || (!(pad_axis_width0 & 3) && !(pad_axis_width1 & 3) && !(pad_inner_dims & 3))) {
                ppl_cukernel_split_nhwc_two_inputs_opt<int8_t,float,4,2,float,4,2><<<GridSize,BlockSize,0,stream>>>(height, inner_dims,pad_inner_dims, axis_width0,pad_axis_width0,axis_width1,pad_axis_width1,
                    block_line,DivModFast(inner_dims),input, outputs[0],outputs[1]);
                use_opt = 1;
            } else {
                ppl_cukernel_split_nhwc_two_inputs_opt<int8_t,int8_t,1,0,int8_t,1,0><<<GridSize,BlockSize,0,stream>>>(height, inner_dims,pad_inner_dims, axis_width0,pad_axis_width0,axis_width1,pad_axis_width1,
                    block_line,DivModFast(inner_dims),input, outputs[0],outputs[1]);
                use_opt = 1;
            }
        } else if(num_byte_elem == 4) {
            ppl_cukernel_split_nhwc_two_inputs_opt<int32_t,int32_t,1,0,int32_t,1,0><<<GridSize,BlockSize,0,stream>>>(height, inner_dims,pad_inner_dims, axis_width0,pad_axis_width0,axis_width1,pad_axis_width1,
                block_line,DivModFast(inner_dims),input, outputs[0],outputs[1]);
            use_opt = 1;
        } else if(num_byte_elem == 8) {
            ppl_cukernel_split_nhwc_two_inputs_opt<long,long,1,0,long,1,0><<<GridSize,BlockSize,0,stream>>>(height, inner_dims,pad_inner_dims, axis_width0,pad_axis_width0,axis_width1,pad_axis_width1,
                block_line,DivModFast(inner_dims),input, outputs[0],outputs[1]);
            use_opt = 1;
        }
    }
    if(use_opt) return ppl::common::RC_SUCCESS;
#endif//__SPLIT_OPT__

            switch (num_byte_elem) {
                SWITCH_CASE(int8_t);
                SWITCH_CASE(int16_t);
                SWITCH_CASE(int32_t);
                SWITCH_CASE(int64_t);
                default:
                    return ppl::common::RC_UNSUPPORTED;
            }
#undef SWITCH_CASE
        }

        ppl::common::TensorShape output_shape(*input_shape);

        SliceKernelParam param;
        param.axes_num  = 1;
        param.starts[0] = 0;
        param.ends[0]   = input_shape->GetDim(split_axis);
        param.axes[0]   = split_axis;
        param.steps[0]  = 1;

        int64_t offset_split_axis = 0;
        for (int i = 0; i < num_outputs; i++) {
            output_shape.Reshape(out_dims[i], num_dims);
            output_shape.CalcPadding();
            param.starts[0]           = offset_split_axis;
            int out_split_axis_length = out_dims[i][split_axis];
            offset_split_axis += out_split_axis_length;
            param.ends[0] = offset_split_axis;
            PPLCUDASliceForwardImp(stream, param, input_shape, input, &output_shape, outputs[i]);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}