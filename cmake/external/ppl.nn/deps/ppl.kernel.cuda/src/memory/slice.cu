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

#include "cudakernel/memory/slice.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "cudakernel/common/common.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

#define MAX_DIM_SIZE SLICE_PARAM_MAX_DIM_SIZE
#ifdef PPLNN_USE_MACA
#define SLICE_OPT
#endif
template <typename T>
__global__ void ppl_cukernel_slice(
    int64_t num_elems,
    int num_dims,
    SliceKernelParam param,
    GArray<int64_t> input_strides,
    const T* input,
    GArray<int64_t> output_strides,
    GArray<DivModFast> output_strides_fast,
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int output_idx[MAX_DIM_SIZE];
    int input_idx[MAX_DIM_SIZE];
    int idx, remain = index;
    for (int it = 0; it < num_dims; ++it) {
        output_strides_fast[it].divmod(remain, idx, remain);
        output_idx[it] = idx;
    }

    // copy output_idx to input_idx
    for (int it = 0; it < num_dims; ++it)
        input_idx[it] = output_idx[it];

    // calc input_idx according to axes[]
    for (int it = 0; it < param.axes_num; ++it) {
        int axis        = param.axes[it];
        input_idx[axis] = output_idx[axis] * param.steps[it] + param.starts[it];
    }

    int64_t input_offset  = 0;
    int64_t output_offset = 0;
    for (int it = 0; it < num_dims; ++it) {
        input_offset += input_idx[it] * input_strides[it];
        output_offset += output_idx[it] * output_strides[it];
    }
    output[output_offset] = input[input_offset];
}

bool isFastSliceSupported(const SliceKernelParam& param, int32_t num_dims) {
    if (num_dims != 4) return false;
    if (param.axes_num > 2) return false;
    for (int32_t i = 0; i < param.axes_num; ++i) {
        int axis        = param.axes[i];
        if (axis < (num_dims - 2)) return false;
        if (param.steps[i] != 1) return false;
        if (param.starts[i] != 0) return false;
    }
    return true;
}

template <typename T>
__global__ void ppl_cukernel_slice_fast_ndarray(const T* input, int src_height, int src_width,
    T* output, int dst_height, int dst_width) {
    int dst_hgt = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_wdt = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_hgt >= dst_height || dst_wdt >= dst_width) return;
    int b_idx = blockIdx.z;
    int dst_idx = b_idx * dst_height * dst_width + dst_hgt * dst_width + dst_wdt;
    int src_idx = b_idx * src_height * src_width + dst_hgt * src_width + dst_wdt;
    output[dst_idx] = input[src_idx];
}

template <typename T>
__global__ void ppl_cukernel_slice_fast_nhwc(int channel, const T* input, int src_height, int src_width,
    int src_pad_channel, T* output, int dst_height, int dst_width, int dst_pad_channel) {
    int dst_hw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_hgt = dst_hw_idx / dst_width;
    int dst_wdt = dst_hw_idx % dst_width;
    int chl_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chl_idx >= channel || dst_hgt >= dst_height) return;
    int b_idx = blockIdx.z;
    int dst_idx = (b_idx * dst_height * dst_width + dst_hgt * dst_width + dst_wdt) * dst_pad_channel + chl_idx;
    int src_idx = (b_idx * src_height * src_width + dst_hgt * src_width + dst_wdt) * src_pad_channel + chl_idx;
    output[dst_idx] = input[src_idx];
}

bool isOptSlice_float2_Supported(const SliceKernelParam& param, int32_t num_dims, const ppl::common::TensorShape* input_shape,  ppl::common::TensorShape* output_shape)
{
    if (num_dims != 3) return false;
    if (param.axes_num != 1) return false;
    if (param.axes[0] !=2)  return false;
    if (param.steps[0] !=1)  return false;
    if (param.starts[0] %2 != 0)  return false;
    if (input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT32) return false;
    if (output_shape->GetDim(2) % 2 != 0) return false;
    if (input_shape->GetDim(2) % 2 != 0) return false;
    return true;
}

__global__ void ppl_cukernel_slice_opt_float2(const float2* input, int src_height, int src_width,
    float2* output, int dst_height, int dst_width, int start_w) {
    int dst_hgt = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_wdt = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_hgt >= dst_height || dst_wdt >= dst_width) return;
    int b_idx = blockIdx.z;
    int dst_idx = b_idx * dst_height * dst_width + dst_hgt * dst_width + dst_wdt;
    int src_idx = b_idx * src_height * src_width + dst_hgt * src_width + dst_wdt + start_w;
    output[dst_idx].x = input[src_idx].x;
    output[dst_idx].y = input[src_idx].y;

}

bool isOptSliceSupported(const SliceKernelParam& param, int32_t num_dims) {
    if (num_dims != 3) return false;
    if (param.axes_num != 1) return false;
    if (param.axes[0] !=2)  return false;
    if (param.steps[0] !=1)  return false;
    return true;
}

template <typename T>
__global__ void ppl_cukernel_slice_opt(const T* input, int src_height, int src_width,
    T* output, int dst_height, int dst_width, int start_w) {
    int dst_hgt = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_wdt = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_hgt >= dst_height || dst_wdt >= dst_width) return;
    int b_idx = blockIdx.z;
    int dst_idx = b_idx * dst_height * dst_width + dst_hgt * dst_width + dst_wdt;
    int src_idx = b_idx * src_height * src_width + dst_hgt * src_width + dst_wdt + start_w;
    output[dst_idx] = input[src_idx];
}

#ifdef SLICE_OPT
template<class T>
__global__ void ppl_cukernel_slice_dim1_nhwc_bhw(const T* input, T* output,
    int batch_block, int sliced_batch_block, int padded_dim1, int sliced_dim1, DivModFast div_padded_sliced_dim1,
    int start, int step)
{
    int bdx = blockIdx.x;
    const T* input_start = input + bdx * batch_block;
    T* output_start = output + bdx * sliced_batch_block;
    for (int i = threadIdx.x; i < sliced_batch_block; i += blockDim.x) {
        int ch, ch_idx;
        div_padded_sliced_dim1.divmod(i, ch, ch_idx);
        int src_offset = ch * padded_dim1 + ch_idx * step + start;
        T out_val;
        if (ch_idx < sliced_dim1) {
            out_val = input_start[src_offset];
        } else {
            out_val = T(0);
        }
        output_start[i] = out_val;
    }
}

template<class T>
__global__ void ppl_cukernel_slice_dim1_nhwc_bhw_no_padding(const T* input, T* output,
    int batch_block, int sliced_batch_block, int dim1_rank, int sliced_dim1_rank,
    int start, int step)
{
    int bdx = blockIdx.x;
    const T* input_start = input + bdx * batch_block;
    T* output_start = output + bdx * sliced_batch_block;
    for (int i = threadIdx.x; i < sliced_batch_block; i += blockDim.x) {
        int ch = i >> sliced_dim1_rank;
        int ch_idx = i - (ch << sliced_dim1_rank);
        int src_offset = (ch << dim1_rank) + ch_idx * step + start;
        T out_val = input_start[src_offset];
        output_start[i] = out_val;
    }
}

int get_rank(int i) {
    if (i <= 1) return 0;
    int rank = 0;
    int j = i;
    while (j % 2 == 0 && j / 2 >= 1) {
        rank += 1;
        j /= 2;
    }
    if (j != 1) return 0;
    return rank;
}

template<class T>
int call_slice_nhwc(const ppl::common::TensorShape* input_shape,
    const ppl::common::TensorShape* output_shape,
    int64_t axes, int64_t start, int64_t end, int64_t step,
    const T* input, T* output, cudaStream_t stream) {
    int64_t input_pad = input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
    int64_t output_pad = output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
    int padded_dim1 = input_shape->GetDim(1) + input_pad, padded_sliced_dim1 = output_shape->GetDim(1) + output_pad;
    int dim1_rank = get_rank(padded_dim1);
    int sliced_dim1_rank = get_rank(padded_sliced_dim1);
    bool not_2n = dim1_rank == 0 || sliced_dim1_rank == 0;
    int batch_block = input_shape->GetDim(2) * input_shape->GetDim(3) * padded_dim1;
    int sliced_batch_block = output_shape->GetDim(2) * output_shape->GetDim(3) * padded_sliced_dim1;

    //balance grid size and sliced_batch_block size
    int grid_size = input_shape->GetDim(0);
    if (grid_size < 32) {
        grid_size *= output_shape->GetDim(2);
        batch_block = input_shape->GetDim(3) * padded_dim1;
        sliced_batch_block = output_shape->GetDim(3) * padded_sliced_dim1;
    }
    if (sliced_batch_block > 8192) {
        int b_size = output_shape->GetDim(2) * output_shape->GetDim(3);
        while (sliced_batch_block > 8192) {
            bool can_div = false;
            for (int i = 2; i < 8; i++) {
                if (b_size % i == 0) {
                    b_size /= i;
                    grid_size *= i;
                    can_div = true;
                    sliced_batch_block = b_size * padded_sliced_dim1;
                    batch_block = b_size * padded_dim1;
                    break;
                }
            }
            if (!can_div) break;
        }
    }

    int sliced_dim1 = output_shape->GetDim(1);
    bool no_padding = input_pad == 0 && output_pad == 0;
    int block_size = 256;
    if (sliced_batch_block >= 2048) block_size = 512;

    if (!no_padding || not_2n) {
        ppl_cukernel_slice_dim1_nhwc_bhw<T><<<grid_size, block_size, 0, stream>>>(
            input, output,
            batch_block, sliced_batch_block, padded_dim1, sliced_dim1, DivModFast(padded_sliced_dim1),
            start, step
        );
    } else {
        ppl_cukernel_slice_dim1_nhwc_bhw_no_padding<T><<<grid_size, block_size, 0, stream>>>(
            input, output,
            batch_block, sliced_batch_block, dim1_rank, sliced_dim1_rank,
            start, step
        );
    }
    return ppl::common::RC_SUCCESS;;
}

//slice on the rdim0, which means the last dim
template<class T>
__global__ void ppl_cukernel_slice_rdim0(const T* input, T* output,
    int slice_elems, int slice_batch, int rdim0, DivModFast div_sliced_rdim0,
    int start, int step)
{
    int bdx = blockIdx.x;
    T* output_start = output + bdx * slice_batch;
    const T* input_start = input + div_sliced_rdim0.div(bdx * slice_batch) * rdim0;
    for (int i = threadIdx.x; i < slice_batch; i += blockDim.x) {
        int line_idx;// = i / sliced_rdim0;
        int idx_in_line; //= i % sliced_rdim0;
        div_sliced_rdim0.divmod(i, line_idx, idx_in_line);
        int src_pixel = line_idx * rdim0 + idx_in_line * step + start;
        output_start[i] = input_start[src_pixel];
    }
}

template<class T>
int call_slice_rdim0(const ppl::common::TensorShape* input_shape,
    const ppl::common::TensorShape* output_shape,
    int64_t axes, int64_t start, int64_t end, int64_t step,
    const T* input, T* output, cudaStream_t stream) {
    int dims = output_shape->GetDimCount();
    int rdim0 = input_shape->GetDim(dims-1);
    int sliced_rdim0 = output_shape->GetDim(dims-1);
    int block_size = 512;
    int grid_size = 1;
    int64_t elems = 1;
    for (int i = 0; i < dims; i++) {
        elems *= output_shape->GetDim(i);
    }

    int batch_size = 1;
    for (int i = 2; i < dims; i++) batch_size *= output_shape->GetDim(i);
    const int min_data_size_per_block = 4096;
    int dim1 = output_shape->GetDim(1);
    //Trying to split from dim 1
    int i = 10;
    for (; i >= 1; i--) {
        if (dim1 % i == 0 && dim1 / i * batch_size >= min_data_size_per_block) {
            batch_size = batch_size * dim1 / i;
            break;
        }
    }
    if (i == 0) {
        i = 1;
        batch_size = batch_size * dim1;
        if (batch_size < min_data_size_per_block) block_size /= 2;
    }
    grid_size = output_shape->GetDim(0) * i;
    ppl_cukernel_slice_rdim0<<<grid_size, block_size>>>(
        input, output,
        elems, batch_size, rdim0, DivModFast(sliced_rdim0),
        start, step
    );
    return ppl::common::RC_SUCCESS;
}

template<class T>
__global__ void ppl_cukernel_slice_rdim(const T* input, T* output,
    int slice_elems, int slice_batch, int rdim, DivModFast div_sliced_rdim, int block_size, DivModFast div_block_size,
    int start, int step)
{
    int bdx = blockIdx.x;
    T* output_start = output + bdx * slice_batch * block_size;
    const T* input_start = input + div_sliced_rdim.div(bdx * slice_batch) * rdim *block_size;
    for (int i = threadIdx.x; i < slice_batch * block_size; i += blockDim.x) {
        int channels, idx_in_channels, imgs, idx_in_imgs;
        div_block_size.divmod(i, imgs, idx_in_imgs);
        div_sliced_rdim.divmod(imgs, channels, idx_in_channels);
        int src_pixel = channels * rdim * block_size + (idx_in_channels * step + start) * block_size + idx_in_imgs;
        output_start[i] = input_start[src_pixel];
    }
}

template<class T>
__global__ void ppl_cukernel_slice_dim1(const T* input, T* output,
    int slice_elems, int slice_batch, int rdim, DivModFast div_sliced_rdim, int block_size, DivModFast div_block_size,
    int start, int step)
{
    int bdx = blockIdx.x;
    T* output_start = output + bdx * slice_batch * block_size;
    int current_batch, current_channel;
    div_sliced_rdim.divmod(bdx*slice_batch, current_batch, current_channel);
    current_channel = current_channel * step + start;
    const T* input_start = input + current_batch * rdim *block_size;
    for (int i = threadIdx.x; i < slice_batch * block_size; i += blockDim.x) {
        int channels, idx_in_channels, imgs, idx_in_imgs;
        div_block_size.divmod(i, imgs, idx_in_imgs);
        div_sliced_rdim.divmod(imgs, channels, idx_in_channels);
        int src_pixel = channels * rdim * block_size + (idx_in_channels * step + current_channel) * block_size + idx_in_imgs;
        output_start[i] = input_start[src_pixel];
    }
}

template<class T>
__global__ void ppl_cukernel_slice_dim1_step1(const T* input, T* output,
    int slice_elems, int slice_batch, int rdim, DivModFast div_sliced_rdim, int block_size,
    int start)
{
    int bdx = blockIdx.x;
    T* output_start = output + bdx * slice_batch * block_size;
    int current_batch, current_channel;
    div_sliced_rdim.divmod(bdx*slice_batch, current_batch, current_channel);
    current_channel = current_channel + start;
    const T* input_start = input + current_batch * rdim *block_size + current_channel * block_size;
    for (int i = threadIdx.x; i < slice_batch * block_size; i += blockDim.x) {
        output_start[i] = input_start[i];
    }
}

template<class T>
int call_slice_rdim(const ppl::common::TensorShape* input_shape,
    const ppl::common::TensorShape* output_shape,
    int64_t axes, int64_t start, int64_t end, int64_t step,
    const T* input, T* output, cudaStream_t stream) {
    int dims = output_shape->GetDimCount();
    int sliced_block_size=1, slice_batch=1;

    for (int i = axes + 1; i < dims; i++) sliced_block_size *= output_shape->GetDim(i);
    int rdim = input_shape->GetDim(axes);
    int sliced_rdim = output_shape->GetDim(axes);
    int block_size = 512;
    int grid_size = 1;
    int64_t sliced_elems = 1;
    for (int i = 0; i <= axes; i++) {
        sliced_elems *= output_shape->GetDim(i);
    }
    for (int i = 2; i <= axes; i++) slice_batch *= output_shape->GetDim(i);

    int dim1 = output_shape->GetDim(1);
    const int min_data_size_per_block = 4096;
    int i = 10;
    for (; i >= 1; i--) {
        if (dim1 % i == 0 && dim1 / i * slice_batch * sliced_block_size >= min_data_size_per_block) {
            slice_batch = slice_batch * dim1 / i;
            break;
        }
    }
    if (i == 0) {
        i = 1;
        slice_batch = slice_batch * dim1;
        if (slice_batch < min_data_size_per_block) block_size /= 2;
    }
    grid_size = output_shape->GetDim(0) * i;
    if (axes == 1) {
        if (step == 1) {
            ppl_cukernel_slice_dim1_step1<<<grid_size, block_size>>>(
                input, output,
                sliced_elems, slice_batch, rdim, DivModFast(sliced_rdim), sliced_block_size,
                start
            );
        } else {
            ppl_cukernel_slice_dim1<<<grid_size, block_size>>>(
                input, output,
                sliced_elems, slice_batch, rdim, DivModFast(sliced_rdim), sliced_block_size, DivModFast(sliced_block_size),
                start, step
            );
        }
    } else {
        ppl_cukernel_slice_rdim<<<grid_size, block_size>>>(
            input, output,
            sliced_elems, slice_batch, rdim, DivModFast(sliced_rdim), sliced_block_size, DivModFast(sliced_block_size),
            start, step
        );
    }
    return ppl::common::RC_SUCCESS;;
}

template <typename T, typename DST_T, int N, int SHIFT, typename SRC_T, int S_N, int S_SHIFT>
__global__ void ppl_cukernel_slice_roi_opt(const T* input, int src_height, int src_width,
    T* output, int dst_height, int dst_width, int start_w, DivModFast dst_width_fast) {
    constexpr int N1 = 16 / sizeof(T);
    int b_idx = blockIdx.y;
    int dst_batch_offset = b_idx * dst_height * dst_width;
    int src_batch_offset = b_idx * src_height * src_width;
    int dst_offset_pb = blockIdx.x * blockDim.x * N1;
    int dst_len = min(dst_height * dst_width - dst_offset_pb, blockDim.x*N1);
    if(dst_len <= 0) return;
    T* ptr_block_output = output + dst_batch_offset + dst_offset_pb;
    const T* ptr_block_input = input + src_batch_offset + start_w;
    dst_len = dst_len >> SHIFT;
    for(int i = threadIdx.x; i < dst_len; i += blockDim.x) {
        int h, w;
        DST_T reg_dst;
        SRC_T *ptr_reg_dst = (SRC_T*)&reg_dst;
        dst_width_fast.divmod((dst_offset_pb + (i<<SHIFT)), h, w);
        const T* ptr_input = ptr_block_input + h * src_width + w;
        #pragma unroll
        for(int j = 0; j < N; j += S_N) {
            ptr_reg_dst[j>>S_SHIFT] = *(SRC_T*)(ptr_input + j);
        }
        *(DST_T*)(ptr_block_output + (i << SHIFT)) = reg_dst;
    }
}

template <typename T, typename VT, int N, int SHIFT>
__global__ void ppl_cukernel_slice_s2_opt(
    int64_t num_elems,
    int num_dims,
    SliceKernelParam param,
    GArray<int64_t> input_strides,
    const T* input,
    GArray<int64_t> output_strides,
    GArray<DivModFast> output_strides_fast,
    T* output)
{
    int index = (blockIdx.x * blockDim.x + threadIdx.x) << SHIFT;
    if(index >= num_elems) return;
    int io_idx[MAX_DIM_SIZE];
    int idx, remain = index;
    for(int it = 0; it < num_dims; ++it) {
        output_strides_fast[it].divmod(remain, idx, remain);
        io_idx[it] = idx;
    }
    for(int it = 0; it < param.axes_num - 1; ++it) {
        int axis = param.axes[it];
        io_idx[axis] = io_idx[axis] * param.steps[it] + param.starts[it];
    }
    int last_index = param.axes_num - 1;
    int axis = param.axes[last_index];
    io_idx[axis] = io_idx[axis] * param.steps[last_index];
    
    int input_offset = 0;
    for(int it = 0; it < num_dims; ++it) {
        input_offset += io_idx[it] * input_strides[it];
    }
    T buffer_cache[N<<1];
    *(VT*)(buffer_cache) = *(VT*)(input + input_offset);
    *(VT*)(buffer_cache + N) = *(VT*)(input + input_offset + N);
    VT dst;
    T* ptr_dst = (T*)&dst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        ptr_dst[i] = buffer_cache[i*param.steps[last_index] + param.starts[last_index]];
    }
    T* ptr_output = output + index;
    *(VT*)ptr_output = dst;
}
#endif//SLICE_OPT

ppl::common::RetCode PPLCUDASliceForwardImp(
    cudaStream_t stream,
    SliceKernelParam param,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    ppl::common::TensorShape* output_shape,
    void* output)
{
    if (output_shape->CalcElementsIncludingPadding() == 0)
        return ppl::common::RC_SUCCESS;
    int num_dims       = output_shape->GetDimCount();

    if (isOptSlice_float2_Supported(param, num_dims, input_shape, output_shape))
    {
            int batch = input_shape->CalcElementsToDimensionExcludingPadding(1);
            int dst_height = output_shape->GetDim(1);
            int dst_width  = output_shape->GetDim(2)/2;
            int src_height = input_shape->GetDim(1);
            int src_width  = input_shape->GetDim(2)/2;
            int start_w = param.starts[0]/2;
            dim3 block_size(1, 1024, 1);
            dim3 grid_size(DivUp(dst_width,1), DivUp(dst_height, 1024), batch);
            ppl_cukernel_slice_opt_float2<<<grid_size, block_size, 0, stream>>>(
                (const float2*)input, src_height, src_width, (float2*)output, dst_height, dst_width, start_w);
            return ppl::common::RC_SUCCESS;
    }

    if (isOptSliceSupported(param, num_dims))
    {
        #define SWITCH_CASE(TYPE) \
        case sizeof(TYPE):{ \
            int batch = input_shape->CalcElementsToDimensionExcludingPadding(1); \
            int dst_height = output_shape->GetDim(1); \
            int dst_width  = output_shape->GetDim(2); \
            int src_height = input_shape->GetDim(1); \
            int src_width  = input_shape->GetDim(2); \
            int start_w = param.starts[0]; \
            dim3 block_size(2, 512, 1); \
            dim3 grid_size(DivUp(dst_width,2), DivUp(dst_height, 512), batch); \
            ppl_cukernel_slice_opt<<<grid_size, block_size, 0, stream>>>( \
                (const TYPE*)input, src_height, src_width, (TYPE*)output, dst_height, dst_width, start_w); \
            return ppl::common::RC_SUCCESS; \
        }
#ifdef SLICE_OPT
        int batch = input_shape->CalcElementsToDimensionExcludingPadding(1);
        int dst_height = output_shape->GetDim(1);
        int dst_width  = output_shape->GetDim(2);
        int src_height = input_shape->GetDim(1);
        int src_width  = input_shape->GetDim(2);
        int start_w = param.starts[0];

        if(start_w + dst_width <= src_width)
        {
            int inputTypeSize = ppl::common::GetSizeOfDataType(input_shape->GetDataType());
            int blockPixel = 8192/ inputTypeSize;
            dim3 blockSize(512,1,1);
            dim3 gridSize((dst_width * dst_height + blockPixel - 1) / blockPixel, batch, 1);
            
#define SLICE_KERNEL_OPT(T,T1,N1,N2,T3,N3,N4) \
    ppl_cukernel_slice_roi_opt<T,T1,N1,N2,T3,N3,N4><<<gridSize,blockSize,0,stream>>>\
                        ((const T*)input,src_height,src_width,(T*)output,dst_height,dst_width,start_w,DivModFast(dst_width)); \
    return ppl::common::RC_SUCCESS;

            if(inputTypeSize == 2) {    
                if((dst_width&7)==0) {
                    if((src_width&7) == 0 && (start_w & 7) == 0) {
                        SLICE_KERNEL_OPT(short,float4,8,3,float4,8,3)
                    } else if((src_width&3)==0 && (start_w & 3) == 0) {
                        SLICE_KERNEL_OPT(short, float4, 8, 3, float2, 4, 2)
                    } else if((src_width&1) == 0 && (start_w & 1) == 0) {
                        SLICE_KERNEL_OPT(short, float4, 8, 3,float, 2, 1)
                    } else { 
                        SLICE_KERNEL_OPT(short, float4, 8, 3, short, 1, 0)
                    }
                } else if((dst_width&3) == 0) {
                    if((src_width&3) == 0 && (start_w & 3) == 0) {
                        SLICE_KERNEL_OPT(short, float2, 4, 2, float2, 4, 2)
                    } else if((src_width&1) == 0 && (start_w & 1) == 0) {
                        SLICE_KERNEL_OPT(short, float2, 4, 2, float, 2, 1)
                    } else {
                        SLICE_KERNEL_OPT(short, float2, 4, 2, short, 1, 0)
                    }
                } else {
                    SLICE_KERNEL_OPT(short, short, 1, 0, short, 1, 0)
                }
            } else if(inputTypeSize == 1) {
                if((dst_width&7)==0) {
                    if((src_width&7) == 0 && (start_w & 7) == 0) {
                        SLICE_KERNEL_OPT(int8_t, float2, 8, 3, float2, 8, 3)
                    } else if((src_width&3)==0 && (start_w & 3) == 0) {
                        SLICE_KERNEL_OPT(int8_t, float2, 8, 3, float, 4, 2)
                    } else if((src_width&1) == 0 && (start_w & 1) == 0) {
                        SLICE_KERNEL_OPT(int8_t, float2, 8, 3, short, 2, 1)
                    } else {
                        SLICE_KERNEL_OPT(int8_t, float2, 8, 3, int8_t, 1, 0)
                    }
                } else if((dst_width&3) == 0) {
                    if((src_width&3) == 0 && (start_w & 3) == 0) {
                        SLICE_KERNEL_OPT(int8_t, float, 4, 2, float, 4, 2)
                    } else if((src_width&1) == 0 && (start_w & 1) == 0) {
                        SLICE_KERNEL_OPT(int8_t, float, 4, 2, short, 2, 1)
                    } else {
                        SLICE_KERNEL_OPT(int8_t, float, 4, 2, int8_t, 1, 0)
                    }
                } else {
                    SLICE_KERNEL_OPT(int8_t, int8_t, 1, 0, int8_t, 1, 0)
                }
            } else if(inputTypeSize == 4) {
                if((dst_width&3) == 0) {
                    if((src_width&3) == 0 && (start_w & 3) == 0) {
                        SLICE_KERNEL_OPT(int32_t, float4, 4, 2, float4, 4, 2)
                    } else if((src_width&1) == 0 && (start_w & 1) == 0) {
                        SLICE_KERNEL_OPT(int32_t, float4, 4, 2, float2, 2, 1)
                    } else {
                        SLICE_KERNEL_OPT(int32_t, float4, 4, 2, int32_t, 1, 0)
                    }
                } else {
                    SLICE_KERNEL_OPT(int32_t, int32_t , 1, 0, int32_t, 1, 0)
                }
            } else if(inputTypeSize == 8) {
                SLICE_KERNEL_OPT(int64_t, int64_t, 1, 0, int64_t, 1, 0)
            }
#undef SLICE_KERNEL_OPT
        }
#endif//SLICE_OPT
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

    if (isFastSliceSupported(param, num_dims)) {
        #define SWITCH_CASE(TYPE) \
        case sizeof(TYPE): { \
            if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) { \
                int batch = input_shape->CalcElementsToDimensionExcludingPadding(num_dims - 2); \
                int dst_height = output_shape->GetDim(num_dims - 2); \
                int dst_width  = output_shape->GetDim(num_dims - 1); \
                int src_height = input_shape->GetDim(num_dims - 2); \
                int src_width  = input_shape->GetDim(num_dims - 1); \
                dim3 block_size(16, 16, 1); \
                dim3 grid_size(DivUp(dst_width, 16), DivUp(dst_height, 16), batch); \
                ppl_cukernel_slice_fast_ndarray<<<grid_size, block_size, 0, stream>>>( \
                    (const TYPE*)input, src_height, src_width, (TYPE*)output, dst_height, dst_width); \
                return ppl::common::RC_SUCCESS; \
            } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 || \
                        output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||\
                        output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) { \
                int batch = input_shape->GetDim(0); \
                int channel = input_shape->GetDim(1); \
                int src_pad_channel = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1); \
                int dst_pad_channel = output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1); \
                int dst_height = output_shape->GetDim(num_dims - 2); \
                int dst_width  = output_shape->GetDim(num_dims - 1); \
                int src_height = input_shape->GetDim(num_dims - 2); \
                int src_width  = input_shape->GetDim(num_dims - 1); \
                dim3 block_size(16, 16, 1); \
                dim3 grid_size(DivUp(channel, 16), DivUp(dst_height * dst_width, 16), batch); \
                ppl_cukernel_slice_fast_nhwc<<<grid_size, block_size, 0, stream>>>(channel, \
                    (const TYPE*)input, src_height, src_width, src_pad_channel, (TYPE*)output, dst_height, dst_width, dst_pad_channel); \
                return ppl::common::RC_SUCCESS; \
            } \
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
#ifdef SLICE_OPT
    bool nhwc = (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
        output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
        output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) && output_shape->GetDimCount() == 4;
    if (param.axes_num == 1 && num_dims >= 4 && param.axes[0] >= num_dims-3) {
        if (nhwc && param.axes[0] == 1) {
#define SWITCH_CASE(TYPE)                                                                                                       \
    case sizeof(TYPE): {                                                                                                        \
        return call_slice_nhwc(                                                                                                 \
            input_shape, output_shape,                                                                                          \
            param.axes[0], param.starts[0], param.ends[0], param.steps[0],                                                      \
            (const TYPE*)input, (TYPE*)output, stream);                                                                         \
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
        } else if (param.axes[0] == num_dims - 1 && !nhwc) {
#define SWITCH_CASE(TYPE)                                                                                                       \
    case sizeof(TYPE): {                                                                                                        \
        return call_slice_rdim0(                                                                                                \
            input_shape, output_shape,                                                                                          \
            param.axes[0], param.starts[0], param.ends[0], param.steps[0],                                                      \
            (const TYPE*)input, (TYPE*)output, stream);                                                                         \
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
        } else if (param.axes[0] >= 1 && !nhwc) {
#define SWITCH_CASE(TYPE)                                                                                                       \
    case sizeof(TYPE): {                                                                                                        \
        return call_slice_rdim(                                                                                                 \
            input_shape, output_shape,                                                                                          \
            param.axes[0], param.starts[0], param.ends[0], param.steps[0],                                                      \
            (const TYPE*)input, (TYPE*)output, stream);                                                                         \
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
    }
#endif
    int block_size     = 512;
    uint64_t num_elems = output_shape->CalcElementsExcludingPadding();
    int grid_size      = (num_elems + block_size - 1) / block_size;
    GArray<int64_t> input_strides(num_dims);
    GArray<int64_t> output_strides(num_dims);
    GArray<DivModFast> output_strides_fast(num_dims);
    int64_t acc_output_stride = 1;
    int64_t acc_input_stride  = 1;
    for (int it = num_dims - 1; it >= 0; --it) {
        input_strides[it]       = acc_input_stride;
        output_strides[it]      = acc_output_stride;
        output_strides_fast[it] = DivModFast(acc_output_stride);
        acc_input_stride *= input_shape->GetDim(it);
        acc_output_stride *= output_shape->GetDim(it);
    }
    
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
        output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
        output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
        acc_output_stride = 1;
        acc_input_stride  = 1;
        for (int it = num_dims - 1; it >= 0; --it) {
            if (it == num_dims - 1) {
                input_strides[1]  = acc_input_stride;
                output_strides[1] = acc_output_stride;
                acc_input_stride *= input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
                acc_output_stride *= output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
            } else if (it == 0) {
                input_strides[it]  = acc_input_stride;
                output_strides[it] = acc_output_stride;
                acc_input_stride *= input_shape->GetDim(it);
                acc_output_stride *= output_shape->GetDim(it);
            } else {
                input_strides[it + 1]  = acc_input_stride;
                output_strides[it + 1] = acc_output_stride;
                acc_input_stride *= input_shape->GetDim(it + 1);
                acc_output_stride *= output_shape->GetDim(it + 1);
            }
        }
    }
#ifdef SLICE_OPT
    else {
        int size_of_t = ppl::common::GetSizeOfDataType(input_shape->GetDataType());
        int N = 16 / size_of_t;
        if(param.axes[param.axes_num - 1] == num_dims - 1 && (output_shape->GetDim(num_dims - 1) & (N - 1)) == 0 && param.steps[param.axes_num - 1] == 2) {
            grid_size = (num_elems + block_size * N - 1) / (block_size * N);
            if(size_of_t == 1) {
                ppl_cukernel_slice_s2_opt<int8_t, float4, 16, 4><<<grid_size, block_size, 0, stream>>>(num_elems, num_dims, param, input_strides, (const int8_t*)input, output_strides, output_strides_fast, (int8_t*)output);
            } else if(size_of_t == 2) {
                ppl_cukernel_slice_s2_opt<short, float4, 8, 3><<<grid_size, block_size, 0, stream>>>(num_elems, num_dims, param, input_strides, (const short*)input, output_strides, output_strides_fast, (short*)output);
            } else if(size_of_t == 4){
                ppl_cukernel_slice_s2_opt<int, float4, 4, 2><<<grid_size, block_size, 0, stream>>>(num_elems, num_dims, param, input_strides, (const int*)input, output_strides, output_strides_fast, (int*)output);
            } else if(size_of_t == 8){
                ppl_cukernel_slice_s2_opt<int64_t, float4, 2, 1><<<grid_size, block_size, 0, stream>>>(num_elems, num_dims, param, input_strides, (const int64_t*)input, output_strides, output_strides_fast, (int64_t*)output);
            } else {
                return ppl::common::RC_UNSUPPORTED;
            }
            return ppl::common::RC_SUCCESS;
        }
    }
#endif//SLICE_OPT

#define SWITCH_CASE(TYPE)                                                                                                       \
    case sizeof(TYPE): {                                                                                                        \
        ppl_cukernel_slice<<<grid_size, block_size, 0, stream>>>(                                                               \
            num_elems, num_dims, param, input_strides, (const TYPE*)input, output_strides, output_strides_fast, (TYPE*)output); \
        return ppl::common::RC_SUCCESS;                                                                                         \
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
