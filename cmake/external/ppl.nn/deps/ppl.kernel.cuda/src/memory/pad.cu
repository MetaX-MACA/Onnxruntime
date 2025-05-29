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

#include "cudakernel/memory/pad.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "cudakernel/common/common.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>
#ifdef __MACACC__
#define OPT_PAD
#endif//__MACACC__

template <int MODE>
__device__ int pad_calc_in_idx(int out_idx, int64_t start_pad_val, int64_t input_dim, bool& use_pad_value) {
    int res = 0;
    if (out_idx < start_pad_val || out_idx >= start_pad_val + input_dim)
        use_pad_value = true;
    else
        res = out_idx - start_pad_val;
    return res;
}

// PadKernelParam::PAD_MODE_REFLECT --> 1
template <>
__device__ int pad_calc_in_idx<PadKernelParam::PAD_MODE_REFLECT>(int out_idx, int64_t start_pad_val, int64_t input_dim, bool& use_pad_value) {
    int res = 0;
    if (out_idx < start_pad_val) {
        res = start_pad_val - out_idx;
    } else if (out_idx >= start_pad_val + input_dim) {
        res = input_dim - 2 - (out_idx - (start_pad_val + input_dim));
    } else {
        res = out_idx - start_pad_val;
    }
    return res;
}

// PadKernelParam::PAD_MODE_EDGE --> 1
template <>
__device__ int pad_calc_in_idx<PadKernelParam::PAD_MODE_EDGE>(int out_idx, int64_t start_pad_val, int64_t input_dim, bool& use_pad_value) {
    int res = 0;
    if (out_idx < start_pad_val) {
        res = 0;
    } else if (out_idx >= start_pad_val + input_dim) {
        res = input_dim - 1;
    } else {
        res = out_idx - start_pad_val;
    }
    return res;
}

template <typename T, int MODE>
__global__ void ppl_cukernel_pad(
    int64_t num_elems,
    int num_dims,
    PadKernelParam param,
    GArray<int64_t> input_dims,
    GArray<int64_t> input_strides,
    const T* input,
    const int64_t* pads,
    GArray<DivModFast> output_strides_fast,
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    bool use_pad_value   = false;
    int64_t input_offset = 0;
    int out_idx, remain = index;
    for (int it = 0; (it < num_dims) && !use_pad_value; ++it) {
        output_strides_fast[it].divmod(remain, out_idx, remain);
        int64_t start_pad_val = pads[it];
        int in_idx            = 0;
        in_idx = pad_calc_in_idx<MODE>(out_idx, start_pad_val, input_dims[it], use_pad_value);
        input_offset += in_idx * input_strides[it];
    }
    output[index] = use_pad_value ? (T)param.constant_value : input[input_offset];
}

template <typename T, int MODE>
__global__ void ppl_cukernel_pad_fast3(
    int64_t num_elems,
    int num_dims,
    int vec_factor,
    int vec_dim_size,
    PadKernelParam param,
    GArray<int64_t> input_dims,
    GArray<int64_t> input_strides,
    const float4* input,
    const int64_t* pads,
    GArray<DivModFast> output_strides_fast,
    float4* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    bool use_pad_value   = false;
    float4 constant_val;
    T* constant_val_ptr = reinterpret_cast<T*>(&constant_val);
    for (int i = 0; i < vec_factor; ++i) {
        constant_val_ptr[i] = (T)param.constant_value;
    }
    int64_t input_offset = 0;
    int out_idx, remain = index;
    for (int it = 0; (it < num_dims) && !use_pad_value; ++it) {
        output_strides_fast[it].divmod(remain, out_idx, remain);
        int64_t start_pad_val = pads[it];
        int in_idx            = 0;
        in_idx = pad_calc_in_idx<MODE>(out_idx, start_pad_val, input_dims[it], use_pad_value);
        input_offset += in_idx * input_strides[it];
    }
    input_offset *= vec_dim_size;
    for (int i = 0; i < vec_dim_size; ++i) {
        output[index * vec_dim_size + i] = use_pad_value ? constant_val : input[input_offset + i];
    }

}

bool isFastPadSupported(const std::vector<int32_t>& pads, int32_t num_dims) {
    if (num_dims < 3) return false;
    int32_t diff_cnt = num_dims - 2;
    for (int32_t i = 0; i < diff_cnt; ++i) {
        if (pads[i] != 0) return false; // start
        if (pads[num_dims + i] != 0) return false; //end
    }
    if (pads[num_dims - 1] != 0) return false;
    if (pads[num_dims - 2] != 0) return false;
    return true;
}

template <typename T>
__global__ void ppl_cukernel_pad_fast(const T* input, int src_height, int src_width,
    T* output, int dst_height, int dst_width) {
    int dst_hgt = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_wdt = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_hgt >= dst_height || dst_wdt >= dst_width) return;
    int b_idx = blockIdx.z;
    int dst_idx = b_idx * dst_height * dst_width + dst_hgt * dst_width + dst_wdt;
    if (dst_hgt >= src_height || dst_wdt >= src_width) {
        output[dst_idx] = T(0);
    } else {
        int src_idx = b_idx * src_height * src_width + dst_hgt * src_width + dst_wdt;
        output[dst_idx] = input[src_idx];
    }
}

#ifdef __MACACC__
template<typename T, int IOCOMPAT>
__device__ __forceinline__ void fast_copy_to(T* dst, const T* src, int len, int tid, int thread_stride) {
    for (int i = tid; i < len; i+= thread_stride) dst[i] =src[i];
}


template<>
__device__ __forceinline__ void fast_copy_to<int8_t, 16>(int8_t *dst, const int8_t *src, int len, int tid, int thread_stride) {
    uint64_t* sp = (uint64_t*)dst;
    const uint64_t* ip = (uint64_t*)src;
    for (int i = tid; i < len/8; i+= thread_stride) sp[i] = ip[i];
}


template<>
__device__ __forceinline__ void fast_copy_to<int8_t, 8>(int8_t *dst, const int8_t *src, int len, int tid, int thread_stride) {
    uint64_t* sp = (uint64_t*)dst;
    const uint64_t* ip = (uint64_t*)src;
    for (int i = tid; i < len/8; i+= thread_stride) sp[i] = ip[i];
}

template<>
__device__ __forceinline__ void fast_copy_to<int8_t, 4>(int8_t *dst, const int8_t *src, int len, int tid, int thread_stride) {
    uint32_t* sp = (uint32_t*)dst;
    const uint32_t* ip = (uint32_t*)src;
    for (int i = tid; i < len/4; i+= thread_stride) sp[i] = ip[i];
}

template<typename T, int LINES_PER_WARP>
__global__ void ppl_cukernel_pad_fast_opt(int num_lines, const T* input, int src_height, int src_width,
    T* output, int dst_height, int dst_width) {
    __shared__ T shared[16*1024/sizeof(T)];
    const int WARPS_PER_BLOCK = blockDim.z;
    const int tiny_warp_thread_count = blockDim.x;
    const int LINE_SIZE = (dst_width + 127) / 128 * 128;
    const int WARP_SIZE = 64;
    T* s1 = shared;
    T* s2 = shared+8*1024/sizeof(T);

    int warp_id = blockIdx.x * WARPS_PER_BLOCK + threadIdx.z;
    int warp_idx_in_block = threadIdx.z;
    int warp_line_start = warp_id * LINES_PER_WARP;
    if (warp_line_start >= num_lines) return;
    int tiny_warp_idx = threadIdx.y;
    int tiny_warp_tid = threadIdx.x;

    const T* input_src = input + warp_line_start * src_width;
    T* this_s1 = s1 + warp_idx_in_block * LINE_SIZE * LINES_PER_WARP;
    fast_copy_to<T,LINES_PER_WARP>(this_s1, input_src, src_width*LINES_PER_WARP, tiny_warp_idx * tiny_warp_thread_count + tiny_warp_tid, WARP_SIZE);

    __syncthreads();

    //Re-arrange data in s1 into s2
    int8_t *copy_from_addr = s1 + src_width * tiny_warp_idx + warp_idx_in_block * LINE_SIZE * LINES_PER_WARP;
    int8_t *copy_to_addr = s2 + dst_width * tiny_warp_idx + warp_idx_in_block * LINE_SIZE * LINES_PER_WARP;
    for (int i = tiny_warp_tid; i < src_width; i+=tiny_warp_thread_count) copy_to_addr[i] = copy_from_addr[i];
    for (int i = src_width + tiny_warp_tid; i < dst_width; i+=tiny_warp_thread_count) copy_to_addr[i] = 0;
    __syncthreads();
    T *dp = output + warp_line_start * dst_width;
    T *this_s2 = s2+warp_idx_in_block * LINE_SIZE * LINES_PER_WARP;
    fast_copy_to<T, LINES_PER_WARP>(dp, this_s2, dst_width*LINES_PER_WARP, tiny_warp_idx * tiny_warp_thread_count + tiny_warp_tid, WARP_SIZE);
}
#endif


// last 2-dim padded
bool isFastPadSupported2(const std::vector<int32_t>& pads, int32_t num_dims) {
    if (num_dims < 3) return false;
    int32_t diff_cnt = num_dims - 2;
    for (int32_t i = 0; i < diff_cnt; ++i) {
        if (pads[i] != 0) return false; // start
        if (pads[num_dims + i] != 0) return false; //end
    }
    return true;
}

template <typename T, int MODE>
__global__ void ppl_cukernel_pad_fast2(const T* input, int src_height, int src_width,
    T* output, int dst_height, int dst_width, int num_dims, const int64_t* pads, PadKernelParam param) {
    int dst_hgt = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_wdt = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_hgt >= dst_height || dst_wdt >= dst_width) return;
    int b_idx = blockIdx.z;
    int dst_idx = b_idx * dst_height * dst_width + dst_hgt * dst_width + dst_wdt;
    bool use_pad_value = false;
    int in_hgt = pad_calc_in_idx<MODE>(dst_hgt, pads[num_dims - 2], src_height, use_pad_value);
    int in_wdt = pad_calc_in_idx<MODE>(dst_wdt, pads[num_dims - 1], src_width, use_pad_value);
    int src_idx = b_idx * src_height * src_width + in_hgt * src_width + in_wdt;
    output[dst_idx] = use_pad_value ? (T)param.constant_value : input[src_idx];
}

#ifdef __MACACC__
template<int MODE>
__global__ void ppl_cukernel_pad_fast2_int8_opt(const int8_t* input, int src_height, int src_width,
    int8_t* output, int dst_height, int dst_width, int num_dims, const int64_t* pads, PadKernelParam param, int images_per_block, DivModFast dsz_mod, DivModFast dst_width_mod) {
    __shared__ int8_t s1[16*1024];
    __shared__ int8_t s2[16*1024];
    int block_image_start_idx = images_per_block * blockIdx.x;
    int block_size = blockDim.x;
    int tid = threadIdx.x;
    const int ssz = src_width * src_height;
    const int dsz = dst_width * dst_height;
    fast_copy_to<int8_t, 4>(s1, input + block_image_start_idx * ssz, ssz*images_per_block, tid, block_size);

    __syncthreads();

    //pads and param may stores in private memory
    //use them in loop will slow kernel dramatically
    int w_pad = pads[num_dims - 1];
    int h_pad = pads[num_dims - 2];
    int8_t pad_val = (int8_t)param.constant_value;
    for (int i = tid; i < dsz*images_per_block; i+= block_size) {
        int image_idx = 0, pixels = 0, w = 0, h = 0;
        dsz_mod.divmod(i, image_idx, pixels);
        dst_width_mod.divmod(pixels, h, w);
        bool use_pad_value = false;
        int in_h = pad_calc_in_idx<MODE>(h, h_pad, src_height, use_pad_value);
        int in_w = pad_calc_in_idx<MODE>(w, w_pad, src_width, use_pad_value);
        int8_t val = use_pad_value ? pad_val : s1[in_w+in_h*src_width+image_idx*ssz];
        s2[i] = val;
    }

    __syncthreads();

    //Finally, copy all images from s2 to output
    fast_copy_to<int8_t, 4>(output + block_image_start_idx * dsz, s2, dsz*images_per_block, tid, block_size);
}

__global__ void ppl_cukernel_pad_nhwc16_opt(const float4* input, int src_height, int src_width,
                                            float4* output, int dst_height, int dst_width,
                                            int num_elems,
                                            int height_width_channels,
                                            int width_channels,
                                            int channels,
                                            DivModFast channels_mod,
                                            DivModFast dst_width_mod,
                                            DivModFast dst_height_mod) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) {
        return;
    }

    int outer_idx, inner_idx, outer_idx_w, inner_idx_w, outer_idx_h, inner_idx_h;
    channels_mod.divmod(index, outer_idx, inner_idx);
    dst_width_mod.divmod(outer_idx, outer_idx_w, inner_idx_w);
    dst_height_mod.divmod(outer_idx_w, outer_idx_h, inner_idx_h);
    if (inner_idx_w < src_width && inner_idx_h < src_height) {
        int input_offset = outer_idx_h * height_width_channels + inner_idx_h * width_channels + inner_idx_w * channels + inner_idx;
        output[index] = input[input_offset];
    } else {
        output[index] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
}

template <typename T, int MODE, int N>
__global__ void ppl_cukernel_pad_fast_opt2(const T* input, int src_height, int src_width,
    T* output, int dst_height, int dst_width, int num_dims, const int64_t* pads, PadKernelParam param, int batch) {
    int dst_hgt = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_wdt = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_hgt >= dst_height || dst_wdt >= dst_width) return;
    int64_t pad_h = pads[num_dims - 2];
    int64_t pad_w = pads[num_dims - 1];
    int b_idx = blockIdx.z * N;
    int in_image_size = src_height * src_width;
    int out_image_size = dst_height * dst_width;
    int dst_idx = b_idx * out_image_size + dst_hgt * dst_width + dst_wdt;
    bool use_pad_value = false;
    int in_hgt = pad_calc_in_idx<MODE>(dst_hgt, pad_h, src_height, use_pad_value);
    int in_wdt = pad_calc_in_idx<MODE>(dst_wdt, pad_w, src_width, use_pad_value);
    int src_idx = b_idx * in_image_size + in_hgt * src_width + in_wdt;
    T r[N];
    T reg_value = (T)param.constant_value;
    #pragma unroll N
    for(int b = 0; b < N; b++) {
        r[b] = reg_value;
    }
    int gridDim_z = gridDim.z;
    if(blockIdx.z != gridDim_z - 1) {
        if(!use_pad_value) {
            const T * ptr_input = input + src_idx;
            #pragma unroll N
            for(int b = 0; b < N; b++) {
                r[b] = ptr_input[0];
                ptr_input += in_image_size;
            }
        }
        T* ptr_output = output + dst_idx;
        #pragma unroll N
        for(int b = 0; b < N; b++) {
            ptr_output[0] = r[b];
            ptr_output += out_image_size;
        }
    } else {
        int length = min(batch - blockIdx.z * N, N);
        if(!use_pad_value) {
            const T * ptr_input = input + src_idx;
            for(int b = 0; b < length; b++) {
                r[b] = ptr_input[0];
                ptr_input += in_image_size;
            }
        }
        T* ptr_output = output + dst_idx;
        for(int b = 0; b < length; b++) {
            ptr_output[0] = r[b];
            ptr_output += out_image_size;
        }
    }
}
#endif

template<typename T, typename VT, int MODE, int N>
__global__ void ppl_cukernel_pad_nhwc_opt(const T* input, int src_height, int src_width,
    T* output, int dst_height, int dst_width, int num_dims, const int64_t* pads, PadKernelParam param, int channels,
    DivModFast channels_fast, DivModFast width_fast, int blockDim_x,int num_elems)
{
    int batch = blockIdx.y;
    int index = blockIdx.x * blockDim_x * N + threadIdx.x * N;
    if(index >= num_elems) return;
    int batch_offset = batch * num_elems + index;
    int out_h, out_w, out_c;
    int temp;
    channels_fast.divmod(index, temp, out_c);
    width_fast.divmod(temp, out_h, out_w);
    bool use_pad_value = false;
    int in_hgt = pad_calc_in_idx<MODE>(out_h, pads[num_dims - 2], src_height, use_pad_value);
    int in_wdt = pad_calc_in_idx<MODE>(out_w, pads[num_dims - 1], src_width, use_pad_value);
    VT reg_dst;
    T *ptr_reg_dst = (T*)&reg_dst;
    T reg_value = (T)param.constant_value;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        ptr_reg_dst[i] = reg_value;
    }
    if(!use_pad_value){
        int src_idx = (batch * src_height * src_width + in_hgt * src_width + in_wdt) * channels + out_c;
        reg_dst = *(VT*)(input + src_idx);
    }
    *(VT*)(output + batch_offset) = reg_dst;
}

// last n-dim not padded
bool isFastPadSupported3(ppl::common::TensorShape* input_shape,
        const std::vector<int32_t>& pads, int32_t num_dims, int32_t& num_last) {
    constexpr int float4_as_bytes = 16;
    num_last = 0;
    int vec_last_cnt = 1;
    for (int32_t i = num_dims - 1; i >= 0; --i) {
        if (pads[i] == 0 && pads[num_dims + i] == 0) {
            vec_last_cnt *= input_shape->GetDim(i);
            ++num_last;
        } else break;
    }
    int vec_last_size = ppl::common::GetSizeOfDataType(input_shape->GetDataType()) * vec_last_cnt;
    return ((num_last != 0) && (vec_last_size >= float4_as_bytes));
}

#define CALL_PAD_FAST_INT8_OPT(LINES_PER_WARP,WARPS_PER_BLOCK) \
{ \
    const int LINES_PER_BLOCK = WARPS_PER_BLOCK * LINES_PER_WARP; \
    dim3 b(64/LINES_PER_WARP, LINES_PER_WARP, WARPS_PER_BLOCK); \
    dim3 g((batch*src_height+LINES_PER_BLOCK-1)/LINES_PER_BLOCK, 1, 1); \
    ppl_cukernel_pad_fast_opt<int8_t, LINES_PER_WARP><<<g, b, 0, stream>>>( \
        num_lines, (const int8_t*)input, src_height, src_width, (int8_t*)output, dst_height, dst_width); \
}


ppl::common::RetCode PPLCUDAPadForwardImp(
    cudaStream_t stream,
    PadKernelParam param,
    ppl::common::TensorShape* input_shape,
    const void* input,
    const int64_t* pads,
    ppl::common::TensorShape* output_shape,
    void* output)
{
    int num_dims       = output_shape->GetDimCount();
    int num_last = 0; // last n dims is not padded
    if (isFastPadSupported(param.pads, num_dims)) {
        int batch = input_shape->CalcElementsToDimensionExcludingPadding(num_dims - 2);
        int dst_height = output_shape->GetDim(num_dims - 2);
        int dst_width  = output_shape->GetDim(num_dims - 1);
        int src_height = input_shape->GetDim(num_dims - 2);
        int src_width  = input_shape->GetDim(num_dims - 1);
        dim3 block_size(16, 16, 1);
        dim3 grid_size(DivUp(dst_width, 16), DivUp(dst_height, 16), batch);
        switch (input_shape->GetDataType()) {
            case ppl::common::DATATYPE_INT8: {
#ifdef __MACACC__
            if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
                int block_size = 256;
                int num_elems = output_shape->CalcElementsIncludingPadding() >> 4;
                int grid_size = (num_elems + block_size - 1) / block_size;
                int channels = (input_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1)) >> 4;
                int height_width_channels = src_height * src_width * channels;
                int width_channels = src_width * channels;
                DivModFast channels_mod(channels);
                DivModFast dst_width_mod(dst_width);
                DivModFast dst_height_mod(dst_height);
                ppl_cukernel_pad_nhwc16_opt<<<grid_size, block_size, 0, stream>>>(
                                            (const float4*)input,
                                            src_height,
                                            src_width,
                                            (float4*)output,
                                            dst_height,
                                            dst_width,
                                            num_elems,
                                            height_width_channels,
                                            width_channels,
                                            channels,
                                            channels_mod,
                                            dst_width_mod,
                                            dst_height_mod);

            } else {
                if (batch*dst_height % 4 != 0 || src_height != dst_height || dst_width <= 64) {
                    ppl_cukernel_pad_fast<<<grid_size, block_size, 0, stream>>>(
                    (const int8_t*)input, src_height, src_width, (int8_t*)output, dst_height, dst_width);
                } else {
                    int w_tile_size = (dst_width + 128)/128;
                    int num_lines = src_height * batch;
                    bool kernel_launched = false;
                    //TODO: There is a limitation that we will only use 16K shared memory
                    //So we are restricted to solve larger dst_width pad problems
                    //If you need to improve pad efficiency, increase shared memory usage is preferred
                    if (num_lines % 8 == 0) {
                    //We are using at most 8K shared memory for 8 lines
                        int max_warps_per_block = 8*1024 / w_tile_size / 128 / 8;
                        if (max_warps_per_block > 1) { //max_warps_per_block == 1 will lead to poor performance as block is too small
                            CALL_PAD_FAST_INT8_OPT(8, max_warps_per_block);
                            kernel_launched = true;
                        }
                    }

                    if (!kernel_launched && num_lines % 4 == 0) {
                        int max_warps_per_block = 8*1024 / w_tile_size / 128 / 4;
                        if (max_warps_per_block > 0) {
                            CALL_PAD_FAST_INT8_OPT(4, max_warps_per_block);
                            kernel_launched = true;
                        }
                    }

                    if (!kernel_launched) {
                        ppl_cukernel_pad_fast<<<grid_size, block_size>>>(
                                    (const int8_t*)input, src_height, src_width, (int8_t*)output, dst_height, dst_width);
                    }
                }
            }
#else
                ppl_cukernel_pad_fast<<<grid_size, block_size, 0, stream>>>(
                    (const int8_t*)input, src_height, src_width, (int8_t*)output, dst_height, dst_width);
#endif
                return ppl::common::RC_SUCCESS;
            }
            case ppl::common::DATATYPE_INT32: {
                if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 || output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
                    output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
                    return ppl::common::RC_UNSUPPORTED;
                }
                ppl_cukernel_pad_fast<<<grid_size, block_size, 0, stream>>>(
                    (const int32_t*)input, src_height, src_width, (int32_t*)output, dst_height, dst_width);
                return ppl::common::RC_SUCCESS;
            }
            case ppl::common::DATATYPE_FLOAT16: {
#ifdef __MACACC__
                if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
                    int block_size = 512;
                    int num_elems = output_shape->CalcElementsIncludingPadding() >> 3;
                    int grid_size = (num_elems + block_size - 1) / block_size;
                    int channels = (input_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1)) >> 3;
                    int height_width_channels = src_height * src_width * channels;
                    int width_channels = src_width * channels;
                    DivModFast channels_mod(channels);
                    DivModFast dst_width_mod(dst_width);
                    DivModFast dst_height_mod(dst_height);
                    ppl_cukernel_pad_nhwc16_opt<<<grid_size, block_size, 0, stream>>>(
                                                (const float4*)input,
                                                src_height,
                                                src_width,
                                                (float4*)output,
                                                dst_height,
                                                dst_width,
                                                num_elems,
                                                height_width_channels,
                                                width_channels,
                                                channels,
                                                channels_mod,
                                                dst_width_mod,
                                                dst_height_mod);
                    return ppl::common::RC_SUCCESS;
                }
#endif//__MACACC__
                ppl_cukernel_pad_fast<<<grid_size, block_size, 0, stream>>>(
                    (const half*)input, src_height, src_width, (half*)output, dst_height, dst_width);
                return ppl::common::RC_SUCCESS;
            }
            case ppl::common::DATATYPE_FLOAT32: {
                if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 || output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
                    output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
                    return ppl::common::RC_UNSUPPORTED;
                }
                ppl_cukernel_pad_fast<<<grid_size, block_size, 0, stream>>>(
                    (const float*)input, src_height, src_width, (float*)output, dst_height, dst_width);
                return ppl::common::RC_SUCCESS;
            }
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
    } else if (isFastPadSupported2(param.pads, num_dims)) {
        int batch = input_shape->CalcElementsToDimensionExcludingPadding(num_dims - 2);
        int dst_height = output_shape->GetDim(num_dims - 2);
        int dst_width  = output_shape->GetDim(num_dims - 1);
        int src_height = input_shape->GetDim(num_dims - 2);
        int src_width  = input_shape->GetDim(num_dims - 1);
        dim3 block_size(16, 16, 1);
        dim3 grid_size(DivUp(dst_width, 16), DivUp(dst_height, 16), batch);
        dim3 GridSize(grid_size.x, grid_size.y, (batch + 7) / 8);
#define PAD_EXEC_FAST2(TYPE, MODE) \
    ppl_cukernel_pad_fast2<TYPE, MODE><<<grid_size, block_size, 0, stream>>>( \
                    (const TYPE*)input, src_height, src_width, (TYPE*)output, dst_height, dst_width, \
                    num_dims, pads, param); \
    break;
#ifdef __MACACC__
#define PAD_EXEC_FAST2_MACA(MODE) \
    ppl_cukernel_pad_fast2_int8_opt<MODE><<<batch/images_per_block,512,0,stream>>>( \
                    (const int8_t*)input, src_height, src_width, (int8_t*)output, dst_height, dst_width, \
                    num_dims, pads, param, images_per_block, DivModFast(dst_height*dst_width), DivModFast(dst_width)); \
    break;
#define PAD_EXEC_FAST_OPT2(TYPE, MODE) \
    ppl_cukernel_pad_fast_opt2<TYPE, MODE, 8><<<GridSize, block_size, 0, stream>>>( \
                    (const TYPE*)input, src_height, src_width, (TYPE*)output, dst_height, dst_width, \
                    num_dims, pads, param, batch); \
    break;
#endif
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 || output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
            output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
            int element_size = ppl::common::GetSizeOfDataType(input_shape->GetDataType());
            int N = 16 / element_size;
            int channels = input_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
            int src_height = input_shape->GetDim(num_dims - 2);
            int src_width  = input_shape->GetDim(num_dims - 1);
            int batch = input_shape->GetDim(0);
            int dst_height = output_shape->GetDim(num_dims - 2);
            int dst_width = output_shape->GetDim(num_dims - 1);
            int num_elems = dst_height * dst_width * channels;
            if((channels & (N - 1))==0) {
                int blockSize = 512;                
                dim3 gridSize = dim3((num_elems + blockSize * N - 1) / (blockSize * N),batch,1);
                switch(element_size) {
                    case 1:
                    switch(param.mode) {
                        case PadKernelParam::PAD_MODE_CONSTANT:
                        ppl_cukernel_pad_nhwc_opt<int8_t, float4, PadKernelParam::PAD_MODE_CONSTANT, 16><<<gridSize, blockSize,0,stream>>>(
                            (const int8_t *)input, src_height,src_width,(int8_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_REFLECT:
                        ppl_cukernel_pad_nhwc_opt<int8_t, float4, PadKernelParam::PAD_MODE_REFLECT, 16><<<gridSize, blockSize,0,stream>>>(
                            (const int8_t *)input, src_height,src_width,(int8_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_EDGE:
                        ppl_cukernel_pad_nhwc_opt<int8_t, float4, PadKernelParam::PAD_MODE_EDGE, 16><<<gridSize, blockSize,0,stream>>>(
                            (const int8_t *)input, src_height,src_width,(int8_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                    }
                    break;
                    case 2:
                    switch(param.mode) {
                        case PadKernelParam::PAD_MODE_CONSTANT:
                        ppl_cukernel_pad_nhwc_opt<half, float4, PadKernelParam::PAD_MODE_CONSTANT, 8><<<gridSize, blockSize,0,stream>>>(
                            (const half *)input, src_height,src_width,(half*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_REFLECT:
                        ppl_cukernel_pad_nhwc_opt<half, float4, PadKernelParam::PAD_MODE_REFLECT, 8><<<gridSize, blockSize,0,stream>>>(
                            (const half *)input, src_height,src_width,(half*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_EDGE:
                        ppl_cukernel_pad_nhwc_opt<half, float4, PadKernelParam::PAD_MODE_EDGE, 8><<<gridSize, blockSize,0,stream>>>(
                            (const half *)input, src_height,src_width,(half*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                    }
                    break;
                    case 4:
                    switch(param.mode) {
                        case PadKernelParam::PAD_MODE_CONSTANT:
                        ppl_cukernel_pad_nhwc_opt<int32_t, float4, PadKernelParam::PAD_MODE_CONSTANT, 4><<<gridSize, blockSize,0,stream>>>(
                            (const int32_t *)input, src_height,src_width,(int32_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_REFLECT:
                        ppl_cukernel_pad_nhwc_opt<int32_t, float4, PadKernelParam::PAD_MODE_REFLECT, 4><<<gridSize, blockSize,0,stream>>>(
                            (const int32_t *)input, src_height,src_width,(int32_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_EDGE:
                        ppl_cukernel_pad_nhwc_opt<int32_t, float4, PadKernelParam::PAD_MODE_EDGE, 4><<<gridSize, blockSize,0,stream>>>(
                            (const int32_t *)input, src_height,src_width,(int32_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                    }
                    break;
                    case 8:
                    switch(param.mode) {
                        case PadKernelParam::PAD_MODE_CONSTANT:
                        ppl_cukernel_pad_nhwc_opt<int64_t, float4, PadKernelParam::PAD_MODE_CONSTANT, 2><<<gridSize, blockSize,0,stream>>>(
                            (const int64_t *)input, src_height,src_width,(int64_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_REFLECT:
                        ppl_cukernel_pad_nhwc_opt<int64_t, float4, PadKernelParam::PAD_MODE_REFLECT, 2><<<gridSize, blockSize,0,stream>>>(
                            (const int64_t *)input, src_height,src_width,(int64_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_EDGE:
                        ppl_cukernel_pad_nhwc_opt<int64_t, float4, PadKernelParam::PAD_MODE_EDGE, 2><<<gridSize, blockSize,0,stream>>>(
                            (const int64_t *)input, src_height,src_width,(int64_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                    }
                    break;
                    default:
                    return ppl::common::RC_UNSUPPORTED;   
                }
            } else {
                int blockSize = 512;                
                dim3 gridSize = dim3((num_elems + blockSize - 1) / (blockSize),batch,1);
                switch(element_size) {
                    case 1:
                    switch(param.mode) {
                        case PadKernelParam::PAD_MODE_CONSTANT:
                        ppl_cukernel_pad_nhwc_opt<int8_t, int8_t, PadKernelParam::PAD_MODE_CONSTANT, 1><<<gridSize, blockSize,0,stream>>>(
                            (const int8_t *)input, src_height,src_width,(int8_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_REFLECT:
                        ppl_cukernel_pad_nhwc_opt<int8_t, int8_t, PadKernelParam::PAD_MODE_REFLECT, 1><<<gridSize, blockSize,0,stream>>>(
                            (const int8_t *)input, src_height,src_width,(int8_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_EDGE:
                        ppl_cukernel_pad_nhwc_opt<int8_t, int8_t, PadKernelParam::PAD_MODE_EDGE, 1><<<gridSize, blockSize,0,stream>>>(
                            (const int8_t *)input, src_height,src_width,(int8_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                    }
                    break;
                    case 2:
                    switch(param.mode) {
                        case PadKernelParam::PAD_MODE_CONSTANT:
                        ppl_cukernel_pad_nhwc_opt<half, half, PadKernelParam::PAD_MODE_CONSTANT, 1><<<gridSize, blockSize,0,stream>>>(
                            (const half *)input, src_height,src_width,(half*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_REFLECT:
                        ppl_cukernel_pad_nhwc_opt<half, half, PadKernelParam::PAD_MODE_REFLECT, 1><<<gridSize, blockSize,0,stream>>>(
                            (const half *)input, src_height,src_width,(half*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_EDGE:
                        ppl_cukernel_pad_nhwc_opt<half, half, PadKernelParam::PAD_MODE_EDGE, 1><<<gridSize, blockSize,0,stream>>>(
                            (const half *)input, src_height,src_width,(half*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                    }
                    break;
                    case 4:
                    switch(param.mode) {
                        case PadKernelParam::PAD_MODE_CONSTANT:
                        ppl_cukernel_pad_nhwc_opt<int32_t, int32_t, PadKernelParam::PAD_MODE_CONSTANT, 1><<<gridSize, blockSize,0,stream>>>(
                            (const int32_t *)input, src_height,src_width,(int32_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_REFLECT:
                        ppl_cukernel_pad_nhwc_opt<int32_t, int32_t, PadKernelParam::PAD_MODE_REFLECT, 1><<<gridSize, blockSize,0,stream>>>(
                            (const int32_t *)input, src_height,src_width,(int32_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_EDGE:
                        ppl_cukernel_pad_nhwc_opt<int32_t, int32_t, PadKernelParam::PAD_MODE_EDGE, 1><<<gridSize, blockSize,0,stream>>>(
                            (const int32_t *)input, src_height,src_width,(int32_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                    }
                    break;
                    case 8:
                    switch(param.mode) {
                        case PadKernelParam::PAD_MODE_CONSTANT:
                        ppl_cukernel_pad_nhwc_opt<int64_t, int64_t, PadKernelParam::PAD_MODE_CONSTANT, 1><<<gridSize, blockSize,0,stream>>>(
                            (const int64_t *)input, src_height,src_width,(int64_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_REFLECT:
                        ppl_cukernel_pad_nhwc_opt<int64_t, int64_t, PadKernelParam::PAD_MODE_REFLECT, 1><<<gridSize, blockSize,0,stream>>>(
                            (const int64_t *)input, src_height,src_width,(int64_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                        case PadKernelParam::PAD_MODE_EDGE:
                        ppl_cukernel_pad_nhwc_opt<int64_t, int64_t, PadKernelParam::PAD_MODE_EDGE, 1><<<gridSize, blockSize,0,stream>>>(
                            (const int64_t *)input, src_height,src_width,(int64_t*)output,dst_height,dst_width,num_dims,pads,param,channels,
                            DivModFast(channels),DivModFast(dst_width),blockSize,num_elems
                        );
                        break;
                    }
                    break;
                    default:
                    return ppl::common::RC_UNSUPPORTED;   
                }
            }
            return ppl::common::RC_SUCCESS;
        }
        switch (input_shape->GetDataType()) {
            case ppl::common::DATATYPE_INT8: {
#ifdef __MACACC__
                int images_per_block = 0;
                //If image is small enough, we prefered to process more images per block
                if (dst_width*dst_height<=16*16 && batch % 16 == 0) images_per_block = 16;
                else if (dst_width*dst_height<=32*32 && batch % 8 == 0) images_per_block = 8;
                else if (dst_width*dst_height<64*64 && batch % 4 == 0) images_per_block = 4;
#endif
                switch(param.mode) {
                    case PadKernelParam::PAD_MODE_CONSTANT:
#ifdef __MACACC__
                        if (images_per_block > 0) {
                            PAD_EXEC_FAST2_MACA(PadKernelParam::PAD_MODE_CONSTANT)
                        } else {
                            PAD_EXEC_FAST2(int8_t, PadKernelParam::PAD_MODE_CONSTANT)
                        }
#else
                        PAD_EXEC_FAST2(int8_t, PadKernelParam::PAD_MODE_CONSTANT)
#endif
                    case PadKernelParam::PAD_MODE_REFLECT:
#ifdef __MACACC__
                        if (images_per_block > 0) {
                            PAD_EXEC_FAST2_MACA(PadKernelParam::PAD_MODE_REFLECT)
                        } else {
                            PAD_EXEC_FAST2(int8_t, PadKernelParam::PAD_MODE_REFLECT)
                        }
#else
                        PAD_EXEC_FAST2(int8_t, PadKernelParam::PAD_MODE_REFLECT)
#endif
                    case PadKernelParam::PAD_MODE_EDGE:
#ifdef __MACACC__
                        if (images_per_block > 0) {
                            PAD_EXEC_FAST2_MACA(PadKernelParam::PAD_MODE_EDGE)
                        } else {
                            PAD_EXEC_FAST2(int8_t, PadKernelParam::PAD_MODE_EDGE)
                        }
#else
                        PAD_EXEC_FAST2(int8_t, PadKernelParam::PAD_MODE_EDGE)
#endif
                }
                return ppl::common::RC_SUCCESS;
            }
            case ppl::common::DATATYPE_INT32: {
                switch(param.mode) {
                    case PadKernelParam::PAD_MODE_CONSTANT:
                        PAD_EXEC_FAST2(int32_t, PadKernelParam::PAD_MODE_CONSTANT)
                    case PadKernelParam::PAD_MODE_REFLECT:
                        PAD_EXEC_FAST2(int32_t, PadKernelParam::PAD_MODE_REFLECT)
                    case PadKernelParam::PAD_MODE_EDGE:
                        PAD_EXEC_FAST2(int32_t, PadKernelParam::PAD_MODE_EDGE)
                }
                return ppl::common::RC_SUCCESS;
            }
            case ppl::common::DATATYPE_FLOAT16: {
                switch(param.mode) {
                    case PadKernelParam::PAD_MODE_CONSTANT:
#ifdef OPT_PAD
                        PAD_EXEC_FAST_OPT2(half, PadKernelParam::PAD_MODE_CONSTANT)
#else//!OPT_PAD
                        PAD_EXEC_FAST2(half, PadKernelParam::PAD_MODE_CONSTANT)
#endif//OPT_PAD
                    case PadKernelParam::PAD_MODE_REFLECT:
#ifdef OPT_PAD
                        PAD_EXEC_FAST_OPT2(half, PadKernelParam::PAD_MODE_REFLECT)
#else//!OPT_PAD
                        PAD_EXEC_FAST2(half, PadKernelParam::PAD_MODE_REFLECT)
#endif//OPT_PAD
                    case PadKernelParam::PAD_MODE_EDGE:
#ifdef OPT_PAD
                        PAD_EXEC_FAST_OPT2(half, PadKernelParam::PAD_MODE_EDGE)
#else//!OPT_PAD
                        PAD_EXEC_FAST2(half, PadKernelParam::PAD_MODE_EDGE)
#endif//OPT_PAD
                }
                return ppl::common::RC_SUCCESS;
            }
            case ppl::common::DATATYPE_FLOAT32: {
                switch(param.mode) {
                    case PadKernelParam::PAD_MODE_CONSTANT:
                        PAD_EXEC_FAST2(float, PadKernelParam::PAD_MODE_CONSTANT)
                    case PadKernelParam::PAD_MODE_REFLECT:
                        PAD_EXEC_FAST2(float, PadKernelParam::PAD_MODE_REFLECT)
                    case PadKernelParam::PAD_MODE_EDGE:
                        PAD_EXEC_FAST2(float, PadKernelParam::PAD_MODE_EDGE)
                }
                return ppl::common::RC_SUCCESS;
            }
            default:
                return ppl::common::RC_UNSUPPORTED;
            }
    } else if (isFastPadSupported3(input_shape, param.pads, num_dims, num_last)) {
        constexpr int float4_as_bytes = 16;
        int vec_factor = float4_as_bytes / ppl::common::GetSizeOfDataType(input_shape->GetDataType());
        int vec_dim_size = 1;
        for (int i = 0; i < num_last; ++i) {
            vec_dim_size *= input_shape->GetDim(num_dims - 1 - i);
        }
        int calc_dims = num_dims - num_last;
        int block_size     = 256;
        uint64_t num_elems = output_shape->CalcElementsToDimensionIncludingPadding(calc_dims);
        int grid_size      = (num_elems + block_size - 1) / block_size;
        vec_dim_size /= vec_factor;

        GArray<int64_t> input_dims(calc_dims);
        GArray<int64_t> input_strides(calc_dims);
        GArray<DivModFast> output_strides_fast(calc_dims);
        int64_t acc_output_stride = 1;
        int64_t acc_input_stride  = 1;
        for (int it = calc_dims - 1; it >= 0; --it) {
            input_dims[it]          = input_shape->GetDim(it);
            input_strides[it]       = acc_input_stride;
            output_strides_fast[it] = DivModFast(acc_output_stride);
            acc_input_stride *= input_shape->GetDim(it);
            acc_output_stride *= output_shape->GetDim(it);
        }

        #define PAD_EXEC_FAST3(TYPE, MODE) \
    ppl_cukernel_pad_fast3<TYPE, MODE><<<grid_size, block_size, 0, stream>>>( \
                num_elems, calc_dims, vec_factor, vec_dim_size, param, input_dims, input_strides, (const float4*)input, pads, output_strides_fast, (float4*)output); \
    break;
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 || output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
            output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
            return ppl::common::RC_UNSUPPORTED;
        }
        switch (input_shape->GetDataType()) {
            case ppl::common::DATATYPE_INT8: {
                switch(param.mode) {
                    case PadKernelParam::PAD_MODE_CONSTANT:
                        PAD_EXEC_FAST3(int8_t, PadKernelParam::PAD_MODE_CONSTANT)
                    case PadKernelParam::PAD_MODE_REFLECT:
                        PAD_EXEC_FAST3(int8_t, PadKernelParam::PAD_MODE_REFLECT)
                    case PadKernelParam::PAD_MODE_EDGE:
                        PAD_EXEC_FAST3(int8_t, PadKernelParam::PAD_MODE_EDGE)
                }
                return ppl::common::RC_SUCCESS;
            }
            case ppl::common::DATATYPE_INT32: {
                switch(param.mode) {
                    case PadKernelParam::PAD_MODE_CONSTANT:
                        PAD_EXEC_FAST3(int32_t, PadKernelParam::PAD_MODE_CONSTANT)
                    case PadKernelParam::PAD_MODE_REFLECT:
                        PAD_EXEC_FAST3(int32_t, PadKernelParam::PAD_MODE_REFLECT)
                    case PadKernelParam::PAD_MODE_EDGE:
                        PAD_EXEC_FAST3(int32_t, PadKernelParam::PAD_MODE_EDGE)
                }
                return ppl::common::RC_SUCCESS;
            }
            case ppl::common::DATATYPE_FLOAT16: {
                switch(param.mode) {
                    case PadKernelParam::PAD_MODE_CONSTANT:
                        PAD_EXEC_FAST3(half, PadKernelParam::PAD_MODE_CONSTANT)
                    case PadKernelParam::PAD_MODE_REFLECT:
                        PAD_EXEC_FAST3(half, PadKernelParam::PAD_MODE_REFLECT)
                    case PadKernelParam::PAD_MODE_EDGE:
                        PAD_EXEC_FAST3(half, PadKernelParam::PAD_MODE_EDGE)
                }
                return ppl::common::RC_SUCCESS;
            }
            case ppl::common::DATATYPE_FLOAT32: {
                switch(param.mode) {
                    case PadKernelParam::PAD_MODE_CONSTANT:
                        PAD_EXEC_FAST3(float, PadKernelParam::PAD_MODE_CONSTANT)
                    case PadKernelParam::PAD_MODE_REFLECT:
                        PAD_EXEC_FAST3(float, PadKernelParam::PAD_MODE_REFLECT)
                    case PadKernelParam::PAD_MODE_EDGE:
                        PAD_EXEC_FAST3(float, PadKernelParam::PAD_MODE_EDGE)
                }
                return ppl::common::RC_SUCCESS;
            }
            default:
                return ppl::common::RC_UNSUPPORTED;
            }
    }
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 || output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
            output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
        return ppl::common::RC_UNSUPPORTED;
    }
    int block_size     = 256;
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int grid_size      = (num_elems + block_size - 1) / block_size;
    GArray<int64_t> input_dims(num_dims);
    GArray<int64_t> input_strides(num_dims);
    GArray<DivModFast> output_strides_fast(num_dims);
    int64_t acc_output_stride = 1;
    int64_t acc_input_stride  = 1;
    for (int it = num_dims - 1; it >= 0; --it) {
        input_dims[it]          = input_shape->GetDim(it);
        input_strides[it]       = acc_input_stride;
        output_strides_fast[it] = DivModFast(acc_output_stride);
        acc_input_stride *= input_shape->GetDim(it);
        acc_output_stride *= output_shape->GetDim(it);
    }

#define PAD_EXEC(TYPE, MODE) \
    ppl_cukernel_pad<TYPE, MODE><<<grid_size, block_size, 0, stream>>>( \
                num_elems, num_dims, param, input_dims, input_strides, (const TYPE*)input, pads, output_strides_fast, (TYPE*)output); \
    break;

    switch (input_shape->GetDataType()) {
        case ppl::common::DATATYPE_INT8: {
            switch(param.mode) {
                case PadKernelParam::PAD_MODE_CONSTANT:
                    PAD_EXEC(int8_t, PadKernelParam::PAD_MODE_CONSTANT)
                case PadKernelParam::PAD_MODE_REFLECT:
                    PAD_EXEC(int8_t, PadKernelParam::PAD_MODE_REFLECT)
                case PadKernelParam::PAD_MODE_EDGE:
                    PAD_EXEC(int8_t, PadKernelParam::PAD_MODE_EDGE)
            }
            return ppl::common::RC_SUCCESS;
        }
        case ppl::common::DATATYPE_INT32: {
            switch(param.mode) {
                case PadKernelParam::PAD_MODE_CONSTANT:
                    PAD_EXEC(int32_t, PadKernelParam::PAD_MODE_CONSTANT)
                case PadKernelParam::PAD_MODE_REFLECT:
                    PAD_EXEC(int32_t, PadKernelParam::PAD_MODE_REFLECT)
                case PadKernelParam::PAD_MODE_EDGE:
                    PAD_EXEC(int32_t, PadKernelParam::PAD_MODE_EDGE)
            }
            return ppl::common::RC_SUCCESS;
        }
        case ppl::common::DATATYPE_FLOAT16: {
            switch(param.mode) {
                case PadKernelParam::PAD_MODE_CONSTANT:
                    PAD_EXEC(half, PadKernelParam::PAD_MODE_CONSTANT)
                case PadKernelParam::PAD_MODE_REFLECT:
                    PAD_EXEC(half, PadKernelParam::PAD_MODE_REFLECT)
                case PadKernelParam::PAD_MODE_EDGE:
                    PAD_EXEC(half, PadKernelParam::PAD_MODE_EDGE)
            }
            return ppl::common::RC_SUCCESS;
        }
        case ppl::common::DATATYPE_FLOAT32: {
            switch(param.mode) {
                case PadKernelParam::PAD_MODE_CONSTANT:
                    PAD_EXEC(float, PadKernelParam::PAD_MODE_CONSTANT)
                case PadKernelParam::PAD_MODE_REFLECT:
                    PAD_EXEC(float, PadKernelParam::PAD_MODE_REFLECT)
                case PadKernelParam::PAD_MODE_EDGE:
                    PAD_EXEC(float, PadKernelParam::PAD_MODE_EDGE)
            }
            return ppl::common::RC_SUCCESS;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
}
