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

#include "cudakernel/memory/expand.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_fp16.h>
#define MAX_BLOCK_DIM_YZ (65536)

#ifdef __MACACC__
#define _OPT_EXPAND_
#endif//__MACACC__

template <typename T>
__global__ void ppl_cukernel_expand(
    int64_t num_elems,
    int num_output_dim,
    GArray<DivModFast> output_strides_fast,
    GArray<int64_t> input_strides,
    const T *input,
    T *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int64_t input_offset = 0;
    int idx, remain = index;
    for (int it = 0; it < num_output_dim; ++it) {
        output_strides_fast[it].divmod(remain, idx, remain);
        input_offset += idx * input_strides[it];
    }
    output[index] = input[input_offset];
}

template <int N, typename T1, typename T2>
__global__ void ppl_cukernel_expand_three_broadcast_opt(
    int64_t inner_elems,
    int axis_width,
    const T1 *input,
    T2 *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= inner_elems)
        return;

    int axis_idx         = blockIdx.y;
    int outer_idx        = blockIdx.z;
    float4 val;
    T1* val_ptr          = (T1*)&val;

    int64_t output_index = outer_idx * axis_width * inner_elems + axis_idx * inner_elems + index;
    T1 input_val         = input[axis_idx];

    #pragma unroll N
    for (int i = 0; i < N; i++) {
        val_ptr[i] = input_val;
    }

    output[output_index] = *(T2*)&val;
}

template <typename T>
__global__ void ppl_cukernel_expand_one_broadcast(
    int64_t inner_elems,
    int axis_width,
    const T *input,
    T *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= inner_elems)
        return;

    int axis_idx         = blockIdx.y;
    int outer_idx        = blockIdx.z;
    int64_t input_index  = outer_idx * inner_elems + index;
    int64_t output_index = outer_idx * axis_width * inner_elems + axis_idx * inner_elems + index;
    output[output_index] = input[input_index];
}

template <typename T>
__global__ void ppl_cukernel_expand_last_dim(
    int64_t inner_dim,
    const T *input,
    T *output)
{
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    if (index >= inner_dim)
        return;
    int blk_idx   = blockIdx.x;
    int out_index = blk_idx * inner_dim + index;

    output[out_index] = input[blk_idx];
}

#ifdef _OPT_EXPAND_
//tileSize / inner_dim must be Divisible
//inner_dim / N must be Divisble
//tile_height= 256 / (inner_dim / N)
template<int N,typename T>
__global__ void ppl_cukernel_expand_last_dim_optN(
    int inner_dim,
    int outer_dim,
    int tile_width,
    int tile_height,
    const T* input,
    T* output
)
{
    int tidx = threadIdx.x;
    int tiley = (int)tidx * __builtin_mxc_rcpf(tile_width);
    int heightId = blockIdx.x*tile_height;
    int height = min(outer_dim - heightId,tile_height);
    const T* ptr_input = input + heightId + tiley;
    T* ptr_output = output + heightId* inner_dim;

    if(tiley < height)
    {
        T tmp = ptr_input[0];
        T buffer[N];
        #pragma unroll N
        for(int i = 0; i < N; i++){
            buffer[i] = tmp;
        }
        if constexpr(N==4){
            *((float4 *)ptr_output + tidx) = *(float4*)buffer;
        }else if constexpr(N==2){
            *((float2 *)ptr_output + tidx) = *(float2*)buffer;
        }else{
            ptr_output[tidx] = buffer[0];
        }
    }
}

template<int N>
__global__ void ppl_cukernel_expand_last_dim_optN(
    int inner_dim,
    int outer_dim,
    int tile_width,
    int tile_height,
    const half* input,
    half* output
)
{
    int tidx = threadIdx.x;
    int tiley = (int)tidx * __builtin_mxc_rcpf(tile_width);
    int heightId = blockIdx.x*tile_height;
    int height = min(outer_dim - heightId,tile_height);
    const half* ptr_input = input + heightId + tiley;
    half* ptr_output = output + heightId* inner_dim;

    if(tiley < height)
    {
        half tmp = ptr_input[0];
        half buffer[N];
        #pragma unroll N
        for(int i = 0; i < N; i++)
        {
            buffer[i] = tmp;
        }
        if constexpr(N==8){
            *((float4 *)ptr_output + tidx) = *(float4*)buffer;
        }else if constexpr(N==4){
            *((float2 *)ptr_output + tidx) = *(float2*)buffer;
        }else if constexpr(N==2){
            *((half2 *)ptr_output + tidx) = *(half2*)buffer;
        }else{
            ptr_output[tidx] = buffer[0];
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_expand_last_dim_opt_small(
    int inner_dim,
    int outer_dim,
    int tile_height,
    const T* input,
    T* output
)
{
    int tid = threadIdx.x;
    int heightId = blockIdx.x*tile_height;
    int height = min(outer_dim - heightId,tile_height);
    int blockSize = height*inner_dim;

    const T *ptr_input = input + heightId;
    T *ptr_output = output + heightId*inner_dim;

    for(int i = tid; i < blockSize; i += blockDim.x)
    {
        T tmp = ptr_input[(int)(i *__builtin_mxc_rcpf(inner_dim))];
        ptr_output[i] = tmp;
    }
}

template <typename T, typename VEC_T, int N, int SHIFT>
__global__ void ppl_cukernel_expand_opt_large(
    int out_len,
    int in_len,
    int per_block,
    const T *input,
    T *output,
    DivModFast in_len_fast)
{
    extern __shared__ char sm_buffer[];
    T * sm_input = (T*)sm_buffer;
    int y_offset = blockIdx.x * per_block;
    int len_y = min(out_len - y_offset, per_block);
    if(len_y <= 0 ) return;
    const T * ptr_input = input + y_offset;
    for(int i = threadIdx.x; i < len_y; i += blockDim.x) {
        sm_input[i] = ptr_input[i];
    }
    __syncthreads();

    VEC_T* ptr_output = (VEC_T*)(output + y_offset * in_len);
    int out_block_size = len_y * in_len;
    out_block_size = out_block_size >> SHIFT;
    for(int i = threadIdx.x; i < out_block_size; i += blockDim.x) {
        int offset = in_len_fast.div(i<<SHIFT);
        T reg_input = sm_input[offset];
        T array_input[N];
        
        #pragma unroll N
        for(int j = 0; j < N; j++) {
            array_input[j] = reg_input;
        }
        VEC_T vec_input = *(VEC_T*)array_input;
        ptr_output[i] = vec_input;
    }
}

template <typename T, typename VEC_T, int N, int Shift>
__global__ void ppl_cukernel_expand_opt_small(
    int out_len,
    int in_len,
    const T *input,
    T *output)
{
    __shared__ T sm_input;
    const T* ptr_input = input + blockIdx.y;
    if(threadIdx.x == 0) {
        sm_input = *ptr_input;
    }
    __syncthreads();
    T reg_input = sm_input;
    T reg_array[N];
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        reg_array[i] = reg_input;
    }
    VEC_T vec_input = *(VEC_T*)reg_array;
    int offset = blockIdx.x * 4096;
    int block_size = min(in_len - offset, 4096);
    if(block_size  <= 0) return;
    block_size = block_size >> Shift;
    VEC_T *ptr_output = (VEC_T*)(output + blockIdx.y * in_len + offset);
    for(int i = threadIdx.x; i < block_size; i += blockDim.x) {
        ptr_output[i] = vec_input;
    }
}

#endif//_OPT_EXPAND_
static void ppl_pad_tensor_shape(const ppl::common::TensorShape *tensor_shape0,
                                 const ppl::common::TensorShape *tensor_shape1,
                                 ppl::common::TensorShape *pad_tensor_shape0,
                                 ppl::common::TensorShape *pad_tensor_shape1)
{
    int max_dims = std::max(tensor_shape0->GetDimCount(), tensor_shape1->GetDimCount());
    if (pad_tensor_shape0->GetDimCount() < pad_tensor_shape1->GetDimCount()) {
        pad_tensor_shape0->SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape0->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape0->SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape0->SetDim(i, tensor_shape0->GetDim(i - offset));
        }
    } else {
        pad_tensor_shape1->SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape1->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape1->SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape1->SetDim(i, tensor_shape1->GetDim(i - offset));
        }
    }
}

static int ppl_get_num_broadcast_dims(const ppl::common::TensorShape *tensor_shape0,
                                      const ppl::common::TensorShape *tensor_shape1,
                                      int &aixs)
{
    ppl::common::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::common::TensorShape pad_tensor_shape1 = *tensor_shape1;
    ppl_pad_tensor_shape(tensor_shape0, tensor_shape1, &pad_tensor_shape0, &pad_tensor_shape1);
    int dim_count          = pad_tensor_shape0.GetDimCount();
    int num_broadcast_dims = 0;
    for (int it = 0; it < dim_count; ++it) {
        if (pad_tensor_shape0.GetDim(it) != pad_tensor_shape1.GetDim(it))
            ++num_broadcast_dims;
    }
    if (num_broadcast_dims == 1) {
        for (int it = 0; it < dim_count; ++it) {
            if (pad_tensor_shape0.GetDim(it) != pad_tensor_shape1.GetDim(it))
                aixs = it;
        }
    }
    if (num_broadcast_dims > 1 && num_broadcast_dims == (dim_count - 1)) {
        for (int it = 0; it < dim_count; ++it) {
            if (pad_tensor_shape0.GetDim(it) == pad_tensor_shape1.GetDim(it))
                aixs = it;
        }
    }
    return num_broadcast_dims;
}

ppl::common::RetCode PPLCUDAExpandForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape *input_shape,
    const void *input,
    const ppl::common::TensorShape *output_shape,
    void *output)
{
    int dim_count      = output_shape->GetDimCount();
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    if (num_elems == input_shape->CalcElementsIncludingPadding()) { // no expand, just copy, as reshape
        cudaMemcpyAsync(output, input, ppl::common::GetSizeOfDataType(input_shape->GetDataType()) * num_elems, cudaMemcpyDeviceToDevice, stream);
        return ppl::common::RC_SUCCESS;
    }
    int axis               = 0;
    int num_broadcast_dims = ppl_get_num_broadcast_dims(input_shape, output_shape, axis);
    int inner_elems        = 1;
    int outer_elems        = 1;
    int32_t axis_width     = output_shape->GetDim(axis);
    for (int it = dim_count - 1; it > axis; inner_elems *= output_shape->GetDim(it), --it)
        ;
    for (int it = 0; it < axis; outer_elems *= output_shape->GetDim(it), ++it)
        ;

    // Avoid using memcpy on Cxx
    // if (num_broadcast_dims == 1 && (axis == 0 || axis == dim_count - 1)) {
    if (num_broadcast_dims == 1 && (axis == dim_count - 1)) { // only one broadcast dim, normal case
        if (axis == 0) { // just for-copy
            int iters        = output_shape->GetDim(axis);
            int input_size   = input_shape->CalcElementsIncludingPadding() * ppl::common::GetSizeOfDataType(input_shape->GetDataType());
            char *output_ptr = static_cast<char *>(output);

            for (int it = 0; it < iters; ++it) {
                cudaMemcpyAsync(output_ptr + it * input_size, input, input_size, cudaMemcpyDeviceToDevice, stream);
            }
        } else if (axis == dim_count - 1) {
            int inner_dim = output_shape->GetDim(axis);
            int outer_dim = 1;
            for (int it = 0; it < axis; outer_dim *= output_shape->GetDim(it), ++it)
                ;
#ifdef _OPT_EXPAND_
            if(ppl::common::GetSizeOfDataType(input_shape->GetDataType())==2){
                if((inner_dim & 0x7) == 0 && inner_dim <= 2048){
                    int blockX = inner_dim >> 3;
                    int blockY = 256 / blockX;
                    int blockSize = 256;
                    int grid_size = (outer_dim + blockY - 1) / blockY;
                    ppl_cukernel_expand_last_dim_optN<8><<<grid_size,blockSize,0,stream>>>(inner_dim,outer_dim,blockX,blockY,(const half*)input,(half*)output);
                    return ppl::common::RC_SUCCESS;
                }
                else if((inner_dim & 0x3) == 0 && inner_dim <= 1024){
                    int blockX = inner_dim >> 2;
                    int blockY = 256 / blockX;
                    int blockSize = 256;
                    int grid_size = (outer_dim + blockY - 1) / blockY;
                    ppl_cukernel_expand_last_dim_optN<4><<<grid_size,blockSize,0,stream>>>(inner_dim,outer_dim,blockX,blockY,(const half*)input,(half*)output);
                    return ppl::common::RC_SUCCESS;
                }
                else if((inner_dim & 0x1) == 0 && inner_dim <= 512)
                {
                    int blockX = inner_dim >> 1;
                    int blockY = 256 / blockX;
                    int blockSize = 256;
                    int grid_size = (outer_dim + blockY - 1) / blockY;
                    ppl_cukernel_expand_last_dim_optN<2><<<grid_size,blockSize,0,stream>>>(inner_dim,outer_dim,blockX,blockY,(const half*)input,(half*)output);
                    return ppl::common::RC_SUCCESS;
                }
                else if(inner_dim <= 256)
                {
                    int blockX = inner_dim;
                    int blockY = 256 / blockX;
                    int blockSize = 256;
                    int grid_size = (outer_dim + blockY - 1) / blockY;
                    ppl_cukernel_expand_last_dim_optN<1><<<grid_size,blockSize,0,stream>>>(inner_dim,outer_dim,blockX,blockY,(const half*)input,(half*)output);
                    return ppl::common::RC_SUCCESS;
                }
            }
            else if(ppl::common::GetSizeOfDataType(input_shape->GetDataType())==4){
                if((inner_dim & 0x3) == 0 && inner_dim <= 1024){
                    int blockX = inner_dim >> 2;
                    int blockY = 256 / blockX;
                    int blockSize = 256;
                    int grid_size = (outer_dim + blockY - 1) / blockY;
                    ppl_cukernel_expand_last_dim_optN<4,float><<<grid_size,blockSize,0,stream>>>(inner_dim,outer_dim,blockX,blockY,(const float*)input,(float*)output);
                    return ppl::common::RC_SUCCESS;
                }
                else if((inner_dim & 0x1) == 0 && inner_dim <= 512)
                {
                    int blockX = inner_dim >> 1;
                    int blockY = 256 / blockX;
                    int blockSize = 256;
                    int grid_size = (outer_dim + blockY - 1) / blockY;
                    ppl_cukernel_expand_last_dim_optN<2,float><<<grid_size,blockSize,0,stream>>>(inner_dim,outer_dim,blockX,blockY,(const float*)input,(float*)output);
                    return ppl::common::RC_SUCCESS;
                }
                else if(inner_dim <= 256)
                {
                    int blockX = inner_dim;
                    int blockY = 256 / blockX;
                    int blockSize = 256;
                    int grid_size = (outer_dim + blockY - 1) / blockY;
                    ppl_cukernel_expand_last_dim_optN<1,float><<<grid_size,blockSize,0,stream>>>(inner_dim,outer_dim,blockX,blockY,(const float*)input,(float*)output);
                    return ppl::common::RC_SUCCESS;
                }
            }

            if(inner_dim <= 512)
            {
                int block_size = 512;
                int tile_height = block_size / inner_dim;
                int grid_size = (outer_dim + tile_height - 1) / tile_height;
                switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())){
                    case 1:{
                        ppl_cukernel_expand_last_dim_opt_small<int8_t><<<grid_size,block_size,0,stream>>>(
                            inner_dim, outer_dim, tile_height,(const int8_t*)input,(int8_t*)output
                        );
                        return ppl::common::RC_SUCCESS;
                    }
                    case 2:{
                        ppl_cukernel_expand_last_dim_opt_small<int16_t><<<grid_size,block_size,0,stream>>>(
                            inner_dim, outer_dim, tile_height,(const int16_t*)input,(int16_t*)output
                        );
                        return ppl::common::RC_SUCCESS;
                    }
                    case 4:{
                        ppl_cukernel_expand_last_dim_opt_small<int32_t><<<grid_size,block_size,0,stream>>>(
                            inner_dim, outer_dim, tile_height,(const int32_t*)input,(int32_t*)output
                        );
                        return ppl::common::RC_SUCCESS;
                    }
                    case 8:{
                        ppl_cukernel_expand_last_dim_opt_small<int64_t><<<grid_size,block_size,0,stream>>>(
                            inner_dim, outer_dim, tile_height,(const int64_t*)input,(int64_t*)output
                        );
                        return ppl::common::RC_SUCCESS;
                    }
                    default:
                        return ppl::common::RC_UNSUPPORTED;
                }
            }
#endif//_OPT_EXPAND_
            int block_size = 256;
            dim3 grid_size(outer_dim, (inner_dim + block_size - 1) / block_size, 1);

#define SWITCH_CASE(TYPE)                                                   \
    case sizeof(TYPE): {                                                    \
        ppl_cukernel_expand_last_dim<<<grid_size, block_size, 0, stream>>>( \
            inner_dim, (const TYPE *)input, (TYPE *)output);                \
        return ppl::common::RC_SUCCESS;                                     \
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
        return ppl::common::RC_SUCCESS;
    } else if (num_broadcast_dims == 1 && (axis_width < MAX_BLOCK_DIM_YZ) && (outer_elems < MAX_BLOCK_DIM_YZ)) {
        if (ppl::common::GetSizeOfDataType(input_shape->GetDataType()) == 4 && !(inner_elems & 0xf)) {
            int block_size = 256;
            dim3 grid_size(((inner_elems >> 2) + block_size - 1) / block_size, axis_width, outer_elems);
            ppl_cukernel_expand_one_broadcast<<<grid_size, block_size, 0, stream>>>(inner_elems >> 2,
                                                                                    axis_width,
                                                                                    (const float4 *)input,
                                                                                    (float4 *)output);
        } else {
            int block_size = 256;
            dim3 grid_size((inner_elems + block_size - 1) / block_size, axis_width, outer_elems);

#define SWITCH_CASE(TYPE)                                                                            \
    case sizeof(TYPE): {                                                                             \
        ppl_cukernel_expand_one_broadcast<<<grid_size, block_size, 0, stream>>>(inner_elems,         \
                                                                                axis_width,          \
                                                                                (const TYPE *)input, \
                                                                                (TYPE *)output);     \
        return ppl::common::RC_SUCCESS;                                                              \
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
        return ppl::common::RC_SUCCESS;
#ifdef _OPT_EXPAND_
    } else if (num_broadcast_dims == 3 && (axis_width < MAX_BLOCK_DIM_YZ) && (outer_elems < MAX_BLOCK_DIM_YZ)) {
        int block_size = 256;
        dim3 grid_size((inner_elems + block_size - 1) / block_size, axis_width, outer_elems);

#define EXPAND_THREE_BROADCAST_OPT(TYPE, TYPE2, N)                                                               \
        grid_size.x = (grid_size.x + N - 1) / N;                                                                 \
        ppl_cukernel_expand_three_broadcast_opt<N><<<grid_size, block_size, 0, stream>>>(inner_elems / N,       \
                                                                                axis_width,                      \
                                                                                (const TYPE *)input,             \
                                                                                (TYPE2 *)output);                \
        return ppl::common::RC_SUCCESS;

        if (ppl::common::GetSizeOfDataType(input_shape->GetDataType()) == 1) {
            if (inner_elems % 16 == 0) {
                EXPAND_THREE_BROADCAST_OPT(int8_t, float4, 16);
            } else if (inner_elems % 8 == 0) {
                EXPAND_THREE_BROADCAST_OPT(int8_t, int64_t, 8);
            } else if (inner_elems % 4 == 0) {
                EXPAND_THREE_BROADCAST_OPT(int8_t, int32_t, 4);
            } else if (inner_elems % 2 == 0) {
                EXPAND_THREE_BROADCAST_OPT(int8_t, int16_t, 2);
            } else {
                EXPAND_THREE_BROADCAST_OPT(int8_t, int8_t, 1);
            }
        } else if (ppl::common::GetSizeOfDataType(input_shape->GetDataType()) == 2) {
            if (inner_elems % 8 == 0) {
                EXPAND_THREE_BROADCAST_OPT(int16_t, float4, 8);
            } else if (inner_elems % 4 == 0) {
                EXPAND_THREE_BROADCAST_OPT(int16_t, int64_t, 4);
            } else if (inner_elems % 2 == 0) {
                EXPAND_THREE_BROADCAST_OPT(int16_t, int32_t, 2);
            } else {
                EXPAND_THREE_BROADCAST_OPT(int16_t, int16_t, 1);
            }
        } else if (ppl::common::GetSizeOfDataType(input_shape->GetDataType()) == 4) {
            if (inner_elems % 4 == 0) {
                EXPAND_THREE_BROADCAST_OPT(int32_t, float4, 4);
            } else if (inner_elems % 2 == 0) {
                EXPAND_THREE_BROADCAST_OPT(int32_t, int64_t, 2);
            } else {
                EXPAND_THREE_BROADCAST_OPT(int32_t, int32_t, 1);
            }
        } else if (ppl::common::GetSizeOfDataType(input_shape->GetDataType()) == 8) {
            if (inner_elems % 2 == 0) {
                EXPAND_THREE_BROADCAST_OPT(int64_t, float4, 2);
            } else {
                EXPAND_THREE_BROADCAST_OPT(int64_t, int64_t, 1);
            }
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
#undef EXPAND_THREE_BROADCAST_OPT
#endif//_OPT_EXPAND_
    } else {
        uint32_t num_output_dim = output_shape->GetDimCount();
        GArray<DivModFast> output_strides_fast(num_output_dim);
        GArray<int64_t> input_strides(num_output_dim);
        ppl::common::TensorShape pad_input_shape = *input_shape;
        if (pad_input_shape.GetDimCount() < num_output_dim) {
            pad_input_shape.SetDimCount(num_output_dim);
            // pad 1 to shape_min_pad's higher dim
            uint32_t offset = num_output_dim - input_shape->GetDimCount();
            for (uint32_t i = 0; i < offset; i++) {
                pad_input_shape.SetDim(i, 1);
            }
            for (uint32_t i = offset; i < num_output_dim; i++) {
                pad_input_shape.SetDim(i, input_shape->GetDim(i - offset));
            }
        }

        int64_t acc_output_stride = 1;
        int64_t acc_input_stride  = 1;
        for (int it = num_output_dim - 1; it >= 0; --it) {
            if (pad_input_shape.GetDim(it) == 1) {
                input_strides[it] = 0;
            } else {
                input_strides[it] = acc_input_stride;
            }
            output_strides_fast[it] = DivModFast(acc_output_stride);
            acc_input_stride *= pad_input_shape.GetDim(it);
            acc_output_stride *= output_shape->GetDim(it);
        }
#ifdef _OPT_EXPAND_
        {
            int inner_dim = 1, out_dim = 1,it, check_dim = 1;
            for(it = num_output_dim - 1; it >= 0; --it) {
                if(pad_input_shape.GetDim(it) == 1) {
                    inner_dim *= output_shape->GetDim(it);
                } else {
                    break;
                }
            }

            for(; it >= 0; --it) {
                check_dim *= pad_input_shape.GetDim(it);
                out_dim *= output_shape->GetDim(it);
            }

            if(check_dim == out_dim) {
#define OPT_EXPAND(T,T1,N,SHIFT)                                                                                                                                                            \
                if(inner_dim < 4096)                                                                                                                                                        \
                {                                                                                                                                                                           \
                    int per_block = 4096 / inner_dim;                                                                                                                                       \
                    dim3 gridSize((out_dim + per_block - 1) / per_block, 1, 1);                                                                                                             \
                    int blockSize = 512;                                                                                                                                                    \
                    ppl_cukernel_expand_opt_large<T,T1,N,SHIFT><<<gridSize, blockSize, per_block*sizeof(T),stream>>>(out_dim,inner_dim, per_block, (const T*)input,(T*)output,DivModFast(inner_dim)); \
                } else {                                                                                                                                                                    \
                    dim3 gridSize((inner_dim + 4095) / 4096, out_dim, 1);                                                                                                                   \
                    int blockSize = 512;                                                                                                                                                    \
                    ppl_cukernel_expand_opt_small<T,T1,N,SHIFT><<<gridSize, blockSize, 0,stream>>>(out_dim,inner_dim, (const T*)input,(T*)output);                                          \
                }                                                                                                                                                                           \
                return ppl::common::RC_SUCCESS;
                
                switch(ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
                    case 1:
                        if((inner_dim &7) == 0) {
                            OPT_EXPAND(int8_t, float2, 8, 3)
                        } 
                        else if((inner_dim & 3) == 0) {
                            OPT_EXPAND(int8_t, int32_t, 4, 2)
                        }
                        else if((inner_dim & 1) == 0) {
                            OPT_EXPAND(int8_t, int16_t, 2, 1)
                        } else {
                            OPT_EXPAND(int8_t, int8_t, 1, 0)
                        }
                    case 2:
                        if((inner_dim &7) == 0) {
                            OPT_EXPAND(int16_t, float4, 8, 3)
                        } 
                        else if((inner_dim & 3) == 0) {
                            OPT_EXPAND(int16_t, float2, 4, 2)
                        }
                        else if((inner_dim & 1) == 0) {
                            OPT_EXPAND(int16_t, int32_t, 2, 1)
                        } else {
                            OPT_EXPAND(int16_t, int16_t, 1, 0)
                        }
                    case 4:
                        if((inner_dim &3) == 0) {
                            OPT_EXPAND(int32_t, float4, 4, 2)
                        } 
                        else if((inner_dim & 1) == 0) {
                            OPT_EXPAND(int32_t, float2, 2, 1)
                        }
                        else {
                            OPT_EXPAND(int32_t, int32_t, 1, 0)
                        }
                    case 8:
                        if((inner_dim &1) == 0) {
                            OPT_EXPAND(int64_t, float4, 2, 1)
                        }
                        else {
                            OPT_EXPAND(int64_t, int64_t, 1, 0)
                        }
                    default:
                        return ppl::common::RC_UNSUPPORTED;
                }
#undef OPT_EXPAND
            }
        }
#endif//_OPT_EXPAND_
        int block_size = 256;
        int grid_size  = (num_elems + block_size - 1) / block_size;

#define SWITCH_CASE(TYPE)                                                                                        \
    case sizeof(TYPE): {                                                                                         \
        ppl_cukernel_expand<<<grid_size, block_size, 0, stream>>>(                                               \
            num_elems, num_output_dim, output_strides_fast, input_strides, (const TYPE *)input, (TYPE *)output); \
        return ppl::common::RC_SUCCESS;                                                                          \
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
