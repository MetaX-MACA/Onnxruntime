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

#include "cudakernel/memory/gather_nd.h"
#include "cudakernel/common/divmod_fast.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/common/types.h"
#include <cuda_runtime.h>
#include <assert.h>
#include <vector>
#ifdef __MACACC__
#define _OPT_GATHER_
#endif//__MACACC__

#ifdef _OPT_GATHER_
#define MIN(a,b) ((a) < (b) ? (a):(b))
struct stride_param {
    int64_t input_dims_gpu[16];
    int64_t input_strides[16];
};

template <typename IndexT>
__global__ void ppl_cukernel_gather_nd_offset_opt(
    int64_t num_pieces,
    DivModFast num_pieces_per_batch_fast,
    int batch_dim,
    int num_input_dim,
    stride_param input_dims,
    int input_batch_stride,
    int indices_last_dim_size,
    const IndexT* indices_data,
    int64_t* piece_offsets)
{
    int64_t * input_strides_gpu = input_dims.input_strides;
    int64_t* input_dims_gpu = input_dims.input_dims_gpu;
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_pieces)
        return;
    // batch offset
    int batch_idx             = num_pieces_per_batch_fast.div(index);
    int64_t batch_offset      = batch_idx * input_batch_stride;
    const IndexT* indices_ptr = indices_data + index * indices_last_dim_size;
    int64_t rel_offset        = 0;

    for (int it = 0; it < indices_last_dim_size; ++it) {
        IndexT cor_val = indices_ptr[it];
        int64_t reg = input_dims_gpu[batch_dim + it];
        if (cor_val < 0)
            cor_val += reg;
// #ifdef DEBUG
//         assert(cor_val >= 0 && cor_val < reg);
// #endif
        rel_offset += cor_val * input_strides_gpu[it];
    }
    piece_offsets[index] = batch_offset + rel_offset;
}

template <typename T, typename DST_T, int N, int shift,int SHIFT>
__global__ void ppl_cukernel_gather_nd_opt(
    int64_t num_elems,
    DivModFast piece_size_fast,
    int num_piece,
    int piece_size_per_block,
    int64_t* piece_offsets,
    const T* input,
    T* output)
{
    __shared__ int64_t sm_piece_offsets[2048];
    int index0 = (blockIdx.x * blockDim.x) << SHIFT;
    int piece_idx0 = piece_idx0 = piece_size_fast.div(index0);
    int64_t * piece_block_offset = piece_offsets + piece_idx0;
    int piece_size_block = min(num_piece - piece_idx0, piece_size_per_block);
    for(int i = threadIdx.x; i < piece_size_block; i += blockDim.x) {
        sm_piece_offsets[i] = piece_block_offset[i];
    }
    __syncthreads();
    
    T *ptr_output = output + index0;
    if(blockIdx.x == gridDim.x - 1) {
        int block_size = min(num_elems - index0, (blockDim.x<<SHIFT));
        for(int i = threadIdx.x; i < block_size; i += blockDim.x) {
            T reg_dst;
            T* ptr_local_output = ptr_output + i;
            int piece_idx, offset;
            piece_size_fast.divmod(index0 + i, piece_idx, offset);
            int64_t base_offset = sm_piece_offsets[piece_idx - piece_idx0];
            const T* ptr_input = input + base_offset + offset;
            reg_dst = *ptr_input;
            *ptr_local_output = reg_dst;
        }
    } else {
        int block_size = blockDim.x << SHIFT;
        block_size = block_size >> shift;
        for(int i = threadIdx.x; i < block_size; i += blockDim.x) {
            DST_T reg_dst;
            T * ptr_local_output = ptr_output + (i << shift);
            T * ptr_reg_dst = (T*)(&reg_dst);
            int piece_idx, offset;
            piece_size_fast.divmod(index0 + (i<<shift), piece_idx, offset);
            int64_t base_offset = sm_piece_offsets[piece_idx - piece_idx0];
            const T* ptr_input = input + base_offset + offset;
            reg_dst = *(DST_T*)(ptr_input);
            *(DST_T*)(ptr_local_output) = reg_dst;
        }
    }
}

#endif//_OPT_GATHER_
template <typename T>
__global__ void ppl_cukernel_gather_nd(
    int64_t num_elems,
    DivModFast piece_size_fast,
    int64_t* piece_offsets,
    const T* input,
    T* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int piece_idx, offset;
    piece_size_fast.divmod(index, piece_idx, offset);
    int64_t base_offset = piece_offsets[piece_idx];
    output[index]       = input[base_offset + offset];
}

template <typename IndexT>
__global__ void ppl_cukernel_gather_nd_offset(
    int64_t num_pieces,
    DivModFast num_pieces_per_batch_fast,
    int batch_dim,
    int64_t* input_dims_gpu,
    int input_batch_stride,
    int64_t* input_strides_gpu,
    int indices_last_dim_size,
    const IndexT* indices_data,
    int64_t* piece_offsets)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_pieces)
        return;
    // batch offset
    int batch_idx             = num_pieces_per_batch_fast.div(index);
    int64_t batch_offset      = batch_idx * input_batch_stride;
    // inner offset
    const IndexT* indices_ptr = indices_data + index * indices_last_dim_size;
    int64_t rel_offset        = 0;
    for (int it = 0; it < indices_last_dim_size; ++it) {
        IndexT cor_val = indices_ptr[it];
        if (cor_val < 0)
            cor_val += input_dims_gpu[batch_dim + it];
        assert(cor_val >= 0 && cor_val < input_dims_gpu[batch_dim + it]);
        rel_offset += cor_val * input_strides_gpu[it];
    }
    piece_offsets[index] = batch_offset + rel_offset;
}

int64_t pplGatherNDGetTempBufferSize(
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* indices_shape,
    const void* indices)
{
    int num_input_dim   = input_shape->GetDimCount();
    int num_indices_dim = indices_shape->GetDimCount();
    int num_pieces      = indices_shape->CalcElementsToDimensionIncludingPadding(num_indices_dim - 1);
    // pieces offsets and input strides and input_dims
    int64_t total_size  = (num_pieces + 2 * num_input_dim) * sizeof(int64_t);
    return total_size;
}

ppl::common::RetCode PPLCUDAGatherNDForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* indices_shape,
    const void* indices,
    const ppl::common::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    int batch_dim)
{
    int num_batches           = input_shape->CalcElementsToDimensionIncludingPadding(batch_dim);
    int input_batch_stride    = input_shape->CalcElementsFromDimensionIncludingPadding(batch_dim);
    int num_indices_dim       = indices_shape->GetDimCount();
    int num_input_dim         = input_shape->GetDimCount();
    int indices_last_dim_size = indices_shape->GetDim(num_indices_dim - 1);
    int num_pieces            = indices_shape->CalcElementsToDimensionIncludingPadding(num_indices_dim - 1);
    DivModFast num_pieces_per_batch_fast(num_pieces / num_batches);
    int piece_size = input_shape->CalcElementsFromDimensionIncludingPadding(
        batch_dim + indices_last_dim_size);
    int block_size             = 256;
    // step 1: calcalute each piece's offset first
    int64_t* piece_offsets     = static_cast<int64_t*>(temp_buffer);
    int64_t* input_strides_gpu = piece_offsets + num_pieces;
    int64_t* input_dims_gpu    = input_strides_gpu + num_input_dim;
    std::vector<int64_t> input_strides(indices_last_dim_size);
    std::vector<int64_t> input_dims(num_input_dim);
    // dimension is partitioned as batch--indices_last_dim_size--piece_size
    int64_t acc_strides = piece_size;
    for (int it = 0; it < indices_last_dim_size; ++it) {
        input_strides[indices_last_dim_size - 1 - it] = acc_strides;
        acc_strides *= input_shape->GetDim(batch_dim + indices_last_dim_size - 1 - it);
    }
    for (int it = 0; it < num_input_dim; ++it)
        input_dims[it] = input_shape->GetDim(it);
#ifdef _OPT_GATHER_
    stride_param param;
    for(int it = 0; it < indices_last_dim_size; it++) {
        param.input_strides[it] = input_strides[it];
    }
    for(int it = 0; it < num_input_dim; it++) {
        param.input_dims_gpu[it] = input_dims[it];
    }
    switch (ppl::common::GetSizeOfDataType(indices_shape->GetDataType())) {
        case sizeof(int32_t): {
            int gridSize = (num_pieces + 511) >> 9;
            int blockSize = 512;
            ppl_cukernel_gather_nd_offset_opt<int32_t><<<gridSize, blockSize, 0, stream>>>
                (num_pieces, num_pieces_per_batch_fast, batch_dim, num_input_dim, param, input_batch_stride, indices_last_dim_size, (const int32_t*)indices, piece_offsets);
            break;
        }
        case sizeof(int64_t): {
            int gridSize = (num_pieces + 511) >> 9;
            int blockSize = 512;
            ppl_cukernel_gather_nd_offset_opt<int64_t><<<gridSize, blockSize, 0, stream>>>
                (num_pieces, num_pieces_per_batch_fast, batch_dim, num_input_dim, param, input_batch_stride, indices_last_dim_size, (const int64_t*)indices, piece_offsets);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#else//!_OPT_GATHER_
    cudaMemcpyAsync(input_strides_gpu, input_strides.data(), sizeof(int64_t) * indices_last_dim_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(input_dims_gpu, input_dims.data(), sizeof(int64_t) * num_input_dim, cudaMemcpyHostToDevice, stream);
    int cal_offset_grid = (num_pieces + block_size - 1) / block_size;
    switch (ppl::common::GetSizeOfDataType(indices_shape->GetDataType())) {
        case sizeof(int32_t): {
            ppl_cukernel_gather_nd_offset<<<cal_offset_grid, block_size, 0, stream>>>(num_pieces, num_pieces_per_batch_fast, batch_dim, input_dims_gpu, input_batch_stride, input_strides_gpu, indices_last_dim_size, (const int32_t*)indices, piece_offsets);
            break;
        }
        case sizeof(int64_t): {
            ppl_cukernel_gather_nd_offset<<<cal_offset_grid, block_size, 0, stream>>>(num_pieces, num_pieces_per_batch_fast, batch_dim, input_dims_gpu, input_batch_stride, input_strides_gpu, indices_last_dim_size, (const int64_t*)indices, piece_offsets);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#endif//_OPT_GATHER_

    // step2: begiin gather elements
    int64_t num_elems    = output_shape->CalcElementsIncludingPadding();
    int gather_grid_size = (num_elems + block_size - 1) / block_size;
    DivModFast piece_size_fast(piece_size);

#define SWITCH_CASE(TYPE)                                                                                                                                  \
    case sizeof(TYPE): {                                                                                                                                   \
        ppl_cukernel_gather_nd<<<gather_grid_size, block_size, 0, stream>>>(num_elems, piece_size_fast, piece_offsets, (const TYPE*)input, (TYPE*)output); \
        return ppl::common::RC_SUCCESS;                                                                                                                    \
    }

#ifdef _OPT_GATHER_
    int use_opt = 0;
    switch(ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        case 1:{
            if((piece_size & 7) == 0 ) {
                int piece_size_per_block = MIN((4096 + piece_size - 1) / piece_size + 2, 4096);
                if(piece_size <= 4096) {
                    if(piece_size_per_block <= 2048) {
                        use_opt = 1;
                        int gridSize = (num_elems + 4095) >> 12;
                        int blockSize = 512;
                        ppl_cukernel_gather_nd_opt<int8_t, float2, 8, 3, 3><<<gridSize, blockSize, 0, stream>>>(num_elems, piece_size_fast, num_pieces, piece_size_per_block, piece_offsets, (const int8_t*)input, (int8_t*)output);
                        break;
                    }
                }
            }
            if((piece_size & 3) == 0) {
                if(piece_size <= 2048) {
                    use_opt = 1;
                    int piece_size_per_block = MIN((2048 + piece_size - 1) / piece_size + 2, 2048);
                    int gridSize = (num_elems + 2047) >> 11;
                    int blockSize = 512;
                    ppl_cukernel_gather_nd_opt<int8_t, float, 4, 2, 2><<<gridSize, blockSize, 0, stream>>>(num_elems, piece_size_fast, num_pieces, piece_size_per_block, piece_offsets, (const int8_t*)input, (int8_t*)output);
                }
            } else {
                if(piece_size <= 2048) {
                    use_opt = 1;
                    int piece_size_per_block = MIN((2048 + piece_size - 1) / piece_size + 2, 2048);
                    int gridSize = (num_elems + 2047) >> 11;
                    int blockSize = 512;
                    ppl_cukernel_gather_nd_opt<int8_t, int8_t, 1, 0, 2><<<gridSize, blockSize, 0, stream>>>(num_elems, piece_size_fast, num_pieces, piece_size_per_block, piece_offsets, (const int8_t*)input, (int8_t*)output);
                }
            }
            break;
        };
        case 2:{
            if((piece_size & 7) == 0 ) {
                int piece_size_per_block = MIN((4096 + piece_size - 1) / piece_size + 2, 4096);
                if(piece_size <= 4096) {
                    if(piece_size_per_block <= 2048) {
                        use_opt = 1;
                        int gridSize = (num_elems + 4095) >> 12;
                        int blockSize = 512;
                        ppl_cukernel_gather_nd_opt<int16_t, float4, 8, 3, 3><<<gridSize, blockSize, 0, stream>>>(num_elems, piece_size_fast, num_pieces, piece_size_per_block, piece_offsets, (const int16_t*)input, (int16_t*)output);
                        break;
                    }
                }
            }
            if((piece_size & 3) == 0) {
                if(piece_size <= 2048) {
                    use_opt = 1;
                    int piece_size_per_block = MIN((2048 + piece_size - 1) / piece_size + 2, 2048);
                    int gridSize = (num_elems + 2047) >> 11;
                    int blockSize = 512;
                    ppl_cukernel_gather_nd_opt<int16_t, float2, 4, 2, 2><<<gridSize, blockSize, 0, stream>>>(num_elems, piece_size_fast, num_pieces, piece_size_per_block, piece_offsets, (const int16_t*)input, (int16_t*)output);
                }
            } else {
                if(piece_size <= 2048) {
                    use_opt = 1;
                    int piece_size_per_block = MIN((2048 + piece_size - 1) / piece_size + 2, 2048);
                    int gridSize = (num_elems + 2047) >> 11;
                    int blockSize = 512;
                    ppl_cukernel_gather_nd_opt<int16_t, int16_t, 1, 0, 2><<<gridSize, blockSize, 0, stream>>>(num_elems, piece_size_fast, num_pieces, piece_size_per_block, piece_offsets, (const int16_t*)input, (int16_t*)output);
                }
            }
            break;
        }
        case 4:{
            if(piece_size <= 2048) {
                use_opt = 1;
                int piece_size_per_block = MIN((2048 + piece_size - 1) / piece_size + 2, 2048);
                int gridSize = (num_elems + 2047) >> 11;
                int blockSize = 512;
                ppl_cukernel_gather_nd_opt<int32_t, int32_t, 1, 0, 2><<<gridSize, blockSize, 0, stream>>>(num_elems, piece_size_fast, num_pieces, piece_size_per_block, piece_offsets, (const int32_t*)input, (int32_t*)output);
            }
            break; 
        }
        case 8:{
            if(piece_size <= 2048) {
                use_opt = 1;
                int piece_size_per_block = MIN((2048 + piece_size - 1) / piece_size + 2, 2048);
                int gridSize = (num_elems + 2047) >> 11;
                int blockSize = 512;
                ppl_cukernel_gather_nd_opt<int64_t, int64_t, 1, 0, 2><<<gridSize, blockSize, 0, stream>>>(num_elems, piece_size_fast, num_pieces, piece_size_per_block, piece_offsets, (const int64_t*)input, (int64_t*)output);
            }
            break; 
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
    if(use_opt) return ppl::common::RC_SUCCESS; 
#endif//_OPT_GATHER_

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
