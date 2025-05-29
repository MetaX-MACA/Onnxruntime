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

#include "cudakernel/nn/trilu.h"
#include "cudakernel/common/divmod_fast.h"
#include "ppl/common/types.h"
#include "ppl/common/log.h"
#include <cuda_fp16.h>
#include <algorithm>

#define DivUp(x,y) (((x) + (y) - 1) / (y))

template<typename T>
__global__ void ppl_cukernel_trilu(
    const uint64_t num_elems,
    const T *input, T *output,
    int MN, int M, int upper, int *k) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;  
    int K_value = __ldg(k);
    int index_inner = index % MN;
    int row = index_inner / M;
    int col = index_inner % M;  
    if (upper == 1) {  // triu
        if (col-row < K_value) output[index] = 0;
        else output[index] = input[index];
    } else {    // tril
        if (col-row > K_value) output[index] = 0;
        else output[index] = input[index];
    }
}

template<typename T, typename PackT, int PackSize>
__global__ void ppl_cukernel_trilu_opt(
    const uint64_t num_elems,
    const T *input, T *output,
    DivModFast fastMN, DivModFast fastM,
    int upper, int *k) {
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) * PackSize;
    if (index >= num_elems) return;  
    int col, row, index_inner;
    int K_value = __ldg(k);
    if (num_elems - index >= PackSize){
        PackT in_val = *((PackT*)(input + index));
        PackT *out_val = (PackT*)(output + index);
        PackT reg = make_float4(1,2,3,4);
        T* ptr_data = (T*)&in_val;
        T* ptr_reg = (T*)&reg;
        for (int i = 0; i < PackSize; i++){
            index_inner = fastMN.mod(index + i);
            fastM.divmod(index_inner, row, col);
            if (upper == 1) {  // triu
                if (col-row < K_value) ptr_reg[i] = 0;
                else ptr_reg[i] = ptr_data[i];
            } else {    // tril
                if (col-row > K_value) ptr_reg[i] = 0;
                else ptr_reg[i] = ptr_data[i];
            }
        }
        out_val[0] = reg;
    } else {
        for (int i = index ; i < num_elems; i++){
            index_inner = fastMN.mod(i);
            fastM.divmod(index_inner, row, col);
            if (upper == 1) {  // triu
                if (col-row < K_value) output[i] = 0;
                else output[i] = input[i];
            } else {    // tril
                if (col-row > K_value) output[i] = 0;
                else output[i] = input[i];
            }
        }
    }
}

ppl::common::RetCode PPLCUDATriluForwardImp(
    cudaStream_t stream,
    int upper, const void* k,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output) {
    
    int dim_count = output_shape->GetDimCount();

    int M = output_shape->GetDim(dim_count - 1);
    int N = output_shape->GetDim(dim_count - 2);
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int block_size = 256;
            
    switch(output_shape->GetDataType()) {
        case ppl::common::DATATYPE_FLOAT32: {
            uint64_t grid_size = DivUp(num_elems, block_size * 4);
            ppl_cukernel_trilu_opt<float, float4, 4><<<grid_size, block_size>>>(num_elems, (const float*) input, (float*) output, DivModFast(M*N), DivModFast(M), upper, (int*) k);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16: {
            uint64_t grid_size = DivUp(num_elems, block_size * 8);
            ppl_cukernel_trilu_opt<half, float4, 8><<<grid_size, block_size>>>(num_elems, (const half*) input, (half*) output, DivModFast(M*N), DivModFast(M), upper, (int*) k);
            break;
        }
        case ppl::common::DATATYPE_INT8: {
            uint64_t grid_size = DivUp(num_elems, block_size * 16);
            ppl_cukernel_trilu_opt<int8_t, float4, 16><<<grid_size, block_size>>>(num_elems, (const int8_t*) input, (int8_t*) output, DivModFast(M*N), DivModFast(M), upper, (int*) k);
            break;
        }
        default:
            // uint64_t grid_size = DivUp(num_elems, block_size);
            // ppl_cukernel_trilu<<<grid_size, block_size>>>(num_elems, (const T*) input, (T*) output, M*N, M, upper, (int*) k);
            LOG(WARNING) << "Need implement PPLCUDATriluForwardImp";
            break;
    }
    
    return ppl::common::RC_SUCCESS;
}
