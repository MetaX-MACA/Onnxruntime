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

#include "cudakernel/nn/one_hot.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>

template <typename T>
__global__ void ppl_cukernel_one_hot(const int64_t* incides, T off_value, T on_value, T* output, int64_t outer, int64_t depth, int64_t inner){

        int tid = threadIdx.x;
        int outer_id = blockIdx.x;
        int inner_id = blockIdx.y;

        for(int id = tid; id < depth; id += blockDim.x){
            if(id < depth){
                uint64_t output_offset = outer_id * depth * inner + id * inner + inner_id;
                uint64_t incides_offset = outer_id * inner + inner_id;
                int64_t on_index = (incides[incides_offset] >= 0 ? incides[incides_offset] : (depth + incides[incides_offset]));
                if(id == on_index)
                    output[output_offset] = on_value;
                else
                    output[output_offset] = off_value;
            }
        }
}

template <typename T>
T  get_mask_value(cudaStream_t stream, const T* values, T* output, uint64_t nelem){
    T* host_value = (T*)malloc(sizeof(T)*2);
    cudaMemcpyAsync(host_value, (T*)values, sizeof(T) * 2, cudaMemcpyDeviceToHost, stream);
    T off_value = host_value[0];
    cudaMemsetAsync((void*)output, off_value, sizeof(T)*nelem, stream); // set off value
    T on_value = host_value[1];
    free(host_value);
    return on_value;
}

ppl::common::RetCode PPLCUDAOneHotForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* indices_shape,
    const void* indices,
    ppl::common::TensorShape* values_shape,
    const void* values,
    ppl::common::TensorShape* output_shape,
    void* output,
    uint32_t real_axis)
{
    uint64_t num_elems = output_shape->CalcElementsExcludingPadding();

    auto outer = output_shape->CalcElementsToDimensionExcludingPadding(real_axis);
    auto depth_val = output_shape->GetDim(real_axis);
    auto inner = output_shape->CalcElementsFromDimensionExcludingPadding(real_axis+1);

    int block_size     = 256;
    dim3 block(block_size);
    dim3 grid(outer, inner);

    auto datatype = values_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();

    int64_t *indices_cuda = nullptr;
    int64_t *indices_cpu_src = nullptr;
    int64_t *indices_cpu_dst = nullptr;
    if(indices_shape->GetDataType() != ppl::common::DATATYPE_INT64)
    {
        int64_t indices_ele_num = indices_shape->CalcElementsExcludingPadding();
        cudaMalloc(&indices_cuda, indices_ele_num * sizeof(int64_t));
        indices_cpu_src = (int64_t*)malloc(indices_ele_num * sizeof(int64_t));
        indices_cpu_dst = (int64_t*)malloc(indices_ele_num * sizeof(int64_t));
        if(indices_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32)
        {
            cudaMemcpyAsync(indices_cpu_src, indices, sizeof(float) * indices_ele_num, cudaMemcpyDeviceToHost, stream);
            //cudaStreamSynchronize(stream);
            float *indices_cpu_src_temp = (float *)indices_cpu_src;
            for(int i = 0; i < indices_ele_num; i++)
            {
                indices_cpu_dst[i] = (int64_t)indices_cpu_src_temp[i];
            }
            cudaMemcpyAsync(indices_cuda, indices_cpu_dst, sizeof(int64_t) * indices_ele_num, cudaMemcpyHostToDevice, stream);
        }
        else if (indices_shape->GetDataType() == ppl::common::DATATYPE_INT32)
        {
            cudaMemcpyAsync(indices_cpu_src, indices, sizeof(int32_t) * indices_ele_num, cudaMemcpyDeviceToHost, stream);
            int32_t *indices_cpu_src_temp = (int32_t *)indices_cpu_src;
            for(int i = 0; i < indices_ele_num; i++)
            {
                indices_cpu_dst[i] = (int64_t)indices_cpu_src_temp[i];
            }
            cudaMemcpyAsync(indices_cuda, indices_cpu_dst, sizeof(int64_t) * indices_ele_num, cudaMemcpyHostToDevice, stream);
        }
        else
        {
            cudaFree(indices_cuda);
            free(indices_cpu_src);
            free(indices_cpu_dst);
            return ppl::common::RC_UNSUPPORTED;
        }
        indices = indices_cuda;
    }

    switch(datatype){
        case ppl::common::DATATYPE_FLOAT32:{
            //float on_value = get_mask_value<float>(stream, (const float*)values, (float*)output, num_elems);
            float value[2];
            cudaMemcpyAsync(value, (float*)values, sizeof(float) * 2, cudaMemcpyDeviceToHost, stream);
            float off_value = value[0];
            float on_value = value[1];
            ppl_cukernel_one_hot<float><<<grid, block, 0, stream>>>((const int64_t*)indices, off_value, on_value, (float*)output, outer, depth_val, inner);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16:{
            //half on_value = get_mask_value<half>(stream, (const half*)values, (half*)output, num_elems);
            half value[2];
            cudaMemcpyAsync(value, (half*)values, sizeof(half) * 2, cudaMemcpyDeviceToHost, stream);
            half off_value = value[0];
            half on_value = value[1];
            ppl_cukernel_one_hot<half><<<grid, block, 0, stream>>>((const int64_t*)indices, off_value, on_value, (half*)output, outer, depth_val, inner);
            break;
        }
        case ppl::common::DATATYPE_INT64:{
            //int64_t on_value = get_mask_value<int64_t>(stream, (const int64_t*)values, (int64_t*)output, num_elems);
            int64_t value[2];
            cudaMemcpyAsync(value, (int64_t*)values, sizeof(int64_t) * 2, cudaMemcpyDeviceToHost, stream);
            int64_t off_value = value[0];
            int64_t on_value = value[1];
            ppl_cukernel_one_hot<int64_t><<<grid, block, 0, stream>>>((const int64_t*)indices, off_value, on_value, (int64_t*)output, outer, depth_val, inner);
            break;
        }
        case ppl::common::DATATYPE_INT32:{
            //int32_t on_value = get_mask_value<int32_t>(stream, (const int32_t*)values, (int32_t*)output, num_elems);
            int32_t value[2];
            cudaMemcpyAsync(value, (int32_t*)values, sizeof(int32_t) * 2, cudaMemcpyDeviceToHost, stream);
            int32_t off_value = value[0];
            int32_t on_value = value[1];
            ppl_cukernel_one_hot<int32_t><<<grid, block, 0, stream>>>((const int64_t*)indices, off_value, on_value, (int32_t*)output, outer, depth_val, inner);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }

    if(indices_shape->GetDataType() != ppl::common::DATATYPE_INT64)
    {
        cudaFree(indices_cuda);
        free(indices_cpu_src);
        free(indices_cpu_dst);
    }


    return ppl::common::RC_SUCCESS;
}
