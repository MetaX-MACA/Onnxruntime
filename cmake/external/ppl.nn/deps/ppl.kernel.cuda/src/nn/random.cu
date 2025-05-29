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
#include <cuda_fp16.h>
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

template<typename T>
__global__ void random_uniform(unsigned long long seed,float low,float scale,int num_elems,T*output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed,idx,0,&state);
    float tmp = curand_uniform(&state);
    float value = tmp * scale + low;
    output[idx] = value;
}

template<typename T>
__global__ void random_uniformlike(unsigned long long seed,float low,float scale,int num_elems,T*output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed,idx,0,&state);
    uint32_t tmp = curand(&state);
    uint32_t tmp1 = (uint32_t)scale;
    int32_t tmp2 = (int32_t)low;
    T value = (T)(tmp % tmp1) + tmp2;
    output[idx] = value;
}

template<typename T>
__global__ void random_normal(unsigned long long seed,float mean,float scale,int num_elems,T*output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed,idx,0,&state);
    float tmp = curand_normal(&state);
    float value = tmp * scale + mean;
    output[idx] = value;
}

ppl::common::RetCode PPLCUDARandomUniformForwardImp(
    cudaStream_t stream,
    float low,
    float high,
    float seed,
    ppl::common::TensorShape* output_shape,
    void* output)
{
    int num_elems = output_shape->CalcElementsIncludingPadding();
    int blocksize = 512;
    int gridsize = (num_elems  + 511) >> 9;
    float scale = high - low;
    unsigned long long value;
    srand((unsigned int)time(NULL));
    if(seed != -65535){
        value = (unsigned long long)seed;
    }else{
        value = rand();
    }
    if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32){
        random_uniform<float><<<gridsize,blocksize,0,stream>>>(value,low,scale,num_elems,(float*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16){
        random_uniform<half><<<gridsize,blocksize,0,stream>>>(value,low,scale,num_elems,(half*)output);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDARandomUniformLikeForwardImp(
    cudaStream_t stream,
    float low,
    float high,
    float seed,
    ppl::common::TensorShape* output_shape,
    void* output)
{
    int num_elems = output_shape->CalcElementsIncludingPadding();
    int blocksize = 512;
    int gridsize = (num_elems  + 511) >> 9;
    float scale = high - low;
    if(scale == 0){
        return ppl::common::RC_UNSUPPORTED;
    }
    unsigned long long value;
    srand((unsigned int)time(NULL));
    if(seed != -65535){
        value = (unsigned long long)seed;
    }else{
        value = rand();
    }
    if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32){
        random_uniform<float><<<gridsize,blocksize,0,stream>>>(value,low,scale,num_elems,(float*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16){
        random_uniform<half><<<gridsize,blocksize,0,stream>>>(value,low,scale,num_elems,(half*)output);
    }else if (output_shape->GetDataType() == ppl::common::DATATYPE_UINT8) {
        random_uniformlike<uint8_t><<<gridsize,blocksize,0,stream>>>(value,low,scale,num_elems,(uint8_t*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        random_uniformlike<int8_t><<<gridsize,blocksize,0,stream>>>(value,low,scale,num_elems,(int8_t*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_UINT16) {
        random_uniformlike<uint16_t><<<gridsize,blocksize,0,stream>>>(value,low,scale,num_elems,(uint16_t*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT16) {
        random_uniformlike<int16_t><<<gridsize,blocksize,0,stream>>>(value,low,scale,num_elems,(int16_t*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_UINT32) {
        random_uniformlike<uint32_t><<<gridsize,blocksize,0,stream>>>(value,low,scale,num_elems,(uint32_t*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT32) {
        random_uniformlike<int32_t><<<gridsize,blocksize,0,stream>>>(value,low,scale,num_elems,(int32_t*)output);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDARandomNormalForwardImp(
    cudaStream_t stream,
    float mean,
    float scale,
    float seed,
    ppl::common::TensorShape* output_shape,
    void* output)
{
    int num_elems = output_shape->CalcElementsIncludingPadding();
    int blocksize = 512;
    int gridsize = (num_elems  + 511) >> 9;
    unsigned long long value;
    srand((unsigned int)time(NULL));
    if(seed != -65535){
        value = (unsigned long long)seed;
    }else{
        value = rand();
    }
    if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32){
        random_normal<float><<<gridsize,blocksize,0,stream>>>(value,mean,scale,num_elems,(float*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16){
        random_normal<half><<<gridsize,blocksize,0,stream>>>(value,mean,scale,num_elems,(half*)output);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDARandomNormalLikeForwardImp(
    cudaStream_t stream,
    float mean,
    float scale,
    float seed,
    ppl::common::TensorShape* output_shape,
    void* output)
{
    int num_elems = output_shape->CalcElementsIncludingPadding();
    int blocksize = 512;
    int gridsize = (num_elems  + 511) >> 9;
    unsigned long long value;
    srand((unsigned int)time(NULL));
    if(seed != -65535){
        value = (unsigned long long)seed;
    }else{
        value = rand();
    }
    if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32){
        random_normal<float><<<gridsize,blocksize,0,stream>>>(value,mean,scale,num_elems,(float*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16){
        random_normal<half><<<gridsize,blocksize,0,stream>>>(value,mean,scale,num_elems,(half*)output);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}
