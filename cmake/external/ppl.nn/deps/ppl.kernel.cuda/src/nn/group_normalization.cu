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

#include "cudakernel/nn/group_normalization.h"
#include "cudakernel/common/common.cuh"
#include "cudakernel/math/math.h"
#include "cudakernel/common/divmod_fast.h"
#include "ppl/common/tensor_shape.h"
#include <cuda_fp16.h>
#include <iostream>
#define DivUp(x,y) (((x) + (y) - 1) / (y))
#ifdef PPLNN_USE_MACA
#define GROUP_NORMAL_OPT
#endif
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

template<typename T>
__inline__ __device__ __host__ T convert_to(float t) {
    return T(t);
}

template<>
__inline__ __device__ __host__ int8_t convert_to<int8_t>(float t) {
    int64_t temp = round(t);
    if (temp > 127)         temp = 127;
    else if (temp < -128)   temp = -128;
    return (int8_t)temp;
}

template<typename T, typename TPar>
__global__ void ppl_cukernel_instancenorm(const T* in, const TPar* alpha,
                                const TPar* beta, T* out, const int channels,
                                const int HW, bool with_relu, bool with_swish, const float eps = 1e-5,
                                const float input_scale = 1.0f, const float output_scale = 1.0f) {

    int b_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    auto cur_in = in + b_idx * channels * HW + c_idx * HW;
    auto cur_out = out + b_idx * channels * HW + c_idx * HW;
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    for(auto tid = threadIdx.x; tid < HW; tid += blockDim.x) {
        float v = (float)__ldg(cur_in + tid);
        sum.x += v;
        sum.y += v * v;
    }
    if(sizeof(T) == 1){
        sum.x *= input_scale;
        sum.y *= (input_scale * input_scale);
    }
    //BlockReduceSum
    BlockDoubleReduceSum(sum.x, sum.y);
    float mean = sum.x / HW;
    float rstd = rsqrtf(sum.y / HW - mean * mean + float(eps));
    for(auto tid = threadIdx.x; tid < HW; tid += blockDim.x) {
        float out_val;
        if(sizeof(T) > 1)
            out_val = ((float)__ldg(cur_in + tid) - mean) * rstd * (float)__ldg(alpha + c_idx) + (float)__ldg(beta + c_idx);
        else
            out_val = ((float)__ldg(cur_in + tid) * input_scale - mean) * rstd * (float)__ldg(alpha + c_idx) + (float)__ldg(beta + c_idx);
        if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
        if(with_swish) {
            float div_value = 1.0 + expf(-out_val);
            out_val = out_val / div_value;
        }
        if(sizeof(T) == 1) out_val = out_val / output_scale;
        cur_out[tid] = convert_to<T>(out_val);
    }

}

#ifdef GROUP_NORMAL_OPT
template<typename T, typename TPar>
__global__ void ppl_cukernel_instancenorm_fusion(const T* in, const TPar* alpha,
                                const TPar* beta, T* out, const int channels,
                                const int HW, bool with_relu, bool with_swish,
                                const int channels_per_group, DivModFast hw_div,
                                const float eps = 1e-5,
                                const float input_scale = 1.0f, const float output_scale = 1.0f) {

    int b_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    __shared__ TPar shared_alpha[256];
    __shared__ TPar shared_beta[256];
    int start_channel = c_idx * channels_per_group;
    for (int i = 0; i < channels_per_group; i++) {
        shared_alpha[i] = __ldg(alpha + start_channel + i);
        shared_beta[i] = __ldg(beta + start_channel + i);
    }
    __syncthreads();
    auto cur_in = in + b_idx * channels * HW + c_idx * HW;
    auto cur_out = out + b_idx * channels * HW + c_idx * HW;
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    for(auto tid = threadIdx.x; tid < HW; tid += blockDim.x) {
        float v = (float)__ldg(cur_in + tid);
        sum.x += v;
        sum.y += v * v;
    }
    if(sizeof(T) == 1){
        sum.x *= input_scale;
        sum.y *= (input_scale * input_scale);
    }
    //BlockReduceSum
    BlockDoubleReduceSum(sum.x, sum.y);
    float mean = sum.x / HW;
    float rstd = rsqrtf(sum.y / HW - mean * mean + float(eps));
    for(auto tid = threadIdx.x; tid < HW; tid += blockDim.x) {
        TPar this_alpha = shared_alpha[hw_div.div(tid)];
        TPar this_beta = shared_beta[hw_div.div(tid)];
        float out_val;
        if(sizeof(T) > 1)
            out_val = ((float)__ldg(cur_in + tid) - mean) * rstd *
                            (float)this_alpha + (float)this_beta;
        else
            out_val = ((float)__ldg(cur_in + tid) * input_scale - mean) * rstd *
                            (float)this_alpha + (float)this_beta;
        if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
        if(with_swish) {
            float div_value = 1.0 + expf(-out_val);
            out_val = out_val / div_value;
        }
        if(sizeof(T) == 1) out_val = out_val / output_scale;
        cur_out[tid] = convert_to<T>(out_val);
    }

}

/**
    Cache opt will trying to store data in registers for each thread when which
    firstly visit the input.
    But we are not sure that all the data can be stored into cache, so the original
    loop still remains working.
*/
template<typename T, typename TPar, int CACHE_SIZE=64>
__launch_bounds__(1024)
__global__ void ppl_cukernel_instancenorm_cache_opt(const T* in, const TPar* alpha,
                                const TPar* beta, T* out, const int channels,
                                const int HW, bool with_relu, bool with_swish,const float eps = 1e-5,
                                const float input_scale = 1.0f, const float output_scale = 1.0f) {
    int b_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    auto cur_in = in + b_idx * channels * HW + c_idx * HW;
    auto cur_out = out + b_idx * channels * HW + c_idx * HW;
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    T b[CACHE_SIZE];
    auto tid = threadIdx.x;
    int idx = 0;
    for(; idx < CACHE_SIZE; tid += blockDim.x) {
        auto vh = __ldg(cur_in + tid);
        b[idx++] = vh;
        float v = (float)vh;
        sum.x += v;
        sum.y += v * v;
    }
    for(; tid < HW; tid += blockDim.x) {
        float v = (float)__ldg(cur_in + tid);
        sum.x += v;
        sum.y += v * v;
    }
    if(sizeof(T) == 1){
        sum.x *= input_scale;
        sum.y *= (input_scale * input_scale);
    }
    TPar alpha_t = __ldg(alpha + c_idx);
    TPar beta_t = __ldg(beta + c_idx);
    //BlockReduceSum
    BlockDoubleReduceSum(sum.x, sum.y);
    float mean = sum.x / HW;
    float rstd = rsqrtf(sum.y / HW - mean * mean + float(eps));
    float alpha_f = (float)alpha_t;
    float beta_f = (float)beta_t;
    idx = 0;
    tid = threadIdx.x;
    for(; idx < CACHE_SIZE; tid += blockDim.x) {
        float out_val;
        if(sizeof(T) > 1)
            out_val = ((float)b[idx++]-mean) * rstd *
                            alpha_f + beta_f;
        else
            out_val = ((float)b[idx++] * input_scale -mean) * rstd *
                            alpha_f + beta_f;
        if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
        if(sizeof(T) == 1) out_val = out_val / output_scale;
        cur_out[tid] = convert_to<T>(out_val);
    }
    for(; tid < HW; tid += blockDim.x) {
        float out_val;
        if(sizeof(T) > 1)
            out_val = ((float)__ldg(cur_in + tid)-mean) * rstd *
                            alpha_f + beta_f;
        else
            out_val = ((float)__ldg(cur_in + tid) * input_scale -mean) * rstd *
                            alpha_f + beta_f;
        if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
        if(with_swish) {
            float div_value = 1.0 + expf(-out_val);
            out_val = out_val / div_value;
        }
        if(sizeof(T) == 1) out_val = out_val / output_scale;
        cur_out[tid] = convert_to<T>(out_val);
    }
}


template<typename T, typename TPar, int CACHE_SIZE=64>
__launch_bounds__(1024)
__global__ void ppl_cukernel_instancenorm_cache_fusion_opt(const T* in, const TPar* alpha,
                                const TPar* beta, T* out, const int channels,
                                const int HW, bool with_relu, bool with_swish,
                                const int channels_per_group, DivModFast hw_div,
                                const float eps = 1e-5,
                                const float input_scale = 1.0f, const float output_scale = 1.0f) {
    int b_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    __shared__ TPar shared_alpha[256];
    __shared__ TPar shared_beta[256];
    int start_channel = c_idx * channels_per_group;
    for (int i = 0; i < channels_per_group; i++) {
        shared_alpha[i] = __ldg(alpha + start_channel + i);
        shared_beta[i] = __ldg(beta + start_channel + i);
    }
    __syncthreads();
    auto cur_in = in + b_idx * channels * HW + c_idx * HW;
    auto cur_out = out + b_idx * channels * HW + c_idx * HW;
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    T b[CACHE_SIZE];
    auto tid = threadIdx.x;
    int idx = 0;
    for(; idx < CACHE_SIZE; tid += blockDim.x) {
        auto vh = __ldg(cur_in + tid);
        b[idx++] = vh;
        float v = (float)vh;
        sum.x += v;
        sum.y += v * v;
    }
    for(; tid < HW; tid += blockDim.x) {
        float v = (float)__ldg(cur_in + tid);
        sum.x += v;
        sum.y += v * v;
    }
    if(sizeof(T) == 1){
        sum.x *= input_scale;
        sum.y *= (input_scale * input_scale);
    }
    //BlockReduceSum
    BlockDoubleReduceSum(sum.x, sum.y);
    float mean = sum.x / HW;
    float rstd = rsqrtf(sum.y / HW - mean * mean + float(eps));
    idx = 0;
    tid = threadIdx.x;
    for(; idx < CACHE_SIZE; tid += blockDim.x) {
        TPar this_alpha = shared_alpha[hw_div.div(tid)];
        TPar this_beta = shared_beta[hw_div.div(tid)];
        float out_val;
        if(sizeof(T) > 1)
            out_val = ((float)b[idx++]-mean) * rstd *
                            (float)this_alpha + (float)this_beta;
        else
            out_val = ((float)b[idx++] * input_scale -mean) * rstd *
                            (float)this_alpha + (float)this_beta;
        if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
        if(with_swish) {
            float div_value = 1.0 + expf(-out_val);
            out_val = out_val / div_value;
        }
        if(sizeof(T) == 1) out_val = out_val / output_scale;
        cur_out[tid] = convert_to<T>(out_val);
    }
    for(; tid < HW; tid += blockDim.x) {
        TPar this_alpha = shared_alpha[hw_div.div(tid)];
        TPar this_beta = shared_beta[hw_div.div(tid)];
        float out_val;
        if(sizeof(T) > 1)
            out_val = ((float)__ldg(cur_in + tid)-mean) * rstd *
                            (float)this_alpha + (float)this_beta;
        else
            out_val = ((float)__ldg(cur_in + tid) * input_scale -mean) * rstd *
                            (float)this_alpha + (float)this_beta;
        if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
        if(with_swish) {
            float div_value = 1.0 + expf(-out_val);
            out_val = out_val / div_value;
        }
        if(sizeof(T) == 1) out_val = out_val / output_scale;
        cur_out[tid] = convert_to<T>(out_val);
    }
}



/**
    We have tested that if we are trying to store more than 64 data into cache
    we may get negative optimization effects.
    no-cache opt will works when data is aligned with 64bit, and we will load
    and store data in a packed way.
    We have also tested that, we have the best performance when PackT = uint64_t
*/
template<typename T, typename TPar>
__launch_bounds__(1024)
__global__ void ppl_cukernel_instancenorm_nocache_opt(const T* in, const TPar* alpha,
                                const TPar* beta, T* out, const int channels,
                                const int HW, bool with_relu,  bool with_swish, const float eps = 1e-5,
                                const float input_scale = 1.0f, const float output_scale = 1.0f) {
    int b_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    using PackT = uint64_t;
    auto cur_in = reinterpret_cast<const PackT*>(in + b_idx * channels * HW + c_idx * HW);
    auto cur_out = reinterpret_cast<PackT*>(out + b_idx * channels * HW + c_idx * HW);
    constexpr int PackSize = sizeof(PackT) / sizeof(T);
    union Package {
        PackT packed;
        T     unpacked[PackSize];
    };
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    for(auto tid = threadIdx.x; tid < HW / PackSize; tid += blockDim.x) {
        Package p;
        p.packed = __ldg(cur_in + tid);
        #pragma unroll
        for (int i = 0; i < PackSize; i++) {
            float v = (float)p.unpacked[i];
            sum.x += v;
            sum.y += v * v;
        }
    }
    if(sizeof(T) == 1){
        sum.x *= input_scale;
        sum.y *= (input_scale * input_scale);
    }
    TPar alpha_t = __ldg(alpha + c_idx);
    TPar beta_t = __ldg(beta + c_idx);
    //BlockReduceSum
    BlockDoubleReduceSum(sum.x, sum.y);
    float mean = sum.x / HW;
    float rstd = rsqrtf(sum.y / HW - mean * mean + float(eps));
    float alpha_f = (float)alpha_t;
    float beta_f = (float)beta_t;
    for(auto tid = threadIdx.x; tid < HW/PackSize; tid += blockDim.x) {
        Package p,o;
        p.packed = __ldg(cur_in + tid);
        #pragma unroll
        for (int i = 0; i < PackSize; i++) {
            float v = (float)p.unpacked[i];
            float out_val;
            if(sizeof(T) > 1)
                out_val = (v-mean) * rstd *
                            alpha_f + beta_f;
            else
                out_val = (v * input_scale -mean) * rstd *
                            alpha_f + beta_f;
            if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
            if(with_swish) {
                float div_value = 1.0 + expf(-out_val);
                out_val = out_val / div_value;
            }
            if(sizeof(T) == 1) out_val = out_val / output_scale;
            o.unpacked[i] = convert_to<T>(out_val);
        }
        cur_out[tid] = o.packed;
    }
}

template<typename T, typename TPar>
__launch_bounds__(1024)
__global__ void ppl_cukernel_instancenorm_nocache_fusion_opt(const T* in, const TPar* alpha,
                                const TPar* beta, T* out, const int channels,
                                const int HW, bool with_relu, bool with_swish,
                                const int channels_per_group, DivModFast hw_div,
                                const float eps = 1e-5,
                                const float input_scale = 1.0f, const float output_scale = 1.0f) {
    int b_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    __shared__ TPar shared_alpha[256];
    __shared__ TPar shared_beta[256];
    int start_channel = c_idx * channels_per_group;
    for (int i = 0; i < channels_per_group; i++) {
        shared_alpha[i] = __ldg(alpha + start_channel + i);
        shared_beta[i] = __ldg(beta + start_channel + i);
    }
    __syncthreads();
    using PackT = uint64_t;
    auto cur_in = reinterpret_cast<const PackT*>(in + b_idx * channels * HW + c_idx * HW);
    auto cur_out = reinterpret_cast<PackT*>(out + b_idx * channels * HW + c_idx * HW);
    constexpr int PackSize = sizeof(PackT) / sizeof(T);
    union Package {
        PackT packed;
        T     unpacked[PackSize];
    };
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;

    for(auto tid = threadIdx.x; tid < HW / PackSize; tid += blockDim.x) {
        Package p;
        p.packed = __ldg(cur_in + tid);
        #pragma unroll
        for (int i = 0; i < PackSize; i++) {
            float v = (float)p.unpacked[i];
            sum.x += v;
            sum.y += v * v;
        }
    }
    if(sizeof(T) == 1){
        sum.x *= input_scale;
        sum.y *= (input_scale * input_scale);
    }
    //BlockReduceSum
    BlockDoubleReduceSum(sum.x, sum.y);
    float mean = sum.x / HW;
    float rstd = rsqrtf(sum.y / HW - mean * mean + float(eps));
    for(auto tid = threadIdx.x; tid < HW/PackSize; tid += blockDim.x) {
        Package p,o;
        p.packed = __ldg(cur_in + tid);
        TPar this_alpha = shared_alpha[hw_div.div(tid*PackSize)];
        TPar this_beta = shared_beta[hw_div.div(tid*PackSize)];
        #pragma unroll
        for (int i = 0; i < PackSize; i++) {
            float v = (float)p.unpacked[i];
            float out_val;
            if(sizeof(T) > 1)
                out_val = (v-mean) * rstd *
                            (float)this_alpha + (float)this_beta;
            else
                out_val = (v * input_scale -mean) * rstd *
                            (float)this_alpha + (float)this_beta;
            if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
            if(with_swish) {
                float div_value = 1.0 + expf(-out_val);
                out_val = out_val / div_value;
            }
            if(sizeof(T) == 1) out_val = out_val / output_scale;
            o.unpacked[i] = convert_to<T>(out_val);
        }
        cur_out[tid] = o.packed;
    }
}

template<typename T, typename TPar>
__launch_bounds__(1024)
__global__ void ppl_cukernel_instancenorm_fusion_nhwc(
                                const T* in, const TPar* alpha, const TPar* beta, T* out,
                                const int groups, const int P_channels, const int  channels,
                                const int GHW, int HW, bool with_relu, const int channels_per_group,
                                const float eps = 1e-5, const float input_scale = 1.0f, const float output_scale = 1.0f) {
    int b_idx = blockIdx.y;
    int g_idx = blockIdx.x;
    int tid = threadIdx.x;

    int start_channel = g_idx * channels_per_group;

    auto cur_in = in + b_idx * groups * GHW;
    auto cur_out = out + b_idx * channels * HW;

    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    
    for(int hw_idx = tid; hw_idx < HW; hw_idx += blockDim.x) {
        auto this_cur_in = cur_in + hw_idx * P_channels;
        for (int i = 0; i < channels_per_group; i++) {
            int c_idx = start_channel + i;
            float v = (float)__ldg(this_cur_in + c_idx);
            sum.x += v;
            sum.y += v * v;
        }
    }
    if(sizeof(T) == 1){
        sum.x *= input_scale;
        sum.y *= (input_scale * input_scale);
    }
    //BlockReduceSum
    BlockDoubleReduceSum(sum.x, sum.y);

    float mean = sum.x / GHW;
    float rstd = rsqrtf(sum.y / GHW - mean * mean + float(eps));
    for(int hw_idx = tid; hw_idx < HW; hw_idx += blockDim.x) {
        auto this_cur_in = cur_in + hw_idx * P_channels;
        auto this_cur_out = cur_out + hw_idx * channels;
        for (int i = 0; i < channels_per_group; i++) {
            int c_idx = start_channel + i;
            float v = (float)__ldg(this_cur_in + c_idx);
            float out_val;
            if(sizeof(T) > 1)
                out_val = (v-mean) * rstd *
                            (float)__ldg(alpha + c_idx) + (float)__ldg(beta + c_idx);
            else
                out_val = (v * input_scale -mean) * rstd *
                            (float)__ldg(alpha + c_idx) + (float)__ldg(beta + c_idx);
            if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
            if(sizeof(T) == 1) out_val = out_val / output_scale;
            this_cur_out[c_idx] = convert_to<T>(out_val);;
        }
    }
}

template<typename T, typename TPar, typename PackT, int CACHE_SIZE>
__launch_bounds__(1024)
__global__ void ppl_cukernel_instancenorm_cache_fusion_nhwc16_opt_align(
                                const T* __restrict__ in, const TPar* __restrict__ alpha, const TPar* __restrict__ beta, T* __restrict__ out,
                                const int groups, const int  channels,
                                const int GHW, int HW, bool with_relu, const int channels_per_group,
                                const float eps = 1e-5, const float input_scale = 1.0f, const float output_scale = 1.0f) {

    int b_idx = blockIdx.y;
    int g_idx = blockIdx.x;
    __shared__ TPar shared_alpha[256];
    __shared__ TPar shared_beta[256];
    int start_channel = g_idx * channels_per_group;
    for (int i = 0; i < channels_per_group; i++) {
        int c_idx = start_channel + i;
        shared_alpha[i] = __ldg(alpha + c_idx);
        shared_beta[i] = __ldg(beta + c_idx);
    }
    __syncthreads();
    int tid = threadIdx.x;
    auto cur_in = reinterpret_cast<const PackT*>(in + b_idx * groups * GHW + start_channel);
    auto cur_out = reinterpret_cast<PackT*>(out + b_idx * channels * HW + start_channel);

    constexpr int PackSize = sizeof(PackT) / sizeof(T);
    union Package {
        PackT packed;
        T     unpacked[PackSize];
    };
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    
    Package p[CACHE_SIZE];
    int idx = 0;
    for(int hw_idx = tid; hw_idx < HW; hw_idx += blockDim.x) {
        int hw_offset = hw_idx * groups;
        p[idx].packed = __ldg(cur_in + hw_offset);
        #pragma unroll
        for (int i = 0; i < PackSize; i++) {
            float v = (float)p[idx].unpacked[i];
            sum.x += v;
            sum.y += v * v;
        }
        idx++;
    }
    if(sizeof(T) == 1){
        sum.x *= input_scale;
        sum.y *= (input_scale * input_scale);
    }

    //BlockReduceSum
    BlockDoubleReduceSum(sum.x, sum.y);

    float mean = sum.x / GHW;
    float rstd = rsqrtf(sum.y / GHW - mean * mean + float(eps));

    idx = 0;
    for(int hw_idx = tid; hw_idx < HW; hw_idx += blockDim.x) {
        T unpacked[PackSize];
        int hw_offset = hw_idx * groups;
        // PackT Temp = 0;
        Package o;
        #pragma unroll
        for (int i = 0; i < PackSize; i++) {
            TPar this_alpha = shared_alpha[i];
            TPar this_beta = shared_beta[i];
            float v = (float)p[idx].unpacked[i];
            float out_val;
            
            if(sizeof(T) > 1)
                out_val = (v-mean) * rstd *
                            (float)this_alpha + (float)this_beta;
            else
                out_val = (v * input_scale -mean) * rstd *
                            (float)this_alpha + (float)this_beta;
            
            if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
            
            if(sizeof(T) == 1) out_val = out_val / output_scale;
            o.unpacked[i] = convert_to<T>(out_val);
            // Temp |= (PackT(unpacked[i]&((T)(-1)))) << (i*sizeof(T)*8);
        }
        idx++;
        
        // cur_out[hw_offset] = Temp;
        cur_out[hw_offset] = o.packed;
    }
}

template<typename T, typename PackT>
__global__ void ppl_cukernel_instancenorm_fusion_nhwc_first(
                            const T* in, float* means, float* rstds, const int GHW, int HW, 
                            const int channels, const int groups, const int channels_per_group,
                            const float eps = 1e-5, const float input_scale = 1.0f, const float output_scale = 1.0f){
    
    int b_idx  = blockIdx.x;
    int tid = threadIdx.x;
    int g_idx = blockIdx.y;
    auto cur_in = in + b_idx * HW * channels;
    constexpr int PackSize = sizeof(PackT) / sizeof(T);
    union Package {
        PackT packed;
        T     unpacked[PackSize];
    };
    
    float sum = 0.0f;
    float pow = 0.0f;
    for(int hw_idx = tid; hw_idx < HW; hw_idx += blockDim.x) {
        auto this_cur_in = reinterpret_cast<const PackT*>(cur_in + hw_idx * channels);   
        Package p;
        p.packed = __ldg(this_cur_in + g_idx);
        for(int j = 0; j < PackSize; j++){
            float v = (float)p.unpacked[j];
            sum += v;
            pow += v * v;
        }
    }
    if(sizeof(T) == 1){
        sum *= input_scale;
        pow *= (input_scale * input_scale);
    }

    BlockDoubleReduceSum(sum, pow);
    
    if (tid == 0){
        float mean = sum / GHW;
        float rstd = rsqrtf(pow / GHW - mean * mean + float(eps));
        int out_offset = b_idx * groups + g_idx;
        means[out_offset] = mean;
        rstds[out_offset] = rstd;
    }
}

template<typename T, typename TPar, typename PackT>
__global__ void ppl_cukernel_instancenorm_fusion_nhwc_second(
                                const T* in, const TPar* alpha, const TPar* beta, T* out, const float* means, const float* rstds, 
                                const int groups, const int padding, const int  channels,const int GHW, int HW, bool with_relu, 
                                const int channels_per_group, const float eps = 1e-5, const float input_scale = 1.0f, const float output_scale = 1.0f){
    int b_idx  = blockIdx.x;
    int hwblock_idx = blockIdx.y;
    int block_hwidx = threadIdx.x;
    int g_idx = threadIdx.y;
    __shared__ TPar shared_alpha[256];
    __shared__ TPar shared_beta[256];

    int start_channel = g_idx * channels_per_group;
    int c_idx = start_channel + (block_hwidx & 0x7);
    shared_alpha[c_idx] = __ldg(alpha + c_idx);
    shared_beta[c_idx] = __ldg(beta + c_idx);
    __syncthreads();
    int poffset = b_idx * groups + g_idx;
    float mean = __ldg(means + poffset);
    float rstd = __ldg(rstds + poffset);
    constexpr int PackSize = sizeof(PackT) / sizeof(T);
    union Package {
        PackT packed;
        T     unpacked[PackSize];
    };
   
    int hw_idx = hwblock_idx * blockDim.x + block_hwidx;
    int inout_offset = (b_idx * HW + hw_idx) * channels;

    auto this_cur_in = reinterpret_cast<const PackT*>(in + inout_offset);
    auto this_cur_out = reinterpret_cast<PackT*>(out + inout_offset);
    Package p, o;
    p.packed = __ldg(this_cur_in + g_idx);
    for (int i = 0; i < channels_per_group; i++) {
        int c_idx = start_channel + i;
        TPar this_alpha = shared_alpha[c_idx];
        TPar this_beta = shared_beta[c_idx];
        float v = (float)p.unpacked[i];
        float out_val;
        if(sizeof(T) > 1)
            out_val = (v-mean) * rstd *
                        (float)this_alpha + (float)this_beta;
        else
            out_val = (v * input_scale -mean) * rstd *
                        (float)this_alpha + (float)this_beta;
        if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
        if(sizeof(T) == 1) out_val = out_val / output_scale;
        o.unpacked[i] = convert_to<T>(out_val);
    }
    this_cur_out[g_idx] = o.packed;
}

#define USE_CACHE(T, TPar, CACHE_SIZE) \
    if (fusion) {   \
        ppl_cukernel_instancenorm_cache_fusion_opt<T,TPar,CACHE_SIZE><<<grid_size, block_size, 0, stream>>>(    \
            (const T*)input, (const TPar*)scale, (const TPar*)B, (T*)output,                                    \
                    channels, hw_count, with_relu, with_swish, n_groups, DivModFast(H*W), epsilon, in_scale, out_scale                           \
        );                                                                                                      \
    } else {                                                                                                    \
        ppl_cukernel_instancenorm_cache_opt<T,TPar,CACHE_SIZE><<<grid_size, block_size, 0, stream>>>(           \
            (const T*)input, (const TPar*)scale, (const TPar*)B, (T*)output,                                    \
                    channels, hw_count, with_relu, with_swish,epsilon, in_scale, out_scale                                                      \
        );                                                                                                      \
    }

#define USE_OPT(T, TPar) \
    if (data_size_per_thread > 64) {                                                                        \
        if (hw_count % 4 == 0) {                                                                            \
            if (fusion) {                                                                                   \
                ppl_cukernel_instancenorm_nocache_fusion_opt<T,TPar><<<grid_size, block_size, 0, stream>>>( \
                (const T*)input, (const TPar*)scale, (const TPar*)B, (T*)output,                            \
                    channels, hw_count, with_relu, with_swish, n_groups, DivModFast(H*W), epsilon, in_scale, out_scale                       \
                );                                                                                          \
            } else {                                                                                        \
                ppl_cukernel_instancenorm_nocache_opt<T,TPar><<<grid_size, block_size, 0, stream>>>(        \
                (const T*)input, (const TPar*)scale, (const TPar*)B, (T*)output,                            \
                    channels, hw_count, with_relu, with_swish, epsilon, in_scale, out_scale                                                  \
                );                                                                                          \
            }                                                                                               \
        } else {                                                                                            \
            if (fusion) {                                                                                   \
                ppl_cukernel_instancenorm_fusion<T,TPar><<<grid_size, block_size, 0, stream>>>(             \
                (const T*)input, (const TPar*)scale, (const TPar*)B, (T*)output,                            \
                    channels, hw_count, with_relu, with_swish, n_groups, DivModFast(H*W), epsilon, in_scale, out_scale                       \
                );                                                                                          \
            } else {                                                                                        \
                ppl_cukernel_instancenorm<T,TPar><<<grid_size, block_size, 0, stream>>>(                    \
                (const T*)input, (const TPar*)scale, (const TPar*)B, (T*)output,                            \
                    channels, hw_count, with_relu, with_swish, epsilon, in_scale, out_scale                                           \
                );                                                                                          \
            }                                                                                               \
        }                                                                                           \
    } else if (data_size_per_thread == 64) {                                                        \
        USE_CACHE(T, TPar, 64)                                                                      \
    } else if (data_size_per_thread >= 32) {                                                        \
        USE_CACHE(T, TPar, 32)                                                                      \
    } else if (data_size_per_thread >= 16) {                                                        \
        USE_CACHE(T, TPar, 16)                                                                      \
    } else if (data_size_per_thread >= 8) {                                                         \
        USE_CACHE(T, TPar, 8)                                                                       \
    } else if (data_size_per_thread >= 4) {                                                         \
        USE_CACHE(T, TPar, 4)                                                                       \
    } else if (data_size_per_thread >= 2) {                                                         \
        USE_CACHE(T, TPar, 2)                                                                       \
    } else {                                                                                        \
        USE_CACHE(T, TPar, 1)                                                                       \
    }
#endif

#define NHWC_CACHE_OPT(T, TPar, PackT, CACHE_SIZE) \
    ppl_cukernel_instancenorm_cache_fusion_nhwc16_opt_align<T, TPar, PackT, CACHE_SIZE><<<grid_size, block_size, 0, stream>>>(   \
        (const T*)input, (const TPar*)scale, (const TPar*)B, (T*)output,                           \
        groups, C, GHW, HW, with_relu, n_groups, epsilon, in_scale, out_scale);   

ppl::common::RetCode PPLCUDAGroupNormalizationForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input,
    ppl::common::TensorShape* scale_shape,
    const void* scale,
    // share scale shape
    const void* B,
    ppl::common::TensorShape* output_shape,
    void* output,
    int n_groups,
    float epsilon,
    float in_scale,
    float out_scale,
    bool with_relu,
    bool with_swish)
{
    // Since we just support NCHW format, We can use InstanceNorm to impl.
    auto N = input_shape->GetDim(0);
    auto C = input_shape->GetDim(1);
    auto H = input_shape->GetDim(2);
    auto W = input_shape->GetDim(3);
#ifdef GROUP_NORMAL_OPT
    bool fusion = scale_shape->GetDim(0) == C;
#endif

    int numElements = N * C * H * W;
    int batch = N;

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        int channels = C/n_groups;
        int hw_count = n_groups*H*W;
#ifdef GROUP_NORMAL_OPT
        int data_size_per_thread = (hw_count + 1024 - 1) / 1024;
        int max_block_size = 1024;
        if (data_size_per_thread > 64) {
            max_block_size = 1024;
        } else if (data_size_per_thread >= 32 && data_size_per_thread <= 64) {
            if (fusion) max_block_size = 512;
            else max_block_size = 1024;
        } else {
            max_block_size = 512;
        }
        int block_size    = GetBlockSize(hw_count, max_block_size);
#else
        int block_size    = GetBlockSize(hw_count);
#endif
        dim3 grid_size(channels, batch, 1);
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
#ifdef GROUP_NORMAL_OPT
            data_size_per_thread = hw_count / block_size;
            USE_OPT(float, float)
#else
            ppl_cukernel_instancenorm<float, float><<<grid_size, block_size, 0, stream>>>(
                (const float*)input, (const float*)scale, (const float*)B, (float*)output,
                channels, hw_count, with_relu, with_swish,epsilon);
#endif
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
#ifdef GROUP_NORMAL_OPT
            data_size_per_thread = hw_count / block_size;
            USE_OPT(half, half)
#else
            ppl_cukernel_instancenorm<half, half><<<grid_size, block_size, 0, stream>>>(
                (const half*)input, (const half*)scale, (const half*)B, (half*)output,
                channels, hw_count, with_relu, with_swish,epsilon);
#endif
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
#ifdef GROUP_NORMAL_OPT
            data_size_per_thread = hw_count / block_size;
            USE_OPT(int8_t, float)
#else
            ppl_cukernel_instancenorm<int8_t, float><<<grid_size, block_size, 0, stream>>>(
                (const int8_t*)input, (const float*)scale, (const float*)B, (int8_t*)output,
                channels, hw_count, with_relu,with_swish, epsilon, in_scale, out_scale);
#endif
        } else {
             return ppl::common::RC_UNSUPPORTED;
        }
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16){
#ifdef GROUP_NORMAL_OPT
        int padding = input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
        int HW = H * W;
        int GHW = n_groups * HW;

        if (padding == 0){
            int groups = C / n_groups;
            dim3 grid_size(groups, batch, 1);
            int block_size = GetBlockSize(HW, 1024);
            int cache_size = GHW >> 10;

            if (n_groups == 8){
                if (cache_size > 128){
                    float* means, *rstds;
                    cudaMalloc((float**)&means, batch * groups * sizeof(float));
                    cudaMemset(means, 0, batch * groups * sizeof(float));
                    cudaMalloc((float**)&rstds, batch * groups * sizeof(float));
                    cudaMemset(rstds, 0, batch * groups * sizeof(float));
                    int blocksize2 = 1024;
                    dim3 GridSize(batch, groups, 1);
                    dim3 BlockSize(blocksize2, 1, 1);
                    ppl_cukernel_instancenorm_fusion_nhwc_first<int8_t, uint64_t><<<GridSize, BlockSize, 0, stream>>>(
                        (const int8_t*)input, (float*)means, (float*)rstds, GHW, HW, C, groups, n_groups, epsilon, in_scale, out_scale);

                    blocksize2 = 512;
                    int hwcount_per_block = DivUp(GetBlockSize(HW, blocksize2), groups);
                    int hwblock = DivUp(HW, hwcount_per_block);  // 167x167 / 16 = 1744
                    dim3 GridSize_2(batch, hwblock, 1);
                    dim3 BlockSize_2(hwcount_per_block, groups, 1);
                    ppl_cukernel_instancenorm_fusion_nhwc_second<int8_t, float, uint64_t><<<GridSize_2, BlockSize_2, 0, stream>>>(
                            (const int8_t*)input, (const float*)scale, (const float*)B, (int8_t*)output, (float*)means, (float*)rstds,
                                groups, padding, C, GHW, HW, with_relu, n_groups,
                                epsilon, in_scale, out_scale);
                } else if (cache_size > 8){
                    NHWC_CACHE_OPT(int8_t, float, uint64_t, 64)
                } else {
                    block_size = GetBlockSize(HW, 512);
                    cache_size = GHW >> 9;
                    if (cache_size < 1) NHWC_CACHE_OPT(int8_t, float, uint64_t, 1)
                    else if (cache_size < 2) NHWC_CACHE_OPT(int8_t, float, uint64_t, 2)
                    else if (cache_size < 4) NHWC_CACHE_OPT(int8_t, float, uint64_t, 4)
                    else if (cache_size < 8) NHWC_CACHE_OPT(int8_t, float, uint64_t, 8)
                    else if (cache_size < 16) NHWC_CACHE_OPT(int8_t, float, uint64_t, 16)
                    else if (cache_size < 32) NHWC_CACHE_OPT(int8_t, float, uint64_t, 32)
                    else NHWC_CACHE_OPT(int8_t, float, uint64_t, 64)
                    
                }
            } else {
                ppl_cukernel_instancenorm_fusion_nhwc<int8_t, float><<<grid_size, block_size, 0, stream>>>(
                    (const int8_t*)input, (const float*)scale, (const float*)B, (int8_t*)output, 
                        groups, C, C, GHW, HW, with_relu, n_groups,
                        epsilon, in_scale, out_scale);
            }
        } else {  //exist padding
            int P_channels = C + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
            int groups = P_channels / n_groups;
            dim3 grid_size(groups, batch, 1);
            int block_size = GetBlockSize(HW, 512);
            ppl_cukernel_instancenorm_fusion_nhwc<int8_t, float><<<grid_size, block_size, 0, stream>>>(
                (const int8_t*)input, (const float*)scale, (const float*)B, (int8_t*)output, 
                    groups, P_channels, C, GHW, HW, with_relu, n_groups,
                    epsilon, in_scale, out_scale);
        }
#else
        int P_channels = C + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
        int groups = P_channels / n_groups;
        dim3 grid_size(groups, batch, 1);
        int block_size = GetBlockSize(HW, 512);
        ppl_cukernel_instancenorm_fusion_nhwc<int8_t, float><<<grid_size, block_size, 0, stream>>>(
            (const int8_t*)input, (const float*)scale, (const float*)B, (int8_t*)output, 
                groups, P_channels, C, GHW, HW, with_relu, n_groups,
                epsilon, in_scale, out_scale);
#endif
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8){
#ifdef GROUP_NORMAL_OPT
        int padding = input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
        int HW = H * W;
        int GHW = n_groups * HW;

        if (padding == 0){
            int groups = C / n_groups;
            dim3 grid_size(groups, batch, 1);
            int block_size = GetBlockSize(HW, 1024);
            int cache_size = GHW >> 10;

            if (n_groups == 8){
                if (cache_size > 32){
                    half* means, *rstds;
                    cudaMalloc((float**)&means, batch * groups * sizeof(float));
                    cudaMemset(means, 0, batch * groups * sizeof(float));
                    cudaMalloc((float**)&rstds, batch * groups * sizeof(float));
                    cudaMemset(rstds, 0, batch * groups * sizeof(float));
                    int blocksize2 = 1024;
                    dim3 GridSize(batch, groups, 1);
                    dim3 BlockSize(blocksize2, 1, 1);
                    ppl_cukernel_instancenorm_fusion_nhwc_first<half, float4><<<GridSize, BlockSize, 0, stream>>>(
                        (const half*)input, (float*)means, (float*)rstds, GHW, HW, C, groups, n_groups, epsilon, in_scale, out_scale);

                    blocksize2 = 512;
                    int hwcount_per_block = DivUp(GetBlockSize(HW, blocksize2), groups);
                    int hwblock = DivUp(HW, hwcount_per_block);  
                    dim3 GridSize_2(batch, hwblock, 1);
                    dim3 BlockSize_2(hwcount_per_block, groups, 1);
                    ppl_cukernel_instancenorm_fusion_nhwc_second<half, half, float4><<<GridSize_2, BlockSize_2, 0, stream>>>(
                            (const half*)input, (const half*)scale, (const half*)B, (half*)output, (float*)means, (float*)rstds,
                                groups, padding, C, GHW, HW, with_relu, n_groups,
                                epsilon, in_scale, out_scale);
                } else if (cache_size > 8){
                    NHWC_CACHE_OPT(half, half, float4, 32)
                } else {
                    block_size = GetBlockSize(HW, 512);
                    cache_size = GHW >> 9;
                    if (cache_size < 1) NHWC_CACHE_OPT(half, half, float4, 1)
                    else if (cache_size < 2) NHWC_CACHE_OPT(half, half, float4, 2)
                    else if (cache_size < 4) NHWC_CACHE_OPT(half, half, float4, 4)
                    else if (cache_size < 8) NHWC_CACHE_OPT(half, half, float4, 8)
                    else if (cache_size < 16) NHWC_CACHE_OPT(half, half, float4, 16)
                    else NHWC_CACHE_OPT(half, half, float4, 32)
                    
                }
            } else {
                ppl_cukernel_instancenorm_fusion_nhwc<half, half><<<grid_size, block_size, 0, stream>>>(
                    (const half*)input, (const half*)scale, (const half*)B, (half*)output, 
                        groups, C, C, GHW, HW, with_relu, n_groups,
                        epsilon, in_scale, out_scale);
            }
        } else {  //exist padding
            int P_channels = C + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
            int groups = P_channels / n_groups;
            dim3 grid_size(groups, batch, 1);
            int block_size = GetBlockSize(HW, 512);
            ppl_cukernel_instancenorm_fusion_nhwc<half, half><<<grid_size, block_size, 0, stream>>>(
                (const half*)input, (const half*)scale, (const half*)B, (half*)output, 
                    groups, P_channels, C, GHW, HW, with_relu, n_groups,
                    epsilon, in_scale, out_scale);
        }
#else
        int P_channels = C + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
        int groups = P_channels / n_groups;
        dim3 grid_size(groups, batch, 1);
        int block_size = GetBlockSize(HW, 512);
        ppl_cukernel_instancenorm_fusion_nhwc<half, half><<<grid_size, block_size, 0, stream>>>(
            (const half*)input, (const half*)scale, (const half*)B, (half*)output, 
                groups, P_channels, C, GHW, HW, with_relu, n_groups,
                epsilon, in_scale, out_scale);
#endif
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}
