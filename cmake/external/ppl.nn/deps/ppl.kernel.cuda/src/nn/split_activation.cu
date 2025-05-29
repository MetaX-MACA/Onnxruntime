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
#include "ppl/common/tensor_shape.h"
#include <cuda_fp16.h>
#include "cudakernel/common/divmod_fast.h"
#ifdef __MACACC__
#define SPLIT_ACTIVATION_OPT
#endif//__MACACC__

// -------------  For Add + Activation Split   -------------------
template <typename T, int32_t HHS, int32_t TPB>
__global__ void biasSplitGeluKernel(T const* input, T const* bias, T* output) {
  int32_t index_input = blockIdx.x * HHS * 2 + threadIdx.x;
  int32_t index_output = blockIdx.x * HHS + threadIdx.x;
  int32_t index_bias = threadIdx.x;

#pragma unroll
  for (int32_t i = 0; i < HHS / TPB; ++i) {
    auto value_left = (float)(input[index_input] + bias[index_bias]);
    auto value_right = (float)(input[index_input + HHS] + bias[index_bias + HHS]);

    // Gelu is applied to right side only: Gelu(x) = x * 0.5 * (erf(x / sqrt(2)) + 1.0)
    float gelu_right = value_right * 0.5f * (erff(value_right / static_cast<float>(M_SQRT2)) + 1.0f);
    float result = value_left * gelu_right;
    output[index_output] = static_cast<T>(result);
    index_input += TPB;
    index_output += TPB;
    index_bias += TPB;
  }
  return;
}

#ifdef SPLIT_ACTIVATION_OPT
template <typename T,typename VT, int N, int SHIFT>
__global__ void biasSplitGeluKernel_opt(T const* input, T const* bias, T* output, int num_elems, DivModFast hws_fast, int32_t HHS) {
  int index_output = (blockIdx.x * blockDim.x + threadIdx.x) << SHIFT;
  if(index_output >= num_elems) return;
  int ver_id, hor_id;
  hws_fast.divmod(index_output, ver_id, hor_id);
  const T * ptr_input0 = input + ver_id * (HHS<<1) + hor_id;
  const T *ptr_input1 = ptr_input0 + HHS;
  const T* ptr_bias0 = bias + hor_id;
  const T* ptr_bias1 = ptr_bias0 + HHS;
  T* ptr_out = output + index_output;
  VT vsrc0 = *(VT*)ptr_input0;
  VT vsrc1 = *(VT*)ptr_input1;
  VT vbias0 = *(VT*)ptr_bias0;
  VT vbias1 = *(VT*)ptr_bias1;
  VT vdst;
  T* ptr_reg_src0 = (T*)&vsrc0;
  T* ptr_reg_src1 = (T*)&vsrc1;
  T* ptr_reg_bias0 = (T*)&vbias0;
  T* ptr_reg_bias1 = (T*)&vbias1;
  T* ptr_reg_dst = (T*)&vdst;
  #pragma unroll N
  for(int i = 0; i < N; i++) {
    float value_left = (float)ptr_reg_src0[i] + (float)(ptr_reg_bias0[i]);
    float value_right = (float)ptr_reg_src1[i] + (float)(ptr_reg_bias1[i]);
    float gelu_right = value_right * 0.5f * (erff(value_right * 0.70710678118654752440084) + 1.0f);
    float result = value_left * gelu_right;
    ptr_reg_dst[i] = static_cast<T>(result);
  }
  *(VT*)(ptr_out) = vdst;
}

template <typename T,typename VT, int N, int SHIFT>
__global__ void biasSplitGeluKernel_opt(T const* input, T* output, int num_elems, DivModFast hws_fast, int32_t HHS) {
  int index_output = (blockIdx.x * blockDim.x + threadIdx.x) << SHIFT;
  if(index_output >= num_elems) return;
  int ver_id, hor_id;
  hws_fast.divmod(index_output, ver_id, hor_id);
  const T * ptr_input0 = input + ver_id * (HHS<<1) + hor_id;
  const T *ptr_input1 = ptr_input0 + HHS;
  T* ptr_out = output + index_output;
  VT vsrc0 = *(VT*)ptr_input0;
  VT vsrc1 = *(VT*)ptr_input1;
  VT vdst;
  T* ptr_reg_src0 = (T*)&vsrc0;
  T* ptr_reg_src1 = (T*)&vsrc1;
  T* ptr_reg_dst = (T*)&vdst;
  #pragma unroll N
  for(int i = 0; i < N; i++) {
    float value_left = (float)ptr_reg_src0[i];
    float value_right = (float)ptr_reg_src1[i];
    float gelu_right = value_right * 0.5f * (erff(value_right * 0.70710678118654752440084) + 1.0f);
    float result = value_left * gelu_right;
    ptr_reg_dst[i] = static_cast<T>(result);
  }
  *(VT*)(ptr_out) = vdst;
}

#endif//SPLIT_ACTIVATION_OPT


template <typename T>
void LaunchBiasSplitGeluKernel(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size,
                               T const* input, T const* bias, T* output) {
  constexpr int32_t TPB = 256;  // thread per block
  switch (half_hidden_size) {
    case 1280:
      (biasSplitGeluKernel<T, 1280, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, output);
      break;
    case 2560:
      (biasSplitGeluKernel<T, 2560, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, output);
      break;
    case 5120:
      (biasSplitGeluKernel<T, 5120, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, output);
      break;
    case 3072:
      (biasSplitGeluKernel<T, 3072, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, output);
      break;
    case 6144:
      (biasSplitGeluKernel<T, 6144, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, output);
      break;
    default:
      // Notice we need default support.
      break;
  }
}

// -------------  For  Activation Split   -------------------

template <typename T, int32_t HHS, int32_t TPB>
__global__ void biasSplitGeluKernel(T const* input,  T* output) {
  int32_t index_input = blockIdx.x * HHS * 2 + threadIdx.x;
  int32_t index_output = blockIdx.x * HHS + threadIdx.x;

#pragma unroll
  for (int32_t i = 0; i < HHS / TPB; ++i) {
    auto value_left = (float)(input[index_input] );
    auto value_right = (float)(input[index_input + HHS]);

    // Gelu is applied to right side only: Gelu(x) = x * 0.5 * (erf(x / sqrt(2)) + 1.0)
    float gelu_right = value_right * 0.5f * (erff(value_right / static_cast<float>(M_SQRT2)) + 1.0f);
    float result = value_left * gelu_right;
    output[index_output] = static_cast<T>(result);
    index_input += TPB;
    index_output += TPB;
  }
  return;
}


template <typename T>
void LaunchBiasSplitGeluKernel(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size,
                               T const* input,  T* output) {
  constexpr int32_t TPB = 256;  // thread per block
  switch (half_hidden_size) {
    case 1280:
      (biasSplitGeluKernel<T, 1280, TPB>)<<<grid_size, TPB, 0, stream>>>(input, output);
      break;
    case 2560:
      (biasSplitGeluKernel<T, 2560, TPB>)<<<grid_size, TPB, 0, stream>>>(input, output);
      break;
    case 5120:
      (biasSplitGeluKernel<T, 5120, TPB>)<<<grid_size, TPB, 0, stream>>>(input, output);
      break;
    case 3072:
      (biasSplitGeluKernel<T, 3072, TPB>)<<<grid_size, TPB, 0, stream>>>(input, output);
      break;
    case 6144:
      (biasSplitGeluKernel<T, 6144, TPB>)<<<grid_size, TPB, 0, stream>>>(input, output);
      break;
    default:
      // Notice we need default support.
      break;
  }
}


//  ------ CALL   -------
ppl::common::RetCode PPLCUDASplitActivationForwardImp(
        cudaStream_t stream,
        const void* input,
        ppl::common::TensorShape* input_shape,
        const void* bias,
        ppl::common::TensorShape* bias_shape,
        int activation_type,
        void* output)
{

    const int32_t grid_size = static_cast<int32_t>(input_shape->GetDim(0) * input_shape->GetDim(1));
    const int32_t half_hidden_size = static_cast<int32_t>(input_shape->GetDim(2) / 2);
    if (bias != nullptr) {
#ifdef SPLIT_ACTIVATION_OPT
        int N = 16 / ppl::common::GetSizeOfDataType(input_shape->GetDataType());
        if((half_hidden_size & (N - 1)) == 0) {
          int num_elems = grid_size * half_hidden_size;
          int blockSize = 512;
          int gridSize = (num_elems + blockSize * N - 1) / (blockSize * N);
          if(N == 8) {
              biasSplitGeluKernel_opt<half, float4, 8, 3><<<gridSize, blockSize, 0 , stream>>>(reinterpret_cast<const half*>(input), 
                  reinterpret_cast<const half*>(bias),reinterpret_cast<half*>(output),num_elems, DivModFast(half_hidden_size), half_hidden_size);
          } else if(N == 4) {
              biasSplitGeluKernel_opt<float, float4, 4, 2><<<gridSize, blockSize, 0 , stream>>>(reinterpret_cast<const float*>(input),
                reinterpret_cast<const float*>(bias),reinterpret_cast<float*>(output),num_elems, DivModFast(half_hidden_size), half_hidden_size);
          } else {
            return ppl::common::RC_UNSUPPORTED;
          }
          return ppl::common::RC_SUCCESS;
        }
#endif//SPLIT_ACTIVATION_OPT
        if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            LaunchBiasSplitGeluKernel<half>(stream, grid_size, half_hidden_size,
                                        reinterpret_cast<const half*>(input),
                                        reinterpret_cast<const half*>(bias),
                                        reinterpret_cast<half*>(output));
            return ppl::common::RC_SUCCESS;
        } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            LaunchBiasSplitGeluKernel<float>(stream, grid_size, half_hidden_size,
                                        reinterpret_cast<const float*>(input),
                                        reinterpret_cast<const float*>(bias),
                                        reinterpret_cast<float*>(output));
            return ppl::common::RC_SUCCESS;
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }

    } else {
#ifdef SPLIT_ACTIVATION_OPT
        int N = 16 / ppl::common::GetSizeOfDataType(input_shape->GetDataType());
        if((half_hidden_size & (N - 1)) == 0) {
          int num_elems = grid_size * half_hidden_size;
          int blockSize = 512;
          int gridSize = (num_elems + blockSize * N - 1) / (blockSize * N);
          if(N == 8) {
              biasSplitGeluKernel_opt<half, float4, 8, 3><<<gridSize, blockSize, 0 , stream>>>(reinterpret_cast<const half*>(input), 
                reinterpret_cast<half*>(output),num_elems, DivModFast(half_hidden_size), half_hidden_size);
          } else if(N == 4) {
              biasSplitGeluKernel_opt<float, float4, 4, 2><<<gridSize, blockSize, 0 , stream>>>(reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),num_elems, DivModFast(half_hidden_size), half_hidden_size);
          } else {
            return ppl::common::RC_UNSUPPORTED;
          }
          return ppl::common::RC_SUCCESS;
        }
#endif//SPLIT_ACTIVATION_OPT
        if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            LaunchBiasSplitGeluKernel<half>(stream, grid_size, half_hidden_size,
                                        reinterpret_cast<const half*>(input),
                                        reinterpret_cast<half*>(output));
            return ppl::common::RC_SUCCESS;
        } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            LaunchBiasSplitGeluKernel<float>(stream, grid_size, half_hidden_size,
                                        reinterpret_cast<const float*>(input),
                                        reinterpret_cast<float*>(output));
            return ppl::common::RC_SUCCESS;
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    }

}
