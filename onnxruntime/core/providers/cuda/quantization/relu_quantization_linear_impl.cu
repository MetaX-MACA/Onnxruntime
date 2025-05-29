// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "relu_quantization_linear_impl.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace cuda {

template <class T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void QLinearReluKernelNoScale(const T* A, T* B, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      B[id] = (A[id] > 0) ? A[id] : 0;
      id += NumThreadsPerBlock;
    }
  }
}

template <class T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void QLinearReluKernel(const T* A, T* B,
              float a_scale, T a_zp, float b_scale, T b_zp, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      float a = static_cast<float>(A[id] - a_zp) * a_scale;
      a = (a > 0) ? a : 0;
      float b = a / b_scale + b_zp;
      b = (b > 127) ? 127 : (b < -128) ? -128 : b;
      B[id] = static_cast<T>(__float2int_rn(b));
      id += NumThreadsPerBlock;
    }
  }
}

template <class T>
Status CudaQLinearRelu(cudaStream_t stream, const T* A, T* B,
              float a_scale, T a_zp, float b_scale, T b_zp, size_t count) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (a_scale == b_scale && a_zp == b_zp) {
    QLinearReluKernelNoScale<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      A, B, N);
  } else {
    QLinearReluKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      A, B, a_scale, a_zp, b_scale, b_zp, N);
  }

  return Status::OK();
}

template Status CudaQLinearRelu<int8_t>(cudaStream_t stream, const int8_t* A, int8_t* B,
              float a_scale, int8_t a_zp, float b_scale, int8_t b_zp, size_t count);


}  // namespace cuda
}  // namespace onnxruntime
