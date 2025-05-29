// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "add_quantization_linear_impl.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace cuda {

template <class T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void QLinearAddKernel(const T* A, const T* B, T* C,
              float a_scale, T a_zp, float b_scale, T b_zp, float c_scale, T c_zp, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      float c = (static_cast<float>(A[id] - a_zp) * a_scale + static_cast<float>(B[id] - b_zp) * b_scale) / c_scale + c_zp;
      int res = __float2int_rn(c);
      if (res > 127) res = 127;
      else if (res < -128) res = -128;
      C[id] = static_cast<T>(res);
      id += NumThreadsPerBlock;
    }
  }
}

template <class T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void QLinearAddKernelNoScale(const T* A, const T* B, T* C, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      float c = static_cast<float>(A[id]) + static_cast<float>(B[id]);
      int res = __float2int_rn(c);
      if (res > 127) res = 127;
      else if (res < -128) res = -128;
      C[id] = static_cast<T>(res);
      id += NumThreadsPerBlock;
    }
  }
}

template <class T>
Status CudaQLinearAdd(cudaStream_t stream, const T* A, const T* B, T* C,
              float a_scale, T a_zp, float b_scale, T b_zp, float c_scale, T c_zp, size_t count) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  if (a_scale == b_scale && a_scale == c_scale && a_zp == b_zp && a_zp == c_zp && a_zp == 0) {
    QLinearAddKernelNoScale<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      A, B, C, N);
  } else {
    QLinearAddKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      A, B, C, a_scale, a_zp, b_scale, b_zp, c_scale, c_zp, N);
  }

  return Status::OK();
}

template Status CudaQLinearAdd<int8_t>(cudaStream_t stream, const int8_t* A, const int8_t* B, int8_t* C,
              float a_scale, int8_t a_zp, float b_scale, int8_t b_zp, float c_scale, int8_t c_zp, size_t count);


}  // namespace cuda
}  // namespace onnxruntime
