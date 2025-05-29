// Copyright 2023 metax-tech.com Inc.

#pragma once

#include "core/common/common.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------
template <typename ERRTYPE, bool THRW>
bool CudaCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg = "");


#define CUDA_CALL(expr) (CudaCall<cudaError, false>((expr), #expr, "CUDA", cudaSuccess))

#define CUDA_CALL_THROW(expr) (CudaCall<cudaError, true>((expr), #expr, "CUDA", cudaSuccess))

}  // namespace onnxruntime
