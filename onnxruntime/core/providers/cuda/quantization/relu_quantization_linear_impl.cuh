// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {
template <class Tin>
Status CudaQLinearRelu(cudaStream_t stream,
                        const Tin* A,
                        Tin* B,
                        float a_scale,
                        Tin a_zp,
                        float b_scale,
                        Tin b_zp, size_t count);

}  // namespace cuda
}  // namespace onnxruntime
