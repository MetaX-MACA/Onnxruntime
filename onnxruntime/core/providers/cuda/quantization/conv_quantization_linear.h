// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "gsl/gsl"

#ifndef UNUSED
    #define UNUSED(x) (void)(x)
#endif

namespace onnxruntime {
namespace cuda {

template <typename T>
class QLinearConv final : public CudaKernel {

 public:
  QLinearConv(const OpKernelInfo& info) : CudaKernel(info) { }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status CheckInputs(const Tensor* input,
                     const Tensor* weights,
                     const Tensor* bias,
                     const Tensor* input_scale_tensor,
                     const Tensor* weight_scale_tensor,
                     const Tensor*& mask_index,
                     const Tensor* i_zp_tensor,
                     const Tensor* w_zp_tensor,
                     const Tensor* past_tensor) const;
};

}  // namespace cuda
}  // namespace onnxruntime
