// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "gsl/gsl"

#ifndef UNUSED
    #define UNUSED(x) (void)(x)
#endif

namespace onnxruntime {
namespace cuda {

template <typename T, typename TQuant>
class QLinearAdd;

template <typename T>
class QLinearAdd<T, int8_t> final : public CudaKernel {

 public:
  QLinearAdd(const OpKernelInfo& info) : CudaKernel(info) { }

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
// template <class T, class U = int8_t>
// class QLinearAdd final : public CudaKernel {
//  public:
//   QLinearAdd(const OpKernelInfo& info) : CudaKernel(info) {}

//   Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;
// };

}  // namespace cuda
}  // namespace onnxruntime
