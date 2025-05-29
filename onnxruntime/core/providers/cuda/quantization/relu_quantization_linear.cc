// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "relu_quantization_linear.h"
#include "relu_quantization_linear_impl.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/shared_inc/integer_gemm.h"
#include "core/providers/cuda/tensor/quantize_linear.h"

using namespace onnxruntime::cuda;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, TQuant)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      QLinearRelu,                                                       \
      kMetaxDomain,                                                      \
      1,                                                                 \
      T##_##TQuant,                                                      \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 4)                        \
          .TypeConstraint("X_scale", DataTypeImpl::GetTensorType<float>())   \
          .TypeConstraint("Y_scale", DataTypeImpl::GetTensorType<float>()),  \
      QLinearRelu<T, TQuant>);

REGISTER_KERNEL_TYPED(int8_t, int8_t)

template <typename T>
Status QLinearRelu<T, int8_t>::CheckInputs(const Tensor* input,
                                          const Tensor* weights,
                                          const Tensor* bias,
                                          const Tensor* input_scale_tensor,
                                          const Tensor* weight_scale_tensor,
                                          const Tensor*& mask_index,
                                          const Tensor* i_zp_tensor,
                                          const Tensor* w_zp_tensor,
                                          const Tensor* past_tensor) const {

  return Status::OK();
}

template <typename T>
Status QLinearRelu<T, int8_t>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* A = context->Input<Tensor>(0);
  const Tensor* A_scale = context->Input<Tensor>(1);
  const Tensor* A_zp = context->Input<Tensor>(2);
  const Tensor* B_scale = context->Input<Tensor>(3);
  const Tensor* B_zp = context->Input<Tensor>(4);

  const auto& A_shape = A->Shape();
  TensorShapeVector output_shape(4);

  output_shape[0] = A_shape[0];
  output_shape[1] = A_shape[1];
  output_shape[2] = A_shape[2];
  output_shape[3] = A_shape[3];

  Tensor* B = context->Output(0, output_shape);

  typedef typename ToCudaType<T>::MappedType CudaT;

  float input_a_scale = *(A_scale->template Data<float>());
  CudaT input_a_zp = *(reinterpret_cast<const CudaT*>(A_zp->template Data<T>()));
  float input_b_scale = *(B_scale->template Data<float>());
  CudaT input_b_zp = *(reinterpret_cast<const CudaT*>(B_zp->template Data<T>()));

  size_t count = A_shape.Size();

  ORT_RETURN_IF_ERROR(CudaQLinearRelu(Stream(),
                      reinterpret_cast<const CudaT*>(A->template Data<T>()),
                      reinterpret_cast<CudaT*>(B->template MutableData<T>()),
                      input_a_scale,
                      input_a_zp,
                      input_b_scale,
                      input_b_zp,
                      count));
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
