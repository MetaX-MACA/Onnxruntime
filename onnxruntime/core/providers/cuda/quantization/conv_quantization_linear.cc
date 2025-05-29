// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv_quantization_linear.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/shared_inc/integer_gemm.h"
#include "core/providers/cuda/tensor/quantize_linear.h"

using namespace onnxruntime::cuda;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      QLinearConv,                                                        \
      kMSDomain,                                                         \
      1,                                                                 \
      T,                                                      \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 4)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 5)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 6)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 7)                        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("x_scale", DataTypeImpl::GetTensorType<float>())   \
          .TypeConstraint("w_scale", DataTypeImpl::GetTensorType<float>())   \
          .TypeConstraint("y_scale", DataTypeImpl::GetTensorType<float>()),  \
      QLinearConv<T>);

REGISTER_KERNEL_TYPED(int8_t)

template <typename T>
Status QLinearConv<T>::CheckInputs(const Tensor* input,
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
Status QLinearConv<T>::ComputeInternal(OpKernelContext* context) const {
  // =================== call cudnn api ===================
  const Tensor* A = context->Input<Tensor>(0);
  // const Tensor* B = context->Input<Tensor>(3);
  const auto& A_shape = A->Shape();
  // TensorShapeVector output_shape(4);

  // output_shape[0] = 1;
  // output_shape[1] = 1;
  // output_shape[2] = 2; // A_shape[0];
  // output_shape[3] = 2; // A_shape[1];

  Tensor* C = context->Output(0, A_shape);

  typedef typename ToCudaType<T>::MappedType CudaT;

  CudnnTensor a_tensor;
  const void* a_data = nullptr;
  // TensorShapeVector a_dims(4, 1);
  // a_dims[0] = 1;
  // a_dims[1] = 1;
  // a_dims[2] = 2;
  // a_dims[3] = 2;
  ORT_RETURN_IF_ERROR(a_tensor.Set(A_shape.AsShapeVector(), CudnnTensor::GetDataType<CudaT>()));
  a_data = reinterpret_cast<const CudaT*>(A->template Data<T>());

  // CudnnTensor b_tensor;
  // const void* b_data = nullptr;
  // TensorShapeVector b_dims(4, 1);
  // b_dims[0] = 1;
  // b_dims[1] = 1;
  // b_dims[2] = 2;
  // b_dims[3] = 2;
  // ORT_RETURN_IF_ERROR(b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
  // b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());

  CudnnTensor c_tensor;
  void* c_data = nullptr;
  // TensorShapeVector c_dims(4, 1);
  // c_dims[0] = 1;
  // c_dims[1] = 1;
  // c_dims[2] = 2;
  // c_dims[3] = 2;
  ORT_RETURN_IF_ERROR(c_tensor.Set(A_shape.AsShapeVector(), CudnnTensor::GetDataType<CudaT>()));
  c_data = reinterpret_cast<CudaT*>(C->template MutableData<T>());

  cudnnHandle_t handle = CudnnHandle();
  const auto alpha = Consts<float>::One;
  const auto beta = Consts<float>::Zero;
  // // CUDNN_RETURN_IF_ERROR(cudnnAddTensor(handle, &alpha, a_tensor, a_data,
  // //                                        &alpha, s_.y_tensor, s_.y_data));
  cudnnOpTensorDescriptor_t opTensorDesc;
  cudnnCreateOpTensorDescriptor(&opTensorDesc);
  cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT,
                            CUDNN_PROPAGATE_NAN);
  CUDNN_RETURN_IF_ERROR(cudnnOpTensor(
                        handle,
                        opTensorDesc,
                        &alpha,
                        a_tensor,
                        a_data,
                        &alpha,
                        a_tensor,
                        a_data,
                        &beta,
                        c_tensor,
                        c_data
  ));
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
