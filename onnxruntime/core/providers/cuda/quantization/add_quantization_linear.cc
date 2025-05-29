// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "add_quantization_linear.h"
#include "add_quantization_linear_impl.cuh"
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
      QLinearAdd,                                                        \
      kMSDomain,                                                         \
      1,                                                                 \
      T##_##TQuant,                                                      \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 4)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 5)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 6)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 7)                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("A_scale", DataTypeImpl::GetTensorType<float>())   \
          .TypeConstraint("B_scale", DataTypeImpl::GetTensorType<float>())   \
          .TypeConstraint("C_scale", DataTypeImpl::GetTensorType<float>()),  \
      QLinearAdd<T, TQuant>);

REGISTER_KERNEL_TYPED(int8_t, int8_t)

template <typename T>
Status QLinearAdd<T, int8_t>::CheckInputs(const Tensor* input,
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
Status QLinearAdd<T, int8_t>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* A = context->Input<Tensor>(0);
  const Tensor* A_scale = context->Input<Tensor>(1);
  const Tensor* A_zp = context->Input<Tensor>(2);
  const Tensor* B = context->Input<Tensor>(3);
  const Tensor* B_scale = context->Input<Tensor>(4);
  const Tensor* B_zp = context->Input<Tensor>(5);
  const Tensor* C_scale = context->Input<Tensor>(6);
  const Tensor* C_zp = context->Input<Tensor>(7);

  const auto& A_shape = A->Shape();
  const auto& B_shape = B->Shape();

  if (A_shape != B_shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA QLinearAdd not support broadcast");
  }

  // TensorShapeVector output_shape(4);

  // output_shape[0] = A_shape[0];
  // output_shape[1] = A_shape[1];
  // output_shape[2] = A_shape[2]; // A_shape[0];
  // output_shape[3] = A_shape[3]; // A_shape[1];

  Tensor* C = context->Output(0, A_shape);

  typedef typename ToCudaType<T>::MappedType CudaT;

  float input_a_scale = *(A_scale->template Data<float>());
  CudaT input_a_zp = *(reinterpret_cast<const CudaT*>(A_zp->template Data<T>()));
  float input_b_scale = *(B_scale->template Data<float>());
  CudaT input_b_zp = *(reinterpret_cast<const CudaT*>(B_zp->template Data<T>()));
  float output_c_scale = *(C_scale->template Data<float>());
  CudaT output_c_zp = *(reinterpret_cast<const CudaT*>(C_zp->template Data<T>()));

  ORT_RETURN_IF_ERROR(CudaQLinearAdd(Stream(),
                      reinterpret_cast<const CudaT*>(A->template Data<T>()),
                      reinterpret_cast<const CudaT*>(B->template Data<T>()),
                      reinterpret_cast<CudaT*>(C->template MutableData<T>()),
                      input_a_scale,
                      input_a_zp,
                      input_b_scale,
                      input_b_zp,
                      output_c_scale,
                      output_c_zp,
                      A_shape.Size()));

  // =================== call cudnn api ===================
  // const Tensor* A = context->Input<Tensor>(0);
  // const Tensor* B = context->Input<Tensor>(3);
  // TensorShapeVector output_shape(4);

  // output_shape[0] = 1;
  // output_shape[1] = 1;
  // output_shape[2] = 2; // A_shape[0];
  // output_shape[3] = 2; // A_shape[1];

  // Tensor* C = context->Output(0, output_shape);

  // typedef typename ToCudaType<T>::MappedType CudaT;

  // CudnnTensor a_tensor;
  // const void* a_data = nullptr;
  // TensorShapeVector a_dims(4, 1);
  // a_dims[0] = 1;
  // a_dims[1] = 1;
  // a_dims[2] = 2;
  // a_dims[3] = 2;
  // ORT_RETURN_IF_ERROR(a_tensor.Set(a_dims, CudnnTensor::GetDataType<CudaT>()));
  // a_data = reinterpret_cast<const CudaT*>(A->template Data<T>());

  // CudnnTensor b_tensor;
  // const void* b_data = nullptr;
  // TensorShapeVector b_dims(4, 1);
  // b_dims[0] = 1;
  // b_dims[1] = 1;
  // b_dims[2] = 2;
  // b_dims[3] = 2;
  // ORT_RETURN_IF_ERROR(b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
  // b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());

  // CudnnTensor c_tensor;
  // void* c_data = nullptr;
  // TensorShapeVector c_dims(4, 1);
  // c_dims[0] = 1;
  // c_dims[1] = 1;
  // c_dims[2] = 2;
  // c_dims[3] = 2;
  // ORT_RETURN_IF_ERROR(c_tensor.Set(c_dims, CudnnTensor::GetDataType<CudaT>()));
  // c_data = reinterpret_cast<CudaT*>(C->template MutableData<T>());

  // cudnnHandle_t handle = CudnnHandle();
  // const auto alpha = Consts<float>::One;
  // const auto beta = Consts<float>::Zero;
  // // CUDNN_RETURN_IF_ERROR(cudnnAddTensor(handle, &alpha, a_tensor, a_data,
  // //                                        &alpha, s_.y_tensor, s_.y_data));
  // cudnnOpTensorDescriptor_t opTensorDesc;
  // cudnnCreateOpTensorDescriptor(&opTensorDesc);
  // cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT,
  //                           CUDNN_PROPAGATE_NAN);
  // CUDNN_RETURN_IF_ERROR(cudnnOpTensor(
  //                       handle,
  //                       opTensorDesc,
  //                       &alpha,
  //                       a_tensor,
  //                       a_data,
  //                       &alpha,
  //                       b_tensor,
  //                       b_data,
  //                       &beta,
  //                       c_tensor,
  //                       c_data
  // ));
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
