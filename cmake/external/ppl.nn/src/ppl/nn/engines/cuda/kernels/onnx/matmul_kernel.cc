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

#include "ppl/nn/engines/cuda/kernels/onnx/matmul_kernel.h"
#include "ppl/nn/engines/cuda/module/cuda_module.h"
#include "ppl/common/destructor.h"
#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/gemm/bgemm.h"

namespace ppl { namespace nn { namespace cuda {
#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
extern
ppl::common::RetCode mcblasGemmExBiasActFp16(mcblasLtHandle_t &ltHandle,
                                             mcblasLtMatmulDesc_t operationDesc,
                                             mcblasLtMatrixLayout_t Adesc,
                                             mcblasLtMatrixLayout_t Bdesc,
                                             mcblasLtMatrixLayout_t Cdesc,
                                             cublasOperation_t transa,
                                             cublasOperation_t transb,
                                             int32_t m,
                                             int32_t n,
                                             int32_t k,
                                             float alpha,
                                             const void *A, /* device pointer */
                                             int64_t lda,
                                             const void *B, /* device pointer */
                                             int64_t ldb,
                                             void *C, /* device pointer */
                                             int64_t ldc,
                                             float *bias_coef,     /* device pointer */
                                             void *bias, /* device pointer */
                                             int64_t ldbias,
                                             const int32_t bias_mode, /* 0-none, 1-scale, 2-vector, 3-matrix */
                                             const int32_t act_type,  /* 0-none, 1-Relu, 2-Gelu, 3-HardSwish */
                                             cudaStream_t stream,
                                             void *D);
#endif

// StridedBatchedGemm can be used for the following GEMM computation
// C[pnm] = A[pnk]*B[km] or C[pnm] = A[pnk]*B[pkm]
static bool CanUseStridedBatchedGemm(const TensorShape& left_shape, const TensorShape& right_shape,
                                     bool transa, bool transb, bool trans_batch_a, bool trans_batch_b,
                                     int64_t& stride_A, int64_t& stride_B, int64_t& stride_C, int64_t& batch_count) {
  auto left_num_dims = left_shape.GetDimCount();
  auto right_num_dims = right_shape.GetDimCount();

  if (!(left_num_dims >= 3 && right_num_dims >= 2)) {
    return false;
  }

  size_t left_leading_axis = trans_batch_a ? 0 : left_num_dims - 2;
  size_t right_leading_axis = trans_batch_b ? 0 : right_num_dims - 2;
  int64_t left_p = left_shape.CalcElementsToDimensionExcludingPadding(left_num_dims - 2);
  if (trans_batch_a) {
    left_p = left_p * left_shape.GetDim(left_num_dims - 2) / left_shape.GetDim(0);
  }
  int64_t left_k = transa ? left_shape.GetDim(left_leading_axis) : left_shape.GetDim(left_num_dims - 1);

  if (right_num_dims >= 3) {
    int64_t right_p = right_shape.CalcElementsToDimensionExcludingPadding(right_num_dims - 2);
    if (trans_batch_b) {
      right_p = right_p * right_shape.GetDim(right_num_dims - 2) / right_shape.GetDim(0);
    }
    if (left_p != right_p) {
      return false;
    }
  }

  int64_t right_k = transb ? right_shape.GetDim(right_num_dims - 1) : right_shape.GetDim(right_leading_axis);
  if (left_k != right_k) {
    return false;
  }

  int64_t n = transa ? left_shape.GetDim(left_num_dims - 1) : left_shape.GetDim(left_leading_axis);
  int64_t m = transb ? right_shape.GetDim(right_leading_axis) : right_shape.GetDim(right_num_dims - 1);
  stride_A = n * left_k / (trans_batch_a ? left_shape.GetDim(0) : 1);
  stride_B = right_num_dims == 2 ? 0 : right_k * m / (trans_batch_b ? right_shape.GetDim(0) : 1);
  stride_C = n * m;
  batch_count = left_p;
  return true;
}

bool MatMulKernel::CanDoExecute(const KernelExecContext& ctx) const {
    const TensorShape& input0 = *ctx.GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& input1 = *ctx.GetInput<TensorImpl>(1)->GetShape();
    if (input0.CalcBytesIncludingPadding() == 0) {
        return false;
    }
    if (input1.CalcBytesIncludingPadding() == 0) {
        return false;
    }
    // K must be the same
    uint32_t dim_count0 = input0.GetDimCount();
    uint32_t dim_count1 = input1.GetDimCount();
    if (input0.GetDim(dim_count0 - 1) != input1.GetDim(dim_count1 - 2)) {
        return false;
    }

    return true;
}

uint64_t MatMulKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    // TODO
    auto A = ctx.GetInput<TensorImpl>(0)->GetShape();
    return PPLBgemmCUDAGetBufSize(A, param_->param.transA);
}
#ifdef PPLNN_USE_DNN
#define  CHECK_CUBLAS_STATUS(cubalas_api_str, status)                                                           \
            do{                                                                                                 \
                if(status != CUBLAS_STATUS_SUCCESS){                                                            \
                    LOG(ERROR) <<"cublas interface(" << #cubalas_api_str << ") return error code: " << (int)status;   \
                    return ppl::common::RC_OTHER_ERROR;                                                         \
                }                                                                                               \
            }while(0)
#endif
ppl::common::RetCode MatMulKernel::DoExecute(KernelExecContext* ctx) {
#ifdef PPLNN_USE_DNN
    auto left_X = ctx->GetInput<TensorImpl>(0);
    auto right_X = ctx->GetInput<TensorImpl>(1);

    const TensorShape& shape_left = *left_X->GetShape();
    const TensorShape& shape_right = *right_X->GetShape();
    uint32_t dim_left = shape_left.GetDimCount();
    uint32_t dim_right = shape_right.GetDimCount();
    bool is_need_k8_align = false;
    bool isFp16 = (shape_left.GetDataType() == ppl::common::DATATYPE_FLOAT16);
#if USE_MATMUL_K8_ALIGN
    if(shape_left.GetDim(shape_left.GetDimCount() - 1) % 8 != 0 && shape_left.GetDataType() == ppl::common::DATATYPE_FLOAT16){
        if (shape_left.GetDim(shape_left.GetDimCount() - 1) > 4 )
        {
            is_need_k8_align = true;
        }
    }
    if(is_need_k8_align){
        LOG(DEBUG) << "is_need_k8_align condition is true!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
        auto temp_shape_left = shape_left;
        auto k = temp_shape_left.GetDim(temp_shape_left.GetDimCount() - 1);
        temp_shape_left.SetDim(temp_shape_left.GetDimCount() - 1, ALIGN_UP(k, 8));
        if(temp_shape_left.CalcBytesExcludingPadding() != matmul_k8_a_len_){
            if(matmul_k8_a_buffer_ != nullptr){
                cudaFree(matmul_k8_a_buffer_);
                matmul_k8_a_buffer_ = nullptr;
            }
            cudaMalloc(&matmul_k8_a_buffer_, temp_shape_left.CalcBytesIncludingPadding());
            if(matmul_k8_a_buffer_ == nullptr)
                return RC_OUT_OF_MEMORY;
            cudaMemsetAsync(matmul_k8_a_buffer_, 0, temp_shape_left.CalcBytesIncludingPadding(), GetStream());
            matmul_k8_a_len_ = temp_shape_left.CalcBytesExcludingPadding();
        }
        cudaMemcpy2DAsync(matmul_k8_a_buffer_, ALIGN_UP(k, 8) * GetSizeOfDataType(shape_left.GetDataType()),
                          left_X->GetBufferPtr(),
                          k * GetSizeOfDataType(shape_left.GetDataType()),
                          k * GetSizeOfDataType(shape_left.GetDataType()),
                          shape_left.CalcBytesIncludingPadding() / (k * GetSizeOfDataType(shape_left.GetDataType())),
                          cudaMemcpyDeviceToDevice, GetStream());
        if (!param_->extra_param.is_initializer_weight){
            auto temp_shape_right = shape_right;
            if(shape_right.GetDimCount() == 2){
                auto k = temp_shape_right.GetDim(0);
                temp_shape_right.SetDim(0, ALIGN_UP(k, 8));
                if(temp_shape_right.CalcBytesExcludingPadding() != matmul_k8_b_len_){
                    if(matmul_k8_b_buffer_ != nullptr){
                        cudaFree(matmul_k8_b_buffer_);
                        matmul_k8_b_buffer_ = nullptr;
                    }
                    cudaMalloc(&matmul_k8_b_buffer_, temp_shape_right.CalcBytesIncludingPadding());
                    if(matmul_k8_b_buffer_ == nullptr)
                        return RC_OUT_OF_MEMORY;
                    cudaMemsetAsync(matmul_k8_b_buffer_, 0, temp_shape_right.CalcBytesIncludingPadding(), GetStream());
                    matmul_k8_b_len_ = temp_shape_right.CalcBytesExcludingPadding();
                }
                cudaMemcpyAsync(matmul_k8_b_buffer_, right_X->GetBufferPtr(), shape_right.CalcBytesIncludingPadding(), cudaMemcpyDeviceToDevice, GetStream());
            } else{
                auto k = temp_shape_right.GetDim(temp_shape_right.GetDimCount() - 2);
                temp_shape_right.SetDim(temp_shape_right.GetDimCount() - 2, ALIGN_UP(k, 8));
                if(temp_shape_right.CalcBytesExcludingPadding() != matmul_k8_b_len_){
                    if(matmul_k8_b_buffer_ != nullptr){
                        cudaFree(matmul_k8_b_buffer_);
                        matmul_k8_b_buffer_ = nullptr;
                    }
                    cudaMalloc(&matmul_k8_b_buffer_, temp_shape_right.CalcBytesIncludingPadding());
                    if(matmul_k8_b_buffer_ == nullptr)
                        return RC_OUT_OF_MEMORY;
                    cudaMemsetAsync(matmul_k8_b_buffer_, 0, temp_shape_right.CalcBytesIncludingPadding(), GetStream());
                    matmul_k8_b_len_ = temp_shape_right.CalcBytesExcludingPadding();
                }
                cudaMemcpy2DAsync(matmul_k8_b_buffer_,
                                  temp_shape_right.CalcBytesFromDimesionIncludingPadding(temp_shape_right.GetDimCount() - 2),
                                  right_X->GetBufferPtr(),
                                  shape_right.CalcBytesFromDimesionIncludingPadding(shape_right.GetDimCount() - 2),
                                  shape_right.CalcBytesFromDimesionIncludingPadding(shape_right.GetDimCount() - 2),
                                  shape_right.CalcElementsToDimensionExcludingPadding(shape_right.GetDimCount() - 2),
                                  cudaMemcpyDeviceToDevice,
                                  GetStream());
            }
        }
    }
#endif //USE_MATMUL_K8_ALIGN

    bool transa = false;
    bool transb = false;
    bool trans_batch_a = false;
    bool trans_batch_b = false;
    MatMulComputeHelper helper;
    auto temp_shape_left = shape_left;
    temp_shape_left.SetDim(temp_shape_left.GetDimCount() - 1, ALIGN_UP(temp_shape_left.GetDim(temp_shape_left.GetDimCount() - 1), 8));
    auto temp_shape_right = shape_right;
    temp_shape_right.SetDim(temp_shape_right.GetDimCount() - 2, ALIGN_UP(temp_shape_right.GetDim(temp_shape_left.GetDimCount() - 2), 8));
    auto status = helper.Compute(is_need_k8_align ? temp_shape_left : shape_left,
                                 is_need_k8_align ? temp_shape_right : shape_right,
                                 transa, transb, trans_batch_a, trans_batch_b, false);

    auto Y = ctx->GetOutput<TensorImpl>(0);

    cublasOperation_t transA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    float alpha = param_->param.alpha; // or half
    float beta = 0.0; // or half

    int lda = helper.Lda(transa);
    int ldb = helper.Ldb(transb);
    int ldc = helper.Ldc();

    int64_t stride_A, stride_B, stride_C, batch_count;
    if (helper.OutputOffsets().size() == 1) {
        if (shape_left.GetDataType() == ppl::common::DATATYPE_INT8 && shape_right.GetDataType() == ppl::common::DATATYPE_INT8) {
        #ifdef PPLNN_USE_MACA
            auto shape_bias_broadcast = *Y->GetShape();
            shape_bias_broadcast.SetDataType(ppl::common::DATATYPE_FLOAT32);
            if(gemm_bias_len_ != shape_bias_broadcast.CalcBytesIncludingPadding()){
                if(gemm_bias_buffer_ != nullptr){
                    cudaFree(gemm_bias_buffer_);
                    gemm_bias_buffer_ = nullptr;
                }
                cudaMalloc(&gemm_bias_buffer_, shape_bias_broadcast.CalcBytesIncludingPadding());
                if(gemm_bias_buffer_ == nullptr)
                    return RC_OUT_OF_MEMORY;
                cudaMemsetAsync(gemm_bias_buffer_, 0, shape_bias_broadcast.CalcBytesIncludingPadding(), GetStream());
                gemm_bias_len_ = shape_bias_broadcast.CalcBytesIncludingPadding();
            }

            auto alpha_X = ctx->GetInput<TensorImpl>(2);
            auto alpha_shape = alpha_X->GetShape();
            int quantMode = (alpha_shape->GetDim(0) > 1) ? 1 : 0;
            cublasSetPointerMode(GetCublasHandle(), CUBLAS_POINTER_MODE_DEVICE);
            mcblasStatus_t cublasStatus = mcblasGemmExQuantInt8(GetCublasHandle(),
                                                    transB,
                                                    transA,
                                                    helper.N(),
                                                    helper.M(),
                                                    helper.K(),
                                                    alpha_X->GetBufferPtr(),
                                                    right_X->GetBufferPtr(), CUDA_R_8I, ldb,
                                                    left_X->GetBufferPtr(), CUDA_R_8I, lda,
                                                    Y->GetBufferPtr(), CUDA_R_8I, ldc,
                                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT,
                                                    quantMode, gemm_bias_buffer_);
            cublasSetPointerMode(GetCublasHandle(), CUBLAS_POINTER_MODE_HOST);
        #endif // end ifdef PPLNN_USE_MACA
        } else {
        #if 0
            bool canUseCublassApi = param_->extra_param.fuse_info.types.size() == 0 || param_->extra_param.fuse_info.types.size() == 1;
            if(canUseCublassApi){
                if(param_->extra_param.fuse_info.types.size() == 1){
                    auto add_input1 = ctx->GetInput<TensorImpl>(param_->extra_param.fuse_info.input_inds[0]);
                    auto shape_add_input1 = *add_input1->GetShape();
                    auto shape_out0 = *Y->GetShape();
                    if(shape_out0.CalcElementsIncludingPadding() % shape_add_input1.CalcElementsIncludingPadding() != 0){
                        LOG(ERROR) << "add_input1 size: " << shape_add_input1.CalcElementsIncludingPadding() << " not adjust to add_input 0: "
                                   << shape_out0.CalcElementsIncludingPadding();
                        return ppl::common::RC_UNSUPPORTED;
                    }
                    int m = shape_out0.CalcElementsIncludingPadding() / shape_add_input1.CalcElementsIncludingPadding();
                    int n = shape_add_input1.CalcElementsIncludingPadding();
                    auto tmp_shape = TensorShape();
                    tmp_shape.SetDimCount(1);
                    tmp_shape.SetDim(0, m);
                    tmp_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                    tmp_shape.SetDataType(shape_left.GetDataType());

                    BufferDesc gemm_tmp_;
                    status = GetCudaDevice()->Realloc(tmp_shape, &gemm_tmp_);
                    if (status != ppl::common::RC_SUCCESS) {
                        LOG(ERROR) << "alloc buffer for gemm bias space failed: " << GetRetCodeStr(status);
                        return status;
                    }
                    Destructor tmp_buf_guard__([this, &gemm_tmp_]() -> void {
                        GetCudaDevice()->Free(&gemm_tmp_);
                    });

                    std::vector<float> ones(m, 1.0f);
                    auto tmp_shape_float = tmp_shape;
                    tmp_shape_float.SetDataType(DATATYPE_FLOAT32);
                    status = GetCudaDevice()->GetDataConverter()->ConvertFromHost(&gemm_tmp_, tmp_shape,
                                                                                ones.data(), tmp_shape_float);
                    if (status != ppl::common::RC_SUCCESS) {
                        LOG(ERROR) << "copy data failed: " << ppl::common::GetRetCodeStr(status);
                        return status;
                    }
                    cublasStatus_t cublasStatus = cublasGemmEx(GetCublasHandle(),
                                                            (cublasOperation_t)0,
                                                            (cublasOperation_t)0,
                                                            n,
                                                            m,
                                                            1,
                                                            &alpha,
                                                            add_input1->GetBufferPtr(),
                                                            isFp16 ? CUDA_R_16F : CUDA_R_32F,
                                                            n,
                                                            gemm_tmp_.addr,
                                                            isFp16 ? CUDA_R_16F : CUDA_R_32F,
                                                            1,
                                                            &beta,
                                                            Y->GetBufferPtr(),
                                                            isFp16 ? CUDA_R_16F : CUDA_R_32F,
                                                            n,
                                                            CUDA_R_32F,
                                                            CUBLAS_GEMM_DEFAULT);
                    CHECK_CUBLAS_STATUS(cublasGemmEx, cublasStatus);
                    beta = 1.0f;
                }
                cublasStatus_t cublasStatus = cublasGemmEx(GetCublasHandle(),
                                                    transB,
                                                    transA,
                                                    helper.N(),
                                                    helper.M(),
                                                    helper.K(),
                                                    &alpha,
                                                    (is_need_k8_align && !param_->extra_param.is_initializer_weight) ? matmul_k8_b_buffer_ : right_X->GetBufferPtr(),
                                                    isFp16 ? CUDA_R_16F : CUDA_R_32F, ldb,
                                                    is_need_k8_align ? matmul_k8_a_buffer_ : left_X->GetBufferPtr(),
                                                    isFp16 ? CUDA_R_16F : CUDA_R_32F, lda,
                                                    &beta,
                                                    Y->GetBufferPtr(), isFp16 ? CUDA_R_16F : CUDA_R_32F, ldc,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);
            }
        #else
            cublasOperation_t transa = CUBLAS_OP_N;
            cublasOperation_t transb = CUBLAS_OP_N;
            int m = shape_right.GetDim(shape_right.GetDimCount() - 1);//shape_right.GetDim(1);
            int n = shape_left.CalcElementsToDimensionExcludingPadding(shape_left.GetDimCount() - 1);
            int k = shape_right.CalcElementsToDimensionExcludingPadding(shape_right.GetDimCount() - 1);//shape_right.GetDim(0);
            k = is_need_k8_align ? ALIGN_UP(k, 8) : k;
            int lda = m;
            int ldb = k;
            int ldc = m;
            int ldbias = ctx->GetInputCount() > 2 ? ldc : 1;
            int32_t bias_mode = 0;
            if(ctx->GetInputCount() > 2){
                auto bias = ctx->GetInput<TensorImpl>(2);
                if(bias->GetShape()->GetDimCount() == 2){
                    if(bias->GetShape()->GetDim(0) == 1)
                        bias_mode = 2;
                    else
                        bias_mode = 3;
                }
                if(bias->GetShape()->GetDimCount() == 1){
                    if(bias->GetShape()->GetDim(0) == 1)
                        bias_mode = 1;
                    else{
                        bias_mode = 2;
                    }
                }
            }
            if(((bias_mode == 2 || bias_mode == 1) && gemm_bias_len_ != m)
                || (bias_mode == 3 && gemm_bias_len_ != m * n)){
                auto bias = ctx->GetInput<TensorImpl>(2);
                auto bias_tmp_shape = TensorShape();
                if(bias_mode == 2 || bias_mode == 1){
                    bias_tmp_shape.SetDimCount(1);
                    bias_tmp_shape.SetDim(0, m);
                    bias_tmp_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                    bias_tmp_shape.SetDataType(DATATYPE_FLOAT16);
                    if(gemm_bias_buffer_ != nullptr){
                        cudaFree(gemm_bias_buffer_);
                        gemm_bias_buffer_ = nullptr;
                    }
                    cudaMalloc(&gemm_bias_buffer_, bias_tmp_shape.CalcBytesIncludingPadding());
                    gemm_bias_len_ = m;
                }else{
                    bias_tmp_shape = *bias->GetShape();
                    if(gemm_bias_buffer_ != nullptr){
                        cudaFree(gemm_bias_buffer_);
                        gemm_bias_buffer_ = nullptr;
                    }
                    cudaMalloc(&gemm_bias_buffer_, bias_tmp_shape.CalcBytesIncludingPadding());
                    gemm_bias_len_ = m * n;
                }
                if(gemm_bias_buffer_ == nullptr)
                    return RC_OUT_OF_MEMORY;
                cudaMemsetAsync(gemm_bias_buffer_, 0, bias_tmp_shape.CalcBytesIncludingPadding(), GetStream());
                cudaMemcpyAsync(gemm_bias_buffer_, ctx->GetInput<TensorImpl>(2)->GetBufferPtr(), bias->GetShape()->CalcBytesIncludingPadding(), cudaMemcpyDeviceToDevice, GetStream());
                GemmKernelParam param_kernel;
                param_kernel.alpha = 1.0f;
                param_kernel.beta  = param_->param.beta;
                param_kernel.transA = 0;
                param_kernel.transB = 0;
                PPLCUDAGemmModifyBias( GetStream(), shape_left.GetDataType(), &bias_tmp_shape, gemm_bias_buffer_, &param_kernel);
            }
            mcblasLtHandle_t mcblasLtHandle = GetMcblasLtHandle();
            if(mcblasLtHandle == nullptr){
                LOG(ERROR) <<"DoExecute_BLAS: mcblasLtHandle == nullptr";
                return ppl::common::RC_NOT_FOUND;
            }

            //printf("run param: m:%d, n:%d, k:%d, lda:%d, ldb:%d, ldc:%d, bias_mode: %d\n", m, n, k, lda, ldb, ldc, bias_mode);
            void* bias_data_ptr = nullptr;
            if(bias_mode != 0)
                bias_data_ptr =  gemm_bias_buffer_;
            void* add_data_ptr = nullptr;
            int32_t act_type = 0;
            if(param_->extra_param.fuse_info.types.size() > 0){
                if(param_->extra_param.fuse_info.types[param_->extra_param.fuse_info.types.size() - 1] == "Relu"){
                    act_type = 1;
                } else if(param_->extra_param.fuse_info.types[param_->extra_param.fuse_info.types.size() - 1] == "Gelu"){
                    act_type = 2;
                } else if(param_->extra_param.fuse_info.types[param_->extra_param.fuse_info.types.size() - 1] == "HardSwish"){
                    act_type = 3;
                } else if(param_->extra_param.fuse_info.types.size() == 2
                          && param_->extra_param.fuse_info.types[param_->extra_param.fuse_info.types.size() - 1] == "Add"){
                    add_data_ptr = ctx->GetInput<TensorImpl>(ctx->GetInputCount() - 1)->GetBufferPtr();
                }
            }

            if(!ismcblasLtInit_ || m_ != m || n_ != n || k_ != k){
                cublasStatus_t status = mcblasLtMatmulDescCreate(&mcblasLtOperationDesc_, CUBLAS_COMPUTE_32F, CUDA_R_32F);
                CHECK_CUBLAS_STATUS(mcblasLtMatmulDescCreate, status);
                status = mcblasLtMatrixLayoutCreate(&Adesc_, isFp16 ? CUDA_R_16F : CUDA_R_32F, m, k, lda);
                CHECK_CUBLAS_STATUS(mcblasLtMatrixLayoutCreate, status);
                status = mcblasLtMatrixLayoutCreate(&Bdesc_, isFp16 ? CUDA_R_16F : CUDA_R_32F, k, n, ldb);
                CHECK_CUBLAS_STATUS(mcblasLtMatrixLayoutCreate, status);
                status = mcblasLtMatrixLayoutCreate(&Cdesc_, isFp16 ? CUDA_R_16F : CUDA_R_32F, m, n, ldc);
                CHECK_CUBLAS_STATUS(mcblasLtMatrixLayoutCreate, status);
                ismcblasLtInit_ = true;
                m_ = m;
                n_ = n;
                k_ = k;
                auto beta_tmp_shape = TensorShape();
                beta_tmp_shape.SetDimCount(1);
                beta_tmp_shape.SetDim(0, 1);
                beta_tmp_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                beta_tmp_shape.SetDataType(DATATYPE_FLOAT32);
                if(gemm_beta_buffer_ != nullptr){
                    cudaFree(gemm_beta_buffer_);
                    gemm_beta_buffer_ = nullptr;
                }
                if(!(bias_mode==2 && act_type !=3 && transa == MCBLAS_OP_N && transb == MCBLAS_OP_N)){
                    cudaMalloc(&gemm_beta_buffer_, beta_tmp_shape.CalcBytesIncludingPadding());
                    if(gemm_beta_buffer_ == nullptr)
                        return RC_OUT_OF_MEMORY;
                    float beta = 1.0f;
                    cudaMemcpyAsync(gemm_beta_buffer_, &beta, sizeof(beta), cudaMemcpyHostToDevice, GetStream());
                    gemm_beta_len_ = sizeof(float);
                }
            }

            //printf("######################## use mcblasGemmExBiasActFp16  api ################ act_type %d\n", act_type);
            mcblasGemmExBiasActFp16(mcblasLtHandle,
                                    mcblasLtOperationDesc_,
                                    Adesc_,
                                    Bdesc_,
                                    Cdesc_,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    param_->param.alpha,
                                    (is_need_k8_align && !param_->extra_param.is_initializer_weight) ? matmul_k8_b_buffer_ : right_X->GetBufferPtr(),
                                    lda,
                                    is_need_k8_align ? matmul_k8_a_buffer_ : left_X->GetBufferPtr(),
                                    ldb,
                                    Y->GetBufferPtr(),
                                    ldc,
                                    (float*)gemm_beta_buffer_,
                                    bias_data_ptr,
                                    ldbias,
                                    bias_mode,
                                    act_type,
                                    GetStream(),
                                    add_data_ptr);
        #endif
        }
        return status;
        // CHECK_CUBLAS_STATUS(cublasGemmEx, cublasStatus);
    } else if (CanUseStridedBatchedGemm(is_need_k8_align ? temp_shape_left : shape_left,
                                        is_need_k8_align ? temp_shape_right : shape_right,
                                        transa, transb, trans_batch_a, trans_batch_b,
                                        stride_A, stride_B, stride_C, batch_count)) {
        if (shape_left.GetDataType() == ppl::common::DATATYPE_INT8 && shape_right.GetDataType() == ppl::common::DATATYPE_INT8) {
            return ppl::common::RC_UNSUPPORTED;
        } else {
            cublasStatus_t cublasStatus = cublasGemmStridedBatchedEx(GetCublasHandle(),
                                                transB,
                                                transA,
                                                helper.N(),
                                                helper.M(),
                                                helper.K(),
                                                &alpha,
                                                (is_need_k8_align && !param_->extra_param.is_initializer_weight) ? matmul_k8_b_buffer_ : right_X->GetBufferPtr(),
                                                isFp16 ? CUDA_R_16F : CUDA_R_32F, ldb, stride_B,
                                                is_need_k8_align ? matmul_k8_a_buffer_ : left_X->GetBufferPtr(),
                                                isFp16 ? CUDA_R_16F : CUDA_R_32F, lda, stride_A,
                                                &beta,
                                                Y->GetBufferPtr(), isFp16 ? CUDA_R_16F : CUDA_R_32F, ldc, stride_C,
                                                batch_count,
                                                CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT);
        }
        return status;
    }

    helper.FillOffsets();
    if ((shape_left.GetDataType() == ppl::common::DATATYPE_FLOAT16 && shape_right.GetDataType() == ppl::common::DATATYPE_FLOAT16)
        || (shape_left.GetDataType() == ppl::common::DATATYPE_FLOAT32 && shape_right.GetDataType() == ppl::common::DATATYPE_FLOAT32)) {
        if (!left_arrays) cudaMallocHost(&left_arrays, helper.LeftOffsets().size() * sizeof(void *));
        if (!right_arrays) cudaMallocHost(&right_arrays, helper.RightOffsets().size() * sizeof(void *));
        if (!output_arrays) cudaMallocHost(&output_arrays, helper.OutputOffsets().size()* sizeof(void *));

        // half *left_arrays[helper.LeftOffsets().size()];
        // half *right_arrays[helper.RightOffsets().size()];
        // half *output_arrays[helper.OutputOffsets().size()];

        for (size_t i = 0; i < helper.LeftOffsets().size(); i++) {
            left_arrays[i] = (void*)(is_need_k8_align ? matmul_k8_a_buffer_ : left_X->GetBufferPtr()) + helper.LeftOffsets()[i];
        }
        for (size_t i = 0; i < helper.RightOffsets().size(); i++) {
            right_arrays[i] = (void*)((is_need_k8_align && !param_->extra_param.is_initializer_weight) ? matmul_k8_b_buffer_ : right_X->GetBufferPtr()) + helper.RightOffsets()[i];
        }
        for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
            output_arrays[i] = (void*)Y->GetBufferPtr() + helper.OutputOffsets()[i];
        }

        // half **left_gpu, **right_gpu, **output_gpu;
        // cudaMalloc(&left_gpu, helper.LeftOffsets().size() * sizeof(half *));
        // cudaMalloc(&right_gpu, helper.RightOffsets().size()* sizeof(half *));
        // cudaMalloc(&output_gpu, helper.OutputOffsets().size()* sizeof(half *));

        // cudaMemcpyAsync(left_gpu, left_arrays, helper.LeftOffsets().size() * sizeof(half *), cudaMemcpyHostToDevice, GetStream());
        // cudaMemcpyAsync(right_gpu, right_arrays, helper.RightOffsets().size() * sizeof(half *), cudaMemcpyHostToDevice, GetStream());
        // cudaMemcpyAsync(output_gpu, output_arrays, helper.OutputOffsets().size() * sizeof(half *), cudaMemcpyHostToDevice, GetStream());
        cublasStatus_t cublasStatus = cublasGemmBatchedEx(GetCublasHandle(),
                                    transB,
                                    transA,
                                    helper.N(),
                                    helper.M(),
                                    helper.K(),
                                    &alpha,
                                    (const void**)right_arrays, isFp16 ? CUDA_R_16F : CUDA_R_32F, ldb,
                                    (const void**)left_arrays, isFp16 ? CUDA_R_16F : CUDA_R_32F, lda,
                                    &beta,
                                    (void**)output_arrays, isFp16 ? CUDA_R_16F : CUDA_R_32F, ldc,
                                    helper.OutputOffsets().size(),
                                    CUBLAS_COMPUTE_32F,
                                    CUBLAS_GEMM_DEFAULT);

        return status;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
#else //PPLNN_USE_DNN
    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_bytes = CalcTmpBufferSize(*ctx);
    auto status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_bytes << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;

    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);
    GemmKernelParam param_kernel_;
    param_kernel_.alpha = param_->param.alpha;
    param_kernel_.beta = param_->param.beta;
    param_kernel_.transA = param_->param.transA;
    param_kernel_.transB = param_->param.transB;

    // convert filter only if the filter tensor is an output of another kernel
    BufferDesc weight_buffer;
    auto newshape = *weight->GetShape();
    {
        auto align_size = 8;
        auto dim_count = newshape.GetDimCount();
        newshape.SetDim(dim_count - 2, (newshape.GetDim(dim_count - 2) + align_size - 1) / align_size * align_size);

        auto status = GetCudaDevice()->Realloc(newshape, &weight_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        PPLCUDABgemmModifyWeights(stream, weight->GetShape(), weight->GetBufferPtr(), weight_buffer.addr,
                                  &param_kernel_);
    }
    ppl::common::Destructor __tmp_buffer_guard__([this, &weight_buffer]() -> void {
        GetCudaDevice()->Free(&weight_buffer);
    });

    BufferDesc input0_buffer;
    auto newshape0 = *input0->GetShape();
    auto dim_count = newshape0.GetDimCount();
    auto K = newshape0.GetDim(dim_count - 1);
    auto align_size = 8;
    auto K_pad = (K + align_size - 1) / align_size * align_size;
    bool is_input0_pad = K != K_pad;
    void* bmm_input0;
    if (is_input0_pad) {
        newshape0.SetDim(dim_count - 1, K_pad);
        auto status = GetCudaDevice()->Realloc(newshape0, &input0_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        PPLCUDABgemmPadInput(stream, input0->GetShape(), input0->GetBufferPtr(), input0_buffer.addr, &param_kernel_);
        bmm_input0 = input0_buffer.addr;
    } else {
        bmm_input0 = input0->GetBufferPtr();
    }
    ppl::common::Destructor __input0_buffer_guard__([this, &input0_buffer]() -> void {
        GetCudaDevice()->Free(&input0_buffer);
    });

    auto newshape_out = *output->GetShape();
    auto out_dim_count = newshape_out.GetDimCount();
    auto N = newshape_out.GetDim(out_dim_count - 1);
    auto N_pad = (N + align_size - 1) / align_size * align_size;
    BufferDesc output_buffer;
    bool is_output_pad = N != N_pad;
    void* bgemm_out;
    if (is_output_pad) {
        newshape_out.SetDim(out_dim_count - 1, N_pad);
        auto status = GetCudaDevice()->Realloc(newshape_out, &output_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        bgemm_out = output_buffer.addr;
    } else {
        bgemm_out = output->GetBufferPtr();
    }
    ppl::common::Destructor __output_buffer_guard__([this, &output_buffer]() -> void {
        GetCudaDevice()->Free(&output_buffer);
    });

    fuse_param_t temp_fuse_param;
    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);

    auto stream = GetStream();
    CUfunction module_func = nullptr;
#ifdef PPLNN_ENABLE_CUDA_JIT
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);
    module_func = module->GetKernelFunc();
#endif

    const TensorShape& shape_in0 = *input0->GetShape();

    if (shape_in0.GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        status = PPLCUDABgemmForwardImp(GetCudaDevice()->GetDeviceProp(), stream, module_func, input0->GetShape(), bmm_input0,
                                        weight->GetShape(), weight_buffer.addr, output->GetShape(), bgemm_out,
                                        param_kernel_, tmp_buffer, temp_fuse_param, param_->extra_param.algo_info);
    }

    if (is_output_pad) {
        PPLCUDABgemmCvtOutput(stream, output->GetShape(), output->GetBufferPtr(), bgemm_out);
    }

    return status;
#endif //PPLNN_USE_DNN
}

}}} // namespace ppl::nn::cuda
