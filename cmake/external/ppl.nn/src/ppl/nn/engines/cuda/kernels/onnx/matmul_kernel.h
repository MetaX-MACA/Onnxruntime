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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_BGEMM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_BGEMM_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"
#include "ppl/nn/engines/cuda/params/gemm_extra_param.h"
#include "ppl/nn/engines/cuda/kernels/onnx/matmul_helper.h"
#include <cublas_api.h>

namespace ppl { namespace nn { namespace cuda {

class MatMulKernel : public CudaKernel {
public:
    MatMulKernel(const ir::Node* node) : CudaKernel(node) {
    #ifdef PPLNN_USE_DNN
        gemm_bias_buffer_   = nullptr;
        gemm_alpha_buffer_  = nullptr;
        gemm_beta_buffer_   = nullptr;
        matmul_k8_a_buffer_ = nullptr;
        matmul_k8_b_buffer_ = nullptr;
    #endif
    }
#ifdef PPLNN_USE_DNN
    ~MatMulKernel(){
        if(gemm_bias_buffer_ != nullptr){
            cudaFree(gemm_bias_buffer_);
        }
        if(gemm_alpha_buffer_ != nullptr){
            cudaFree(gemm_alpha_buffer_);
        }
        if(gemm_beta_buffer_ != nullptr){
            cudaFree(gemm_beta_buffer_);
        }
        if(matmul_k8_a_buffer_ != nullptr){
            cudaFree(matmul_k8_a_buffer_);
        }
        if(matmul_k8_b_buffer_ != nullptr){
            cudaFree(matmul_k8_b_buffer_);
        }
        if (left_arrays) cudaFreeHost(left_arrays);
        if (right_arrays) cudaFreeHost(right_arrays);
        if (output_arrays) cudaFreeHost(output_arrays);
    #ifdef PPLNN_USE_MACA
        if(ismcblasLtInit_){
            mcblasLtMatrixLayoutDestroy(Cdesc_);
            mcblasLtMatrixLayoutDestroy(Bdesc_);
            mcblasLtMatrixLayoutDestroy(Adesc_);
            mcblasLtMatmulDescDestroy(mcblasLtOperationDesc_);
        }
    #endif
    }
#endif
    void SetParam(const CudaGemmParam* p) {
        param_ = p;
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext&) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool CanDoExecute(const KernelExecContext&) const override;

private:
    const CudaGemmParam* param_ = nullptr;
#ifdef PPLNN_USE_DNN
    void* gemm_bias_buffer_;
    uint32_t gemm_bias_len_ = 0;
    void* gemm_alpha_buffer_;
    uint32_t gemm_alpha_len_ = 0;
    void* gemm_beta_buffer_;
    uint32_t gemm_beta_len_ = 0;
    void **left_arrays = nullptr;
    void **right_arrays = nullptr;
    void **output_arrays = nullptr;
    void* matmul_k8_a_buffer_;
    uint32_t matmul_k8_a_len_ = 0;
    void* matmul_k8_b_buffer_;
    uint32_t matmul_k8_b_len_ = 0;
#ifdef PPLNN_USE_MACA
    bool ismcblasLtInit_ = false;
    mcblasLtHandle_t mcblasLtHandle_ = nullptr;
    mcblasLtMatmulDesc_t mcblasLtOperationDesc_ = nullptr;
    mcblasLtMatrixLayout_t Adesc_ = nullptr, Bdesc_ = nullptr, Cdesc_ = nullptr;
    int m_ = 0, n_ = 0, k_ = 0;
#endif//PPLNN_USE_MACA
#endif
};

}}} // namespace ppl::nn::cuda

#endif
