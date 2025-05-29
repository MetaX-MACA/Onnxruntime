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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_GEMM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_GEMM_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"
#include "ppl/nn/engines/cuda/params/gemm_extra_param.h"
#include "ppl/nn/engines/cuda/kernels/onnx/matmul_helper.h"

namespace ppl { namespace nn { namespace cuda {

class GemmKernel : public CudaKernel {
public:
    GemmKernel(const ir::Node* node) : CudaKernel(node) {
        gemm_weight_buffer_ = nullptr;
        gemm_bias_buffer_ = nullptr;
        gemm_beta_buffer_ = nullptr;
    }

    void SetParam(const CudaGemmParam* p) {
        param_ = p;
    }
#ifdef PPLNN_USE_DNN
    ~GemmKernel(){
        if(gemm_weight_buffer_ != nullptr){
            cudaFree(gemm_weight_buffer_);
        }
        if(gemm_bias_buffer_ != nullptr){
            cudaFree(gemm_bias_buffer_);
        }
        if(gemm_beta_buffer_ != nullptr){
            cudaFree(gemm_beta_buffer_);
        }
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
private:
    ppl::common::RetCode DoExecute_PPL(KernelExecContext*);
#ifdef PPLNN_USE_DNN
    ppl::common::RetCode DoExecute_BLAS(KernelExecContext*);
#ifdef PPLNN_USE_MACA
    bool ismcblasLtInit_ = false;
    mcblasLtHandle_t mcblasLtHandle_ = nullptr;
    mcblasLtMatmulDesc_t mcblasLtOperationDesc_ = nullptr;
    mcblasLtMatrixLayout_t Adesc_ = nullptr, Bdesc_ = nullptr, Cdesc_ = nullptr;
    int m_ = 0, n_ = 0, k_ = 0;
#endif//PPLNN_USE_MACA
#endif
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool CanDoExecute(const KernelExecContext&) const override;

private:
    const CudaGemmParam* param_ = nullptr;
    void* gemm_weight_buffer_;
    uint32_t gemm_weight_len_ = 0;
    void* gemm_bias_buffer_;
    uint32_t gemm_bias_len_ = 0;
    void* gemm_beta_buffer_;
    uint32_t gemm_beta_len_ = 0;
};

}}} // namespace ppl::nn::cuda

#endif
