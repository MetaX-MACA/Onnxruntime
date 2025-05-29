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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CONV_FMMA_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_CONV_FMMA_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"

#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)

namespace ppl { namespace nn { namespace cuda {

class ConvFmmaKernel : public CudaKernel {
public:
    ConvFmmaKernel(const ir::Node* node) : CudaKernel(node) {
        cudnnCreateFilterDescriptor(&w_desc_);
        cudnnCreateTensorDescriptor(&x_tensor_);
        cudnnCreateTensorDescriptor(&b_tensor_);
        cudnnCreateTensorDescriptor(&ele_input_tensor_);
        cudnnCreateTensorDescriptor(&y_tensor_);
        cudnnCreateActivationDescriptor(&activation_desc_);
        cudnnCreateActivationDescriptor(&activation_desc2_);
        cudnnCreateConvolutionDescriptor(&conv_desc_);
        conv_work_buffer_ = nullptr;
        conv_bias_buffer_ = nullptr;
    }

    ~ConvFmmaKernel(){
        cudnnDestroyFilterDescriptor(w_desc_);w_desc_ = nullptr;
        cudnnDestroyTensorDescriptor(x_tensor_);x_tensor_ = nullptr;
        cudnnDestroyTensorDescriptor(b_tensor_);b_tensor_ = nullptr;
        cudnnDestroyTensorDescriptor(ele_input_tensor_);ele_input_tensor_ = nullptr;
        cudnnDestroyTensorDescriptor(y_tensor_);y_tensor_ = nullptr;
        cudnnDestroyActivationDescriptor(activation_desc_);activation_desc_ = nullptr;
        cudnnDestroyActivationDescriptor(activation_desc2_);activation_desc2_ = nullptr;
        cudnnDestroyConvolutionDescriptor(conv_desc_);conv_desc_ = nullptr;
        if(conv_work_buffer_ != nullptr){
            cudaFree(conv_work_buffer_);
        }
        if(conv_bias_buffer_ != nullptr){
            cudaFree(conv_bias_buffer_);
        }
    }

    void SetParam(const CudaConvParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode BeforeExecute(KernelExecContext*) override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
private:
    const CudaConvParam* param_ = nullptr;
    cudnnFilterDescriptor_t w_desc_ = nullptr;
    cudnnTensorDescriptor_t x_tensor_ = nullptr;
    cudnnTensorDescriptor_t b_tensor_ = nullptr;
    cudnnTensorDescriptor_t ele_input_tensor_ = nullptr;
    cudnnTensorDescriptor_t y_tensor_ = nullptr;
    cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
    cudnnActivationDescriptor_t activation_desc_ = nullptr;
    cudnnActivationDescriptor_t activation_desc2_ = nullptr;
    void* conv_work_buffer_;
    uint32_t conv_work_len_ = 0;
    void* conv_bias_buffer_;
    uint32_t conv_bias_len_ = 0;
};

}}} // namespace ppl::nn::cuda

#endif //#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)

#endif
