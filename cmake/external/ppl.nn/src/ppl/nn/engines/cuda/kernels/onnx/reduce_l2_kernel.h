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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_REDUCE_L2_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_REDUCE_L2_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/reduce_param.h"

namespace ppl { namespace nn { namespace cuda {

class ReduceL2Kernel : public CudaKernel {
public:
    ReduceL2Kernel(const ir::Node* node) : CudaKernel(node) {
        tmp_buffer_desc.addr = nullptr;
        tmp_buffer_out_desc.addr = nullptr;
    }

    void SetParam(const ppl::nn::onnx::ReduceParam* p) {
        param_ = p;
    }
    ~ReduceL2Kernel(){
        if(tmp_buffer_desc.addr != nullptr){
            GetCudaDevice()->Free(&tmp_buffer_desc);
        }
        if(tmp_buffer_out_desc.addr != nullptr){
            GetCudaDevice()->Free(&tmp_buffer_out_desc);
        }
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::onnx::ReduceParam* param_ = nullptr;

    BufferDesc tmp_buffer_desc;
    uint32_t tmp_buffer_len_ = 0;
    BufferDesc tmp_buffer_out_desc;
    uint32_t tmp_buffer_out_len_ = 0;
};

}}} // namespace ppl::nn::cuda

#endif
