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

#include "ppl/nn/engines/cuda/kernels/onnx/random_kernel.h"
#include "cudakernel/nn/random.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode RandomUniformKernel::DoExecute(KernelExecContext* ctx) {
    auto output = ctx->GetOutput<TensorImpl>(0);
    float seed;
    if(param_->seed.empty()){
        seed = -65535;
    } else {
        seed = param_->seed[0];
    }
    return PPLCUDARandomUniformForwardImp(GetStream(), param_->low, param_->high, seed, output->GetShape(), output->GetBufferPtr());
}

ppl::common::RetCode RandomUniformLikeKernel::DoExecute(KernelExecContext* ctx) {
    auto output = ctx->GetOutput<TensorImpl>(0);
    float seed;
    if(param_->seed.empty()){
        seed = -65535;
    } else {
        seed = param_->seed[0];
    }
    
    return PPLCUDARandomUniformLikeForwardImp(GetStream(), param_->low, param_->high, seed, output->GetShape(), output->GetBufferPtr());
}

ppl::common::RetCode RandomNormalKernel::DoExecute(KernelExecContext* ctx) {
    auto output = ctx->GetOutput<TensorImpl>(0);
    float seed;
    if(param_->seed.empty()){
        seed = -65535;
    } else {
        seed = param_->seed[0];
    }
    return PPLCUDARandomNormalForwardImp(GetStream(), param_->mean, param_->scale, seed, output->GetShape(), output->GetBufferPtr());
}

ppl::common::RetCode RandomNormalLikeKernel::DoExecute(KernelExecContext* ctx) {
    auto output = ctx->GetOutput<TensorImpl>(0);
    float seed;
    if(param_->seed.empty()){
        seed = -65535;
    } else {
        seed = param_->seed[0];
    }
    
    return PPLCUDARandomNormalLikeForwardImp(GetStream(), param_->mean, param_->scale, seed, output->GetShape(), output->GetBufferPtr());
}

}}} // namespace ppl::nn::cuda
