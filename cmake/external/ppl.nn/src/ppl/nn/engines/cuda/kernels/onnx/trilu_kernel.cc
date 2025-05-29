
// 2024 - Created by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
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

#include "cudakernel/nn/trilu.h"
#include "ppl/nn/engines/cuda/kernels/onnx/trilu_kernel.h"

namespace ppl { namespace nn { namespace cuda {
ppl::common::RetCode TriluKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    // if(ctx->GetInputCount() > 1){
    //     auto input1 = ctx->GetInput<TensorImpl>(1);
    // }
    auto k = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);
    auto upper = param_->upper;

    // ppl::nn::onnx::TriluKernelParam param_kernel_;
    // param_kernel_.upper = param_->upper;
    // ppl::common::RetCode status = PPLCUDATriluForwardImp(
    //     GetStream(), param_kernel_, input->GetShape(), input->GetBufferPtr(), output->GetShape(), output->GetBufferPtr());
    ppl::common::RetCode status = PPLCUDATriluForwardImp(
        GetStream(), upper, k->GetBufferPtr(), input0->GetShape(), input0->GetBufferPtr(), output->GetShape(), output->GetBufferPtr());
    
    return 0;
}
}}} // namespace ppl::nn::cuda
