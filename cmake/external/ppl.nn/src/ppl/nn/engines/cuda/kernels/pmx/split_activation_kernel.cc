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

#include "ppl/nn/engines/cuda/kernels/pmx/split_activation_kernel.h"

#include "cudakernel/nn/split_activation.h"


namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode SplitActivationKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);

    void *bias_data;
    TensorShape *bias_shape;
    if (ctx->GetInputCount() == 2) {
        auto bias = ctx->GetInput<TensorImpl>(1);
        bias_data = bias->GetBufferPtr();
        bias_shape = bias->GetShape();
    } else {
        bias_data = nullptr;
        bias_shape = nullptr;
    }

    auto input_shape = input->GetShape();
    if (input_shape->GetDimCount() != 3) {
        LOG(ERROR) << "input is expected to have 3 dimensions, got " << input_shape->GetDimCount();
        return ppl::common::RC_INVALID_VALUE;
    }

    auto input_dims_2_val = input_shape->GetDim(2);
    if (input_dims_2_val != 2560 &&
      input_dims_2_val != 5120 &&
      input_dims_2_val != 6144 &&
      input_dims_2_val != 10240 &&
      input_dims_2_val != 12288) {
        LOG(ERROR) << "hidden size should be 2560, 5120, 6144, 10240 or 12288, got " << input_dims_2_val;
        return ppl::common::RC_INVALID_VALUE;

    }

    if (bias_data != nullptr) {
        if (bias_shape->GetDimCount() != 1) {
            LOG(ERROR) << "bias is expected to have 1 dimensions, got " << bias_shape->GetDimCount();
            return ppl::common::RC_INVALID_VALUE;
        }

        if (bias_shape->GetDim(0) != input_shape->GetDim(2)) {
            LOG(ERROR) << "last dimension of input and bias are not the same";
            return ppl::common::RC_INVALID_VALUE;
        }
    }


    auto output = ctx->GetOutput<TensorImpl>(0);
    std::string activation = "Gelu";
    int activation_type = 0;

    auto status = PPLCUDASplitActivationForwardImp(
                        GetStream(),
                        input->GetBufferPtr(),
                        input->GetShape(),
                        bias_data,
                        bias_shape,
                        activation_type,
                        output->GetBufferPtr());

    return status;
}

}}} // namespace ppl::nn::cuda
