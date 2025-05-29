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

#include "ppl/nn/oputils/onnx/reshape_deformconv.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ReshapeDeformConv(InputOutputInfo* info, const ir::Attr* arg) {
    auto param = (const DeformConvParam*)arg;
    auto input = info->GetInput<TensorImpl>(0)->GetShape();
    auto weight = info->GetInput<TensorImpl>(1)->GetShape();
    auto output = info->GetOutput<TensorImpl>(0)->GetShape();
    auto num_output = weight->GetDim(0);

    output->SetDimCount(input->GetDimCount());
    output->SetDim(0, input->GetDim(0));
    output->SetDim(1, num_output);

    const int64_t kernel_dims = 2;
    for (int64_t i = 0; i < kernel_dims; ++i) {
        const int64_t j = i + 2;
        const int64_t kernel_shape_eff = (weight->GetDim(j) - 1) * param->dilations[i] + 1;
        const int64_t out_dim = (input->GetDim(j) + param->pads[i] * 2 - kernel_shape_eff) / param->strides[i] + 1;
        if (out_dim <= 0) {
            return RC_INVALID_VALUE;
        }
        output->SetDim(j, out_dim);
    }
    output->CalcPadding();

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::mmcv
