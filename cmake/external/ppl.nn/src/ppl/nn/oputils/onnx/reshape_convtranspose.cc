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

#include <cmath>
#include "ppl/nn/params/onnx/auto_pad_type.h"
#include "ppl/nn/oputils/onnx/reshape_convtranspose.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ReshapeConvTranspose(InputOutputInfo* info, ir::Attr* arg) {
    auto param = static_cast<ConvTransposeParam*>(arg);
    auto x = info->GetInput<TensorImpl>(0)->GetShape();
    auto w = info->GetInput<TensorImpl>(1)->GetShape();
    auto y = info->GetOutput<TensorImpl>(0)->GetShape();
    auto num_output = w->GetDim(1);

    y->SetDimCount(x->GetDimCount());
    y->SetDim(0, x->GetDim(0));
    y->SetDim(1, num_output);

    const int32_t kernel_dims = (int32_t)x->GetDimCount() - 2;
    if(param->dilations.size() == 0)
    {
        param->dilations.resize(kernel_dims, 1);
    }
    if(param->pads.size() == 0)
    {
        param->pads.resize(kernel_dims * 2, 0);
    }
    if(param->strides.size() == 0)
    {
        param->strides.resize(kernel_dims, 1);
    }
    if(param->output_padding.size() == 0)
    {
        param->output_padding.resize(kernel_dims, 0);
    }
    if(param->kernel_shape.size() == 0)
    {
        param->kernel_shape.resize(kernel_dims);
        for(int32_t i = 0; i < kernel_dims; i++)
        {
            param->kernel_shape[i] = w->GetDim(2 + i);
        }
    }
    
    for (int32_t i = 0; i < kernel_dims; ++i) {
        const int32_t j = i + 2;
        int64_t out_dim;
        if(param->output_shape.size() > 0)
        {
            out_dim = param->output_shape[i];
            int pad_temp = (x->GetDim(j) - 1) * param->strides[i] + param->output_padding[i] + ((param->kernel_shape[i] - 1) * param->dilations[i] + 1)  - out_dim;
            param->pads[i] = pad_temp / 2;
            param->pads[kernel_dims + i] = pad_temp - pad_temp / 2;
        }
        else if(AUTO_PAD_SAME_UPPER == param->auto_pad || AUTO_PAD_SAME_LOWER == param->auto_pad)
        {
            out_dim = x->GetDim(j) * param->strides[i];
            int pad_temp = (x->GetDim(j) - 1) * param->strides[i] + param->output_padding[i] + ((param->kernel_shape[i] - 1) * param->dilations[i] + 1)  - out_dim;
            param->pads[i] = AUTO_PAD_SAME_LOWER == param->auto_pad ? (pad_temp / 2 + pad_temp % 2) : (pad_temp / 2);
            param->pads[kernel_dims + i] = pad_temp - pad_temp / 2;
        }
        else
        {
            const int32_t kernel_shape_eff = (param->kernel_shape[i] - 1) * param->dilations[i] + 1;
            out_dim = param->strides[i] * (x->GetDim(j) - 1) + param->output_padding[i] + kernel_shape_eff - param->pads[i] - param->pads[i + kernel_dims];
            //out_dim = param->strides[i] * (x->GetDim(j) - 1) + kernel_shape_eff - param->pads[i] - param->pads[i + kernel_dims];
        }
        
        if (out_dim <= 0) {
            LOG(DEBUG) << "ERROR: output dim[" << out_dim << "] < 0.";
            return RC_INVALID_VALUE;
        }
        y->SetDim(j, out_dim);
    }
    y->CalcPadding();

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
