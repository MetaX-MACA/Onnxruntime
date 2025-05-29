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

#include "ppl/nn/engines/cuda/kernels/onnx/resize_kernel.h"

#include "cudakernel/nn/resize.h"

namespace ppl { namespace nn { namespace cuda {

bool ResizeKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto& X = *ctx.GetInput<TensorImpl>(0)->GetShape();
    if (X.CalcBytesIncludingPadding() == 0) {
        return false;
    }
    if(ctx.GetInput<TensorImpl>(2)){
        auto& scales = *ctx.GetInput<TensorImpl>(2)->GetShape();
        if (ctx.GetInputCount() == 3 && scales.CalcBytesIncludingPadding() == 0) {
            return false;
        }
        if(ctx.GetInputCount() >= 4){
            auto& sizes = *ctx.GetInput<TensorImpl>(3)->GetShape();
            if (scales.CalcBytesIncludingPadding() == 0 && sizes.CalcBytesIncludingPadding() == 0) {
                return false;
            }
        }
    }else{
        if (ctx.GetInputCount() >= 4) {
            auto& sizes = *ctx.GetInput<TensorImpl>(3)->GetShape();
            if(sizes.CalcBytesIncludingPadding() == 0){
                return false;
            }
        }
    }
    return true;
}

ppl::common::RetCode ResizeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    // same size, just transfer
    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        input->GetShape()->GetDimCount() == 4 && input->GetShape()->GetDim(2) == output->GetShape()->GetDim(2) &&
        input->GetShape()->GetDim(3) == output->GetShape()->GetDim(3)) {
        output->TransferBufferFrom(input);
        return ppl::common::RC_SUCCESS;
    }
    //roi info
    std::vector<float> roi_data;
    float roi_0 = 0,roi_1 = 0,roi_2 = 0,roi_3 = 0;
    if(ctx->GetInput<TensorImpl>(1)&&!ctx->GetInput<TensorImpl>(1)->GetShape()->IsEmpty()) {
        if(roi_data_.empty()) {
            const TensorShape& shape = *ctx->GetInput<TensorImpl>(1)->GetShape();
            roi_data.resize(shape.CalcElementsIncludingPadding());
            auto status = ctx->GetInput<TensorImpl>(1)->CopyToHost(roi_data.data());
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "Copy roi data failed: " << ppl::common::GetRetCodeStr(status);
                return status;
            }
            if(roi_data.size() < 8) return ppl::common::RC_INVALID_VALUE;
            roi_0 = roi_data[3];roi_1 = roi_data[7];
            roi_2 = roi_data[2];roi_3 = roi_data[6];
        } else {
            if(roi_data_.size() < 8) return ppl::common::RC_INVALID_VALUE;
            roi_0 = roi_data_[3];roi_1 = roi_data_[7];
            roi_2 = roi_data_[2];roi_3 = roi_data_[6];
        }
    }
    // deal with pre-set h_scale&w_scale
    bool scale_pre_set = false;
    float h_scale = 0.f, w_scale = 0.f;
    std::vector<float> scales_data;
    float* ptr_scales = nullptr;
    if (ctx->GetInput<TensorImpl>(2)&&!ctx->GetInput<TensorImpl>(2)->GetShape()->IsEmpty()) {
        scale_pre_set = true;
        const TensorShape& shape = *ctx->GetInput<TensorImpl>(2)->GetShape();
        if(scales_data_.empty()){
            scales_data.resize(shape.CalcElementsIncludingPadding());
            auto status = ctx->GetInput<TensorImpl>(2)->CopyToHost(scales_data.data());
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "Copy scales data failed: " << ppl::common::GetRetCodeStr(status);
                return status;
            }
            ptr_scales = scales_data.data();
        }else {
            ptr_scales = scales_data_.data();
        }

        if(param_->coord_trans_mode==2){
            int dim_count  = output->GetShape()->GetDimCount();
            int in_height = input->GetShape()->GetDim(2), in_width = 1;
            for (int it = 3; it < dim_count - 1; ++it) {
                in_height *= input->GetShape()->GetDim(it);
            }

            if (dim_count >= 4) {
                in_width  = input->GetShape()->GetDim(dim_count - 1);
            }
            h_scale = (in_height - 1) / (in_height*ptr_scales[2] - 1);
            w_scale = (in_width - 1) / (in_width*ptr_scales[3] - 1);
        }else{
            h_scale = 1.f / ptr_scales[2];
            w_scale = 1.f / ptr_scales[3];
        }
    }
    auto input_id = input->GetEdge()->GetId();
    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input_id);
    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);
    auto status = PPLCUDAResizeForwardImp(
        GetStream(), input->GetShape(), input->GetBufferPtr(), output->GetShape(), output->GetBufferPtr(),
        scale_pre_set, h_scale, w_scale,roi_0,roi_1,roi_2,roi_3, param_->exclude_outside, param_->coord_trans_mode, param_->mode, param_->cubic_coeff_a,
        param_->nearest_mode, input_quant.scale[0], output_quant.scale[0],param_->extrapolation_value);
    return status;
}

}}} // namespace ppl::nn::cuda
