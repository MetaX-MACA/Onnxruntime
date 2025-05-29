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

#include "ppl/nn/engines/cuda/kernels/onnx/deformconv_kernel.h"
#include "ppl/nn/engines/cuda/module/cuda_module.h"
#include "ppl/common/destructor.h"
#include "cudakernel/nn/deform_conv.h"

namespace ppl { namespace nn { namespace cuda {

bool DeformConvKernel::CanDoExecute(const KernelExecContext& ctx) const {
    for (uint32_t i = 0; i < ctx.GetInputCount(); ++i) {
        if (i == 2)
            continue;
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor || tensor->GetShape()->CalcBytesIncludingPadding() == 0) {
            LOG(WARNING) << "Cannot execute " << GetName();
            return false;
        }
    }
    return true;
}

ppl::common::RetCode DeformConvKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);
    auto offset = ctx->GetInputCount() > 2 ? ctx->GetInput<TensorImpl>(2) : nullptr;
    auto bias = ctx->GetInputCount() > 3 ? ctx->GetInput<TensorImpl>(3) : nullptr;
    auto mask = ctx->GetInputCount() > 4 ? ctx->GetInput<TensorImpl>(4) : nullptr;

    const TensorShape* shape_in0 = ctx->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape* shape_in3 = ctx->GetInput<TensorImpl>(1)->GetShape();
    const TensorShape* shape_out = ctx->GetOutput<TensorImpl>(0)->GetShape();

    int64_t size = PPLCUDADeformConvGetBufSize(shape_in0, shape_in3, shape_out);
    BufferDesc tmp_buffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });

    const int64_t num_output = weight->GetShape()->GetDim(0);
    const int64_t channels = weight->GetShape()->GetDim(1) * param_->groups;
    const int64_t kernel_h = weight->GetShape()->GetDim(2);
    const int64_t kernel_w = weight->GetShape()->GetDim(3);

    auto stream = GetStream();
    CudaDevice* device = GetCudaDevice();
    CUfunction module_func = nullptr;
#ifdef PPLNN_ENABLE_CUDA_JIT
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);
    module_func = module->GetKernelFunc();
#endif

    status = PPLOnnxCUDADeformConvForward(device->GetDeviceProp(), stream, module_func, output->GetShape(), input->GetShape(),
                                      output->GetBufferPtr(), input->GetBufferPtr(), weight->GetBufferPtr(),
                                      offset->GetBufferPtr(),
                                      mask ? mask->GetBufferPtr() : nullptr, bias ? bias->GetBufferPtr() : nullptr,
                                      param_->groups, param_->deform_groups, channels, num_output,
                                      param_->strides[0], param_->strides[1], kernel_h, kernel_w,
                                      param_->pads[0], param_->pads[1], param_->dilations[0], param_->dilations[1],
                                      mask, tmp_buffer_desc.addr);

    return status;
}

}}} // namespace ppl::nn::cuda
