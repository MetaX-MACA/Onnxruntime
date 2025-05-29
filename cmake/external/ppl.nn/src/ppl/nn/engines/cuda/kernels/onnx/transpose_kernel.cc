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

#include "ppl/nn/engines/cuda/kernels/onnx/transpose_kernel.h"

#include "cudakernel/memory/transpose.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode TransposeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    const TensorShape& in_shape0 = *input->GetShape();

    ppl::nn::onnx::TransposeParam modified_param = *param_;
    if (modified_param.perm.empty()) {
        int32_t dim_count = in_shape0.GetDimCount();
        modified_param.perm.resize(dim_count);
        for (int it = 0; it < dim_count; ++it) {
            modified_param.perm[it] = dim_count - it - 1;
        }
    }
    auto in_shape = input->GetShape();
    auto out_shape = output->GetShape();
    int32_t start_change_index = - 1;
    int32_t end_change_index = -1;
    for(uint i = 0; i < in_shape->GetDimCount(); i++){
        if(i != modified_param.perm[i]){
            if(start_change_index == -1){
                start_change_index = i;
            }else{
                end_change_index = i;
            }
        }
    }
    bool can_trans = false;
    if(start_change_index != -1 && end_change_index != -1){
        uint64_t elementAcc = 1;
        for(int32_t i = start_change_index; i < end_change_index + 1; i++){
            elementAcc *= in_shape->GetDim(i);
        }
        for(int32_t i = start_change_index; i < end_change_index + 1; i++){
            if(in_shape->GetDim(i) == elementAcc){
                can_trans = true;
            }
        }
    }
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    if(can_trans && in_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY){
        //output->TransferBufferFrom(input);
        cudaMemcpyAsync(output->GetBufferPtr(), input->GetBufferPtr(), input->GetShape()->CalcBytesIncludingPadding(), cudaMemcpyDeviceToDevice, GetStream());
    }else{
        TransposeKernelParam param_kernel_;
        param_kernel_.perm = modified_param.perm;
        status = PPLCUDATransposeForwardImp(GetStream(), param_kernel_, input->GetShape(), input->GetBufferPtr(),
                                            output->GetShape(), output->GetBufferPtr());
    }
    return status;
}

}}} // namespace ppl::nn::cuda
