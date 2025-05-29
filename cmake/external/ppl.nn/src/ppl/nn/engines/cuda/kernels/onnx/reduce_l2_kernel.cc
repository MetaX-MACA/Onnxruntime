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

#include "ppl/nn/engines/cuda/kernels/onnx/reduce_l2_kernel.h"

#include <numeric>

#include "cudakernel/reduce/reduce.h"
#include "cudakernel/unary/unary.h"
#include "ppl/common/destructor.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode ReduceL2Kernel::DoExecute(KernelExecContext* ctx) {
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ReduceParam param = ReduceSum;
    const TensorShape& input_shape = *input->GetShape();
    uint32_t n_outer = 1, n_reduce = 1, n_inner = 1;

    const uint32_t dim_count = input_shape.GetDimCount();
    if (param_->axes.empty()) { // empty axes means reduce all dims
        n_reduce =
            accumulate(input_shape.GetDims(), input_shape.GetDims() + dim_count, n_reduce, std::multiplies<uint32_t>());
    } else {
        std::vector<uint32_t> real_axis(param_->axes.size());

        for (uint32_t i = 0; i < param_->axes.size(); ++i) {
            real_axis[i] = (param_->axes[i] + dim_count) % dim_count;
            if (i > 0 && real_axis[i] != real_axis[i - 1] + 1) {
                return ppl::common::RC_UNSUPPORTED;
            }
            n_reduce *= input_shape.GetDim(real_axis[i]);
        }
        n_outer = accumulate(input_shape.GetDims(), input_shape.GetDims() + real_axis[0], n_outer,
                             std::multiplies<uint32_t>());
        n_inner = accumulate(input_shape.GetDims() + real_axis[param_->axes.size() - 1] + 1,
                             input_shape.GetDims() + dim_count, n_inner, std::multiplies<uint32_t>());
    }
    PPLReduceDimDes des(n_inner, n_reduce, n_outer);

    if(input->GetShape()->GetDataType() == ppl::common::DATATYPE_FLOAT16){
        auto tmp_input_shape = *input->GetShape();
        tmp_input_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        auto size = tmp_input_shape.CalcBytesIncludingPadding();
        if(size > tmp_buffer_len_){
            tmp_buffer_len_ = size;
            auto status = GetCudaDevice()->Realloc(tmp_input_shape, &tmp_buffer_desc);
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for tmp_buffer_desc failed: " << ppl::common::GetRetCodeStr(status);
                return status;
            }
        }
        auto tmp_output_shape = *output->GetShape();
        tmp_output_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        auto size_out = tmp_output_shape.CalcBytesIncludingPadding();
        if(size_out > tmp_buffer_out_len_){
            tmp_buffer_out_len_ = size_out;
            auto status = GetCudaDevice()->Realloc(tmp_output_shape, &tmp_buffer_out_desc);
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for tmp_buffer_out_desc failed: " << ppl::common::GetRetCodeStr(status);
                return status;
            }
        }
        auto converter = output->GetDevice()->GetDataConverter();
        status = converter->Convert(&tmp_buffer_desc, tmp_input_shape, input->GetBufferDesc(), *input->GetShape());
        if (status != ppl::common::RC_SUCCESS){
            LOG(ERROR) << "Convert fp16 to fp32 failed";
            return status;
        }
        status = PPLCUDAUnarySquareForwardImp(GetStream(), &tmp_input_shape, tmp_buffer_desc.addr, &tmp_input_shape,
                                            tmp_buffer_desc.addr);
        if (status != ppl::common::RC_SUCCESS){
            LOG(ERROR) << "PPLCUDAUnarySquareForwardImp failed";
            return status;
        }
        status = PPLCUDAReduceForwardImp(GetStream(), param, des, &tmp_input_shape, tmp_buffer_desc.addr,
                                        &tmp_output_shape, tmp_buffer_out_desc.addr);
        if (status != ppl::common::RC_SUCCESS){
            LOG(ERROR) << "PPLCUDAReduceForwardImp failed";
            return status;
        }
        status = PPLCUDAUnarySqrtForwardImp(GetStream(), &tmp_output_shape, tmp_buffer_out_desc.addr, &tmp_output_shape,
                                            tmp_buffer_out_desc.addr);
        if (status != ppl::common::RC_SUCCESS){
            LOG(ERROR) << "PPLCUDAUnarySqrtForwardImp failed";
            return status;
        }
        status = converter->Convert(&output->GetBufferDesc(), *output->GetShape(), tmp_buffer_out_desc, tmp_output_shape);
        if (status != ppl::common::RC_SUCCESS){
            LOG(ERROR) << "Convert fp32 to fp16 failed";
            return status;
        }
    }else if(input->GetShape()->GetDataType() == ppl::common::DATATYPE_FLOAT32){
        auto tmp_input_shape = *input->GetShape();
        auto size = tmp_input_shape.CalcBytesIncludingPadding();
        if(size > tmp_buffer_len_){
            tmp_buffer_len_ = size;
            auto status = GetCudaDevice()->Realloc(tmp_input_shape, &tmp_buffer_desc);
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for tmp_buffer_desc failed: " << ppl::common::GetRetCodeStr(status);
                return status;
            }
        }
        status = PPLCUDAUnarySquareForwardImp(GetStream(), input->GetShape(), input->GetBufferPtr(), input->GetShape(),
                                            tmp_buffer_desc.addr);
        if (status != ppl::common::RC_SUCCESS){
            LOG(ERROR) << "PPLCUDAUnarySquareForwardImp failed";
            return status;
        }
        status = PPLCUDAReduceForwardImp(GetStream(), param, des, input->GetShape(), tmp_buffer_desc.addr,
                                        output->GetShape(), output->GetBufferPtr());
        if (status != ppl::common::RC_SUCCESS){
            LOG(ERROR) << "PPLCUDAReduceForwardImp failed";
            return status;
        }
        status = PPLCUDAUnarySqrtForwardImp(GetStream(), output->GetShape(), output->GetBufferPtr(), output->GetShape(),
                                            output->GetBufferPtr());
        if (status != ppl::common::RC_SUCCESS){
            LOG(ERROR) << "PPLCUDAUnarySqrtForwardImp failed";
            return status;
        }
    }else{
        LOG(ERROR) << "Unsupport input data type" << ppl::common::GetDataTypeStr(input->GetShape()->GetDataType());
    }

    return status;
}

}}} // namespace ppl::nn::cuda
