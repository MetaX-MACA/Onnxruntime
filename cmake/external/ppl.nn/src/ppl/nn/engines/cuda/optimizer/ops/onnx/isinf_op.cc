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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/isinf_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/isinf_kernel.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace cuda {

RetCode IsinfOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<IsinfParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    
    return RC_SUCCESS;
}

void IsinfOp::CopyParam(void*& param) {
    if (param == nullptr) {
        param = new IsinfParam();
    }
    *(IsinfParam*)param = param_;
    return;
}

IsinfOp::IsinfOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        TensorShape& in_shape = *info->GetInput<TensorImpl>(0)->GetShape();
        if(in_shape.GetDataType() != ppl::common::DATATYPE_FLOAT16 && in_shape.GetDataType() != ppl::common::DATATYPE_FLOAT32
           && in_shape.GetDataType() != ppl::common::DATATYPE_FLOAT64)
        {
            LOG(ERROR) << "IsinfOp input datatye must be : FLOAT16/FLOAT32/FLOAT64";
            return RC_UNSUPPORTED;
        }
        TensorShape& out_shape = *info->GetOutput<TensorImpl>(0)->GetShape();
        out_shape.SetDataType(ppl::common::DATATYPE_BOOL);
        return RC_SUCCESS;
    };

    infer_dims_func_ = GenericInferDims;
}

RetCode IsinfOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* IsinfOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<IsinfKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
