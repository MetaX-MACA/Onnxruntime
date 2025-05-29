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

#include "ppl/nn/engines/cuda/optimizer/ops/pmx/fused_mul_add_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/pmx/fused_mul_add_kernel.h"
#include "ppl/nn/oputils/broadcast.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode FusedMulAddOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

FusedMulAddOp::FusedMulAddOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [this](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        uint64_t mask = 0;
        for (uint32_t i = 0; i < info->GetInputCount(); ++i) {
            auto in_tensor = info->GetInput<TensorImpl>(i);
            if (in_tensor->GetType() == TENSORTYPE_RESERVED)
                mask |= 1<<i;
        }
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            //status = InferHighestType(info, type, mask);
            status = InferDefaultType(info, info->GetInput<TensorImpl>(0)->GetShape()->GetDataType());
        } else if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        return status;
    };

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        if (info->GetInputCount() != 3) {
            return RC_INVALID_VALUE;
        }
        const TensorShape& mul_lhs = *info->GetInput<TensorImpl>(0)->GetShape();
        const TensorShape& mul_rhs = *info->GetInput<TensorImpl>(1)->GetShape();
        MultiDirectionalBroadCaster multi_mul_bc;
        multi_mul_bc.SetInputTensorShapes(mul_lhs, mul_rhs);
        multi_mul_bc.CalcBroadCast();
        if (!multi_mul_bc.CanBroadCast()) {
            LOG(DEBUG) << "ERROR: cannot broadcast.";
            return RC_INVALID_VALUE;
        }
        auto& mul_output_shape = multi_mul_bc.OutputTensorShape();
        const TensorShape& add_rhs = *info->GetInput<TensorImpl>(2)->GetShape();
        MultiDirectionalBroadCaster multi_add_bc;
        multi_add_bc.SetInputTensorShapes(mul_output_shape, add_rhs);
        multi_add_bc.CalcBroadCast();
        if (!multi_add_bc.CanBroadCast()) {
            LOG(DEBUG) << "ERROR: cannot broadcast.";
            return RC_INVALID_VALUE;
        }
        auto& add_output_shape = multi_add_bc.OutputTensorShape();
        if (add_output_shape.IsScalar()) {
            info->GetOutput<TensorImpl>(0)->GetShape()->ReshapeAsScalar();
        } else {
            info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(add_output_shape.GetDims(), add_output_shape.GetDimCount());
        }
        return RC_SUCCESS;
    };
}

RetCode FusedMulAddOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* FusedMulAddOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<FusedMulAddKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
