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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/one_hot_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/one_hot_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_one_hot.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace cuda {

RetCode OneHotOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<OneHotParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

OneHotOp::OneHotOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status = ppl::common::RC_SUCCESS;
        if (type == DATATYPE_UNKNOWN) {
            //status = InferInheritedType(info);
            auto value_shape = info->GetInput<TensorImpl>(2)->GetShape();
            auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
            output_shape->SetDataType(value_shape->GetDataType());
        } else if (type == DATATYPE_INT8) {
            //status = UnifyToOutputQuant(info, quant);
            status = CopyQuantType(info, quant);
        } else {
            //status = InferDefaultType(info, type);
            auto value_shape = info->GetInput<TensorImpl>(2)->GetShape();
            auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
            output_shape->SetDataType(value_shape->GetDataType());
        }
        //auto shape = info->GetInput<TensorImpl>(1)->GetShape();
        //shape->SetDataType(DATATYPE_INT64);
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
            int64_t depth;
            auto shape = info->GetInput<TensorImpl>(1)->GetShape();
            if(shape->GetDataType() == DATATYPE_FLOAT32){
                float_t  depth_f;
                info->GetInput<TensorImpl>(1)->CopyToHost(&depth_f);
                depth = (int64_t)depth_f;
            }else if(shape->GetDataType() == DATATYPE_INT32){
                int32_t  depth_int32;
                info->GetInput<TensorImpl>(1)->CopyToHost(&depth_int32);
                depth = (int64_t)depth_int32;
            }else if(shape->GetDataType() == DATATYPE_INT64){
                info->GetInput<TensorImpl>(1)->CopyToHost(&depth);
            }else{
                LOG(ERROR) << "Oneshot  depth not suppport: " << GetDataTypeStr(shape->GetDataType());
            }
            return onnx::ReshapeOneHot(info, &param_, depth);
    };
}

RetCode OneHotOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* OneHotOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<OneHotKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
