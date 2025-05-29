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

#include "ppl/nn/engines/cuda/optimizer/ops/pmx/multi_head_attn_v1_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/pmx/multi_head_attention_v1_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_multi_head_attention_v1.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;


namespace ppl { namespace nn { namespace cuda {

RetCode MultiHeadAttnV1Op::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<MultiHeadAttnV1Param>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

MultiHeadAttnV1Op::MultiHeadAttnV1Op(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        type = ppl::common::DATATYPE_FLOAT16; // only support fp16 for now

        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        return status;
    };
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return ppl::nn::onnx::ReshapeMultiHeadAttnV1(info, &param_);
    };

}

RetCode MultiHeadAttnV1Op::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* MultiHeadAttnV1Op::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MultiHeadAttnV1Kernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode MultiHeadAttnV1Op::SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream* ds) const {
/*         flatbuffers::FlatBufferBuilder builder; */
        /* auto fb_param = ppl::nn::pmx::onnx::SerializeReduceParam(param_, &builder); */
        /* auto fb_op_param = ppl::nn::pmx::onnx::CreateOpParam(builder, ppl::nn::pmx::onnx::OpParamType_ReduceParam, fb_param.Union()); */
        /* ppl::nn::pmx::onnx::FinishOpParamBuffer(builder, fb_op_param); */
        /* return ds->Write(builder.GetBufferPointer(), builder.GetSize()); */
        return ppl::common::RC_SUCCESS;
    }
    ppl::common::RetCode MultiHeadAttnV1Op::DeserializeData(const ppl::nn::pmx::DeserializationContext&, const void* base, uint64_t size) {
/*         auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base); */
        /* auto fb_argmax_param = fb_op_param->value_as_ReduceParam(); */
        /* ppl::nn::pmx::onnx::DeserializeReduceParam(*fb_argmax_param, &param_); */
        /* if (GetNode()->GetType().name == "ReduceSum") { */
            /* param_.type = ReduceParam::ReduceSum; */
        /* } else if (GetNode()->GetType().name == "ReduceMax") { */
            /* param_.type = ReduceParam::ReduceMax; */
        /* } else if (GetNode()->GetType().name == "ReduceMin") { */
            /* param_.type = ReduceParam::ReduceMin; */
        /* } else if (GetNode()->GetType().name == "ReduceProd") { */
            /* param_.type = ReduceParam::ReduceProd; */
        /* } else if (GetNode()->GetType().name == "ReduceMean") { */
            /* param_.type = ReduceParam::ReduceMean; */
        /* } else { */
            /* param_.type = ReduceParam::ReduceUnknown; */
/*         } */
        return ppl::common::RC_SUCCESS;
    }
#endif

}}} // namespace ppl::nn::cuda
