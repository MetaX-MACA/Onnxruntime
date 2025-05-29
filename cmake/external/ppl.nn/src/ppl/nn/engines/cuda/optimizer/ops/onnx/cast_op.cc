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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/cast_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/cast_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_cast.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/cast.h"
#endif

namespace ppl { namespace nn { namespace cuda {

RetCode CastOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<CastParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

CastOp::CastOp(const ir::Node* node) : CudaOptKernel(node)  {
    infer_type_func_ = [this](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
        #if 1
            TensorShape& in_shape = *info->GetInput<TensorImpl>(0)->GetShape();
            if (in_shape.GetDataType() == ppl::common::DATATYPE_UNKNOWN) {
                LOG(ERROR) << "Input edge has unknown type.";
                return ppl::common::RC_UNSUPPORTED;
            }
            status = ppl::common::RC_SUCCESS;
        #else
            status = InferInheritedType(info);
        #endif
        } else if (type == DATATYPE_INT8) {
            // TODO
            LOG(ERROR) << "Cast Op not support INT8";
        } else {
            status = InferDefaultType(info, type);
        }
        auto output = info->GetOutput<TensorImpl>(0)->GetShape();
        output->SetDataType(param_.to);
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeCast(info, &param_);
    };
}

RetCode CastOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* CastOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<CastKernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode CastOp::SerializeData(const pmx::SerializationContext&, utils::DataStream* ds) const {
        flatbuffers::FlatBufferBuilder builder;
        auto fb_param = pmx::onnx::SerializeCastParam(param_, &builder);
        auto fb_op_param = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_CastParam, fb_param.Union());
        pmx::onnx::FinishOpParamBuffer(builder, fb_op_param);
        return ds->Write(builder.GetBufferPointer(), builder.GetSize());
    }
    ppl::common::RetCode CastOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
        auto fb_op_param = pmx::onnx::GetOpParam(base);
        auto fb_argmax_param = fb_op_param->value_as_CastParam();
        pmx::onnx::DeserializeCastParam(*fb_argmax_param, &param_);
        return ppl::common::RC_SUCCESS;
    }
#endif

RetCode CastOp::DeleteCastNode(ir::Node* node, ir::Graph* graph,
                               std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                               std::vector<CudaTensorQuant>* quants){
    auto topo = graph->topo.get();
    auto preedge_id = node->GetInput(0);
    auto postedge_id = node->GetOutput(0);
    auto preedge = topo->GetEdge(preedge_id);
    auto nextnode_id = topo->GetEdge(postedge_id)->CreateConsumerIter().Get(); // consumer0
    auto nextnode = topo->GetNode(nextnode_id);
    if (topo->GetEdge(postedge_id)->CalcConsumerCount() == 0) { // final bridge node
        return RC_UNSUPPORTED;
    }
    auto input_shape = (*tensors).find(preedge_id)->second->GetShape();
    auto output_shape = (*tensors).find(postedge_id)->second->GetShape();
    if(input_shape->GetDataType() == output_shape->GetDataType()){
        preedge->DelConsumer(node->GetId());
        for (auto it = topo->GetEdge(postedge_id)->CreateConsumerIter(); it.IsValid(); it.Forward()) {
            nextnode_id = it.Get();
            preedge->AddConsumer(nextnode_id);
            auto nextnode = topo->GetNode(nextnode_id);
            nextnode->ReplaceInput(postedge_id, preedge_id);
        }
        topo->DelEdge(postedge_id);
        topo->DelNode(node->GetId());
        return RC_SUCCESS;
    }
    return RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::cuda
