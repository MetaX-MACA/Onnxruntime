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


#include "ppl/nn/engines/cuda/optimizer/fusions/fs_gather_reshape_add.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/fused_gather_reshape_add_op.h"
#include "ppl/nn/params/onnx/gemm_param.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

bool GatherReshapeAddFusion::CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) {
    auto data = options.graph->data.get();
    auto gather_fusion_node = fusion_tree_.fusion_nodes[0];
    if(gather_fusion_node->input_edges[0].is_constant == false){
        return false;
    }
    auto data_shape = GetConstantShape(options , gather_fusion_node->input_edges[0].edge->GetId());
    if(data_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16){
        return false;
    }
    if(((ppl::nn::onnx::GatherParam*)(gather_fusion_node->param))->axis != 0){
        return false;
    }

    auto reshape_fusion_node = fusion_tree_.fusion_nodes[1];
    if(reshape_fusion_node->input_edges[1].is_constant == false){
        return false;
    }
    auto shape_edge_id = reshape_fusion_node->input_edges[1].edge->GetId();
    auto shape_data = data->constants[shape_edge_id].data;
    std::vector<int64_t> shape_vec(shape_data.GetSize() / 8);
    memcpy(shape_vec.data(), shape_data.GetData(), shape_data.GetSize());
    output_shape_ = shape_vec;
    return true;
};

ppl::common::RetCode GatherReshapeAddFusion::FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) {
    return RC_SUCCESS;
};

ppl::common::RetCode GatherReshapeAddFusion::InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) {

    auto topo = options.graph->topo.get();
    node->SetType(ir::Node::Type("pmx", "FusedGatherReshapeAdd", 1));
    auto creator = OptKernelCreatorManager::GetInstance()->Find(node->GetType().domain, node->GetType().name,
                                                                node->GetType().version);
    if (!creator) {
        LOG(ERROR) << "Cannot find creator for FusedGatherReshapeAdd kernel";
        return RC_UNSUPPORTED;
    }
    auto opt_kernel = unique_ptr<CudaOptKernel>((*creator)(node));
    if (!opt_kernel) {
        LOG(ERROR) <<"create Kernel failed: oom";
        return RC_UNSUPPORTED;
    }
    opt_kernel->Init(options);
    ((FusedGatherReshapeAddOp* )(opt_kernel.get()))->SetOutputShape(output_shape_);

    options.info->kernels.erase(node->GetId());
    options.info->kernels.emplace(node->GetId(), std::move(opt_kernel));
    return RC_SUCCESS;
};

}}} // namespace ppl::nn::cuda
