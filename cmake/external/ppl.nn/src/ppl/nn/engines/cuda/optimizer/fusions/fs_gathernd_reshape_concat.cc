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

#include "ppl/nn/engines/cuda/params/fused_gathernd_reshape_concat_param.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_gathernd_reshape_concat.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

bool GatherNDReshapeConcatFusion::CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) {
    if(fusion_tree_.fusion_nodes.size() != 3){
        return false;
    }
    if(!SupportNodeDataType(node, options, {ppl::common::DATATYPE_FLOAT16})){
        return false;
    }
    for(auto p : fusion_tree_.fusion_nodes){
        if(p->output_edges.size() > 1){
            return false;
        }
        if(p->output_edges[0].edge->CalcConsumerCount() > 1){
            return false;
        }
    }
    auto fusion_concat_node = fusion_tree_.fusion_nodes[2];
    if(fusion_concat_node->input_edges.size() != 2){
        return false;
    }
    return true;
}

ppl::common::RetCode GatherNDReshapeConcatFusion::FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) {
    return RC_SUCCESS;
}

ppl::common::RetCode GatherNDReshapeConcatFusion::InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    node->SetType(ir::Node::Type("pmx", "FusedGatherNDReshapeConcat", 1));
    auto creator = OptKernelCreatorManager::GetInstance()->Find(node->GetType().domain, node->GetType().name,
                                                                node->GetType().version);
    if (!creator) {
        LOG(ERROR) << "Cannot find creator for FusedGatherNDReshapeConcat kernel";
        return RC_UNSUPPORTED;
    }

    std::shared_ptr<ppl::nn::ir::Attr> param = std::make_shared<FusedGatherNDReshapeConcatParam>();
    ((FusedGatherNDReshapeConcatParam* )param.get())->gathernd_param = *((ppl::nn::onnx::GatherNDParam *)(fusion_tree_.fusion_nodes[0]->param));
    ((FusedGatherNDReshapeConcatParam* )param.get())->concat_param = *((ppl::nn::onnx::ConcatParam *)(fusion_tree_.fusion_nodes[2]->param));
    auto opt_kernel = unique_ptr<CudaOptKernel>((*creator)(node));
    if (!opt_kernel) {
        LOG(ERROR) << "create Kernel failed: oom";
        return RC_UNSUPPORTED;
    }
    options.graph->data.get()->attrs[node->GetId()] = param;
    opt_kernel->Init(options);
    options.info->kernels.erase(node->GetId());
    options.info->kernels.emplace(node->GetId(), std::move(opt_kernel));
    return RC_SUCCESS;
}


bool GatherNDReshapeConcatBridgeFusion::CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) {
    auto bridge_fusion = fusion_tree_.fusion_nodes[1];
    auto output_edge = bridge_fusion->output_edges[0].edge;
    auto output_shape = *options.tensors->find(output_edge->GetId())->second->GetShape();
    if(output_shape.GetDataFormat() != DATAFORMAT_NHWC8){
        return false;
    }
    return true;
};
ppl::common::RetCode GatherNDReshapeConcatBridgeFusion::FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) {
    return RC_SUCCESS;
};
ppl::common::RetCode GatherNDReshapeConcatBridgeFusion::InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) {
    return RC_SUCCESS;
};

ppl::common::RetCode GatherNDReshapeConcatFusion_Set::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    auto status = base_fusion_.FuseNode(node, reliable, options);
    if(status != RC_SUCCESS){
        return status;
    }
    status = base_bridge_fusion_.FuseNode(node, reliable, options);
    return RC_SUCCESS;
};

}}} // namespace ppl::nn::cuda
