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

#include "ppl/nn/engines/cuda/params/instance_normalization_extra_param.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_add_layernorm.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

bool AddLayerNormFusion::CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options){
    std::shared_ptr<FusionNodeInfo> fusion_node = fusion_tree_.fusion_nodes[0];
    TensorShape& add_input0_shape = *options.tensors->find(fusion_node->input_edges[0].edge->GetId())->second->GetShape();
    TensorShape& add_input1_shape = *options.tensors->find(fusion_node->input_edges[1].edge->GetId())->second->GetShape();
    if(add_input0_shape.GetDimCount() != add_input1_shape.GetDimCount()){
        return false;
    }
    if(!SupportNodeDataType(node, options, {ppl::common::DATATYPE_FLOAT16, ppl::common::DATATYPE_INT8})){
        return false;
    }
    return true;
}

ppl::common::RetCode AddLayerNormFusion::FilteNodesInfo(ir::Node* node, const OptKernelOptions& options){
    return RC_SUCCESS;
}

ppl::common::RetCode AddLayerNormFusion::InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options){
    options.graph->data.get()->attrs[node->GetId()] = options.graph->data.get()->attrs[fusion_tree_.found_node_ids[1]];
    node->SetType(ir::Node::Type("pmx", "FusedAddLayerNorm", 1));
    auto creator = OptKernelCreatorManager::GetInstance()->Find(node->GetType().domain, node->GetType().name,
                                                                node->GetType().version);
    if (!creator) {
        LOG(ERROR) << "Cannot find creator for FusedAddLayerNorm kernel";
        return RC_UNSUPPORTED;
    }
    auto opt_kernel = unique_ptr<CudaOptKernel>((*creator)(node));
    opt_kernel->Init(options);
    if (!opt_kernel) {
        LOG(ERROR) << "create Kernel failed: oom";
        return RC_UNSUPPORTED;
    }
    options.info->kernels.erase(node->GetId());
    options.info->kernels.emplace(node->GetId(), std::move(opt_kernel));
    return RC_SUCCESS;
};

}}} // namespace ppl::nn::cuda
