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
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_mul_add.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

bool MulAddFusion::CanFuse(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options) {
    if (nextnode->GetType().name == "Add") {
        if(node->GetInputCount() > 2)
            return false;
        auto topo = options.graph->topo.get();
        auto connect_edge_id = node->GetOutput(0);
        int add_input_idx = 0;
        for(int i = 0; i < nextnode->GetInputCount(); i++){
            if(nextnode->GetInput(i) != connect_edge_id)
                add_input_idx = i;
        }
        TensorShape& mul_input0_shape = *options.tensors->find(node->GetInput(0))->second->GetShape();
        TensorShape& mul_input1_shape = *options.tensors->find(node->GetInput(1))->second->GetShape();
        TensorShape& add_input1_shape = *options.tensors->find(nextnode->GetInput(add_input_idx))->second->GetShape();
        
        int mul0_dim = mul_input0_shape.GetDimCount();
        int mul1_dim = mul_input1_shape.GetDimCount();
        int add1_dim = add_input1_shape.GetDimCount();

        if(mul0_dim == 4 && add1_dim == 4
           && mul_input0_shape.GetDim(1) == add_input1_shape.GetDim(1)
           && mul_input0_shape.GetDim(2) == add_input1_shape.GetDim(2) 
           && mul_input0_shape.GetDim(3) == add_input1_shape.GetDim(3)){
            if (mul_input1_shape.IsScalar() || mul1_dim == 1){
                if (mul_input0_shape.GetDim(0) == add_input1_shape.GetDim(0)){ 
                    return true;
                }
            } 
            if (mul1_dim > 2 && mul_input1_shape.GetDim(mul1_dim - 2) == 1 && mul_input1_shape.GetDim(mul1_dim - 1) == 1){ 
                if (mul_input0_shape.GetDim(0) == add_input1_shape.GetDim(0)){
                    if (mul1_dim == 4 && mul_input0_shape.GetDim(0) == mul_input1_shape.GetDim(0) 
                    && mul_input0_shape.GetDim(1) == mul_input1_shape.GetDim(1)){
                        return true;
                    }
                } 
                if (add_input1_shape.GetDim(0) == 1){
                    if (mul1_dim == 3 && mul_input0_shape.GetDim(1) == mul_input1_shape.GetDim(0)){
                        return true; 
                    }
                    if (mul1_dim == 4 && mul_input1_shape.GetDim(0) == 1 
                       && mul_input0_shape.GetDim(1) == mul_input1_shape.GetDim(1)){
                        return true;
                    }
                }
            }
        }
       
        if(mul0_dim == mul1_dim && mul0_dim == add1_dim){
            bool isSameExpectBatch = true;
            for(int i = 1; i < mul0_dim; i++){
                if(mul_input0_shape.GetDim(i) != mul_input1_shape.GetDim(i)
                   || mul_input0_shape.GetDim(i) != add_input1_shape.GetDim(i))
                    isSameExpectBatch = false;
            }
            return isSameExpectBatch;
        }
    }
    return false;
}

RetCode MulAddFusion::FuseWithNextAdd(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto connect_edge_id = node->GetOutput(0);
    auto connect_edge = topo->GetEdge(connect_edge_id);
    auto next_edge_id = nextnode->GetOutput(0);
    auto next_edge = topo->GetEdge(next_edge_id);
    auto next_node_id = nextnode->GetId();
    std::string fuseOpName = node->GetName() + "_Fused_" + nextnode->GetName();

    for (auto it = next_edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
        auto tempnode_id = it.Get();
        auto tempnode = topo->GetNode(tempnode_id);
        connect_edge->AddConsumer(tempnode_id);
        tempnode->ReplaceInput(next_edge_id, connect_edge_id);
    }
    for(int i = 0; i < nextnode->GetInputCount(); i++){
        auto input_edge_id = nextnode->GetInput(i);
        if(input_edge_id == INVALID_EDGEID || input_edge_id == connect_edge_id)
            continue;
        auto input_edge = topo->GetEdge(input_edge_id);
        input_edge->AddConsumer(node->GetId());
        input_edge->DelConsumer(nextnode->GetId());
        node->AddInput(input_edge_id);
    }
    for(int i = 0; i < topo->GetOutputCount(); i++){
        auto output_id = topo->GetOutput(i);
        if(output_id != next_edge_id)
            continue;
        topo->MarkAsOutput(connect_edge_id);
    }
    connect_edge->DelConsumer(nextnode->GetId());
    connect_edge->SetName(next_edge->GetName());
    node->SetName(fuseOpName);
    topo->DelEdge(next_edge->GetId());
    topo->DelNode(nextnode->GetId());
    node->SetType(ir::Node::Type("pmx", "FusedMulAdd", 1));
    auto creator = OptKernelCreatorManager::GetInstance()->Find(node->GetType().domain, node->GetType().name,
                                                                node->GetType().version);
    if (!creator) {
        LOG(ERROR) << "Cannot find creator for channel shuffle kernel";
        return RC_UNSUPPORTED;
    }
    auto opt_kernel = unique_ptr<CudaOptKernel>((*creator)(node));
    if (!opt_kernel) {
        LOG(ERROR) << "create Kernel failed: oom";
        return RC_UNSUPPORTED;
    }
    options.info->kernels.erase(next_node_id);
    options.info->kernels.erase(node->GetId());
    options.info->kernels.emplace(node->GetId(), std::move(opt_kernel));
    return RC_SUCCESS;
}

RetCode MulAddFusion::FuseWithNextActNode(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto connect_edge_id = node->GetOutput(0);

    for (uint32_t i = 0; i < nextnode->GetOutputCount(); ++i) {
        auto edge_id = nextnode->GetOutput(i);
        auto temp_edge = topo->GetEdge(edge_id);
        temp_edge->SetProducer(node->GetId());
        if (i == 0) {
            node->ReplaceOutput(connect_edge_id, edge_id);
        } else {
            node->AddOutput(edge_id);
        }
    }

    for (uint32_t i = 0; i < nextnode->GetInputCount(); ++i) {
        auto edge_id = nextnode->GetInput(i);
        if (edge_id == connect_edge_id || edge_id == INVALID_EDGEID) {
            continue;
        }
        ir::Edge* edge = topo->GetEdge(edge_id);
        edge->DelConsumer(nextnode->GetId());
        edge->AddConsumer(node->GetId());
        node->AddInput(edge_id);
    }

    topo->DelEdge(connect_edge_id);
    topo->DelNode(nextnode->GetId());
    return RC_SUCCESS;
}

RetCode MulAddFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto node_id = node->GetId();
    auto edge_id = node->GetOutput(0);
    if (edge_id == INVALID_EDGEID) {
        return RC_UNSUPPORTED;
    }
    auto edge = topo->GetEdge(edge_id);
    if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
        return RC_UNSUPPORTED;
    }
    if(topo->GetEdge(edge_id)->CalcConsumerCount() > 1){
        return RC_UNSUPPORTED;
    }
    auto nextnode_id = topo->GetEdge(edge_id)->CreateConsumerIter().Get();
    if (nextnode_id == INVALID_NODEID) {
        return RC_UNSUPPORTED;
    }
    auto nextnode = topo->GetNode(nextnode_id);
    if (CanFuse(node, nextnode, options)) {
        LOG(DEBUG) << "Fuse mul node[" << node->GetName() << "] with nextnode[" << nextnode->GetName() << "]";
        FuseWithNextAdd(node, nextnode, options);
        node = topo->GetNode(node_id);
        edge_id = node->GetOutput(0);
        edge = topo->GetEdge(edge_id);
        if(topo->GetEdge(edge_id)->CalcConsumerCount() > 1){
            return RC_UNSUPPORTED;
        }
        auto consumer_iter = topo->GetEdge(edge_id)->CreateConsumerIter();
        if(consumer_iter.IsValid() ){
            nextnode_id = consumer_iter.Get();
        }else{
            nextnode_id = INVALID_NODEID;
        }
        if (nextnode_id == INVALID_NODEID) {
            return RC_UNSUPPORTED;
        }
        nextnode = topo->GetNode(nextnode_id);
        if(nextnode->GetType().name == "Relu"){
            auto nextnode_name = nextnode->GetName();
            LOG(DEBUG) << "    continue fuse MulAdd node[" << node->GetName() << "] with nextnode[" << nextnode->GetName() << "]";
            FuseWithNextActNode(node, nextnode, options);
            node->SetName(node->GetName() + "_Fused_" + nextnode_name);
            auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node_id].get());
            CudaMulAddParam* param = (CudaMulAddParam*)opt_kernel->GetParam();
            param->extra_param.act_type = 1;
        }
    }

    return RC_SUCCESS;
}
}}} // namespace ppl::nn::cuda
