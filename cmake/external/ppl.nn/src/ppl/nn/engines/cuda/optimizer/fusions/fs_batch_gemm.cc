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


#include "ppl/nn/engines/cuda/optimizer/fusions/fs_batch_gemm.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/batch_gemm_op.h"
#include "ppl/nn/params/onnx/gemm_param.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

bool BatchGemmFusion::CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) {
    auto golden_fusion_node = fusion_tree_.fusion_nodes[0];
    // only support A,B,C
    if(golden_fusion_node->input_edges.size() !=3 ){
        return false;
    }
    for(auto p : fusion_tree_.fusion_nodes){
        ppl::nn::onnx::GemmParam* param = (ppl::nn::onnx::GemmParam *)(p->param);
        if(param->alpha!=1 || param->beta!=1 || param->transA!=0 || param->transB!=0){
            return false;
        }
    }
    for(size_t i=1; i<fusion_tree_.fusion_nodes.size(); i++){
        auto fusion_node = fusion_tree_.fusion_nodes[i];
        if(golden_fusion_node->input_edges.size() != fusion_node->input_edges.size() ||
            golden_fusion_node->output_edges.size() != fusion_node->output_edges.size()
        ){
            return false;
        }else{
            for(size_t j=1; j<golden_fusion_node->input_edges.size(); j++){
                if(!golden_fusion_node->input_edges[j].is_constant ||
                !fusion_node->input_edges[j].is_constant
                ){
                    return false;
                }
                if(!((*options.tensors->find(golden_fusion_node->input_edges[j].edge->GetId())->second->GetShape()) ==
                (*options.tensors->find(fusion_node->input_edges[j].edge->GetId())->second->GetShape()))){
                    return false;
                }
            }
        }
    }
    return true;
};

ppl::common::RetCode BatchGemmFusion::FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto obj_fusion_node = fusion_tree_.fusion_nodes[0];
    for(size_t i=1; i< obj_fusion_node->input_edges.size(); i++){
        auto obj_edge_id = obj_fusion_node->input_edges[i].edge->GetId();
        // reshape
        auto shape_ori = options.tensors->find(obj_edge_id)->second->GetShape();
        std::vector<int64_t> shape_new(shape_ori->GetDimCount() + 1);
        shape_new[0] = fusion_tree_.fusion_nodes.size();
        for(size_t j = 0; j < shape_ori->GetDimCount(); j++){
            shape_new[j+1] = shape_ori->GetDim(j);
        }
        shape_ori->Reshape(shape_new);
        // modify const buffer
        auto fused_constant = data->constants[obj_edge_id];
        for(size_t j=1; j<fusion_tree_.fusion_nodes.size(); j++){
            auto fusion_node = fusion_tree_.fusion_nodes[j];
            fusion_node->input_edges[i].keep = false;
            need_clear_edge_ids_.push_back(fusion_node->input_edges[i].edge->GetId());
            // JoinConstantInputToClearList(fusion_node, j);
            auto edge_constant = data->constants[fusion_node->input_edges[i].edge->GetId()];
            fused_constant.data.Append(edge_constant.data.GetData(), edge_constant.data.GetSize());
        }
        data->constants[obj_edge_id] = fused_constant;
    }
    return RC_SUCCESS;
};

ppl::common::RetCode BatchGemmFusion::InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    node->SetType(ir::Node::Type("pmx", "BatchGemm", 1));
    auto creator = OptKernelCreatorManager::GetInstance()->Find(node->GetType().domain, node->GetType().name,
                                                                node->GetType().version);
    if (!creator) {
        LOG(ERROR) << "Cannot find creator for BatchGemm kernel";
        return RC_UNSUPPORTED;
    }
    auto opt_kernel = unique_ptr<CudaOptKernel>((*creator)(node));
    if (!opt_kernel) {
        LOG(ERROR) <<"create Kernel failed: oom";
        return RC_UNSUPPORTED;
    }
    opt_kernel->Init(options);
    options.info->kernels.erase(node->GetId());
    options.info->kernels.emplace(node->GetId(), std::move(opt_kernel));
    return RC_SUCCESS;
};

bool BatchGemmReshapeFusion::CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();

    auto reshape_fusion_node = fusion_tree_.fusion_nodes[1];
    if(reshape_fusion_node->input_edges[1].is_constant == false){
        return false;
    }
    auto shape_edge_id = reshape_fusion_node->input_edges[1].edge->GetId();
    auto shape_data = data->constants[shape_edge_id].data;
    std::vector<int64_t> shape_vec(shape_data.GetSize() / 8);
    memcpy(shape_vec.data(), shape_data.GetData(), shape_data.GetSize());
    for(size_t i=2; i<fusion_tree_.fusion_nodes.size(); i++){
        if(fusion_tree_.fusion_nodes[i]->input_edges[1].is_constant == false){
            return false;
        }
        auto temp_shape_edge_id = fusion_tree_.fusion_nodes[i]->input_edges[1].edge->GetId();
        auto temp_shape_data = data->constants[temp_shape_edge_id].data;
        std::vector<int64_t> temp_shape_vec(temp_shape_data.GetSize() / 8);
        memcpy(temp_shape_vec.data(), temp_shape_data.GetData(), temp_shape_data.GetSize());
        if(shape_vec !=  temp_shape_vec){
            return false;
        }
    }
    output_shape_ = shape_vec;
    return true;
};
ppl::common::RetCode BatchGemmReshapeFusion::FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) {
    for(size_t i=1; i<fusion_tree_.fusion_nodes.size(); i++){
        fusion_tree_.fusion_nodes[i]->input_edges[1].keep = false;
        fusion_tree_.fusion_nodes[i]->input_edges[1].edge->DelConsumer(fusion_tree_.found_node_ids[i]);
        if(fusion_tree_.fusion_nodes[i]->input_edges[1].edge->CalcConsumerCount() == 0){
            need_clear_edge_ids_.push_back(fusion_tree_.fusion_nodes[i]->input_edges[1].edge->GetId());
        }
    }
    return RC_SUCCESS;
};
ppl::common::RetCode BatchGemmReshapeFusion::InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) {
    auto opt_kernel = (BatchGemmOp* )(options.info->kernels[node->GetId()].get());
    opt_kernel->SetOutputShape(output_shape_);
    return RC_SUCCESS;
};




}}} // namespace ppl::nn::cuda
