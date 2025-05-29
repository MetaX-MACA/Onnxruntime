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


#include "ppl/nn/engines/cuda/optimizer/fusions/fs_matmul_split.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/batch_gemm_op.h"
#include "ppl/nn/params/onnx/split_param.h"
#include "ppl/nn/engines/cuda/params/gemm_extra_param.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

bool MatMulSplitFusion::CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();

    //judge matmul node
    if(fusion_tree_.fusion_nodes[0]->input_edges[1].is_constant == false){
        return false;
    }
    auto matmul_B_shape = GetConstantShape(options , fusion_tree_.fusion_nodes[0]->input_edges[1].edge->GetId());
    if(matmul_B_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16){
        return false;
    }

    //judge split node
    auto split_fusion_node = fusion_tree_.fusion_nodes[1];
    if(split_fusion_node->input_edges[1].is_constant)
    {
        auto shape_edge_id = split_fusion_node->input_edges[1].edge->GetId();
        auto shape_data = data->constants[shape_edge_id].data;
        std::vector<int64_t> shape_vec(shape_data.GetSize() / 8);
        memcpy(shape_vec.data(), shape_data.GetData(), shape_data.GetSize());
        if(shape_vec.size()!=3){
            return false;
        }
        if(shape_vec[0] != shape_vec[1] || shape_vec[0] != shape_vec[2] || shape_vec[2] != shape_vec[1] ){
            return false;
        }
    }else{
        return false;
    }
    ppl::nn::onnx::SplitParam* param = (ppl::nn::onnx::SplitParam *)(split_fusion_node->param);
    if(param->axis != -1){
        return false;
    }

    //judge reshape node
    auto reshape_fusion_node = fusion_tree_.fusion_nodes[2];
    if(reshape_fusion_node->input_edges[1].is_constant == false){
        return false;
    }
    auto shape_edge_id = reshape_fusion_node->input_edges[1].edge->GetId();
    auto shape_data = data->constants[shape_edge_id].data;
    std::vector<int64_t> shape_vec(shape_data.GetSize() / 8);
    memcpy(shape_vec.data(), shape_data.GetData(), shape_data.GetSize());
    for(size_t i=3; i<fusion_tree_.fusion_nodes.size(); i++){
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

ppl::common::RetCode MatMulSplitFusion::FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto obj_fusion_node = fusion_tree_.fusion_nodes[0];
    auto matmul_B_shape = GetConstantShape(options , obj_fusion_node->input_edges[1].edge->GetId());
    int k = matmul_B_shape->GetDim(0);
    int n = matmul_B_shape->GetDim(1);

    auto edge_constant = data->constants[obj_fusion_node->input_edges[1].edge->GetId()];
    char* ori_ptr = (char*)(edge_constant.data.GetData());

    std::vector<char> new_matmul_B( k * n * 2);
    char* new_ptr = (char*)(new_matmul_B.data());
    int64_t new_n = n / 3;

    for(size_t i=0; i<k; i++){
        for(size_t j=0; j<3; j++){
            memcpy((void*)(new_ptr + (i+j*k)*new_n*2), (void*)(ori_ptr+(i*3+j)*new_n*2), new_n*2);
        }
    }
    memcpy(edge_constant.data.GetData(),new_matmul_B.data(),new_matmul_B.size());
    data->constants[obj_fusion_node->input_edges[1].edge->GetId()] = edge_constant;

    std::vector<int64_t> new_shape = {3, k, new_n};
    matmul_B_shape->Reshape(new_shape);

    for(size_t i=1; i<fusion_tree_.fusion_nodes.size(); i++){
        fusion_tree_.fusion_nodes[i]->input_edges[1].keep = false;
        fusion_tree_.fusion_nodes[i]->input_edges[1].edge->DelConsumer(fusion_tree_.found_node_ids[i]);
        if(fusion_tree_.fusion_nodes[i]->input_edges[1].edge->CalcConsumerCount() == 0){
            need_clear_edge_ids_.push_back(fusion_tree_.fusion_nodes[i]->input_edges[1].edge->GetId());
        }
    }
    return RC_SUCCESS;
};

ppl::common::RetCode MatMulSplitFusion::InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) {
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
    ((BatchGemmOp* )opt_kernel.get())->SetOutputShape(output_shape_);
    options.info->kernels.erase(node->GetId());
    options.info->kernels.emplace(node->GetId(), std::move(opt_kernel));
    return RC_SUCCESS;
};

bool BatchMatMulFusion::CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto golden_fusion_node = fusion_tree_.fusion_nodes[0];
    for(size_t i=1; i<matmul_num_; i++){
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
    //judge reshape node
    auto reshape_fusion_node = fusion_tree_.fusion_nodes[matmul_num_];
    if(reshape_fusion_node->input_edges[1].is_constant == false){
        return false;
    }
    auto shape_edge_id = reshape_fusion_node->input_edges[1].edge->GetId();
    auto shape_data = data->constants[shape_edge_id].data;
    std::vector<int64_t> shape_vec(shape_data.GetSize() / 8);
    memcpy(shape_vec.data(), shape_data.GetData(), shape_data.GetSize());
    for(size_t i=matmul_num_; i<fusion_tree_.fusion_nodes.size(); i++){
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

ppl::common::RetCode BatchMatMulFusion::FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto obj_fusion_node = fusion_tree_.fusion_nodes[0];
    for(size_t i=1; i< obj_fusion_node->input_edges.size(); i++){
        auto obj_edge_id = obj_fusion_node->input_edges[i].edge->GetId();
        // reshape
        auto shape_ori = options.tensors->find(obj_edge_id)->second->GetShape();
        std::vector<int64_t> shape_new(shape_ori->GetDimCount() + 1);
        shape_new[0] = matmul_num_;
        for(size_t j = 0; j < shape_ori->GetDimCount(); j++){
            shape_new[j+1] = shape_ori->GetDim(j);
        }
        shape_ori->Reshape(shape_new);
        // modify const buffer
        auto fused_constant = data->constants[obj_edge_id];
        for(size_t j=1; j<matmul_num_; j++){
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

ppl::common::RetCode BatchMatMulFusion::InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) {
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
    ((BatchGemmOp* )opt_kernel.get())->SetOutputShape(output_shape_);
    options.info->kernels.erase(node->GetId());
    options.info->kernels.emplace(node->GetId(), std::move(opt_kernel));
    return RC_SUCCESS;
};


bool BatchMatMulFusion::InitNodeRule(ir::Node* node, const OptKernelOptions& options){
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto fuse_node_info = ParseNodeToFusionNode(node, options);
    if(fuse_node_info->input_edges[1].is_constant == false){
        return false;
    }
    rule_shape_ = *options.tensors->find(fuse_node_info->input_edges[1].edge->GetId())->second->GetShape();
    return true;
};

bool BatchMatMulFusion::RuleJudge(ir::Node* node, const OptKernelOptions& options){
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto fuse_node_info = ParseNodeToFusionNode(node, options);
    if(node->GetName() != "MatMul"){
        return false;
    }
    if(fuse_node_info->input_edges.size() < 2 || fuse_node_info->input_edges[1].is_constant == false){
        return false;
    }
    if(!(rule_shape_ == *options.tensors->find(fuse_node_info->input_edges[1].edge->GetId())->second->GetShape())){
        return false;
    }
    return true;
};


bool FusedMatMulDiv::CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) {
    auto div_node = fusion_tree_.fusion_nodes[1];
    if(div_node->input_edges[1].is_constant == false){
        return false;
    }
    auto div_input_1_shape = GetConstantShape(options, div_node->input_edges[1].edge->GetId());
    if(div_input_1_shape->GetDimCount() != 1 || div_input_1_shape->GetDim(0) != 1){
        return false;
    }
    return true;
};

ppl::common::RetCode FusedMatMulDiv::FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) {
    auto div_node = fusion_tree_.fusion_nodes[1];
    JoinConstantInputToClearList(div_node, 1);
    return RC_SUCCESS;
};
ppl::common::RetCode FusedMatMulDiv::InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();

    auto div_node = fusion_tree_.fusion_nodes[1];
    auto div_constant = data->constants[div_node->input_edges[1].edge->GetId()];

    auto node_id = node->GetId();
    auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node_id].get());
    CudaGemmParam* param = (CudaGemmParam*)opt_kernel->GetParam();
    param->param.alpha *=  1.0 / __half2float(((__half*)(div_constant.data.GetData()))[0]);
    return RC_SUCCESS;
};

}}} // namespace ppl::nn::cuda
