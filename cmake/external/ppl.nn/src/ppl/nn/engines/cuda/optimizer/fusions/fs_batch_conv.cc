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


#include "ppl/nn/engines/cuda/optimizer/fusions/fs_batch_conv.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/params/onnx/conv_param.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"

using namespace ppl::common;
using namespace ppl::nn::onnx;
namespace ppl { namespace nn { namespace cuda {

bool BatchConvFusion::CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) {
    output_channels_.clear();
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto tensor_params = &(options.args->quant_info.tensor_params);

    // w and b must be constant
    // only support int8 & output scale is same
    std::string output_scale = "";
    for(size_t i=1; i< fusion_tree_.fusion_nodes.size(); i++){
        auto f_node = fusion_tree_.fusion_nodes[i];
        if(i< conv_num_){
            for(size_t j=1; j<f_node->input_edges.size(); j++){
                auto input_edge_name = f_node->input_edges[j].edge->GetName();
                if(j==1 && tensor_params->find(input_edge_name) == tensor_params->end()){
                    return false;
                }
                if(f_node->input_edges[j].is_constant == false){
                    return false;
                }
            }
        }else{
            auto output_edge_name = f_node->output_edges[0].edge->GetName();
            if(tensor_params->find(output_edge_name) == tensor_params->end()){
                return false;
            }
            if(tensor_params->at(output_edge_name).fields.find("scale") == tensor_params->at(output_edge_name).fields.end()){
                return false;
            }
            if(output_scale.size() == 0){
                output_scale = tensor_params->at(output_edge_name).fields["scale"].content;
            }else{
                if(output_scale != tensor_params->at(output_edge_name).fields["scale"].content){
                    return false;
                }
            }
        }
    }

    // filter shape must be same, filter num can different, attribute must be same
    auto golden_fusion_node = fusion_tree_.fusion_nodes[0];
    int input_num = golden_fusion_node->input_edges.size();
    std::vector<int64_t> golden_w_shape = data->shapes[golden_fusion_node->input_edges[1].edge->GetId()].dims;
    output_channels_.push_back(golden_w_shape[0]);
    golden_w_shape.erase(golden_w_shape.begin());
    auto golden_conv_param = *((ConvParam*)golden_fusion_node->param);
    for(size_t j=1; j< conv_num_; j++){
        auto f_node = fusion_tree_.fusion_nodes[j];
        if(input_num != f_node->input_edges.size()){
            return false;
        }

        auto conv_param = *((ConvParam*)golden_fusion_node->param);
        if(!(golden_conv_param == conv_param)){
            return false;
        }
        auto w_shape = data->shapes[f_node->input_edges[1].edge->GetId()].dims;
        output_channels_.push_back(w_shape[0]);
        w_shape.erase(w_shape.begin());
        if(w_shape != golden_w_shape){
            return false;
        }

    }

    return true;
};

ppl::common::RetCode BatchConvFusion::FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto quants = options.quants;
    auto tensor_params = &(options.args->quant_info.tensor_params);

    auto golden_fusion_node = fusion_tree_.fusion_nodes[0];
    for(size_t i=1; i< golden_fusion_node->input_edges.size(); i++){
        auto obj_edge_id = golden_fusion_node->input_edges[i].edge->GetId();
        // reshape
        auto shape_ori = options.tensors->find(obj_edge_id)->second->GetShape();
        auto obj_edge_name = golden_fusion_node->input_edges[i].edge->GetName();
        shape_ori->SetDim(0, std::accumulate(output_channels_.begin(), output_channels_.end(), 0));
        // modify const buffer
        auto fused_constant = data->constants[obj_edge_id];
        for(size_t j=1; j<conv_num_; j++){
            auto fusion_node = fusion_tree_.fusion_nodes[j];
            fusion_node->input_edges[i].keep = false;
            JoinConstantInputToClearList(fusion_node, i);
            fusion_node->input_edges[i].edge->DelConsumer(fusion_tree_.found_node_ids[j]);
            auto fusion_edge_id = fusion_node->input_edges[i].edge->GetId();
            auto edge_constant = data->constants[fusion_edge_id];
            auto fusion_edge_name = fusion_node->input_edges[i].edge->GetName();

            fused_constant.data.Append(edge_constant.data.GetData(), edge_constant.data.GetSize());
            if(tensor_params->find(obj_edge_name) == tensor_params->end()){
                continue;
            }
            tensor_params->at(obj_edge_name).fields["scale"].content += tensor_params->at(fusion_edge_name).fields["scale"].content;
            tensor_params->at(obj_edge_name).fields["zero_point"].content += tensor_params->at(fusion_edge_name).fields["zero_point"].content;
            tensor_params->at(obj_edge_name).fields["tensor_max"].content += tensor_params->at(fusion_edge_name).fields["tensor_max"].content;
            tensor_params->at(obj_edge_name).fields["tensor_min"].content += tensor_params->at(fusion_edge_name).fields["tensor_min"].content;
        }
        data->constants[obj_edge_id] = fused_constant;
    }

    return RC_SUCCESS;
};

ppl::common::RetCode BatchConvFusion::InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();

    auto opt_kernel =(CudaOptKernel*)(options.info->kernels[fusion_tree_.found_node_ids[0]].get());
    if (!opt_kernel) {
        LOG(ERROR) <<"create Kernel failed: oom";
        return RC_UNSUPPORTED;
    }
    CudaConvParam* param = (CudaConvParam*)opt_kernel->GetParam();
    opt_kernel->Init(options);
    param->extra_param.fuse_info.output_channels = output_channels_;
    if(act_type_.size()){
        param->extra_param.fuse_info.types.emplace_back(act_type_);
        auto act_kernel = (CudaOptKernel*)(options.info->kernels[fusion_tree_.found_node_ids[conv_num_]].get());
        void* temp_param = nullptr;
        act_kernel->CopyParam(temp_param);
        param->extra_param.fuse_info.fuse_attrs.emplace_back(temp_param);
        param->extra_param.fuse_info.input_inds.emplace_back(1);
    }
    // ((BatchConvOp* )(opt_kernel.get()))->SetOutputChannels(output_channels_);
    return RC_SUCCESS;
};



}}} // namespace ppl::nn::cuda
