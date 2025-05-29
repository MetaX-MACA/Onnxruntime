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

#include "ppl/nn/engines/cuda/optimizer/fusions/maca_fusion.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {


bool MacaFusion::need_parse_env_ = true;
std::map<std::string, int> MacaFusion::fuse_env_info_ = std::map<std::string, int>{
        {"FusedGatherNDReshapeConcat",0},
        {"FusedGatherNDReshapeConcatBridge",0},
        {"FusedAddLayerNorm", 1},
        {"BatchGemm",1},
        {"BatchGemmReshape",1},
        {"FusedConcatBridge",1},
        {"FusedMatMulSplit",1},
        {"FusedGatherReshapeAdd",1},
        {"BatchConv",1},
        {"BatchMatMulReshape",1},
        {"FusedMatMulDiv",1}
    };

int MacaFusion::ParseEnv(){
    const char* env_var = std::getenv("MACART_GRAPH_FUSION");
    if (!env_var) {
        LOG(DEBUG) << "Environment variable MACART_GRAPH_FUSION is not set. Will use default value" ;
        return 0;
    }

    std::string env_str(env_var);

    std::istringstream env_stream(env_str);
    std::string pair;

    while (std::getline(env_stream, pair, ';')) {
        if (pair.empty()) continue;

        size_t colon_pos = pair.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = pair.substr(0, colon_pos);
            int value = std::stoi(pair.substr(colon_pos + 1));

            if (!key.empty() && key[0] == '-') {
                key = key.substr(1);
            }

            fuse_env_info_[key] = value;
        }
    }

    for (const auto& kv : fuse_env_info_) {
        LOG(DEBUG) <<"env -> " << kv.first << ": " << kv.second ;
    }
    return fuse_env_info_.size();
}

std::string MacaFusion::GetFusedName(){
    std::string fused_node_name = "";
    for(auto fusion_node : fusion_tree_.fusion_nodes){
        if( fusion_node->node->GetType().name == "Bridge"){
            continue;
        }
        fused_node_name = fused_node_name  + "" + fusion_node->node->GetName() + " + ";
    }
    if (fused_node_name.length() >= 3) {
        fused_node_name.erase(fused_node_name.length() - 3);
    }
    return fused_node_name;
}

bool MacaFusion::SupportNodeDataType(ir::Node* node, const OptKernelOptions& options, std::vector<ppl::common::datatype_t> data_types){
    auto input_edge_id = node->GetInput(0);
    auto input_type = (*options.tensors->find(input_edge_id)->second->GetShape()).GetDataType();
    if(std::find(data_types.begin(), data_types.end(), input_type) == data_types.end()){
        return false;
    }
    auto output_edge_id = node->GetOutput(0);
    auto output_type = (*options.tensors->find(output_edge_id)->second->GetShape()).GetDataType();
    if(input_type != output_type){
        return false;
    }
    return true;
}
TensorShape* MacaFusion::GetConstantShape(const OptKernelOptions& options, edgeid_t constant_edged_id){
    return options.tensors->find(constant_edged_id)->second->GetShape();
}

void MacaFusion::CheckConectEdgeStatus(const OptKernelOptions& options){
    auto topo = options.graph->topo.get();
    //delete parell node same input edge
    for(int i = 1; i < fusion_tree_.connect_edge_ids.size(); i++){
        auto connect_pair = fusion_tree_.connect_infos[i];
        if(connect_pair.first == -1){
            auto remove_edge_id = fusion_tree_.connect_edge_ids[i];
            auto fusion_node = fusion_tree_.fusion_nodes[connect_pair.second];
            for(size_t j=0; j<fusion_node->input_edges.size(); j++){
                if(remove_edge_id == fusion_node->input_edges[j].edge->GetId()){
                    fusion_node->input_edges[j].keep = false;
                }
            }
        }
    }

    // delete unused connect edge
    std::map<int32_t, bool> connect_edge_can_delete_status;
    for(size_t i = 0; i < fusion_tree_.connect_edge_ids.size(); i++){
        if(fusion_tree_.connect_infos[i].first == -1){
            continue;
        }
        auto connect_edge_id = fusion_tree_.connect_edge_ids[i];
        auto edge = topo->GetEdge(connect_edge_id);
        auto producer_id = edge->GetProducer();
        if(connect_edge_can_delete_status.find(connect_edge_id) != connect_edge_can_delete_status.end()){
            continue;
        }
        if(!fusion_tree_.IsIncludeNodeId(producer_id)){
            connect_edge_can_delete_status[connect_edge_id] = false;
            continue;
        }
        auto it = edge->CreateConsumerIter();
        for (; it.IsValid(); it.Forward()) {
            auto consumer_node_id = it.Get();
            if(!fusion_tree_.IsIncludeNodeId(consumer_node_id)){
                connect_edge_can_delete_status[connect_edge_id] = false;
                break;
            }
        }
        if(!it.IsValid()){
            connect_edge_can_delete_status[connect_edge_id] = true;
        }
    }

    for(int i = 0; i < fusion_tree_.fusion_nodes.size(); i++){
        std::shared_ptr<FusionNodeInfo> fusion_node = fusion_tree_.fusion_nodes[i];
        for(size_t j=0; j<fusion_node->input_edges.size(); j++){
            auto edge_id = fusion_node->input_edges[j].edge->GetId();
            if(fusion_tree_.IsIncludeConnectEdgeId(edge_id) && i !=0 ){
                // connect id only keep output edge
                fusion_node->input_edges[j].keep = false;
            }
        }
        for(size_t j=0; j<fusion_node->output_edges.size(); j++){
            auto edge_id = fusion_node->output_edges[j].edge->GetId();
            if(connect_edge_can_delete_status.find(edge_id) != connect_edge_can_delete_status.end()){
                fusion_node->output_edges[j].keep = !connect_edge_can_delete_status[edge_id];
            }
        }
    }

    for(int i=1; i<fusion_tree_.found_node_ids.size(); i++){
        need_clear_node_ids_.push_back(fusion_tree_.found_node_ids[i]);
    }
    for(auto p:connect_edge_can_delete_status){
        if(p.second){
            need_clear_edge_ids_.push_back(p.first);
        }
    }
}

ppl::common::RetCode MacaFusion::ClearUnusedNodeEdge(const OptKernelOptions& options){
    auto topo = options.graph->topo.get();
    if(debug_){
        std::cout << fusion_type_ << " clear edge id ";
        for(auto p : need_clear_edge_ids_){
            std::cout << p << " ";
        }
        std::cout << std::endl << fusion_type_ << " clear node id ";
        for(auto p : need_clear_node_ids_){
          std::cout << p << " ";
        }
        std::cout << std::endl;
    }
    for(auto p : need_clear_edge_ids_){
        topo->DelEdge(p);
    }
    for(auto p : need_clear_node_ids_){
        topo->DelNode(p);
        options.info->kernels.erase(p);
    }
    return RC_SUCCESS;
};

bool MacaFusion::FindTreeNodes(ir::Node* node, const OptKernelOptions& options){
    if(debug_){
        std::cout << "finding "  << fusion_type_ << "; Node name " << node->GetName() << std::endl;
    }
    auto topo = options.graph->topo.get();

    // first node
    std::shared_ptr<FusionNodeInfo> fusion_node = ParseNodeToFusionNode(node, options);
    fusion_tree_.fusion_nodes[0] = fusion_node;
    fusion_tree_.found_node_ids[0] = fusion_node->node->GetId();
    // deal parallel nodes
    std::vector<std::string> parallel_node_types;
    for(auto connect_pair : fusion_tree_.connect_infos){
        if(connect_pair.first == -1){
            parallel_node_types.push_back(fusion_tree_.node_types[connect_pair.second]);
        }else{
            break;
        }
    }

    if(parallel_node_types.empty() == false){
        int32_t parent_edge_id = -1;
        std::vector<int32_t> parallel_node_id(parallel_node_types.size(), -1);
        for(int i = 0; i < node->GetInputCount(); i++){
            parallel_node_id = std::vector<int32_t>(parallel_node_types.size(), -1);
            parallel_node_id[0] = node->GetId();
            auto edge_id = node->GetInput(i);
            auto edge = topo->GetEdge(edge_id);
            fusion_tree_.connect_edge_ids[0] = edge_id;
            if(edge->CalcConsumerCount() < parallel_node_types.size()){
                continue;
            }else{
                for (auto it = edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                    auto node_id = it.Get();
                    if(node_id  == parallel_node_id[0]){
                        continue;
                    }
                    auto sub_node = topo->GetNode(node_id);
                    auto type_name = node->GetType().name;
                    bool set_node_id = false;
                    for(int j = 0; j < parallel_node_id.size(); j++){
                        if(parallel_node_types[j] == type_name && parallel_node_id[j] == -1 && RuleJudge(sub_node, options)){
                            parallel_node_id[j] = node_id;
                            fusion_tree_.connect_edge_ids[j] = edge_id;
                            set_node_id = true;
                            break;
                        }
                    }
                    // if(!set_node_id && std::count(parallel_node_types.begin(), parallel_node_types.end(), type_name)){
                    //     return false;
                    // }
                }
            }
            if(std::count(parallel_node_id.begin(), parallel_node_id.end(), -1) == 0){
                break;
            }
        }
        if(std::count(parallel_node_id.begin(), parallel_node_id.end(), -1) != 0 ||
            std::count(parallel_node_id.begin(), parallel_node_id.end(), fusion_tree_.found_node_ids[0]) == 0
        ){
            return false;
        }else{
            auto fusion_nodes_iter = fusion_tree_.fusion_nodes.begin() + 1;
            auto fusion_node_ids_iter = fusion_tree_.found_node_ids.begin() + 1;
            for( auto p : parallel_node_id){
                if(p != fusion_tree_.found_node_ids[0]){
                    (*fusion_node_ids_iter) = p;
                    fusion_node_ids_iter++;
                    (*fusion_nodes_iter) = ParseNodeToFusionNode(topo->GetNode(p), options);
                    fusion_nodes_iter++;
                }
            }
        }

    }

    //
    for(int connect_id = parallel_node_types.size(); connect_id<fusion_tree_.connect_infos.size(); connect_id++){
        auto connect_pair = fusion_tree_.connect_infos[connect_id];
        int32_t src_node_fusion_id = connect_pair.first;
        auto src_fusion_node = fusion_tree_.fusion_nodes[connect_pair.first];
        int32_t dst_node_fusion_id = connect_pair.second;
        auto dst_fusion_node = fusion_tree_.fusion_nodes[connect_pair.second];

        if(!src_fusion_node && !dst_fusion_node){
            return false;
        }

        if(src_fusion_node && !dst_fusion_node){
            ir::Node* current_node = src_fusion_node->node;
            std::string dst_node_type = fusion_tree_.node_types[dst_node_fusion_id];
            for(int i = 0; i < current_node->GetOutputCount(); i++){
                auto output_edge_id = current_node->GetOutput(i);
                auto output_edge = topo->GetEdge(output_edge_id);
                for (auto it = output_edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                    auto dst_node_id = it.Get();
                    auto dst_node = topo->GetNode(dst_node_id);
                    // for edge consumer has two same type node
                    if(dst_node->GetType().name == dst_node_type &&
                        !fusion_tree_.IsIncludeNodeId(dst_node_id)){
                        fusion_tree_.fusion_nodes[dst_node_fusion_id] = ParseNodeToFusionNode(dst_node, options);
                        fusion_tree_.found_node_ids[dst_node_fusion_id] = dst_node_id;
                        fusion_tree_.connect_edge_ids[connect_id] = output_edge_id;
                        break;
                    }
                }
                if(fusion_tree_.connect_edge_ids[connect_id] != -1){
                    break;
                }
            }
            if(!fusion_tree_.fusion_nodes[dst_node_fusion_id]){
                return false;
            }
            continue;
        }

        if(!src_fusion_node && dst_fusion_node){
            ir::Node* current_node = dst_fusion_node->node;
            std::string src_node_type = fusion_tree_.node_types[src_node_fusion_id];
            for(int i = 0; i < current_node->GetInputCount(); i++){
                auto input_edge_id = current_node->GetInput(i);
                auto input_edge = topo->GetEdge(input_edge_id);
                auto src_node_id = input_edge->GetProducer();
                auto src_node = topo->GetNode(src_node_id);
                if(src_node->GetType().name == src_node_type &&
                    !fusion_tree_.IsIncludeNodeId(src_node_id)){
                    fusion_tree_.fusion_nodes[src_node_fusion_id] = ParseNodeToFusionNode(src_node, options);
                    fusion_tree_.found_node_ids[src_node_fusion_id] = src_node_id;
                    fusion_tree_.connect_edge_ids[connect_id] = input_edge_id;
                    break;
                }
            }
            if(!fusion_tree_.fusion_nodes[src_node_fusion_id]){
                return false;
            }
            continue;
        }
    }

    for(auto p : fusion_tree_.found_node_ids){
        if(p == -1){
            return false;
        }
    }
    for(auto p : fusion_tree_.connect_edge_ids){
        if(p == -1){
            return false;
        }
    }
    CheckConectEdgeStatus(options);
    if(debug_){
        std::cout << "found "  << fusion_type_ << "; Node name " << node->GetName() << std::endl;
    }
    return true;
}

/*
Input edge :
1 : consumer replace other node id to nodes[0]

Output edge:
1 : if(comsumer include any node id) remove node id
2 : producer set to nodes[0]
*/
ppl::common::RetCode MacaFusion::FuseTreeToNode(ir::Node* node, const OptKernelOptions& options){
    if(debug_){
        fusion_tree_.PrintInfo();
    }
    std::string fused_name = GetFusedName();
    LOG(DEBUG) << "Fused nodes " << fused_name;
    node->SetName(fused_name);
    int32_t input_ids = 0;
    int32_t output_ids = 0;
    for(int i = 0; i < fusion_tree_.fusion_nodes.size(); i++){
        std::shared_ptr<FusionNodeInfo> fusion_node = fusion_tree_.fusion_nodes[i];
        // modify input consumer
        for(size_t j=0; j<fusion_node->input_edges.size(); j++){
            FusionEdgeInfo fuse_edge_info = fusion_node->input_edges[j];
            auto edge_id = fuse_edge_info.edge->GetId();
            if(!fuse_edge_info.keep){
                continue;
            }
            if(input_ids < node->GetInputCount()){
                node->ReplaceInput(node->GetInput(input_ids++), edge_id);
            }else{
                node->InsertInput(input_ids++, edge_id);
            }
            std::vector<int32_t> remove_consumers;
            for (auto it = fuse_edge_info.edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                if(fusion_tree_.IsIncludeNodeId(it.Get())){
                    remove_consumers.push_back(it.Get());
                }
            }
            for(auto p : remove_consumers){
                fuse_edge_info.edge->DelConsumer(p);
            }
            fuse_edge_info.edge->AddConsumer(fusion_tree_.found_node_ids[0]);
        }

        for(size_t j=0; j<fusion_node->output_edges.size(); j++){
            FusionEdgeInfo fuse_edge_info = fusion_node->output_edges[j];
            auto edge_id = fuse_edge_info.edge->GetId();
            if(!fuse_edge_info.keep){
                continue;
            }
            if(output_ids < node->GetOutputCount()){
                node->ReplaceOutput(node->GetOutput(output_ids++), edge_id);
            }else{
                node->InsertOutput(output_ids++, edge_id);
            }
            std::vector<int32_t> remove_consumers;
            for (auto it = fuse_edge_info.edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                if(fusion_tree_.IsIncludeNodeId(it.Get())){
                    remove_consumers.push_back(it.Get());
                }
            }
            for(auto p : remove_consumers){
                fuse_edge_info.edge->DelConsumer(p);
            }
            fuse_edge_info.edge->SetProducer(fusion_tree_.found_node_ids[0]);
        }
    }
    LOG(DEBUG) << "input ids " << input_ids << "; output ids " << output_ids;
    return RC_SUCCESS;
}

ppl::common::RetCode MacaFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options){
    // set node rule
    if(InitNodeRule(node, options) == false){
        return RC_UNSUPPORTED;
    }

    // fusion control by env
    if(GetEnvValue(fusion_type_, 0) == 0){
        return RC_UNSUPPORTED;
    }
    ResetFusionInfo();
    RetCode status = RC_SUCCESS;
    if(FindTreeNodes(node,options) && CanFuseTreeNodes(node, options)){
        status = FilteNodesInfo(node, options);
        if(status != RC_SUCCESS){
            LOG(ERROR) << "filte nodes info failed";
            return status;
        }
        status = FuseTreeToNode(node, options);
        if(status != RC_SUCCESS){
            LOG(ERROR) << "fuse tree to node failed";
            return status;
        }
        status = InsertNodeOptKernel(node, options);
        if(status != RC_SUCCESS){
            LOG(ERROR) << "insert node optkernel failed";
            return status;
        }
        status = ClearUnusedNodeEdge(options);
        if(status != RC_SUCCESS){
            LOG(ERROR) << "clear unsed node edge failed";
            return status;
        }
    }else{
        return RC_UNSUPPORTED;
    }
    if(status == RC_SUCCESS && debug_){
        PrintFusedResult(node,options);
    }
    return status;
}

void MacaFusion::PrintFusedResult(ir::Node* node, const OptKernelOptions& options){
    std::cout <<"node id : "<< node->GetId() << "; node name : " <<node->GetName()  <<"; node type : "<< node->GetType().name  << std::endl;
    std::cout << "input edges ";
    for(size_t i=0; i< node->GetInputCount(); i++){
        std::cout << node->GetInput(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "output edges ";
    for(size_t i=0; i< node->GetOutputCount(); i++){
        std::cout << node->GetOutput(i) << " ";
    }
    std::cout << std::endl;
}

std::shared_ptr<FusionNodeInfo> MacaFusion::ParseNodeToFusionNode(ir::Node* node, const OptKernelOptions& options){
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto param = options.graph->data.get()->attrs[node->GetId()].get();

    std::shared_ptr<FusionNodeInfo> node_info = std::make_shared<FusionNodeInfo>();
    node_info->node = node;
    node_info->param = param;

    for(int i = 0; i < node->GetInputCount(); i++){
        auto input_edge_id = node->GetInput(i);
        auto edge = topo->GetEdge(input_edge_id);
        if(data->constants.find(input_edge_id) != data->constants.end()){
            node_info->input_edges.push_back(FusionEdgeInfo(true, edge, true));
        }else{
            node_info->input_edges.push_back(FusionEdgeInfo(true, edge, false));
        }
    }

    for(int i = 0; i < node->GetOutputCount(); i++){
        auto output_edge_id = node->GetOutput(i);
        auto edge = topo->GetEdge(output_edge_id);
        node_info->output_edges.push_back(FusionEdgeInfo(true, edge, false));
    }
    return node_info;
}


ir::Node* MacaFusion::FuseWithNextNode(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options, bool append_netxnode_input) {
    auto topo = options.graph->topo.get();
    auto node_id = node->GetId();
    auto next_node_id = nextnode->GetId();

    auto connect_edge_id = node->GetOutput(0);
    for(size_t i=0; i<node->GetOutputCount(); i++){
        auto edge_id = node->GetOutput(i);
        auto edge  = topo->GetEdge(edge_id);
        for(auto  it= edge->CreateConsumerIter();  it.IsValid(); it.Forward()){
            if(it.Get() == next_node_id){
                connect_edge_id = edge_id;
                break;
            }
        }
    }

    auto connect_edge = topo->GetEdge(connect_edge_id);
    if(connect_edge->CalcConsumerCount() != 1){
        return nullptr;
    }


    for(size_t i=0; i< nextnode->GetOutputCount(); i++){
        auto edge_id = node->GetOutput(i);
        auto edge  = topo->GetEdge(edge_id);
        edge->SetProducer(node_id);
        if(i == 0){
            node->ReplaceOutput(connect_edge_id, edge_id);
        }else{
            node->AddOutput(edge_id);
        }
    }

    if(append_netxnode_input){
        for(int i = 0; i < nextnode->GetInputCount(); i++){
            auto input_edge_id = nextnode->GetInput(i);
            if(input_edge_id == INVALID_EDGEID || input_edge_id == connect_edge_id)
                continue;
            auto input_edge = topo->GetEdge(input_edge_id);
            input_edge->AddConsumer(node->GetId());
            input_edge->DelConsumer(nextnode->GetId());
            node->AddInput(input_edge_id);
        }
    }

    return node;
}

bool MacaFusion::JoinConstantInputToClearList(std::shared_ptr<FusionNodeInfo> fusion_node, int input_idx){

    if(input_idx >= fusion_node->input_edges.size()){
        LOG(ERROR) << "JoinConstantInputToClearList failed, input_idx > input_num";
        return false;
    }
    if(fusion_node->input_edges[input_idx].is_constant == false){
        LOG(ERROR) << "JoinConstantInputToClearList failed, not constant edge";
        return false;
    }

    auto consumer_node_id = fusion_node->node->GetId();
    auto edge = fusion_node->input_edges[input_idx].edge;
    edge->DelConsumer(consumer_node_id);
    if(edge->CalcConsumerCount() == 0){
        need_clear_edge_ids_.push_back(edge->GetId());
    }
    return true;
}

}}} // namespace ppl::nn::cuda
