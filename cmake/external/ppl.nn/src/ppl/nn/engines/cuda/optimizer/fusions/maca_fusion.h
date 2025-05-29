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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_MACA_FUSION_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_MACA_FUSION_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/fusion.h"
#include "ppl/nn/ir/attr.h"
using namespace ppl::nn::ir;
namespace ppl { namespace nn { namespace cuda {

struct FusionEdgeInfo{
    bool keep = true;
    Edge* edge = nullptr;
    bool is_constant = false;
    FusionEdgeInfo(bool keep_i = true, Edge* edge_i=nullptr, bool is_constant_i =false) : keep(keep_i), edge(edge_i), is_constant(is_constant_i)
    {}
};

struct FusionNodeInfo{
    bool keep = true;
    Node* node = nullptr;
    std::vector<FusionEdgeInfo> input_edges;
    std::vector<FusionEdgeInfo> output_edges;
    Attr* param = nullptr;
};

struct FusionTreeInfo{
    // 广度优先方式存储
    std::vector<std::pair<int32_t, int32_t>> connect_infos;
    std::vector<std::string> node_types;

    // 检索并存储节点信息
    std::vector<std::shared_ptr<FusionNodeInfo>> fusion_nodes;
    std::vector<int32_t> found_node_ids;
    std::vector<int32_t> connect_edge_ids;

public:
    bool IsIncludeNodeId(int32_t node_id){
        return std::count(found_node_ids.begin(), found_node_ids.end(), node_id);
    }
    bool IsIncludeConnectEdgeId(int32_t edge_id){
        return std::count(connect_edge_ids.begin(), connect_edge_ids.end(), edge_id);
    }
    void PrintInfo(){
        for(auto fusion_node : fusion_nodes){
            std::cout << "node id "<< fusion_node->node->GetId() << "; node name : " << fusion_node->node->GetName()  <<"; node type : "<< fusion_node->node->GetType().name  <<"; keep status : " << fusion_node->keep << std::endl;
            for(auto p :fusion_node->input_edges){
                std::cout << "input edge id : "<< p.edge->GetId() << "; is constant : "<< p.is_constant  << "; keep stauts : " << p.keep  << std::endl;
            }
            for(auto p :fusion_node->output_edges){
                std::cout << "output edge id : " << p.edge->GetId() << "; is constant : "<< p.is_constant  << "; keep stauts : " << p.keep  << std::endl;
            }
        }


    }
};

class MacaFusion : public Fusion {

public:
    virtual ~MacaFusion() {}
    virtual ppl::common::RetCode FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) override;
    virtual bool CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) {return 0;};
    virtual ppl::common::RetCode FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) {return 0;};
    virtual ppl::common::RetCode InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) {return 0;};
    virtual bool InitNodeRule(ir::Node* node, const OptKernelOptions& options) {return true;};
    virtual bool RuleJudge(ir::Node* node, const OptKernelOptions& options) {return true;};
    static int ParseEnv();
    static int GetEnvValue(std::string env, int default_value = 1){
        if(need_parse_env_){
            ParseEnv();
            need_parse_env_ = false;
        }
        if(fuse_env_info_.find(env) != fuse_env_info_.end()){
            return fuse_env_info_[env];
        }else{
            fuse_env_info_[env] = default_value;
        }
        return default_value;
    }

private:
    bool FindTreeNodes(ir::Node* node, const OptKernelOptions& options);
    ppl::common::RetCode FuseTreeToNode(ir::Node* node, const OptKernelOptions& options);
    void CheckConectEdgeStatus(const OptKernelOptions& options);
    ppl::common::RetCode ClearUnusedNodeEdge(const OptKernelOptions& options);
    void PrintFusedResult(ir::Node* node, const OptKernelOptions& options);

protected:
    std::shared_ptr<FusionNodeInfo> ParseNodeToFusionNode(ir::Node* node, const OptKernelOptions& options);
    bool JoinConstantInputToClearList(std::shared_ptr<FusionNodeInfo> fusion_node, int input_idx);
    std::string GetFusedName();
    TensorShape* GetConstantShape(const OptKernelOptions& options, edgeid_t constant_edged_id);
    bool SupportNodeDataType(ir::Node* node, const OptKernelOptions& options, std::vector<ppl::common::datatype_t> data_types = {ppl::common::DATATYPE_FLOAT16});
    ir::Node* FuseWithNextNode(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options, bool append_netxnode_input = true);
    void ResetFusionInfo(){
        fusion_tree_.fusion_nodes.clear();
        fusion_tree_.fusion_nodes = std::vector<std::shared_ptr<FusionNodeInfo>>(fusion_tree_.node_types.size(), nullptr);
        fusion_tree_.found_node_ids.clear();
        fusion_tree_.found_node_ids = std::vector<int32_t>(fusion_tree_.node_types.size(), -1);
        fusion_tree_.connect_edge_ids.clear();
        fusion_tree_.connect_edge_ids = std::vector<int32_t>(fusion_tree_.connect_infos.size(), -1);
        need_clear_node_ids_.clear();
        need_clear_edge_ids_.clear();
    }

protected:
    std::vector<int32_t> need_clear_node_ids_;
    std::vector<int32_t> need_clear_edge_ids_;

    FusionTreeInfo fusion_tree_;
    std::string fusion_type_="";
    bool debug_ = false;

public:
    // env info
    static bool need_parse_env_ ;
    static std::map<std::string, int> fuse_env_info_;

};



}}} // namespace ppl::nn::cuda

#endif
