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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_MATMUL_SPLIT_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_MATMUL_SPLIT_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/maca_fusion.h"

namespace ppl { namespace nn { namespace cuda {

// only support fp16
class MatMulSplitFusion : public MacaFusion {
public:
    MatMulSplitFusion(){
        fusion_tree_.node_types = {"MatMul", "Split", "Reshape","Reshape","Reshape"};
        fusion_tree_.connect_infos = {{0,1}, {1,2}, {1,3}, {1,4}};
        fusion_type_ = "FusedMatMulSplit";
    };
    bool CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) override;
private:
    std::vector<int64_t> output_shape_;
};

class BatchMatMulFusion : public MacaFusion {
public:
    BatchMatMulFusion(int32_t matmul_num = 2, std::string append_type = "Reshape"):matmul_num_(matmul_num),append_type_(append_type) {
        for(int i=0; i<matmul_num_; i++){
            fusion_tree_.node_types.push_back("MatMul");
            fusion_tree_.connect_infos.push_back({-1,i});
        }
        if(append_type_.size()){
            for(int i=0; i<matmul_num_; i++){
                fusion_tree_.node_types.push_back(append_type_);
                fusion_tree_.connect_infos.push_back({i,i+matmul_num_});
            }
        }
        fusion_type_ = "BatchMatMul" + append_type_;
    };
    bool CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) override;
    bool InitNodeRule(ir::Node* node, const OptKernelOptions& options) override;
    bool RuleJudge(ir::Node* node, const OptKernelOptions& options) override;
private:
    int32_t matmul_num_;
    std::string append_type_ = "";
    std::vector<int64_t> output_shape_;

    ppl::common::TensorShape rule_shape_;
};

class FusedMatMulDiv : public MacaFusion{
public:
    FusedMatMulDiv(){
        fusion_tree_.node_types = {"MatMul", "Div"};
        fusion_tree_.connect_infos = {{0,1}};
        fusion_type_ = "FusedMatMulDiv";
    };
    bool CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) override;
};

}}} // namespace ppl::nn::cuda

#endif
