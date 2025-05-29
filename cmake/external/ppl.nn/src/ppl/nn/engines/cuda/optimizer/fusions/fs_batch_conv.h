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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_BATCH_CONV_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_BATCH_CONV_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/maca_fusion.h"

namespace ppl { namespace nn { namespace cuda {

// only support fp16
class BatchConvFusion : public MacaFusion {
public:
    BatchConvFusion(size_t conv_num = 2, std::string act_type = ""){
        conv_num_ = conv_num;
        act_type_ = act_type;
        for(int i=0; i<conv_num_; i++){
            fusion_tree_.node_types.push_back("Conv");
            fusion_tree_.connect_infos.push_back({-1,i});
        }
        if(act_type_.size()){
            for(int i=0; i<conv_num_; i++){
                fusion_tree_.node_types.push_back(act_type_);
                fusion_tree_.connect_infos.push_back({i,i+conv_num_});
            }
        }
        fusion_type_ = "BatchConv";
    };
    bool CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) override;

private:
    size_t conv_num_;
    std::string act_type_;
    std::vector<int32_t> output_channels_;
};




}}} // namespace ppl::nn::cuda

#endif
