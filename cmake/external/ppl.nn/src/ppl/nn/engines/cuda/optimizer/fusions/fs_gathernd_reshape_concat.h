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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_GATHERND_RESHAPE_CONCAT_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_GATHERND_RESHAPE_CONCAT_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/maca_fusion.h"

namespace ppl { namespace nn { namespace cuda {


// only support fp16
class GatherNDReshapeConcatFusion : public MacaFusion {
public:
    GatherNDReshapeConcatFusion(){
        fusion_tree_.node_types = {"GatherND", "Reshape", "Concat"};
        fusion_tree_.connect_infos = {{0,1},{1,2}};
        fusion_type_ = "FusedGatherNDReshapeConcat";
    }
    bool CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) override;
};

class GatherNDReshapeConcatBridgeFusion : public MacaFusion {
public:
    GatherNDReshapeConcatBridgeFusion(){
        fusion_tree_.node_types = {"FusedGatherNDReshapeConcat", "Bridge"};
        fusion_tree_.connect_infos = {{0,1}};
        fusion_type_ = "FusedGatherNDReshapeConcatBridge";
    }
    bool CanFuseTreeNodes(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode FilteNodesInfo(ir::Node* node, const OptKernelOptions& options) override;
    ppl::common::RetCode InsertNodeOptKernel(ir::Node* node, const OptKernelOptions& options) override;
};

class GatherNDReshapeConcatFusion_Set : public Fusion{
    ppl::common::RetCode FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) override;
    GatherNDReshapeConcatFusion base_fusion_;
    GatherNDReshapeConcatBridgeFusion base_bridge_fusion_;
};


}}} // namespace ppl::nn::cuda

#endif
