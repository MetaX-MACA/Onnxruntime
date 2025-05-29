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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_GEMM_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_GEMM_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/fusion.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_batch_gemm.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_matmul_split.h"

#define SUPPORT_MATMUL_ADD_ADD_FUSE     0

namespace ppl { namespace nn { namespace cuda {

class GemmFusion : public Fusion {
public:
    ppl::common::RetCode FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) override;

private:
    bool CanFuse(ir::Node* nextnode, const OptKernelOptions& options, uint32_t flag);
    bool CanFuseMatmulAdd(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options, uint32_t flag);
    ppl::common::RetCode FuseGemmWithNextNode(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options);

private:
#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
    std::set<std::string> fuse_type{"Relu",
                                    "Gelu",
                                    "HardSwish"};
#else
    std::set<std::string> fuse_type{"Relu",
                                    "Clip",
                                    "Sigmoid"};
#endif
    std::set<std::string> matmul_fuse_type{"Add"};

    static bool edgeIsQuant(edgeid_t edgeId, const OptKernelOptions& options){
        if(options.quants->at(edgeId).bit_width == 8){
            return true;
        }else{
            return false;
        }
    }
};

class GemmFusion_Set: public Fusion{
    ppl::common::RetCode FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) override;
    GemmFusion gemm_fusion_;
    BatchGemmFusion batchgemm_base_fusion_;
    BatchGemmReshapeFusion batchgemm_reshape_fusion_;
    MatMulSplitFusion matmul_split_fusion_;
    FusedMatMulDiv matmul_div_;
    BatchMatMulFusion batchmatmul_reshape_2_fusion_;
    BatchMatMulFusion batchmatmul_reshape_4_fusion_ = BatchMatMulFusion(4,"Reshape");
};

}}} // namespace ppl::nn::cuda

#endif
