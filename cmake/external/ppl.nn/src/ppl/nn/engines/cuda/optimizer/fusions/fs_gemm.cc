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

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_gemm.h"

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/engines/cuda/params/gemm_extra_param.h"
#include "ppl/nn/common/logger.h"


using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

bool GemmFusion::CanFuseMatmulAdd(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options, uint32_t flag) {
#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
    if (matmul_fuse_type.find(nextnode->GetType().name) != matmul_fuse_type.end()
        || fuse_type.find(nextnode->GetType().name) != fuse_type.end()) {
        auto inputEdgeId = node->GetInput(0);
        if(edgeIsQuant(inputEdgeId, options) == true){
            return false; // now, not support matmul+add int8 fuse
        }
        auto matmul_input1_edegid = node->GetInput(1);
        auto matmul_input1_ref = options.graph->data->constants.find(matmul_input1_edegid);
        if(matmul_input1_ref == options.graph->data->constants.end()){
            return false;
        }
        auto matmul_input1_shape = options.tensors->find(matmul_input1_edegid)->second->GetShape();
        if(matmul_input1_shape->GetDimCount() > 2){
            return false;
        }
        if(matmul_fuse_type.find(nextnode->GetType().name) != matmul_fuse_type.end()){
            if(edgeIsQuant(nextnode->GetOutput(0), options) == true){
                return false; // now, not support matmul+add int8 fuse
            }
            auto matmul_output_edegid = node->GetOutput(0);
            int add_input1_edegid = -1;
            for(int i = 0; i < nextnode->GetInputCount(); i++){
                if(matmul_output_edegid != nextnode->GetInput(i)){
                    add_input1_edegid = nextnode->GetInput(i);
                    break;
                }
            }
            if(0 == flag){
                auto add_input1_ref = options.graph->data->constants.find(add_input1_edegid);
                if(add_input1_ref == options.graph->data->constants.end()){
                    return false;
                }
                auto add_input1_shape = options.tensors->find(add_input1_edegid)->second->GetShape();
                if(add_input1_shape->GetDimCount() <= 1){
                    return true;
                }
            } else if(1 == flag){
                auto add_input1_shape = options.tensors->find(add_input1_edegid)->second->GetShape();
                auto matmul_output_shape = options.tensors->find(matmul_output_edegid)->second->GetShape();
                if(add_input1_shape->GetDimCount() != matmul_output_shape->GetDimCount())
                    return false;
                for(int i = 0; i < matmul_output_shape->GetDimCount(); i++){
                    if(matmul_output_shape->GetDim(i) != add_input1_shape->GetDim(i)){
                        return false;
                    }
                }
                return true;
            }
        }
        //return true;
    }
    return false;
#else
    return false;
#endif
}


bool GemmFusion::CanFuse(ir::Node* nextnode, const OptKernelOptions& options, uint32_t flag) {
#if defined(PPLNN_USE_MACA) || !defined(PPLNN_USE_DNN)
    if (fuse_type.find(nextnode->GetType().name) != fuse_type.end()) {
        const TensorShape& out_shape = *options.tensors->find(nextnode->GetOutput(0))->second->GetShape();
        if(out_shape.GetDataType() != DATATYPE_FLOAT16){
        #if defined(PPLNN_USE_MACA) && defined(PPLNN_USE_DNN)
            return false;
        #endif
        }
        return true;
    }
#endif
    return false;
}

RetCode GemmFusion::FuseGemmWithNextNode(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options) {
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

RetCode GemmFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto node_id = node->GetId();
    auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node_id].get());
    CudaGemmParam* param = (CudaGemmParam*)opt_kernel->GetParam();

    for (uint32_t i = 0; i < 1 && node->GetOutputCount() == 1; ++i) {
        auto edge_id = node->GetOutput(0);
        auto edge = topo->GetEdge(edge_id);
        if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
            break;
        }

        auto iter = topo->GetEdge(edge_id)->CreateConsumerIter();
        if (!iter.IsValid()) {
            break;
        }

        auto nextnode_id = iter.Get();
        auto nextnode = topo->GetNode(nextnode_id);

        iter.Forward();
        if (iter.IsValid()) { // Do not fuse if the edge has more than one consumer
            break;
        }

        if (node->GetType().name == "Gemm" && CanFuse(nextnode, options, i)) {
            LOG(DEBUG) << "Fuse node[" << node->GetName() << "] and nextnode[" << nextnode->GetName() << "]";
            param->extra_param.fuse_info.types.emplace_back(nextnode->GetType().name);
            param->extra_param.fuse_info.input_inds.emplace_back(node->GetInputCount());
            if (nextnode->GetType().name != "Clip") {
                auto next_kernel = (CudaOptKernel*)(options.info->kernels[nextnode_id].get());
                void* temp_param = nullptr;
                next_kernel->CopyParam(temp_param);
                param->extra_param.fuse_info.fuse_attrs.emplace_back(std::move(temp_param));
            } else {
                auto clip_param = new CudaClipParam();
                auto min_iter = data->constants.find(nextnode->GetInput(1));
                if (min_iter != data->constants.end()) {
                    clip_param->min_value = *(float*)(min_iter->second.data.GetData());
                }
                auto max_iter = data->constants.find(nextnode->GetInput(2));
                if (max_iter != data->constants.end()) {
                    clip_param->max_value = *(float*)(max_iter->second.data.GetData());
                }
                param->extra_param.fuse_info.fuse_attrs.emplace_back((void*)clip_param);
            }
            options.info->kernels.erase(nextnode_id);
            FuseGemmWithNextNode(node, nextnode, options);
        }
        if (node->GetType().name == "MatMul" && CanFuseMatmulAdd(node, nextnode, options, 0)) {
            if(param->extra_param.fuse_info.types.size() > 0){
                return RC_UNSUPPORTED;  // Only support fp16 MatMul+Add+Act(opt)
            }
            LOG(DEBUG) << "#### Fuse node[" << node->GetName() << "] and nextnode[" << nextnode->GetName() << "]";
            param->extra_param.fuse_info.types.emplace_back(nextnode->GetType().name);
            param->extra_param.fuse_info.input_inds.emplace_back(node->GetInputCount());
            options.info->kernels.erase(nextnode_id);
            node->SetName(node->GetName() + "_fused_" + nextnode->GetName());
            FuseGemmWithNextNode(node, nextnode, options);
            edge_id = node->GetOutput(0);
            if(topo->GetEdge(edge_id)->CalcConsumerCount() > 1){
                break;
            }
            auto iter = topo->GetEdge(edge_id)->CreateConsumerIter();
            if (!iter.IsValid()) {
                break;
            }
            auto nextnode_id = iter.Get();
            auto nextnode = topo->GetNode(nextnode_id);
            if(edgeIsQuant(nextnode->GetOutput(0), options) == true){
                return false; // now, not support matmul+add+act int8 fuse
            }
            if(fuse_type.find(nextnode->GetType().name) != fuse_type.end()){
                LOG(DEBUG) << "#### Fuse node[" << node->GetName() << "] continue and nextnode[" << nextnode->GetName() << "]";
                param->extra_param.fuse_info.types.emplace_back(nextnode->GetType().name);
                param->extra_param.fuse_info.input_inds.emplace_back(node->GetInputCount());
                options.info->kernels.erase(nextnode_id);
                node->SetName(node->GetName() + "_fused_" + nextnode->GetName());
                FuseGemmWithNextNode(node, nextnode, options);
            }else if(nextnode->GetType().name == "Add" && CanFuseMatmulAdd(node, nextnode, options, 1)){
            #if SUPPORT_MATMUL_ADD_ADD_FUSE
                LOG(DEBUG) << "#### Fuse node[" << node->GetName() << "] continue and nextnode[" << nextnode->GetName() << "]";
                param->extra_param.fuse_info.types.emplace_back(nextnode->GetType().name);
                param->extra_param.fuse_info.input_inds.emplace_back(node->GetInputCount());
                options.info->kernels.erase(nextnode_id);
                node->SetName(node->GetName() + "_fused_" + nextnode->GetName());
                FuseGemmWithNextNode(node, nextnode, options);
            #endif
            }
        }
    }
    return RC_SUCCESS;
}


ppl::common::RetCode GemmFusion_Set::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    auto status = 0;
    if(node->GetType().name == "MatMul"){
        status = matmul_split_fusion_.FuseNode(node, reliable, options);
        if(status == RC_SUCCESS){
            return RC_SUCCESS;
        }
        status = batchmatmul_reshape_4_fusion_.FuseNode(node, reliable, options);
        if(status == RC_SUCCESS){
            return RC_SUCCESS;
        }
        status = batchmatmul_reshape_2_fusion_.FuseNode(node, reliable, options);
        if(status == RC_SUCCESS){
            return RC_SUCCESS;
        }
        status = matmul_div_.FuseNode(node, reliable, options);
    }
    status = batchgemm_base_fusion_.FuseNode(node, reliable, options);
    if(status == RC_SUCCESS){
        status = batchgemm_reshape_fusion_.FuseNode(node, reliable, options);
        return RC_SUCCESS;
    }
    status = gemm_fusion_.FuseNode(node, reliable, options);
    return RC_SUCCESS;
};

}}} // namespace ppl::nn::cuda
