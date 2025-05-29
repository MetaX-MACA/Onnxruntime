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

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_cast.h"

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/params/onnx/cast_param.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

bool CastFusion::CanFuse(ir::Node* node, ir::Node* prenode) {
    if (prenode->GetInputCount() == 1 && prenode->GetOutputCount() == 1 && node->GetType().name == "Cast" &&
        prenode->GetType().name == "Cast") {
        return true;
    }
    return false;
}

RetCode CastFusion::FuseWithPreviousCast(ir::Node* node, ir::Node* prenode, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto connect_edge_id = node->GetInput(0);
    auto connect_edge = topo->GetEdge(connect_edge_id);
    auto next_edge_id = node->GetOutput(0);
    auto next_edge = topo->GetEdge(next_edge_id);

    for (auto it = next_edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
        auto tempnode_id = it.Get();
        auto tempnode = topo->GetNode(tempnode_id);
        connect_edge->AddConsumer(tempnode_id);
        tempnode->ReplaceInput(next_edge_id, connect_edge_id);
    }

    connect_edge->DelConsumer(node->GetId());
    topo->DelEdge(next_edge->GetId());
    topo->DelNode(node->GetId());
    return RC_SUCCESS;
}

void CastFusion::FuseCastBridge(ir::Node* node, const OptKernelOptions& options)
{
    auto topo = options.graph->topo.get();
    auto node_id = node->GetId();
    auto edge_id = node->GetInput(0);
    auto prenode_id = topo->GetEdge(edge_id)->GetProducer();
    auto prenode = topo->GetNode(prenode_id);
    auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node_id].get());
    ppl::nn::onnx::CastParam* param = (ppl::nn::onnx::CastParam*)opt_kernel->GetParam();
    if(prenode != nullptr && prenode->GetType().name == "Bridge"){
        TensorShape& input_shape = *options.tensors->find(node->GetInput(0))->second->GetShape();
        TensorShape& output_shape = *options.tensors->find(node->GetOutput(0))->second->GetShape();
        input_shape.SetDataType(output_shape.GetDataType());
        auto output_edge_id = node->GetOutput(0);
        if(topo->GetEdge(output_edge_id)->CalcConsumerCount() == 1){
            auto nextnode_id = topo->GetEdge(output_edge_id)->CreateConsumerIter().Get(); // Get Output(0)
            auto nextnode = topo->GetNode(nextnode_id);
            if(nextnode->GetType().name == "Bridge"){
                TensorShape& next_input_shape = *options.tensors->find(nextnode->GetInput(0))->second->GetShape();
                TensorShape& next_output_shape = *options.tensors->find(nextnode->GetOutput(0))->second->GetShape();
                input_shape.SetDataType(next_output_shape.GetDataType());
                input_shape.SetDataFormat(next_output_shape.GetDataFormat());
                output_shape.SetDataType(next_output_shape.GetDataType());
                output_shape.SetDataFormat(next_output_shape.GetDataFormat());
                param->to = input_shape.GetDataType();
                auto cur_kernel = options.info->kernels.find(node_id);
                if (cur_kernel == options.info->kernels.end()) {
                    LOG(ERROR) << "Can not find kernel[" << node->GetName() << "].";
                }
                auto cur_cudakernel = (CudaOptKernel*)options.info->kernels.find(node_id)->second.get();
                auto status = cur_cudakernel->Finalize(options);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "update for kernel[" << node->GetName() << "]  common param failed: " << GetRetCodeStr(status);
                }
                LOG(DEBUG) << "Fuse bridge node[" << prenode->GetName() << "] with cast node[" << node->GetName() << "] with bridge node[" << nextnode->GetName() << "]";
            }else{
                LOG(DEBUG) << "Fuse bridge node[" << prenode->GetName() << "] with cast node[" << node->GetName() << "]";
            }
        }else{
            LOG(DEBUG) << "Fuse bridge node[" << prenode->GetName() << "] with cast node[" << node->GetName() << "]";
        }
        auto kernel = options.info->kernels.find(prenode_id);
        if (kernel == options.info->kernels.end()) {
            LOG(ERROR) << "Can not find kernel[" << node->GetName() << "].";
        }
        auto cudakernel = (CudaOptKernel*)options.info->kernels.find(prenode_id)->second.get(); //update output data type change of op
        auto status = cudakernel->Finalize(options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "update for kernel[" << node->GetName() << "]  common param failed: " << GetRetCodeStr(status);
        }
        return;
    }
    edge_id = node->GetOutput(0);
    if(topo->GetEdge(edge_id)->CalcConsumerCount() == 1)
    {
        auto nextnode_id = topo->GetEdge(edge_id)->CreateConsumerIter().Get(); // Get Output(0)
        auto nextnode = topo->GetNode(nextnode_id);
        if(nextnode->GetType().name == "Bridge"){
            TensorShape& input_shape = *options.tensors->find(node->GetInput(0))->second->GetShape();
            TensorShape& output_shape = *options.tensors->find(node->GetOutput(0))->second->GetShape();
            output_shape.SetDataType(input_shape.GetDataType());
            output_shape.SetDataFormat(input_shape.GetDataFormat());
            param->to = input_shape.GetDataType();
            auto kernel = options.info->kernels.find(node_id);
            if (kernel == options.info->kernels.end()) {
                LOG(ERROR) << "Can not find kernel[" << node->GetName() << "].";
            }
            auto cudakernel = (CudaOptKernel*)options.info->kernels.find(node_id)->second.get();
            auto status = cudakernel->Finalize(options);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "update for kernel[" << node->GetName() << "]  common param failed: " << GetRetCodeStr(status);
            }
            LOG(DEBUG) << "Fuse cast node[" << node->GetName() << "] with bridge node[" << nextnode->GetName() << "]";
            return;
        }
    }
}

RetCode CastFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto node_id = node->GetId();
    auto edge_id = node->GetInput(0);
    if (edge_id == INVALID_EDGEID) {
        return RC_UNSUPPORTED;
    }

    auto prenode_id = topo->GetEdge(edge_id)->GetProducer();
    if (prenode_id == INVALID_NODEID) {
        auto edge_out_id = node->GetOutput(0);
        if(topo->GetEdge(edge_out_id)->CalcConsumerCount() == 1){
            FuseCastBridge(node, options);
            return RC_SUCCESS;
        }
        return RC_UNSUPPORTED;
    }

    auto prenode = topo->GetNode(prenode_id);
    if (node->GetInputCount() != 1 || node->GetOutputCount() != 1) {
        return RC_UNSUPPORTED;
    }

    auto edge = topo->GetEdge(node->GetOutput(0));
    if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
        return RC_UNSUPPORTED;
    }

    if (CanFuse(node, prenode)) {
        LOG(DEBUG) << "Fuse cast node[" << node->GetName() << "] with prenode[" << prenode->GetName() << "]";
        options.info->kernels.erase(node_id);
        FuseWithPreviousCast(node, prenode, options);
        FuseCastBridge(prenode, options);
    }
    else if(node->GetType().name == "Cast")
    {
        FuseCastBridge(node, options);
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
