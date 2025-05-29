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

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_conv.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ConvFusion::FuseConvWithNextNode(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options) {
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

bool ConvFusion::FuseTest(ir::Node* node, const OptKernelOptions& options,
                          std::function<ppl::common::RetCode(ir::Node*, const OptKernelOptions&)> canfuse) {
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto node_id = node->GetId();
    auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node_id].get());
    CudaConvParam* param = (CudaConvParam*)opt_kernel->GetParam();

    auto edge_id = node->GetOutput(0);
    auto edge = topo->GetEdge(edge_id);
    if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
        return false;
    }
    if (topo->GetEdge(edge_id)->CalcConsumerCount() != 1) { // Can not fuse multi-consumer edge
        return false;
    }

    auto nextnode_id = topo->GetEdge(edge_id)->CreateConsumerIter().Get(); // Get Output(0)
    auto nextnode = topo->GetNode(nextnode_id);
#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
    if(fuseInt8NodeVector_.size() > 0){
        uint32_t i = 0;
        for(; i < fuseInt8NodeVector_.size(); i++){
            if(nextnode == fuseInt8NodeVector_[i]){
                break;
            }
        }
        if(i >= fuseInt8NodeVector_.size()){
            return false;
        }
    } else if(fuseFp16NodeVector_.size() > 0){
         uint32_t i = 0;
        for(; i < fuseFp16NodeVector_.size(); i++){
            if(nextnode == fuseFp16NodeVector_[i]){
                break;
            }
        }
        if(i >= fuseFp16NodeVector_.size()){
            return false;
        }
    }
#endif
    if (canfuse(nextnode, options)) {
        /*
        auto inputEdgeId = node->GetInput(0);
        if(  (edgeIsQuant(inputEdgeId, options) != true ||
             (edgeIsQuant(inputEdgeId, options) == true && fuseInt8NodeVector_.size() == 0))
            && nextnode->GetType().name != "Relu"){ //fp16 only support conv+relu
            return false;
        }*/
        LOG(DEBUG) << "Fuse node[" << node->GetName() << "] and nextnode[" << nextnode->GetName() << "]";
        // avoid conv+add+add case
        for (auto& type : param->extra_param.fuse_info.types) {
            if (type == nextnode->GetType().name)
                return false;
        }
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
                auto size = min_iter->second.data.GetSize();
                if (size == 2) {
                    clip_param->min_value = *(half*)(min_iter->second.data.GetData());
                } else if (size == 4) {
                    clip_param->min_value = *(float*)(min_iter->second.data.GetData());
                }
            }
            auto max_iter = data->constants.find(nextnode->GetInput(2));
            if (max_iter != data->constants.end()) {
                auto size = max_iter->second.data.GetSize();
                if (size == 2) {
                    clip_param->max_value = *(half*)(max_iter->second.data.GetData());
                } else if (size == 4) {
                    clip_param->max_value = *(float*)(max_iter->second.data.GetData());
                }
            }
            param->extra_param.fuse_info.fuse_attrs.emplace_back((void*)clip_param);
        }
        options.info->kernels.erase(nextnode_id);
        FuseConvWithNextNode(node, nextnode, options);
        return true;
    }
    return false;
}

RetCode ConvFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    const TensorShape& input_shape = *options.tensors->find(node->GetInput(0))->second->GetShape();
    const TensorShape& weight_shape = *options.tensors->find(node->GetInput(1))->second->GetShape();
    // if(input_shape.GetDataType() == DATATYPE_FLOAT32)
    //     return RC_UNSUPPORTED;
    if(weight_shape.GetDimCount() == 5){
        //return RC_UNSUPPORTED;
    }
#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
    CanFuseConvInt8(node, reliable, options);
    if(fuseInt8NodeVector_.size() > 1){
        std::set<std::string> elementwise_fuse_op{"Add", "Mul"};
        for (size_t i = 0; i < (fuseInt8NodeVector_.size() - 1); i++){
            auto curNode = node;
            auto nextNode = fuseInt8NodeVector_[i + 1];
            if (elementwise_fuse_op.find(nextNode->GetType().name) != elementwise_fuse_op.end()){
                if (FuseTest(curNode, options, CanFuseElementwise) != true) {
                    break;;
                }
            }else{
                if (FuseTest(curNode, options, CanFuseRelu) != true) {
                    break;;
                }
            }
        }
        return RC_SUCCESS;
    } else if(fuseFp16NodeVector_.size() > 1){
        std::set<std::string> elementwise_fuse_op{"Add", "Mul"};
        for (size_t i = 0; i < (fuseFp16NodeVector_.size() - 1); i++){
            auto curNode = node;
            auto nextNode = fuseFp16NodeVector_[i + 1];
            if (elementwise_fuse_op.find(nextNode->GetType().name) != elementwise_fuse_op.end()){
                if (FuseTest(curNode, options, CanFuseElementwise) != true) {
                    break;;
                }
            }else{
                if (FuseTest(curNode, options, CanFuseRelu) != true) {
                    break;;
                }
            }
        }
        return RC_SUCCESS;
    }
#else
    FuseTest(node, options, CanFuseRelu);
    if (reliable) {
        if (FuseTest(node, options, CanFuseElementwise)) {
            FuseTest(node, options, CanFuseRelu);
        }
    }
#endif
    return RC_SUCCESS;
}

ppl::common::RetCode ConvFusion_Set::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    auto opt_kernel =(CudaOptKernel*)(options.info->kernels[node->GetId()].get());
    if (!opt_kernel) {
        CudaConvParam* param = (CudaConvParam*)opt_kernel->GetParam();
        if(param->extra_param.fuse_info.output_channels.empty() == false){
            return RC_UNSUPPORTED;
        }
    }
    auto status = 0;
    status = batch_conv_fusion_.FuseNode(node, reliable, options);
    if(status != RC_SUCCESS){
        status = conv_fusion_.FuseNode(node, reliable, options);
        return status;
    }
    return RC_SUCCESS;
};

}}} // namespace ppl::nn::cuda
