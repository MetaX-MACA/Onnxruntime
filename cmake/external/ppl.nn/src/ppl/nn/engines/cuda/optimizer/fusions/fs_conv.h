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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_CONV_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_FUSIONS_FS_CONV_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/fusion.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_batch_conv.h"

namespace ppl { namespace nn { namespace cuda {

class ConvFusion : public Fusion {
public:
    ppl::common::RetCode FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) override;

private:
    std::vector<ir::Node*> fuseInt8NodeVector_;
    std::vector<ir::Node*> fuseFp16NodeVector_;
    bool FuseTest(ir::Node* node, const OptKernelOptions& options,
                  std::function<ppl::common::RetCode(ir::Node*, const OptKernelOptions&)>);
    ppl::common::RetCode FuseConvWithNextNode(ir::Node* node, ir::Node* nextnode, const OptKernelOptions& options);

    static bool CanFuseRelu(ir::Node* nextnode, const OptKernelOptions& options) {
#ifdef PPLNN_USE_DNN
    #ifndef PPLNN_USE_MACA
        std::set<std::string> relu_fuse_op{"Relu"};//与NV文档描述不一致，"Sigmoid", "Swish", "Tanh"};
    #else
        std::set<std::string> relu_fuse_op{"Relu", "Sigmoid", "HardSigmoid", "Swish", "HardSwish", "Clip", "PRelu", "LeakyRelu", "Mish"};
    #endif
#else
    #ifdef PPLNN_ENABLE_CUDA_JIT
        std::set<std::string> relu_fuse_op{"Relu", "Clip", "PRelu", "LeakyRelu"};
    #else
        std::set<std::string> relu_fuse_op{"Relu", "Clip"};
    #endif
#endif
        if (relu_fuse_op.find(nextnode->GetType().name) != relu_fuse_op.end()) {
            if (nextnode->GetType().name == "PRelu") { // extra check for PRelu
                // slope must be an 1-d array or a scalar
                const TensorShape& shape1 = *options.tensors->find(nextnode->GetInput(0))->second->GetShape();
                const TensorShape& shape2 = *options.tensors->find(nextnode->GetInput(1))->second->GetShape();
                if (shape2.IsScalar() || (shape2.GetDimCount() == 1 && shape1.GetDim(1) == shape2.GetDim(0))
                    || shape1.GetDim(1) == shape2.CalcElementsExcludingPadding()) {
                    return true;
                }
                return false;
            }
            return true;
        }
        return false;
    }

    static bool CanFuseElementwise(ir::Node* nextnode, const OptKernelOptions& options) {
    #if 0//#ifdef PPLNN_USE_DNN
        return false;
    #endif
        std::set<std::string> elementwise_fuse_op{"Add", "Mul"};
        if (elementwise_fuse_op.find(nextnode->GetType().name) != elementwise_fuse_op.end()) {
            // two inputs must have same dims size for conv-add fusion
            const TensorShape& shape1 = *options.tensors->find(nextnode->GetInput(0))->second->GetShape();
            const TensorShape& shape2 = *options.tensors->find(nextnode->GetInput(1))->second->GetShape();
            if (shape1.GetDimCount() != shape2.GetDimCount()) {
                return false;
            }
            for (uint32_t i = 0; i < shape1.GetDimCount(); ++i) {
                if (shape1.GetDim(i) != shape2.GetDim(i)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    static bool NodeCanFuseNextNode(ir::Node* node, const OptKernelOptions& options){
        auto topo = options.graph->topo.get();
        auto edge_id = node->GetOutput(0);
        auto edge = topo->GetEdge(edge_id);
        if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
            return false;
        }
        if (topo->GetEdge(edge_id)->CalcConsumerCount() != 1) { // Can not fuse multi-consumer edge
            return false;
        }
        return true;
    }

    static bool edgeIsQuant(edgeid_t edgeId, const OptKernelOptions& options){
        if(options.quants->at(edgeId).bit_width == 8){
            return true;
        }else{
            return false;
        }
    }

    static ir::Node* NextNodeCanFused(ir::Node* node, const OptKernelOptions& options,
                                      std::function<ppl::common::RetCode(ir::Node*, const OptKernelOptions&)> canfuseRelu,
                                      std::function<ppl::common::RetCode(ir::Node*, const OptKernelOptions&)> canfuseElementwise){
        auto topo = options.graph->topo.get();
        if(NodeCanFuseNextNode(node, options) != true){
            return nullptr;
        }
        if(canfuseRelu == nullptr && canfuseElementwise == nullptr){
            return nullptr;
        }
        auto connectEdgeId = node->GetOutput(0);
        auto nextnode_id = topo->GetEdge(connectEdgeId)->CreateConsumerIter().Get(); // Get Output(0)
        auto nextnode = topo->GetNode(nextnode_id);
        //printf(" nextnode: %s\n", nextnode->GetName().c_str());
        if(canfuseRelu != nullptr && canfuseRelu(nextnode, options)){
            return nextnode;
        }
        if(canfuseElementwise != nullptr && canfuseElementwise(nextnode, options)){
            return nextnode;
        }
        return nullptr;
    }

    void ReplaceNodeName(std::string ConvOpName, ir::Node* ActNode, int opNum){
        std::set<std::string> relu_fuse_op{"Relu", "Sigmoid", "HardSigmoid", "Swish", "HardSwish", "Clip", "PRelu", "LeakyRelu", "Mish"};
        if (relu_fuse_op.find(ActNode->GetType().name) != relu_fuse_op.end()){
            std::string actOpName = ActNode->GetName();
            if(actOpName.find("ppl_anonymous_node_") != string::npos){
                ActNode->SetName(ConvOpName + "_" + ActNode->GetType().name + "_" + std::to_string(opNum));
            }
        }
    }

    bool CanFuseConvInt8(ir::Node* node, bool reliable, const OptKernelOptions& options){
        std::vector<ir::Node*> nodeVector;
        fuseInt8NodeVector_.clear();
        fuseFp16NodeVector_.clear();
        nodeVector.clear();
        nodeVector.push_back(node);  //Conv + ...
        auto nextNode = NextNodeCanFused(node, options, CanFuseRelu, CanFuseElementwise);
        if(nullptr == nextNode){
            goto NODE_PROCESS;
        }
        ReplaceNodeName(node->GetName(), nextNode, nodeVector.size());
        nodeVector.push_back(nextNode);
        if(CanFuseRelu(nextNode, options) == true){ //Conv + Act + ...
            nextNode = NextNodeCanFused(nextNode, options, nullptr, CanFuseElementwise);
            if(nullptr == nextNode){
                goto NODE_PROCESS;
            }
            nodeVector.push_back(nextNode); //Conv + Act + Elemetwise + ...
            nextNode = NextNodeCanFused(nextNode, options, CanFuseRelu, nullptr);
            if(nullptr == nextNode){
                goto NODE_PROCESS;
            }
            ReplaceNodeName(node->GetName(), nextNode, nodeVector.size());
            nodeVector.push_back(nextNode); //Conv + Act + Elemetwise + Act
        }else{ //Conv + Elementwise + ...
            nextNode = NextNodeCanFused(nextNode, options, CanFuseRelu, nullptr);
            if(nullptr == nextNode){
                goto NODE_PROCESS;
            }
            ReplaceNodeName(node->GetName(), nextNode, nodeVector.size());
            nodeVector.push_back(nextNode);; //Conv + Elemetwise + Act
        }
    NODE_PROCESS:
        auto inputEdgeId = node->GetInput(0);
        auto weightEdgeId = node->GetInput(1);
        std::set<std::string> elementwise_fuse_op{"Add", "Mul"};
        std::set<std::string> equal_quant_param_fuse_op{"Relu", "Sigmoid", "HardSigmoid", "Swish", "HardSwish", "Clip", "PRelu", "LeakyRelu", "Mish"};
        if(edgeIsQuant(inputEdgeId, options) == true && edgeIsQuant(weightEdgeId, options) == true){
            uint32_t i = 0;
            for(; i < nodeVector.size(); i++){
                auto outputEdgeId = nodeVector[i]->GetOutput(0);
                if(i < (nodeVector.size() - 1)){
                    ir::Node* nodeTemp = nodeVector[i + 1];
                    if (elementwise_fuse_op.find(nodeTemp->GetType().name) != elementwise_fuse_op.end()){
                        for(uint32_t j = 0; j < nodeTemp->GetInputCount(); j++){
                            auto eleInputEdgeId = nodeTemp->GetInput(j);
                            if(eleInputEdgeId != outputEdgeId){
                                if(!reliable || edgeIsQuant(eleInputEdgeId, options) != true){
                                    if(edgeIsQuant(outputEdgeId, options) != true){
                                        i = nodeVector.size();
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    if (equal_quant_param_fuse_op.find(nodeTemp->GetType().name) != equal_quant_param_fuse_op.end() && edgeIsQuant(outputEdgeId, options) != true){
                        continue;
                    }
                }
                if(edgeIsQuant(outputEdgeId, options) == true){
                    break;
                }
            }
            if(i != nodeVector.size()){
                for(uint32_t j = 0; j <= i; j++){
                    fuseInt8NodeVector_.push_back(nodeVector[j]);
                }
            }
        } else if(edgeIsQuant(inputEdgeId, options) != true && edgeIsQuant(weightEdgeId, options) != true){
            const TensorShape& shape1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
            auto node_id = node->GetId();
            auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node_id].get());
            CudaConvParam* param = (CudaConvParam*)opt_kernel->GetParam();
            bool fp16_can_fuse = true;//!(param->param.group > 1 && shape1.GetDim(1) == 1);
            if(fp16_can_fuse){
                for(uint32_t i = 0; i < nodeVector.size(); i++){
                    bool canFuse = true;
                    for(uint32_t j = 0; j < nodeVector[i]->GetInputCount(); j++){
                        auto eleInputEdgeId = nodeVector[i]->GetInput(j);
                        if(edgeIsQuant(eleInputEdgeId, options) == true){
                            canFuse = false;
                        }
                    }
                    auto outputEdgeId = nodeVector[i]->GetOutput(0);
                    if(edgeIsQuant(outputEdgeId, options) == true){
                        canFuse = false;
                    }
                    if(canFuse)
                        fuseFp16NodeVector_.push_back(nodeVector[i]);
                    else
                        break;
                }
            } else{
                if(nodeVector.size() > 1 && nodeVector[1]->GetType().name == "Relu"){
                    fuseFp16NodeVector_.push_back(nodeVector[0]);
                    fuseFp16NodeVector_.push_back(nodeVector[1]);
                }
            }
        }
        if(nodeVector.size() > 1){
            std::string fuseNodeName = nodeVector[0]->GetName() + "_fuse";
            std::string findFuseNodesStr = "[";
            for(uint32_t i = 0; i < nodeVector.size(); i++){
                findFuseNodesStr += nodeVector[i]->GetName() + (i < (nodeVector.size() - 1) ? " + " : "");
            }
            findFuseNodesStr += "]";
            std::string finalFuseNodesStr = "";
            if(fuseInt8NodeVector_.size() > 0){
                finalFuseNodesStr += " int8 fuse ";
                finalFuseNodesStr += "[";
                for(uint32_t i = 0; i < fuseInt8NodeVector_.size(); i++){
                    finalFuseNodesStr += fuseInt8NodeVector_[i]->GetName() + (i < (fuseInt8NodeVector_.size() - 1) ? " + " : "");
                    if(i > 0){
                        fuseNodeName += ("_" + fuseInt8NodeVector_[i]->GetType().name);
                        if (elementwise_fuse_op.find(fuseInt8NodeVector_[i]->GetType().name) != elementwise_fuse_op.end()){
                            const TensorShape& shape1 = *options.tensors->find(fuseInt8NodeVector_[i]->GetInput(0))->second->GetShape();
                            const TensorShape& shape2 = *options.tensors->find(fuseInt8NodeVector_[i]->GetInput(1))->second->GetShape();
                            fuseNodeName += "_";
                            for (uint32_t i = 0; i < shape1.GetDimCount(); ++i) {
                                if(i != 0){
                                    fuseNodeName += "x";
                                }
                                fuseNodeName += std::to_string(shape1.GetDim(i));
                            }
                            fuseNodeName += "_";
                            for (uint32_t i = 0; i < shape2.GetDimCount(); ++i) {
                                if(i != 0){
                                    fuseNodeName += "x";
                                }
                                fuseNodeName += std::to_string(shape2.GetDim(i));
                            }
                        }
                    }
                }
            } else if(fuseFp16NodeVector_.size() > 0){
                finalFuseNodesStr += " fp16 fuse ";
                finalFuseNodesStr += "[";
                for(uint32_t i = 0; i < fuseFp16NodeVector_.size(); i++){
                    finalFuseNodesStr += fuseFp16NodeVector_[i]->GetName() + (i < (fuseFp16NodeVector_.size() - 1) ? " + " : "");
                    if(i > 0){
                        fuseNodeName += ("_" + fuseFp16NodeVector_[i]->GetType().name);
                        if (elementwise_fuse_op.find(fuseFp16NodeVector_[i]->GetType().name) != elementwise_fuse_op.end()){
                            const TensorShape& shape1 = *options.tensors->find(fuseFp16NodeVector_[i]->GetInput(0))->second->GetShape();
                            const TensorShape& shape2 = *options.tensors->find(fuseFp16NodeVector_[i]->GetInput(1))->second->GetShape();
                            fuseNodeName += "_";
                            for (uint32_t i = 0; i < shape1.GetDimCount(); ++i) {
                                if(i != 0){
                                    fuseNodeName += "x";
                                }
                                fuseNodeName += std::to_string(shape1.GetDim(i));
                            }
                            fuseNodeName += "_";
                            for (uint32_t i = 0; i < shape2.GetDimCount(); ++i) {
                                if(i != 0){
                                    fuseNodeName += "x";
                                }
                                fuseNodeName += std::to_string(shape2.GetDim(i));
                            }
                        }
                    }
                }
            }
            finalFuseNodesStr += "]";

            LOG(DEBUG) << "## Conv fuse info: find Conv+Act+Ele+Act struct" << findFuseNodesStr << " -> final fused struct" << finalFuseNodesStr;
            if(fuseInt8NodeVector_.size() > 1){
                fuseInt8NodeVector_[0]->SetName(fuseNodeName);
                LOG(DEBUG) << "New fuse op name: " << fuseNodeName;
            } else if(fuseFp16NodeVector_.size() > 0){
                fuseFp16NodeVector_[0]->SetName(fuseNodeName);
                LOG(DEBUG) << "New fuse op name: " << fuseNodeName;
            }
        }
        return fuseInt8NodeVector_.size() > 0;
    }
};

class ConvFusion_Set: public Fusion{
    ppl::common::RetCode FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) override;
    ConvFusion conv_fusion_;
    BatchConvFusion batch_conv_fusion_ = BatchConvFusion(2, "Swish");
};


}}} // namespace ppl::nn::cuda

#endif
