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

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_transpose.h"

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/params/onnx/transpose_param.h"
#include "ppl/nn/params/onnx/reshape_param.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

bool TransposeFusion::CanFuse(ir::Node* node, ir::Node* prenode) {
    if (node->GetType().name == "Transpose" && prenode->GetType().name == "Bridge") {
        return true;
    }
    return false;
}

bool TransposeFusion::CanFuseBridgeReshapeTranspose(ir::Node* node, ir::Node* prenode, ir::Node* nextnode){
    if (node->GetType().name == "Reshape" && prenode->GetType().name == "Bridge" && nextnode->GetType().name == "Transpose") {
        return true;
    }
    return false;
}

RetCode TransposeFusion::FuseWithPreviousBridge(ir::Node* node, ir::Node* prenode, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto connect_edge_id = node->GetInput(0);
    auto connect_edge = topo->GetEdge(connect_edge_id);
    auto next_edge_id = node->GetOutput(0);
    auto next_edge = topo->GetEdge(next_edge_id);
    auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node->GetId()].get());
    ppl::nn::onnx::TransposeParam* param = (ppl::nn::onnx::TransposeParam*)opt_kernel->GetParam();
    if((param->perm.size() != 4 || !(param->perm[0] == 0 && param->perm[1] == 2 && param->perm[2] == 3 && param->perm[3] == 1)) &&
        (param->perm.size() != 3 || !(param->perm[0] == 0 && param->perm[1] == 2 && param->perm[2] == 1 ))){
        return RC_UNSUPPORTED;
    }
    TensorShape& input_shape = *options.tensors->find(node->GetInput(0))->second->GetShape();
    TensorShape& output_shape = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    if(!(input_shape.GetDataFormat() == DATAFORMAT_NDARRAY && output_shape.GetDataFormat() == DATAFORMAT_NDARRAY)){
        return RC_UNSUPPORTED;
    }
    TensorShape& pre_input_shape = *options.tensors->find(prenode->GetInput(0))->second->GetShape();
    if(!(pre_input_shape.GetDataFormat() == DATAFORMAT_NHWC8 || pre_input_shape.GetDataFormat() == DATAFORMAT_NHWC16 || pre_input_shape.GetDataFormat() == DATAFORMAT_NHWC)){
        return RC_UNSUPPORTED;
    }
    input_shape.SetDataFormat(DATAFORMAT_NHWC);
    input_shape.SetDataFormatForceTo(DATAFORMAT_NDARRAY);

    auto kernel = options.info->kernels.find(prenode->GetId());
    if (kernel == options.info->kernels.end()) {
        LOG(ERROR) << "Can not find kernel[" << node->GetName() << "].";
    }
    auto cudakernel = (CudaOptKernel*)options.info->kernels.find(prenode->GetId())->second.get(); //update output data type change of op
    auto status = cudakernel->Finalize(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "update for kernel[" << node->GetName() << "]  common param failed: " << GetRetCodeStr(status);
    }

    for (auto it = next_edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
        auto tempnode_id = it.Get();
        auto tempnode = topo->GetNode(tempnode_id);
        connect_edge->AddConsumer(tempnode_id);
        tempnode->ReplaceInput(next_edge_id, connect_edge_id);
    }
    LOG(DEBUG) << "Fuse Transpose node[" << node->GetName() << "] with prenode[" << prenode->GetName() << "]";
    connect_edge->DelConsumer(node->GetId());
    topo->DelEdge(next_edge->GetId());
    topo->DelNode(node->GetId());
    options.info->kernels.erase(node->GetId());
    return RC_SUCCESS;
}

RetCode TransposeFusion::FusePostReshapeTransposeWithBridge(ir::Node* node, ir::Node* prenode, ir::Node* nextnode, const OptKernelOptions& options){
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto connect_edge_id = node->GetInput(0);
    auto connect_edge = topo->GetEdge(connect_edge_id);
    auto opt_kernel = (CudaOptKernel*)(options.info->kernels[node->GetId()].get());
    auto opt_next_kernel = (CudaOptKernel*)(options.info->kernels[nextnode->GetId()].get());
    auto perm = ((ppl::nn::onnx::TransposeParam*)opt_next_kernel->GetParam())->perm;
    int64_t perm_size = perm.size();
    if(perm_size == 3 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1){
        auto iter = data->constants.find(node->GetInput(1));
        if(iter != data->constants.end()){
            int64_t* shape_data = (int64_t*)(iter->second.data.GetData());
            auto data_iter = data->shapes.find(node->GetInput(1));
            if(data_iter != data->shapes.end()){
                int64_t shape_ele_num = data_iter->second.dims[0];
                if(shape_ele_num !=3 ){
                    return RC_UNSUPPORTED;
                }
                TensorShape& input_shape = *options.tensors->find(node->GetInput(0))->second->GetShape();
                TensorShape& output_shape = *options.tensors->find(node->GetOutput(0))->second->GetShape();
                if(input_shape.GetDimCount() == 4 && shape_ele_num == perm_size){
                    TensorShape& pre_input_shape = *options.tensors->find(prenode->GetInput(0))->second->GetShape();
                    if(pre_input_shape.GetDataFormat() != DATAFORMAT_NHWC8){
                        return RC_UNSUPPORTED;
                    }
                    input_shape.SetDataFormat(DATAFORMAT_NHWC);
                    input_shape.SetDataFormatForceTo(DATAFORMAT_NDARRAY);
                    std::vector<int64_t> c_split_dims;
                    c_split_dims.push_back(0);
                    for(int i = 1; i < shape_ele_num ; i++){
                        c_split_dims.push_back(output_shape.GetDim(shape_ele_num - i));
                    }
                    input_shape.SetChannelSpilt(c_split_dims);
                    auto kernel = options.info->kernels.find(prenode->GetId());
                    auto cudakernel = (CudaOptKernel*)options.info->kernels.find(prenode->GetId())->second.get(); //update output data type change of op
                    auto status = cudakernel->Finalize(options);
                    if (status != RC_SUCCESS) {
                        LOG(ERROR) << "update for kernel[" << node->GetName() << "]  common param failed: " << GetRetCodeStr(status);
                        return RC_UNSUPPORTED;
                    }
                    auto next_edge_id = nextnode->GetOutput(0);
                    auto next_edge = topo->GetEdge(next_edge_id);
                    for (auto it = next_edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                        auto tempnode_id = it.Get();
                        auto tempnode = topo->GetNode(tempnode_id);
                        connect_edge->AddConsumer(tempnode_id);
                        tempnode->ReplaceInput(next_edge_id, connect_edge_id);
                    }
                    LOG(DEBUG) << "Fuse reshape+transpose node[" << node->GetName() << " + " << nextnode->GetName() << "] with prenode[" << prenode->GetName() << "]";
                    connect_edge->DelConsumer(node->GetId());
                    options.info->constants.erase(node->GetInput(1));
                    topo->DelEdge(next_edge->GetId());
                    topo->DelEdge(node->GetOutput(0));
                    topo->DelEdge(node->GetInput(1));
                    topo->DelNode(nextnode->GetId());
                    topo->DelNode(node->GetId());
                    options.info->kernels.erase(node->GetId());
                    options.info->kernels.erase(nextnode->GetId());
                }
            }else{
                return RC_UNSUPPORTED;
            }
        }
    }
    if(perm_size < 5 || perm[0] != 0 || perm[perm_size - 2] != 1 || perm[perm_size - 1] != 2)
        return RC_UNSUPPORTED;
    auto iter = data->constants.find(node->GetInput(1));
    if(iter != data->constants.end()){
        int64_t* shape_data = (int64_t*)(iter->second.data.GetData());
        auto data_iter = data->shapes.find(node->GetInput(1));
        if(data_iter != data->shapes.end()){
            int64_t shape_ele_num = data_iter->second.dims[0];
            TensorShape& input_shape = *options.tensors->find(node->GetInput(0))->second->GetShape();
            TensorShape& output_shape = *options.tensors->find(node->GetOutput(0))->second->GetShape();
            if(input_shape.GetDimCount() == 4 && shape_ele_num == perm_size){
                if(input_shape.GetDim(0) == output_shape.GetDim(0) && input_shape.GetDim(2) == output_shape.GetDim(shape_ele_num - 2)
                   && input_shape.GetDim(3) == output_shape.GetDim(shape_ele_num - 1)){
                    TensorShape& pre_input_shape = *options.tensors->find(prenode->GetInput(0))->second->GetShape();
                    if(!(pre_input_shape.GetDataFormat() == DATAFORMAT_NHWC8 || pre_input_shape.GetDataFormat() == DATAFORMAT_NHWC16
                         || pre_input_shape.GetDataFormat() == DATAFORMAT_NHWC)){
                        return RC_UNSUPPORTED;
                    }
                    input_shape.SetDataFormat(DATAFORMAT_NHWC);
                    input_shape.SetDataFormatForceTo(DATAFORMAT_NDARRAY);
                    std::vector<int64_t> c_split_dims;
                    for(int i = 1; i < shape_ele_num - 2; i++){
                        c_split_dims.push_back(output_shape.GetDim(i));
                    }
                    input_shape.SetChannelSpilt(c_split_dims);
                    auto kernel = options.info->kernels.find(prenode->GetId());
                    auto cudakernel = (CudaOptKernel*)options.info->kernels.find(prenode->GetId())->second.get(); //update output data type change of op
                    auto status = cudakernel->Finalize(options);
                    if (status != RC_SUCCESS) {
                        LOG(ERROR) << "update for kernel[" << node->GetName() << "]  common param failed: " << GetRetCodeStr(status);
                        return RC_UNSUPPORTED;
                    }
                    auto next_edge_id = nextnode->GetOutput(0);
                    auto next_edge = topo->GetEdge(next_edge_id);
                    for (auto it = next_edge->CreateConsumerIter(); it.IsValid(); it.Forward()) {
                        auto tempnode_id = it.Get();
                        auto tempnode = topo->GetNode(tempnode_id);
                        connect_edge->AddConsumer(tempnode_id);
                        tempnode->ReplaceInput(next_edge_id, connect_edge_id);
                    }
                    LOG(DEBUG) << "Fuse reshape+transpose node[" << node->GetName() << " + " << nextnode->GetName() << "] with prenode[" << prenode->GetName() << "]";
                    connect_edge->DelConsumer(node->GetId());
                    options.info->constants.erase(node->GetInput(1));
                    topo->DelEdge(next_edge->GetId());
                    topo->DelEdge(node->GetOutput(0));
                    topo->DelEdge(node->GetInput(1));
                    topo->DelNode(nextnode->GetId());
                    topo->DelNode(node->GetId());
                    options.info->kernels.erase(node->GetId());
                    options.info->kernels.erase(nextnode->GetId());
                }
            }
        }
    }

    return RC_SUCCESS;
}


RetCode TransposeFusion::FuseNode(ir::Node* node, bool reliable, const OptKernelOptions& options) {
    auto topo = options.graph->topo.get();
    auto node_id = node->GetId();
    auto edge_id = node->GetInput(0);
    if (edge_id == INVALID_EDGEID) {
        return RC_UNSUPPORTED;
    }

    auto prenode_id = topo->GetEdge(edge_id)->GetProducer();
    if (prenode_id == INVALID_NODEID) {
        return RC_UNSUPPORTED;
    }

    auto prenode = topo->GetNode(prenode_id);
    if (node->GetOutputCount() != 1) {
        return RC_UNSUPPORTED;
    }

    auto edge = topo->GetEdge(node->GetOutput(0));
    if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID) { // Can not fuse an output edge
        return RC_UNSUPPORTED;
    }

    if (CanFuse(node, prenode)) {
        FuseWithPreviousBridge(node, prenode, options);
        return RC_SUCCESS;
    }

    if(edge->CalcConsumerCount() == 1){
        auto netxnode_id = edge->CreateConsumerIter().Get();
        auto nextnode = topo->GetNode(netxnode_id);
        if (CanFuseBridgeReshapeTranspose(node, prenode, nextnode)) {
            FusePostReshapeTransposeWithBridge(node, prenode, nextnode, options);
        }
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
