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

#include "ppl/nn/engines/cuda/optimizer/opt_graph.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/engine.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/cuda/optimizer/algos/algo_graph.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/bridge_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/cast_op.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"
#include "ppl/nn/engines/cuda/params/gemm_extra_param.h"
#include "ppl/nn/params/onnx/transpose_param.h"

using namespace std;
using namespace ppl::common;
extern bool gIsOpKeepOnnxPrecision;
namespace ppl { namespace nn { namespace cuda {

OptGraph::OptGraph(ir::Graph* graph, RuntimePartitionInfo* info, CudaArgs* args, CompileInfo* compile_set)
    : graph_(graph), info_(info), args_(args), compile_set_(compile_set) {
    acquire_tensor_func_ = [this](edgeid_t eid, uint32_t) -> EdgeObject* {
        auto it = tensor_impls_.find(eid);
        if (it == tensor_impls_.end()) {
            return nullptr;
        }
        return it->second.get();
    };
}

OptGraph::~OptGraph() {
    // destroy tensors before default_cpu_device_
    tensor_impls_.clear();
}

int32_t OptGraph::LastLegalNodeIndex() {
    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i) {
        if (illegal_dims_.find(sorted_node_ids_[i]) != illegal_dims_.end()) {
            return (int32_t)i;
        }
    }
    return sorted_node_ids_.size() - 1;
}

void OptGraph::UpdateTopologicalSort() {
    ir::GraphTopo* topo = graph_->topo.get();
    sorted_node_ids_.clear();
    topo->TopologicalSort([this](nodeid_t nid) -> void {
        sorted_node_ids_.push_back(nid);
    });
}

void OptGraph::DumpGraphInfo(std::string stage) {
    ir::GraphTopo* topo = graph_->topo.get();
    std::vector<edgeid_t> edge_out_v;

    char *env_ptr = getenv("MACART_GRAPH_DUMP");
    if(env_ptr != nullptr){
        std::string graph_dump_str(env_ptr);
        if(graph_dump_str != stage){
            return;
        }
    }else{
        return;
    }
    LOG(INFO) << "############################    " << stage <<"    Graph info  ####################################################################";
    LOG(INFO) << "  Graph input Info:";
    for (uint32_t j = 0; j < topo->GetInputCount(); ++j){
        auto edge_id = topo->GetInput(j);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto edge = topo->GetEdge(edge_id);
        auto shape = tensor_impls_.find(edge_id)->second.get();
        LOG(DEBUG) << "    graph input(" << j  << "): " << edge->GetName() << ", format: "
                   << GetDataFormatStr(shape->GetShape()->GetDataFormat()) << ", datatype: "
                   << GetDataTypeStr(shape->GetShape()->GetDataType());
    }
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()){
        auto node = it->Get();
        LOG(INFO) << "  Node( " << node->GetName() << ") Info:";
        for (uint32_t j = 0; j < node->GetInputCount(); ++j) {
            auto edge_id = node->GetInput(j);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }
            auto edge = topo->GetEdge(edge_id);
            auto shape = tensor_impls_.find(edge_id)->second.get();
            LOG(DEBUG) << "    node input(" << j  << "): " << edge->GetName() << ", format: "
                       << GetDataFormatStr(shape->GetShape()->GetDataFormat()) << ", datatype: "
                       << GetDataTypeStr(shape->GetShape()->GetDataType());
        }
        for (uint32_t j = 0; j < node->GetOutputCount(); ++j) {
            auto edge_id = node->GetOutput(j);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }
            auto edge = topo->GetEdge(edge_id);
            if(edge->CalcConsumerCount() == 0){
                edge_out_v.push_back(edge_id);
            }
            auto shape = tensor_impls_.find(edge_id)->second.get();
            LOG(DEBUG) << "    node output(" << j  << "): " << edge->GetName() << ", format: "
                       << GetDataFormatStr(shape->GetShape()->GetDataFormat()) << ", datatype: "
                       << GetDataTypeStr(shape->GetShape()->GetDataType());
        }
    }
    LOG(INFO) << "  Graph output Info:";
    uint32_t graph_output_cnt = 0;
    for (uint32_t j = 0; j < topo->GetOutputCount(); ++j){
        auto edge_id = topo->GetOutput(j);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto edge = topo->GetEdge(edge_id);
        auto shape = tensor_impls_.find(edge_id)->second.get();
        LOG(DEBUG) << "    graph output(" << graph_output_cnt  << "): " << edge->GetName() << ", format: "
                   << GetDataFormatStr(shape->GetShape()->GetDataFormat()) << ", datatype: "
                   << GetDataTypeStr(shape->GetShape()->GetDataType());
        graph_output_cnt++;
    }
    for (uint32_t j = 0; j < edge_out_v.size(); ++j){
        auto edge = topo->GetEdge(edge_out_v[j]);
        if(topo->GetOutput(edge->GetName()) != INVALID_EDGEID)
            continue;
        auto shape = tensor_impls_.find(edge_out_v[j])->second.get();
        LOG(DEBUG) << "    graph output(" << graph_output_cnt << "): " << edge->GetName() << ", format: "
                   << GetDataFormatStr(shape->GetShape()->GetDataFormat()) << ", datatype: "
                   << GetDataTypeStr(shape->GetShape()->GetDataType());
        graph_output_cnt++;
    }
    LOG(INFO) << "################################################################################################";
}

RetCode OptGraph::graphPreOptProcess(const utils::SharedResource& resource){
    auto topo = graph_->topo.get();
    auto graph_data = graph_->data.get();
    UpdateTopologicalSort();
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        auto& type = node->GetType();
        if(type.name == "Gemm"){
            auto param_ref = graph_data->attrs.find(node->GetId());
            if (param_ref == graph_data->attrs.end()) {
                continue;
            }
            auto param = ((ppl::nn::onnx::GemmParam *)param_ref->second.get());
            if(param->transA){
                auto ret_pair = topo->AddNode(node->GetName() + "_transpose_a");
                if (!ret_pair.second) {
                    continue;
                }
                auto edge = topo->GetEdge(node->GetInput(0));
                auto node_tranpose_a = ret_pair.first;
                auto nodeType = node->GetType();
                nodeType.name = "Transpose";
                node_tranpose_a->SetType(nodeType);
                shared_ptr<ir::Attr> param_trans_a = make_shared<ppl::nn::onnx::TransposeParam>();
                auto tranpose_param = static_cast<ppl::nn::onnx::TransposeParam*>(param_trans_a.get());
                tranpose_param->perm.push_back(1);
                tranpose_param->perm.push_back(0);
                graph_data->attrs.emplace(node_tranpose_a->GetId(), std::move(param_trans_a));
                auto ret_pair_2 = topo->AddEdge(node->GetName() + "_transpose_a_out");
                auto edge_new = ret_pair_2.first;
                edge->DelConsumer(node->GetId());
                edge->AddConsumer(node_tranpose_a->GetId());
                node_tranpose_a->AddInput(edge->GetId());
                node_tranpose_a->AddOutput(edge_new->GetId());
                edge_new->SetProducer(node_tranpose_a->GetId());
                edge_new->AddConsumer(node->GetId());
                node->ReplaceInput(edge->GetId(), edge_new->GetId());
                param->transA = 0;
            }
            if(param->transB){
                auto weight_iter = graph_data->constants.find(node->GetInput(1));
                if(weight_iter != graph_data->constants.end()){
                    continue;
                }
                auto ret_pair = topo->AddNode(node->GetName() + "_transpose_b");
                if (!ret_pair.second) {
                    continue;
                }
                auto edge = topo->GetEdge(node->GetInput(1));
                auto node_tranpose_b = ret_pair.first;
                auto nodeType = node->GetType();
                nodeType.name = "Transpose";
                node_tranpose_b->SetType(nodeType);
                shared_ptr<ir::Attr> param_trans_b = make_shared<ppl::nn::onnx::TransposeParam>();
                auto tranpose_param = static_cast<ppl::nn::onnx::TransposeParam*>(param_trans_b.get());
                tranpose_param->perm.push_back(1);
                tranpose_param->perm.push_back(0);
                graph_data->attrs.emplace(node_tranpose_b->GetId(), std::move(param_trans_b));
                auto ret_pair_2 = topo->AddEdge(node->GetName() + "_transpose_b_out");
                auto edge_new = ret_pair_2.first;
                edge->DelConsumer(node->GetId());
                edge->AddConsumer(node_tranpose_b->GetId());
                node_tranpose_b->AddInput(edge->GetId());
                node_tranpose_b->AddOutput(edge_new->GetId());
                edge_new->SetProducer(node_tranpose_b->GetId());
                edge_new->AddConsumer(node->GetId());
                node->ReplaceInput(edge->GetId(), edge_new->GetId());
                param->transB = 0;
            }
        }
    }
    #if 0
    for (int32_t i = sorted_node_ids_.size() - 1; i >= 0; --i) {
        auto node = topo->GetNode(sorted_node_ids_[i]);
        if (node) {
            if(node->GetType().name == "Mul"){
                auto fs_filter_manager = FsFilterManager::Instance();
                OptKernelOptions options(graph_, &resource);
                auto fuse = fs_filter_manager->FindFusion(node->GetType().name);
                if (fuse) {
                    fuse->FuseNode(node, true, options);
                }
            }
        }
    }
    #endif
    return RC_SUCCESS;
}

RetCode OptGraph::InitKernels() {
    auto topo = graph_->topo.get();
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        auto& type = node->GetType();
        auto creator = OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version);
        if (!creator) {
            LOG(ERROR) << "cannot find creator for CudaOptKernel[" << node->GetName() << "] type[" << type.domain << ":"
                       << type.name << "]" << type.version;
            return RC_NOT_FOUND;
        }

        auto opt_kernel = unique_ptr<CudaOptKernel>((*creator)(node));
        if (!opt_kernel) {
            LOG(ERROR) << "create CudaOptKernel failed: oom";
            return RC_OUT_OF_MEMORY;
        }

        info_->kernels.emplace(node->GetId(), std::move(opt_kernel));
    }

    return RC_SUCCESS;
}

RetCode OptGraph::findGraphInputConv(const utils::SharedResource& resource){
    auto topo = graph_->topo.get();
    UpdateTopologicalSort();
    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i){
        auto node = topo->GetNode(sorted_node_ids_[i]);
        if(node->GetType().name != "Conv"){
            continue;
        }
        auto edge_input = topo->GetEdge(node->GetInput(0));
        auto pre_node_id = edge_input->GetProducer();
        if(pre_node_id != INVALID_NODEID){
            auto pre_node = topo->GetNode(pre_node_id);
            if(pre_node->GetType().name != "Cast"){
                continue;
            }
        #if 1
            if(edge_input->CalcConsumerCount() > 1){
                continue;
            }
        #endif
            auto pre_edge_input = topo->GetEdge(pre_node->GetInput(0));
            auto pre_pre_node_id = pre_edge_input->GetProducer();
            if(pre_pre_node_id != INVALID_NODEID){
                continue;
            }
        }
        CudaOptKernel* kernel = (CudaOptKernel*)(info_->kernels.find(node->GetId())->second.get());
        CudaConvParam* param = (CudaConvParam*)kernel->GetParam();
        param->extra_param.is_graph_input_conv = true;
    }
    return RC_SUCCESS;
}

RetCode OptGraph::UpdateDims(const utils::SharedResource& resource) {
    auto topo = graph_->topo.get();
    auto data = graph_->data.get();

    vector<nodeid_t> sorted_node_ids;
    topo->TopologicalSort([&sorted_node_ids](nodeid_t nid) -> void {
        sorted_node_ids.push_back(nid);
    });

    OptKernelOptions options(graph_, &resource);
    UpdateTopologicalSort();

    InputOutputInfo IOinfo;
    IOinfo.SetAcquireFunc(acquire_tensor_func_);

    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i) {
        auto node = topo->GetNode(sorted_node_ids_[i]);
        IOinfo.SetNode(node);

        CudaOptKernel* kernel = (CudaOptKernel*)(info_->kernels.find(node->GetId())->second.get());
        auto status = kernel->Init(options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Init kernel failed.";
            return RC_UNSUPPORTED;
        }

        for (uint32_t j = 0; j < node->GetInputCount(); ++j) {
            auto edge_id = node->GetInput(j);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }

            auto edge = topo->GetEdge(edge_id);
            auto impl_pair = tensor_impls_.insert(
                make_pair(edge_id, unique_ptr<TensorImpl>(new TensorImpl(edge, TENSORTYPE_NORMAL))));
            if (impl_pair.second) {
                // default shape
                TensorShape temp_tensor_shape;

                // replace to model-given shape
                auto ir_shape = data->shapes.find(edge_id);
                if (ir_shape != data->shapes.end()) {
                    utils::IrShape2TensorShape(ir_shape->second, &temp_tensor_shape);
                    auto dim_count = temp_tensor_shape.GetRealDimCount();
                    if (dim_count > 0) {
                        // replace dynamic dims to default values
                        if (temp_tensor_shape.GetDim(0) == INVALID_DIM_VALUE) {
                            temp_tensor_shape.SetDim(0, 1);
                        }
                        for (uint32_t k = 1; k < dim_count; ++k) {
                            if (temp_tensor_shape.GetDim(k) == INVALID_DIM_VALUE) {
                                temp_tensor_shape.SetDim(k, 224);
                            }
                        }
                    }
                } else {
                    temp_tensor_shape.Reshape({1, 3, 224, 224});
                    temp_tensor_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                    temp_tensor_shape.SetDataType(DATATYPE_UNKNOWN);
                }

                if (topo->GetInput(edge->GetName()) != INVALID_EDGEID) { // input j is a graph input edge
                    if (j < args_->input_dims.size() && !args_->input_dims[j].empty()) { // args include input shape
                        const vector<int64_t>* dims = &args_->input_dims[j];
                        temp_tensor_shape.SetDimCount(dims->size());
                        for (uint32_t k = 0; k < dims->size(); ++k) {
                            temp_tensor_shape.SetDim(k, dims->at(k));
                        }
                    }
                }

                *impl_pair.first->second->GetShape() = temp_tensor_shape;
                auto constant_ref = graph_->data->constants.find(edge_id);
                if (constant_ref != graph_->data->constants.end()) { // constant tensor
                    auto tensor = impl_pair.first->second.get();
                    tensor->SetDevice(&default_cpu_device_);
                    status = tensor->CopyFromHost(constant_ref->second.data.GetData());
                    if (status != RC_SUCCESS) {
                        LOG(ERROR) << "copy constant [" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
                        return status;
                    }
                }
            }
        }

        for (uint32_t j = 0; j < node->GetOutputCount(); ++j) {
            auto edge_id = node->GetOutput(j);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }
            auto edge = topo->GetEdge(edge_id);
            auto impl_pair = tensor_impls_.insert(make_pair(edge_id, unique_ptr<TensorImpl>(new TensorImpl(edge, TENSORTYPE_NORMAL))));
            auto ir_shape = data->shapes.find(edge_id);
            if (ir_shape != data->shapes.end()) {
                impl_pair.first->second->GetShape()->SetDataFormat(ir_shape->second.data_format);
            }
        }

        status = kernel->InferDims(&IOinfo);
        if (status != RC_SUCCESS) {
            isShapeUnsafe_ = true;
            LOG(DEBUG) << "Can not reshape safely for node[" << node->GetName() << "]";
            illegal_dims_.emplace(sorted_node_ids_[i]);
            std::set<uint32_t> illegal_inputs;
            for (uint32_t j = 0; j < node->GetInputCount(); ++j) {
                auto preedge_id = node->GetInput(j);
                auto prenode_edg = topo->GetEdge(preedge_id);
                if(!prenode_edg || preedge_id==INVALID_EDGEID){
                    continue;
                }
                auto prenode_id = prenode_edg->GetProducer();
                if (illegal_dims_.find(prenode_id) == illegal_dims_.end()) {
                    illegal_inputs.emplace(j);
                }
            }
            kernel->InferUnsafeDims(&IOinfo, &illegal_inputs);
        }
    }

    LOG(DEBUG) << "Create " << tensor_impls_.size() << " TensorImpl";
    return RC_SUCCESS;
}

RetCode OptGraph::FuseOperator(const utils::SharedResource& resource) {
    ir::GraphTopo* topo = graph_->topo.get();
    UpdateTopologicalSort();
    auto fs_filter_manager = FsFilterManager::Instance();

    OptKernelOptions options(graph_, info_, &resource, &tensor_impls_, &args_->tensor_quants);
    options.args = args_;
    int32_t index = LastLegalNodeIndex();
    for (int32_t i = sorted_node_ids_.size() - 1; i >= 0; --i) {
        auto node = topo->GetNode(sorted_node_ids_[i]);
        if (node && node->GetType().name == "Conv"){
            auto fuse = fs_filter_manager->FindFusion(node->GetType().name);
            if (fuse) {
                fuse->FuseNode(node, i <= index, options);
            }
        }
    }
    UpdateTopologicalSort();
    index = LastLegalNodeIndex();
    for (int32_t i = sorted_node_ids_.size() - 1; i >= 0; --i) {
        auto node = topo->GetNode(sorted_node_ids_[i]);
        if (node) {
            auto fuse = fs_filter_manager->FindFusion(node->GetType().name);
            if (fuse) {
                fuse->FuseNode(node, i <= index, options);
            }
        }
    }
    return RC_SUCCESS;
}

RetCode OptGraph::AddBridgeKernels(const utils::SharedResource& resource) {
    auto topo = graph_->topo.get();
    auto& tensor_params = args_->quant_info.tensor_params;
    uint32_t count = 0;
    OptKernelOptions options(graph_, &resource);

    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();
        if (node->GetType().name == "Bridge") {
            continue;
        }

        for (uint32_t j = 0; j < node->GetInputCount(); ++j) {
            auto edge_id = node->GetInput(j);
            if (edge_id == INVALID_EDGEID) {
                continue;
            }
            auto edge = topo->GetEdge(edge_id);
            if (edge->GetName().find("Bridge_Edge") != string::npos) {
                continue;
            }

            auto creator = OptKernelCreatorManager::GetInstance()->Find("pmx", "Bridge", 1);
            auto ret_pair = topo->AddNode("Bridge_Node_" + node->GetName() + "_" + edge->GetName());
            if (!ret_pair.second) {
                LOG(ERROR) << "create a new node for [" << edge->GetName() << "] failed.";
                return RC_OUT_OF_MEMORY;
            }
            auto new_node = ret_pair.first;
            new_node->SetType(ir::Node::Type("pmx", "Bridge", 1));
            auto bridge_kernel = unique_ptr<CudaOptKernel>((*creator)(new_node));
            ((BridgeOp*)bridge_kernel.get())->AddInternalBridgeNode(node, new_node, edge, graph_);

            auto preedge_id = new_node->GetInput(0);
            auto postedge_id = new_node->GetOutput(0);
            auto new_edge = topo->GetEdge(postedge_id);
            auto impl_pair = tensor_impls_.insert(
                make_pair(postedge_id, unique_ptr<TensorImpl>(new TensorImpl(new_edge, TENSORTYPE_NORMAL))));
            auto pre_shape = tensor_impls_.find(preedge_id)->second.get();
            impl_pair.first->second->GetShape()->Reshape(pre_shape->GetShape()->GetDims(),
                                                         pre_shape->GetShape()->GetRealDimCount());

            bridge_kernel.get()->Init(options);
            info_->kernels.emplace(new_node->GetId(), std::move(bridge_kernel));
            auto tensor_pair = tensor_params.find(edge->GetName());
            if (tensor_pair != tensor_params.end())
                tensor_params.emplace(new_edge->GetName(), tensor_pair->second);
            count++;
        }

        for (uint32_t j = 0; j < node->GetOutputCount(); ++j) {
            auto edge = topo->GetEdge(node->GetOutput(j));
            if (topo->GetOutput(edge->GetName()) != INVALID_EDGEID || // it is marked as an output node
                edge->CalcConsumerCount() == 0) { // it is an finel node for the graph
                auto creator = OptKernelCreatorManager::GetInstance()->Find("pmx", "Bridge", 1);

                auto ret_pair = topo->AddNode("Bridge_Final_" + node->GetName() + "_" + edge->GetName());
                if (!ret_pair.second) {
                    LOG(ERROR) << "create a new node for [" << edge->GetName() << "] failed.";
                    return RC_OUT_OF_MEMORY;
                }
                auto new_node = ret_pair.first;

                new_node->SetType(ir::Node::Type("pmx", "Bridge", 1));
                auto bridge_kernel = unique_ptr<CudaOptKernel>((*creator)(new_node));
                ((BridgeOp*)bridge_kernel.get())->AddFinalBridgeNode(node, new_node, edge, graph_);

                auto preedge_id = new_node->GetInput(0);
                auto postedge_id = new_node->GetOutput(0);
                auto new_edge = topo->GetEdge(preedge_id);
                auto impl_pair = tensor_impls_.insert(
                    make_pair(preedge_id, unique_ptr<TensorImpl>(new TensorImpl(new_edge, TENSORTYPE_NORMAL))));
                auto post_shape = tensor_impls_.find(postedge_id)->second.get();

                impl_pair.first->second->GetShape()->Reshape(post_shape->GetShape()->GetDims(),
                                                             post_shape->GetShape()->GetRealDimCount());
                if(post_shape->GetShape()->GetDataFormat() == DATAFORMAT_UNKNOWN){
                    post_shape->GetShape()->SetDataFormat(DATAFORMAT_NDARRAY);
                }

                bridge_kernel.get()->Init(options);
                info_->kernels.emplace(new_node->GetId(), std::move(bridge_kernel));
                auto tensor_pair = tensor_params.find(edge->GetName());
                if (tensor_pair != tensor_params.end())
                    tensor_params.emplace(new_edge->GetName(), tensor_pair->second);
                count++;
            }
        }
    }

    LOG(DEBUG) << "added " << count << " new bridge kernels";
    return RC_SUCCESS;
}

RetCode OptGraph::InitQuantization() {
    auto topo = graph_->topo.get();
    auto& graph_quants = args_->tensor_quants;
    graph_quants.resize(topo->GetCurrentEdgeIdBound());


    // Load node quant to args_->node_type
    auto& node_params = args_->quant_info.node_params;
    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();
        auto pair = node_params.find(node->GetName());
        if (pair != node_params.end()) {
            auto str = pair->second.fields.find("data_type")->second;
            if (str.content == "INT8") {
                args_->node_types.emplace(node->GetName(), DATATYPE_INT8);
            } else if (str.content == "FLOAT32") {
                args_->node_types.emplace(node->GetName(), DATATYPE_FLOAT32);
            } else if (str.content == "FLOAT16") {
                args_->node_types.emplace(node->GetName(), DATATYPE_FLOAT16);
            } else {
                LOG(ERROR) << "Not support set to such datatype: " << str.content;
            }
        }
    }
    // Load tensor quant to args_->quant_info
    auto& tensor_params = args_->quant_info.tensor_params;
    for (auto iter = topo->CreateEdgeIter(); iter->IsValid(); iter->Forward()) {
        auto edge = iter->Get();
        auto pair = tensor_params.find(edge->GetName());
        // Can not find quant info. It means quant info is not exist.
        if (pair == tensor_params.end()) {
            continue;
        }
        auto& temp_tensor_quant = graph_quants[edge->GetId()];
        auto str = pair->second.fields.find("per_channel")->second;
        temp_tensor_quant.per_channel = *(bool*)(str.content.data());
        auto bit_width = pair->second.fields.find("bit_width")->second;
        temp_tensor_quant.bit_width = *(int*)(bit_width.content.data());
        if (temp_tensor_quant.per_channel) {
            auto max_str = pair->second.fields.find("tensor_max")->second;
            auto min_str = pair->second.fields.find("tensor_min")->second;
            auto scale_str = pair->second.fields.find("scale")->second;
            auto zero_point_str = pair->second.fields.find("zero_point")->second;
            uint32_t size = max_str.content.length() / 8;
            temp_tensor_quant.scale.resize(size);
            temp_tensor_quant.zero_point.resize(size);
            for (uint32_t i = 0; i < size; ++i) {
#ifdef USE_QDQ
                temp_tensor_quant.scale[i] = *((double*)(scale_str.content.data()) + i);
                temp_tensor_quant.zero_point[i] = *((double*)(zero_point_str.content.data()) + i);
#else
                auto tensor_max = *((double*)(max_str.content.data()) + i);
                auto tensor_min = *((double*)(min_str.content.data()) + i);
                temp_tensor_quant.scale[i] =
                    (double)(tensor_max - tensor_min) / ((1 << temp_tensor_quant.bit_width) - 1);
                temp_tensor_quant.zero_point[i] = tensor_max + tensor_min;
#endif
            }
        } else {
#ifdef USE_QDQ
            auto scale_str = pair->second.fields.find("scale")->second;
            auto zero_point_str = pair->second.fields.find("zero_point")->second;
            temp_tensor_quant.scale[0] = *(double*)(scale_str.content.data());
            temp_tensor_quant.zero_point[0] = *(double*)(zero_point_str.content.data());

#else
            str = pair->second.fields.find("tensor_max")->second;
            auto tensor_max = *(double*)(str.content.data());
            str = pair->second.fields.find("tensor_min")->second;
            auto tensor_min = *(double*)(str.content.data());
            auto scale_str = pair->second.fields.find("scale")->second;
            temp_tensor_quant.scale[0] = (double)(tensor_max - tensor_min) / ((1 << temp_tensor_quant.bit_width) - 1);
            temp_tensor_quant.zero_point[0] = tensor_max + tensor_min;
#endif
        }

    }

    return RC_SUCCESS;
}

RetCode OptGraph::UpdateType() {
    auto topo = graph_->topo.get();
    auto& graph_quants = args_->tensor_quants;
    UpdateTopologicalSort();

    InputOutputInfo IOinfo;
    IOinfo.SetAcquireFunc(acquire_tensor_func_);

    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i) {
        auto node = topo->GetNode(sorted_node_ids_[i]);

        IOinfo.SetNode(node);
        CudaOptKernel* kernel = (CudaOptKernel*)(info_->kernels.find(node->GetId())->second.get());

        datatype_t kernel_type = args_->default_kernel_type;

        auto conf_pair = args_->node_types.find(node->GetName());
        if (conf_pair != args_->node_types.end()) {
            kernel_type = conf_pair->second;
        }
        if(node->GetType().name == "Conv"){
            CudaConvParam* param = (CudaConvParam*)kernel->GetParam();
            if(param->extra_param.fuse_info.types.size() > 0)
            {
                auto input_edge_id = node->GetInput(0);
                auto output_edge_id = node->GetOutput(0);
                if(graph_quants.at(input_edge_id).bit_width == 8 && graph_quants.at(output_edge_id).bit_width == 8){
                    kernel_type = DATATYPE_INT8;
                }
            }
        }
        if(node->GetType().name == "FusedMulAdd"){
            auto input0_edge_id = node->GetInput(0);
            auto input1_edge_id = node->GetInput(1);
            auto input2_edge_id = node->GetInput(2);
            auto output_edge_id = node->GetOutput(0);
            if(graph_quants.at(input0_edge_id).bit_width == graph_quants.at(input1_edge_id).bit_width
               && graph_quants.at(input0_edge_id).bit_width == graph_quants.at(input2_edge_id).bit_width
               && graph_quants.at(input0_edge_id).bit_width == graph_quants.at(output_edge_id).bit_width){
                if(graph_quants.at(input0_edge_id).bit_width == 8){
                    kernel_type = DATATYPE_INT8;
                }
            }
        }
        if (kernel_type == DATATYPE_INT8) {
            for (uint32_t j = 0; j < node->GetInputCount(); ++j) {
                auto edge_id = node->GetInput(j);
                if (edge_id == INVALID_EDGEID) {
                    continue;
                }
                auto& input_quant = graph_quants.at(edge_id);
                input_quant.type = kernel_type;
            }
            for (uint32_t j = 0; j < node->GetOutputCount(); ++j) {
                auto edge_id = node->GetOutput(j);
                if (edge_id == INVALID_EDGEID) {
                    continue;
                }
                auto& output_quant = graph_quants.at(edge_id);
                output_quant.type = kernel_type;
            }
        }
        if(gIsOpKeepOnnxPrecision){
            kernel_type = DATATYPE_UNKNOWN;
        }
        auto status = kernel->InferType(&IOinfo, &graph_quants, kernel_type);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Set type for node[" << node->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    // update shapes of outputs
    if(topo->GetInputCount()==0 && sorted_node_ids_.size()==0)
    {
      //There is no need to update shapes of outputs. There is only constant node for output.
    }
    else{

        auto data = graph_->data.get();
        for (uint32_t i = 0; i < topo->GetOutputCount(); ++i) {
            auto eid = topo->GetOutput(i);
            auto tensor_ref = tensor_impls_.find(eid);
            if (tensor_ref == tensor_impls_.end()) {
                auto edge = topo->GetEdge(eid);
                LOG(ERROR) << "cannot find instance of edge[" << edge->GetName() << "]";
                return RC_NOT_FOUND;
            }
            auto shape = tensor_ref->second->GetShape();

            auto ir_shape_ref = data->shapes.find(eid);
            if (ir_shape_ref != data->shapes.end()) {
                shape->SetDataType(ir_shape_ref->second.data_type);
            } else {
                if (shape->GetDataType() == DATATYPE_FLOAT16 || shape->GetDataType() == DATATYPE_INT8) {
                    shape->SetDataType(DATATYPE_FLOAT32);
                }
            }
        }

    }

    return RC_SUCCESS;
}

RetCode OptGraph::SelectAlgos(const utils::SharedResource& resource, CudaDevice* device) {
    auto topo = graph_->topo.get();
    auto& graph_quants = args_->tensor_quants;

    OptKernelOptions options(graph_, info_, &resource, args_, compile_set_, device, &tensor_impls_, &graph_quants);
    UpdateTopologicalSort();

    AlgoGraph algo_graph(topo);
    // calculate the least time consuming
    for (uint32_t i = 0; i < sorted_node_ids_.size(); ++i) {
        auto node = topo->GetNode(sorted_node_ids_[i]);
        CudaOptKernel* kernel = (CudaOptKernel*)(info_->kernels.find(node->GetId())->second.get());

        auto status = algo_graph.CreateNode(node, kernel);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Create the node[" << node->GetName() << "] failed." << GetRetCodeStr(status);
            return status;
        }
        status = algo_graph.UpdateNode(node, options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Update the node[" << node->GetName() << "] failed." << GetRetCodeStr(status);
            return status;
        }
    }

    // select algorithm method and its format
    for (int32_t i = sorted_node_ids_.size() - 1; i >= 0; --i) {
        auto node_id = sorted_node_ids_[i];
        auto node = topo->GetNode(node_id);

        auto kernel = info_->kernels.find(node_id);
        if (kernel == info_->kernels.end()) {
            LOG(ERROR) << "Can not find kernel[" << node->GetName() << "].";
            return RC_NOT_FOUND;
        }
        auto status = algo_graph.DetermineNode((CudaOptKernel*)(kernel->second.get()), options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Determine the node[" << node->GetName() << "] failed." << GetRetCodeStr(status);
            return status;
        }
    }

    algo_graph.Delete();
    return RC_SUCCESS;
}

RetCode OptGraph::LoadConstants(CudaDevice* device) {
    auto topo = graph_->topo.get();
    auto graph_data = graph_->data.get();
    auto& graph_quants = args_->tensor_quants;

    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();
        if (node->GetType().name != "Bridge") {
            continue;
        }

        auto preedge_id = node->GetInput(0);
        auto postedge_id = node->GetOutput(0);
        const TensorShape& preshape = *tensor_impls_.find(preedge_id)->second->GetShape();
        const TensorShape& postshape = *tensor_impls_.find(postedge_id)->second->GetShape();

        auto constant_ref = graph_data->constants.find(preedge_id);
        if (constant_ref != graph_data->constants.end() &&
            info_->constants.find(preedge_id) == info_->constants.end()) { // constant tensor
            RuntimeConstantInfo constant_info;
            constant_info.SetDevice(device);
            constant_info.Reshape(postshape);

            auto status = constant_info.ReallocBuffer();
            if (status != RC_SUCCESS && postshape.CalcBytesIncludingPadding() > 0) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            auto converter = (CudaDataConverter*)device->GetDataConverter();
            status =
                converter->ConvertFromHost(&constant_info.GetBufferDesc(), postshape, graph_quants[postedge_id],
                                           constant_ref->second.data.GetData(), preshape, graph_quants[preedge_id]);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
                return status;
            }

            info_->constants[preedge_id] = std::move(constant_info);
            *tensor_impls_.find(preedge_id)->second->GetShape() = postshape;
            graph_quants[preedge_id] = graph_quants[postedge_id];
        }
    }

    // load the rest of constants that are not quantized
    auto status = utils::LoadConstants(*graph_, device, &info_->constants);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load constants failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode OptGraph::AdjustGemmInOutFormat(const utils::SharedResource& resource, CudaDevice* device) {
    auto topo = graph_->topo.get();
    //OptKernelOptions options(graph_, info_, &resource, &tensor_impls_, &args_->tensor_quants);
    auto& graph_quants = args_->tensor_quants;
    OptKernelOptions options(graph_, info_, &resource, args_, compile_set_, device, &tensor_impls_, &graph_quants);
    options.param = nullptr;
    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();
        if (node->GetType().name != "Gemm" || isShapeUnsafe_) {
            continue;
        }
        auto node_id = node->GetId();
        auto preedge_id = node->GetInput(0);
        auto postedge_id = node->GetOutput(0);
        TensorShape& preshape = *tensor_impls_.find(preedge_id)->second->GetShape();
        TensorShape& postshape = *tensor_impls_.find(postedge_id)->second->GetShape();
        auto predege = topo->GetEdge(preedge_id);
        auto prebridge_node_id = predege->GetProducer();
        auto prebridge_node = topo->GetNode(prebridge_node_id);
        const TensorShape& prebridge_preshape = *tensor_impls_.find(prebridge_node->GetInput(0))->second->GetShape();
        if(prebridge_preshape.GetDimCount() == 2
           && (
               #if 0
               (prebridge_preshape.GetDataType() == DATATYPE_INT8 && prebridge_preshape.GetDim(1) % 16 == 0) ||
               #endif
               (prebridge_preshape.GetDataType() == DATATYPE_FLOAT16 && prebridge_preshape.GetDim(1) % 8 == 0)))
        {
            preshape.SetDataFormat(prebridge_preshape.GetDataFormat());
            auto pre_bridge_kernel = info_->kernels.find(prebridge_node_id);
            if (pre_bridge_kernel == info_->kernels.end()) {
                LOG(ERROR) << "cannot find gemm prebridge kernel for node[" << prebridge_node->GetName() << "]";
                return RC_NOT_FOUND;
            }
            auto pre_bridge_cudakernel = (CudaOptKernel*)pre_bridge_kernel->second.get();
            pre_bridge_cudakernel->Finalize(options);
        }
        auto postedge = topo->GetEdge(postedge_id);
        auto postbridge_node_id = postedge->CreateConsumerIter().Get();
        auto postbridge_node = topo->GetNode(postbridge_node_id);
        const TensorShape& postbridge_postshape = *tensor_impls_.find(postbridge_node->GetOutput(0))->second->GetShape();
        if(postbridge_postshape.GetDimCount() == 2
           && (
               #if 0
               (postbridge_postshape.GetDataType() == DATATYPE_INT8 && postbridge_postshape.GetDim(1) % 16 == 0) ||
               #endif
               (postbridge_postshape.GetDataType() == DATATYPE_FLOAT16 && postbridge_postshape.GetDim(1) % 8 == 0)))
        {
            postshape.SetDataFormat(postbridge_postshape.GetDataFormat());
            auto gemm_kernel = info_->kernels.find(node->GetId());
            if (gemm_kernel == info_->kernels.end()) {
                LOG(ERROR) << "cannot find gemm kernel for node[" << node->GetName() << "]";
                return RC_NOT_FOUND;
            }
            auto gemm_cudakernel = (CudaOptKernel*)gemm_kernel->second.get();
            gemm_cudakernel->Finalize(options);
        }
    }
    return RC_SUCCESS;
}

RetCode OptGraph::DeleteBridgeKernels() {
    auto topo = graph_->topo.get();
    auto& graph_quants = args_->tensor_quants;
    uint32_t count = 0;

    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();

        if (node->GetType().name != "Bridge") {
            continue;
        }

        auto bridge_kernel = info_->kernels.find(node->GetId());
        if (bridge_kernel == info_->kernels.end()) {
            LOG(ERROR) << "cannot find bridge kernel for node[" << node->GetName() << "]";
            return RC_NOT_FOUND;
        }

        auto node_id = node->GetId();
        auto status =
            ((BridgeOp*)(bridge_kernel->second.get()))->DeleteBridgeNode(node, graph_, &tensor_impls_, &graph_quants);
        if (status == RC_SUCCESS) {
            info_->kernels.erase(node_id);
            count++;
        }
    }

    LOG(DEBUG) << "deleted " << count << " bridge kernels";
    return RC_SUCCESS;
}

RetCode OptGraph::DeleteCastKernels() {
    auto topo = graph_->topo.get();
    auto& graph_quants = args_->tensor_quants;
    uint32_t count = 0;

    for (auto iter = topo->CreateNodeIter(); iter->IsValid(); iter->Forward()) {
        auto node = iter->Get();

        if (node->GetType().name != "Cast") {
            continue;
        }

        auto cast_kernel = info_->kernels.find(node->GetId());
        if (cast_kernel == info_->kernels.end()) {
            LOG(ERROR) << "cannot find cast kernel for node[" << node->GetName() << "]";
            return RC_NOT_FOUND;
        }

        auto node_id = node->GetId();
        auto status =
            ((CastOp*)(cast_kernel->second.get()))->DeleteCastNode(node, graph_, &tensor_impls_, &graph_quants);
        if (status == RC_SUCCESS) {
            info_->kernels.erase(node_id);
            count++;
        }
    }

    LOG(DEBUG) << "deleted " << count << " cast kernels";
    return RC_SUCCESS;
}

RetCode OptGraph::DoOptimize(const utils::SharedResource& resource, CudaDevice* dev) {
    auto status = graphPreOptProcess(resource);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init kernels failed: " << GetRetCodeStr(status);
        return status;
    }
    status = InitKernels();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init kernels failed: " << GetRetCodeStr(status);
        return status;
    }

    status = findGraphInputConv(resource);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "findGraphInputConv failed: " << GetRetCodeStr(status);
        return status;
    }

    status = UpdateDims(resource);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Update dims failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitQuantization();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init quantization failed: " << GetRetCodeStr(status);
        return status;
    }

    DumpGraphInfo("DUMP_ORIGINAL");

    if(!gIsOpKeepOnnxPrecision)
        status = FuseOperator(resource);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Fuse operators failed: " << GetRetCodeStr(status);
        return status;
    }

    DumpGraphInfo("DUMP_ORIGINAL_FUSED");

    status = AddBridgeKernels(resource);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Add Bridge nodes failed: " << GetRetCodeStr(status);
        return status;
    }

    DumpGraphInfo("DUMP_ADD_BRIDGE");

    status = InitQuantization();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init quantization failed: " << GetRetCodeStr(status);
        return status;
    }

    status = UpdateType();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Update type failed: " << GetRetCodeStr(status);
        return status;
    }

    status = SelectAlgos(resource, dev);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Selec algos for each kernel failed: " << GetRetCodeStr(status);
        return status;
    }

    status = LoadConstants(dev);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Load constant tensors failed: " << GetRetCodeStr(status);
        return status;
    }

    status = AdjustGemmInOutFormat(resource, dev);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "AdjustGemmInOutFormat failed: " << GetRetCodeStr(status);
        return status;
    }

    status = DeleteBridgeKernels();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Delete some of bridge nodes failed: " << GetRetCodeStr(status);
        return status;
    }

    DumpGraphInfo("DUMP_DELETE_BRIDGE");

    if(!gIsOpKeepOnnxPrecision)
        status = FuseOperator(resource);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Fuse operators failed: " << GetRetCodeStr(status);
        return status;
    }

    status = DeleteCastKernels();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Delete some of cast nodes failed: " << GetRetCodeStr(status);
        return status;
    }

    DumpGraphInfo("DUMP_FINAL_FUSED");

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
