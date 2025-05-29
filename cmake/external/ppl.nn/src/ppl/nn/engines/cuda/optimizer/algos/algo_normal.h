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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_NORMAL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_NORMAL_H_

#include <stdlib.h>
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/algos/algorithm.h"
#include "ppl/nn/oputils/broadcast.h"
extern bool gSupportNc1hwc0;
using namespace ppl::common;
namespace ppl { namespace nn { namespace cuda {

class NormalAlgorithm : public Algorithm {
public:
    NormalAlgorithm() {
        std::set<dataformat_t> ndarray{DATAFORMAT_NDARRAY};
        std::set<dataformat_t> nhwc8{DATAFORMAT_NHWC8};
        std::set<dataformat_t> nhwc8_nd{DATAFORMAT_NHWC8, DATAFORMAT_NDARRAY};
        std::set<dataformat_t> nhwc16{DATAFORMAT_NHWC16};
        std::set<dataformat_t> nhwc16_nd{DATAFORMAT_NHWC16, DATAFORMAT_NDARRAY};
        std::set<dataformat_t> nhwc{DATAFORMAT_NHWC};
        std::set<dataformat_t> nhwc_nd{DATAFORMAT_NHWC, DATAFORMAT_NDARRAY};
        std::set<dataformat_t> nchw8{DATAFORMAT_NCHW8};
        std::set<dataformat_t> nchw8_nd{DATAFORMAT_NCHW8, DATAFORMAT_NDARRAY};
        std::set<dataformat_t> nchw16{DATAFORMAT_NCHW16};
        std::set<dataformat_t> nchw16_nd{DATAFORMAT_NCHW16, DATAFORMAT_NDARRAY};

        ndarray_formats_.emplace(DATAFORMAT_NDARRAY, ndarray);
        inherited_formats_.emplace(DATAFORMAT_NDARRAY, ndarray);
        inherited_formats_.emplace(DATAFORMAT_NHWC8, nhwc8);
        inherited_formats_.emplace(DATAFORMAT_NHWC16, nhwc16);
    #ifdef PPLNN_USE_DNN
        inherited_formats_.emplace(DATAFORMAT_NHWC, nhwc);
        inherited_formats_.emplace(DATAFORMAT_NCHW8, nchw8);
        inherited_formats_.emplace(DATAFORMAT_NCHW16, nchw16);
    #endif
        arbitrary_formats_.emplace(DATAFORMAT_NDARRAY, ndarray);
        arbitrary_formats_.emplace(DATAFORMAT_NHWC8, ndarray);
        arbitrary_formats_.emplace(DATAFORMAT_NHWC16, ndarray);
    #ifdef PPLNN_USE_DNN
        arbitrary_formats_.emplace(DATAFORMAT_NHWC, ndarray);
        arbitrary_formats_.emplace(DATAFORMAT_NCHW8, ndarray);
        arbitrary_formats_.emplace(DATAFORMAT_NCHW16, ndarray);
    #endif
        inherited_nd_formats_.emplace(DATAFORMAT_NDARRAY, ndarray);
        inherited_nd_formats_.emplace(DATAFORMAT_NHWC8, nhwc8_nd);
        inherited_nd_formats_.emplace(DATAFORMAT_NHWC16, nhwc16_nd);
    #ifdef PPLNN_USE_DNN
        inherited_nd_formats_.emplace(DATAFORMAT_NHWC, nhwc_nd);
        inherited_nd_formats_.emplace(DATAFORMAT_NCHW8, nchw8_nd);
        inherited_nd_formats_.emplace(DATAFORMAT_NCHW16, nchw16_nd);
    #endif
    }

    bool DimsAreAllOne(const int64_t *dims, uint32_t dim_count){
        bool result = true;
        for(uint32_t i = 0; i < dim_count; i++){
            if(dims[i] != 1)
                result = false;
        }
        return result;
    }

    void GetAttrParam(void*& param) const override {
        return;
    };
    void DeleteAttrParam(void*& param) override {
        return;
    };
    bool IsSupported(const ir::Node* node, const OptKernelOptions& options, dataformat_t input_format) const override{
        if (inherited_set_.find(node->GetType().name) != inherited_set_.end() ||
            inherited_nd_set_.find(node->GetType().name) != inherited_nd_set_.end()) {
            char *env_ptr = getenv("MACART_OP_FORMAT");
            if(env_ptr != nullptr){
                string op_format_str(env_ptr);
                if(op_format_str == "ND" && input_format == DATAFORMAT_NDARRAY){
                    return true;
                }else if(op_format_str == "NHWC8" && input_format == DATAFORMAT_NHWC8){
                    return true;
                }
                else if(op_format_str == "NHWC16" && input_format == DATAFORMAT_NHWC16){
                    return true;
                }
                else if(op_format_str == "NHWC" && input_format == DATAFORMAT_NHWC){
                    return true;
                }
                else if(gSupportNc1hwc0 && op_format_str == "NCHW16" && input_format == DATAFORMAT_NCHW16){
                    return true;
                }
                else if(gSupportNc1hwc0 && op_format_str == "NCHW8" && input_format == DATAFORMAT_NCHW8){
                    return true;
                }
                if(!isFormatSupport(op_format_str))
                    LOG(WARNING) << "MACART_OP_FORMAT only support ND/NHWC/NHWC8/NHWC16/NCHW8/NCHW16, but set:" << env_ptr;
                return false;
            }
            const TensorShape& tensor0_shape = *options.tensors->find(node->GetInput(0))->second->GetShape();
            if((input_format == DATAFORMAT_NHWC16 || input_format == DATAFORMAT_NCHW16) && tensor0_shape.GetDataType() == DATATYPE_FLOAT16)
                return false;
            if((input_format == DATAFORMAT_NHWC8 || input_format == DATAFORMAT_NCHW8) && tensor0_shape.GetDataType() == DATATYPE_INT8)
                return false;
        }
        return true;
    }
    const std::map<dataformat_t, std::set<dataformat_t>> Getformats(const std::string& type_name) const override {
        if (inherited_nd_set_.find(type_name) != inherited_nd_set_.end()) {
        #ifdef PPLNN_USE_DNN
            if(gSupportNc1hwc0 == false){
                std::map<dataformat_t, std::set<dataformat_t>> inherited_nd_formats_tmp = inherited_nd_formats_;
                inherited_nd_formats_tmp.erase(DATAFORMAT_NCHW8);
                inherited_nd_formats_tmp.erase(DATAFORMAT_NCHW16);
                return inherited_nd_formats_tmp;
            }else{
                std::map<dataformat_t, std::set<dataformat_t>> inherited_nd_formats_tmp = inherited_nd_formats_;
                inherited_nd_formats_tmp.erase(DATAFORMAT_NHWC8);
                inherited_nd_formats_tmp.erase(DATAFORMAT_NHWC16);
                return inherited_nd_formats_tmp;
            }
        #endif
            return inherited_nd_formats_;
        }
        if (inherited_set_.find(type_name) != inherited_set_.end()) {
        #ifdef PPLNN_USE_DNN
            if(gSupportNc1hwc0 == false){
                std::map<dataformat_t, std::set<dataformat_t>> inherited_formats_tmp = inherited_formats_;
                inherited_formats_tmp.erase(DATAFORMAT_NCHW8);
                inherited_formats_tmp.erase(DATAFORMAT_NCHW16);
                return inherited_formats_tmp;
            }else{
                std::map<dataformat_t, std::set<dataformat_t>> inherited_formats_tmp = inherited_formats_;
                inherited_formats_tmp.erase(DATAFORMAT_NHWC8);
                inherited_formats_tmp.erase(DATAFORMAT_NHWC16);
                return inherited_formats_tmp;
            }
        #endif
            return inherited_formats_;
        }
        if (arbitrary_set_.find(type_name) != arbitrary_set_.end()) {
        #ifdef PPLNN_USE_DNN
            if(gSupportNc1hwc0 == false){
                std::map<dataformat_t, std::set<dataformat_t>> arbitrary_formats_tmp = arbitrary_formats_;
                arbitrary_formats_tmp.erase(DATAFORMAT_NCHW8);
                arbitrary_formats_tmp.erase(DATAFORMAT_NCHW16);
                return arbitrary_formats_tmp;
            }else{
                std::map<dataformat_t, std::set<dataformat_t>> arbitrary_formats_tmp = arbitrary_formats_;
                arbitrary_formats_tmp.erase(DATAFORMAT_NHWC8);
                arbitrary_formats_tmp.erase(DATAFORMAT_NHWC16);
                return arbitrary_formats_tmp;
            }
        #endif
            return arbitrary_formats_;
        }
        return ndarray_formats_;
    }

    double ExcuteTimer(const ir::Node* node, OptKernelOptions& options) override;
    RetCode ModifyParam(ir::Node* node, OptKernelOptions& options) override {
        if (elementwise_op_set_.find(node->GetType().name) != elementwise_op_set_.end()){
            const TensorShape& tensor0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
            const TensorShape& tensor1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
            auto data = options.graph->data.get();
            auto topo = options.graph->topo.get();
            auto input0_edge = topo->GetEdge(node->GetInput(0));
            auto input1_edge = topo->GetEdge(node->GetInput(1));
            auto input0_node = topo->GetNode(input0_edge->GetProducer());
            auto input1_node = topo->GetNode(input1_edge->GetProducer());
            auto input0_iter = data->constants.find(input0_node->GetInput(0));
            auto input1_iter = data->constants.find(input1_node->GetInput(0));
            if ((input0_iter == data->constants.end() && input1_iter == data->constants.end()) ||
                (input0_iter != data->constants.end() && input1_iter != data->constants.end())){
                return RC_SUCCESS;
            }
            int const_index = (input0_iter != data->constants.end() ? 0 : 1);
            int var_index = 1 - const_index;
            auto post_const_edge_id = node->GetInput(const_index);
            auto var_edge_id = node->GetInput(var_index);
            auto post_const_shape = *options.tensors->find(post_const_edge_id)->second->GetShape();
            auto var_shape = *options.tensors->find(var_edge_id)->second->GetShape();
            if(post_const_shape.GetRealDimCount() > 0 && DimsAreAllOne(post_const_shape.GetDims(), post_const_shape.GetDimCount())){
                std::vector<int64_t> dim_const = {1};
                post_const_shape.Reshape(dim_const);
                post_const_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                *options.tensors->find(post_const_edge_id)->second->GetShape() = post_const_shape;
                auto const_edge = topo->GetEdge(node->GetInput(const_index));
                auto const_node = topo->GetNode(const_edge->GetProducer());
                auto pre_const_edge_id = const_node->GetInput(0);
                post_const_shape.SetDataType(options.tensors->find(pre_const_edge_id)->second->GetShape()->GetDataType());
                post_const_shape.SetDataFormat(options.tensors->find(pre_const_edge_id)->second->GetShape()->GetDataFormat());
                *options.tensors->find(pre_const_edge_id)->second->GetShape() = post_const_shape;
                return RC_SUCCESS;
            }
            if(var_shape.GetDimCount() != post_const_shape.GetDimCount() && var_shape.GetDataFormat() != DATAFORMAT_NDARRAY){
                if(post_const_shape.GetDimCount() <= 1){
                    return RC_SUCCESS;
                }else{
                    MultiDirectionalBroadCaster multi_bc;
                    multi_bc.SetInputTensorShapes(var_shape, post_const_shape);
                    multi_bc.CalcBroadCast();
                    if (!multi_bc.CanBroadCast()) {
                        return RC_SUCCESS;
                    }
                    auto& broadcast_shape = multi_bc.OutputTensorShape();
                    if(var_shape.GetRealDimCount() != broadcast_shape.GetRealDimCount()){
                        return RC_SUCCESS;
                    }
                    for(int i = 0; i < var_shape.GetRealDimCount(); i++){
                        if(var_shape.GetDim(i) != broadcast_shape.GetDim(i)){
                            return RC_SUCCESS;
                        }
                    }
                    auto const_dim_size = post_const_shape.GetRealDimCount();
                    auto broadcast_dim_size = broadcast_shape.GetRealDimCount();
                    std::vector<int64_t> dim_const;
                    for(int i = 0; i < broadcast_dim_size - const_dim_size; i++)
                        dim_const.push_back(1);
                    for(int i = 0; i < const_dim_size; i++)
                        dim_const.push_back(post_const_shape.GetDim(i));
                    post_const_shape.Reshape(dim_const);
                }
                *options.tensors->find(post_const_edge_id)->second->GetShape() = post_const_shape;
                auto const_edge = topo->GetEdge(node->GetInput(const_index));
                auto const_node = topo->GetNode(const_edge->GetProducer());
                auto pre_const_edge_id = const_node->GetInput(0);
                post_const_shape.SetDataType(options.tensors->find(pre_const_edge_id)->second->GetShape()->GetDataType());
                post_const_shape.SetDataFormat(options.tensors->find(pre_const_edge_id)->second->GetShape()->GetDataFormat());
                *options.tensors->find(pre_const_edge_id)->second->GetShape() = post_const_shape;
            } else{
                auto data = options.graph->data.get();
                auto topo = options.graph->topo.get();
                auto input0_edge = topo->GetEdge(node->GetInput(0));
                auto input0_node = topo->GetNode(input0_edge->GetProducer());
                auto input0_iter = data->constants.find(input0_node->GetInput(0));
                if (input0_iter != data->constants.end()){
                    auto input0_shape = *options.tensors->find(input0_node->GetInput(0))->second->GetShape();
                    if(input0_shape.GetDimCount() == 3){
                        int i = 0;
                        for(; i < input0_shape.GetRealDimCount(); i++){
                            if(input0_shape.GetDim(i) == input0_shape.CalcElementsExcludingPadding()){
                                break;;
                            }
                        }
                        if(i != input0_shape.GetDimCount()){
                            std::vector<int64_t> dim_const;
                            dim_const.push_back(1);
                            for(int i = 0; i < input0_shape.GetRealDimCount(); i++){
                                dim_const.push_back(input0_shape.GetDim(i));
                            }
                            input0_shape.Reshape(dim_const);
                            *options.tensors->find(input0_node->GetInput(0))->second->GetShape() = input0_shape;
                            input0_shape.SetDataFormat(options.tensors->find(node->GetInput(0))->second->GetShape()->GetDataFormat());
                            input0_shape.SetDataType(options.tensors->find(node->GetInput(0))->second->GetShape()->GetDataType());
                            *options.tensors->find(node->GetInput(0))->second->GetShape() = input0_shape;
                        }
                    }
                }
                auto input1_edge = topo->GetEdge(node->GetInput(1));
                auto input1_node = topo->GetNode(input1_edge->GetProducer());
                auto input1_iter = data->constants.find(input1_node->GetInput(0));
                if (input1_iter != data->constants.end()){
                    auto input1_shape = *options.tensors->find(input1_node->GetInput(0))->second->GetShape();
                    if(input1_shape.GetDimCount() == 3){
                        int i = 0;
                        for(; i < input1_shape.GetRealDimCount(); i++){
                            if(input1_shape.GetDim(i) == input1_shape.CalcElementsExcludingPadding()){
                                break;;
                            }
                        }
                        if(i != input1_shape.GetDimCount()){
                            std::vector<int64_t> dim_const;
                            dim_const.push_back(1);
                            for(int i = 0; i < input1_shape.GetRealDimCount(); i++){
                                dim_const.push_back(input1_shape.GetDim(i));
                            }
                            input1_shape.Reshape(dim_const);
                            *options.tensors->find(input1_node->GetInput(0))->second->GetShape() = input1_shape;
                            input1_shape.SetDataFormat(options.tensors->find(node->GetInput(1))->second->GetShape()->GetDataFormat());
                            input1_shape.SetDataType(options.tensors->find(node->GetInput(1))->second->GetShape()->GetDataType());
                            *options.tensors->find(node->GetInput(1))->second->GetShape() = input1_shape;
                        }
                    }
                }
            }
        }
        return RC_SUCCESS;
    }
    void ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                        ppl::common::dataformat_t input_format, ppl::common::dataformat_t output_format) override;

private:
    std::map<dataformat_t, std::set<dataformat_t>> ndarray_formats_;
    std::map<dataformat_t, std::set<dataformat_t>> nhwc8_formats_;
    std::map<dataformat_t, std::set<dataformat_t>> inherited_formats_;
    std::map<dataformat_t, std::set<dataformat_t>> arbitrary_formats_;
    std::map<dataformat_t, std::set<dataformat_t>> inherited_nd_formats_;
    dataformat_t input_format_ = 0;
    dataformat_t output_format_ = 0;
    std::set<std::string> inherited_set_{"Add",
                                         "Mul",
                                         "Div",
                                         "Sub",
                                         "Relu",
                                         "Clip",
                                         "LeakyRelu",
                                         "Cast",
                                         //"MaxPool",
                                         "GlobalMaxPool",
                                         "AveragePool",
                                         "GlobalAveragePool",
                                         "BatchNormalization",
                                         "InstanceNormalization",
                                         "Slice",
                                         "Split",
                                         //"Sigmoid",
                                         "ChannelShuffle",
                                         "HardSigmoid",
                                         "Swish",
                                         "BitwiseAnd",
                                         "BitwiseNot",
                                         "BitwiseOr",
                                         "BitwiseXor",
                                         "Selu",
                                         "Elu",
                                         "Celu",
                                         "Gelu",
                                         "Mish",
                                         "HardMish",
                                         "HardSwish",
                                         "Softsign",
                                         "Shrink",
                                         "ThresholdedRelu",
                                         "IsInf",
                                         "IsNaN",
                                         "Resize",
                                         "ReduceMean",
                                         "Identity",
                                         "FusedMulAdd",
                                    //  "Reshape"
                                         };
    std::set<std::string> arbitrary_set_{"Shape"};
    std::set<std::string> inherited_nd_set_{"Sigmoid", "MaxPool"};
    std::set<std::string> elementwise_op_set_{"Add", "Mul", "Sub", "Div"};
    std::set<std::string> keep_const_input_nd_op_set_{"Resize"};
};

}}} // namespace ppl::nn::cuda

#endif
