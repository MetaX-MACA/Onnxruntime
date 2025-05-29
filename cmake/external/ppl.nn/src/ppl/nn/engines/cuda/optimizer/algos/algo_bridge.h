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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_BRIDGE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_BRIDGE_H_

#include "ppl/nn/engines/cuda/optimizer/algos/algorithm.h"
extern bool gSupportNc1hwc0;
using namespace ppl::common;
namespace ppl { namespace nn { namespace cuda {

class BridgeAlgorithm : public Algorithm {
public:
    BridgeAlgorithm() {
    #ifdef PPLNN_USE_DNN
        std::set<dataformat_t> all_formats{DATAFORMAT_NDARRAY, DATAFORMAT_NHWC8, DATAFORMAT_NHWC16, DATAFORMAT_NHWC, DATAFORMAT_NCHW8, DATAFORMAT_NCHW16};
    #else
        std::set<dataformat_t> all_formats{DATAFORMAT_NDARRAY, DATAFORMAT_NHWC8, DATAFORMAT_NHWC16};
    #endif
        bridge_formats_.emplace(DATAFORMAT_NDARRAY, all_formats);
        bridge_formats_.emplace(DATAFORMAT_NHWC8, all_formats);
        bridge_formats_.emplace(DATAFORMAT_NHWC16, all_formats);
    #ifdef PPLNN_USE_DNN
        bridge_formats_.emplace(DATAFORMAT_NHWC, all_formats);
        bridge_formats_.emplace(DATAFORMAT_NCHW8, all_formats);
        bridge_formats_.emplace(DATAFORMAT_NCHW16, all_formats);
    #endif
    }

    void GetAttrParam(void*& param) const override {
        return;
    };
    void DeleteAttrParam(void*& param) override {
        return;
    };

    bool IsSupported(const ir::Node* node, const OptKernelOptions& options, dataformat_t input_format) const {
        const TensorShape& tensor0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
        if (tensor0.GetDataType() == ppl::common::DATATYPE_FLOAT16){
            if ((input_format == DATAFORMAT_NHWC16 || input_format == DATAFORMAT_NCHW16))
                return false;
            else
                return true;
        }
        if (tensor0.GetDataType() == ppl::common::DATATYPE_INT8) {
            if (input_format == DATAFORMAT_NHWC8 || input_format == DATAFORMAT_NCHW8)
                return false;
            else
                return true;
        }
        return true;
    }

    const std::map<dataformat_t, std::set<dataformat_t>> Getformats(const std::string& type_name) const override {
    #ifdef PPLNN_USE_DNN
        if(gSupportNc1hwc0 == false){
            std::map<dataformat_t, std::set<dataformat_t>> bridge_formats_tmp = bridge_formats_;
            bridge_formats_tmp.erase(DATAFORMAT_NCHW8);
            bridge_formats_tmp.erase(DATAFORMAT_NCHW16);
            for(auto iter = bridge_formats_tmp.begin(); iter != bridge_formats_tmp.end(); iter++){
                iter->second.erase(DATAFORMAT_NCHW8);
                iter->second.erase(DATAFORMAT_NCHW16);
            }
            return bridge_formats_tmp;
        }else{
            std::map<dataformat_t, std::set<dataformat_t>> bridge_formats_tmp = bridge_formats_;
            bridge_formats_tmp.erase(DATAFORMAT_NHWC8);
            bridge_formats_tmp.erase(DATAFORMAT_NHWC16);
            for(auto iter = bridge_formats_tmp.begin(); iter != bridge_formats_tmp.end(); iter++){
                iter->second.erase(DATAFORMAT_NHWC8);
                iter->second.erase(DATAFORMAT_NHWC16);
            }
            return bridge_formats_tmp;
        }
    #endif
        return bridge_formats_;
    }

    double ExcuteTimer(const ir::Node* node, OptKernelOptions& options) override;

    RetCode ModifyParam(ir::Node* node, OptKernelOptions& options) override {
        auto edge_i = node->GetInput(0);
        auto edge_o = node->GetOutput(0);
        #if 0 //need in future
        auto topo = options.graph->topo.get();
        auto edge_output = topo->GetEdge(edge_o);
        if(edge_output->CalcConsumerCount() > 0){
            auto consumer = edge_output->CreateConsumerIter().Get();
            auto nextnode = topo->GetNode(consumer);
            if(nextnode->GetType().name == "Conv"){
                return RC_SUCCESS;
            }
        }
        #endif
        auto shape_i = options.tensors->find(edge_i)->second->GetShape();
        auto shape_o = options.tensors->find(edge_o)->second->GetShape();
        int dim_count = shape_i->GetDimCount();

        if (shape_i->GetDataFormat() == DATAFORMAT_NDARRAY &&
            (shape_o->GetDataFormat() == DATAFORMAT_NHWC8 || shape_o->GetDataFormat() == DATAFORMAT_NHWC16
             || shape_o->GetDataFormat() == DATAFORMAT_NCHW8 || shape_o->GetDataFormat() == DATAFORMAT_NCHW16) &&
            dim_count == 3) {
            int dst_dim_count = 4;
            vector<int64_t> shape_data(dst_dim_count, 1);
            for (int i = 1; i < dst_dim_count; i++) {
                shape_data[i] = shape_i->GetDim(i - 1);
            }
            shape_i->Reshape(shape_data.data(), dst_dim_count);
            shape_o->Reshape(shape_data.data(), dst_dim_count);
        }
        return RC_SUCCESS;
    }

    void ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                        ppl::common::dataformat_t input_format, ppl::common::dataformat_t output_format) override;

private:
    std::map<dataformat_t, std::set<dataformat_t>> bridge_formats_;
    dataformat_t input_format_ = 0;
    dataformat_t output_format_ = 0;
};

}}} // namespace ppl::nn::cuda

#endif
