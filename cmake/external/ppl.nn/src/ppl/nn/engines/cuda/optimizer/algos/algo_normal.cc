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

#include "ppl/nn/engines/cuda/optimizer/algos/algo_normal.h"

bool isOpKeepOnnxPrecision(void){
    char *env_ptr = getenv("MACART_OP_KEEP_ONNX_PRECISION");
        if(env_ptr != nullptr){
            string op_keep_fp32_str(env_ptr);
            if(op_keep_fp32_str == "ON"){
                return true;
            }
        }
        return false;
}
bool gIsOpKeepOnnxPrecision = isOpKeepOnnxPrecision();

bool isSupportNc1hwc0(void){
    char *env_ptr = getenv("MACART_USE_FORMAT_NC1HWC0");
        if(env_ptr != nullptr){
            string op_format_str(env_ptr);
            if(op_format_str == "ON"){
                return true;
            }
        }
        return false;
}
bool gSupportNc1hwc0 = isSupportNc1hwc0();

bool isFormatSupport(std::string df_str){
    bool flag = df_str == GetDataFormatStr(DATAFORMAT_NDARRAY) || df_str == GetDataFormatStr(DATAFORMAT_NHWC);
    if(gSupportNc1hwc0)
        flag =  flag || df_str == GetDataFormatStr(DATAFORMAT_NCHW8) || df_str == GetDataFormatStr(DATAFORMAT_NCHW16);
    else
        flag =  flag || df_str == GetDataFormatStr(DATAFORMAT_NHWC8) || df_str == GetDataFormatStr(DATAFORMAT_NHWC16);
    return flag;
}

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

double NormalAlgorithm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    if (inherited_nd_set_.find(node->GetType().name) != inherited_nd_set_.end()) {
        auto preedge_id = node->GetInput(0);
        TensorShape preshape = *options.tensors->find(preedge_id)->second.get()->GetShape();
        if(output_format_ != DATAFORMAT_NDARRAY && output_format_ != DATAFORMAT_NHWC){
            if(preshape.GetDimCount() >= 3 && preshape.GetDim(1) > preshape.GetPadding1(1)){
                if(input_format_ != DATAFORMAT_NDARRAY
                   && ((preshape.GetDataType() == DATATYPE_FLOAT16 && preshape.GetDim(1) >= 4)
                        || (preshape.GetDataType() == DATATYPE_INT8 && preshape.GetDim(1) >= 8))){
                    return 1e-7 * preshape.CalcElementsIncludingPadding() / 4;
                }else{
                    return 1e-7 * preshape.CalcElementsExcludingPadding();
                }
            }else{
                return 1e-7 * preshape.CalcElementsIncludingPadding();
            }
        }else{
            return 1e-7 * preshape.CalcElementsExcludingPadding();
        }
    }
    return 1.0e-5;
}

void NormalAlgorithm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                     dataformat_t input_format, dataformat_t output_format) {
    input_format_ = input_format;
    output_format_ = output_format;
    if (inherited_nd_set_.find(node->GetType().name) != inherited_nd_set_.end()){
        char *env_ptr = getenv("MACART_OP_FORMAT");
        if(env_ptr != nullptr){
            string op_format_str(env_ptr);
            if(op_format_str == "ND" && input_format == DATAFORMAT_NDARRAY){
                output_format = DATAFORMAT_NDARRAY;
            }else if(op_format_str == "NHWC8" && input_format == DATAFORMAT_NHWC8){
                output_format = DATAFORMAT_NHWC8;
            }
            else if(op_format_str == "NHWC16" && input_format == DATAFORMAT_NHWC16){
                output_format = DATAFORMAT_NHWC16;
            }
            else if(op_format_str == "NHWC" && input_format == DATAFORMAT_NHWC){
                output_format = DATAFORMAT_NHWC;
            }
            else if(op_format_str == "NCHW8" && input_format == DATAFORMAT_NCHW8){
                output_format = DATAFORMAT_NCHW8;
            }
            else if(op_format_str == "NCHW16" && input_format == DATAFORMAT_NCHW16){
                output_format = DATAFORMAT_NCHW16;
            }
        }
        input_format = output_format;
    }
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto shape = tensors->find(edge_id)->second->GetShape();
        if(i >= 1 && keep_const_input_nd_op_set_.find(node->GetType().name) != keep_const_input_nd_op_set_.end()){
            shape->SetDataFormat(DATAFORMAT_NDARRAY);
        }else{
            if (shape->GetDimCount() > 1) {
                shape->SetDataFormat(input_format);
            } else {
                shape->SetDataFormat(DATAFORMAT_NDARRAY);
            }
        }
    }

    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto shape = tensors->find(edge_id)->second->GetShape();
        if (shape->GetDimCount() > 1) {
            shape->SetDataFormat(output_format);
        } else {
            shape->SetDataFormat(DATAFORMAT_NDARRAY);
        }
    }
    return;
}

}}} // namespace ppl::nn::cuda
