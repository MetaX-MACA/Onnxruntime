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

#include "ppl/nn/engines/cuda/optimizer/algos/algo_conv.h"

#include <chrono>

#include "ppl/common/cuda/cuda_types.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/utils.h"

#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)

using namespace ppl::common;

extern std::string str_replace(std::string org_str, std::string replaced_str, std::string to_str);

namespace ppl { namespace nn { namespace cuda {

void TuringFMMAImpgemm::DeleteAttrParam(void*& param) {
    delete (CudaConvParam*)param;
    return;
}

void TuringFMMAImpgemm::GetAttrParam(void*& param) const {
    if (param == nullptr)
        param = new CudaConvParam();
    *(CudaConvParam*)param = attr_param_;
    return;
}

bool TuringFMMAImpgemm::IsSupported(const ir::Node* node, const OptKernelOptions& options,
                                    dataformat_t input_format) const {
    uint32_t group = (reinterpret_cast<CudaConvParam*>(options.param))->param.group;
    // check if conv is depthwise
    const TensorShape& tensor0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    const TensorShape& tensor1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    if (tensor0.GetDataType() != ppl::common::DATATYPE_FLOAT32) {
        return false;
    }
    if (input_format != DATAFORMAT_NDARRAY) {//if (input_format != DATAFORMAT_NHWC) {
        return false;
    }
    return true;
}

double TuringFMMAImpgemm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    attr_param_.extra_param.algo_info.algo_type = "TuringFMMAImpgemm";
    return 0.0f;
}

RetCode TuringFMMAImpgemm::ModifyParam(ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto weight_edge = topo->GetEdge(node->GetInput(1));
    auto weight_node = topo->GetNode(weight_edge->GetProducer());

    const TensorShape& shape_in0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    const TensorShape& shape_in1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    const TensorShape& shape_out = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    RetCode status;

    // Split weight format to group padding
    auto stream = options.device->GetStream();
    auto weight_iter = data->constants.find(weight_node->GetInput(0));
    if (weight_iter != data->constants.end() && // is a constant tensor and has not be loaded
        options.info->constants.find(weight_node->GetInput(0)) == options.info->constants.end()) {
        auto preedge_id = weight_node->GetInput(0);
        auto postedge_id = node->GetInput(1);
        TensorShape& preshape = *options.tensors->find(preedge_id)->second->GetShape();
        TensorShape& postshape = *options.tensors->find(postedge_id)->second->GetShape();
        if(attr_param_.param.kernel_shape.size() == 1){
            std::vector<int64_t> dim;
            for(int i = 0; i < postshape.GetDimCount(); i++)
                dim.push_back(postshape.GetDim(i));
            dim.push_back((int64_t)1);
            postshape.Reshape(dim.data(), dim.size());
        }
        auto newshape = postshape;
        RuntimeConstantInfo weight_constat_info;
        {
            BufferDesc buffer;
            status = options.device->Realloc(newshape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }
            cudaMemsetAsync(buffer.addr, 0, newshape.CalcBytesIncludingPadding(), stream);
            weight_constat_info.SetBuffer(buffer, options.device, true);
            weight_constat_info.Reshape(newshape); // give the converted shape, but the dims are not changed
        }

        if (preshape.GetDataType() == DATATYPE_FLOAT32){
            std::vector<int64_t> postshape_dim_new;
            for(int i = 0; i < postshape.GetDimCount(); i++)
                postshape_dim_new.push_back(postshape.GetDim(i));
            std::vector<int64_t> preshape_dim_new;
            for(int i = 0; i < preshape.GetDimCount(); i++)
                preshape_dim_new.push_back(preshape.GetDim(i));
            if(postshape.GetDimCount() == 5){
                postshape_dim_new[3] *= postshape_dim_new[4];
                postshape.Reshape(postshape_dim_new.data(), 4);
                preshape_dim_new[3] *= preshape_dim_new[4];
                preshape.Reshape(preshape_dim_new.data(), 4);
            }
            if(0){
                float *temp_input = (float*)weight_iter->second.data.GetData();
                char file_input_txt_name[128] = {0};
                sprintf(file_input_txt_name, "%s_weight_org.txt", str_replace(node->GetName(), "/", "_").c_str());
                FILE *fp_input_txt = fopen(file_input_txt_name, "wb");
                int dim_sum = weight_constat_info.GetShape()->GetDimCount();
                uint n = weight_constat_info.GetShape()->GetDim(0);
                uint c = weight_constat_info.GetShape()->GetDim(1);
                uint hw = weight_constat_info.GetShape()->CalcElementsFromDimensionExcludingPadding(2);
                printf("~~~~~~~~~~~~~ n:%d, c:%d, hw:%d\n", n, c, hw);
                for(uint i = 0; i < n; i++){
                    for(uint j = 0; j < c; j++){
                        for(uint k = 0; k < hw; k++){
                            fprintf(fp_input_txt, "%f\n", temp_input[i * c * hw + j * hw + k]);
                        }
                    }
                }
                fclose(fp_input_txt);
            }
            if (shape_in0.GetDataType() == DATATYPE_FLOAT16 && (preshape.GetDataType() == DATATYPE_FLOAT16 || preshape.GetDataType() == DATATYPE_FLOAT32)){
                status = options.device->GetDataConverter()->ConvertFromHost(&weight_constat_info.GetBufferDesc(), postshape,
                                                                             weight_iter->second.data.GetData(), preshape);
            }else{
                auto quants = options.quants;
                status = ((CudaDataConverter*)options.device->GetDataConverter())->ConvertFromHost(&weight_constat_info.GetBufferDesc(), postshape, (*quants)[postedge_id],
                                                                                                   weight_iter->second.data.GetData(), preshape, (*quants)[preedge_id]);
            }
            if (status != RC_SUCCESS) {
                LOG(ERROR) << node->GetName() << " copy constant failed: " << GetRetCodeStr(status);
                return status;
            }
            if(0){
                float *temp_input = (float*)malloc(weight_constat_info.GetShape()->CalcBytesIncludingPadding());
                cudaMemcpy(temp_input, weight_constat_info.GetBufferPtr(), weight_constat_info.GetShape()->CalcBytesIncludingPadding(), cudaMemcpyDeviceToHost);
                printf("Weight CalcBytesIncludingPadding: %d\n", weight_constat_info.GetShape()->CalcBytesIncludingPadding());
                char file_input_txt_name[128] = {0};
                sprintf(file_input_txt_name, "%s_weight_process.txt", str_replace(node->GetName(), "/", "_").c_str());
                FILE *fp_input_txt = fopen(file_input_txt_name, "wb");
                int dim_sum = weight_constat_info.GetShape()->GetDimCount();
                uint n = weight_constat_info.GetShape()->GetDim(0);
                uint c_pad = weight_constat_info.GetShape()->GetDim(1) +  weight_constat_info.GetShape()->GetPadding0(1) + weight_constat_info.GetShape()->GetPadding1(1);
                uint hw = weight_constat_info.GetShape()->CalcElementsFromDimensionExcludingPadding(2);
                printf("################ n:%d, c_pad:%d, hw:%d\n", n, c_pad, hw);
                for(uint i = 0; i < n; i++){
                    for(uint j = 0; j < weight_constat_info.GetShape()->GetDim(1); j++){
                        for(uint k = 0; k < hw; k++){
                            fprintf(fp_input_txt, "%f\n", temp_input[i * c_pad * hw + k * c_pad + j]);
                        }
                    }
                }
                for(uint i = 0; i < n; i++){
                    for(uint j = weight_constat_info.GetShape()->GetDim(1); j < c_pad; j++){
                        for(uint k = 0; k < hw; k++){
                            fprintf(fp_input_txt, "%f\n", temp_input[i * c_pad * hw + k * c_pad + j]);
                        }
                    }
                }
                fclose(fp_input_txt);
            }
            if(postshape.GetDimCount() == 5){
                postshape_dim_new[3] /= postshape_dim_new[4];
                postshape.Reshape(postshape_dim_new.data(), 5);
                preshape_dim_new[3] /= preshape_dim_new[4];
                preshape.Reshape(preshape_dim_new.data(), 5);
            }
            options.info->constants.emplace(preedge_id, std::move(weight_constat_info));
            *options.tensors->find(preedge_id)->second->GetShape() = postshape;
            *options.tensors->find(postedge_id)->second->GetShape() = postshape;
            reinterpret_cast<CudaConvParam*>(options.param)->extra_param.is_initializer_weight = true;
        }
    }
    if (attr_param_.extra_param.bias_term != 0){
        // Split bias format to group padding
        auto bias_edge = topo->GetEdge(node->GetInput(2));
        auto bias_node = topo->GetNode(bias_edge->GetProducer());
        auto bias_iter = data->constants.find(bias_node->GetInput(0));
        if (bias_iter != data->constants.end() && // is a constant tensor and has not be loaded
            options.info->constants.find(bias_node->GetInput(0)) == options.info->constants.end()) {
            auto preedge_id = bias_node->GetInput(0);
            auto postedge_id = node->GetInput(2);
            const TensorShape& preshape = *options.tensors->find(preedge_id)->second->GetShape();
            const TensorShape& postshape = *options.tensors->find(postedge_id)->second->GetShape();
            auto newshape = postshape;
            RuntimeConstantInfo bias_constat_info;
            {
                BufferDesc buffer;
                status = options.device->Realloc(newshape, &buffer);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                    return status;
                }
                cudaMemsetAsync(buffer.addr, 0, newshape.CalcBytesIncludingPadding(), stream);
                bias_constat_info.Reshape(postshape);
                bias_constat_info.SetBuffer(buffer, options.device, true);
            }
            status = options.device->GetDataConverter()->ConvertFromHost(&bias_constat_info.GetBufferDesc(), postshape,
                                                                        bias_iter->second.data.GetData(), preshape);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
                return status;
            }
            options.info->constants.emplace(preedge_id, std::move(bias_constat_info));
            *options.tensors->find(preedge_id)->second->GetShape() = postshape;
            *options.tensors->find(postedge_id)->second->GetShape() = postshape;
        }
    }
    return RC_SUCCESS;
}

void TuringFMMAImpgemm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                       dataformat_t input_format, dataformat_t output_format) {
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) { // only reset formats of input0 and weight
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto shape = tensors->find(edge_id)->second->GetShape();
        if (i == 0 || i == 1 || shape->GetDimCount() > 1)
            shape->SetDataFormat(input_format);
        else
            shape->SetDataFormat(DATAFORMAT_NDARRAY);
    }

    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto shape = tensors->find(edge_id)->second->GetShape();
        shape->SetDataFormat(output_format);
    }
    return;
}

}}} // namespace ppl::nn::cuda

#endif
