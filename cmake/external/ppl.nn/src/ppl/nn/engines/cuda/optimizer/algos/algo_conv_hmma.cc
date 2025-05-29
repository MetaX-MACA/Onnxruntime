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

using namespace ppl::common;

std::string str_replace(std::string org_str, std::string replaced_str, std::string to_str){
    size_t start_pos = 0;
    while((start_pos = org_str.find(replaced_str, start_pos)) != std::string::npos) {
        org_str.replace(start_pos, replaced_str.length(), to_str);
        start_pos += to_str.length();
    }
    return org_str;
}

namespace ppl { namespace nn { namespace cuda {

void TuringHMMAImpgemm::DeleteAttrParam(void*& param) {
    delete (CudaConvParam*)param;
    return;
}

void TuringHMMAImpgemm::GetAttrParam(void*& param) const {
    if (param == nullptr)
        param = new CudaConvParam();
    *(CudaConvParam*)param = attr_param_;
    return;
}

bool TuringHMMAImpgemm::IsSupported(const ir::Node* node, const OptKernelOptions& options,
                                    dataformat_t input_format) const {
    uint32_t group = (reinterpret_cast<CudaConvParam*>(options.param))->param.group;
    // check if conv is depthwise
    const TensorShape& tensor0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    const TensorShape& tensor1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    if (tensor0.GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        return false;
    }
#ifndef PPLNN_USE_DNN
    if (group == tensor1.GetDim(0) && tensor1.GetDim(1) == 1 && group != 1) {
        return false;
    }
#else
    if(group > 1 && !(group == tensor0.GetDim(1) && group == tensor1.GetDim(0))){
        if (input_format != DATAFORMAT_NHWC) {
            return false;
        }
        return true;
    }
#endif
    // check if conv is quantization
    auto quant0 = options.quants->at(node->GetInput(0));
    if (quant0.type == DATATYPE_INT8) {
        return false;
    }
#ifdef PPLNN_USE_DNN
    if(gSupportNc1hwc0){
        if (input_format != DATAFORMAT_NCHW8) {
            return false;
        }
        return true;
    }
#endif
    if (input_format != DATAFORMAT_NHWC8) {
        return false;
    }
    return true;
}

double TuringHMMAImpgemm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    attr_param_.extra_param.algo_info.algo_type = "TuringHMMAImpgemm";
    options.compile_set->emplace(node->GetId());

    const std::string& key_str = node->GetName();
    auto algo_info = options.args->alog_selects.find(key_str);
    if (algo_info != options.args->alog_selects.end()) {
        attr_param_.extra_param.algo_info.kid = algo_info->second.kid;
        attr_param_.extra_param.algo_info.splitk = algo_info->second.splitk;
        attr_param_.extra_param.algo_info.splitf = algo_info->second.splitf;
        attr_param_.extra_param.algo_info.algo_name = algo_info->second.kname;
        if (algo_info->second.splitk > 1)
            attr_param_.extra_param.algo_info.algo_name += "_spk" + std::to_string(algo_info->second.splitk);
        attr_param_.extra_param.algo_info.ParseAlgoName();
        return 0.0f;
    } else { // Give the default kernel
#ifdef PPLNN_CUDA_ENABLE_KERNEL_CUT
        attr_param_.extra_param.algo_info.algo_name = "nvSwzlSm75Fp16Conv_hmma1688_nhwc_fn_b32x256_w32x64_k8_buf2";
        attr_param_.extra_param.algo_info.kid = 685;
#else
        attr_param_.extra_param.algo_info.algo_name = "nvSwzlSm75Fp16Conv_hmma1688_nhwc_fn_b128x64_w64x32_k64_buf2";
        attr_param_.extra_param.algo_info.kid = 5197;
#endif
        attr_param_.extra_param.algo_info.splitk = 1;
        attr_param_.extra_param.algo_info.splitf = 1;
        attr_param_.extra_param.algo_info.ParseAlgoName();
    }

    auto shape_in0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    auto shape_in1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    auto shape_in2 = TensorShape();
    const TensorShape& shape_out = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());
    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param;
    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, attr_param_, temp_conv_param);

    // input shape is invalid
    if (shape_in0.GetDimCount() != 4 || shape_in1.GetDimCount() != 4) {
        return 0.0f;
    }
    // input H or W is too small
    if (shape_in0.GetDim(2) + 2 * temp_conv_param.pad_height < shape_in1.GetDim(2) ||
        shape_in0.GetDim(3) + 2 * temp_conv_param.pad_width < shape_in1.GetDim(3)) {
        shape_in0.SetDim(2, shape_in1.GetDim(2));
        shape_in0.SetDim(3, shape_in1.GetDim(3));
    }

#ifdef PPLNN_USE_DNN
    return 0.0f;
#endif

    if (options.args->quick_select) {
        return 0.0f;
    }

    // Padding
    shape_in0.SetDim(1, shape_in1.GetDim(1) * attr_param_.param.group);
    uint32_t k_per_grp = shape_in1.GetDim(0) / attr_param_.param.group;
    uint32_t k_per_grp_pad = (k_per_grp + align_size - 1) / align_size * align_size;
    shape_in1.SetDim(0, k_per_grp_pad * attr_param_.param.group);
    if (temp_conv_param.has_bias) {
        shape_in2 = *options.tensors->find(node->GetInput(2))->second->GetShape();
        shape_in2.SetDim(0, k_per_grp_pad * temp_conv_param.num_grp);
    }

    RetCode status;
    ALLOC_BUFFERF_FOR_ALGO_SELECT(input_buffer, shape_in0.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(weight_buffer, shape_in1.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(bias_buffer, shape_in2.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(output_buffer, shape_out.CalcBytesIncludingPadding(), ALGO_MAX_TIME)

    uint64_t size = PPLCUDAConvolutionGetCompilationBufSize(shape_in0.GetDataType(), temp_conv_param);
    ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, size, ALGO_MAX_TIME)

    auto stream = options.device->GetStream();

#ifdef PPLNN_ENABLE_CUDA_JIT
    // Do select
    LOG(INFO) << "Compiling " << node->GetName();
    auto timer = PPLCUDAConvolutionJitSelectKernel(options.device->GetDeviceProp(), stream, shape_in0.GetDataType(), (int4*)input_buffer.addr,
                                                   (int4*)weight_buffer.addr, (int4*)output_buffer.addr,
                                                   (int4*)bias_buffer.addr, (int4*)temp_buffer.addr,
                                                   attr_param_.extra_param.algo_info, temp_conv_param, temp_fuse_param);
    LOG(INFO) << "select kernel " << attr_param_.extra_param.algo_info.algo_name;
#else
    // Do select
    auto timer = PPLCUDAConvolutionSelectKernel(options.device->GetDeviceProp(), stream, shape_in0.GetDataType(), (int4*)input_buffer.addr,
                                                (int4*)weight_buffer.addr, (int4*)output_buffer.addr,
                                                (int4*)bias_buffer.addr, (int4*)temp_buffer.addr,
                                                attr_param_.extra_param.algo_info, temp_conv_param, temp_fuse_param);
#endif
    CudaArgs::AlgoSelects algo_select;
    algo_select.kname = attr_param_.extra_param.algo_info.algo_name;
    algo_select.kid = attr_param_.extra_param.algo_info.kid;
    algo_select.splitk = attr_param_.extra_param.algo_info.splitk;
    algo_select.splitf = attr_param_.extra_param.algo_info.splitf;
    options.args->alog_selects.emplace(key_str, std::move(algo_select));
    return timer;
}

void TuringHMMAImpgemm::AddPreluSlopeDeQuantInput(ir::Node* node, OptKernelOptions& options, int fuse_index){
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto fuse_info = attr_param_.extra_param.fuse_info;
    //float prelu_slope_quant_scale = options.quants->at(node->GetInput(fuse_info.input_inds[fuse_index])).scale[0];
    auto slope_edge_idx = node->GetInput(fuse_info.input_inds[fuse_index]);
    const TensorShape& shape_slope = *options.tensors->find(slope_edge_idx)->second->GetShape();
    auto slope_edge = topo->GetEdge(slope_edge_idx);
    auto slope_node = topo->GetNode(slope_edge->GetProducer());
    auto slope_iter = data->constants.find(slope_node->GetInput(0));
    if (slope_iter != data->constants.end()){
        int slope_ele_num = shape_slope.CalcElementsExcludingPadding();
        void *slope_data = (void*)slope_iter->second.data.GetData();
        std::vector<float> slope_float_host(slope_ele_num);
        for(int i = 0; i < slope_ele_num; i++){
            if(shape_slope.GetDataType() == DATATYPE_FLOAT32)
                slope_float_host[i] = ((float*)slope_data)[i];
            else if(shape_slope.GetDataType() == DATATYPE_FLOAT16)
                slope_float_host[i] = cpu_half2float(((unsigned short*)slope_data)[i]);
            else{
                LOG(ERROR) << "slope data unsupport " << GetDataTypeStr(shape_slope.GetDataType());
                return;
            }
        }
        TensorShape prelu_slope_float_shape;
        prelu_slope_float_shape.SetDataFormat(DATAFORMAT_NDARRAY);
        prelu_slope_float_shape.SetDataType(DATATYPE_FLOAT32);
        prelu_slope_float_shape.SetDimCount(1);
        prelu_slope_float_shape.SetDim(0, slope_ele_num);
        prelu_slope_float_shape.CalcPadding();
        RuntimeConstantInfo prelu_slope_float_constat_info;
        {
            BufferDesc buffer;
            auto status = options.device->Realloc(prelu_slope_float_shape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return;
            }
            prelu_slope_float_constat_info.Reshape(prelu_slope_float_shape);
            prelu_slope_float_constat_info.SetBuffer(buffer, options.device, true);
        }
        auto status = options.device->GetDataConverter()->ConvertFromHost(&prelu_slope_float_constat_info.GetBufferDesc(),
                                                                          prelu_slope_float_shape,
                                                                          slope_float_host.data(),
                                                                          prelu_slope_float_shape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Prelu float slope ConvertFromHost failed: " << GetRetCodeStr(status);
            return;
        }
        auto ret_pair = topo->AddEdge("prelu_float_" + node->GetName() + to_string(fuse_index));
        auto slope_dequant_edge = ret_pair.first;
        auto slope_dequant_edge_id = slope_dequant_edge->GetId();
        node->AddInput(slope_dequant_edge_id);
        slope_dequant_edge->AddConsumer(node->GetId());
        options.tensors->insert(
        make_pair(slope_dequant_edge_id, unique_ptr<TensorImpl>(new TensorImpl(slope_dequant_edge, TENSORTYPE_NORMAL))));
        *options.tensors->find(slope_dequant_edge_id)->second->GetShape() = prelu_slope_float_shape;
        options.quants->resize(topo->GetCurrentEdgeIdBound());
        options.quants->at(slope_dequant_edge_id).format = prelu_slope_float_shape.GetDataFormat();
        options.quants->at(slope_dequant_edge_id).type = prelu_slope_float_shape.GetDataType();
        options.info->constants.emplace(slope_dequant_edge_id, std::move(prelu_slope_float_constat_info));
    }
}

RetCode TuringHMMAImpgemm::ModifyParam(ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto weight_edge = topo->GetEdge(node->GetInput(1));
    auto weight_node = topo->GetNode(weight_edge->GetProducer());

    const TensorShape& shape_in0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    const TensorShape& shape_in1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    const TensorShape& shape_out = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(shape_in1.GetDataFormat());
    bool is_need_to_c4 = !gSupportNc1hwc0 && (this->attr_param_.extra_param.is_graph_input_conv && shape_in1.GetDim(1) <= 4)
                         && this->attr_param_.param.group == 1 && topo->GetEdge(weight_node->GetInput(0))->CalcConsumerCount() == 1;

    RetCode status;
    conv_param_t temp_conv_param;
    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, attr_param_, temp_conv_param);

#ifndef PPLNN_USE_DNN
    uint32_t k_per_grp = shape_in1.GetDim(0) / temp_conv_param.num_grp;
    uint32_t k_per_grp_pad = (k_per_grp + align_size - 1) / align_size * align_size;
#endif

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
    #ifdef PPLNN_USE_DNN
        newshape.SetDim(0, (newshape.GetDim(0) + 7) / 8 * 8);
    #else
        newshape.SetPadding1(0, k_per_grp_pad * temp_conv_param.num_grp - newshape.GetDim(0));
    #endif

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
        #ifdef PPLNN_USE_DNN
            if(attr_param_.param.group > 1){
                postshape.SetDataFormat(DATAFORMAT_NHWC);
                postshape.CalcPadding();
            }else{
                if(is_need_to_c4){
                    postshape.SetDataFormat(DATAFORMAT_NHWC4);
                    postshape.CalcPadding();
                }
            }
            weight_constat_info.Reshape(postshape);
        #else
            weight_constat_info.Reshape(newshape); // give the converted shape, but the dims are not changed
        #endif
        }

    #ifdef PPLNN_USE_DNN
        if (shape_in0.GetDataType() == DATATYPE_FLOAT16 && (preshape.GetDataType() == DATATYPE_FLOAT16 || preshape.GetDataType() == DATATYPE_FLOAT32)){
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
            status = options.device->GetDataConverter()->ConvertFromHost(&weight_constat_info.GetBufferDesc(), postshape,
                                                                     weight_iter->second.data.GetData(), preshape);
            if(0){
                u_int16_t *temp_input = (u_int16_t*)malloc(weight_constat_info.GetShape()->CalcBytesIncludingPadding());
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
                            fprintf(fp_input_txt, "%f\n", cpu_half2float(temp_input[i * c_pad * hw + k * c_pad + j]));
                        }
                    }
                }
                for(uint i = 0; i < n; i++){
                    for(uint j = weight_constat_info.GetShape()->GetDim(1); j < c_pad; j++){
                        for(uint k = 0; k < hw; k++){
                            fprintf(fp_input_txt, "%f\n", cpu_half2float(temp_input[i * c_pad * hw + k * c_pad + j]));
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
        }else{
            auto quants = options.quants;
            status = ((CudaDataConverter*)options.device->GetDataConverter())
                         ->ConvertFromHost(&weight_constat_info.GetBufferDesc(), postshape, (*quants)[postedge_id],
                                           weight_iter->second.data.GetData(), preshape, (*quants)[preedge_id]);
        }
        if (status != RC_SUCCESS) {
            LOG(ERROR) << node->GetName() << " copy constant failed: " << GetRetCodeStr(status);
            return status;
        }
        options.info->constants.emplace(preedge_id, std::move(weight_constat_info));
        *options.tensors->find(preedge_id)->second->GetShape() = postshape;
        *options.tensors->find(postedge_id)->second->GetShape() = postshape;
    #else
        ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, newshape.CalcBytesIncludingPadding(), RC_OUT_OF_MEMORY)
        status = options.device->GetDataConverter()->ConvertFromHost(&temp_buffer, postshape,
                                                                     weight_iter->second.data.GetData(), preshape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << node->GetName() << " copy constant failed: " << GetRetCodeStr(status);
            return status;
        }

        PPLCUDAConvolutionCvtFlt(stream, weight_constat_info.GetBufferDesc().addr, temp_buffer.addr,
                                 shape_in0.GetDataType(), temp_conv_param);
        options.info->constants.emplace(preedge_id, std::move(weight_constat_info));
        *options.tensors->find(preedge_id)->second->GetShape() = newshape;
        *options.tensors->find(postedge_id)->second->GetShape() = newshape;
    #endif
        options.quants->at(preedge_id) = options.quants->at(postedge_id);
        options.quants->at(preedge_id).format = postshape.GetDataFormat();
        options.quants->at(preedge_id).type = postshape.GetDataType();
    }
    reinterpret_cast<CudaConvParam*>(options.param)->extra_param.is_initializer_weight =
        weight_iter != data->constants.end();
    if (attr_param_.extra_param.bias_term == 0) {
        //return RC_SUCCESS;
    }

#ifdef PPLNN_USE_DNN
//#if 1
    fuse_param_t fuse_param;
    auto fuse_info = attr_param_.extra_param.fuse_info;
    for(int fuse_index = 0; fuse_index < fuse_info.types.size(); fuse_index++){
        int type = GetRelueType(fuse_info.types[fuse_index]);
        switch (type) {
            case 7: // PRelu
                AddPreluSlopeDeQuantInput(node, options, fuse_index);
                break;
            default:
                ;
        }
    }
#endif
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
        #ifdef PPLNN_USE_DNN
            newshape.SetDim(0, (newshape.GetDim(0) + 7) / 8 * 8);
        #else
            newshape.SetDim(0, k_per_grp_pad * temp_conv_param.num_grp);
        #endif
            RuntimeConstantInfo bias_constat_info;
            {
                BufferDesc buffer;
                status = options.device->Realloc(newshape, &buffer);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                    return status;
                }
                cudaMemsetAsync(buffer.addr, 0, newshape.CalcBytesIncludingPadding(), stream);
            #ifdef PPLNN_USE_DNN
                bias_constat_info.Reshape(postshape);
            #else
                bias_constat_info.Reshape(newshape);
            #endif
                bias_constat_info.SetBuffer(buffer, options.device, true);
            }
        #ifdef PPLNN_USE_DNN
            status = options.device->GetDataConverter()->ConvertFromHost(&bias_constat_info.GetBufferDesc(), postshape,
                                                                        bias_iter->second.data.GetData(), preshape);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
                return status;
            }
            options.info->constants.emplace(preedge_id, std::move(bias_constat_info));
            *options.tensors->find(preedge_id)->second->GetShape() = postshape;
            *options.tensors->find(postedge_id)->second->GetShape() = postshape;
        #else
            ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, newshape.CalcBytesIncludingPadding(), RC_OUT_OF_MEMORY)
            status = options.device->GetDataConverter()->ConvertFromHost(&temp_buffer, postshape,
                                                                        bias_iter->second.data.GetData(), preshape);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
                return status;
            }

            PPLCUDAConvolutionCvtBias(stream, bias_constat_info.GetBufferDesc().addr, temp_buffer.addr,
                                    shape_in0.GetDataType(), temp_conv_param);
            options.info->constants.emplace(preedge_id, std::move(bias_constat_info));
            *options.tensors->find(preedge_id)->second->GetShape() = newshape;
            *options.tensors->find(postedge_id)->second->GetShape() = newshape;
        #endif
            options.quants->at(preedge_id) = options.quants->at(postedge_id);
            options.quants->at(preedge_id).format = postshape.GetDataFormat();
            options.quants->at(preedge_id).type = postshape.GetDataType();
        }
    }
#ifdef PPLNN_USE_DNN
    auto shape_input = *options.tensors->find(node->GetInput(0))->second->GetShape();
    if(shape_input.GetDataFormat() == DATAFORMAT_NHWC && shape_input.GetDim(1) % 8 == 0){
        if(gSupportNc1hwc0){
            //shape_input.SetDataFormat(DATAFORMAT_NCHW8);
        } else{
            shape_input.SetDataFormat(DATAFORMAT_NHWC8);
        }
        *options.tensors->find(node->GetInput(0))->second->GetShape() = shape_input;
    }
    if(is_need_to_c4){
        shape_input.SetDataFormat(DATAFORMAT_NHWC4);
        *options.tensors->find(node->GetInput(0))->second->GetShape() = shape_input;
    }
    auto shape_output = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    if(shape_output.GetDataFormat() == DATAFORMAT_NHWC && shape_output.GetDim(1) % 8 == 0){
        if(gSupportNc1hwc0){
            //shape_output.SetDataFormat(DATAFORMAT_NCHW8);
        } else{
            shape_output.SetDataFormat(DATAFORMAT_NHWC8);
        }
        *options.tensors->find(node->GetOutput(0))->second->GetShape() = shape_output;
    }
#endif
    return RC_SUCCESS;
}

void TuringHMMAImpgemm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
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
