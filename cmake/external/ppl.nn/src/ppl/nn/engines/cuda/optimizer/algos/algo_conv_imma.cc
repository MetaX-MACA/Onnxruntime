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

#include <string.h>
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

void TuringIMMAImpgemm::DeleteAttrParam(void*& param) {
    delete (CudaConvParam*)param;
    return;
}

void TuringIMMAImpgemm::GetAttrParam(void*& param) const {
    if (param == nullptr)
        param = new CudaConvParam();
    *(CudaConvParam*)param = attr_param_;
    return;
}

bool TuringIMMAImpgemm::IsSupported(const ir::Node* node, const OptKernelOptions& options,
                                    dataformat_t input_format) const {
    uint32_t group = (reinterpret_cast<CudaConvParam*>(options.param))->param.group;
    // check if conv is depthwise
    const TensorShape& tensor0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    const TensorShape& tensor1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
#if !(defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA))
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
    // check if conv quant to INT8
    auto quant0 = options.quants->at(node->GetInput(0));
    if (quant0.type != DATATYPE_INT8) {
        return false;
    }
#ifdef PPLNN_USE_DNN
    if(gSupportNc1hwc0){
        if (input_format != DATAFORMAT_NCHW16) {
            return false;
        }
        return true;
    }
#endif
    if (input_format != DATAFORMAT_NHWC16) {
        return false;
    }
    return true;
}

double TuringIMMAImpgemm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    attr_param_.extra_param.algo_info.algo_type = "TuringIMMAImpgemm";
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
        attr_param_.extra_param.algo_info.algo_name = "nvSwzlSm75Int8Conv_imma8816_nhwc_fn_b64x128_w16x128_k16_buf2";
        attr_param_.extra_param.algo_info.kid = 718;
#else
        attr_param_.extra_param.algo_info.algo_name = "nvSwzlSm75Int8Conv_imma8816_nhwc_fn_b256x64_w64x64_k64_buf2";
        attr_param_.extra_param.algo_info.kid = 3775;
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

#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
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

    auto group = ((CudaConvParam*)options.param)->param.group;
    auto channel_per_grp = shape_in1.GetDim(0) / group;
    auto channel_per_grp_pad = (channel_per_grp + align_size - 1) / align_size * align_size;
    auto total_size = channel_per_grp_pad * group;
    ALLOC_BUFFERF_FOR_ALGO_SELECT(wegiht_quant, total_size * sizeof(float), ALGO_MAX_TIME)
    quant_param_t temp_quant_param;
    temp_quant_param.in_scale = options.quants->at(node->GetId()).scale[0];
    temp_quant_param.out_scale = 1.0f / options.quants->at(node->GetId()).scale[0];
    temp_quant_param.d_flt_scale = wegiht_quant.addr;
    temp_quant_param.pre_scale = 0.0f;

    auto stream = options.device->GetStream();

#ifdef PPLNN_ENABLE_CUDA_JIT
    // Do select
    LOG(INFO) << "Compiling " << node->GetName();
    auto timer = PPLCUDAConvolutionJitSelectKernelInt8(
        options.device->GetDeviceProp(), stream, shape_in0.GetDataType(), (int4*)input_buffer.addr, (int4*)weight_buffer.addr,
        (int4*)output_buffer.addr, (int4*)bias_buffer.addr, (int4*)temp_buffer.addr, attr_param_.extra_param.algo_info,
        temp_conv_param, temp_quant_param, temp_fuse_param);
    LOG(INFO) << "select kernel " << attr_param_.extra_param.algo_info.algo_name;
#else
    // Do select
    auto timer = PPLCUDAConvolutionSelectKernelInt8(options.device->GetDeviceProp(), stream, shape_in0.GetDataType(), (int4*)input_buffer.addr,
                                                   (int4*)weight_buffer.addr, (int4*)output_buffer.addr,
                                                   (int4*)bias_buffer.addr, (int4*)temp_buffer.addr,
                                                   attr_param_.extra_param.algo_info, temp_conv_param, temp_quant_param, temp_fuse_param);
#endif
    CudaArgs::AlgoSelects algo_select;
    algo_select.kname = attr_param_.extra_param.algo_info.algo_name;
    algo_select.kid = attr_param_.extra_param.algo_info.kid;
    algo_select.splitk = attr_param_.extra_param.algo_info.splitk;
    algo_select.splitf = attr_param_.extra_param.algo_info.splitf;
    options.args->alog_selects.emplace(key_str, std::move(algo_select));
    return timer;
}

void TuringIMMAImpgemm::AddPreluSlopeDeQuantInput(ir::Node* node, OptKernelOptions& options, int fuse_index){
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto fuse_info = attr_param_.extra_param.fuse_info;
    float prelu_slope_quant_scale = options.quants->at(node->GetInput(fuse_info.input_inds[fuse_index])).scale[0];
    auto slope_edge = topo->GetEdge(node->GetInput(fuse_info.input_inds[fuse_index]));
    auto slope_node = topo->GetNode(slope_edge->GetProducer());
    auto slope_iter = data->constants.find(slope_node->GetInput(0));
    if (slope_iter != data->constants.end()){
        int slope_size = slope_iter->second.data.GetSize();
        char *slope_quant_data = (char *)slope_iter->second.data.GetData();
        std::vector<float> slope_dequant_host(slope_size / sizeof(char));
        for(int i = 0; i < slope_size  / sizeof(char); i++){
            slope_dequant_host[i] = slope_quant_data[i] * prelu_slope_quant_scale;
        }
        TensorShape prelu_slope_float_shape;
        prelu_slope_float_shape.SetDataFormat(DATAFORMAT_NDARRAY);
        prelu_slope_float_shape.SetDataType(DATATYPE_FLOAT32);
        prelu_slope_float_shape.SetDimCount(1);
        prelu_slope_float_shape.SetDim(0, slope_size / sizeof(char));
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
                                                                          slope_dequant_host.data(),
                                                                          prelu_slope_float_shape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Prelu float slope ConvertFromHost failed: " << GetRetCodeStr(status);
            return;
        }
        auto ret_pair = topo->AddEdge("prelu_dequant_" + node->GetName() + to_string(fuse_index));
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

RetCode TuringIMMAImpgemm::ModifyParam(ir::Node* node, OptKernelOptions& options) {
#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
//#if 1
    auto stream = options.device->GetStream();
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto fuse_info = attr_param_.extra_param.fuse_info;
    auto weight_edge = topo->GetEdge(node->GetInput(1));
    auto weight_node = topo->GetNode(weight_edge->GetProducer());
    auto quants = options.quants;
    const TensorShape& shape_in0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    const TensorShape& shape_in1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    const TensorShape& shape_out = *options.tensors->find(node->GetOutput(0))->second->GetShape();
#if 1
    bool is_need_to_c4 = !gSupportNc1hwc0 && (this->attr_param_.extra_param.is_graph_input_conv && shape_in1.GetDim(1) <= 4)
                         && this->attr_param_.param.group == 1 && topo->GetEdge(weight_node->GetInput(0))->CalcConsumerCount() == 1;
    is_need_to_c4 = is_need_to_c4
                    && this->attr_param_.param.kernel_shape.size() == 2
                    && (    (this->attr_param_.param.kernel_shape[0] == 7 && this->attr_param_.param.kernel_shape[1] == 7)
                         || (this->attr_param_.param.kernel_shape[0] == 6 && this->attr_param_.param.kernel_shape[1] == 6)
                         || (this->attr_param_.param.kernel_shape[0] == 3 && this->attr_param_.param.kernel_shape[1] == 3));
#else
    bool is_need_to_c4 = false;
#endif
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
        newshape.SetDim(0, (newshape.GetDim(0) + 15) / 16 * 16);
        RuntimeConstantInfo weight_constat_info;
        {
            BufferDesc buffer;
            auto status = options.device->Realloc(newshape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }
            cudaMemsetAsync(buffer.addr, 0, newshape.CalcBytesIncludingPadding(), stream);

            weight_constat_info.SetBuffer(buffer, options.device, true);
            weight_constat_info.Reshape(postshape);
        }
        if(attr_param_.param.group > 1){
            postshape.SetDataFormat(DATAFORMAT_NHWC);
            postshape.CalcPadding();
            weight_constat_info.Reshape(postshape);
        }else{
            if(is_need_to_c4){
                postshape.SetDataFormat(DATAFORMAT_NHWC4);
                postshape.CalcPadding();
                weight_constat_info.Reshape(postshape);
            }
        }
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
        auto status = ((CudaDataConverter*)options.device->GetDataConverter())
                       ->ConvertFromHost(&weight_constat_info.GetBufferDesc(), postshape, (*quants)[postedge_id],
                                       weight_iter->second.data.GetData(), preshape, (*quants)[preedge_id]);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << node->GetName() << " copy constant failed: " << GetRetCodeStr(status);
            return status;
        }
        if(postshape.GetDimCount() == 5){
            postshape_dim_new[3] /= postshape_dim_new[4];
            postshape.Reshape(postshape_dim_new.data(), 5);
            preshape_dim_new[3] /= preshape_dim_new[4];
            preshape.Reshape(preshape_dim_new.data(), 5);
        }

        *options.tensors->find(preedge_id)->second->GetShape() = postshape;
        *options.tensors->find(postedge_id)->second->GetShape() = postshape;
        options.info->constants.emplace(preedge_id, std::move(weight_constat_info));
        options.quants->at(preedge_id) = (*quants)[postedge_id];
        options.quants->at(preedge_id).type = postshape.GetDataType();
        options.quants->at(preedge_id).format = postshape.GetDataFormat();
        reinterpret_cast<CudaConvParam*>(options.param)->extra_param.is_initializer_weight = weight_iter != data->constants.end();
    }

    fuse_param_t fuse_param;

    const std::set<std::string> relu_set{"Relu", "Sigmoid", "HardSigmoid", "Swish", "HardSwish", "Clip", "PRelu", "LeakyRelu", "Mish"};
    const std::set<std::string> elemenwise_set{"Add", "Mul"};

    int fuse_index = 0;
    int fuse_size = fuse_info.types.size();

    RetCode status = RC_SUCCESS;

    if (fuse_index < fuse_size && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
        int type = GetRelueType(fuse_info.types[fuse_index]);
        switch (type) {
            case 0: // Relu
            case 1: // Sigmoid
            case 2: // HardSigmoid
            case 3: // Swish
            case 4: // HardSwish
            case 5: // Clip
            case 6: // LeakyRelu
            case 8: // Mish
                fuse_param.has_activation = type + 1;
                break;
            case 7: // PRelu
                fuse_param.has_activation = type + 1;
                AddPreluSlopeDeQuantInput(node, options, fuse_index);
                break;
            default:
                return RC_UNSUPPORTED;
        }
        fuse_index += 1;
    }

    for(; fuse_index < fuse_size; fuse_index++){
        if (elemenwise_set.find(fuse_info.types[fuse_index]) != elemenwise_set.end()) {
            fuse_param.has_elt = true;
            break;
        }
    }
    if(fuse_param.has_elt){
        //auto& ele_input_quant = options.quants->at(node->GetInput(attr_param_.extra_param.bias_term ? 3 : 2));
        auto& ele_input_quant = options.quants->at(node->GetInput(fuse_info.input_inds[fuse_index]));
        auto ele_input_quant_size = ele_input_quant.scale.size();
        TensorShape ele_input_quant_shape;
        ele_input_quant_shape.SetDataFormat(DATAFORMAT_NDARRAY);
        ele_input_quant_shape.SetDataType(DATATYPE_FLOAT32);
        ele_input_quant_shape.SetDimCount(1);
        ele_input_quant_shape.SetDim(0, ele_input_quant_size);
        ele_input_quant_shape.CalcPadding();
        RuntimeConstantInfo ele_input_quant_constat_info;
        {
            BufferDesc buffer;
            auto status = options.device->Realloc(ele_input_quant_shape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            ele_input_quant_constat_info.Reshape(ele_input_quant_shape);
            ele_input_quant_constat_info.SetBuffer(buffer, options.device, true);
        }

        std::vector<float> ele_input_quant_host(ele_input_quant_size);
        for(uint i=0; i<ele_input_quant_size; i++) {
            ele_input_quant_host[i] = ele_input_quant.scale[i];
        }
        status = options.device->GetDataConverter()->ConvertFromHost(&ele_input_quant_constat_info.GetBufferDesc(), 
                                                                     ele_input_quant_shape,
                                                                     ele_input_quant_host.data(), 
                                                                     ele_input_quant_shape);

        auto ret_pair = topo->AddEdge("ele_quant_" + node->GetName());
        auto ele_input_quant_edge = ret_pair.first;
        auto ele_input_quant_edge_id = ele_input_quant_edge->GetId();
        node->AddInput(ele_input_quant_edge_id);
        ele_input_quant_edge->AddConsumer(node->GetId());

        options.tensors->insert(
            make_pair(ele_input_quant_edge_id, unique_ptr<TensorImpl>(new TensorImpl(ele_input_quant_edge, TENSORTYPE_NORMAL))));
        *options.tensors->find(ele_input_quant_edge_id)->second->GetShape() = ele_input_quant_shape;
        options.quants->resize(topo->GetCurrentEdgeIdBound());
        options.quants->at(ele_input_quant_edge_id).format = ele_input_quant_shape.GetDataFormat();
        options.quants->at(ele_input_quant_edge_id).type = ele_input_quant_shape.GetDataType();
        options.info->constants.emplace(ele_input_quant_edge_id, std::move(ele_input_quant_constat_info));
        fuse_index += 1;
    }

    if (fuse_index < fuse_size && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
        int type = GetRelueType(fuse_info.types[fuse_index]);
        switch (type) {
            case 7: // PRelu
                AddPreluSlopeDeQuantInput(node, options, fuse_index);
                break;
            default:
                ;
        }
    }

    const TensorShape& weight_shape = *options.tensors->find(node->GetInput(1))->second->GetShape();
    if (attr_param_.extra_param.bias_term != 0) {
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
            newshape.SetDim(0, (newshape.GetDim(0) + 15) / 16 * 16);

            RuntimeConstantInfo bias_constat_info;
            {
                BufferDesc buffer;
                auto status = options.device->Realloc(newshape, &buffer);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                    return status;
                }
                cudaMemsetAsync(buffer.addr, 0, newshape.CalcBytesIncludingPadding(), stream);
                bias_constat_info.Reshape(postshape);
                bias_constat_info.SetBuffer(buffer, options.device, true);
            }

            auto& out_quant = options.quants->at(node->GetOutput(0));
            int bias_len = postshape.CalcElementsIncludingPadding();
            float* bias_data = (float*)bias_iter->second.data.GetData();
            std::vector<float> bias_tmp_data(bias_len);
            for(int i=0; i<postshape.CalcElementsIncludingPadding(); i++) {
            #ifdef QUANT_CONV_FORWARD_ALL_USE_FUSE_API
                bias_tmp_data[i] = bias_data[i];
            #else
                bias_tmp_data[i] = (fuse_param.has_activation == 0 && fuse_param.has_elt == false) ? (bias_data[i] / out_quant.scale[0]) : bias_data[i];
            #endif
            }

            auto status = options.device->GetDataConverter()->ConvertFromHost(&bias_constat_info.GetBufferDesc(), postshape,
                                                                        bias_tmp_data.data(), preshape);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
                return status;
            }

            auto ret_pair = topo->AddEdge("bias_" + node->GetName());
            auto bias_edge = ret_pair.first;
            auto bias_edge_id = bias_edge->GetId();
            node->AddInput(bias_edge_id);
            bias_edge->AddConsumer(node->GetId());

            options.tensors->insert(
                make_pair(bias_edge_id, unique_ptr<TensorImpl>(new TensorImpl(bias_edge, TENSORTYPE_NORMAL))));
            *options.tensors->find(bias_edge_id)->second->GetShape() = postshape;
            options.quants->resize(topo->GetCurrentEdgeIdBound());
            options.quants->at(bias_edge_id).format = preshape.GetDataFormat();
            options.quants->at(bias_edge_id).type = preshape.GetDataType();
            options.info->constants.emplace(bias_edge_id, std::move(bias_constat_info));
        }
    } else {
        // no bias
        auto bias_shape = TensorShape();
        bias_shape.SetDimCount(1);
        bias_shape.SetDim(0, (weight_shape.GetDim(0) + 15) / 16 * 16);
        bias_shape.SetDataFormat(DATAFORMAT_NDARRAY);
        bias_shape.SetDataType(DATATYPE_FLOAT32);
        bias_shape.CalcPadding();

        RuntimeConstantInfo bias_constat_info;
        {
            BufferDesc buffer;
            auto status = options.device->Realloc(bias_shape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            cudaMemsetAsync(buffer.addr, 0, bias_shape.CalcBytesIncludingPadding(), stream);
            bias_shape.SetDim(0, weight_shape.GetDim(0));
            bias_constat_info.Reshape(bias_shape);
            bias_constat_info.SetBuffer(buffer, options.device, true);
        }

        auto ret_pair = topo->AddEdge("bias_" + node->GetName());
        auto bias_edge = ret_pair.first;
        auto bias_edge_id = bias_edge->GetId();
        node->AddInput(bias_edge_id);
        bias_edge->AddConsumer(node->GetId());

        options.tensors->insert(
            make_pair(bias_edge_id, unique_ptr<TensorImpl>(new TensorImpl(bias_edge, TENSORTYPE_NORMAL))));
        *options.tensors->find(bias_edge_id)->second->GetShape() = bias_shape;
        options.quants->resize(topo->GetCurrentEdgeIdBound());
        options.quants->at(bias_edge_id).format = bias_shape.GetDataFormat();
        options.quants->at(bias_edge_id).type = bias_shape.GetDataType();
        options.info->constants.emplace(bias_edge_id, std::move(bias_constat_info));
    }

    {
        auto& in_quant = options.quants->at(node->GetInput(0));
        auto& weight_quant = options.quants->at(node->GetInput(1));
        auto& out_quant = options.quants->at(node->GetOutput(0));

        auto weight_size = weight_quant.scale.size();
        TensorShape alpha_shape;
        alpha_shape.SetDataFormat(DATAFORMAT_NDARRAY);
        alpha_shape.SetDataType(DATATYPE_FLOAT32);
        alpha_shape.SetDimCount(1);
        if(weight_size > 1)
            alpha_shape.SetDim(0, (weight_size + 15) / 16 * 16);
        else
            alpha_shape.SetDim(0, weight_size);
        alpha_shape.CalcPadding();
        RuntimeConstantInfo alpha_constat_info;
        {
            BufferDesc buffer;
            auto status = options.device->Realloc(alpha_shape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            alpha_constat_info.Reshape(alpha_shape);
            alpha_constat_info.SetBuffer(buffer, options.device, true);
        }

        int weight_size_temp = weight_size;
        if(weight_size > 1)
            weight_size_temp = (weight_size + 15) / 16 * 16;
        std::vector<float> alpha(weight_size_temp);
        memset(alpha.data(), 0, weight_size_temp * sizeof(float));
        for(uint i=0; i<weight_size; i++) {
        #ifdef QUANT_CONV_FORWARD_ALL_USE_FUSE_API
            alpha[i] = (in_quant.scale[0] * weight_quant.scale[i]);
        #else
            alpha[i] = (fuse_param.has_activation == 0 && fuse_param.has_elt == false) ? (in_quant.scale[0] * weight_quant.scale[i] / out_quant.scale[0]) :
                                                                                         (in_quant.scale[0] * weight_quant.scale[i]);
        #endif
        }

        options.device->CopyFromHost(&alpha_constat_info.GetBufferDesc(), alpha.data(), alpha_shape);

        auto ret_pair = topo->AddEdge("alpha_" + node->GetName());
        auto alpha_edge = ret_pair.first;
        auto alpha_edge_id = alpha_edge->GetId();
        node->AddInput(alpha_edge_id);
        alpha_edge->AddConsumer(node->GetId());

        options.tensors->insert(
            make_pair(alpha_edge_id, unique_ptr<TensorImpl>(new TensorImpl(alpha_edge, TENSORTYPE_NORMAL))));
        *options.tensors->find(alpha_edge_id)->second->GetShape() = alpha_shape;
        options.quants->resize(topo->GetCurrentEdgeIdBound());
        options.quants->at(alpha_edge_id).format = alpha_shape.GetDataFormat();
        options.quants->at(alpha_edge_id).type = alpha_shape.GetDataType();
        options.info->constants.emplace(alpha_edge_id, std::move(alpha_constat_info));
    }

    {
        auto& out_quant = options.quants->at(node->GetOutput(0));
        auto beta_size = out_quant.scale.size();
        TensorShape beta_shape;
        beta_shape.SetDataFormat(DATAFORMAT_NDARRAY);
        beta_shape.SetDataType(DATATYPE_FLOAT32);
        beta_shape.SetDimCount(1);
        beta_shape.SetDim(0, beta_size);
        beta_shape.CalcPadding();
        RuntimeConstantInfo beta_constat_info;
        {
            BufferDesc buffer;
            auto status = options.device->Realloc(beta_shape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            beta_constat_info.Reshape(beta_shape);
            beta_constat_info.SetBuffer(buffer, options.device, true);
        }

        std::vector<float> beta(beta_size);
        for(uint i=0; i<beta_size; i++) {
        #ifdef QUANT_CONV_FORWARD_ALL_USE_FUSE_API
            beta[i] = 1 / out_quant.scale[0];
        #else
            beta[i] = (fuse_param.has_activation == 0 && fuse_param.has_elt == false) ? out_quant.scale[0] : 1 / out_quant.scale[0];
        #endif
        }
        status = options.device->GetDataConverter()->ConvertFromHost(&beta_constat_info.GetBufferDesc(), beta_shape,
                                                                        beta.data(), beta_shape);

        auto ret_pair = topo->AddEdge("beta_" + node->GetName());
        auto beta_edge = ret_pair.first;
        auto beta_edge_id = beta_edge->GetId();
        node->AddInput(beta_edge_id);
        beta_edge->AddConsumer(node->GetId());

        options.tensors->insert(
            make_pair(beta_edge_id, unique_ptr<TensorImpl>(new TensorImpl(beta_edge, TENSORTYPE_NORMAL))));
        *options.tensors->find(beta_edge_id)->second->GetShape() = beta_shape;
        options.quants->resize(topo->GetCurrentEdgeIdBound());
        options.quants->at(beta_edge_id).format = beta_shape.GetDataFormat();
        options.quants->at(beta_edge_id).type = beta_shape.GetDataType();
        options.info->constants.emplace(beta_edge_id, std::move(beta_constat_info));
    }

    auto shape_input = *options.tensors->find(node->GetInput(0))->second->GetShape();
    if(shape_input.GetDataFormat() == DATAFORMAT_NHWC && shape_input.GetDim(1) % 16 == 0){
        if(gSupportNc1hwc0){
            //shape_input.SetDataFormat(DATAFORMAT_NCHW16);
        } else{
            shape_input.SetDataFormat(DATAFORMAT_NHWC16);
        }
        *options.tensors->find(node->GetInput(0))->second->GetShape() = shape_input;
    }
    if(is_need_to_c4){
        shape_input.SetDataFormat(DATAFORMAT_NHWC4);
        *options.tensors->find(node->GetInput(0))->second->GetShape() = shape_input;
    }
    auto shape_output = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    if(shape_output.GetDataFormat() == DATAFORMAT_NHWC && shape_output.GetDim(1) % 16 == 0){
        if(gSupportNc1hwc0){
            //shape_output.SetDataFormat(DATAFORMAT_NCHW16);
        } else{
            shape_output.SetDataFormat(DATAFORMAT_NHWC16);
        }
        *options.tensors->find(node->GetOutput(0))->second->GetShape() = shape_output;
    }

    return status;
#else // PPLNN_USE_DNN && PPLNN_USE_MACA
    this->attr_param_ = *(reinterpret_cast<CudaConvParam*>(options.param));
    auto topo = options.graph->topo.get();
    auto data = options.graph->data.get();
    auto weight_edge = topo->GetEdge(node->GetInput(1));
    auto weight_node = topo->GetNode(weight_edge->GetProducer());
    auto quants = options.quants;

    const TensorShape& shape_in0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    const TensorShape& shape_in1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    const TensorShape& shape_out = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());

    RetCode status;
    conv_param_t temp_conv_param;
    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, attr_param_, temp_conv_param);

    // Add quant to conv inputs
    auto group = ((CudaConvParam*)options.param)->param.group;
    auto channel_per_grp = shape_in1.GetDim(0) / group;
    auto channel_per_grp_pad = (channel_per_grp + align_size - 1) / align_size * align_size;
    auto total_size = channel_per_grp_pad * group;
    auto& weight_quant = options.quants->at(node->GetInput(1));

    if (!weight_quant.per_channel) {
        weight_quant.scale.insert(weight_quant.scale.begin(), total_size, weight_quant.scale[0]);
    }

    std::vector<float> scales(total_size);
    for (int i = 0; i < channel_per_grp_pad * group; i++) {
        if (i % channel_per_grp_pad >= channel_per_grp) {
            scales[i] = 0.0f;
        } else {
            scales[i] = weight_quant.scale[i / channel_per_grp_pad * channel_per_grp + i % channel_per_grp_pad];
        }
    }

    auto quant_shape = TensorShape();
    quant_shape.SetDimCount(1);
    quant_shape.SetDim(0, total_size);
    quant_shape.SetDataFormat(DATAFORMAT_NDARRAY);
    quant_shape.SetDataType(DATATYPE_FLOAT32);

    RuntimeConstantInfo quant_constat_info;
    {
        BufferDesc buffer;
        status = options.device->Realloc(quant_shape, &buffer);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }

        quant_constat_info.Reshape(quant_shape);
        quant_constat_info.SetBuffer(buffer, options.device, true);
    }

    auto ret_pair = topo->AddEdge("Quant_" + node->GetName());
    auto quant_edge = ret_pair.first;
    auto quant_edge_id = quant_edge->GetId();
    node->AddInput(quant_edge_id);
    quant_edge->AddConsumer(node->GetId());

    options.tensors->insert(
        make_pair(quant_edge_id, unique_ptr<TensorImpl>(new TensorImpl(quant_edge, TENSORTYPE_NORMAL))));
    *options.tensors->find(quant_edge_id)->second->GetShape() = quant_shape;
    options.quants->resize(topo->GetCurrentEdgeIdBound());
    options.quants->at(quant_edge_id).format = quant_shape.GetDataFormat();
    options.quants->at(quant_edge_id).type = quant_shape.GetDataType();

    options.device->CopyFromHost(&quant_constat_info.GetBufferDesc(), scales.data(), quant_shape);
    options.info->constants.emplace(quant_edge_id, std::move(quant_constat_info));

    // Split weight format to group padding
    uint32_t k_per_grp = shape_in1.GetDim(0) / temp_conv_param.num_grp;
    uint32_t k_per_grp_pad = (k_per_grp + align_size - 1) / align_size * align_size;
    auto stream = options.device->GetStream();
    auto weight_iter = data->constants.find(weight_node->GetInput(0));
    if (weight_iter != data->constants.end() && // is a constant tensor and has not be loaded
        options.info->constants.find(weight_node->GetInput(0)) == options.info->constants.end()) {
        auto preedge_id = weight_node->GetInput(0);
        auto postedge_id = node->GetInput(1);
        const TensorShape& preshape = *options.tensors->find(preedge_id)->second->GetShape();
        const TensorShape& postshape = *options.tensors->find(postedge_id)->second->GetShape();
        auto newshape = postshape;
        newshape.SetPadding1(0, k_per_grp_pad * temp_conv_param.num_grp - newshape.GetDim(0));

        RuntimeConstantInfo weight_constat_info;
        {
            BufferDesc buffer;
            status = options.device->Realloc(newshape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            weight_constat_info.Reshape(newshape); // give the converted shape, but the dims are not changed
            weight_constat_info.SetBuffer(buffer, options.device, true);
        }

        ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, newshape.CalcBytesIncludingPadding(), RC_OUT_OF_MEMORY)
        status = ((CudaDataConverter*)options.device->GetDataConverter())
                     ->ConvertFromHost(&temp_buffer, postshape, (*quants)[postedge_id],
                                       weight_iter->second.data.GetData(), preshape, (*quants)[preedge_id]);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << node->GetName() << " copy constant failed: " << GetRetCodeStr(status);
            return status;
        }
        cudaMemcpyAsync(weight_constat_info.GetBufferDesc().addr, temp_buffer.addr,
                   shape_in1.CalcElementsIncludingPadding() * sizeof(int8_t), cudaMemcpyDeviceToDevice,  options.device->GetStream());

        options.info->constants.emplace(preedge_id, std::move(weight_constat_info));
        *options.tensors->find(preedge_id)->second->GetShape() = newshape;
        *options.tensors->find(postedge_id)->second->GetShape() = newshape;
        options.quants->at(preedge_id) = (*quants)[postedge_id];
        options.quants->at(preedge_id).type = postshape.GetDataType();
        options.quants->at(preedge_id).format = postshape.GetDataFormat();
    }

    reinterpret_cast<CudaConvParam*>(options.param)->extra_param.is_initializer_weight =
        weight_iter != data->constants.end();

    if (attr_param_.extra_param.bias_term == 0) {
        return RC_SUCCESS;
    }

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
        newshape.SetDim(0, k_per_grp_pad * temp_conv_param.num_grp);
        RuntimeConstantInfo bias_constat_info;
        {
            BufferDesc buffer;
            status = options.device->Realloc(newshape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            bias_constat_info.Reshape(newshape); // give the init shape, but the actual shape is padded
            bias_constat_info.SetBuffer(buffer, options.device, true);
        }

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
        options.quants->at(preedge_id) = (*quants)[postedge_id];
        options.quants->at(preedge_id).format = postshape.GetDataFormat();
        options.quants->at(preedge_id).type = postshape.GetDataType();
    }

    return RC_SUCCESS;
#endif // PPLNN_USE_DNN && PPLNN_USE_MACA
}

void TuringIMMAImpgemm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                       dataformat_t input_format, dataformat_t output_format) {
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) { // only reset formats of input0 and weight
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto shape = tensors->find(edge_id)->second->GetShape();
        if (shape->GetDimCount() > 1)
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
