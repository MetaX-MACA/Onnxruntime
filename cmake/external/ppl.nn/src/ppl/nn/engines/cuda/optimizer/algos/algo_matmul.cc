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

#include "ppl/nn/engines/cuda/optimizer/algos/algo_matmul.h"

#include <chrono>

#include "ppl/common/cuda/cuda_types.h"
#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/gemm/bgemm.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/utils.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

void MatMulAlgorithm::DeleteAttrParam(void*& param) {
    delete (CudaGemmParam*)param;
    return;
}

void MatMulAlgorithm::GetAttrParam(void*& param) const {
    if (param == nullptr) {
        param = new CudaGemmParam();
    }
    *(CudaGemmParam*)param = attr_param_;
    return;
}

bool MatMulAlgorithm::IsSupported(const ir::Node* node, const OptKernelOptions& options,
                                  dataformat_t input_format) const {
    // check if matmul is fp32 type
    const TensorShape& tensor0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
#ifdef PPLNN_USE_DNN
    if (tensor0.GetDataType() != ppl::common::DATATYPE_FLOAT16 && tensor0.GetDataType() != ppl::common::DATATYPE_INT8
        && tensor0.GetDataType() != ppl::common::DATATYPE_FLOAT32) {
        return false;
    }
    return true;
#else
    if (tensor0.GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        return false;
    }
    // check if matmul is quantization
    auto quant0 = options.quants->at(node->GetInput(0));
    if (quant0.type == DATATYPE_INT8 && input_format != DATAFORMAT_NHWC16) {
        return false;
    }
    if (quant0.type == DATATYPE_FLOAT16 && input_format != DATAFORMAT_NHWC8) {
        return false;
    }
    if (quant0.type == DATATYPE_FLOAT32) {
        return false;
    }
    return true;
#endif
}

double MatMulAlgorithm::ExcuteTimer(const ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaGemmParam*>(options.param));
#ifdef PPLNN_USE_DNN
    return 0.0f;
#endif
    options.compile_set->emplace(node->GetId());
    if (node->GetInputCount() == 3) {
        attr_param_.extra_param.bias_term = true;
    }

    auto shape_in0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    auto shape_in1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    auto shape_in2 = TensorShape();
    auto shape_out = *options.tensors->find(node->GetOutput(0))->second->GetShape();
    auto align_size = shape_in0.GetDataType() == DATATYPE_FLOAT16 ? 8 : 16;
    auto dim_count0 = shape_in0.GetDimCount();
    auto dim_count1 = shape_in1.GetDimCount();
    auto out_dim_count = shape_out.GetDimCount();

    { // Give the default kernel
        if (shape_in0.GetDataType() == DATATYPE_FLOAT16) {
            attr_param_.extra_param.algo_info.algo_name = "nv2spkSm75Fp16Conv_hmma1688_nhwc_f1_b128x128_w64x64_k32_s32_buf1";
        } else if (shape_in0.GetDataType() == DATATYPE_INT8) {
            attr_param_.extra_param.algo_info.algo_name = "nv2spkSm75Int8Conv_imma8816_nhwc_f1_b64x64_w64x32_k32_s16_buf1";
        } else {
            return ALGO_MAX_TIME;
        }
        attr_param_.extra_param.algo_info.kid = 0; // TODO
        attr_param_.extra_param.algo_info.splitk = 1;
        attr_param_.extra_param.algo_info.splitf = 1;
        attr_param_.extra_param.algo_info.ParseAlgoName();
    }

    if (dim_count0 < 2 || dim_count1 < 2) {
        return 0.0f;
    }

    // Padding
    auto K = shape_in0.GetDim(dim_count0 - 1);
    auto N = shape_in1.GetDim(dim_count1 - 1);

    shape_in0.SetDim(dim_count0 - 1, (K + align_size - 1) / align_size * align_size);
    shape_in1.SetDim(dim_count1 - 2, (K + align_size - 1) / align_size * align_size);
    shape_out.SetDim(out_dim_count - 1, (N + align_size - 1) / align_size * align_size);
    if (attr_param_.extra_param.bias_term) {
        shape_in2 = *options.tensors->find(node->GetInput(2))->second->GetShape();
        shape_in2.SetDim(0, (shape_in2.GetDim(0) + align_size - 1) / align_size * align_size);
    }

    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param;
    temp_conv_param.in_num =
        attr_param_.param.transA ? shape_in0.GetDim(dim_count0 - 1) : shape_in0.GetDim(dim_count0 - 2);
    int m_id = dim_count0 - 2;
#if 0
    if (temp_conv_param.in_num == 1) {
        int m_id = dim_count0 - 3;
        while (m_id && shape_in0.GetDim(m_id) == 1)
            m_id--;
        temp_conv_param.in_num = shape_in0.GetDim(m_id);
    }
#endif
    int batch = 1;
    for (int i = 0; i < m_id; i++) {
        batch *= shape_in0.GetDim(i);
    }
    if (dim_count1 == 2) {
        temp_conv_param.in_num *= batch;
        batch = 1;
    }
    temp_conv_param.num_chl =
        attr_param_.param.transB ? shape_in1.GetDim(dim_count1 - 1) : shape_in1.GetDim(dim_count1 - 2);
    temp_conv_param.num_flt =
        attr_param_.param.transB ? shape_in1.GetDim(dim_count1 - 2) : shape_in1.GetDim(dim_count1 - 1);
    temp_conv_param.in_height = 1;
    temp_conv_param.in_width = 1;
    temp_conv_param.flt_height = 1;
    temp_conv_param.flt_width = 1;
    temp_conv_param.out_height = 1;
    temp_conv_param.out_width = 1;
    temp_conv_param.pad_height = 1;
    temp_conv_param.pad_width = 1;
    temp_conv_param.stride_height = 1;
    temp_conv_param.stride_width = 1;
    temp_conv_param.hole_height = 1;
    temp_conv_param.hole_width = 1;
    temp_conv_param.num_grp = 1;
    temp_conv_param.has_bias = 0;

    const std::string& key_str = node->GetName();
    auto algo_info = options.args->alog_selects.find(key_str);
    if (algo_info != options.args->alog_selects.end()) {
        attr_param_.extra_param.algo_info.kid = algo_info->second.kid;
        attr_param_.extra_param.algo_info.splitk = algo_info->second.splitk;
        attr_param_.extra_param.algo_info.splitf = algo_info->second.splitf;
	    attr_param_.extra_param.algo_info.gemm_batch = batch;
        attr_param_.extra_param.algo_info.algo_name = algo_info->second.kname;
        if (algo_info->second.splitk > 1)
            attr_param_.extra_param.algo_info.algo_name += "_spk" + std::to_string(algo_info->second.splitk);
        attr_param_.extra_param.algo_info.ParseAlgoName();
        return 0.0f;
    }

    if (options.args->quick_select) {
        return 0.0f;
    }

    RetCode status;
    ALLOC_BUFFERF_FOR_ALGO_SELECT(input_buffer, shape_in0.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(weight_buffer, shape_in1.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(bias_buffer, shape_in2.CalcBytesIncludingPadding(), ALGO_MAX_TIME)
    ALLOC_BUFFERF_FOR_ALGO_SELECT(output_buffer, shape_out.CalcBytesIncludingPadding(), ALGO_MAX_TIME)

    uint64_t size = PPLBgemmCUDAGetBufSize(&shape_in0, attr_param_.param.transA);
    ALLOC_BUFFERF_FOR_ALGO_SELECT(temp_buffer, size, ALGO_MAX_TIME)

    auto stream = options.device->GetStream();

    double timer = ALGO_MAX_TIME;
#ifdef PPLNN_ENABLE_CUDA_JIT
    // Do select
    LOG(INFO) << "Compiling " << node->GetName();
    if (shape_in0.GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        timer = PPLCUDABgemmJITSelectKernel(options.device->GetDeviceProp(), stream, shape_in0.GetDataType(), &shape_in0, input_buffer.addr,
                                            &shape_in1, weight_buffer.addr, bias_buffer.addr, &shape_out,
                                            output_buffer.addr, temp_buffer.addr, temp_conv_param, temp_fuse_param,
                                            attr_param_.extra_param.algo_info);
    }
    LOG(INFO) << "select kernel " << attr_param_.extra_param.algo_info.algo_name;
#else
    // Do Select
    GemmKernelParam param_kernel_;
    param_kernel_.alpha = attr_param_.param.alpha;
    param_kernel_.beta = attr_param_.param.beta;
    param_kernel_.transA = attr_param_.param.transA;
    param_kernel_.transB = attr_param_.param.transB;
    if (shape_in0.GetDataType()==ppl::common::DATATYPE_FLOAT16) {
        timer = PPLCUDABgemmSelectKernel(options.device->GetDeviceProp(), stream, &shape_in0, input_buffer.addr, &shape_in1, weight_buffer.addr,
                                         &shape_out, output_buffer.addr, temp_buffer.addr,
                                         param_kernel_, temp_fuse_param, attr_param_.extra_param.algo_info);
    }
#endif
    CudaArgs::AlgoSelects algo_select;
    algo_select.kname = attr_param_.extra_param.algo_info.algo_name;
    algo_select.kid = attr_param_.extra_param.algo_info.kid;
    algo_select.splitk = attr_param_.extra_param.algo_info.splitk;
    algo_select.splitf = attr_param_.extra_param.algo_info.splitf;
    options.args->alog_selects.emplace(key_str, std::move(algo_select));
    return timer;
}

RetCode MatMulAlgorithm::ModifyParam(ir::Node* node, OptKernelOptions& options) {
    this->attr_param_ = *(reinterpret_cast<CudaGemmParam*>(options.param));
    const TensorShape& shape_in0 = *options.tensors->find(node->GetInput(0))->second->GetShape();
    const TensorShape& shape_in1 = *options.tensors->find(node->GetInput(1))->second->GetShape();
    const TensorShape& shape_out = *options.tensors->find(node->GetOutput(0))->second->GetShape();
#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
    if (shape_in0.GetDataType() == DATATYPE_INT8 && shape_in1.GetDataType() == DATATYPE_INT8) {
        auto topo = options.graph->topo.get();

        auto& input_quant = options.quants->at(node->GetInput(0));
        auto& weight_quant = options.quants->at(node->GetInput(1));
        auto& output_quant = options.quants->at(node->GetOutput(0));

        int total_size = weight_quant.scale.size();
        std::vector<float> scales(total_size);
        for (int i = 0; i < total_size; i++) {
            scales[i] = input_quant.scale[0] * weight_quant.scale[i] / output_quant.scale[0];
        }

        auto quant_shape = TensorShape();
        quant_shape.SetDimCount(1);
        quant_shape.SetDim(0, total_size);
        quant_shape.SetDataFormat(DATAFORMAT_NDARRAY);
        quant_shape.SetDataType(DATATYPE_FLOAT32);

        RuntimeConstantInfo quant_constat_info;
        {
            BufferDesc buffer;
            auto status = options.device->Realloc(quant_shape, &buffer);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                return status;
            }

            quant_constat_info.Reshape(quant_shape);
            quant_constat_info.SetBuffer(buffer, options.device, true);
        }

        auto ret_pair = topo->AddEdge("Quant_alpha_" + node->GetName());
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
    } else if(shape_in0.GetDataType() == DATATYPE_FLOAT16 && shape_in1.GetDataType() == DATATYPE_FLOAT16){
    #if USE_MATMUL_K8_ALIGN
        auto data = options.graph->data.get();
        auto topo = options.graph->topo.get();
        auto inshape1_edge = topo->GetEdge(node->GetInput(1));
        auto inshape1_node = topo->GetNode(inshape1_edge->GetProducer());
        auto inshape1_iter = data->constants.find(inshape1_node->GetInput(0));
        const TensorShape& preshape = *options.tensors->find(inshape1_node->GetInput(0))->second->GetShape();
        if(inshape1_iter != data->constants.end() && options.info->constants.find(inshape1_node->GetInput(0)) == options.info->constants.end()){
            if(shape_in1.GetDimCount() == 2){
                auto temp_shape_in1 = shape_in1;
                temp_shape_in1.SetDim(0, ALIGN_UP(shape_in1.GetDim(0), 8));
                RuntimeConstantInfo shape_in1_constat_info;
                {
                    BufferDesc buffer;
                    auto status = options.device->Realloc(temp_shape_in1, &buffer);
                    if (status != RC_SUCCESS) {
                        LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                        return status;
                    }
                    shape_in1_constat_info.Reshape(shape_in1);
                    shape_in1_constat_info.SetBuffer(buffer, options.device, true);
                }
                cudaMemsetAsync(shape_in1_constat_info.GetBufferPtr(), 0, temp_shape_in1.CalcBytesIncludingPadding(), options.device->GetStream());
                auto status = options.device->GetDataConverter()->ConvertFromHost(&shape_in1_constat_info.GetBufferDesc(), shape_in1,
                                                                                  inshape1_iter->second.data.GetData(), preshape);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Matmul input1 ConvertFromHost failed: " << GetRetCodeStr(status);
                    return status;
                }
                options.info->constants.emplace(inshape1_node->GetInput(0), std::move(shape_in1_constat_info));
                *options.tensors->find(inshape1_node->GetInput(0))->second->GetShape() = shape_in1;
                *options.tensors->find(node->GetInput(1))->second->GetShape() = shape_in1;
            } else if(shape_in1.GetDimCount() >= 3){
                BufferDesc shape1_temp_buffer;
                auto status = options.device->Realloc(shape_in1, &shape1_temp_buffer);
                if (status != ppl::common::RC_SUCCESS) {
                    LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                    return status;
                }
                p_options_ = &options;
                ppl::common::Destructor __tmp_buffer_guard__([this, &shape1_temp_buffer]() -> void {
                    p_options_->device->Free(&shape1_temp_buffer);
                });
                cudaMemsetAsync(shape1_temp_buffer.addr, 0, shape_in1.CalcBytesIncludingPadding(), options.device->GetStream());
                status = options.device->GetDataConverter()->ConvertFromHost(&shape1_temp_buffer, shape_in1,
                                                                             inshape1_iter->second.data.GetData(), preshape);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Matmul input1 ConvertFromHost failed: " << GetRetCodeStr(status);
                    return status;
                }
                auto temp_shape_in1 = shape_in1;
                temp_shape_in1.SetDim(shape_in1.GetDimCount() - 2, ALIGN_UP(shape_in1.GetDim(shape_in1.GetDimCount() - 2), 8));
                RuntimeConstantInfo shape_in1_constat_info;
                {
                    BufferDesc buffer;
                    auto status = options.device->Realloc(temp_shape_in1, &buffer);
                    if (status != RC_SUCCESS) {
                        LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                        return status;
                    }
                    shape_in1_constat_info.Reshape(shape_in1);
                    shape_in1_constat_info.SetBuffer(buffer, options.device, true);
                }
                cudaMemsetAsync(shape_in1_constat_info.GetBufferPtr(), 0, temp_shape_in1.CalcBytesIncludingPadding(), options.device->GetStream());
                cudaMemcpy2DAsync(shape_in1_constat_info.GetBufferPtr(),
                                  temp_shape_in1.CalcBytesFromDimesionIncludingPadding(temp_shape_in1.GetDimCount() - 2),
                                  shape1_temp_buffer.addr,
                                  shape_in1.CalcBytesFromDimesionIncludingPadding(shape_in1.GetDimCount() - 2),
                                  shape_in1.CalcBytesFromDimesionIncludingPadding(shape_in1.GetDimCount() - 2),
                                  shape_in1.CalcElementsToDimensionExcludingPadding(shape_in1.GetDimCount() - 2),
                                  cudaMemcpyDeviceToDevice, options.device->GetStream());
                options.info->constants.emplace(inshape1_node->GetInput(0), std::move(shape_in1_constat_info));
                *options.tensors->find(inshape1_node->GetInput(0))->second->GetShape() = shape_in1;
                *options.tensors->find(node->GetInput(1))->second->GetShape() = shape_in1;
            }
            reinterpret_cast<CudaGemmParam*>(options.param)->extra_param.is_initializer_weight = 1;
        }
    #endif//end USE_MATMUL_K8_ALIGN
    }
#endif
    return RC_SUCCESS;
}

void MatMulAlgorithm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                     dataformat_t input_format, dataformat_t output_format) {
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) { // only reset formats of input0 and weight
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto shape = tensors->find(edge_id)->second->GetShape();
        if (shape->GetDimCount() > 1) {
            shape->SetDataFormat(input_format);
        } else {
            shape->SetDataFormat(DATAFORMAT_NDARRAY);
        }
    }

    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto shape = tensors->find(edge_id)->second->GetShape();
        shape->SetDataFormat(output_format);
    }
    return;
}

}}} // namespace ppl::nn::cuda
