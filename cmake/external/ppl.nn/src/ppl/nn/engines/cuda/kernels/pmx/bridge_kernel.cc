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

#include "ppl/nn/engines/cuda/kernels/pmx/bridge_kernel.h"

#include "cudakernel/reformat/reformat.h"
#include "ppl/common/cuda/cuda_types.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn { namespace cuda {

bool BridgeKernel::EqualTypeAndFormat(const TensorImpl* input, const TensorImpl* output, const CudaTensorQuant& in_quant, const CudaTensorQuant& out_quant) {
    auto src_align_size = ppl::common::cuda::GetDataFormatChannelAlignment(input->GetShape()->GetDataFormat());
    auto dst_align_size = ppl::common::cuda::GetDataFormatChannelAlignment(output->GetShape()->GetDataFormat());
    CudaTensorKernelQuant in_quant_kernel, out_quant_kernel;
    in_quant_kernel.format = in_quant.format; in_quant_kernel.type = in_quant.type;
    in_quant_kernel.per_channel = in_quant.per_channel; in_quant_kernel.bit_width = in_quant.bit_width;
    in_quant_kernel.scale = in_quant.scale; in_quant_kernel.zero_point = in_quant.zero_point;
    out_quant_kernel.format = out_quant.format; out_quant_kernel.type = out_quant.type;
    out_quant_kernel.per_channel = out_quant.per_channel; out_quant_kernel.bit_width = out_quant.bit_width;
    out_quant_kernel.scale = out_quant.scale; out_quant_kernel.zero_point = out_quant.zero_point;

    if (input->GetShape()->GetDataType() != output->GetShape()->GetDataType()) {
        return false;
    }

    if (input->GetShape()->GetDataType() == ppl::common::DATATYPE_INT8 && !EqualQuant(in_quant_kernel, out_quant_kernel)) {
        return false;
    }

    if (input->GetShape()->GetDataFormat() == output->GetShape()->GetDataFormat()) {
        return true;
    }

    if (input->GetShape()->GetDimCount() == 1 && output->GetShape()->GetDimCount() == 1) {
        return true;
    }

    if ((src_align_size != 0 && input->GetShape()->GetDim(1) % src_align_size != 0) ||
        (dst_align_size != 0 && output->GetShape()->GetDim(1) % dst_align_size != 0)) {
        return false;
    }

    if (input->GetShape()->GetDimCount() == 2 && output->GetShape()->GetDimCount() == 2) {
        return true;
    }

    if (input->GetShape()->GetDimCount() == 4 && output->GetShape()->GetDimCount() == 4 &&
        input->GetShape()->GetDim(2) == 1 && input->GetShape()->GetDim(3) == 1) {
        return true;
    }

    return false;
}

ppl::common::RetCode BridgeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    auto converter = output->GetDevice()->GetDataConverter();

    auto input_id = input->GetEdge()->GetId();
    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input_id);
    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);

    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        EqualTypeAndFormat(input, output, input_quant, output_quant)) {
        output->TransferBufferFrom(input);
        if(output->GetShape()->GetDataFormatForceTo() != ppl::common::DATAFORMAT_UNKNOWN){
            if(output->GetShape()->GetDataFormat() == ppl::common::DATAFORMAT_NHWC && output->GetShape()->GetDataFormatForceTo() == ppl::common::DATAFORMAT_NDARRAY){
                if(output->GetShape()->GetDimCount() == 4){
                    auto tensor_dim_count = output->GetShape()->GetDimCount();
                    std::string tensor_dims = "[";
                    for (uint32_t j = 0; j < tensor_dim_count; ++j) {
                        tensor_dims += std::to_string(output->GetShape()->GetDim(j)) + " ";
                    }
                    tensor_dims += "]";
                    std::string tensor_format = ppl::common::GetDataFormatStr(output->GetShape()->GetDataFormat());
                    std::vector<int64_t> dims;
                    auto dim_data = output->GetShape()->GetDims();
                    dims.push_back(dim_data[0]);
                    dims.push_back(dim_data[2]);
                    dims.push_back(dim_data[3]);
                    dims.push_back(dim_data[1]);
                    output->GetShape()->Reshape(dims);
                    std::string new_tensor_dims = "[";
                    for (uint32_t j = 0; j < tensor_dim_count; ++j) {
                        new_tensor_dims += std::to_string(output->GetShape()->GetDim(j)) + " ";
                    }
                    new_tensor_dims += "]";
                    output->GetShape()->SetDataFormat(output->GetShape()->GetDataFormatForceTo());
                    std::string new_tensor_format = ppl::common::GetDataFormatStr(output->GetShape()->GetDataFormat());
                    LOG(DEBUG) << "bridge output: " << tensor_format << tensor_dims << "---> " << new_tensor_format << new_tensor_dims;
                    if(output->GetShape()->GetChannelSpiltCount() > 0){
                        dims.pop_back();
                        for(int i = 0; i < output->GetShape()->GetChannelSpiltCount(); i++){
                            dims.push_back((output->GetShape()->GetChannelSpilt())[i]);
                        }
                        output->GetShape()->Reshape(dims);
                        std::string new_tensor_dims_2 = "[";
                        for (uint32_t j = 0; j < dims.size(); ++j) {
                            new_tensor_dims_2 += std::to_string(output->GetShape()->GetDim(j)) + " ";
                        }
                        new_tensor_dims_2 += "]";
                        LOG(DEBUG) << "bridge output channel split: " << new_tensor_format << new_tensor_dims << "---> " << new_tensor_format << new_tensor_dims_2;
                    }
                }
            }
        }
        return status;
    }
    auto input_shape = *input->GetShape();
    if(input_shape.GetDimCount() == 5){
        std::vector<int64_t> input_dim_new;
        for(int i = 0; i < 4; i++)
            input_dim_new.push_back(input_shape.GetDim(i));
        input_dim_new[3] *= input_shape.GetDim(4);
        input_shape.Reshape(input_dim_new);
    }
    auto output_shape = *output->GetShape();
    if(output_shape.GetDimCount() == 5){
        std::vector<int64_t> output_dim_new;
        for(int i = 0; i < 4; i++)
            output_dim_new.push_back(output_shape.GetDim(i));
        output_dim_new[3] *= output_shape.GetDim(4);
        output_shape.Reshape(output_dim_new);
    }
    if (input->GetShape()->GetDataType() != ppl::common::DATATYPE_INT8 &&
        output->GetShape()->GetDataType() != ppl::common::DATATYPE_INT8) {
        if(input->GetShape()->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 &&
            output->GetShape()->GetDataFormat() == ppl::common::DATAFORMAT_NHWC &&
            input->GetShape()->CalcElementsIncludingPadding() == input->GetShape()->CalcElementsExcludingPadding() &&
            input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
            input->GetShape()->GetDataType() == output->GetShape()->GetDataType()
            ){
            output->TransferBufferFrom(input);
        }else{
            status = converter->Convert(&output->GetBufferDesc(), output_shape, input->GetBufferDesc(), input_shape);
        }
    } else {
        status = ((CudaDataConverter*)converter)->Convert(&output->GetBufferDesc(), output_shape, output_quant, input->GetBufferDesc(), input_shape, input_quant);
    }
    if(output->GetShape()->GetDataFormatForceTo() != ppl::common::DATAFORMAT_UNKNOWN){
        if(output->GetShape()->GetDataFormat() == ppl::common::DATAFORMAT_NHWC && output->GetShape()->GetDataFormatForceTo() == ppl::common::DATAFORMAT_NDARRAY){
            if(output->GetShape()->GetDimCount() == 4){
                auto tensor_dim_count = output->GetShape()->GetDimCount();
                std::string tensor_dims = "[";
                for (uint32_t j = 0; j < tensor_dim_count; ++j) {
                    tensor_dims += std::to_string(output->GetShape()->GetDim(j)) + " ";
                }
                tensor_dims += "]";
                std::string tensor_format = ppl::common::GetDataFormatStr(output->GetShape()->GetDataFormat());
                std::vector<int64_t> dims;
                auto dim_data = output->GetShape()->GetDims();
                dims.push_back(dim_data[0]);
                dims.push_back(dim_data[2]);
                dims.push_back(dim_data[3]);
                dims.push_back(dim_data[1]);
                output->GetShape()->Reshape(dims);
                std::string new_tensor_dims = "[";
                for (uint32_t j = 0; j < tensor_dim_count; ++j) {
                    new_tensor_dims += std::to_string(output->GetShape()->GetDim(j)) + " ";
                }
                new_tensor_dims += "]";
                output->GetShape()->SetDataFormat(output->GetShape()->GetDataFormatForceTo());
                std::string new_tensor_format = ppl::common::GetDataFormatStr(output->GetShape()->GetDataFormat());
                LOG(DEBUG) << "bridge output: " << tensor_format << tensor_dims << "---> " << new_tensor_format << new_tensor_dims;
                if(output->GetShape()->GetChannelSpiltCount() > 0 ){
                    if((output->GetShape()->GetChannelSpilt())[0] != 0){
                        dims.pop_back();
                        for(int i = 0; i < output->GetShape()->GetChannelSpiltCount(); i++){
                            dims.push_back((output->GetShape()->GetChannelSpilt())[i]);
                        }
                        output->GetShape()->Reshape(dims);
                        std::string new_tensor_dims_2 = "[";
                        for (uint32_t j = 0; j < dims.size(); ++j) {
                            new_tensor_dims_2 += std::to_string(output->GetShape()->GetDim(j)) + " ";
                        }
                        new_tensor_dims_2 += "]";
                        LOG(DEBUG) << "bridge output channel split: " << new_tensor_format << new_tensor_dims << "---> " << new_tensor_format << new_tensor_dims_2;
                    }else{
                        dims.clear();
                        dims.push_back(dim_data[0]);
                        for(int i = 1; i < output->GetShape()->GetChannelSpiltCount(); i++){
                            dims.push_back((output->GetShape()->GetChannelSpilt())[i]);
                        }
                        output->GetShape()->Reshape(dims);
                        std::string new_tensor_dims_2 = "[";
                        for (uint32_t j = 0; j < dims.size(); ++j) {
                            new_tensor_dims_2 += std::to_string(output->GetShape()->GetDim(j)) + " ";
                        }
                        new_tensor_dims_2 += "]";
                        LOG(DEBUG) << "bridge output channel merge: " << new_tensor_format << new_tensor_dims << "---> " << new_tensor_format << new_tensor_dims_2;
                    }

                }
            }else if(output->GetShape()->GetDimCount() == 3){
                auto tensor_dim_count = output->GetShape()->GetDimCount();
                std::string tensor_dims = "[";
                for (uint32_t j = 0; j < tensor_dim_count; ++j) {
                    tensor_dims += std::to_string(output->GetShape()->GetDim(j)) + " ";
                }
                tensor_dims += "]";
                std::string tensor_format = ppl::common::GetDataFormatStr(output->GetShape()->GetDataFormat());
                std::vector<int64_t> dims;
                auto dim_data = output->GetShape()->GetDims();
                dims.push_back(dim_data[0]);
                dims.push_back(dim_data[2]);
                dims.push_back(dim_data[1]);
                output->GetShape()->Reshape(dims);
                std::string new_tensor_dims = "[";
                for (uint32_t j = 0; j < tensor_dim_count; ++j) {
                    new_tensor_dims += std::to_string(output->GetShape()->GetDim(j)) + " ";
                }
                new_tensor_dims += "]";
                output->GetShape()->SetDataFormat(output->GetShape()->GetDataFormatForceTo());
                std::string new_tensor_format = ppl::common::GetDataFormatStr(output->GetShape()->GetDataFormat());
                LOG(DEBUG) << "bridge output: " << tensor_format << tensor_dims << "---> " << new_tensor_format << new_tensor_dims;
                if(output->GetShape()->GetChannelSpiltCount() > 0){
                    dims.pop_back();
                    for(int i = 0; i < output->GetShape()->GetChannelSpiltCount(); i++){
                        dims.push_back((output->GetShape()->GetChannelSpilt())[i]);
                    }
                    output->GetShape()->Reshape(dims);
                    std::string new_tensor_dims_2 = "[";
                    for (uint32_t j = 0; j < dims.size(); ++j) {
                        new_tensor_dims_2 += std::to_string(output->GetShape()->GetDim(j)) + " ";
                    }
                    new_tensor_dims_2 += "]";
                    LOG(DEBUG) << "bridge output channel split: " << new_tensor_format << new_tensor_dims << "---> " << new_tensor_format << new_tensor_dims_2;
                }
            }
        }
    }

    return status;
}

}}} // namespace ppl::nn::cuda
