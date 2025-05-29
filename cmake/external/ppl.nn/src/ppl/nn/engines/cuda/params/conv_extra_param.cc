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

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"

#include "ppl/nn/params/onnx/auto_pad_type.h"
#include "ppl/nn/params/onnx/leaky_relu_param.h"
#include "ppl/nn/params/onnx/hard_sigmoid_param.h"
#include "ppl/nn/params/onnx/swish_param.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;
using namespace ppl::nn::onnx;

namespace ppl { namespace nn { namespace cuda {

float cpu_half2float(unsigned short x)
{
    unsigned sign = ((x >> 15) & 1);
    unsigned exponent = ((x >> 10) & 0x1f);
    unsigned mantissa = ((x & 0x3ff) << 13);
    if (exponent == 0x1f) {  /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) {  /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1;  /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }
    int temp = ((sign << 31) | (exponent << 23) | mantissa);

    return *((float*)((void*)&temp));
}


int GetRelueType(const std::string& name) {
#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
    if (name == "Relu" || name == "ReLU" || name == "ReLU3d")
        return 0;
    if (name == "Sigmoid")
        return 1;
    if (name == "HardSigmoid")
        return 2;
    if (name == "Swish")
        return 3;
    if (name == "HardSwish")
        return 4;
    if (name == "Clip")
        return 5;
    if (name == "LeakyRelu")
        return 6;
    if (name == "PRelu")
        return 7;
    if (name == "Mish")
        return 8;
    return -1;
#else
    if (name == "Relu" || name == "ReLU" || name == "ReLU3d")
        return 0;
    if (name == "Clip")
        return 1;
    if (name == "PRelu")
        return 2;
    if (name == "LeakyRelu")
        return 3;
    if (name == "Sigmoid")
        return 4;
    return -1;
#endif
}

#define GetPadSize(pad_size, type)             \
    {                                          \
        pad_size = 0;                          \
        if (type == DATATYPE_FLOAT32) {        \
            pad_size = 4;                      \
        } else if (type == DATATYPE_FLOAT16) { \
            pad_size = 8;                      \
        } else if (type == DATATYPE_INT8) {    \
            pad_size = 16;                     \
        }                                      \
    }

#define Align(x, y) (((x) + (y)-1) / (y) * (y))

RetCode ConvertToForwardConvParam(const TensorShape& shape_in0, const TensorShape& shape_in1,
                                  const TensorShape& shape_out, const CudaConvParam& cuda_param,
                                  conv_param_t& conv_param) {
    const ConvParam& normal_param = cuda_param.param;

    //if (shape_in1.GetDimCount() == 3) { // 1d-conv case
    if (cuda_param.param.kernel_shape.size() == 1) { // 1d-conv case
        conv_param.in_height = shape_in0.GetDim(2);
        conv_param.in_width = 1;
        conv_param.in_num = shape_in0.GetDim(0);
        conv_param.num_grp = normal_param.group;
        conv_param.num_chl = shape_in1.GetDim(1) * normal_param.group;
        conv_param.num_flt = shape_in1.GetDim(0);
        unsigned int in_pad_size;
        unsigned int flt_pad_size;
        GetPadSize(in_pad_size, shape_in0.GetDataType());
        GetPadSize(flt_pad_size, shape_in1.GetDataType());
        // conv_param.num_chl_pad = (conv_param.num_chl + 7) / 8 * 8;
        // conv_param.num_flt_pad = (conv_param.num_flt + 7) / 8 * 8;
        // std::cout << "in pad size: " << in_pad_size << " flt_pad_size: " << flt_pad_size << std::endl;
        conv_param.num_chl_pad = Align(conv_param.num_chl, in_pad_size);
        conv_param.num_flt_pad = Align(conv_param.num_flt, flt_pad_size);
        conv_param.flt_height = shape_in1.GetDim(2);
        conv_param.flt_width = 1;
        conv_param.out_height = shape_out.GetDim(2);
        conv_param.out_width = 1;
        conv_param.stride_height = normal_param.strides[0];
        conv_param.stride_width = 1;
        if(normal_param.auto_pad == ppl::nn::onnx::AUTO_PAD_SAME_LOWER || normal_param.auto_pad == ppl::nn::onnx::AUTO_PAD_SAME_UPPER)
        {
            const int32_t kernel_dims = shape_in0.GetDimCount() - 2;
            int pad_h_temp = (shape_out.GetDim(kernel_dims) - 1) * normal_param.strides[0] + normal_param.kernel_shape[0] - shape_in0.GetDim(kernel_dims);
            conv_param.pad_height = normal_param.auto_pad == ppl::nn::onnx::AUTO_PAD_SAME_LOWER ? (pad_h_temp / 2 + pad_h_temp % 2) : (pad_h_temp / 2);
        }
        else{
            conv_param.pad_height = normal_param.pads[0];
        }
        conv_param.pad_width = 0;
        conv_param.hole_height = normal_param.dilations[0];
        conv_param.hole_width = 1;
        conv_param.has_bias = cuda_param.extra_param.bias_term;
    } else if(cuda_param.param.kernel_shape.size() == 2){ //2d-conv case
        conv_param.in_height = shape_in0.GetDim(2);
        conv_param.in_width = shape_in0.GetDim(3);
        conv_param.in_num = shape_in0.GetDim(0);
        conv_param.num_grp = normal_param.group;
        conv_param.num_chl = shape_in1.GetDim(1) * normal_param.group;
        conv_param.num_flt = shape_in1.GetDim(0);
        unsigned int in_pad_size;
        unsigned int flt_pad_size;
        GetPadSize(in_pad_size, shape_in0.GetDataType());
        GetPadSize(flt_pad_size, shape_in1.GetDataType());
        // conv_param.num_chl_pad = (conv_param.num_chl + 7) / 8 * 8;
        // conv_param.num_flt_pad = (conv_param.num_flt + 7) / 8 * 8;
        // std::cout << "in pad size: " << in_pad_size << " flt_pad_size: " << flt_pad_size << std::endl;
        conv_param.num_chl_pad = Align(conv_param.num_chl, in_pad_size);
        conv_param.num_flt_pad = Align(conv_param.num_flt, flt_pad_size);
        conv_param.flt_height = shape_in1.GetDim(2);
        conv_param.flt_width = shape_in1.GetDim(3);
        conv_param.out_height = shape_out.GetDim(2);
        conv_param.out_width = shape_out.GetDim(3);
        conv_param.stride_height = normal_param.strides[0];
        conv_param.stride_width = normal_param.strides[1];
        if(normal_param.auto_pad == ppl::nn::onnx::AUTO_PAD_SAME_LOWER || normal_param.auto_pad == ppl::nn::onnx::AUTO_PAD_SAME_UPPER)
        {
            const int32_t kernel_dims = shape_in0.GetDimCount() - 2;
            int pad_h_temp = (shape_out.GetDim(kernel_dims) - 1) * normal_param.strides[0] + normal_param.kernel_shape[0] - shape_in0.GetDim(kernel_dims);
            conv_param.pad_height = normal_param.auto_pad == ppl::nn::onnx::AUTO_PAD_SAME_LOWER ? (pad_h_temp / 2 + pad_h_temp % 2) : (pad_h_temp / 2);
            int pad_w_temp = (shape_out.GetDim(kernel_dims + 1) - 1) * normal_param.strides[1] + normal_param.kernel_shape[1] - shape_in0.GetDim(kernel_dims + 1);
            conv_param.pad_width = normal_param.auto_pad == ppl::nn::onnx::AUTO_PAD_SAME_LOWER ? (pad_w_temp / 2 + pad_w_temp % 2) : (pad_w_temp / 2);
        }
        else
        {
            conv_param.pad_height = normal_param.pads[0];
            conv_param.pad_width = normal_param.pads[1];
        }
        conv_param.hole_height = normal_param.dilations[0];
        conv_param.hole_width = normal_param.dilations[1];
        conv_param.has_bias = cuda_param.extra_param.bias_term;
    }

    return RC_SUCCESS;
}
#undef GetPadSize
#undef Align

RetCode ConvertToPrelu(uint32_t fuse_index, InputOutputInfo* info, CudaDevice* device, ConvFusionInfo fuse_info,
                       fuse_param_t& fuse_param) {
#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
    uint32_t prelu_slope_float_index = 0;
    uint32_t fuse_org_inputs = fuse_info.input_inds[fuse_info.input_inds.size() - 1];
    if( fuse_info.types[fuse_info.input_inds.size() - 1] == "Clip"){
        fuse_org_inputs += 2;
    } else if (fuse_info.types[fuse_info.input_inds.size() - 1] == "PRelu"){
        fuse_org_inputs += 1;
    } else if (fuse_info.types[fuse_info.input_inds.size() - 1] == "Add"){
        fuse_org_inputs += 1;
    } else if (fuse_info.types[fuse_info.input_inds.size() - 1] == "Mul"){
        fuse_org_inputs += 1;
    }

    if(fuse_index == 0){
        prelu_slope_float_index = fuse_org_inputs;
        //printf("prelu_slope_float_index : %d\n", prelu_slope_float_index);
    }else if(fuse_index == 1){
        prelu_slope_float_index = fuse_org_inputs + 1;
    }else if(fuse_index == 2){
        if(fuse_info.types[0] == "PRelu"){
            prelu_slope_float_index = fuse_org_inputs + 2;
        } else{
            prelu_slope_float_index = fuse_org_inputs + 1;
        }
    }else{
        LOG(ERROR) << "ConvertToPrelu fuse_index error: " << fuse_index;
    }
    const TensorShape& shape = *info->GetInput<TensorImpl>(prelu_slope_float_index)->GetShape();
    if (fuse_index == 0){
        fuse_param.has_prelu = shape.CalcElementsExcludingPadding();
        fuse_param.prelu = info->GetInput<TensorImpl>(prelu_slope_float_index)->GetBufferPtr();;
    } else{
        fuse_param.has_elt_prelu = shape.CalcElementsExcludingPadding();
        fuse_param.elt_prelu = info->GetInput<TensorImpl>(prelu_slope_float_index)->GetBufferPtr();;
    }
#else
    uint32_t prelu_input = fuse_info.input_inds[fuse_index];
    const TensorShape& shape = *info->GetInput<TensorImpl>(prelu_input)->GetShape();

    if (fuse_index == 0) {
        fuse_param.has_prelu = shape.IsScalar() ? 1 : 2;
        fuse_param.prelu = info->GetInput<TensorImpl>(prelu_input)->GetBufferPtr();
    } else {
        fuse_param.has_elt_prelu = shape.IsScalar() ? 1 : 2;
        fuse_param.elt_prelu = info->GetInput<TensorImpl>(prelu_input)->GetBufferPtr();
    }
#endif
    return RC_SUCCESS;
}

RetCode ConvertToLeakyrelu(uint32_t fuse_index, InputOutputInfo* info, CudaDevice* device, ConvFusionInfo fuse_info,
                           fuse_param_t& fuse_param) {
    if (fuse_index == 0) {
        fuse_param.has_prelu = 1;
        fuse_param.leaky = ((LeakyReluParam*)fuse_info.fuse_attrs[fuse_index])->alpha;
    } else {
        fuse_param.has_elt_prelu = 1;
        fuse_param.elt_leaky = ((LeakyReluParam*)fuse_info.fuse_attrs[fuse_index])->alpha;
    }

    return RC_SUCCESS;
}

RetCode ConvertToForwardFuseParam(InputOutputInfo* info, CudaDevice* device, const ConvFusionInfo& fuse_info,
                                  fuse_param_t& fuse_param) {
#if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
    const std::set<std::string> relu_set{"Relu", "Sigmoid", "HardSigmoid", "Swish", "HardSwish", "Clip", "PRelu", "LeakyRelu", "Mish"};
#else
    const std::set<std::string> relu_set{"Relu", "ReLU", "ReLU3d", "Clip", "PRelu", "LeakyRelu", "Sigmoid"};
#endif
    int fuse_index = 0;
    int fuse_size = fuse_info.types.size();

    RetCode status;
    CudaClipParam* param = nullptr;

    if (fuse_index < fuse_size && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
        int type = GetRelueType(fuse_info.types[fuse_index]);
        switch (type) {
        #if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
            case 0: // Relu
                fuse_param.has_activation = 1;
                break;
            case 1: // sigmoid
                fuse_param.has_activation = 2;
                break;
            case 2: // hardsigmoid
                fuse_param.has_activation = 3;
                fuse_param.act_alpha = ((HardSigmoidParam*)fuse_info.fuse_attrs[fuse_index])->alpha;
                fuse_param.act_beta = ((HardSigmoidParam*)fuse_info.fuse_attrs[fuse_index])->beta;
                break;
            case 3: // swish
                fuse_param.has_activation = 4;
                fuse_param.act_beta = ((SwishParam*)fuse_info.fuse_attrs[fuse_index])->beta;
                break;
            case 4: // hardswish
                fuse_param.has_activation = 5;
                fuse_param.act_alpha = ((HardSigmoidParam*)fuse_info.fuse_attrs[fuse_index])->alpha;
                fuse_param.act_beta = ((HardSigmoidParam*)fuse_info.fuse_attrs[fuse_index])->beta;
                break;
            case 5: // Clip
                fuse_param.has_activation = 6;
                fuse_param.clip_min = ((CudaClipParam*)fuse_info.fuse_attrs[fuse_index])->min_value;
                fuse_param.clip_max = ((CudaClipParam*)fuse_info.fuse_attrs[fuse_index])->max_value;
                break;
            case 6: // LeakyRelu
                fuse_param.has_activation = 7;
                fuse_param.leaky = ((LeakyReluParam*)fuse_info.fuse_attrs[fuse_index])->alpha;
                break;
            case 7: // PRelu
                fuse_param.has_activation = 8;
                status = ConvertToPrelu(fuse_index, info, device, fuse_info, fuse_param);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Set prelu fuse info failed: " << GetRetCodeStr(status);
                    return status;
                }
                break;
            case 8: // Mish
                fuse_param.has_activation = 9;
                break;
            default:
                return RC_UNSUPPORTED;
        #else
            case 0: // Relu
                fuse_param.has_activation = 1;
                break;
            case 1: // Clip
                fuse_param.has_clip = true;
                param = (CudaClipParam*)fuse_info.fuse_attrs[fuse_index];
                fuse_param.clip_min = param->min_value;
                fuse_param.clip_max = param->max_value;
                break;
            case 2: // PRelu
                status = ConvertToPrelu(fuse_index, info, device, fuse_info, fuse_param);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Set prelu fuse info failed: " << GetRetCodeStr(status);
                    return status;
                }
                break;
            case 3: // LeakyRelu
                status = ConvertToLeakyrelu(fuse_index, info, device, fuse_info, fuse_param);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Set prelu fuse info failed: " << GetRetCodeStr(status);
                    return status;
                }
                break;
            case 4: // Sigmoid
                fuse_param.has_activation = 2;
                break;
            default:
                return RC_UNSUPPORTED;
        #endif
        }
        fuse_index++;
    }

    if (fuse_index < fuse_size && (fuse_info.types[fuse_index] == "Add" || fuse_info.types[fuse_index] == "Mul")) {
        fuse_param.has_elt = true;
        fuse_param.elt_index = fuse_index;
        uint32_t elt_input = fuse_info.input_inds[fuse_index];
        fuse_param.pre_data = info->GetInput<TensorImpl>(elt_input)->GetBufferPtr();
        fuse_index++;
    }

    if (fuse_index < fuse_size && fuse_param.has_elt && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
        int type = GetRelueType(fuse_info.types[fuse_index]);
        switch (type) {
        #if defined(PPLNN_USE_DNN) && defined(PPLNN_USE_MACA)
            case 0: // Relu
                fuse_param.has_elt_activation = 1;
                break;
            case 1: // sigmoid
                fuse_param.has_elt_activation = 2;
                break;
            case 2: // hardsigmoid
                fuse_param.has_elt_activation = 3;
                fuse_param.elt_act_alpha = ((HardSigmoidParam*)fuse_info.fuse_attrs[fuse_index])->alpha;
                fuse_param.elt_act_beta = ((HardSigmoidParam*)fuse_info.fuse_attrs[fuse_index])->beta;
                break;
            case 3: // swish
                fuse_param.has_elt_activation = 4;
                fuse_param.elt_act_beta = ((SwishParam*)fuse_info.fuse_attrs[fuse_index])->beta;
                break;
            case 4: // hardswish
                fuse_param.has_elt_activation = 5;
                fuse_param.elt_act_alpha = ((HardSigmoidParam*)fuse_info.fuse_attrs[fuse_index])->alpha;
                fuse_param.elt_act_beta = ((HardSigmoidParam*)fuse_info.fuse_attrs[fuse_index])->beta;
                break;
            case 5: // Clip
                fuse_param.has_elt_activation = 6;
                fuse_param.elt_clip_min = ((CudaClipParam*)fuse_info.fuse_attrs[fuse_index])->min_value;
                fuse_param.elt_clip_max = ((CudaClipParam*)fuse_info.fuse_attrs[fuse_index])->max_value;
                break;
            case 6: // LeakyRelu
                fuse_param.has_elt_activation = 7;
                fuse_param.elt_leaky = ((LeakyReluParam*)fuse_info.fuse_attrs[fuse_index])->alpha;
                break;
            case 7: // PRelu
                fuse_param.has_elt_activation =8;
                status = ConvertToPrelu(fuse_index, info, device, fuse_info, fuse_param);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Set prelu fuse info failed: " << GetRetCodeStr(status);
                    return status;
                }
                break;
            case 8: // Mish
                fuse_param.has_elt_activation = 9;
            default:
                return RC_UNSUPPORTED;
        #else
            case 0: // Relu
                fuse_param.has_elt_activation = 1;
                break;
            case 1: // Clip
                fuse_param.has_elt_clip = true;
                param = (CudaClipParam*)fuse_info.fuse_attrs[fuse_index];
                fuse_param.elt_clip_min = param->min_value;
                fuse_param.elt_clip_max = param->max_value;
                break;
            case 2: // PRelu
                status = ConvertToPrelu(fuse_index, info, device, fuse_info, fuse_param);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Set prelu fuse info failed: " << GetRetCodeStr(status);
                    return status;
                }
                break;
            case 3: // LeakyRelu
                status = ConvertToLeakyrelu(fuse_index, info, device, fuse_info, fuse_param);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Set prelu fuse info failed: " << GetRetCodeStr(status);
                    return status;
                }
                break;
            case 4: // Sigmoid
                fuse_param.has_elt_activation = 2;
                break;
            default:
                return RC_UNSUPPORTED;
        #endif
        }
    }

    fuse_param.has_concat = fuse_info.channel_offset >= 0;
    if (fuse_param.has_concat) {
        fuse_param.concat_offset = fuse_info.channel_offset;
        fuse_param.concat_stride = fuse_info.channel_size;
        fuse_param.post_concat = info->GetOutput<TensorImpl>(0)->GetBufferPtr();
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
