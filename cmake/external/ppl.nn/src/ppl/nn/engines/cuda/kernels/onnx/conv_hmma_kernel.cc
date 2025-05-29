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
#include "ppl/nn/params/onnx/auto_pad_type.h"
#include "ppl/nn/engines/cuda/kernels/onnx/conv_hmma_kernel.h"
#include "ppl/common/cuda/cuda_types.h"
#include "ppl/common/destructor.h"
#include <cuda_fp16.h>
#include "cudakernel/arithmetic/arithmetic.h"
extern bool gSupportNc1hwc0;
namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode ConvHmmaKernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "Reshape failed: " << GetRetCodeStr(status);
        return status;
    }
    if(param_->extra_param.fuse_info.output_channels.empty() == false){
        auto output_channels = param_->extra_param.fuse_info.output_channels;
        auto output0 = ctx->GetOutput<TensorImpl>(0);
        auto output0_shape = output0->GetShape();

        uint64_t output_buffer_len =  output0_shape->CalcBytesIncludingPadding() * std::accumulate(output_channels.begin(), output_channels.end(), 0) / output_channels[0];

        // alloc output buffer
        CudaDevice* device = GetCudaDevice();
        BufferDesc output_buffer;
        device->Realloc(output_buffer_len, &output_buffer);

        output0->SetBuffer(output_buffer, device, true);
        for (uint32_t i = 1; i < ctx->GetOutputCount(); ++i) {
            output_buffer.addr += ctx->GetOutput<TensorImpl>(i-1)->GetShape()->CalcBytesIncludingPadding();
            auto tensor = ctx->GetOutput<TensorImpl>(i);
            tensor->SetDevice(GetCudaDevice());
            tensor->SetBuffer(output_buffer);
        }
    }else{
        for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
            auto tensor = ctx->GetOutput<TensorImpl>(i);
            auto device = GetCudaDevice();
            tensor->SetDevice(device);
            auto concat_edge_id = param_->extra_param.fuse_info.concat_edge_id;
            if (param_->extra_param.fuse_info.channel_offset >= 0) {
                auto edge2buffer = device->GetEdge2Buffer();
                auto ptr = edge2buffer->find(concat_edge_id);
                if (ptr == edge2buffer->end()) {
                    BufferDesc buffer;
                    auto concat_shape = *tensor->GetShape();
                    auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(concat_shape.GetDataFormat());
                    auto channel_size = param_->extra_param.fuse_info.channel_size;
                    auto channel_size_pad = (channel_size + align_size - 1) / align_size * align_size;
                    concat_shape.SetDim(1, channel_size_pad);
                    status = device->Realloc(concat_shape, &buffer);
                    if (status != RC_SUCCESS) {
                        LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                        return status;
                    }
                    tensor->SetBuffer(buffer);
                    edge2buffer->emplace(concat_edge_id, std::move(buffer));
                } else {
                    tensor->SetBuffer(ptr->second);
                }
            } else {
                status = tensor->ReallocBuffer();
            }
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName() << "] failed.";
                return status;
            }
        }
    }


    return ppl::common::RC_SUCCESS;
}

#ifdef PPLNN_USE_DNN
//#if 1
#define  CHECK_CUDNN_STATUS(cudnn_api_str, status)     \
        do{\
            if(status != CUDNN_STATUS_SUCCESS){\
                LOG(ERROR) <<"cuDNN interface(" << #cudnn_api_str << ") return error code_str: " << cudnnGetErrorString(status);\
                int version = cudnnGetVersion();\
                LOG(ERROR) << "mcDNN version: " << version;\
                int n=0, c=0, h=0, w=0, n_stride=0, c_stride=0, h_stride=0, w_stride=0;\
                cudnnDataType_t dataType=(cudnnDataType_t)0;\
                cudnnGetTensor4dDescriptor(x_tensor_, &dataType, &n, &c, &h, &w, &n_stride, &c_stride, &h_stride, &w_stride);\
                LOG(ERROR) << "x tensor info: dataType(" << (int)dataType<<"),dim(" <<n<<" "<<c<<" "<<h<<" "<<w<<") stride("<<n_stride<<" "<<c_stride<<" "<<h_stride<<" "<<w_stride<<")";\
                cudnnGetTensor4dDescriptor(y_tensor_, &dataType, &n, &c, &h, &w, &n_stride, &c_stride, &h_stride, &w_stride);\
                LOG(ERROR) << "y tensor info: dataType(" << (int)dataType<<"),dim(" <<n<<" "<<c<<" "<<h<<" "<<w<<") stride("<<n_stride<<" "<<c_stride<<" "<<h_stride<<" "<<w_stride<<")";\
                cudnnGetConvolution2dForwardOutputDim(conv_desc_, x_tensor_, w_desc_, &n, &c, &h, &w);\
                LOG(ERROR) << "cudnn dest y tensor shape: " << n<<" "<<c<<" "<<h<<" "<<w;\
                cudnnTensorFormat_t format=(cudnnTensorFormat_t)0;\
                cudnnGetFilter4dDescriptor(w_desc_, &dataType, &format, &n, &c, &h, &w);\
                LOG(ERROR) << "w desc info: dataType(" << (int)dataType<<"),format(" << (int)format<<"),dim(" << n<<" " << c<<" " << h<<" " << w<<")";\
                int pad_w=0, pad_h=0, u=0, v=0, dl_h=0, dl_w=0;\
                cudnnConvolutionMode_t mode=(cudnnConvolutionMode_t)0;\
                cudnnDataType_t computeType=(cudnnDataType_t)0;\
                cudnnGetConvolution2dDescriptor(conv_desc_, &pad_w, &pad_h, &u, &v, &dl_h, &dl_w, &mode, &computeType);\
                LOG(ERROR) << "conv desc info: pad(" << pad_w<<" " << pad_h<<"),uv(" << u<<" " << v<<"),diation(" << dl_h<<" " << dl_w<<"),mode(" << (int)mode<<"),computeType(" << (int)computeType<<")";\
                int group=0;\
                cudnnGetConvolutionGroupCount(conv_desc_, &group);\
                LOG(ERROR) << "group info: " << group;\
                cudnnMathType_t mathType = (cudnnMathType_t)0;\
                cudnnGetConvolutionMathType(conv_desc_, &mathType);\
                LOG(ERROR) << "mathType info: " << (int)mathType;\
                return ppl::common::RC_OTHER_ERROR;\
            }\
        }while(0)
#define  CHECK_CUDNN_ND_STATUS(cudnn_api_str, status)     \
        do{\
            if(status != CUDNN_STATUS_SUCCESS){\
                LOG(ERROR) <<"cuDNN interface(" << #cudnn_api_str << ") return error code_str: " << cudnnGetErrorString(status);\
                int version = cudnnGetVersion();\
                LOG(ERROR) << "mcDNN version: " << version;\
                int nbDims, dimA[5], strideA[5], filterDimA[5], pad[5], dilationA[5];\
                cudnnDataType_t dataType=(cudnnDataType_t)0;\
                cudnnGetTensorNdDescriptor(x_tensor_, 5, &dataType, &nbDims, dimA, strideA);\
                LOG(ERROR) << "x tensor info: dataType(" << (int)dataType<<"),dim(" <<dimA[0]<<" "<<dimA[1]<<" "<<dimA[2]<<" "<<dimA[3]<<" "<<dimA[4]<<") stride("<<strideA[0]<<" "<<strideA[1]<<" "<<strideA[2]<<" "<<strideA[3]<<" "<<strideA[4]<<")";\
                cudnnGetTensorNdDescriptor(y_tensor_, 5, &dataType, &nbDims, dimA, strideA);\
                LOG(ERROR) << "y tensor info: dataType(" << (int)dataType<<"),dim(" <<dimA[0]<<" "<<dimA[1]<<" "<<dimA[2]<<" "<<dimA[3]<<" "<<dimA[4]<<") stride("<<strideA[0]<<" "<<strideA[1]<<" "<<strideA[2]<<" "<<strideA[3]<<" "<<strideA[4]<<")";\
                cudnnGetConvolutionNdForwardOutputDim(conv_desc_, x_tensor_, w_desc_, 5, dimA);\
                LOG(ERROR) << "cudnn dest y tensor shape: " << dimA[0]<<" "<<dimA[1]<<" "<<dimA[2]<<" "<<dimA[3]<<" "<<dimA[4];\
                cudnnTensorFormat_t format=(cudnnTensorFormat_t)0;\
                cudnnGetFilterNdDescriptor(w_desc_, 3, &dataType, &format, &nbDims, filterDimA);\
                LOG(ERROR) << "w desc info: dataType(" << (int)dataType<<"),format(" << (int)format<<"),dim(" << filterDimA[0]<<" " << filterDimA[1]<<" " << filterDimA[2]<<" " << filterDimA[3]<<" " << filterDimA[4]<<")";\
                int pad_w=0, pad_h=0, u=0, v=0, dl_h=0, dl_w=0;\
                cudnnConvolutionMode_t mode=(cudnnConvolutionMode_t)0;\
                cudnnDataType_t computeType=(cudnnDataType_t)0;\
                cudnnGetConvolutionNdDescriptor(conv_desc_, 3, &nbDims, pad, strideA, dilationA, &mode, &computeType);\
                LOG(ERROR) << "conv desc info: pad(" << pad[0]<<" " << pad[1]<<" " << pad[2]<<"),dilationA(" << dilationA[0] <<" " << dilationA[1]<< " " <<dilationA[2]<<"),diation(" << dilationA[0]<<" " << dilationA[1]<<" "<<dilationA[2]<<"),mode(" << (int)mode<<"),computeType(" << (int)computeType<<")";\
                int group=0;\
                cudnnGetConvolutionGroupCount(conv_desc_, &group);\
                LOG(ERROR) << "group info: " << group;\
                cudnnMathType_t mathType = (cudnnMathType_t)0;\
                cudnnGetConvolutionMathType(conv_desc_, &mathType);\
                LOG(ERROR) << "mathType info: " << (int)mathType;\
                return ppl::common::RC_OTHER_ERROR;\
            }\
        }while(0)
ppl::common::RetCode ConvHmmaKernel::SetActivationParam(bool is_conv3d, fuse_param_t *param_ptr, bool is_act1, cudnnActivationDescriptor_t activation_desc){
    cudnnActivationMode_t activation_mode;
    double   coef = std::numeric_limits<double>::max();
    cudnnStatus_t status;
    int act_type = is_act1 ? param_ptr->has_activation : param_ptr->has_elt_activation;
    if(act_type == 0){
        activation_mode = CUDNN_ACTIVATION_IDENTITY;
    }
    else if(act_type == 1){
        activation_mode = CUDNN_ACTIVATION_RELU;
    }
    else if(act_type == 2){
        activation_mode = CUDNN_ACTIVATION_SIGMOID;
    }
    else if(act_type == 3){  //hardsigmoid
        activation_mode = MCDNN_ACTIVATION_HARD_SIGMOID;
    }
    else if(act_type == 4){  //swish
        activation_mode = CUDNN_ACTIVATION_SWISH;
    }
    else if(act_type == 5){  //hardswish
        activation_mode = MCDNN_ACTIVATION_HARD_SWISH;
    }
    else if(act_type == 6){  //Clip
        activation_mode = MCDNN_ACTIVATION_CLIP;
    }
    else if(act_type == 7){  //LeakyRelu
        activation_mode = MCDNN_ACTIVATION_Leaky_Relu;
    }
    else if(act_type == 8){  //PRelu
        activation_mode = MCDNN_ACTIVATION_PRelu;
    }
    else if(act_type == 9){  //Mish
        activation_mode = MCDNN_ACTIVATION_Mish;
    }
    else{
        return ppl::common::RC_UNSUPPORTED;
    }
    status = cudnnSetActivationDescriptor(activation_desc, activation_mode, CUDNN_NOT_PROPAGATE_NAN, coef);
    if(is_conv3d)
        CHECK_CUDNN_ND_STATUS(cudnnSetActivationDescriptor, status);
    else
        CHECK_CUDNN_STATUS(cudnnSetActivationDescriptor, status);

    if(act_type == 4){  //swish
        //printf("swish beta: %f\n", is_act1 ? param_ptr->act_beta : param_ptr->elt_act_beta);
        status = cudnnSetActivationDescriptorSwishBeta(activation_desc,
                                                       is_act1 ? param_ptr->act_beta : param_ptr->elt_act_beta);
        if(is_conv3d)
            CHECK_CUDNN_ND_STATUS(cudnnSetActivationDescriptorSwishBeta, status);
        else
            CHECK_CUDNN_STATUS(cudnnSetActivationDescriptorSwishBeta, status);
    }
    else if(act_type == 3 || act_type == 5){  //hardsigmoid  hardswish
        status = mcdnnSetActivationDescriptorHardSigmoidSwishAlphaBeta(activation_desc,
                                                                       is_act1 ? param_ptr->act_alpha : param_ptr->elt_act_alpha,
                                                                       is_act1 ? param_ptr->act_beta : param_ptr->elt_act_beta);
        if(is_conv3d)
            CHECK_CUDNN_ND_STATUS(mcdnnSetActivationDescriptorHardSigmoidSwishAlphaBeta, status);
        else
            CHECK_CUDNN_STATUS(mcdnnSetActivationDescriptorHardSigmoidSwishAlphaBeta, status);
        //printf("hardsigmoid/swish alpha: %f, beta: %f\n", is_act1 ? param_ptr->act_alpha : param_ptr->elt_act_alpha, is_act1 ? param_ptr->act_beta : param_ptr->elt_act_beta);
    }
    else if(act_type == 6){  //clip
        mcdnnActivationAttribute act_attribute;
        act_attribute.clip.min = is_act1 ? param_ptr->clip_min : param_ptr->elt_clip_min;
        act_attribute.clip.max = is_act1 ? param_ptr->clip_max : param_ptr->elt_clip_max;
        mcdnnSetActivationAttribute(activation_desc, act_attribute);
        if(is_conv3d)
            CHECK_CUDNN_ND_STATUS(mcdnnSetActivationAttribute, status);
        else
            CHECK_CUDNN_STATUS(mcdnnSetActivationAttribute, status);
    }
    else if(act_type == 7){  //leakyrelu
        mcdnnActivationAttribute act_attribute;
        act_attribute.leakyrelu.alpha = is_act1 ? param_ptr->leaky : param_ptr->elt_leaky;
        mcdnnSetActivationAttribute(activation_desc, act_attribute);
        if(is_conv3d)
            CHECK_CUDNN_ND_STATUS(mcdnnSetActivationAttribute, status);
        else
            CHECK_CUDNN_STATUS(mcdnnSetActivationAttribute, status);
        //printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ leakyrelu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~, is_act1(%d),  alpha(%f)\n", is_act1, act_attribute.leakyrelu.alpha);
    }
    else if(act_type == 8){  //prelu
        mcdnnActivationAttribute act_attribute;
        act_attribute.prelu.num = is_act1 ? param_ptr->has_prelu : param_ptr->has_elt_prelu;
        act_attribute.prelu.alpha = is_act1 ? (float*)param_ptr->prelu : (float*)param_ptr->elt_prelu;
        mcdnnSetActivationAttribute(activation_desc, act_attribute);
        if(is_conv3d)
            CHECK_CUDNN_ND_STATUS(mcdnnSetActivationAttribute, status);
        else
            CHECK_CUDNN_STATUS(mcdnnSetActivationAttribute, status);
        //printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ prelu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~, is_act1(%d)\n", is_act1);
    }

    return ppl::common::RC_SUCCESS;
}
#endif

ppl::common::RetCode ConvHmmaKernel::DoExecute(KernelExecContext* ctx) {
    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param;

    const TensorShape& shape_in0 = *ctx->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& shape_in1 = *ctx->GetInput<TensorImpl>(1)->GetShape();
    TensorShape shape_out = *ctx->GetOutput<TensorImpl>(0)->GetShape();
    auto output_channels = param_->extra_param.fuse_info.output_channels;
    if(output_channels.empty() == false){
        shape_out.SetDim(1,shape_in1.GetDim(0));
    }
    bool is_need_to_c4 = !gSupportNc1hwc0 && (param_->extra_param.is_graph_input_conv && shape_in1.GetDim(1) <= 4) && param_->param.group == 1 && shape_in1.GetDataFormat() == DATAFORMAT_NHWC4;

    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, *param_, temp_conv_param);
    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);

    struct algo_param_t algo_param;
    algo_param = param_->extra_param.algo_info;
#ifdef PPLNN_USE_DNN
//#if 1
    if(w_desc_ != nullptr && x_tensor_ != nullptr && b_tensor_ != nullptr && y_tensor_ != nullptr && conv_desc_ != nullptr){
        cudnnStatus_t status;
        //bool is_group_not_depthwise = param_->param.group > 1 && !(param_->param.group == shape_in0.GetDim(1) && param_->param.group == shape_in1.GetDim(0));
        bool is_group_not_depthwise = (shape_in0.GetDataFormat() == DATAFORMAT_NHWC);
        bool is_depthwise = param_->param.group == shape_in0.GetDim(1) && param_->param.group == shape_in1.GetDim(0);
        bool is_3d_conv = shape_in1.GetDimCount() == 5;
        if(is_3d_conv){
            int n, c, d, h, w;
            n = shape_in0.GetDim(0); c = shape_in0.GetDim(1); d = shape_in0.GetDim(2); h = shape_in0.GetDim(3); w = shape_in0.GetDim(4);
            if(is_group_not_depthwise){
                //
            } else if(is_need_to_c4){
                c = (c + 3) / 4 * 4;
            } else{
                c = (c + 7) / 8 * 8;
            }
            int dim_x[5] = {n, c, d, h, w};
            status = cudnnSetTensorNdDescriptorEx(x_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 5, dim_x);
            CHECK_CUDNN_ND_STATUS(cudnnSetTensorNdDescriptorEx, status);

            n = shape_in1.GetDim(0); c = shape_in1.GetDim(1); d = shape_in1.GetDim(2); h = shape_in1.GetDim(3); w = shape_in1.GetDim(4);
            if(param_->param.group > 1){
                if(is_group_not_depthwise){
                    //
                } else{
                    n = (n + 7) / 8 * 8;
                }
            } else if(is_need_to_c4){
                n = (n + 7) / 8 * 8;
                c = (c + 3) / 4 * 4;
            } else{
                n = (n + 7) / 8 * 8;
                c = (c + 7) / 8 * 8;
            }
            int dim_w[5] = {n, c, d, h, w};
            status = cudnnSetFilterNdDescriptor(w_desc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, 5, dim_w);
            CHECK_CUDNN_ND_STATUS(cudnnSetFilterNdDescriptor, status);
            if(output_channels.empty() == false){
                mcdnnSetFilterBatchCount(w_desc_, output_channels.size(), output_channels.data());
            }
            n = shape_out.GetDim(0); c = shape_out.GetDim(1); d = shape_out.GetDim(2); h = shape_out.GetDim(3); w = shape_out.GetDim(4);
            if(is_group_not_depthwise){
                //
            } else{
                c = (c + 7) / 8 * 8;
            }
            int dim_y[5] = {n, c, d, h, w};
            status = cudnnSetTensorNdDescriptorEx(y_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 5, dim_y);
            CHECK_CUDNN_ND_STATUS(cudnnSetTensorNdDescriptorEx, status);

            int pads[3] = {param_->param.pads[0], param_->param.pads[1], param_->param.pads[2]};
            int strides[3] = {param_->param.strides[0], param_->param.strides[1], param_->param.strides[2]};
            int dilations[3] = {param_->param.dilations[0], param_->param.dilations[1], param_->param.dilations[2]};
            status = cudnnSetConvolutionNdDescriptor(conv_desc_,
                                                    3,
                                                    pads,
                                                    strides,
                                                    dilations,
                                                    CUDNN_CROSS_CORRELATION,  // ref onnxruntime
                                                    CUDNN_DATA_FLOAT);  // // ref onnxruntime
            CHECK_CUDNN_ND_STATUS(cudnnGetConvolutionNdDescriptor, status);
        } else{
            int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
            n = shape_in0.GetDim(0); c = shape_in0.GetDim(1); h = shape_in0.GetDim(2); w = shape_in0.GetDimCount() > 3 ? shape_in0.GetDim(3) : 1;
            if(is_group_not_depthwise){
                status = cudnnSetTensor4dDescriptor(x_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, c, h, w);
            }else if(is_need_to_c4){
                status = cudnnSetTensor4dDescriptor(x_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, (c + 3) / 4 * 4, h, w);
            }else{
                status = cudnnSetTensor4dDescriptor(x_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, (c + 7) / 8 * 8, h, w);
            }
            CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor, status);

            n = shape_in1.GetDim(0); c = shape_in1.GetDim(1); h = shape_in1.GetDim(2); w = shape_in1.GetDimCount() > 3 ? shape_in1.GetDim(3) : 1;
            if(param_->param.group > 1){
                if(is_group_not_depthwise)
                    status = cudnnSetFilter4dDescriptor(w_desc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, n, c, h, w);
                else
                    status = cudnnSetFilter4dDescriptor(w_desc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, (n + 7) / 8 * 8, c, h, w);
            } else if(is_need_to_c4){
                status = cudnnSetFilter4dDescriptor(w_desc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, (n + 7) / 8 * 8, (c + 3) / 4 * 4, h, w);
            } else{
                status = cudnnSetFilter4dDescriptor(w_desc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, (n + 7) / 8 * 8, (c + 7) / 8 * 8, h, w);
            }
            CHECK_CUDNN_STATUS(cudnnSetFilter4dDescriptor, status);
            if(output_channels.empty() == false){
                mcdnnSetFilterBatchCount(w_desc_, output_channels.size(), output_channels.data());
            }
            n = shape_out.GetDim(0); c = shape_out.GetDim(1); h = shape_out.GetDim(2); w = shape_out.GetDimCount() > 3 ? shape_out.GetDim(3) : 1;
            if(is_group_not_depthwise)
                status =  cudnnSetTensor4dDescriptor(y_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, c, h, w);
            else
                status =  cudnnSetTensor4dDescriptor(y_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, (c + 7) / 8 * 8, h, w);
            CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor, status);
        #if 1
            int pad_h, pad_w, u, v, dialation_h, dialation_w;
            pad_h = temp_conv_param.pad_height;
            pad_w = temp_conv_param.pad_width;
            status = cudnnSetConvolution2dDescriptor(conv_desc_, pad_h, pad_w,
                                                                param_->param.strides[0], shape_in0.GetDimCount() > 3 ? param_->param.strides[1] : 1,
                                                                param_->param.dilations[0], shape_in0.GetDimCount() > 3 ? param_->param.dilations[1] : 1,
                                                                CUDNN_CROSS_CORRELATION, /*CUDNN_CONVOLUTION,*/ /*CUDNN_CROSS_CORRELATION,*/  // ref onnxruntime
                                                                CUDNN_DATA_FLOAT/*CUDNN_DATA_HALF*//*CUDNN_DATA_FLOAT*/);  // // ref onnxruntime
            CHECK_CUDNN_STATUS(cudnnSetConvolution2dDescriptor, status);
        #else
            int pads[2] = {param_->param.pads[1], param_->param.pads[0]};
            int strides[2] = {param_->param.strides[1], param_->param.strides[0]};
            int dilations[2] = {param_->param.dilations[1], param_->param.dilations[0]};
            status = cudnnSetConvolutionNdDescriptor(conv_desc_,
                                                    2,
                                                    pads,
                                                    strides,
                                                    dilations,
                                                    CUDNN_CROSS_CORRELATION,  // ref onnxruntime
                                                    CUDNN_DATA_FLOAT);  // // ref onnxruntime
            CHECK_CUDNN_STATUS(cudnnGetConvolutionNdDescriptor, status);
        #endif
        }
        if(param_->param.group > 1 && !is_group_not_depthwise)
            status = cudnnSetConvolutionGroupCount(conv_desc_, ((int)(param_->param.group) + 7) / 8 * 8);
        else
            status = cudnnSetConvolutionGroupCount(conv_desc_, (int)(param_->param.group));
        if(is_3d_conv)
            CHECK_CUDNN_ND_STATUS(cudnnSetConvolutionGroupCount, status);
        else
            CHECK_CUDNN_STATUS(cudnnSetConvolutionGroupCount, status);

        status = cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH);
        if(is_3d_conv)
            CHECK_CUDNN_ND_STATUS(cudnnSetConvolutionMathType, status);
        else
            CHECK_CUDNN_STATUS(cudnnSetConvolutionMathType, status);

        size_t work_space_bytes = 0;
        cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        status = cudnnGetConvolutionForwardWorkspaceSize(GetCudnnHandle(), x_tensor_, w_desc_, conv_desc_, y_tensor_, algo, &work_space_bytes);
        if(is_3d_conv)
            CHECK_CUDNN_ND_STATUS(cudnnGetConvolutionForwardWorkspaceSize, status);
        else
            CHECK_CUDNN_STATUS(cudnnGetConvolutionForwardWorkspaceSize, status);
        if(work_space_bytes != conv_work_len_){
            if(work_space_bytes != 0)
            {
                TensorShape conv_work_space_shape;
                conv_work_space_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                conv_work_space_shape.SetDataType(DATATYPE_UINT8);
                conv_work_space_shape.SetDimCount(1);
                conv_work_space_shape.SetDim(0, work_space_bytes);
                if(conv_work_buffer_ != nullptr){
                    cudaFree(conv_work_buffer_);
                    conv_work_buffer_ = nullptr;
                }
                cudaMalloc(&conv_work_buffer_, conv_work_space_shape.CalcBytesIncludingPadding());
                if(conv_work_buffer_ == nullptr)
                    return RC_OUT_OF_MEMORY;
            }
            conv_work_len_ = work_space_bytes;
        }
        if(is_3d_conv){
            int n = 1; int c = shape_in1.GetDim(0); int d = 1; int h = 1; int w = 1;
            if(is_group_not_depthwise){
                //
            }else{
                c = (c + 7) / 8 * 8;
            }
            int dim_b[5] = {n, c, d, h, w};
            status = cudnnSetTensorNdDescriptorEx(b_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 5, dim_b);
            CHECK_CUDNN_ND_STATUS(cudnnSetTensorNdDescriptorEx, status);
        } else{
            int n = 1; int c = shape_in1.GetDim(0); int h = 1; int w = 1;
            if(is_group_not_depthwise)
                status = cudnnSetTensor4dDescriptor(b_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 1, c, 1, 1);
            else
                status = cudnnSetTensor4dDescriptor(b_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 1, (c + 7) / 8 * 8, 1, 1);
            CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor, status);
        }

        float alpha = 1.0f;
        float beta = 0.0f;
        void *input_data_ptr = ctx->GetInput<TensorImpl>(0)->GetBufferPtr();
        void *weight_data_ptr = ctx->GetInput<TensorImpl>(1)->GetBufferPtr();
        void *output_data_ptr = ctx->GetOutput<TensorImpl>(0)->GetBufferPtr();

        if(false){//if(false){//if(is_3d_conv){//if(temp_fuse_param.has_activation == 0 && temp_fuse_param.has_prelu == 0){
            status = cudnnConvolutionForward(GetCudnnHandle(),
                                             &alpha,
                                             x_tensor_,
                                             input_data_ptr,
                                             w_desc_,
                                             weight_data_ptr,
                                             conv_desc_,
                                             algo,
                                             conv_work_buffer_,
                                             work_space_bytes,
                                             &beta,
                                             y_tensor_,
                                             output_data_ptr);
            if(is_3d_conv)
                CHECK_CUDNN_ND_STATUS(cudnnConvolutionForward, status);
            else
                CHECK_CUDNN_STATUS(cudnnConvolutionForward, status);
            if(ctx->GetInputCount() > 2){
            #if 0
                void *bias_data_ptr = ctx->GetInput<TensorImpl>(2)->GetBufferPtr();
                status = cudnnAddTensor(GetCudnnHandle(),
                                        &alpha,
                                        b_tensor_,
                                        bias_data_ptr,
                                        &alpha,
                                        y_tensor_,
                                        output_data_ptr);
                CHECK_CUDNN_STATUS(cudnnAddTensor, status);
            #else
                TensorShape shape_bias = *ctx->GetInput<TensorImpl>(2)->GetShape();
                std::vector<int64_t> dim_bias;
                dim_bias.push_back(1);
                dim_bias.push_back(shape_out.GetDim(1));
                dim_bias.push_back(1);
                if(shape_out.GetDimCount() > 3)
                    dim_bias.push_back(1);
                if(shape_out.GetDimCount() > 4)
                    dim_bias.push_back(1);
                shape_bias.Reshape(dim_bias);
                shape_bias.SetDataFormat(shape_out.GetDataFormat());
                ppl::common::RetCode status1 = PPLCUDAArithMeticAddForwardImp(GetStream(),
                                                                              &shape_out, ctx->GetOutput<TensorImpl>(0)->GetBufferPtr(),
                                                                              &shape_bias, ctx->GetInput<TensorImpl>(2)->GetBufferPtr(),
                                                                              &shape_out, ctx->GetOutput<TensorImpl>(0)->GetBufferPtr());
                return status1;
            #endif
            }
            return ppl::common::RC_SUCCESS;
        //}else if(temp_fuse_param.has_activation == 1 || temp_fuse_param.has_activation == 2){
        }else if(false){//is_depthwise){
            if(activation_desc_ != nullptr){
                cudnnActivationMode_t activation_mode;
                double   coef;
                if(temp_fuse_param.has_activation == 0){
                    activation_mode = CUDNN_ACTIVATION_IDENTITY;
                }
                else if(temp_fuse_param.has_activation == 1){
                    activation_mode = CUDNN_ACTIVATION_RELU;
                    coef = std::numeric_limits<double>::max();
                }
                else if(temp_fuse_param.has_activation == 2){
                    activation_mode = CUDNN_ACTIVATION_SIGMOID;
                    coef = std::numeric_limits<double>::max();
                }
                else{
                    LOG(ERROR) << "Unsupport activation type: " << temp_fuse_param.has_activation;
                    return ppl::common::RC_UNSUPPORTED;
                }
                status = cudnnSetActivationDescriptor(activation_desc_, activation_mode, CUDNN_NOT_PROPAGATE_NAN, coef);
                CHECK_CUDNN_STATUS(cudnnSetActivationDescriptor, status);
                void *bias_data_ptr;
                if(ctx->GetInputCount() > 2){
                    bias_data_ptr = ctx->GetInput<TensorImpl>(2)->GetBufferPtr();
                }else{
                    TensorShape conv_bias_shape;
                    conv_bias_shape.SetDataFormat(DATAFORMAT_NHWC);
                    conv_bias_shape.SetDataType(DATATYPE_FLOAT16);
                    conv_bias_shape.SetDimCount(1);
                    if(is_group_not_depthwise)
                        conv_bias_shape.SetDim(0, shape_in1.GetDim(0));
                    else
                        conv_bias_shape.SetDim(0, (shape_in1.GetDim(0) + 7) / 8 * 8);
                    if(conv_bias_len_ < conv_bias_shape.CalcBytesIncludingPadding()){
                        conv_bias_len_ = conv_bias_shape.CalcBytesIncludingPadding();
                        if(conv_bias_buffer_ != nullptr){
                            cudaFree(conv_bias_buffer_);
                            conv_bias_buffer_ = nullptr;
                        }
                        cudaMalloc(&conv_bias_buffer_, conv_bias_shape.CalcBytesIncludingPadding());
                        if(conv_bias_buffer_ == nullptr)
                            return RC_OUT_OF_MEMORY;
                        cudaMemsetAsync(conv_bias_buffer_, 0, conv_bias_len_, GetStream());
                    }
                    bias_data_ptr = conv_bias_buffer_;
                }
                status = cudnnConvolutionBiasActivationForward(GetCudnnHandle(),
                                                               &alpha,
                                                               x_tensor_,
                                                               input_data_ptr,
                                                               w_desc_,
                                                               weight_data_ptr,
                                                               conv_desc_,
                                                               algo,
                                                               conv_work_buffer_,
                                                               work_space_bytes,
                                                               &beta,
                                                               y_tensor_,
                                                               output_data_ptr,
                                                               b_tensor_,
                                                               bias_data_ptr,
                                                               activation_desc_,
                                                               y_tensor_,
                                                               output_data_ptr);
                CHECK_CUDNN_STATUS(cudnnConvolutionBiasActivationForward, status);
                return ppl::common::RC_SUCCESS;
            }
        }else if(true){
            void *input_data_ptr = ctx->GetInput<TensorImpl>(0)->GetBufferPtr();
            void *weight_data_ptr = ctx->GetInput<TensorImpl>(1)->GetBufferPtr();
            void *bias_data_ptr;
            if((param_->extra_param.fuse_info.input_inds.size() > 0 && param_->extra_param.fuse_info.input_inds[0] > 2)
               || (param_->extra_param.fuse_info.input_inds.size() == 0 && ctx->GetInputCount() > 2)){
                bias_data_ptr = ctx->GetInput<TensorImpl>(2)->GetBufferPtr();
            }else{
                TensorShape conv_bias_shape;
                conv_bias_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                conv_bias_shape.SetDataType(DATATYPE_FLOAT16);
                conv_bias_shape.SetDimCount(1);
                if(is_group_not_depthwise){
                    conv_bias_shape.SetDim(0, shape_in1.GetDim(0));
                }else{
                    conv_bias_shape.SetDim(0, (shape_in1.GetDim(0) + 7) / 8 * 8);
                }
                if(conv_bias_len_ < conv_bias_shape.CalcBytesIncludingPadding()){
                    conv_bias_len_ = conv_bias_shape.CalcBytesIncludingPadding();
                    if(conv_bias_buffer_ != nullptr){
                        cudaFree(conv_bias_buffer_);
                        conv_bias_buffer_ = nullptr;
                    }
                    cudaMalloc(&conv_bias_buffer_, conv_bias_shape.CalcBytesIncludingPadding());
                    if(conv_bias_buffer_ == nullptr)
                        return RC_OUT_OF_MEMORY;
                    cudaMemsetAsync(conv_bias_buffer_, 0, conv_bias_len_, GetStream());
                }
                bias_data_ptr = conv_bias_buffer_;
            }
            void *ele_in1_data_ptr = nullptr;
            if(temp_fuse_param.has_elt){
                uint32_t ele_input_index = param_->extra_param.fuse_info.input_inds[temp_fuse_param.elt_index];
                auto ele_input = ctx->GetInput<TensorImpl>(ele_input_index);
                ele_in1_data_ptr = ctx->GetInput<TensorImpl>(ele_input_index)->GetBufferPtr();
                const TensorShape& shape_ele_in = *ele_input->GetShape();
                if(is_3d_conv){
                    int n, c, d, h, w, n_stride, c_stride, d_stride, h_stride, w_stride;
                    n = shape_ele_in.GetDim(0); c = shape_ele_in.GetDim(1); d = shape_ele_in.GetDim(2); h = shape_ele_in.GetDim(3); w = shape_ele_in.GetDim(4);
                    if(is_group_not_depthwise){
                        //
                    } else{
                        c = (c + 7) / 8 * 8;
                    }
                    int dim_y[5] = {n, c, d, h, w};
                    status = cudnnSetTensorNdDescriptorEx(ele_input_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 5, dim_y);
                    CHECK_CUDNN_ND_STATUS(cudnnConvolutionForward, status);
                }else{
                    int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
                    n = shape_ele_in.GetDim(0); c = shape_ele_in.GetDim(1); h = shape_ele_in.GetDim(2); w = shape_ele_in.GetDimCount() > 3 ? shape_ele_in.GetDim(3) : 1;
                    if(is_group_not_depthwise)
                        status = cudnnSetTensor4dDescriptor(ele_input_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, c, h, w);
                    else
                        status = cudnnSetTensor4dDescriptor(ele_input_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, (c + 7) / 8 * 8, h, w);
                    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor, status);
                }
            }
            void *output_data_ptr = ctx->GetOutput<TensorImpl>(0)->GetBufferPtr();
            if(activation_desc_ != nullptr){
                SetActivationParam(is_3d_conv, &temp_fuse_param, true, activation_desc_);
            }
            if(activation_desc2_ != nullptr){
                SetActivationParam(is_3d_conv, &temp_fuse_param, false, activation_desc2_);
            }
            int ele_mode = MCDNN_CONVOLUTION_QUANT_NONE;
            if(temp_fuse_param.has_elt == true && param_->extra_param.fuse_info.types[temp_fuse_param.elt_index] == "Add"){
                ele_mode = MCDNN_CONVOLUTION_QUANT_ADD;
            }else if(temp_fuse_param.has_elt == true && param_->extra_param.fuse_info.types[temp_fuse_param.elt_index] == "Mul"){
                ele_mode = MCDNN_CONVOLUTION_QUANT_MUL;
            }
            else{
                //return ppl::common::RC_UNSUPPORTED;
            }
            status = mcdnnConvBiasAct1EleAct2FusedForwardFp16(GetCudnnHandle(),
                                                              x_tensor_,
                                                              input_data_ptr,
                                                              w_desc_,
                                                              weight_data_ptr,
                                                              conv_desc_,
                                                              algo,
                                                              conv_work_buffer_,
                                                              work_space_bytes,
                                                              b_tensor_,
                                                              bias_data_ptr,
                                                              activation_desc_,
                                                              ele_mode,
                                                              ele_input_tensor_,
                                                              ele_in1_data_ptr,
                                                              activation_desc2_,
                                                              y_tensor_,
                                                              output_data_ptr);
            if(is_3d_conv)
                CHECK_CUDNN_ND_STATUS(mcdnnConvBiasAct1EleAct2FusedForwardFp16, status);
            else
                CHECK_CUDNN_STATUS(mcdnnConvBiasAct1EleAct2FusedForwardFp16, status);
            return ppl::common::RC_SUCCESS;
        }
        return ppl::common::RC_OTHER_ERROR;
    }else{
        return ppl::common::RC_OTHER_ERROR;
    }
#else  //PPLNN_USE_DNN
    uint64_t size = PPLCUDAConvolutionGetRuntimeBufSize(shape_in0.GetDataType(), temp_conv_param, algo_param.splitk,
                                                        algo_param.splitf, ((uint64_t)8) * 1024 * 1024 * 1024);

    BufferDesc tmp_buffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << size << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;
    auto stream = GetStream();

    // convert filter only if the filter tensor is an output of another kernel
    BufferDesc weight_buffer;
    auto newshape = shape_in1;
    if (!param_->extra_param.is_initializer_weight) {
        auto align_size = 8;
        newshape.SetPadding1(0, (newshape.GetDim(0) + align_size - 1) / align_size * align_size - newshape.GetDim(0));

        auto status = GetCudaDevice()->Realloc(newshape, &weight_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        conv_param_t temp_conv_param;
        ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, *param_, temp_conv_param);
        PPLCUDAConvolutionCvtFlt(stream, weight_buffer.addr, ctx->GetInput<TensorImpl>(1)->GetBufferPtr(), shape_in0.GetDataType(), temp_conv_param);
    }
    ppl::common::Destructor __tmp_buffer_guard__([this, &weight_buffer]() -> void {
        GetCudaDevice()->Free(&weight_buffer);
    });

#ifdef PPLNN_ENABLE_CUDA_JIT
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);
    PPLCUDAConvolutionForwardJitImp(
        GetCudaDevice()->GetDeviceProp(), stream, module->GetKernelFunc(), shape_in0.GetDataType(), (int4*)ctx->GetInput<TensorImpl>(0)->GetBufferPtr(),
        param_->extra_param.is_initializer_weight ? (int4*)ctx->GetInput<TensorImpl>(1)->GetBufferPtr() : (int4*)weight_buffer.addr,
        (int4*)ctx->GetOutput<TensorImpl>(0)->GetBufferPtr(),
        param_->extra_param.bias_term ? (int4*)ctx->GetInput<TensorImpl>(2)->GetBufferPtr() : nullptr,
        (int4*)tmp_buffer, algo_param, temp_conv_param, temp_fuse_param);
#else //PPLNN_ENABLE_CUDA_JIT
    PPLCUDAConvolutionForwardImp(
        GetCudaDevice()->GetDeviceProp(), stream, shape_in0.GetDataType(), (int4*)ctx->GetInput<TensorImpl>(0)->GetBufferPtr(),
        param_->extra_param.is_initializer_weight ? (int4*)ctx->GetInput<TensorImpl>(1)->GetBufferPtr() : (int4*)weight_buffer.addr,
        (int4*)ctx->GetOutput<TensorImpl>(0)->GetBufferPtr(),
        param_->extra_param.bias_term ? (int4*)ctx->GetInput<TensorImpl>(2)->GetBufferPtr() : nullptr,
        (int4*)tmp_buffer, algo_param, temp_conv_param, temp_fuse_param);
#endif //PPLNN_ENABLE_CUDA_JIT
    LOG(DEBUG) << "Excute HMMA conv with kernel id:" << param_->extra_param.algo_info.kid
               << " and temp buffer size: " << size;
    return ppl::common::RC_SUCCESS;
#endif //PPLNN_USE_DNN
}
}}} // namespace ppl::nn::cuda
