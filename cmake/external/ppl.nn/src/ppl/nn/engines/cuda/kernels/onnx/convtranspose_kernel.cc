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

#include "ppl/nn/engines/cuda/kernels/onnx/convtranspose_kernel.h"
#include "ppl/nn/engines/cuda/module/cuda_module.h"
#include "ppl/common/destructor.h"
#include "cudakernel/nn/convtranspose.h"
#include "cudakernel/arithmetic/arithmetic.h"

namespace ppl { namespace nn { namespace cuda {

#define ALGO_MAX_TIME (3.0e+10)

#define ALLOC_BUFFERF_FOR_ALGO_SELECT(___buffer_name___, ___size___, ___ret___)                  \
    BufferDesc ___buffer_name___;  \
    status =  GetCudaDevice()->ReallocWithRandomValue(___size___, &___buffer_name___);             \
    if (status != RC_SUCCESS) {                                                                  \
        LOG(DEBUG) << "alloc " #___buffer_name___ " tensor failed";                              \
        return ppl::common::RC_UNSUPPORTED;                                                                        \
    }                                                                                            \
    ppl::common::Destructor __##___buffer_name___##_guard__([this, &___buffer_name___]() -> void { \
        GetCudaDevice()->Free(&___buffer_name___);                                                \
    });

uint64_t ConvTransposeKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto x = ctx.GetInput<TensorImpl>(0);
    auto y = ctx.GetOutput<TensorImpl>(0);

    ConvTransposeKernelParam param_kernel_;
    param_kernel_.auto_pad = param_->param.auto_pad;
    param_kernel_.group = param_->param.group;
    param_kernel_.dilations = param_->param.dilations;
    param_kernel_.kernel_shape = param_->param.kernel_shape;
    param_kernel_.pads = param_->param.pads;
    param_kernel_.strides = param_->param.strides;
    param_kernel_.output_padding = param_->param.output_padding;
    param_kernel_.output_shape = param_->param.output_shape;

    return PPLConvTransposeGetBufSizeCuda(x->GetShape(), y->GetShape(), &param_kernel_);
}

ppl::common::RetCode ConvTransposeKernel::DoExecute_PPL(KernelExecContext* ctx) {
    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_bytes = CalcTmpBufferSize(*ctx);
    auto status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_bytes << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;

    TensorImpl* X = ctx->GetInput<TensorImpl>(0);
    TensorImpl* W = ctx->GetInput<TensorImpl>(1);
    TensorImpl* B = nullptr;
    TensorImpl* Y = ctx->GetOutput<TensorImpl>(0);
    const float* b_data = nullptr;
    ConvTransposeKernelParam param_kernel_;
    param_kernel_.auto_pad = param_->param.auto_pad;
    param_kernel_.group = param_->param.group;
    param_kernel_.dilations = param_->param.dilations;
    param_kernel_.kernel_shape = param_->param.kernel_shape;
    param_kernel_.pads = param_->param.pads;
    param_kernel_.strides = param_->param.strides;
    param_kernel_.output_padding = param_->param.output_padding;
    param_kernel_.output_shape = param_->param.output_shape;

    // convert filter only if the filter tensor is an output of another kernel
    BufferDesc weight_buffer;
    auto newshape = *W->GetShape();
    if (!param_->extra_param.is_initializer_weight) {
        auto align_size = 8;
        int stride_h = param_->param.strides[0];
        int stride_w = param_->param.strides[1];
        int kernel_u = (newshape.GetDim(2) + stride_h - 1) / stride_h;
        int kernel_v = (newshape.GetDim(3) + stride_w - 1) / stride_w;
        int pattern_num = stride_h * stride_w;
        newshape.SetDim(0, (newshape.GetDim(0) + align_size - 1) / align_size * align_size);
        newshape.SetPadding1(1, (newshape.GetDim(1) + align_size - 1) / align_size * align_size - newshape.GetDim(1));
        newshape.SetDim(2, pattern_num);
        newshape.SetDim(3, kernel_u * kernel_v);
        newshape.SetPadding1(0, (newshape.GetDim(0) + align_size - 1) / align_size * align_size - newshape.GetDim(0));

        auto status = GetCudaDevice()->Realloc(newshape, &weight_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        auto size = PPLConvTransposeGetFilterBufSizeCudaFp16(W->GetShape());
        ALLOC_BUFFERF_FOR_ALGO_SELECT(filter_temp_buffer, size, ALGO_MAX_TIME)
        ALLOC_BUFFERF_FOR_ALGO_SELECT(filter_input_buffer, W->GetShape()->CalcBytesIncludingPadding(), ALGO_MAX_TIME)
        auto filter_shape = *W->GetShape(); filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        GetCudaDevice()->GetDataConverter()->Convert(&filter_input_buffer, filter_shape, W->GetBufferDesc(), *W->GetShape());
        PPLCUDAConvTransposeCvt(GetCudaDevice()->GetDeviceProp(), stream, filter_input_buffer.addr, filter_temp_buffer.addr,
                                weight_buffer.addr, W->GetShape(), &param_kernel_);
    }
    ppl::common::Destructor __tmp_buffer_guard__([this, &weight_buffer]() -> void {
        GetCudaDevice()->Free(&weight_buffer);
    });

    if (ctx->GetInputCount() >= 3) {
        B = ctx->GetInput<TensorImpl>(2);
        b_data = B->GetBufferPtr<float>();
    }
    fuse_param_t temp_fuse_param;
    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);
    CUfunction module_func = nullptr;
#ifdef PPLNN_ENABLE_CUDA_JIT
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);
    module_func = module->GetKernelFunc();
#endif

    status = PPLCUDAConvTransposeForward(GetCudaDevice()->GetDeviceProp(), GetStream(), module_func, X->GetShape(), X->GetBufferPtr(),
                                         param_->extra_param.is_initializer_weight ? (int4*)ctx->GetInput<TensorImpl>(1)->GetBufferPtr() : (int4*)weight_buffer.addr,
                                         b_data, Y->GetShape(), Y->GetBufferPtr(), &param_kernel_, param_->extra_param.algo_info,
                                         temp_fuse_param, tmp_buffer);

    return status;
}

#ifdef PPLNN_USE_DNN
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
                    cudnnTensorFormat_t format=(cudnnTensorFormat_t)0;\
                    cudnnGetFilter4dDescriptor(w_desc_, &dataType, &format, &n, &c, &h, &w);\
                    LOG(ERROR) << "w desc info: dataType(" << (int)dataType<<"),format(" << (int)format<<"),dim(" << n<<" " << c<<" " << h<<" " << w<<")";\
                    int pad_w=0, pad_h=0, u=0, v=0, dl_h=0, dl_w=0;\
                    cudnnConvolutionMode_t mode=(cudnnConvolutionMode_t)0;\
                    cudnnDataType_t computeType=(cudnnDataType_t)0;\
                    cudnnGetConvolution2dDescriptor(conv_desc_, &pad_h, &pad_w, &u, &v, &dl_h, &dl_w, &mode, &computeType);\
                    LOG(ERROR) << "conv desc info: pad(" << pad_h<<" " << pad_w<<"),uv(" << u<<" " << v<<"),diation(" << dl_h<<" " << dl_w<<"),mode(" << (int)mode<<"),computeType(" << (int)computeType<<")";\
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
ppl::common::RetCode ConvTransposeKernel::DoExecute_MCDNN(KernelExecContext* ctx) {
    TensorImpl* X = ctx->GetInput<TensorImpl>(0);
    TensorImpl* W = ctx->GetInput<TensorImpl>(1);
    TensorImpl* B = nullptr;
    TensorImpl* Y = ctx->GetOutput<TensorImpl>(0);
    const float* b_data = nullptr;
    ConvTransposeKernelParam param_kernel_;
    param_kernel_.auto_pad = param_->param.auto_pad;
    param_kernel_.group = param_->param.group;
    param_kernel_.dilations = param_->param.dilations;
    param_kernel_.kernel_shape = param_->param.kernel_shape;
    param_kernel_.pads = param_->param.pads;
    param_kernel_.strides = param_->param.strides;
    param_kernel_.output_padding = param_->param.output_padding;
    param_kernel_.output_shape = param_->param.output_shape;

    const TensorShape& shape_in0 = *ctx->GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& shape_in1 = *ctx->GetInput<TensorImpl>(1)->GetShape();
    const TensorShape& shape_out = *ctx->GetOutput<TensorImpl>(0)->GetShape();

    if(w_desc_ != nullptr && x_tensor_ != nullptr && b_tensor_ != nullptr && y_tensor_ != nullptr && conv_desc_ != nullptr){
        cudnnStatus_t status;
        bool is_depthwise = param_->param.group  > 1 && param_->param.group == shape_in1.GetDim(0); 
        bool is_group_not_depthwise = param_->param.group > 1 && !is_depthwise;
        bool is_3d_conv = shape_in1.GetDimCount() == 5;
        bool is_fp32 = shape_in0.GetDataType() == DATATYPE_FLOAT32;
        if(is_3d_conv){
            int n, c, d, h, w;
            n = shape_in0.GetDim(0); c = shape_in0.GetDim(1); d = shape_in0.GetDim(2); h = shape_in0.GetDim(3); w = shape_in0.GetDim(4);
            if(shape_in0.GetDataType() == DATATYPE_FLOAT16){
                if(param_->param.group == 1 || is_depthwise){
                    c = (c + 7) / 8 * 8;
                }
            }
            int dim_x[5] = {n, c, d, h, w};
            status = cudnnSetTensorNdDescriptorEx(x_tensor_, is_fp32 ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC, is_fp32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF, 5, dim_x);
            CHECK_CUDNN_ND_STATUS(cudnnSetTensorNdDescriptorEx, status);

            n = shape_in1.GetDim(0); c = shape_in1.GetDim(1); d = shape_in1.GetDim(2); h = shape_in1.GetDim(3); w = shape_in1.GetDim(4);
            if(shape_in0.GetDataType() == DATATYPE_FLOAT16){
                if(param_->param.group == 1){
                    n = (n + 7) / 8 * 8;
                    c = (c + 7) / 8 * 8;
                }else if(is_depthwise){
                    n = (n + 7) / 8 * 8;
                }
            }
            int dim_w[5] = {n, c, d, h, w};
            status = cudnnSetFilterNdDescriptor(w_desc_, is_fp32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF, is_fp32 ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC, 5, dim_w);
            CHECK_CUDNN_ND_STATUS(cudnnSetFilterNdDescriptor, status);

            n = shape_out.GetDim(0); c = shape_out.GetDim(1); d = shape_out.GetDim(2); h = shape_out.GetDim(3); w = shape_out.GetDim(4);
            if(shape_in0.GetDataType() == DATATYPE_FLOAT16){
                if(param_->param.group == 1 || is_depthwise){
                    c = (c + 7) / 8 * 8;
                }
            }
            int dim_y[5] = {n, c, d, h, w};
            status = cudnnSetTensorNdDescriptorEx(y_tensor_, is_fp32 ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC, is_fp32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF, 5, dim_y);
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
        }else{
            int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
            n = shape_in0.GetDim(0); c = shape_in0.GetDim(1); h = shape_in0.GetDim(2); w = shape_in0.GetDimCount() > 3 ? shape_in0.GetDim(3) : 1;
            if(shape_in0.GetDataType() == DATATYPE_FLOAT16){
                if(param_->param.group == 1 || is_depthwise)
                    status = cudnnSetTensor4dDescriptor(x_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, (c + 7) / 8 * 8, h, w);
                else
                    status = cudnnSetTensor4dDescriptor(x_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, c, h, w);
            }
            else
                status = cudnnSetTensor4dDescriptor(x_tensor_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
            CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor, status);

            n = shape_in1.GetDim(0); c = shape_in1.GetDim(1); h = shape_in1.GetDim(2); w = shape_in1.GetDimCount() > 3 ? shape_in1.GetDim(3) : 1;
            if(shape_in0.GetDataType() == DATATYPE_FLOAT16){
                if(is_group_not_depthwise)
                    status = cudnnSetFilter4dDescriptor(w_desc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, n, c, h, w);
                else if(is_depthwise)
                    status = cudnnSetFilter4dDescriptor(w_desc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, (n + 7) / 8 * 8, c, h, w);
                else
                    status = cudnnSetFilter4dDescriptor(w_desc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, (n + 7) / 8 * 8, (c + 7) / 8 * 8, h, w);
            }else{
                status = cudnnSetFilter4dDescriptor(w_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w);
            }
            CHECK_CUDNN_STATUS(cudnnSetFilter4dDescriptor, status);

            n = shape_out.GetDim(0); c = shape_out.GetDim(1); h = shape_out.GetDim(2); w = shape_out.GetDimCount() > 3 ? shape_out.GetDim(3) : 1;
            if(shape_in0.GetDataType() == DATATYPE_FLOAT16){
                if(param_->param.group == 1 || is_depthwise)
                    status =  cudnnSetTensor4dDescriptor(y_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, (c + 7) / 8 * 8, h, w);
                else
                    status =  cudnnSetTensor4dDescriptor(y_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, c, h, w);
            }
            else
                status =  cudnnSetTensor4dDescriptor(y_tensor_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
            CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor, status);

            status = cudnnSetConvolution2dDescriptor(conv_desc_, param_->param.pads[0], param_->param.pads[1],
                                                    param_->param.strides[0], shape_in0.GetDimCount() > 3 ? param_->param.strides[1] : 1,
                                                    param_->param.dilations[0], shape_in0.GetDimCount() > 3 ? param_->param.dilations[1] : 1,
                                                    CUDNN_CROSS_CORRELATION, /*CUDNN_CONVOLUTION,*/ /*CUDNN_CROSS_CORRELATION,*/  // ref onnxruntime
                                                    CUDNN_DATA_FLOAT/*CUDNN_DATA_HALF*//*CUDNN_DATA_FLOAT*/);  // // ref onnxruntime
            CHECK_CUDNN_STATUS(cudnnSetConvolution2dDescriptor, status);
        }
        if(param_->param.group > 1){
            if(shape_in0.GetDataType() == DATATYPE_FLOAT16 && is_depthwise)
                status = cudnnSetConvolutionGroupCount(conv_desc_, ((int)(param_->param.group) + 7) / 8 * 8);
            else
                status = cudnnSetConvolutionGroupCount(conv_desc_, (int)(param_->param.group));
        }
        else
            status = cudnnSetConvolutionGroupCount(conv_desc_, (int)(param_->param.group));
        if(is_3d_conv)
            CHECK_CUDNN_ND_STATUS(cudnnSetConvolutionGroupCount, status);
        else
            CHECK_CUDNN_STATUS(cudnnSetConvolutionGroupCount, status);

        status = cudnnSetConvolutionMathType(conv_desc_, shape_in0.GetDataType() == DATATYPE_FLOAT16 ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH);
        if(is_3d_conv)
            CHECK_CUDNN_ND_STATUS(cudnnSetConvolutionMathType, status);
        else
            CHECK_CUDNN_STATUS(cudnnSetConvolutionMathType, status);
        //CHECK_CUDNN_STATUS(cudnnSetConvolutionMathType, (cudnnStatus_t)1);

        size_t work_space_bytes = 0;
        cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        status = cudnnGetConvolutionBackwardDataWorkspaceSize(GetCudnnHandle(), w_desc_, x_tensor_, conv_desc_, y_tensor_, algo, &work_space_bytes);
        if(is_3d_conv)
            CHECK_CUDNN_ND_STATUS(cudnnGetConvolutionBackwardDataWorkspaceSize, status);
        else
            CHECK_CUDNN_STATUS(cudnnGetConvolutionBackwardDataWorkspaceSize, status);
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
            if(is_group_not_depthwise || is_fp32){
                //
            }else{
                c = (c + 7) / 8 * 8;
            }
            int dim_b[5] = {n, c, d, h, w};
            status = cudnnSetTensorNdDescriptorEx(b_tensor_, is_fp32 ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC, is_fp32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF, 5, dim_b);
            CHECK_CUDNN_ND_STATUS(cudnnSetTensorNdDescriptorEx, status);
        }else{
            int n = 1; int c = shape_in1.GetDim(0); int h = 1; int w = 1;
            if(shape_in0.GetDataType() == DATATYPE_FLOAT16){
                if(!is_group_not_depthwise)
                    status = cudnnSetTensor4dDescriptor(b_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 1, (c + 7) / 8 * 8, 1, 1);
                else
                    status = cudnnSetTensor4dDescriptor(b_tensor_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 1, c, 1, 1);
            }
            else
                status = cudnnSetTensor4dDescriptor(b_tensor_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, 1, 1);
        }
        if(is_3d_conv)
            CHECK_CUDNN_ND_STATUS(cudnnSetTensor4dDescriptor, status);
        else
            CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor, status);

        float alpha = 1.0f;
        float beta = 0.0f;
        void *input_data_ptr = ctx->GetInput<TensorImpl>(0)->GetBufferPtr();
        void *weight_data_ptr = ctx->GetInput<TensorImpl>(1)->GetBufferPtr();
        void *output_data_ptr = ctx->GetOutput<TensorImpl>(0)->GetBufferPtr();

        status = cudnnConvolutionBackwardData(GetCudnnHandle(),
                                                &alpha,
                                                w_desc_,
                                                weight_data_ptr,
                                                x_tensor_,
                                                input_data_ptr,
                                                conv_desc_,
                                                algo,
                                                conv_work_buffer_,
                                                work_space_bytes,
                                                &beta,
                                                y_tensor_,
                                                output_data_ptr);
        if(is_3d_conv)
            CHECK_CUDNN_ND_STATUS(cudnnConvolutionBackwardData, status);
        else
            CHECK_CUDNN_STATUS(cudnnConvolutionBackwardData, status);
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
            if(is_3d_conv)
                CHECK_CUDNN_ND_STATUS(cudnnAddTensor, status);
            else
                CHECK_CUDNN_STATUS(cudnnAddTensor, status);
        #else
            TensorShape shape_bias = *ctx->GetInput<TensorImpl>(2)->GetShape();
            std::vector<int64_t> dim_bias;
            dim_bias.push_back(1);
            dim_bias.push_back((shape_in0.GetDataType() == DATATYPE_FLOAT16 && !is_group_not_depthwise) ? ((shape_bias.GetDim(0) + 7) / 8 * 8) : shape_bias.GetDim(0));
            dim_bias.push_back(1);
            if(shape_in1.GetDimCount() > 3)
                dim_bias.push_back(1);
            if(shape_in1.GetDimCount() > 4)
                dim_bias.push_back(1);
            shape_bias.Reshape(dim_bias);
            ppl::common::RetCode status1 = PPLCUDAArithMeticAddForwardImp(GetStream(),
                                                                          &shape_out, ctx->GetOutput<TensorImpl>(0)->GetBufferPtr(),
                                                                          &shape_bias, ctx->GetInput<TensorImpl>(2)->GetBufferPtr(),
                                                                          &shape_out, ctx->GetOutput<TensorImpl>(0)->GetBufferPtr());
            return status1;
        #endif
        }
        return ppl::common::RC_SUCCESS;
    }
    return ppl::common::RC_UNSUPPORTED;
}
#endif

ppl::common::RetCode ConvTransposeKernel::DoExecute(KernelExecContext* ctx) {
//#if 1
#if defined(PPLNN_USE_DNN)
    return DoExecute_MCDNN(ctx);
#else
    return DoExecute_PPL(ctx);
#endif
}

}}} // namespace ppl::nn::cuda
