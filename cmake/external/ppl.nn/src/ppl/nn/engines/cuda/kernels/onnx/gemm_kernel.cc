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

#include "ppl/nn/engines/cuda/kernels/onnx/gemm_kernel.h"
#include "ppl/nn/engines/cuda/module/cuda_module.h"
#include "ppl/common/destructor.h"
#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/gemm/gemm.h"
#include "ppl/common/cuda/cuda_types.h"

namespace ppl { namespace nn { namespace cuda {
#ifdef PPLNN_USE_DNN
#define  CHECK_CUBLAS_STATUS(cublas_api_str, status)                                                            \
            do{                                                                                                 \
                if(status != CUBLAS_STATUS_SUCCESS){                                                            \
                    LOG(ERROR) <<"cublas interface(" << #cublas_api_str << ") return error code: " << (int)status;   \
                    return ppl::common::RC_OTHER_ERROR;                                                         \
                }                                                                                               \
            }while(0)
#ifdef PPLNN_USE_MACA

/*
1、not support bias_coef
2、only support bias vector
3、not support hardswish
4、only support transa = MCBLAS_OP_N, transb = MCBLAS_OP_N
*/

ppl::common::RetCode mcblasGemmExBiasActFp16_New(mcblasLtHandle_t &ltHandle,
                                             mcblasLtMatmulDesc_t operationDesc,
                                             mcblasLtMatrixLayout_t Adesc,
                                             mcblasLtMatrixLayout_t Bdesc,
                                             mcblasLtMatrixLayout_t Cdesc,
                                             cublasOperation_t transa,  /* new fused-kernel only support MCBLAS_OP_N for transa now */
                                             cublasOperation_t transb,  /* new fused-kernel only support MCBLAS_OP_N for transb now */
                                             int32_t m,
                                             int32_t n,
                                             int32_t k,
                                             const float alpha,
                                             const void *A, /* device pointer */
                                             int64_t lda,
                                             const void *B, /* device pointer */
                                             int64_t ldb,
                                             void *C, /* device pointer */
                                             int64_t ldc,
                                            //  float *bias_coef,     /* device pointer */
                                             void *bias, /* device pointer, when bias is nullptr, means no bias */
                                            //  int64_t ldbias,
                                            //  const int32_t bias_mode, /* 0-none, 1-scale, 2-vector, 3-matrix */
                                            //  const int32_t act_type,  /* 0-none, 1-Relu, 2-Gelu, 3-HardSwish */
                                             const int32_t act_type,  /* 0-none, 1-Relu, 2-Gelu, HardSwish is not supported */
                                             cudaStream_t stream,
                                             void *D = nullptr /* device pointer */)
{
    if (A == nullptr) {
        LOG(ERROR) << "Invalid nullptr pointer of matrix A";
    }
    if (B == nullptr) {
        LOG(ERROR) << "Invalid nullptr pointer of matrix B";
    }
    if (act_type < 0 || act_type > 2) {
        LOG(ERROR) << "Invalid act_type: " << act_type;
    }
    cublasStatus_t status;

    mcblasLtEpilogue_t epilogue;;
    if (bias != nullptr) {
        if (act_type == 0) {
            epilogue = MCBLASLT_EPILOGUE_BIAS;
        } if (act_type == 1) {
            epilogue = MCBLASLT_EPILOGUE_RELU_BIAS;
        } else if (act_type == 2) {
            epilogue = MCBLASLT_EPILOGUE_GELU_BIAS;
        }
    } else {
        if (act_type == 0) {
            epilogue = MCBLASLT_EPILOGUE_DEFAULT;
        } if (act_type == 1) {
            epilogue = MCBLASLT_EPILOGUE_RELU;
        } else if (act_type == 2) {
            epilogue = MCBLASLT_EPILOGUE_GELU;
        }
    }

    status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);
    status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);

    status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);

    status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);
    float beta = D != nullptr ? 1.f : 0.f;
    void *work_ptr = nullptr;
    size_t work_size = 0;
    if(k >= 4096){
        cudaMalloc(&work_ptr, 32 * 1024 * 1024);
        if(work_ptr == nullptr)
            return RC_OUT_OF_MEMORY;
        work_size = 32 * 1024 * 1024;
    }
    status = mcblasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,  // we can set alpha to 1.0f if needed
                            A, Adesc,
                            B, Bdesc,
                            &beta,  // beta, it is not needed
                            D != nullptr ? D : C, Cdesc,
                            C, Cdesc,
                            nullptr,  // algo pointer, it is not needed
                            work_ptr,  // workspace pointer, it is not needed
                            work_size,        // workspaceSize, it is not needed
                            stream);
    CHECK_CUBLAS_STATUS(mcblasLtMatmul, status);
    if(work_ptr != nullptr)
        cudaFree(work_ptr);
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode mcblasGemmExBiasActFp16_Old(mcblasLtHandle_t &ltHandle,
                                             mcblasLtMatmulDesc_t operationDesc,
                                             mcblasLtMatrixLayout_t Adesc,
                                             mcblasLtMatrixLayout_t Bdesc,
                                             mcblasLtMatrixLayout_t Cdesc,
                                             cublasOperation_t transa,
                                             cublasOperation_t transb,
                                             int32_t m,
                                             int32_t n,
                                             int32_t k,
                                             const float alpha,
                                             const void *A, /* device pointer */
                                             int64_t lda,
                                             const void *B, /* device pointer */
                                             int64_t ldb,
                                             void *C, /* device pointer */
                                             int64_t ldc,
                                             float *bias_coef,     /* device pointer */
                                             void *bias, /* device pointer */
                                             int64_t ldbias,
                                             const int32_t bias_mode, /* 0-none, 1-scale, 2-vector, 3-matrix */
                                             const int32_t act_type,  /* 0-none, 1-Relu, 2-Gelu, 3-HardSwish */
                                             cudaStream_t stream,
                                             void *D = nullptr /* device pointer */)
{
    if (A == nullptr) {
        LOG(ERROR) << "Invalid nullptr pointer of matrix A";
    }
    if (B == nullptr) {
        LOG(ERROR) << "Invalid nullptr pointer of matrix B";
    }
    if (act_type < 0 || act_type > 3) {
        LOG(ERROR) << "Invalid act_type: " << act_type;
    }
    if (bias_mode < 0 || bias_mode > 3) {
        LOG(ERROR) << "Invalid bias_mode: " << bias_mode;
    }
    if (bias_mode != 0 && bias == nullptr) {
        LOG(ERROR) << "Invalid nullptr pointer of bias";
    }
    cublasStatus_t status;
    mcblasLtEpilogue_t epilogue = MCBLASLT_EPILOGUE_FLEXIBLE_BIAS;
    switch (act_type) {
        case 1:
            epilogue = MCBLASLT_EPILOGUE_RELU_FLEXIBLE_BIAS;
            break;
        case 2:
            epilogue = MCBLASLT_EPILOGUE_GELU_FLEXIBLE_BIAS;
            break;
        case 3:
            epilogue = MCBLASLT_EPILOGUE_HARDSWISH_FLEXIBLE_BIAS;
            break;
        case 0:
        default:
            break;
    }

    status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);
    status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);

    // when bias_mode is matrix, we will need the ldbias
    status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_BIAS_MODE, &bias_mode, sizeof(bias_mode));
    CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);
    if(bias_mode == 3) {
        status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_BIAS_LD, &ldbias, sizeof(ldbias));
        CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);
    }
    status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);
    status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_BIAS_COEF_POINTER, &bias_coef, sizeof(bias_coef));
    CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);
    //float bias_coef_value = 0.f;
    //cudaMemcpyAsync(&bias_coef_value, bias_coef, sizeof(bias_coef_value), cudaMemcpyDeviceToHost, stream);
    //printf("[ort]~~~~~~~~~~~~~~~~~~~~~~ copy from device(%p), value(%f)\n", bias_coef, bias_coef_value);
    macaDataType_t bias_coef_type = MACA_R_32F;
    status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_BIAS_COEF_TYPE, &bias_coef_type, sizeof(bias_coef_type));
    CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);
    status = mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    CHECK_CUBLAS_STATUS(mcblasLtMatmulDescSetAttribute, status);
    float beta = D != nullptr ? 1.f : 0.f;
    void *work_ptr = nullptr;
    size_t work_size = 0;
    if(k >= 4096){
        cudaMalloc(&work_ptr, 32 * 1024 * 1024);
        if(work_ptr == nullptr)
            return RC_OUT_OF_MEMORY;
        work_size = 32 * 1024 * 1024;
    }
    status = mcblasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,  // we can set alpha to 1.0f if needed
                            A, Adesc,
                            B, Bdesc,
                            &beta,  // beta, it is not needed
                            D != nullptr ? D : C, Cdesc,
                            C, Cdesc,
                            nullptr,  // algo pointer, it is not needed
                            work_ptr,  // workspace pointer, it is not needed
                            work_size,        // workspaceSize, it is not needed
                            stream);
    CHECK_CUBLAS_STATUS(mcblasLtMatmul, status);
    if(work_ptr != nullptr)
        cudaFree(work_ptr);
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode mcblasGemmExBiasActFp16(mcblasLtHandle_t &ltHandle,
                                             mcblasLtMatmulDesc_t operationDesc,
                                             mcblasLtMatrixLayout_t Adesc,
                                             mcblasLtMatrixLayout_t Bdesc,
                                             mcblasLtMatrixLayout_t Cdesc,
                                             cublasOperation_t transa,
                                             cublasOperation_t transb,
                                             int32_t m,
                                             int32_t n,
                                             int32_t k,
                                             const float alpha,
                                             const void *A, /* device pointer */
                                             int64_t lda,
                                             const void *B, /* device pointer */
                                             int64_t ldb,
                                             void *C, /* device pointer */
                                             int64_t ldc,
                                             float *bias_coef,     /* device pointer */
                                             void *bias, /* device pointer */
                                             int64_t ldbias,
                                             const int32_t bias_mode, /* 0-none, 1-scale, 2-vector, 3-matrix */
                                             const int32_t act_type,  /* 0-none, 1-Relu, 2-Gelu, 3-HardSwish */
                                             cudaStream_t stream,
                                             void *D = nullptr /* device pointer */){

    if(bias_mode==2 && act_type !=3 && transa == MCBLAS_OP_N && transb == MCBLAS_OP_N && !bias_coef){
        return mcblasGemmExBiasActFp16_New(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, transa, transb, m, n, k, alpha, A,lda,B,ldb,C,ldc,bias,act_type,stream,D);
    }else{
        return mcblasGemmExBiasActFp16_Old(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, transa, transb, m, n, k, alpha, A,lda,B,ldb,C,ldc,bias_coef, bias, ldbias, bias_mode,act_type,stream,D);
    }

}

#endif//PPLNN_USE_MACA
#endif//PPLNN_USE_DNN
bool GemmKernel::CanDoExecute(const KernelExecContext& ctx) const {
    const TensorShape& input = *ctx.GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& weight = *ctx.GetInput<TensorImpl>(1)->GetShape();
    if (input.CalcBytesIncludingPadding() == 0) {
        return false;
    }
    if ((param_->param.transA == 0 && input.GetDim(1) != weight.GetDim(1)) &&
        (param_->param.transA == 1 && input.GetDim(0) != weight.GetDim(1))) {
        return false;
    }

    return true;
}

ppl::common::RetCode GemmKernel::DoExecute_PPL(KernelExecContext* ctx) {
    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param;

    auto input = ctx->GetInput<TensorImpl>(0);
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    const TensorShape& shape_in0 = *input->GetShape();
    const TensorShape& shape_in1 = *weight->GetShape();
    // const TensorShape& shape_out = *output->GetShape();

    temp_conv_param.in_num  = param_->param.transA ? shape_in0.GetDim(1) : shape_in0.GetDim(0);
    temp_conv_param.num_chl = param_->param.transB ? shape_in1.GetDim(1) : shape_in1.GetDim(0);
    temp_conv_param.num_flt = param_->param.transB ? shape_in1.GetDim(0) : shape_in1.GetDim(1);
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

    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);

    struct algo_param_t algo_param;
    algo_param = param_->extra_param.algo_info;

    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_bytes = PPLGemmCUDAGetRuntimeBufSize(&shape_in0, temp_conv_param, algo_param.splitk,
            algo_param.splitf, param_->param.transA);
    int trans_extra_size = 0;
    if(param_->param.transA)
    {
        auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(input->GetShape()->GetDataFormat());
        trans_extra_size = ((input->GetShape()->GetDim(0) + align_size - 1) / align_size * align_size)
                            * ((input->GetShape()->GetDim(1) + align_size - 1) / align_size * align_size)
                            * ppl::common::GetSizeOfDataType(input->GetShape()->GetDataType()) * 2;
    }

    auto status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes + trans_extra_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_bytes << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    cudaMemsetAsync(tmp_buffer_desc.addr, 0, tmp_buffer_bytes + trans_extra_size, GetStream());
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;
    GemmKernelParam param_kernel_;
    param_kernel_.alpha = param_->param.alpha;
    param_kernel_.beta = param_->param.beta;
    param_kernel_.transA = param_->param.transA;
    param_kernel_.transB = param_->param.transB;

    // convert filter only if the filter tensor is an output of another kernel
    BufferDesc weight_buffer;
    auto newshape = *weight->GetShape();
    if (!param_->extra_param.is_initializer_weight) {
        auto align_size = ppl::common::cuda::GetDataFormatChannelAlignment(weight->GetShape()->GetDataFormat());
        newshape.SetPadding1(0, (newshape.GetDim(0) + align_size - 1) / align_size * align_size - newshape.GetDim(0));

        auto status = GetCudaDevice()->Realloc(newshape, &weight_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        PPLCUDAGemmModifyWeights(stream, &newshape, weight->GetBufferPtr(), weight_buffer.addr, &param_kernel_);
    }
    ppl::common::Destructor __tmp_buffer_guard__([this, &weight_buffer]() -> void {
        GetCudaDevice()->Free(&weight_buffer);
    });

    TensorShape bias_shape;
    void* bias = nullptr;
    if (ctx->GetInputCount() >= 3) {
        bias_shape = *ctx->GetInput<TensorImpl>(2)->GetShape();
        bias = ctx->GetInput<TensorImpl>(2)->GetBufferPtr();
    }

    auto stream = GetStream();
    CUfunction module_func = nullptr;
#ifdef PPLNN_ENABLE_CUDA_JIT
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);
    module_func = module->GetKernelFunc();
#endif
    if (shape_in0.GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        if(param_kernel_.transA){
            cudaMemcpyAsync(tmp_buffer, input->GetBufferPtr(), input->GetShape()->CalcBytesIncludingPadding(), cudaMemcpyDeviceToDevice, GetStream());
            status = PPLCUDAGemmForwardImp(GetCudaDevice()->GetDeviceProp(), stream, module_func, input->GetShape(), tmp_buffer,
                                            weight->GetShape(), weight->GetBufferPtr(), bias,
                                            output->GetShape(), output->GetBufferPtr(),
                                            param_kernel_, tmp_buffer + tmp_buffer_bytes + trans_extra_size / 2, temp_fuse_param,
                                            param_->extra_param.algo_info);
        }
        else{
            status = PPLCUDAGemmForwardImp(GetCudaDevice()->GetDeviceProp(), stream, module_func, input->GetShape(), input->GetBufferPtr(),
                                            weight->GetShape(), weight->GetBufferPtr(), bias,
                                            output->GetShape(), output->GetBufferPtr(),
                                            param_kernel_, tmp_buffer, temp_fuse_param, param_->extra_param.algo_info);
        }

    } else if (shape_in0.GetDataType() == ppl::common::DATATYPE_INT8) {
        quant_param_t temp_quant_param;
        auto input_quant = GetCommonParam()->cuda_tensor_info->at(input->GetEdge()->GetId());
        auto output_quant = GetCommonParam()->cuda_tensor_info->at(output->GetEdge()->GetId());
        auto input_scale = input_quant.scale[0];
        auto output_scale = output_quant.scale[0];
        auto d_weight_scale = ctx->GetInput<TensorImpl>(ctx->GetInputCount() - 1)->GetBufferPtr();
        temp_quant_param.in_scale = input_scale;
        temp_quant_param.out_scale = 1 / output_scale;
        temp_quant_param.d_flt_scale = d_weight_scale;
        if (temp_fuse_param.has_elt) {
            auto tps = param_->extra_param.fuse_info.types;
            auto ret = std::find(tps.begin(), tps.end(), "Add");
            if (ret == tps.end())
                LOG(ERROR) << "fuse_info types error: no add op";
            int id = ret - tps.begin();
            auto elt_index = param_->extra_param.fuse_info.input_inds[id];
            auto elt = ctx->GetInput<TensorImpl>(elt_index);
            auto elt_quant = GetCommonParam()->cuda_tensor_info->at(elt->GetEdge()->GetId());
            temp_quant_param.pre_scale = elt_quant.scale[0];
        }
        if (param_->extra_param.fuse_info.channel_offset >= 0) {
            temp_quant_param.out_scale =
                1 / GetCommonParam()->cuda_tensor_info->at(param_->extra_param.fuse_info.concat_edge_id).scale[0];
        }
        if(param_kernel_.transA){
            cudaMemcpyAsync(tmp_buffer, input->GetBufferPtr(), input->GetShape()->CalcBytesIncludingPadding(), cudaMemcpyDeviceToDevice, GetStream());
            status = PPLCUDAGemmForwardImpInt8(GetCudaDevice()->GetDeviceProp(), stream, module_func, input->GetShape(), tmp_buffer,
                                            weight->GetShape(), weight->GetBufferPtr(), bias,
                                            output->GetShape(), output->GetBufferPtr(),
                                            param_kernel_, tmp_buffer + tmp_buffer_bytes + trans_extra_size / 2, temp_quant_param,
                                            temp_fuse_param, param_->extra_param.algo_info);
        }
        else{
            status = PPLCUDAGemmForwardImpInt8(GetCudaDevice()->GetDeviceProp(), stream, module_func, input->GetShape(), input->GetBufferPtr(),
                                            weight->GetShape(), weight->GetBufferPtr(), bias,
                                            output->GetShape(), output->GetBufferPtr(),
                                            param_kernel_, tmp_buffer, temp_quant_param,
                                            temp_fuse_param, param_->extra_param.algo_info);
        }
    }

    return status;
}

#ifdef PPLNN_USE_DNN
ppl::common::RetCode GemmKernel::DoExecute_BLAS(KernelExecContext* ctx) {
    ppl::common::RetCode status;
    auto input = ctx->GetInput<TensorImpl>(0);
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    const TensorShape& shape_in0 = *input->GetShape();
    const TensorShape& shape_in1 = *weight->GetShape();
    const TensorShape& shape_out0 = *output->GetShape();
    cublasHandle_t cublasHandle = GetCublasHandle();
    if(cublasHandle == nullptr){
        LOG(ERROR) <<"DoExecute_BLAS: cublasHandle == nullptr";
        return ppl::common::RC_NOT_FOUND;
    }
    if (shape_in0.GetDataType() == ppl::common::DATATYPE_FLOAT16 || shape_in0.GetDataType() == ppl::common::DATATYPE_FLOAT32){
        bool isFp16 = (shape_in0.GetDataType() == ppl::common::DATATYPE_FLOAT16);
    #ifdef PPLNN_USE_MACA
        //printf("####################################### it is mcblaslt runt ###########################################\n");
        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_N;
        int align_size = isFp16 ? 8 : 1;//ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());
        int m = (shape_in1.GetDim(1) + align_size - 1) / align_size * align_size;
        int n = shape_in0.GetDim(0);
        int k = (shape_in1.GetDim(0) + align_size - 1) / align_size * align_size;
        int lda = m;
        int ldb = k;
        int ldc = m;
        int ldbias = ctx->GetInputCount() > 2 ? ldc : 1;
        int32_t bias_mode = 0;
        if(ctx->GetInputCount() > 2){
            auto bias = ctx->GetInput<TensorImpl>(2);
            if(bias->GetShape()->GetDimCount() == 2){
                if(bias->GetShape()->GetDim(0) == 1)
                    if(bias->GetShape()->GetDim(1) == 1)
                        bias_mode = 1;
                    else
                        bias_mode = 2;
                else
                    bias_mode = 3;
            }
            if(bias->GetShape()->GetDimCount() == 1){
                if(bias->GetShape()->GetDim(0) == 1)
                    bias_mode = 1;
                else{
                    bias_mode = 2;
                }
            }
        }
        if(((bias_mode == 2 || bias_mode == 1) && gemm_bias_len_ != m)
           || (bias_mode == 3 && gemm_bias_len_ != m * n)){
            auto bias = ctx->GetInput<TensorImpl>(2);
            auto bias_tmp_shape = TensorShape();
            if(bias_mode == 2 || bias_mode == 1){
                bias_tmp_shape.SetDimCount(1);
                bias_tmp_shape.SetDim(0, m);
                bias_tmp_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                bias_tmp_shape.SetDataType(shape_in0.GetDataType());
                if(gemm_bias_buffer_ != nullptr){
                    cudaFree(gemm_bias_buffer_);
                    gemm_bias_buffer_ = nullptr;
                }
                cudaMalloc(&gemm_bias_buffer_, bias_tmp_shape.CalcBytesIncludingPadding());
                if(gemm_bias_buffer_ == nullptr)
                    return RC_OUT_OF_MEMORY;
                gemm_bias_len_ = m;
            }else{
                bias_tmp_shape = *bias->GetShape();
                if(gemm_bias_buffer_ != nullptr){
                    cudaFree(gemm_bias_buffer_);
                    gemm_bias_buffer_ = nullptr;
                }
                cudaMalloc(&gemm_bias_buffer_, bias_tmp_shape.CalcBytesIncludingPadding());
                if(gemm_bias_buffer_ == nullptr)
                    return RC_OUT_OF_MEMORY;
                gemm_bias_len_ = m * n;
            }
            cudaMemsetAsync(gemm_bias_buffer_, 0, bias_tmp_shape.CalcBytesIncludingPadding(), GetStream());
            cudaMemcpyAsync(gemm_bias_buffer_, ctx->GetInput<TensorImpl>(2)->GetBufferPtr(),
                            bias_mode != 1 ? bias_tmp_shape.CalcBytesIncludingPadding() : GetSizeOfDataType(shape_in0.GetDataType()),
                            cudaMemcpyDeviceToDevice, GetStream());
            GemmKernelParam param_kernel;
            param_kernel.alpha = 1.0f;
            param_kernel.beta  = param_->param.beta;
            param_kernel.transA = 0;
            param_kernel.transB = 0;
            PPLCUDAGemmModifyBias( GetStream(), shape_in0.GetDataType(), &bias_tmp_shape, gemm_bias_buffer_, &param_kernel);
        }


        mcblasLtHandle_t mcblasLtHandle = GetMcblasLtHandle();
        if(mcblasLtHandle == nullptr){
            LOG(ERROR) <<"DoExecute_BLAS: mcblasLtHandle == nullptr";
            return ppl::common::RC_NOT_FOUND;
        }
        if(!ismcblasLtInit_ || m_ != m || n_ != n || k_ != k){
            cublasStatus_t status = mcblasLtMatmulDescCreate(&mcblasLtOperationDesc_, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            CHECK_CUBLAS_STATUS(mcblasLtMatmulDescCreate, status);
            status = mcblasLtMatrixLayoutCreate(&Adesc_, isFp16 ? CUDA_R_16F : CUDA_R_32F, m, k, lda);
            CHECK_CUBLAS_STATUS(mcblasLtMatrixLayoutCreate, status);
            status = mcblasLtMatrixLayoutCreate(&Bdesc_, isFp16 ? CUDA_R_16F : CUDA_R_32F, k, n, ldb);
            CHECK_CUBLAS_STATUS(mcblasLtMatrixLayoutCreate, status);
            status = mcblasLtMatrixLayoutCreate(&Cdesc_, isFp16 ? CUDA_R_16F : CUDA_R_32F, m, n, ldc);
            CHECK_CUBLAS_STATUS(mcblasLtMatrixLayoutCreate, status);
            ismcblasLtInit_ = true;
            m_ = m;
            n_ = n;
            k_ = k;
        }
        //printf("run param: m:%d, n:%d, k:%d, lda:%d, ldb:%d, ldc:%d, bias_mode: %d\n", m, n, k, lda, ldb, ldc, bias_mode);
        void* bias_data_ptr = nullptr;
        if(bias_mode != 0)
            bias_data_ptr =  gemm_bias_buffer_;
        int32_t act_type = 0;
        if(param_->extra_param.fuse_info.types.size() > 0){
            if(param_->extra_param.fuse_info.types[0] == "Relu")
                act_type = 1;
            if(param_->extra_param.fuse_info.types[0] == "Gelu")
                act_type = 2;
            if(param_->extra_param.fuse_info.types[0] == "HardSwish")
                act_type = 3;
        }

        if(!(bias_mode==2 && act_type !=3 && transa == MCBLAS_OP_N && transb == MCBLAS_OP_N) && gemm_beta_buffer_ == nullptr){
            auto beta_tmp_shape = TensorShape();
            beta_tmp_shape.SetDimCount(1);
            beta_tmp_shape.SetDim(0, 1);
            beta_tmp_shape.SetDataFormat(DATAFORMAT_NDARRAY);
            beta_tmp_shape.SetDataType(DATATYPE_FLOAT32);
            cudaMalloc(&gemm_beta_buffer_, beta_tmp_shape.CalcBytesIncludingPadding());
            if(gemm_beta_buffer_ == nullptr)
                return RC_OUT_OF_MEMORY;
            float beta = 1.f;
            cudaMemcpyAsync(gemm_beta_buffer_, &beta, sizeof(beta), cudaMemcpyHostToDevice, GetStream());
            gemm_beta_len_ = sizeof(float);
        }

        mcblasGemmExBiasActFp16(mcblasLtHandle,
                                mcblasLtOperationDesc_,
                                Adesc_,
                                Bdesc_,
                                Cdesc_,
                                transa,
                                transb,
                                m,
                                n,
                                k,
                                param_->param.alpha,
                                weight->GetBufferPtr(),
                                lda,
                                input->GetBufferPtr(),
                                ldb,
                                output->GetBufferPtr(),
                                ldc,
                                (float*)gemm_beta_buffer_,
                                bias_data_ptr,
                                ldbias,
                                bias_mode,
                                act_type,
                                GetStream());
    #else
        cublasOperation_t transa = (cublasOperation_t)param_->param.transB;
        cublasOperation_t transb = (cublasOperation_t)param_->param.transA;
        int align_size = 8;//ppl::common::cuda::GetDataFormatChannelAlignment(shape_in0.GetDataFormat());
        int m = param_->param.transB ? shape_in1.GetDim(0) : ((shape_in1.GetDim(1) + align_size - 1) / align_size * align_size);
        int n = param_->param.transA ? shape_in0.GetDim(1) : shape_in0.GetDim(0);
        int k = param_->param.transB ? shape_in1.GetDim(1) : ((shape_in1.GetDim(0) + align_size - 1) / align_size * align_size);
        float alpha = param_->param.alpha;
        void *A = weight->GetBufferPtr();
        cudaDataType_t Atype = isFp16 ? CUDA_R_16F : CUDA_R_32F;
        int lda = param_->param.transB ? k : m;
        void *B = input->GetBufferPtr();
        cudaDataType_t Btype = isFp16 ? CUDA_R_16F : CUDA_R_32F;
        int ldb = param_->param.transA ? n : k;
        float beta = ctx->GetInputCount() > 2 ? param_->param.beta : 0.0;
        cudaDataType_t Ctype = isFp16 ? CUDA_R_16F : CUDA_R_32F;
        if (!param_->extra_param.is_initializer_weight){
            if(gemm_weight_len_ != shape_in1.CalcBytesIncludingPadding()){
                auto shape_in1_temp = shape_in1;
                shape_in1_temp.SetPadding1(0, ((shape_in1.GetDim(0) + align_size - 1) / align_size * align_size) - shape_in1.GetDim(0));
                if(gemm_weight_buffer_ != nullptr){
                    cudaFree(gemm_weight_buffer_);
                    gemm_weight_buffer_ = nullptr;
                }
                cudaMalloc(&gemm_weight_buffer_, shape_in1_temp.CalcBytesIncludingPadding());
                if(gemm_weight_buffer_ == nullptr)
                    return RC_OUT_OF_MEMORY;
                cudaMemsetAsync(gemm_weight_buffer_, 0, shape_in1_temp.CalcBytesIncludingPadding(), GetStream());
                gemm_weight_len_ = shape_in1.CalcBytesIncludingPadding();
            }
            cudaMemcpyAsync((void*)gemm_weight_buffer_, A, shape_in1.CalcBytesIncludingPadding(), cudaMemcpyDeviceToDevice, GetStream());
            A = (void*)gemm_weight_buffer_;
        }
        if(gemm_bias_len_ < shape_out0.CalcBytesIncludingPadding()){
            if(gemm_bias_buffer_ != nullptr){
                cudaFree(gemm_bias_buffer_);
                gemm_bias_buffer_ = nullptr;
            }
            cudaMalloc(&gemm_bias_buffer_, shape_out0.CalcBytesIncludingPadding());
            if(gemm_bias_buffer_ == nullptr)
                return RC_OUT_OF_MEMORY;
            gemm_bias_len_ = shape_out0.CalcBytesIncludingPadding();
            cudaMemsetAsync(gemm_bias_buffer_, 0, shape_out0.CalcBytesIncludingPadding(), GetStream());
            if(ctx->GetInputCount() > 2){
                auto bias = ctx->GetInput<TensorImpl>(2);
                auto shape_bias = *ctx->GetInput<TensorImpl>(2)->GetShape();
                if(shape_bias.CalcBytesIncludingPadding() < shape_out0.CalcBytesIncludingPadding()){
                    int n_bias = (shape_out0.GetDim(1) + align_size - 1) / align_size * align_size;
                    int m_bias = shape_out0.CalcElementsIncludingPadding() / n_bias;
                    auto bias_tmp_shape = TensorShape();
                    bias_tmp_shape.SetDimCount(1);
                    bias_tmp_shape.SetDim(0, m_bias);
                    bias_tmp_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                    bias_tmp_shape.SetDataType(shape_in0.GetDataType());

                    BufferDesc gemm_bias_tmp_;
                    status = GetCudaDevice()->Realloc(bias_tmp_shape, &gemm_bias_tmp_);
                    if (status != ppl::common::RC_SUCCESS) {
                        LOG(ERROR) << "alloc buffer for gemm bias space failed: " << GetRetCodeStr(status);
                        return status;
                    }
                    Destructor __bias_tmp_buf_guard__([this, &gemm_bias_tmp_]() -> void {
                        GetCudaDevice()->Free(&gemm_bias_tmp_);
                    });
                    void *bias_gpu_tmp_ptr = nullptr;
                    if(shape_bias.CalcElementsExcludingPadding() == 1){
                        auto bias_tmp_shape1 = bias_tmp_shape;
                        bias_tmp_shape1.SetDim(0, n_bias);
                        if(bias_gpu_tmp_ptr != nullptr){
                            cudaFree(bias_gpu_tmp_ptr);
                            bias_gpu_tmp_ptr = nullptr;
                        }
                        cudaMalloc(&bias_gpu_tmp_ptr, bias_tmp_shape1.CalcBytesExcludingPadding());
                        if(bias_gpu_tmp_ptr == nullptr){
                            LOG(ERROR) << "alloc buffer for bias_gpu_tmp_ptr space failed: ";
                            return RC_OUT_OF_MEMORY;
                        }
                        std::vector<unsigned char> bias_host_float16(n_bias * (isFp16 ? 2 : 4));
                        memset(bias_host_float16.data(), 0, bias_host_float16.size());
                        void *bias_value_single = malloc(isFp16 ? 2 : 4);
                        assert(bias_value_single != nullptr);
                        cudaMemcpyAsync(&bias_value_single, bias->GetBufferPtr(), isFp16 ? 2 : 4, cudaMemcpyDeviceToHost, GetStream());
                        for(int i = 0; i < shape_out0.GetDim(1); i++){
                            if(isFp16){
                                unsigned short *bias_host_temp = (unsigned short *)bias_host_float16.data();
                                bias_host_temp[i] = *(unsigned short *)bias_value_single;
                            }else{
                                float *bias_host_temp = (float *)bias_host_float16.data();
                                bias_host_temp[i] = *(float *)bias_value_single;
                            }
                        }
                        cudaMemcpyAsync(bias_gpu_tmp_ptr, bias_host_float16.data(), bias_host_float16.size(), cudaMemcpyHostToDevice, GetStream());
                    }

                    std::vector<float> ones(m_bias, 1.0f);
                    auto bias_tmp_shape_float = bias_tmp_shape;
                    bias_tmp_shape_float.SetDataType(DATATYPE_FLOAT32);
                    status = GetCudaDevice()->GetDataConverter()->ConvertFromHost(&gemm_bias_tmp_, bias_tmp_shape,
                                                                                ones.data(), bias_tmp_shape_float);
                    if (status != ppl::common::RC_SUCCESS) {
                        LOG(ERROR) << "copy data failed: " << ppl::common::GetRetCodeStr(status);
                        return status;
                    }
                    float alpha1 = 1.0f;
                    cublasStatus_t cublasStatus = cublasGemmEx(GetCublasHandle(),
                                                            (cublasOperation_t)0,
                                                            (cublasOperation_t)0,
                                                            n_bias,
                                                            m_bias,
                                                            1,
                                                            &alpha1,
                                                            shape_bias.CalcElementsExcludingPadding() == 1 ? bias_gpu_tmp_ptr : bias->GetBufferPtr(),
                                                            isFp16 ? CUDA_R_16F : CUDA_R_32F,
                                                            n_bias,
                                                            gemm_bias_tmp_,
                                                            isFp16 ? CUDA_R_16F : CUDA_R_32F,
                                                            1,
                                                            &beta,
                                                            gemm_bias_buffer_,
                                                            isFp16 ? CUDA_R_16F : CUDA_R_32F,
                                                            n_bias,
                                                            CUDA_R_32F,
                                                            CUBLAS_GEMM_DEFAULT);
                    if(bias_gpu_tmp_ptr != nullptr){
                        cudaFree(bias_gpu_tmp_ptr);
                    }
                    CHECK_CUBLAS_STATUS(cublasGemmEx, cublasStatus);
                }
            }
        }
        if(ctx->GetInputCount() > 2)
        {
            auto bias = ctx->GetInput<TensorImpl>(2);
            auto shape_bias = *ctx->GetInput<TensorImpl>(2)->GetShape();
            cudaMemcpyAsync(output->GetBufferPtr(),
                            shape_bias.CalcBytesIncludingPadding() < shape_out0.CalcBytesIncludingPadding() ? gemm_bias_buffer_ : bias->GetBufferPtr(),
                            shape_out0.CalcBytesIncludingPadding(),
                            cudaMemcpyDeviceToDevice,
                            GetStream());
        }
        void *C = output->GetBufferPtr();
        int ldc = m;
        cudaDataType computeType = CUDA_R_32F; //CUDA_R_16F : CUDA_R_32F
        cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
        cublasStatus_t cublasStatus = cublasGemmEx(GetCublasHandle(),
                                                   transa,
                                                   transb,
                                                   m,
                                                   n,
                                                   k,
                                                   &alpha,
                                                   A,
                                                   Atype,
                                                   lda,
                                                   B,
                                                   Btype,
                                                   ldb,
                                                   &beta,
                                                   C,
                                                   Ctype,
                                                   ldc,
                                                   computeType,
                                                   algo);
        CHECK_CUBLAS_STATUS(cublasGemmEx, cublasStatus);
    #endif
    } else if (shape_in0.GetDataType() == ppl::common::DATATYPE_INT8){
    #ifdef PPLNN_USE_MACA
        auto bias = ctx->GetInput<TensorImpl>(ctx->GetInputCount() - 1);
        const TensorShape& shape_in2 = *bias->GetShape();

        const bool transA = param_->param.transA;
        const bool transB = param_->param.transB;
        MatMulComputeHelper helper;
        status = helper.Compute(shape_in0, shape_in1, transA, transB, false, false, false);
        const int m = helper.M();
        const int n = helper.N();
        const int k = helper.K();

        auto shape_bias_broadcast = shape_out0;
        shape_bias_broadcast.SetDataType(ppl::common::DATATYPE_FLOAT32);
        if(gemm_bias_len_ != shape_bias_broadcast.CalcBytesIncludingPadding()){
            if(gemm_bias_buffer_ != nullptr){
                cudaFree(gemm_bias_buffer_);
                gemm_bias_buffer_ = nullptr;
            }
            cudaMalloc(&gemm_bias_buffer_, shape_bias_broadcast.CalcBytesIncludingPadding());
            if(gemm_bias_buffer_ == nullptr)
                return RC_OUT_OF_MEMORY;
            cudaMemsetAsync(gemm_bias_buffer_, 0, shape_bias_broadcast.CalcBytesIncludingPadding(), GetStream());
            if(ctx->GetInputCount() > 2 && param_->extra_param.bias_term){
                int ele_size = (shape_in2.GetDataType() ==  DATATYPE_FLOAT16) ? 2 : 4;
                if(bias->GetShape()->GetDimCount() == 2){
                    if(bias->GetShape()->GetDim(1) != n){
                        LOG(ERROR) << "bias->GetShape()->GetDim(0) != m";
                        return status;
                    }
                    if(bias->GetShape()->GetDim(0) == 1){
                        for(int i = 0; i < m; i++)
                        {
                            cudaMemcpyAsync(gemm_bias_buffer_ + ele_size * m * i, (void*)bias->GetBufferPtr(),
                                            ele_size * m, cudaMemcpyDeviceToDevice, GetStream());
                        }
                    }else if(bias->GetShape()->GetDim(0) == m){
                        cudaMemcpyAsync((void*)gemm_bias_buffer_, (void*)bias->GetBufferPtr(),
                                        bias->GetShape()->CalcBytesIncludingPadding(),
                                        cudaMemcpyDeviceToDevice, GetStream());
                    }
                }else if(bias->GetShape()->GetDimCount() == 1){
                    if(bias->GetShape()->GetDim(0) != n){
                        LOG(ERROR) << "bias->GetShape()->GetDim(0)["<< bias->GetShape()->GetDim(0) <<"] != ["<< n <<"]";
                        return status;
                    }

                    auto bias_tmp_shape = TensorShape();
                    bias_tmp_shape.SetDimCount(1);
                    bias_tmp_shape.SetDim(0, m);
                    bias_tmp_shape.SetDataFormat(DATAFORMAT_NDARRAY);
                    bias_tmp_shape.SetDataType(DATATYPE_FLOAT32);

                    BufferDesc gemm_bias_tmp_;
                    status = GetCudaDevice()->Realloc(bias_tmp_shape, &gemm_bias_tmp_);
                    if (status != ppl::common::RC_SUCCESS) {
                        LOG(ERROR) << "alloc buffer for gemm bias space failed: " << GetRetCodeStr(status);
                        return status;
                    }
                    Destructor __bias_tmp_buf_guard__([this, &gemm_bias_tmp_]() -> void {
                        GetCudaDevice()->Free(&gemm_bias_tmp_);
                    });

                    std::vector<float> ones(m, 1.0f);
                    status = GetCudaDevice()->CopyFromHost(&gemm_bias_tmp_, ones.data(), m * ele_size);
                    if (status != ppl::common::RC_SUCCESS) {
                        LOG(ERROR) << "copy data failed: " << ppl::common::GetRetCodeStr(status);
                        return status;
                    }

                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    cublasStatus_t cublasStatus = cublasSgemm(GetCublasHandle(),
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                n, m, 1,
                                                &alpha,
                                                (float*)bias->GetBufferPtr(), n,
                                                (float*)gemm_bias_tmp_.addr, 1,
                                                &beta,
                                                (float*)gemm_bias_buffer_, n);
                }else{
                    LOG(ERROR) <<"Bias dim count > 2";
                    return ppl::common::RC_UNSUPPORTED;
                }
            }
            gemm_bias_len_ = shape_bias_broadcast.CalcBytesIncludingPadding();
        }

        int quant_idx = param_->extra_param.bias_term ? ctx->GetInputCount() - 2 : ctx->GetInputCount() - 1;
        auto quant_tensor = ctx->GetInput<TensorImpl>(quant_idx);
        auto quant_shape = quant_tensor->GetShape();
        int quantMode = (quant_shape->GetDim(0) > 1) ? 1 : 0;
        auto quant_ptr = quant_tensor->GetBufferPtr();
        cublasSetPointerMode(GetCublasHandle(), CUBLAS_POINTER_MODE_DEVICE);
        mcblasStatus_t cublasStatus = mcblasGemmExQuantInt8(GetCublasHandle(),
                                                   transB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                   transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                   helper.N(), helper.M(), helper.K(),
                                                   quant_ptr,
                                                   weight->GetBufferPtr(), CUDA_R_8I, helper.Ldb(transB),
                                                   input->GetBufferPtr(), CUDA_R_8I, helper.Lda(transA),
                                                   output->GetBufferPtr(), CUDA_R_8I, helper.Ldc(),
                                                   CUDA_R_32F, CUBLAS_GEMM_DEFAULT,
                                                   quantMode, gemm_bias_buffer_);
        cublasSetPointerMode(GetCublasHandle(), CUBLAS_POINTER_MODE_HOST);
    #endif // end ifdef PPLNN_USE_MACA
    }
    return ppl::common::RC_SUCCESS;
}
#endif //end ifdef PPLNN_USE_DNN

ppl::common::RetCode GemmKernel::DoExecute(KernelExecContext* ctx) {
    const TensorShape &tensorshape0 = *ctx->GetInput<TensorImpl>(0)->GetShape();
#ifdef PPLNN_USE_DNN
    #ifndef PPLNN_USE_MACA
    if (tensorshape0.GetDataType() == DATATYPE_INT8) {
        return DoExecute_PPL(ctx);
    }
    #endif
    return DoExecute_BLAS(ctx);
#else
    return DoExecute_PPL(ctx);
#endif
}

}}} // namespace ppl::nn::cuda
