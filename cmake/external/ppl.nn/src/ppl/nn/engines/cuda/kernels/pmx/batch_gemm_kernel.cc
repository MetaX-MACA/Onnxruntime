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

#include "ppl/nn/engines/cuda/kernels/pmx/batch_gemm_kernel.h"
#include "cudakernel/arithmetic/arithmetic.h"
#include "cudakernel/gemm/bgemm.h"
#include <cublasLt.h>

namespace ppl { namespace nn { namespace cuda {
using ppl::common::TensorShape;
ppl::common::RetCode PPLCUDABatchGemmKernel_Blas(cublasHandle_t handle, void* input_A_buf, TensorShape* A_shape_ptr, void* input_B_buf,
            TensorShape* B_shape_ptr, void* input_C_buf, TensorShape* C_shape_ptr, void* ouptut_buf, TensorShape* D_shape_ptr, cudaStream_t stream ){
    int m = A_shape_ptr->GetDim(0);
    if(A_shape_ptr->GetDimCount() == 3){
        m*=A_shape_ptr->GetDim(1);
    }
    int k = B_shape_ptr->GetDim(1);
    int n = B_shape_ptr->GetDim(2);

    int stride_A = 0;
    int stride_B = k * n;
    int stride_D = m * n;
    int batch_count = B_shape_ptr->GetDim(0);
    // float alpha = 1.0f;
    // float beta = 0.0f;
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    cublasStatus_t cu_status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                                                            input_B_buf, CUDA_R_16F, n, stride_B,
                                                            input_A_buf, CUDA_R_16F, k, stride_A,
                                                            &beta,
                                                            ouptut_buf, CUDA_R_16F, n, stride_D,
                                                            batch_count, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
    if(cu_status != CUBLAS_STATUS_SUCCESS){
        return ppl::common::RC_OTHER_ERROR;
    }
    if(C_shape_ptr){
        ppl::common::RetCode status =
            PPLCUDAArithMeticAddForwardImp(stream, D_shape_ptr, ouptut_buf, C_shape_ptr,
                                        input_C_buf, D_shape_ptr,ouptut_buf, 1, 1, 1);
    }

    return 0;
}

ppl::common::RetCode PPLCUDABatchGemmKernel_Blaslt(cublasLtHandle_t ltHandle, void* input_A_buf, TensorShape* A_shape_ptr, void* input_B_buf,
            TensorShape* B_shape_ptr, void* input_C_buf, TensorShape* C_shape_ptr, void* ouptut_buf, TensorShape* D_shape_ptr, cudaStream_t stream ){
    int m = A_shape_ptr->GetDim(0);
    if(A_shape_ptr->GetDimCount() == 3){
        m*=A_shape_ptr->GetDim(1);
    }
    int k = B_shape_ptr->GetDim(1);
    int n = B_shape_ptr->GetDim(2);
    // printf("%d,%d,%d\n",m,n,k);
    int64_t stride_A = 0;
    int64_t stride_B = k * n;
    int64_t stride_C = n;
    int64_t stride_D = m * n;
    int batch_count = B_shape_ptr->GetDim(0);

    // Create matrix layouts
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16F, k, m, k);
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16F, n, k, n);
    cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_16F, n, m, n);

    // Set batch count
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
    cublasLtMatrixLayoutSetAttribute(layoutD, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));

    // Set strides
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_A, sizeof(stride_A));
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_B, sizeof(stride_B));
    cublasLtMatrixLayoutSetAttribute(layoutD, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_D, sizeof(stride_D));


    // Create operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F);


    // float alpha = 1.0f;
    // float beta = 0.0f;
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    if(input_C_buf){
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &input_C_buf,sizeof(input_C_buf));
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, &stride_C,sizeof(stride_C));
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        // __half beta = __float2half(1.0f);
        cublasLtMatmul(ltHandle,operationDesc,
                            &alpha, input_B_buf, layoutB,
                            input_A_buf, layoutA,
                            // &beta, input_C_buf, layoutC,
                            &beta, ouptut_buf, layoutD,
                            ouptut_buf, layoutD,
                            NULL, NULL, 0, stream);
    }else{
        // __half beta = __float2half(0.0f);
        cublasLtMatmul(ltHandle,operationDesc,
                            &alpha, input_B_buf, layoutB,
                            input_A_buf, layoutA,
                            &beta, ouptut_buf, layoutD,
                            ouptut_buf, layoutD,
                            NULL, NULL, 0, stream);
    }

    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutD);


    return 0;
}

/*
ppl::common::RetCode PPLCUDABatchGemmKernel_Blaslt(cublasLtHandle_t ltHandle, void* input_A_buf, TensorShape* A_shape_ptr, void* input_B_buf,
            TensorShape* B_shape_ptr, void* input_C_buf, TensorShape* C_shape_ptr, void* ouptut_buf, TensorShape* D_shape_ptr, cudaStream_t stream ){
    int m = A_shape_ptr->GetDim(0);
    if(A_shape_ptr->GetDimCount() == 3){
        m*=A_shape_ptr->GetDim(1);
    }
    int k = B_shape_ptr->GetDim(1);
    int n = B_shape_ptr->GetDim(2);
    // printf("%d,%d,%d\n",m,n,k);
    int64_t stride_A = 0;
    int64_t stride_B = k * n;
    int64_t stride_C = n;
    int64_t stride_D = m * n;
    int batch_count = B_shape_ptr->GetDim(0);

    // Create matrix layouts
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16F, k, m, k);
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16F, n, k, n);
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16F, n, m, 0);
    cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_16F, n, m, n);

    // Set batch count
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
    cublasLtMatrixLayoutSetAttribute(layoutD, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));

    // Set strides
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_A, sizeof(stride_A));
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_B, sizeof(stride_B));
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_C, sizeof(stride_C));
    cublasLtMatrixLayoutSetAttribute(layoutD, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_D, sizeof(stride_D));

    // Create operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F);

    // float alpha = 1.0f;
    // float beta = 0.0f;
    __half alpha = __float2half(1.0f);

    if(input_C_buf){
        __half beta = __float2half(1.0f);
        cublasLtMatmul(ltHandle,operationDesc,
                            &alpha, input_B_buf, layoutB,
                            input_A_buf, layoutA,
                            &beta, input_C_buf, layoutC,
                            ouptut_buf, layoutD,
                            NULL, NULL, 0, stream);
    }else{
        __half beta = __float2half(0.0f);
        cublasLtMatmul(ltHandle,operationDesc,
                            &alpha, input_B_buf, layoutB,
                            input_A_buf, layoutA,
                            &beta, ouptut_buf, layoutD,
                            ouptut_buf, layoutD,
                            NULL, NULL, 0, stream);
    }


    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatrixLayoutDestroy(layoutD);

    return 0;
}
*/

ppl::common::RetCode BatchGemmKernel::DoExecute(KernelExecContext* ctx) {
    cublasHandle_t handle = GetCublasHandle();
    cublasLtHandle_t ltHandle = GetMcblasLtHandle();
    cudaStream_t stream = GetStream();
    auto input_A = ctx->GetInput<TensorImpl>(0);
    auto A_shape_ptr = input_A->GetShape();
    auto input_B = ctx->GetInput<TensorImpl>(1);
    auto B_shape_ptr = input_B->GetShape();

    int batch_count = ctx->GetOutputCount();
    int m = A_shape_ptr->GetDim(0);
    if(A_shape_ptr->GetDimCount() == 3){
        m*=A_shape_ptr->GetDim(1);
    }
    int k = B_shape_ptr->GetDim(1);
    int n = B_shape_ptr->GetDim(2);

    ppl::common::RetCode status = ppl::common::RC_SUCCESS;

    if(ctx->GetInputCount() == 3){
        auto input_C = ctx->GetInput<TensorImpl>(2);
        auto C_shape = *input_C->GetShape();
        if(C_shape.GetDimCount() == 2){
            C_shape.Reshape({C_shape.GetDim(0),1,C_shape.GetDim(1)});
        }
        auto output = ctx->GetOutput<TensorImpl>(0);
        auto D_shape = *output->GetShape();
        D_shape.Reshape({batch_count,m,n});
        if(m%4 ==0 && k%4==0){
            status = PPLCUDABatchGemmKernel_Blaslt(ltHandle, input_A->GetBufferPtr(), A_shape_ptr, input_B->GetBufferPtr(), B_shape_ptr, input_C->GetBufferPtr(), &C_shape,
                                output->GetBufferPtr(), &D_shape, stream);
        }else{
            status = PPLCUDABatchGemmKernel_Blas(handle, input_A->GetBufferPtr(), A_shape_ptr, input_B->GetBufferPtr(), B_shape_ptr, input_C->GetBufferPtr(), &C_shape,
                                    output->GetBufferPtr(), &D_shape, stream);
        }
    }else{
        auto output = ctx->GetOutput<TensorImpl>(0);
        auto D_shape = *output->GetShape();
        D_shape.Reshape({batch_count,m,n});
        if(m%4 ==0 && k%4==0){
            status = PPLCUDABatchGemmKernel_Blaslt(ltHandle, input_A->GetBufferPtr(), A_shape_ptr, input_B->GetBufferPtr(), B_shape_ptr, nullptr, nullptr,
                                output->GetBufferPtr(), &D_shape, stream);
        }else{
            status = PPLCUDABatchGemmKernel_Blas(handle, input_A->GetBufferPtr(), A_shape_ptr, input_B->GetBufferPtr(), B_shape_ptr, nullptr, nullptr,
                                    output->GetBufferPtr(), &D_shape, stream);
        }
    }



    return status;
}


// create output tensor by address offset
ppl::common::RetCode BatchGemmKernel::BeforeExecute(KernelExecContext* ctx){
    auto status = Reshape(ctx);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "Reshape error";
        return status;
    }
    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        tensor->SetDevice(GetCudaDevice());
        status = tensor->ReallocBuffer();
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName() << "] failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }

    auto output0 = ctx->GetOutput<TensorImpl>(0);
    auto output0_shape = output0->GetShape();

    uint32_t batch_num = ctx->GetOutputCount();
    uint64_t output_buffer_len =  output0_shape->CalcBytesIncludingPadding() * batch_num;

    // alloc output buffer
    CudaDevice* device = GetCudaDevice();
    BufferDesc output_buffer;
    device->Realloc(output_buffer_len, &output_buffer);

    output0->SetBuffer(output_buffer, device, true);
    for (uint32_t i = 1; i < ctx->GetOutputCount(); ++i) {
        output_buffer.addr += output0_shape->CalcBytesIncludingPadding();
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        tensor->SetDevice(GetCudaDevice());
        tensor->SetBuffer(output_buffer);
    }

    return 0;

}
}}} // namespace ppl::nn::cuda
