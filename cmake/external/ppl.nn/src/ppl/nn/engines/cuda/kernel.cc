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

#include "ppl/nn/engines/cuda/kernel.h"

#include <chrono>
#include <memory>
#include <fstream>
#include "ppl/common/allocator.h"
#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

CudaKernel::~CudaKernel() {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    if (exec_begin_event_) {
        cudaEventDestroy(exec_begin_event_);
    }
    if (exec_end_event_) {
        cudaEventDestroy(exec_end_event_);
    }
#endif
}

RetCode CudaKernel::Init() {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    auto err = cudaEventCreate(&exec_begin_event_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaEventCreate failed: " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }

    err = cudaEventCreate(&exec_end_event_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaEventCreate failed: " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
#endif

    return RC_SUCCESS;
}

RetCode CudaKernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Reshape error";
        return status;
    }

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        tensor->SetDevice(GetCudaDevice());
        status = tensor->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
class CudaTimingGuard final {
public:
    CudaTimingGuard(cudaStream_t stream, cudaEvent_t* begin_event, cudaEvent_t* end_event, bool is_profiling_enabled)
        : is_profiling_enabled_(is_profiling_enabled), end_event_(end_event), stream_(stream) {
        if (is_profiling_enabled) {
            cudaEventRecord(*begin_event, stream);
            stream_ = stream;
        }
    }
    ~CudaTimingGuard() {
        if (is_profiling_enabled_) {
            cudaEventRecord(*end_event_, stream_);
        }
    }

private:
    bool is_profiling_enabled_;
    cudaEvent_t* end_event_;
    cudaStream_t stream_;
};
#endif

bool CudaKernel::CanDoExecute(const KernelExecContext& ctx) const {
    for (uint32_t i = 0; i < ctx.GetInputCount(); ++i) {
        if(GetNode()->GetInput(i) == INVALID_EDGEID)
            continue;
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor || tensor->GetShape()->CalcBytesIncludingPadding() == 0) {
            LOG(WARNING) << "Cannot execute " << GetName();
            return false;
        }
    }
    return true;
}

RetCode CudaKernel::Execute(KernelExecContext* ctx) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    CudaTimingGuard __timing_guard__(GetCudaDevice()->GetStream(), &exec_begin_event_, &exec_end_event_,
                                     ctx->IsProfilingEnabled());
#endif
#ifndef NDEBUG
    int debug_log_activate = 1;
#else
    int debug_log_activate = 0;
#endif
    uint32_t total_size = 0;
    auto run_begin_ts = std::chrono::system_clock::now();
    char *env_ptr = getenv("MACART_OP_LOG_LEVEL");
    if(env_ptr != nullptr){
        std::string op_log_level_str(env_ptr);
        if(op_log_level_str == "DEBUG")
            debug_log_activate = 1;
        else if(op_log_level_str == "INFO")
            debug_log_activate = 0;
        else if(op_log_level_str == ""){
            debug_log_activate = 0;
        }
    }
    if(debug_log_activate == 1){
        LOG(INFO) << "Before execute kernel[" << GetName() << "]";
        for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
            auto tensor = ctx->GetInput<TensorImpl>(i);
            if (!tensor) {
                continue;
            }
            auto tensor_size = tensor->GetShape()->CalcBytesIncludingPadding();
            auto tensor_dim_count = tensor->GetShape()->GetDimCount();
            std::string tensor_dims = "";
            for (uint32_t j = 0; j < tensor_dim_count; ++j) {
                tensor_dims += std::to_string(tensor->GetShape()->GetDim(j)) + " ";
            }
            LOG(DEBUG) << "input tensor " << i <<  " datatype: " << ppl::common::GetDataTypeStr(tensor->GetShape()->GetDataType()) << " dataformat "
                    << ppl::common::GetDataFormatStr(tensor->GetShape()->GetDataFormat());
            LOG(DEBUG) << "input tensor " << i <<  " dims: " << tensor_dims;
            LOG(DEBUG) << "input tensor " << i <<  " tensor_size: " << tensor_size << ",  data_ptr: " << tensor->GetBufferPtr();
            total_size += tensor_size;
        }
    }
    auto status = BeforeExecute(ctx);
    if (status != RC_SUCCESS) {
        return status;
    }
    if(debug_log_activate == 1){
//#ifndef NDEBUG
    //uint32_t total_size = 0;
    //LOG(INFO) << "Before execute kernel[" << GetName() << "]";
    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        auto tensor_size = tensor->GetShape()->CalcBytesIncludingPadding();
        auto tensor_dim_count = tensor->GetShape()->GetDimCount();
        std::string tensor_dims = "";
        for (uint32_t j = 0; j < tensor_dim_count; ++j) {
            tensor_dims += std::to_string(tensor->GetShape()->GetDim(j)) + " ";
        }
        LOG(DEBUG) << "output tensor " << i <<  " datatype: " << ppl::common::GetDataTypeStr(tensor->GetShape()->GetDataType()) << " dataformat "
                   << ppl::common::GetDataFormatStr(tensor->GetShape()->GetDataFormat());
        LOG(DEBUG) << "output tensor " << i <<  " dims: " << tensor_dims;
        LOG(DEBUG) << "output tensor " << i <<  " tensor_size: " << tensor_size << ",  data_ptr: " << tensor->GetBufferPtr();
        total_size += tensor_size;
    }
    run_begin_ts = std::chrono::system_clock::now();
    }
//#endif

    if (CanDoExecute(*ctx)) {
        status = DoExecute(ctx);
    }

    if(debug_log_activate == 1){
//#ifndef NDEBUG
    auto run_end_ts = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(run_end_ts - run_begin_ts);
    LOG(INFO) << "After execute kernel[" << GetName() << "] with running time " << (float)diff.count()
              << " ms and memory cost " << total_size;
    }
//#endif

    return status;
}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
void CudaKernel::GetProfilingInfo(InternalProfilingInfo* info) const {
    cudaEventSynchronize(exec_end_event_);
    float ms = 0.0;
    cudaEventElapsedTime(&ms, exec_begin_event_, exec_end_event_);
    info->exec_microseconds = static_cast<uint64_t>(ms * 1000);
}
#endif

}}} // namespace ppl::nn::cuda
