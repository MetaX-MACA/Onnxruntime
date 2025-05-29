// Copyright 2023 metax-tech.com Inc.

#include <string.h>
#include <memory>
#include <chrono>
#include "core/providers/providers.h"
// #include "core/providers/shared_library/provider_api.h"
#include "core/providers/shared_library/provider_host_api.h"
#include "core/providers/maca/maca_provider_factory.h"
#include "maca_execution_provider.h"
// #include "core/framework/provider_options.h"
#include "core/session/abi_session_options_impl.h"
#include "gsl/gsl"
#include "core/providers/maca/maca_allocator.h"
#include "core/providers/maca/gpu_data_transfer.h"
#include "maca_call.h"

using namespace onnxruntime;

namespace onnxruntime {

struct MACAProviderFactory : IExecutionProviderFactory {
  MACAProviderFactory(const OrtMACAProviderOptions& info) : info_{info} {}
  ~MACAProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  OrtMACAProviderOptions info_;
};

std::unique_ptr<IExecutionProvider> MACAProviderFactory::CreateProvider() {
  return std::make_unique<MACAExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MACA(const OrtMACAProviderOptions& info) {
  return std::make_shared<onnxruntime::MACAProviderFactory>(info);
}

struct ProviderInfo_MACA_Impl : ProviderInfo_MACA {
  OrtStatus* SetCurrentGpuDeviceId(_In_ int device_id) override {
    int num_devices;
    auto cuda_err = ::cudaGetDeviceCount(&num_devices);
    if (cuda_err != cudaSuccess) {
      LOGS_DEFAULT(ERROR) << "Failed to set device id since cudaGetDeviceCount failed.";
      return nullptr;
      // return CreateStatus(ORT_FAIL, "Failed to set device id since cudaGetDeviceCount failed.");
    }

    if (device_id >= num_devices) {
      // std::ostringstream ostr;
      // ostr << "Invalid device id. Device id should be less than total number of devices (" << num_devices << ")";
      // return CreateStatus(ORT_INVALID_ARGUMENT, ostr.str().c_str());
      LOGS_DEFAULT(ERROR) <<  "Invalid device id. Device id should be less than total number of devices";
      return nullptr;
    }

    cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      // return CreateStatus(ORT_FAIL, "Failed to set device id.");
      LOGS_DEFAULT(ERROR) <<  "Failed to set device id.";
      return nullptr;
    }
    return nullptr;
  }

  OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) override {
    auto cuda_err = cudaGetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      LOGS_DEFAULT(ERROR) << "Failed to get device id.";
      // return CreateStatus(ORT_FAIL, "Failed to get device id.");
    }
    return nullptr;
  }

  std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) override {
    return std::make_unique<CUDAAllocator>(device_id, name);
  }

  std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) override {
    return std::make_unique<CUDAPinnedAllocator>(device_id, name);
  }

  std::unique_ptr<IDataTransfer> CreateGPUDataTransfer(void* stream) override {
    return std::make_unique<GPUDataTransfer>(static_cast<cudaStream_t>(stream));
  }

  // void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) override {
  //   return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  // }

  // void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) override {
  //   return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  // }

  // void cuda__Impl_Cast(void* stream, const double* input_data, float* output_data, size_t count) override {
  //   return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  // }

  // void cuda__Impl_Cast(void* stream, const float* input_data, double* output_data, size_t count) override {
  //   return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  // }

  // bool CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) override { return CudaCall<cudaError, false>(cudaError(retCode), exprString, libName, cudaError(successCode), msg); }
  // bool CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) override { return CudaCall<cudaError, true>(cudaError(retCode), exprString, libName, cudaError(successCode), msg); }

  void CopyGpuToCpu(void* dst_ptr, const void* src_ptr, const size_t size, const OrtMemoryInfo& dst_location, const OrtMemoryInfo& src_location) override {
    ORT_ENFORCE(dst_location.device.Type() == OrtDevice::CPU);

    // Current CUDA device.
    int device;
    CUDA_CALL(cudaGetDevice(&device));

    if (device != src_location.id) {
      // Need to switch to the allocating device.
      CUDA_CALL(cudaSetDevice(src_location.id));
      // Copy from GPU to CPU.
      CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
      // Switch back to current device.
      CUDA_CALL(cudaSetDevice(device));
    } else {
      // Copy from GPU to CPU.
      CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
    }
  }

  // Used by slice_concatenate_test.cc and onnxruntime_pybind_state.cc

  void cudaMemcpy_HostToDevice(void* dst, const void* src, size_t count) override {
    // cudaMemcpy() operates on the default stream
    CUDA_CALL_THROW(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));

    // To ensure that the copy has completed, invoke a stream sync for the default stream.
    // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
    // For transfers from pageable host memory to device memory, a stream sync is performed before the copy is initiated.
    // The function will return once the pageable buffer has been copied to the staging memory for DMA transfer
    // to device memory, but the DMA to final destination may not have completed.

    CUDA_CALL_THROW(cudaStreamSynchronize(0));
  }

  // Used by onnxruntime_pybind_state.cc
  void cudaMemcpy_DeviceToHost(void* dst, const void* src, size_t count) override {
    // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
    // For transfers from device to either pageable or pinned host memory, the function returns only once the copy has completed.
    CUDA_CALL_THROW(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
  }

  int macaGetDeviceCount() override {
    int num_devices = 0;
    CUDA_CALL_THROW(::cudaGetDeviceCount(&num_devices));
    return num_devices;
  }

  void MACAExecutionProviderInfo__FromProviderOptions(const ProviderOptions& options, MACAExecutionProviderInfo& info) override {
    info = MACAExecutionProviderInfo::FromProviderOptions(options);
  }

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
  cuda::INcclService& GetINcclService() override {
    return cuda::GetINcclService();
  }
#endif

#ifdef ENABLE_NVTX_PROFILE
  void NvtxRangeCreator__BeginImpl(profile::NvtxRangeCreator* p) override { p->BeginImpl(); }
  void NvtxRangeCreator__EndImpl(profile::NvtxRangeCreator* p) override { p->EndImpl(); }
#endif

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const OrtMACAProviderOptions& info) override {
    return std::make_shared<MACAProviderFactory>(info);
  }

  std::shared_ptr<IAllocator> CreateCudaAllocator(int16_t device_id, size_t gpu_mem_limit, int arena_extend_strategy,
                // onnxruntime::CUDAExecutionProviderExternalAllocatorInfo& external_allocator_info,
                OrtArenaCfg* default_memory_arena_cfg) override {
    return MACAExecutionProvider::CreateCudaAllocator(device_id, gpu_mem_limit, arena_extend_strategy,
                                      // external_allocator_info,
                                      default_memory_arena_cfg);
  }

} g_info;

struct MACA_Provider : Provider {
  void* GetInfo() override { return &g_info; }
  void Initialize() override {}
  void Shutdown() override {}
} g_provider;

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_MACA,
                    _In_ OrtSessionOptions* options, const OrtMACAProviderOptions* maca_options) {
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_MACA(*maca_options)
  );
  return nullptr;
}

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
