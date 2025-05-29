// Copyright 2023 metax-tech.com Inc.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"
#include "core/platform/ort_mutex.h"

#include "ppl/nn/models/onnx/runtime_builder_factory.h"
#include "ppl/nn/engines/x86/engine_factory.h"
#include "ppl/nn/engines/cuda/engine_factory.h"
#include "ppl/nn/engines/cuda/options.h"
#include "ppl/nn/utils/array.h"
#include "ppl/nn/quantization/quant_param_info.h"
#include "ppl/common/file_mapping.h"
#include "ppl/nn/common/buffer_desc.h"
#include "ppl/nn/common/device.h"



namespace onnxruntime {

class CUDAAllocator : public IAllocator {
 public:
  CUDAAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id),
                          device_id, OrtMemTypeDefault)) {}
  void* Alloc(size_t size) override;
  void Free(void* p) override;
  FencePtr CreateFence(const SessionState* session_state) override;

 private:
  void CheckDevice(bool throw_when_fail) const;
  void SetDevice(bool throw_when_fail) const;
};

class CUDAExternalAllocator : public CUDAAllocator {
  typedef void* (*ExternalAlloc)(size_t size);
  typedef void (*ExternalFree)(void* p);
  typedef void (*ExternalEmptyCache)();

 public:
  CUDAExternalAllocator(OrtDevice::DeviceId device_id, const char* name, void* alloc, void* free, void* empty_cache)
      : CUDAAllocator(device_id, name) {
    alloc_ = reinterpret_cast<ExternalAlloc>(alloc);
    free_ = reinterpret_cast<ExternalFree>(free);
    empty_cache_ = reinterpret_cast<ExternalEmptyCache>(empty_cache);
  }

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  void* Reserve(size_t size) override;

 private:
  mutable OrtMutex lock_;
  ExternalAlloc alloc_;
  ExternalFree free_;
  ExternalEmptyCache empty_cache_;
  InlinedHashSet<void*> reserved_;
};

//TODO: add a default constructor
class CUDAPinnedAllocator : public IAllocator {
 public:
  CUDAPinnedAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, device_id),
                          device_id, OrtMemTypeCPUOutput)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  FencePtr CreateFence(const SessionState* session_state) override;
};

class PPLRuntimeWrapper;
class PPLOnceAllocator : public IAllocator{
  public:
    PPLOnceAllocator(OrtDevice::DeviceId device_id, ppl::nn::Device* device_context, ppl::nn::BufferDesc buffer_desc, std::shared_ptr<PPLRuntimeWrapper> ppl_runtime_wrapper)
      : IAllocator(
            OrtMemoryInfo("PPL_ALLOCATOR", OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id),
                          device_id, OrtMemTypeDefault)),device_(device_context),buffer_desc_(buffer_desc),ppl_runtime_wrapper_(ppl_runtime_wrapper) {
      init();
    }
    void init();
    void* Alloc(size_t size) override;
    void Free(void* p) override;
    ~PPLOnceAllocator();
  private:
    ppl::nn::Device* device_;
    ppl::nn::BufferDesc buffer_desc_;
    std::shared_ptr<PPLRuntimeWrapper> ppl_runtime_wrapper_;
    bool alloc_ = false;
    bool free_ = false;
};
}  // namespace onnxruntime
