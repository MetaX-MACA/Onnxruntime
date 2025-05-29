// Copyright 2023 metax-tech.com Inc.

#pragma once

#include <limits>

#include "core/framework/arena_extend_strategy.h"
#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

namespace maca {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kMemLimit = "gpu_mem_limit";
constexpr const char* kArenaExtendStrategy = "arena_extend_strategy";
constexpr const char* kCudnnConvAlgoSearch = "cudnn_conv_algo_search";
constexpr const char* kDoCopyInDefaultStream = "do_copy_in_default_stream";
constexpr const char* kGpuExternalAlloc = "gpu_external_alloc";
constexpr const char* kGpuExternalFree = "gpu_external_free";
constexpr const char* kGpuExternalEmptyCache = "gpu_external_empty_cache";
constexpr const char* kCudnnConvUseMaxWorkspace = "cudnn_conv_use_max_workspace";
constexpr const char* kEnableCudaGraph = "enable_cuda_graph";
constexpr const char* kCudnnConv1dPadToNc1d = "cudnn_conv1d_pad_to_nc1d";
}  // namespace provider_option_names
}  // namespace cuda

// Information needed to construct CUDA execution providers.
struct MACAExecutionProviderExternalAllocatorInfo {
  void* alloc{nullptr};
  void* free{nullptr};
  void* empty_cache{nullptr};

  MACAExecutionProviderExternalAllocatorInfo() {
    alloc = nullptr;
    free = nullptr;
    empty_cache = nullptr;
  }

  MACAExecutionProviderExternalAllocatorInfo(void* a, void* f, void* e) {
    alloc = a;
    free = f;
    empty_cache = e;
  }

  bool UseExternalAllocator() const {
    return (alloc != nullptr) && (free != nullptr);
  }
};

struct MACAExecutionProviderInfo {
  OrtDevice::DeviceId device_id{0};
  size_t gpu_mem_limit{std::numeric_limits<size_t>::max()};                         // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)
  ArenaExtendStrategy arena_extend_strategy{ArenaExtendStrategy::kNextPowerOfTwo};  // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)
  OrtCudnnConvAlgoSearch cudnn_conv_algo_search{OrtCudnnConvAlgoSearchExhaustive};
  bool do_copy_in_default_stream{true};
  bool has_user_compute_stream{false};
  void* user_compute_stream{nullptr};
  // The following OrtArenaCfg instance only characterizes the behavior of the default memory
  // arena allocator and not any other auxiliary allocator that may also be part of the CUDA EP.
  // For example, auxiliary allocators `CUDA_PINNED` and `CUDA_CPU` will not be configured using this
  // arena config.
  OrtArenaCfg* default_memory_arena_cfg{nullptr};
  MACAExecutionProviderExternalAllocatorInfo external_allocator_info{};
  // By default use fix workspace size (32M) for Conv algo search, the final algo might not be the best.
  // If set to true, try to use as much as possible memory for algo search.
  bool cudnn_conv_use_max_workspace{false};

  bool enable_cuda_graph{false};

  // By default, for Conv1D, will pad [N,C,D] to [N,C,D,1], if turn on, will pad to [N,C,1,D].
  bool cudnn_conv1d_pad_to_nc1d{false};

  static MACAExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const MACAExecutionProviderInfo& info);
  // static ProviderOptions ToProviderOptions(const OrtCUDAProviderOptionsV2& info);
};
}  // namespace onnxruntime
