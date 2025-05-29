// Copyright 2023 metax-tech.com Inc.

#include "core/providers/maca/maca_execution_provider_info.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "maca_common.h"

namespace onnxruntime {


// namespace {
// const DeleteOnUnloadPtr<EnumNameMapping<OrtCudnnConvAlgoSearch>> ort_cudnn_conv_algo_search_mapping = new EnumNameMapping<OrtCudnnConvAlgoSearch>{
//     {OrtCudnnConvAlgoSearchExhaustive, "EXHAUSTIVE"},
//     {OrtCudnnConvAlgoSearchHeuristic, "HEURISTIC"},
//     {OrtCudnnConvAlgoSearchDefault, "DEFAULT"},
// };

// const DeleteOnUnloadPtr<EnumNameMapping<ArenaExtendStrategy>> arena_extend_strategy_mapping = new EnumNameMapping<ArenaExtendStrategy>{
//     {ArenaExtendStrategy::kNextPowerOfTwo, "kNextPowerOfTwo"},
//     {ArenaExtendStrategy::kSameAsRequested, "kSameAsRequested"},
// };
// }  // namespace

MACAExecutionProviderInfo MACAExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  MACAExecutionProviderInfo info{};
  void* alloc = nullptr;
  void* free = nullptr;
  void* empty_cache = nullptr;
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              maca::provider_option_names::kDeviceId,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                int num_devices{};
                ORT_RETURN_IF_NOT(
                    CUDA_CALL(cudaGetDeviceCount(&num_devices)),
                    "cudaGetDeviceCount() failed.");
                ORT_RETURN_IF_NOT(
                    0 <= info.device_id && info.device_id < num_devices,
                    "Invalid device ID: ", info.device_id,
                    ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                return Status::OK();
              })
          .AddValueParser(
              maca::provider_option_names::kGpuExternalAlloc,
              [&alloc](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                alloc = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              maca::provider_option_names::kGpuExternalFree,
              [&free](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                free = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              maca::provider_option_names::kGpuExternalEmptyCache,
              [&empty_cache](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                empty_cache = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(maca::provider_option_names::kMemLimit, info.gpu_mem_limit)
          // .AddAssignmentToEnumReference(
          //     maca::provider_option_names::kArenaExtendStrategy,
          //     *arena_extend_strategy_mapping, info.arena_extend_strategy)
          // .AddAssignmentToEnumReference(
          //     maca::provider_option_names::kCudnnConvAlgoSearch,
          //     *ort_cudnn_conv_algo_search_mapping, info.cudnn_conv_algo_search)
          .AddAssignmentToReference(maca::provider_option_names::kDoCopyInDefaultStream, info.do_copy_in_default_stream)
          .AddAssignmentToReference(maca::provider_option_names::kCudnnConvUseMaxWorkspace, info.cudnn_conv_use_max_workspace)
          .AddAssignmentToReference(maca::provider_option_names::kEnableCudaGraph, info.enable_cuda_graph)
          .AddAssignmentToReference(maca::provider_option_names::kCudnnConv1dPadToNc1d, info.cudnn_conv1d_pad_to_nc1d)
          .Parse(options));

  MACAExecutionProviderExternalAllocatorInfo alloc_info{alloc, free, empty_cache};
  info.external_allocator_info = alloc_info;
  return info;
}

ProviderOptions MACAExecutionProviderInfo::ToProviderOptions(const MACAExecutionProviderInfo& info) {
  const ProviderOptions options{
      {maca::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
  };
  return options;
}

// ProviderOptions MACAExecutionProviderInfo::ToProviderOptions(const OrtCUDAProviderOptionsV2& info) {
//   const ProviderOptions options{
//       {maca::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
//       {maca::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.gpu_mem_limit)},
//       {maca::provider_option_names::kArenaExtendStrategy, EnumToName(*arena_extend_strategy_mapping, info.arena_extend_strategy)},
//       {maca::provider_option_names::kCudnnConvAlgoSearch, EnumToName(*ort_cudnn_conv_algo_search_mapping, info.cudnn_conv_algo_search)},
//       {maca::provider_option_names::kDoCopyInDefaultStream, MakeStringWithClassicLocale(info.do_copy_in_default_stream)},
//       {maca::provider_option_names::kCudnnConvUseMaxWorkspace, MakeStringWithClassicLocale(info.cudnn_conv_use_max_workspace)},
//       {maca::provider_option_names::kCudnnConv1dPadToNc1d, MakeStringWithClassicLocale(info.cudnn_conv1d_pad_to_nc1d)}
//   };
//
//   return options;
// }

}  // namespace onnxruntime
