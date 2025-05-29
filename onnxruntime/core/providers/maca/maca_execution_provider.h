// Copyright 2023 metax-tech.con Inc.

#pragma once
#include <thread>
#include <queue>

#include "core/framework/allocator.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"

#include "core/providers/maca/maca_allocator.h"
#include "core/providers/maca/gpu_data_transfer.h"
#include "core/providers/maca/maca_execution_provider_info.h"

#include "onnx_converter.h"

#include "ppl/nn/models/onnx/runtime_builder_factory.h"
#include "ppl/nn/engines/x86/engine_factory.h"
#include "ppl/nn/engines/cuda/engine_factory.h"
#include "ppl/nn/engines/cuda/options.h"
#include "ppl/nn/utils/array.h"
#include "ppl/nn/quantization/quant_param_info.h"
#include "ppl/common/file_mapping.h"

namespace onnxruntime {

class MACAExecutionProvider : public IExecutionProvider {
 public:
  explicit MACAExecutionProvider(const OrtMACAProviderOptions& info);
  virtual  ~MACAExecutionProvider();

  Status SetComputeStream(void* stream) override;

  void* GetComputeStream() const override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const GraphViewer& graph,
                        const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                        std::vector<NodeComputeInfo>& node_compute_funcs) override;
  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;
  ProviderOptions GetProviderOptions() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

  void RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) override;

  static AllocatorPtr CreateCudaAllocator(OrtDevice::DeviceId device_id, size_t cuda_mem_limit,
                                        int arena_extend_strategy,
                                        // CUDAExecutionProviderExternalAllocatorInfo external_alloc_info,
                                        OrtArenaCfg* arena_cfg);
  
  /**
     Blocks until the device has completed all preceding requested tasks.
     Currently this is primarily used by the IOBinding object to ensure that all
     inputs have been copied to the device before execution begins.
  */
  common::Status Sync() const override  { return Status::OK(); }

  /**
     Called when InferenceSession::Run started
     NOTE that due to async execution in provider, the actual work of previous
     Run may not be finished on device This function should be regarded as the
     point after which a new Run would start to submit commands from CPU
  */
  common::Status OnRunStart() override  ;

  /**
     Called when InferenceSession::Run ended
     NOTE that due to async execution in provider, the actual work of this Run
     may not be finished on device This function should be regarded as the point
     that all commands of current Run has been submmited by CPU
  */
  common::Status OnRunEnd(bool sync_stream) override;
           
//   virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  ::ppl::nn::QuantParamInfo quant_param_info_;
  bool quantizer_ = false;
  int fuse_rules_;
  mutable char model_path_[4096];  // Reserved for max path length
  mutable char model_file_dir_[4096];  // Reserved for max path length
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, size_t>>> input_info_;
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, size_t>>> output_info_;
  cudaDeviceProp device_prop_;
  OrtMACAProviderOptions info_;
  bool external_stream_ = false;
  cudaStream_t stream_ = nullptr;
  AllocatorPtr allocator_;
  std::unordered_map<std::string, ONNX_NAMESPACE::ModelProto> model_proto_;
  std::vector<std::vector<int>> GetSupportedNodes(const ONNX_NAMESPACE::ModelProto& model_proto) const;
  OnnxConverter converter_;

  // Select stream for each run 
  std::shared_ptr<std::unordered_map<std::thread::id, void*>> thread_map_stream_ = std::make_shared<std::unordered_map<std::thread::id, void*>>();
  std::queue<cudaStream_t> stream_queue_;

  std::shared_ptr<OrtMutex> stream_mutex_ = std::make_shared<OrtMutex>();
  std::shared_ptr<std::unordered_map<std::string, OrtDataLayout>> tensor_data_layout_ = std::make_shared<std::unordered_map<std::string, OrtDataLayout>>();

  private:
// use for enable_profiling
  double run_dur_ = 0;
  uint32_t run_count_ = 0;
};

}  // namespace onnxruntime
