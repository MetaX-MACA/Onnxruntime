// Copyright 2023 metax-tech.com Inc.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "core/framework/data_transfer.h"
#include "core/common/common.h"
#include "core/platform/ort_mutex.h"

#include <thread>
#include <memory>
#include <queue>
#include <unordered_map>
namespace onnxruntime {

enum CUDAStreamType : int {
  kCudaStreamDefault = 0,
  kCudaStreamCopyIn,
  kCudaStreamCopyOut,
  kTotalCudaStreams,
};

class GPUDataTransfer : public IDataTransfer {
 public:
  GPUDataTransfer(cudaStream_t stream, bool do_copy_in_default_stream = true, std::shared_ptr<std::unordered_map<std::thread::id, void*>> thread_map_stream = nullptr, std::shared_ptr<OrtMutex> stream_mutex = nullptr);
  ~GPUDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  // Dumpen MSVC warning about not fully overriding
  using IDataTransfer::CopyTensor;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;

  cudaStream_t GetStream(int queue_id) const {
    if(!thread_map_stream_){
      ORT_ENFORCE(queue_id >= 0 && queue_id < kTotalCudaStreams);
      return streams_[queue_id];
    }else{
      void* get_stream = nullptr;
      std::thread::id thread_id = std::this_thread::get_id();
      std::lock_guard<OrtMutex> lock(*stream_mutex_);
      get_stream = (*thread_map_stream_)[thread_id];
      return reinterpret_cast<cudaStream_t>(get_stream);
    }


  }

 private:
  bool do_copy_in_default_stream_;
  cudaStream_t streams_[kTotalCudaStreams];
  std::shared_ptr<OrtMutex> stream_mutex_;
  std::shared_ptr<std::unordered_map<std::thread::id, void*>> thread_map_stream_;
};

}  // namespace onnxruntime
