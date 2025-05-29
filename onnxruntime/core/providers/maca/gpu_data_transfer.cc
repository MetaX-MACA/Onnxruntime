// Copyright 2023 metax-tech.com Inc.

#include "core/graph/constants.h"
#include "core/providers/maca/gpu_data_transfer.h"
#include "maca_common.h"
#include "core/framework/tensor.h"
#include "core/framework/ortdevice.h"

#include "core/common/logging/logging.h"
#include "core/common/common.h"

// use default stream for copy for now, to avoid racing in BFC arena as in issue #4829
// note this may cause some models to run slower if there are ops running on CPU
// so we leave it as optional, in case user need the previous behavior
// a full fix to BFC arena is being looked at, and once it's in, we can revert this change
namespace onnxruntime {
GPUDataTransfer::GPUDataTransfer(cudaStream_t stream, bool do_copy_in_default_stream ,  std::shared_ptr<std::unordered_map<std::thread::id, void*>> thread_map_stream,std::shared_ptr<OrtMutex> stream_mutex) {
  thread_map_stream_ = thread_map_stream;
  stream_mutex_ = stream_mutex;
  // create streams, default is nullptr
  do_copy_in_default_stream_ = do_copy_in_default_stream;
  
  streams_[kCudaStreamDefault] = stream;
  if (do_copy_in_default_stream) {
    streams_[kCudaStreamCopyIn] = stream;
    streams_[kCudaStreamCopyOut] = stream;
  } else {
    CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyIn], cudaStreamNonBlocking));
    CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyOut], cudaStreamNonBlocking));
  }
  
}

GPUDataTransfer::~GPUDataTransfer() {
  if (!do_copy_in_default_stream_ && streams_[kCudaStreamCopyIn] != nullptr) {
    CUDA_CALL(cudaStreamDestroy(streams_[kCudaStreamCopyIn]));
  }
  if (!do_copy_in_default_stream_ && streams_[kCudaStreamCopyOut] != nullptr) {
    CUDA_CALL(cudaStreamDestroy(streams_[kCudaStreamCopyOut]));
  }
}

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::GPU || src_device.MemType() == OrtDevice::MemType::CUDA_PINNED ||
         dst_device.Type() == OrtDevice::GPU || dst_device.MemType() == OrtDevice::MemType::CUDA_PINNED;
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;
  LOGS_DEFAULT(INFO) << "MacaDataTransfer memcpy from: " << src_data << " to: " << dst_data;
  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::CPU && src_device.MemType() == OrtDevice::MemType::CUDA_PINNED) {
      // copy from pinned memory to GPU, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, GetStream(exec_queue_id)));
      if(!thread_map_stream_)
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(GetStream(exec_queue_id)));
    } else if (src_device.Type() == OrtDevice::GPU) {
      // copying between GPU, this is non-blocking
      // Copy only if the two addresses are different.
      if (dst_data != src_data) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, GetStream(kCudaStreamDefault)));
        if(!thread_map_stream_)
          CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(GetStream(kCudaStreamDefault)));
      }
    } else {
      // copy from other CPU memory to GPU, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, GetStream(kCudaStreamDefault)));
      if(!thread_map_stream_)
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(GetStream(kCudaStreamDefault)));
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    if (dst_device.Type() == OrtDevice::CPU && dst_device.MemType() == OrtDevice::MemType::CUDA_PINNED) {
      // copying from GPU to pinned memory, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, GetStream(exec_queue_id)));
      if(!thread_map_stream_)
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(GetStream(exec_queue_id)));
    } else {
      // copying from GPU to CPU memory, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, GetStream(kCudaStreamDefault)));
      if(!thread_map_stream_)
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(GetStream(kCudaStreamDefault)));

    }
  } else {
    // copying between cpu memory
    memcpy(dst_data, src_data, bytes);
  }
  LOGS_DEFAULT(INFO) << "MacaDataTransfer memcpy from: " << src_data << " to: " << dst_data  << "finished";
  return Status::OK();
}
}  // namespace onnxruntime
