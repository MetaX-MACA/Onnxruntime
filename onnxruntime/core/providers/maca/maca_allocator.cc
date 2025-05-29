// Copyright 2023 metax-tech.com Inc.

#include "maca_allocator.h"
#include "maca_common.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/session_state.h"
#include "maca_fence.h"
#include "gpu_data_transfer.h"
#include "maca_call.h"
#include "ppl_runtime_pool.h"

namespace onnxruntime {

static const GPUDataTransfer* GetGPUDataTransfer(const SessionState* session_state) {
  OrtDevice gpu_device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0);
  OrtDevice cpu_device;
  return static_cast<const GPUDataTransfer*>(session_state->GetDataTransferMgr().GetDataTransfer(gpu_device, cpu_device));
}

void CUDAAllocator::CheckDevice(bool throw_when_fail) const {
// #ifndef NDEBUG
//   // check device to match at debug build
//   // if it's expected to change, call cudaSetDevice instead of the check
//   int current_device;
//   auto cuda_err = cudaGetDevice(&current_device);
//   if (cuda_err == cudaSuccess) {
//     ORT_ENFORCE(current_device == Info().id);
//   } else if (throw_when_fail) {
//     CUDA_CALL_THROW(cuda_err);
//   }
// #else
//   ORT_UNUSED_PARAMETER(throw_when_fail);
// #endif
}

void CUDAAllocator::SetDevice(bool throw_when_fail) const {
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    int allocator_device_id = Info().id;
    if (current_device != allocator_device_id) {
      cuda_err = cudaSetDevice(allocator_device_id);
    }
  }

  if (cuda_err != cudaSuccess && throw_when_fail) {
    CUDA_CALL_THROW(cuda_err);
  }
}

void* CUDAAllocator::Alloc(size_t size) {
  SetDevice(true);
  CheckDevice(true);
  void* p = nullptr;
  if (size > 0) {
    //BFCArena was updated recently to handle the exception and adjust the request size
    CUDA_CALL_THROW(cudaMalloc((void**)&p, size));
  }
  return p;
}

void CUDAAllocator::Free(void* p) {
  SetDevice(false);
  CheckDevice(false);  // ignore CUDA failure when free
  cudaFree(p);         // do not throw error since it's OK for cudaFree to fail during shutdown
}

void* CUDAExternalAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = alloc_(size);

    // review(codemzs): ORT_ENFORCE does not seem appropiate.
    ORT_ENFORCE(p != nullptr);
  }

  return p;
}

void CUDAExternalAllocator::Free(void* p) {
  free_(p);
  std::lock_guard<OrtMutex> lock(lock_);
  auto it = reserved_.find(p);
  if (it != reserved_.end()) {
    reserved_.erase(it);
    if (empty_cache_) empty_cache_();
  }
}

void* CUDAExternalAllocator::Reserve(size_t size) {
  void* p = Alloc(size);
  if (!p) return nullptr;
  std::lock_guard<OrtMutex> lock(lock_);
  ORT_ENFORCE(reserved_.find(p) == reserved_.end());
  reserved_.insert(p);
  return p;
}

FencePtr CUDAAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CUDAFence>(GetGPUDataTransfer(session_state));
}

void* CUDAPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    CUDA_CALL_THROW(cudaMallocHost((void**)&p, size));
  }
  return p;
}

void CUDAPinnedAllocator::Free(void* p) {
  CUDA_CALL_THROW(cudaFreeHost(p));
}

FencePtr CUDAPinnedAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CUDAFence>(GetGPUDataTransfer(session_state));
}


void* PPLOnceAllocator::Alloc(size_t size){
  if(alloc_){
    ORT_THROW("PPLOnceAllocator only allow Alloc once.");
  }
  if(!ppl_runtime_wrapper_){
    ORT_THROW("ppl_runtime_wrapper is nullptr");
  }
  
  // if(buffer_desc_.desc != size){
  //   ORT_THROW("PPLOnceAllocator alloc size not equal bufferDesc");
  // }
  // LOGS_DEFAULT(WARNING) << "Alloc memory " << size;
  alloc_ = true;
  return buffer_desc_.addr;
};
void PPLOnceAllocator::Free(void* p){
  // LOGS_DEFAULT(WARNING) << "Free memory " ;
  if(!free_){
    if(buffer_desc_.addr != p){
      ORT_THROW("PPLOnceAllocator: ptr not belong to PPLOnceAllocator, can't free!");
    }
    device_->Free(&buffer_desc_);
    free_ = true;
    if(ppl_runtime_wrapper_){
      ppl_runtime_wrapper_->decreaseOrtTensorNum();
      ppl_runtime_wrapper_ = nullptr;
    }
  }
};
void PPLOnceAllocator::init(){
    if(!ppl_runtime_wrapper_){
      ORT_THROW("ppl_runtime_wrapper is nullptr");
    } 
    ppl_runtime_wrapper_->increaseOrtTensorNum();
};
PPLOnceAllocator::~PPLOnceAllocator(){
  if(!free_){
    device_->Free(&buffer_desc_);
  }
  if(ppl_runtime_wrapper_){
    ppl_runtime_wrapper_->decreaseOrtTensorNum();
  }
};
}  // namespace onnxruntime
