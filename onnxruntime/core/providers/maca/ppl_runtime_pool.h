// Copyright 2024 metax-tech.com Inc. All Rights Reserved.
#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <unordered_map>
#include <stack>
#include <queue>
#include "maca_execution_provider.h"
#include "maca_call.h"
#include <set>

namespace onnxruntime{

class PPLOnceAllocator;
class PPLRuntimeWrapper{
    public: 
    PPLRuntimeWrapper(size_t release_threshold, std::shared_ptr<::ppl::nn::Runtime> ppl_runtime, cudaStream_t stream): 
        release_threshold_(release_threshold), ppl_runtime_(ppl_runtime), stream_(stream){
            CUDA_CALL_THROW(cudaEventCreateWithFlags(&event_begin_, cudaEventDisableTiming));
            CUDA_CALL_THROW(cudaEventCreateWithFlags(&event_end_, cudaEventDisableTiming));
        }

    std::shared_ptr<::ppl::nn::Runtime> getPPLRuntime();
    bool isRelease(); //wheather release
    void setInPoolFlag(bool is_in_pool);
    void addUnuseNum(bool add_flag);
    cudaStream_t getStream() const { return stream_;}
    bool isInternalStream(cudaStream_t stream) const { return stream==stream_;}
    cudaEvent_t getBeginEvent() const {return event_begin_;}
    void insertBegineEvent(cudaStream_t stream);
    void insertEndEvent(cudaStream_t stream);
    cudaEvent_t getEndEvent() const {return event_end_;}
    friend class PPLOnceAllocator;
    private:
    void decreaseOrtTensorNum();
    void increaseOrtTensorNum();

    private:
    cudaEvent_t event_begin_;
    cudaEvent_t event_end_;
    cudaStream_t stream_;
    const size_t release_threshold_;
    std::shared_ptr<::ppl::nn::Runtime> ppl_runtime_;
    bool is_in_pool_ = false; 
    size_t unuse_num_ = 0; // decision release
    size_t ort_tensor_num_ = 0; // static ort tensor which use ppl allocator
    OrtMutex ppl_runtime_wrapper_mu_;
};

class PPLRuntimePool {
public:
    virtual std::shared_ptr<PPLRuntimeWrapper> allocatePPLRuntimeWrapper(cudaStream_t stream) = 0;
    virtual void recoverPPLRuntimeWrapper(cudaStream_t stream) = 0;
    virtual ~PPLRuntimePool(){}
};

// Use ort stream to create ppl runtime obj.
class ORTStreamPPLRuntimePool : public PPLRuntimePool{
public:
    ORTStreamPPLRuntimePool(std::shared_ptr<::ppl::nn::onnx::RuntimeBuilder> builder, std::queue<cudaStream_t>& stream_queue);
    ~ORTStreamPPLRuntimePool(){}
    std::shared_ptr<PPLRuntimeWrapper> allocatePPLRuntimeWrapper(cudaStream_t stream) override;
    void recoverPPLRuntimeWrapper(cudaStream_t stream) override;
private:
    std::unordered_map<cudaStream_t, std::shared_ptr<PPLRuntimeWrapper>> stream_map_ppl_runtime_wrapper_;
    std::shared_ptr<::ppl::nn::onnx::RuntimeBuilder> builder_;
};

// Use ppl stream to create ppl runtime obj. 
class PPLStreamPPLRuntimePool : public PPLRuntimePool{
public:
    PPLStreamPPLRuntimePool(std::shared_ptr<::ppl::nn::onnx::RuntimeBuilder> builder,size_t max_pool_size=1, size_t shrink_threshold=1000);
    ~PPLStreamPPLRuntimePool(){}
    std::shared_ptr<PPLRuntimeWrapper> allocatePPLRuntimeWrapper(cudaStream_t stream) override;
    void recoverPPLRuntimeWrapper(cudaStream_t stream) override;
private:
    std::set<std::shared_ptr<PPLRuntimeWrapper>> ppl_runtime_wrapper_set_;
    std::unordered_map<cudaStream_t, std::shared_ptr<PPLRuntimeWrapper>> stage_ppl_runtime_wrapper_;
    std::shared_ptr<::ppl::nn::onnx::RuntimeBuilder> builder_;
    const size_t max_pool_size_;
    size_t shrink_threshold_;
    OrtMutex ppl_runtime_mu_;
    size_t current_pool_size_ = 1;
    size_t select_mode_ = 0;
};



}

