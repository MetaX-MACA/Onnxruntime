// Copyright 2024 metax-tech.com Inc. All Rights Reserved.

#include "ppl_runtime_pool.h"
#include "ppl/nn/common/device_context.h"

namespace onnxruntime{

// ==============================================PPLRuntimeWrapper=======================================================
std::shared_ptr<::ppl::nn::Runtime> PPLRuntimeWrapper::getPPLRuntime(){
    std::lock_guard<OrtMutex> lock(ppl_runtime_wrapper_mu_);
    if (is_in_pool_){
        ORT_THROW("PPLRuntimeWrapper is in PPLRuntimePool, cann't getPPLRuntime()");
    }
    return ppl_runtime_;
}

void PPLRuntimeWrapper::setInPoolFlag(bool is_in_pool) {
    std::lock_guard<OrtMutex> lock(ppl_runtime_wrapper_mu_);
    is_in_pool_ = is_in_pool;
}
void PPLRuntimeWrapper::decreaseOrtTensorNum(){
    std::lock_guard<OrtMutex> lock(ppl_runtime_wrapper_mu_);
    ort_tensor_num_--;
}
void PPLRuntimeWrapper::increaseOrtTensorNum(){
    std::lock_guard<OrtMutex> lock(ppl_runtime_wrapper_mu_);
    ort_tensor_num_++;
}
bool PPLRuntimeWrapper::isRelease(){
    std::lock_guard<OrtMutex> lock(ppl_runtime_wrapper_mu_);
    return is_in_pool_ && (!ort_tensor_num_) && (unuse_num_ > release_threshold_);
} //wheather release
void PPLRuntimeWrapper::addUnuseNum(bool add_flag){
    std::lock_guard<OrtMutex> lock(ppl_runtime_wrapper_mu_);
    if(add_flag){
        unuse_num_++;
    } else{
        unuse_num_=0;
    }
}
void PPLRuntimeWrapper::insertBegineEvent(cudaStream_t stream){
    if(isInternalStream(stream) == false){
        CUDA_CALL_THROW(cudaEventRecord(event_begin_, stream));
        CUDA_CALL_THROW(cudaStreamWaitEvent(stream_,event_begin_));
    }
};
void PPLRuntimeWrapper::insertEndEvent(cudaStream_t stream){
    if(isInternalStream(stream) == false){
        CUDA_CALL_THROW(cudaEventRecord(event_begin_, stream_));
        CUDA_CALL_THROW(cudaStreamWaitEvent(stream, event_begin_));
    }
};

// ==============================================ORTStreamPPLRuntimePool=======================================================
ORTStreamPPLRuntimePool::ORTStreamPPLRuntimePool(std::shared_ptr<::ppl::nn::onnx::RuntimeBuilder> builder,std::queue<cudaStream_t>& stream_queue)
                    :builder_(builder) {
    size_t stream_num = stream_queue.size();
    for(size_t i=0; i<stream_num ; i++){
        cudaStream_t stream = stream_queue.front();
        auto ppl_runtime_ptr = std::shared_ptr<::ppl::nn::Runtime>(builder_->CreateRuntime());
        stream_map_ppl_runtime_wrapper_.insert({stream,std::make_shared<PPLRuntimeWrapper>(10000, ppl_runtime_ptr, stream)});
    }
};
std::shared_ptr<PPLRuntimeWrapper> ORTStreamPPLRuntimePool::allocatePPLRuntimeWrapper(cudaStream_t stream){
    if(stream_map_ppl_runtime_wrapper_.count(stream) == 0){
        ORT_THROW("Error stream not in pool runtime.");
    }
    return stream_map_ppl_runtime_wrapper_[stream];
};
void ORTStreamPPLRuntimePool::recoverPPLRuntimeWrapper(cudaStream_t stream){};

// ==============================================PPLStreamPPLRuntimePool=======================================================
PPLStreamPPLRuntimePool::PPLStreamPPLRuntimePool(std::shared_ptr<::ppl::nn::onnx::RuntimeBuilder> builder,size_t max_pool_size, size_t shrink_threshold)
                    :builder_(builder),max_pool_size_(max_pool_size),shrink_threshold_(shrink_threshold) {
    for(size_t i=0; i<1; i++){
        auto ppl_runtime_ptr = std::shared_ptr<::ppl::nn::Runtime>(builder_->CreateRuntime());
        ::ppl::nn::DeviceContext* ppl_device_context =  ppl_runtime_ptr->GetDeviceContext(0);
        cudaStream_t stream = static_cast<cudaStream_t>(ppl_device_context->GetContextStream());
        ppl_runtime_wrapper_set_.insert(std::make_shared<PPLRuntimeWrapper>(shrink_threshold_, ppl_runtime_ptr, stream));
    }
};
std::shared_ptr<PPLRuntimeWrapper> PPLStreamPPLRuntimePool::allocatePPLRuntimeWrapper(cudaStream_t stream){
    stage_ppl_runtime_wrapper_[stream] = nullptr;
    std::lock_guard<OrtMutex> lock(ppl_runtime_mu_);
    if(ppl_runtime_wrapper_set_.empty()){
        return nullptr;
        if(current_pool_size_ < max_pool_size_){
            auto ppl_runtime_ptr = std::shared_ptr<::ppl::nn::Runtime>(builder_->CreateRuntime());
            ::ppl::nn::DeviceContext* ppl_device_context =  ppl_runtime_ptr->GetDeviceContext(0);
            cudaStream_t stream = static_cast<cudaStream_t>(ppl_device_context->GetContextStream());
            ppl_runtime_wrapper_set_.insert(std::make_shared<PPLRuntimeWrapper>(shrink_threshold_, ppl_runtime_ptr,stream));
            current_pool_size_++;
        }else{
            return nullptr;
        }
    }
    std::shared_ptr<PPLRuntimeWrapper> res = (*ppl_runtime_wrapper_set_.begin());
    res->addUnuseNum(false);
    res->setInPoolFlag(false);
    ppl_runtime_wrapper_set_.erase(ppl_runtime_wrapper_set_.begin());

    // shrinking pool
    for(auto iter = ppl_runtime_wrapper_set_.begin(); iter!=ppl_runtime_wrapper_set_.end();){
        (*iter)->addUnuseNum(true);
        if((*iter)->isRelease()){
            iter = ppl_runtime_wrapper_set_.erase(iter);
            current_pool_size_--;
        }else{
            iter++;
        }
    }
    stage_ppl_runtime_wrapper_[stream] = res;
    return res;
};
void PPLStreamPPLRuntimePool::recoverPPLRuntimeWrapper(cudaStream_t stream){
    std::lock_guard<OrtMutex> lock(ppl_runtime_mu_);
    std::shared_ptr<PPLRuntimeWrapper> ppl_runtime_wrapper = stage_ppl_runtime_wrapper_.at(stream);
    stage_ppl_runtime_wrapper_[stream] = nullptr;
    ppl_runtime_wrapper->setInPoolFlag(true);
    ppl_runtime_wrapper_set_.insert(std::move(ppl_runtime_wrapper));
    if(stage_ppl_runtime_wrapper_.size() > 128){
        for(auto iter=stage_ppl_runtime_wrapper_.begin(); iter!=stage_ppl_runtime_wrapper_.end(); ){
        if(!iter->second){
            iter = stage_ppl_runtime_wrapper_.erase(iter);
        }else{
            iter++;
        }
        }
    }
};

}



