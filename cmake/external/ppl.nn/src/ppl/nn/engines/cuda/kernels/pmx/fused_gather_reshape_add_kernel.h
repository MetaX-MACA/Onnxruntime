#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_FUSED_ADD_LAYERNORM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_FUSED_ADD_LAYERNORM_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/gather_param.h"

namespace ppl { namespace nn { namespace cuda {

class FusedGatherReshapeAddKernel : public CudaKernel {
public:
    FusedGatherReshapeAddKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::onnx::GatherParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::onnx::GatherParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
