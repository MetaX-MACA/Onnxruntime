#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_FUSED_GATHERND_RESHAPE_CONCAT_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_FUSED_GATHERND_RESHAPE_CONCAT_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/engines/cuda/params/fused_gathernd_reshape_concat_param.h"

namespace ppl { namespace nn { namespace cuda {

class FusedGatherNDReshapeConcatKernel : public CudaKernel {
public:
    FusedGatherNDReshapeConcatKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const FusedGatherNDReshapeConcatParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const FusedGatherNDReshapeConcatParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
