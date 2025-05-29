// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_GROUPNORM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_GROUPNORM_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/pmx/group_normalization_param.h"
#include "ppl/nn/engines/cuda/params/instance_normalization_extra_param.h"

namespace ppl { namespace nn { namespace cuda {

class GroupNormKernel : public CudaKernel {
public:
    GroupNormKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const CudaGroupNormalizationParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const CudaGroupNormalizationParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
