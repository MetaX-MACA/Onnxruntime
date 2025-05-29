// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_MULTI_HEAD_ATTN_V1_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_PMX_MULTI_HEAD_ATTN_V1_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/pmx/multi_head_attn_v1_param.h"

namespace ppl { namespace nn { namespace cuda {

class MultiHeadAttnV1Kernel : public CudaKernel {
public:
    MultiHeadAttnV1Kernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::pmx::MultiHeadAttnV1Param* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::pmx::MultiHeadAttnV1Param* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
