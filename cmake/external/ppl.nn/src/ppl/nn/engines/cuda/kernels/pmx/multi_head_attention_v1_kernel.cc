// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include "ppl/nn/engines/cuda/kernels/pmx/multi_head_attention_v1_kernel.h"
#include "cudakernel/nn/multi_head_attn_v1.h"

namespace ppl { namespace nn { namespace cuda {


ppl::common::RetCode MultiHeadAttnV1Kernel::DoExecute(KernelExecContext* ctx) {

    auto query = ctx->GetInput<TensorImpl>(0);
    auto key = ctx->GetInput<TensorImpl>(1);
    auto value = ctx->GetInput<TensorImpl>(2);

    void *attn_mask_data;
    TensorShape *attn_mask_shape;

    if (ctx->GetInputCount() == 4) {
        auto attn_mask = ctx->GetInput<TensorImpl>(3);
        attn_mask_data = attn_mask->GetBufferPtr();
        attn_mask_shape = attn_mask->GetShape();
    } else {
        attn_mask_data = nullptr;
        attn_mask_shape  = nullptr;
    }

    auto attn_output = ctx->GetOutput<TensorImpl>(0);

    uint32_t input_dims = query->GetShape()->GetDimCount();
    if (input_dims < 3 || input_dims > 4){
        LOG(ERROR) << "[Unsupported] Query dim:" << input_dims <<  "Only supported dim 3 or 4,";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (input_dims==3) {
        std::vector<int64_t> expand_shape = {1,
                                            key->GetShape()->GetDim(0),
                                            key->GetShape()->GetDim(1),
                                            key->GetShape()->GetDim(2)};
        // assume that q/k/v has the same shape
        query->GetShape()->Reshape(expand_shape);
        key->GetShape()->Reshape(expand_shape);
        value->GetShape()->Reshape(expand_shape);

    }


    int64_t num_heads = param_->num_heads <= 0 ? query->GetShape()->GetDim(2) : param_->num_heads;
    int64_t num_kv_heads = param_->num_kv_heads <= 0  ? key->GetShape()->GetDim(2) : param_->num_kv_heads;
    int64_t head_dim =  key->GetShape()->GetDim(3);
    int64_t is_causal = param_->is_causal == -1 ? 0 : param_->is_causal;

    if (num_heads < 0) {
        LOG(ERROR) << "[Invalid] num_heads:" << num_heads;
        return ppl::common::RC_INVALID_VALUE;
    }


    if (num_kv_heads < 0) {
        LOG(ERROR) << "[Invalid] num_kv_heads: " << num_kv_heads;
        return ppl::common::RC_INVALID_VALUE;
    }

    if (head_dim < 0) {
        LOG(ERROR) << "[Invalid] head_dim: " << head_dim;
        return ppl::common::RC_INVALID_VALUE;
    }

    if (is_causal != 0 && is_causal != 1) {
        LOG(ERROR) << "[Invalid] is_causal: " << is_causal << ". Only support 0 or 1";
        return ppl::common::RC_INVALID_VALUE;
    }

    auto status = PPLCUDAMultiHeadAttentionForwardImp(
                    GetStream(),
                    query->GetShape(),
                    query->GetBufferPtr(),
                    key->GetShape(),
                    key->GetBufferPtr(),
                    value->GetBufferPtr(),
                    attn_mask_shape,
                    attn_mask_data,
                    is_causal,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    attn_output->GetShape(),
                    attn_output->GetBufferPtr());

    return status;
}

}}} // namespace ppl::nn::cuda
