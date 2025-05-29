// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#ifndef _ST_HPC_PPL_NN_PARAMS_PMX_MULTI_HEAD_ATTN_V1_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PMX_MULTI_HEAD_ATTN_V1_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace pmx {

struct MultiHeadAttnV1Param final : public ir::TypedAttr<MultiHeadAttnV1Param> {
    int64_t head_dim;
    int64_t is_causal;
    int64_t num_heads;
    int64_t num_kv_heads;

    bool operator==(const MultiHeadAttnV1Param& p) const {
        return (head_dim == p.head_dim && is_causal == p.is_causal && num_heads==p.num_heads && num_kv_heads==p.num_kv_heads);
    }
};

}}} // namespace ppl::nn::pmx

#endif
