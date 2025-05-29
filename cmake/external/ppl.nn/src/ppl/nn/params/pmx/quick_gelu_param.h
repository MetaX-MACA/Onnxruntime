#ifndef _ST_HPC_PPL_NN_PARAMS_PMX_QUICK_GELU_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PMX_QUICK_GELU_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace pmx {

struct QuickGeluParam final : public ir::TypedAttr<QuickGeluParam> {
    float alpha;

    bool operator==(const QuickGeluParam& p) const {
        return (alpha == p.alpha);
    }
};

}}} // namespace ppl::nn::pmx

#endif
