// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include "ppl/nn/models/onnx/parsers/pmx/parse_multi_head_attn_v1_param.h"
#include "ppl/nn/models/onnx/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseMultiHeadAttnV1Param(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*, ir::Attr* arg) {
    auto param = static_cast<ppl::nn::pmx::MultiHeadAttnV1Param*>(arg);

    utils::GetNodeAttr(pb_node, "head_dim", &param->head_dim, -1);
    utils::GetNodeAttr(pb_node, "is_causal", &param->is_causal, 1);
    utils::GetNodeAttr(pb_node, "num_heads", &param->num_heads, -1);
    utils::GetNodeAttr(pb_node, "num_kv_heads", &param->num_kv_heads, -1);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
