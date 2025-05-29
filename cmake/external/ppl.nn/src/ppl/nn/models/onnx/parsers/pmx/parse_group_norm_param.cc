// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include "ppl/nn/models/onnx/parsers/pmx/parse_group_norm_param.h"
#include "ppl/nn/models/onnx/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseGroupNormParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*, ir::Attr* arg) {
    auto param = static_cast<ppl::nn::pmx::GroupNormalizationParam*>(arg);

    utils::GetNodeAttr(pb_node, "num_groups", &param->num_groups, -1);
    utils::GetNodeAttr(pb_node, "epsilon", &param->epsilon, 1e-5);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::caffe
