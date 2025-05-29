// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include "ppl/nn/models/onnx/parsers/pmx/parse_split_activation_param.h"
#include "ppl/nn/models/onnx/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseSplitActivationParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node*, ir::Attr* arg) {
    auto param = static_cast<ppl::nn::pmx::SplitActivationParam*>(arg);

    utils::GetNodeAttr(pb_node, "activation", &param->activation, "");

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
