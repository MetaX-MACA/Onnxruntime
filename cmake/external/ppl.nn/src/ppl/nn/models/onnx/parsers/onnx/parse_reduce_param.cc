// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
#include "ppl/common/str_utils.h"
#include "ppl/nn/models/utils.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_reduce_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseReduceParam(const ::onnx::NodeProto& pb_node, const ParamParserExtraArgs& args, ir::Node* node,
                         ir::Attr* arg) {
    auto param = static_cast<ReduceParam*>(arg);

    if (node->GetType().name == "ReduceSum") {
        param->type = ReduceParam::ReduceSum;
        auto& node_type = node->GetType();
        if (node_type.version < 13 ) {
            auto topo = args.topo;
            auto data = args.data;

            std::vector<int64_t> axes;
            utils::GetNodeAttr(pb_node, "axes", &axes);

            if(axes.size() > 0)
            {
                auto new_edge_name = node->GetName() + "_axes_" + ToString(topo->GetCurrentEdgeIdBound());
                auto edge = ppl::nn::utils::Add1DInitializer(topo, data, new_edge_name, axes, DATATYPE_INT64);
                if (!edge) {
                    LOG(ERROR) << "add initializer[" << new_edge_name << "] failed.";
                    return RC_OTHER_ERROR;
                }
                node->AddInput(edge->GetId());
                edge->AddConsumer(node->GetId());

                node_type.version = 13;
            }
        }
    } else if (node->GetType().name == "ReduceMax") {
        param->type = ReduceParam::ReduceMax;
    } else if (node->GetType().name == "ReduceMin") {
        param->type = ReduceParam::ReduceMin;
    } else if (node->GetType().name == "ReduceProd") {
        param->type = ReduceParam::ReduceProd;
    } else if (node->GetType().name == "ReduceMean") {
        param->type = ReduceParam::ReduceMean;
    } else {
        param->type = ReduceParam::ReduceUnknown;
    }

    if(node->GetType().name != "ReduceSum") {
        utils::GetNodeAttr(pb_node, "axes", &param->axes);
    }else{
        utils::GetNodeAttr(pb_node, "noop_with_empty_axes", &param->noop_with_empty_axes, false);
    }
    utils::GetNodeAttr(pb_node, "keepdims", &param->keepdims, 1);

    return RC_SUCCESS;
}

RetCode PackReduceParam(const ir::Node*, const ir::Attr* arg, ::onnx::NodeProto* pb_node) {
    auto param = static_cast<const ReduceParam*>(arg);
    if(param->type == ReduceParam::ReduceSum) {
        utils::SetNodeAttr(pb_node, "axes", param->axes);
    }else {
        utils::SetNodeAttr(pb_node, "noop_with_empty_axes", param->noop_with_empty_axes);
    }
    
    utils::SetNodeAttr(pb_node, "keepdims", param->keepdims);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
