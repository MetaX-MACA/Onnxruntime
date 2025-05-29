// Copyright 2024 metax-tech.com Inc. All Rights Reserved.

#include "core/optimizer/insert_cast_custom_node.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/kernel_registry.h"

#include <set>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
static std::string custom_domain = "custom.op";
namespace onnxruntime {

void CustomNodeLayout::init(){
    const KernelCreateInfo* maca_kernel_def = nullptr;
    for (auto registry : kernel_registries_) {
      auto st = registry->TryFindKernel(*node_, kMacaExecutionProvider, &maca_kernel_def);
      // at least one registry has a MACA kernel for this node
      if (st.IsOK())
        break;
    }

    // none of the provided registries has a MACA kernel for this node
    if (maca_kernel_def) {
      for(size_t i =0 ; i < node_->InputDefs().size(); i++){
        inputs_data_layout_.push_back(OrtDataLayout(maca_kernel_def->kernel_def->GetInputDataLayout(i)));
      }
      for(size_t i = 0; i < node_->OutputDefs().size(); i++){
        outputs_data_layout_.push_back(OrtDataLayout(maca_kernel_def->kernel_def->GetOutputDataLayout(i)));
      }
    }else{
        LOGS_DEFAULT(ERROR) << "Not find custom op kernel for node : " << node_->Name();
    }

}

onnxruntime::NodeArg* AddCastNode(onnxruntime::Graph& graph,
                                  onnxruntime::NodeArg* old_arg,
                                  bool new_on_input) {
  // insert cast op to cast input
  std::string node_name = graph.GenerateNodeName("InsertedCast_" + old_arg->Name());

  auto* new_arg = &graph.GetOrCreateNodeArg(node_name, old_arg->TypeAsProto());

  std::vector<onnxruntime::NodeArg*> input_defs = {new_on_input ? new_arg : old_arg};
  std::vector<onnxruntime::NodeArg*> output_defs = {new_on_input ? old_arg : new_arg};

  auto& cast_node = graph.AddNode(node_name, "Cast", "cast node used to transform datalayout.",
                                  input_defs, output_defs);
  cast_node.AddAttribute("to", static_cast<int64_t>(old_arg->TypeAsProto()->tensor_type().elem_type()));
  return new_arg;
}

Status InsertCastCustomNode::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
    for (auto& node : graph.Nodes()){
        if(node.Domain() == custom_domain){

            insertCast(graph, node, logger);
        }
    }
    return Status::OK();
}


Status InsertCastCustomNode::insertCast(Graph& graph, Node& node, const logging::Logger& logger) const {
    std::vector<const KernelRegistry*> kernel_registries = kernel_registry_manager_->GetKernelRegistriesByProviderType(kMacaExecutionProvider);
    CustomNodeLayout custom_op_node_layout(node, kernel_registries);

    std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacement_defs;

    auto input_defs = node.MutableInputDefs();
    std::set<int> not_model_input_arg;
    for (auto input_edge = node.InputEdgesBegin(), end = node.InputEdgesEnd(); input_edge != end; ++input_edge) {
        int src_arg_index = input_edge->GetSrcArgIndex();
        int dst_arg_index = input_edge->GetDstArgIndex();
        not_model_input_arg.insert(dst_arg_index);
        if(input_edge->GetNode().Domain()==custom_domain ){
            CustomNodeLayout custom_src_op_node_layout(input_edge->GetNode(), kernel_registries);
            if(custom_op_node_layout.getInputDataLayout(dst_arg_index) != custom_src_op_node_layout.getOutputDataLayout(src_arg_index)){
                std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> edge_node_replacement_defs;
                onnxruntime::NodeArg* new_arg = AddCastNode(graph, input_defs[dst_arg_index], false);
                edge_node_replacement_defs[input_defs[dst_arg_index]]=new_arg;
                NodeIndex node_index =input_edge->GetNode().Index();
                Node* edge_node = graph.GetNode(node_index);
                edge_node->ReplaceDefs(edge_node_replacement_defs);
            }
        }
    }
    for(size_t i=0; i<input_defs.size(); i++){
        if((not_model_input_arg.find(i) == not_model_input_arg.end()) &&
            (custom_op_node_layout.getInputDataLayout(i)!=OrtDataLayout::NCHW)
            ){
            onnxruntime::NodeArg* new_arg = AddCastNode(graph, input_defs[i], false);
            replacement_defs[input_defs[i]] = new_arg;
        }
    }

    auto output_defs = node.MutableOutputDefs();
    std::set<int> not_model_output_arg;
    for(auto output_edge = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); output_edge != end; ++output_edge) {
        int src_arg_index = output_edge->GetSrcArgIndex();
        not_model_output_arg.insert(src_arg_index);
    }
    for(size_t i=0; i<output_defs.size(); i++){
        if((not_model_output_arg.find(i) == not_model_output_arg.end())&&
            (custom_op_node_layout.getOutputDataLayout(i)!=OrtDataLayout::NCHW)){
            onnxruntime::NodeArg* new_arg = AddCastNode(graph, output_defs[i], true);
            replacement_defs[output_defs[i]] = new_arg;
        }
    }
    node.ReplaceDefs(replacement_defs);
    LOGS(logger, INFO) << "Find custom node " << node.Name();
    return Status::OK();
}






}
