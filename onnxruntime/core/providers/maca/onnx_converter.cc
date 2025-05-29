// Copyright 2023 metax-tech.com Inc.

#include "onnx_converter.h"

namespace onnxruntime {

std::vector<std::vector<int>> OnnxConverter::GetSupportedNodes(
    const ONNX_NAMESPACE::ModelProto& model_proto) {
  std::vector<std::vector<int>> supported_node_vecs;
  std::vector<int> supported_node_vec;
  for (int i = 0; i < model_proto.graph().node_size(); i++) {
    bool supported = true;
    // std::string error_msg;
    // std::tie(supported, error_msg) =
    //     IsNodeSupported(model_proto, model_proto.graph().node(i));
    if (supported) {
      supported_node_vec.push_back(i);
    } else {
    //   const auto& op = model_proto.graph().node(i).op_type();
    //   LOGS_DEFAULT(INFO) << op << ": " << error_msg;
      if (!supported_node_vec.empty()) {
        supported_node_vecs.push_back(supported_node_vec);
        supported_node_vec.clear();
      }
    }
  }
  if (!supported_node_vec.empty()) {
    supported_node_vecs.push_back(supported_node_vec);
  }
  return supported_node_vecs;
}

bool OnnxConverter::RemoveNode(onnxruntime::Graph& graph, onnxruntime::Node& node) {
  // Only have one input edge for Q/DQ
  NodeArg& node_input = *(node.MutableInputDefs()[0]);
  auto scale_n = node.InputDefs()[1]->Name();
  auto zp_n = node.InputDefs()[2]->Name();

  if (graph.NodeProducesGraphOutput(node)) {
    std::vector<graph_utils::GraphEdge> input_edges = graph_utils::GraphEdge::GetNodeInputEdges(node);
    if (graph.IsInitializedTensor(scale_n) && graph.IsInitializedTensor(zp_n)) {
      graph.RemoveInitializedTensor(scale_n);
      graph.RemoveInitializedTensor(zp_n);
    }

    for (const auto& input_edge : input_edges) {
      Node& src_node = (*graph.GetNode(input_edge.src_node));
      graph_utils::FinalizeNodeFusion(graph, src_node, node);
    }
  } else {
    std::vector<graph_utils::GraphEdge> input_edges = graph_utils::GraphEdge::GetNodeInputEdges(node);
    std::vector<graph_utils::GraphEdge> output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(node);
    for (const auto& output_edge : output_edges) {
      Node& dst_node = (*graph.GetNode(output_edge.dst_node));
      graph.RemoveEdge(output_edge.src_node, output_edge.dst_node,
                            output_edge.src_arg_index, output_edge.dst_arg_index);
      graph_utils::ReplaceNodeInput(dst_node, output_edge.dst_arg_index, node_input);
      for (const auto& input_edge : input_edges) {
        Node& src_node = (*graph.GetNode(input_edge.src_node));
        graph.AddEdge(src_node.Index(), dst_node.Index(), input_edge.src_arg_index, output_edge.dst_arg_index);
      }
    }

    if (graph.IsInitializedTensor(scale_n) && graph.IsInitializedTensor(zp_n)) {
      graph.RemoveInitializedTensor(scale_n);
      graph.RemoveInitializedTensor(zp_n);
    }
    graph.RemoveNode(node.Index());
  }
  return true;
}

}  // namespace onnxruntime
