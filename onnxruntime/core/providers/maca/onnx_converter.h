// Copyright 2023 metax-tech.com Inc.

#pragma once

#include <onnx/onnx_pb.h>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include <string>

namespace onnxruntime {

/**
 *  For convert from onnx::ModelProto to rk::nn::Graph.
 */
class OnnxConverter {
 public:
  OnnxConverter() {}
  ~OnnxConverter() { }

  /** Get the supported subgraph.
  */
  std::vector<std::vector<int>> GetSupportedNodes(
      const ONNX_NAMESPACE::ModelProto& model_proto);

  /** Convert onnx::ModelProto to rk::nn::Graph.
   *  Because some attributes of rk::nn::Tensor are used as input in onnx and these attributes
   *  can't be found in onnx::ModelProto, so additional input-related information is required.
  */
//   void Convert(const ONNX_NAMESPACE::ModelProto& model,
//                rk::nn::Graph* graph,
//                const std::vector<const void*>& input_bufs,
//                const std::unordered_map<std::string, int>& input_maps);

  // std::string m(const std::string& str) const;
  bool RemoveNode(onnxruntime::Graph& graph, onnxruntime::Node& node);
 private:
  void Clear();

  OnnxConverter(const OnnxConverter&);
  OnnxConverter& operator=(const OnnxConverter&);
};


}  // namespace onnxruntime
