// Copyright 2024 metax-tech.com Inc. All Rights Reserved.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/graph_transformer.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
/**
@Class InsertCastCustomNode
insert cast node in custom op node

*/
class InsertCastCustomNode : public GraphTransformer {
 public:
  InsertCastCustomNode(KernelRegistryManager* kernel_registry_manager, const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("InsertCastCustomNode", compatible_execution_providers) {
        kernel_registry_manager_ = kernel_registry_manager;
      }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  Status insertCast(Graph& graph, Node& node, const logging::Logger& logger) const;
  KernelRegistryManager* kernel_registry_manager_;
};

class CustomNodeLayout
{
public:
  CustomNodeLayout(const Node& node, std::vector<const KernelRegistry*> kernel_registries): node_(&node){
    kernel_registries_ = kernel_registries;
    init();
  }
  OrtDataLayout getInputDataLayout(int index){
    return inputs_data_layout_[index];
  }
  OrtDataLayout getOutputDataLayout(int index){
    return outputs_data_layout_[index];
  }
private:
  void init();

private:
  std::vector<OrtDataLayout> inputs_data_layout_;
  std::vector<OrtDataLayout> outputs_data_layout_;
  const Node* node_;
  std::vector<const KernelRegistry*> kernel_registries_;
};


}  // namespace onnxruntime
