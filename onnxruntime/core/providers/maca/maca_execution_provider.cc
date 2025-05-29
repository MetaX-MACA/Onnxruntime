// Copyright 2023 metax-tech.com Inc.

#include <fstream>
#include <list>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <limits>
#include <map>
#include <memory>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

#include "maca_execution_provider.h"

#include "core/common/logging/logging.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "core/framework/memcpy.h"
#include "core/graph/graph_proto_serializer.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/indexed_sub_graph.h"

#include "maca_call.h"
#include "maca_profiling.h"
#include "maca_common.h"
#include "ppl_runtime_pool.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/options.h"
#include "ppl/nn/common/device_context.h"
#include <random>
#include <iostream>


namespace onnxruntime {

int parseENV(std::string env_str, int default_value){
  char* env = getenv(env_str.c_str());
  if(env){
    return std::atoi(env);
  }
  return default_value;
}

const int MACART_POOL_SIZE=parseENV("MACART_POOL_SIZE", 1);
const int MACART_EXECUTE_MODE=parseENV("MACART_EXECUTE_MODE", 0);
const int MACART_ENABLE_PROFILING=parseENV("MACART_ENABLE_PROFILING", 0);
struct MACAFuncState {
    std::unique_ptr<::ppl::nn::Engine> engine;
    std::unique_ptr<::ppl::nn::onnx::RuntimeBuilder::Resources> resources;
    std::shared_ptr<::ppl::nn::onnx::RuntimeBuilder> builder;
    std::shared_ptr<PPLRuntimePool> runtime_pool;
    std::unordered_map<std::string, size_t> input_info;
    std::unordered_map<std::string, size_t> output_info;
    std::unordered_map<std::string, size_t> type_info;
    std::unordered_map<std::string, std::vector<bool>> input_dy_axis_info;
};

static std::unordered_map<ppl::common::datatype_t, ONNXTensorElementDataType> DATA_TYPE_PPL_MAP_ORT={
  {ppl::common::DATATYPE_UNKNOWN, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED},
  {ppl::common::DATATYPE_UINT8, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8},
  {ppl::common::DATATYPE_UINT16, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16},
  {ppl::common::DATATYPE_UINT32, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32},
  {ppl::common::DATATYPE_UINT64, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64},
  {ppl::common::DATATYPE_FLOAT16, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16},
  {ppl::common::DATATYPE_FLOAT32, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},
  {ppl::common::DATATYPE_FLOAT64, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED},
  {ppl::common::DATATYPE_BFLOAT16, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16},
  {ppl::common::DATATYPE_INT4B, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED},
  {ppl::common::DATATYPE_INT8, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8},
  {ppl::common::DATATYPE_INT16, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16},
  {ppl::common::DATATYPE_INT32, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32},
  {ppl::common::DATATYPE_INT64, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
  {ppl::common::DATATYPE_BOOL, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL},
  {ppl::common::DATATYPE_COMPLEX64, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64},
  {ppl::common::DATATYPE_COMPLEX128, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128},
};

static std::unordered_map<OrtDataLayout, ppl::common::dataformat_t> LAYOUT_ORT_MAP_PPL = {
  {OrtDataLayout::NCHW, ppl::common::DATAFORMAT_NDARRAY},
  {OrtDataLayout::NHWC8, ppl::common::DATAFORMAT_NHWC8},
  {OrtDataLayout::NHWC16, ppl::common::DATAFORMAT_NHWC16}
};

static std::unordered_map<ONNXTensorElementDataType, size_t > OrtTypeMapLen = {
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, 1},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 1},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, 1},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, 2},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, 2},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, 1},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 2},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, 8},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, 4},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, 8}
};


// NOT USE
void releaseResource(::ppl::nn::onnx::RuntimeBuilder::Resources *obj){
	if(obj){
		if(obj->engines)
		{
      if(obj->engines[0])
      {
        delete obj->engines[0];
      }
			delete [] obj->engines;
		}
		delete obj;
	}
}

AllocatorPtr MACAExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeDefault) {
    return allocator_;
  } else {
    return IExecutionProvider::GetAllocator(id, mem_type);
  }
}

MACAExecutionProvider::MACAExecutionProvider(const OrtMACAProviderOptions& info)
    : IExecutionProvider{onnxruntime::kMacaExecutionProvider, true},
    info_{info} {
  CUDA_CALL_THROW(cudaSetDevice(info_.device_id));
  CUDA_CALL_THROW(cudaDeviceSynchronize());
  CUDA_CALL_THROW(cudaGetDeviceProperties(&device_prop_, info_.device_id));

  if (info.has_user_compute_stream) {
    external_stream_ = true;
    stream_ = static_cast<cudaStream_t>(info.user_compute_stream);
  } else {
    CUDA_CALL_THROW(cudaStreamCreate(&stream_));
  }
  int stream_num = MACART_POOL_SIZE;
  if(MACART_EXECUTE_MODE == 1){
    stream_num = 32;
  }
  for(int i=0; i<stream_num; i++){
    cudaStream_t stream;
    CUDA_CALL_THROW(cudaStreamCreate(&stream));
    stream_queue_.push(stream);
  }

  size_t free = 0;
  size_t total = 0;
  CUDA_CALL_THROW(cudaMemGetInfo(&free, &total));
}


MACAExecutionProvider::~MACAExecutionProvider() {
  if (!external_stream_ && stream_) {
    // [MCR][E]cuda_runtime_wrapper.cpp :1038: 29225: [7f63c5d00740] wcudaStreamDestroy: Returned mcErrorInvalidResourceHandle
    CUDA_CALL(cudaStreamDestroy(stream_));
  }
  while(stream_queue_.empty() == false){
    CUDA_CALL(cudaStreamDestroy(stream_queue_.front()));
    stream_queue_.pop();
  }
}

Status MACAExecutionProvider::SetComputeStream(void* stream) {
  if (stream != stream_) {
    if (stream_) {
      CUDA_RETURN_IF_ERROR(cudaStreamDestroy(stream_));
    }

    external_stream_ = true;
    stream_ = static_cast<cudaStream_t>(stream);
  }
  return Status::OK();
}

void* MACAExecutionProvider::GetComputeStream() const  {
  void* get_stream = nullptr;
  std::thread::id thread_id = std::this_thread::get_id();
  std::lock_guard<OrtMutex> lock(*stream_mutex_);
  get_stream = (*thread_map_stream_)[thread_id];
  return get_stream;
}

/**
    Called when InferenceSession::Run started
    NOTE that due to async execution in provider, the actual work of previous
    Run may not be finished on device This function should be regarded as the
    point after which a new Run would start to submit commands from CPU
*/
common::Status MACAExecutionProvider::OnRunStart()   {
  void* get_stream = nullptr;
  std::thread::id thread_id = std::this_thread::get_id();
  while(!get_stream){
    {
      std::lock_guard<OrtMutex> lock(*stream_mutex_);
      if(stream_queue_.empty() == false){
        get_stream = static_cast<void*>(stream_queue_.front());
        (*thread_map_stream_)[thread_id] = get_stream;
        stream_queue_.pop();
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  return Status::OK();
}

/**
    Called when InferenceSession::Run ended
    NOTE that due to async execution in provider, the actual work of this Run
    may not be finished on device This function should be regarded as the point
    that all commands of current Run has been submmited by CPU
*/
common::Status MACAExecutionProvider::OnRunEnd(bool sync_stream) {
  std::thread::id thread_id = std::this_thread::get_id();
  std::lock_guard<OrtMutex> lock(*stream_mutex_);
  cudaStream_t stream = static_cast<cudaStream_t>((*thread_map_stream_)[thread_id]);
  cudaStreamSynchronize(stream);
  stream_queue_.push(stream);
  (*thread_map_stream_)[thread_id] = nullptr;
  if((*thread_map_stream_).size() > 128){
    for(auto iter=(*thread_map_stream_).begin(); iter!=(*thread_map_stream_).end(); ){
      if(!iter->second){
        iter = (*thread_map_stream_).erase(iter);
      }else{
        iter++;
      }
    }
  }
  return Status::OK();
}


std::unique_ptr<onnxruntime::IDataTransfer> MACAExecutionProvider::GetDataTransfer() const {
  return std::make_unique<onnxruntime::GPUDataTransfer>(static_cast<cudaStream_t>(GetComputeStream()),
                                                        info_.do_copy_in_default_stream, thread_map_stream_, stream_mutex_);
}
ProviderOptions MACAExecutionProvider::GetProviderOptions() const {
  const ProviderOptions options{
      {maca::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info_.device_id)}
  };
  return options;
}

AllocatorPtr MACAExecutionProvider::CreateCudaAllocator(OrtDevice::DeviceId device_id, size_t gpu_mem_limit, int arena_extend_strategy,
                                                        OrtArenaCfg* default_memory_arena_cfg) {
    AllocatorCreationInfo default_memory_info(
        [](OrtDevice::DeviceId id) {
          return std::make_unique<CUDAAllocator>(id, MACA);
        },
        device_id,
        true,
        {default_memory_arena_cfg ? *default_memory_arena_cfg
                                  : OrtArenaCfg(gpu_mem_limit, static_cast<int>(arena_extend_strategy), -1, -1, -1)});

    // CUDA malloc/free is expensive so always use an arena
    return CreateAllocator(default_memory_info);
  // }
}

void MACAExecutionProvider::RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) {
  // Try to get a CUDA allocator from allocator manager first
  // Used to allocate CUDA device memory
  // CUDAExecutionProviderInfo info_;
  allocator_ = allocator_manager->GetAllocator(info_.device_id, OrtMemTypeDefault);
  if (nullptr == allocator_) {
    allocator_ = CreateCudaAllocator(info_.device_id, info_.gpu_mem_limit, info_.arena_extend_strategy,
                                    //  info_.external_allocator_info,
                                     info_.default_memory_arena_cfg);
    allocator_manager->InsertAllocator(allocator_);
  }
  TryInsertAllocator(allocator_);

  // OrtMemTypeCPUOutput -- allocated by cudaMallocHost, used to copy CUDA device memory to CPU
  // Use pinned memory instead of pageable memory make the data transfer faster
  // Used by node MemcpyToHost only
  auto cuda_pinned_alloc = allocator_manager->GetAllocator(DEFAULT_CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPUOutput);
  if (nullptr == cuda_pinned_alloc) {
    AllocatorCreationInfo pinned_memory_info(
        [](OrtDevice::DeviceId device_id) {
          return std::make_unique<CUDAPinnedAllocator>(device_id, MACA_PINNED);
        },
        DEFAULT_CPU_ALLOCATOR_DEVICE_ID);

    cuda_pinned_alloc = CreateAllocator(pinned_memory_info);
    allocator_manager->InsertAllocator(cuda_pinned_alloc);
  }
  TryInsertAllocator(std::move(cuda_pinned_alloc));

  // OrtMemTypeCPUInput -- CUDA op place the input on CPU and will not be accessed by CUDA kernel, no sync issue
  auto cuda_cpu_alloc = allocator_manager->GetAllocator(DEFAULT_CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPUInput);
  if (nullptr == cuda_cpu_alloc) {
    // TODO: this is actually used for the cuda kernels which explicitly ask for inputs from CPU.
    // This will be refactored/removed when allocator and execution provider are decoupled.
    // Need to move the OrtMemoryType out of Allocator, that's one thing blocking us to share it with CPU EP
    // CPUAllocator is OrtMemTypeDefault for CPU EP
    AllocatorCreationInfo cpu_memory_info(
        [](int device_id) {
          return std::make_unique<CPUAllocator>(
              OrtMemoryInfo("MACA_CPU", OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), device_id,
                            OrtMemTypeCPUInput));
        },
        DEFAULT_CPU_ALLOCATOR_DEVICE_ID);

    cuda_cpu_alloc = CreateAllocator(cpu_memory_info);
    allocator_manager->InsertAllocator(cuda_cpu_alloc);
  }
  TryInsertAllocator(std::move(cuda_cpu_alloc));
}

std::vector<std::vector<int>> MACAExecutionProvider::GetSupportedNodes(
    const ONNX_NAMESPACE::ModelProto& model_proto) const {
  OnnxConverter converter;
  return converter.GetSupportedNodes(model_proto);
}

std::vector<std::unique_ptr<ComputeCapability>>
MACAExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::vector<NodeIndex> costom_op_node_index;
  std::vector<std::vector<NodeIndex>> ppl_op_node_index;
  std::vector<NodeIndex> sub_ppl_op_node_index;
  for (auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph_viewer.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    const KernelCreateInfo* maca_kernel_def = nullptr;
    if (!node.GetExecutionProviderType().empty()) {
      LOGS_DEFAULT(ERROR) << "node " << node.Name() << " GetExecutionProviderType: " << node.GetExecutionProviderType() << std::endl;
      continue;
    }

    for (auto registry : kernel_registries) {
      auto st = registry->TryFindKernel(node, Type(), &maca_kernel_def);

      // at least one registry has a CUDA kernel for this node
      if (st.IsOK())
        break;
    }

    // none of the provided registries has a MACA kernel for this node
    if (maca_kernel_def) {
      LOGS_DEFAULT(INFO) << "maca kernel found in custom registries for Op type: " << node.OpType() << ", node name: " << node.Name();
      // auto attributes = node.GetAttributes();
      // for(auto p : attributes){
      //   std::cout << "Names: " << p.first << "; data size : " << p.second.ints()[0]<< std::endl;
      // }{
      for(size_t i =0 ; i < node.InputDefs().size(); i++){
        std::string input_name = node.InputDefs()[i]->Name();
        if((*tensor_data_layout_).count(input_name) == 0){
          (*tensor_data_layout_)[input_name] = OrtDataLayout(maca_kernel_def->kernel_def->GetInputDataLayout(i));
        }else{
          if((*tensor_data_layout_)[input_name] != OrtDataLayout(maca_kernel_def->kernel_def->GetInputDataLayout(i))){
            ORT_THROW("Error! Node " + node.Name() + ", input name : " + input_name + ", DataLayout mismatch (current:" + std::to_string(int((*tensor_data_layout_)[input_name])) + " vs exist:" + std::to_string(maca_kernel_def->kernel_def->GetInputDataLayout(i)) + ")");
          }
        }
      }
      for(size_t i = 0; i < node.OutputDefs().size(); i++){
         std::string output_name = node.OutputDefs()[i]->Name();
        if((*tensor_data_layout_).count(output_name) == 0){
          (*tensor_data_layout_)[output_name] = OrtDataLayout(maca_kernel_def->kernel_def->GetOutputDataLayout(i));
        }else{
          if((*tensor_data_layout_)[output_name]  != OrtDataLayout(maca_kernel_def->kernel_def->GetOutputDataLayout(i))){
            ORT_THROW("Error! Node " + node.Name() + ", output name : " + output_name + ", DataLayout mismatch (current:" + std::to_string(int((*tensor_data_layout_)[output_name])) + " vs exist:" + std::to_string(maca_kernel_def->kernel_def->GetOutputDataLayout(i)) + ")");
          }
        }
      }
      costom_op_node_index.push_back(node.Index());
      if(!sub_ppl_op_node_index.empty()){
        ppl_op_node_index.push_back(sub_ppl_op_node_index);
        sub_ppl_op_node_index.clear();
      }
    }else{
      // LOGS_DEFAULT(INFO) << "maca kernel not found in custom registries for Op type: " << node.OpType() << ", node name: " << node.Name();
      sub_ppl_op_node_index.push_back(node.Index());
    }
  }
  if(sub_ppl_op_node_index.empty() == false){
    ppl_op_node_index.push_back(sub_ppl_op_node_index);
  }

  // Get ModelPath and Model Dir
  const auto& path_string = graph_viewer.ModelPath().ToPathString();
  std::strcpy(model_path_, path_string.c_str());
  std::string parent_dir;
  auto pos = std::string(path_string).find_last_of("/\\");
  if (pos == std::string::npos) {
      parent_dir = ".";
  } else {
      parent_dir.assign(path_string.c_str(), pos);
  }
  std::strcpy(model_file_dir_, parent_dir.c_str());

  // Handle If and Loop operators
  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  // Need access to model_path_
  // for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
  //   if (tensor.second->has_data_location() &&
  //       tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
  //     LOGS_DEFAULT(WARNING) << "MACA: Initializers with external data"
  //                              " location are not currently supported";
  //     return result;
  //   }
  // }

  // Construct modelproto from graph_viewer
  onnxruntime::Model model(graph_viewer.Name(), true, ModelMetaData(),
                           PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(),
                           graph_viewer.DomainToVersionMap(),
                           std::vector<ONNX_NAMESPACE::FunctionProto>(),
                           *GetLogger());
  onnxruntime::Graph& graph_build = model.MainGraph();
  const std::vector<NodeIndex>& node_index =
      graph_viewer.GetNodesInTopologicalOrder();
  std::set<NodeArg*> all_node_inputs;
  for (const auto& node : graph_viewer.Nodes()) {
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (const auto input : node.InputDefs()) {
      auto& n_input = graph_build.GetOrCreateNodeArg(
          input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
      all_node_inputs.insert(&n_input);
    }
    for (const auto output : node.OutputDefs()) {
      auto& n_output = graph_build.GetOrCreateNodeArg(
          output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    graph_build.AddNode(node.Name(), node.OpType(), node.Description(),
                        inputs, outputs, &node.GetAttributes(), node.Domain());
  }
  const auto graph_outputs = graph_viewer.GetOutputs();
  graph_build.SetOutputs(graph_outputs);

  // Add initializer to graph_viewer
  const auto& init_tensors = graph_viewer.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    graph_build.AddInitializedTensor(*(tensor.second));
  }

  // ORT_ENFORCE(graph_build.Resolve().IsOK());
  graph_build.Resolve();
  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // const auto supported_nodes_vector = GetSupportedNodes(model_proto);

  int counter = 0;

  for(auto index : costom_op_node_index){
    auto sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(index);
    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  }

  for (const auto& group : ppl_op_node_index) {
    if (!group.empty()) {
      std::unordered_set<size_t> node_set;
      node_set.reserve(group.size());
      for (const auto& index : group) {
        node_set.insert(index);
      }
      std::unique_ptr<IndexedSubGraph> sub_graph =
          std::make_unique<IndexedSubGraph>();
      // Find inputs and outputs of the subgraph
      std::unordered_map<const NodeArg*, int>
          fused_inputs, fused_outputs, fused_outputs_to_add;
      std::unordered_set<const NodeArg*> erased;
      int input_order = 0;
      int output_order = 0;

      for (const auto& index : group) {
        sub_graph->nodes.push_back(index);
        const auto& node = graph_viewer.GetNode(index);

        for (const auto& input : node->InputDefs()) {
          if (!input->Exists())
          {
            continue;
          }
          // If the input is a initializer, skip
          if (init_tensors.find(input->Name()) != init_tensors.end()) {
            continue;
          }
          const auto& it = fused_outputs.find(input);

          if (it != fused_outputs.end()) {
            fused_outputs.erase(it);
            erased.insert(input);
          }
          // only when input is neither in output list nor erased list, add the
          // input to input list
          else if (erased.find(input) == erased.end()) {
            fused_inputs[input] = input_order++;
          }
        }

        // For output searching, there is a special case:
        // If node's OutputEdges are more than its outputs, meaning certain
        // output is used more than once,
        // if the output is connected to nodes that don't belong to the
        // subgraph, the output need to be added to the output list
        if (node->GetOutputEdgesCount() > node->OutputDefs().size()) {
          for (auto it = node->OutputEdgesBegin(),
                    end = node->OutputEdgesEnd();
               it != end; ++it) {
            const auto& node_idx = it->GetNode().Index();
            const auto& output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];

            if (node_set.find(node_idx) != node_set.end()) {
              const auto& iter = fused_inputs.find(output);

              if (iter != fused_inputs.end()) {
                fused_inputs.erase(iter);
                erased.insert(output);
              } else if (erased.find(output) == erased.end()) {
                fused_outputs[output] = output_order++;
              }

              // If output is graph_view output,need to be added to the output list
              if (std::find(graph_outputs.begin(), graph_outputs.end(), output) != graph_outputs.end()){
                if (fused_outputs_to_add.find(output) == fused_outputs_to_add.end())
                  fused_outputs_to_add[output] = output_order++;
              }
            } else {
              fused_outputs_to_add[output] = output_order++;
            }
          }
        } else {
          for (const auto& output : node->OutputDefs()) {
            const auto& it = fused_inputs.find(output);

            if (it != fused_inputs.end()) {
              fused_inputs.erase(it);
              erased.insert(output);
            }
            // only when output is neither in input list nor erased list,
            // add the output to output list
            else if (erased.find(output) == erased.end()) {
              fused_outputs[output] = output_order++;
            }
            // If output is graph_view output,need to be added to the output list
            if (std::find(graph_outputs.begin(), graph_outputs.end(), output) != graph_outputs.end()){
              fused_outputs_to_add[output] = output_order++;
            }
          }
        }
      }

      fused_outputs.insert(
          fused_outputs_to_add.begin(), fused_outputs_to_add.end());

      // Sort inputs and outputs by the order they were added
      std::map<int, const NodeArg*> inputs, outputs;

      for (auto it = fused_inputs.begin(),
                end = fused_inputs.end();
           it != end; ++it) {
        inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      }

      for (auto it = fused_outputs.begin(),
                end = fused_outputs.end();
           it != end; ++it) {
        if (std::find(graph_outputs.begin(),
                      graph_outputs.end(), it->first) != graph_outputs.end()) {
          outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
        }else{
          for (const auto& x : all_node_inputs) {
            if (x->Name() == it->first->Name()) {
              outputs.insert(
                  std::pair<int, const NodeArg*>(it->second, it->first));
              break;
            }
          }
        }
      }

      // Assign inputs and outputs to subgraph's meta_def
      auto meta_def =
          std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
      meta_def->name = "MACA_" + std::to_string(counter++);
      meta_def->domain = kMSDomain;

      for (const auto& input : inputs) {
        meta_def->inputs.push_back(input.second->Name());
      }

      for (const auto& output : outputs) {
        meta_def->outputs.push_back(output.second->Name());
      }

      meta_def->since_version = 1;
      sub_graph->SetMetaDef(std::move(meta_def));

      result.push_back(
          std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }
  return result;
}

struct CallbackParam{
   std::shared_ptr<PPLRuntimePool> runtime_pool;
   std::shared_ptr<PPLRuntimeWrapper> ppl_runtime_wrapper;
};

void recoverPPLRuntimeCallback(cudaStream_t stream, cudaError_t status, void* user_data){
  PPLRuntimePool* runtime_pool = static_cast<PPLRuntimePool*>(user_data);
  runtime_pool->recoverPPLRuntimeWrapper(stream);
};

common::Status MACAExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                  std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    const Graph& graph_body = graph_body_viewer.GetGraph();
    const Node& fused_node = fused_node_graph.fused_node;

    // Build map from input name to its index in input definitions
    std::unordered_map<std::string, size_t> input_map;
    const auto& input_defs = fused_node.InputDefs();
    input_map.reserve(input_defs.size());
    for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
      input_map[input_defs[i]->Name()] = i;
    }

    // Build map from output name to its index in output definitions
    std::unordered_map<std::string, size_t> output_map;
    const auto& output_defs = fused_node.OutputDefs();
    output_map.reserve(output_defs.size());
    for (size_t i = 0, end = output_defs.size(); i < end; ++i) {
      output_map[output_defs[i]->Name()] = i;
    }

    // Has QuantizeLinear or DequantizeLinear
    for (auto& node_idx : graph_body_viewer.GetNodesInTopologicalOrder()) {
      const onnxruntime::Node& node = *graph_body_viewer.GetNode(node_idx);
      if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "QuantizeLinear", {10, 13, 19}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "DequantizeLinear", {10, 13, 19})) {
        quantizer_ = true;
        break;
      }
    }

    // get QuantParamInfo from Q and DQ
    if (quantizer_) {
      for (auto& node_idx : graph_body_viewer.GetNodesInTopologicalOrder()) {
        auto& op_info = quant_param_info_.node_params;
        ::ppl::nn::QuantParam op_param;
        ::ppl::nn::QuantParam::Value op_value;

        const onnxruntime::Node& node = *graph_body_viewer.GetNode(node_idx);
        const auto& node_inputs = node.InputDefs();
        const auto& node_outputs = node.OutputDefs();

        bool is_Q = graph_utils::IsSupportedOptypeVersionAndDomain(node, "QuantizeLinear", {10, 13, 19});
        bool is_DQ = graph_utils::IsSupportedOptypeVersionAndDomain(node, "DequantizeLinear", {10, 13, 19});

        if (!is_Q && !is_DQ) {
          bool quant_op_input = true;
          bool quant_op_output = true;
          for (auto node_it = node.InputNodesBegin(); node_it != node.InputNodesEnd(); ++node_it) {
            if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node_it, "DequantizeLinear", {10, 13, 19})) {
              quant_op_input = false;
            }
          }
          for (auto node_it = node.OutputNodesBegin(); node_it != node.OutputNodesEnd(); ++node_it) {
            if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node_it, "QuantizeLinear", {10, 13, 19})) {
              quant_op_output = false;
            }
          }
          if (graph_body.NodeProducesGraphOutput(node)) {
            quant_op_output = false;
          }
          // If conv weight is non-quantization, set conv to fp16
          if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1, 11})) {
            if (graph_utils::NodeArgIsConstant(graph_body, *node_inputs[1])) {
              quant_op_input = false;
              quant_op_output = false;
            }
          }

          if (quant_op_input || quant_op_output) {
            if (!quant_op_input && quant_op_output) {
              const std::string data_type = *(node_inputs[0]->Type());
              const std::string op_type = (data_type == "tensor(float)") ? "FLOAT32" : "FLOAT16";
              op_value.content.assign(op_type.c_str(), op_type.length());
            }
            if (quant_op_input && !quant_op_output) {
              const std::string data_type = *(node_outputs[0]->Type());
              const std::string op_type = (data_type == "tensor(float)") ? "FLOAT32" : "FLOAT16";
              op_value.content.assign(op_type.c_str(), op_type.length());
            }
            if (quant_op_input && quant_op_output) {
              const std::string op_type = "INT8";
              op_value.content.assign(op_type.c_str(), op_type.length());
            }
          } else {
            const std::string data_type = *(node_outputs[0]->Type());
            const std::string op_type = (data_type == "tensor(float)") ? "FLOAT32" : "FLOAT16";
            op_value.content.assign(op_type.c_str(), op_type.length());
          }
          op_param.fields.insert(std::make_pair("data_type", op_value));
          op_info.insert(std::make_pair(node.Name().c_str(), op_param));
          continue;
        }

        const ONNX_NAMESPACE::TensorProto* scale_tensor_proto;
        const ONNX_NAMESPACE::TensorProto* zp_tensor_proto;
        scale_tensor_proto = graph_utils::GetConstantInitializer(graph_body, node_inputs[1]->Name());
        zp_tensor_proto = graph_utils::GetConstantInitializer(graph_body, node_inputs[2]->Name());
        auto dim_size = scale_tensor_proto->dims_size();
        int64_t element_count = 1;
        if (dim_size > 0) {
          element_count = scale_tensor_proto->dims(0);
        }

        Initializer scale_ini{*scale_tensor_proto, graph_body.ModelPath()};
        Initializer zp_ini{*zp_tensor_proto, graph_body.ModelPath()};
        const float* scale_data = scale_ini.data<float>();
        const int8_t* zp_data = zp_ini.data<int8_t>();

        auto& quant_info = quant_param_info_.tensor_params;
        ::ppl::nn::QuantParam param;
        ::ppl::nn::QuantParam::Value scale_value, zero_point_value, per_channel_value;
        ::ppl::nn::QuantParam::Value tensor_max_value, tensor_min_value, bit_width_value;

        if (element_count == 1) {
          double scale = (double)scale_data[0];
          double zero_point = (double)zp_data[0];
          double tensor_max = 0;
          double tensor_min = 0;
          bool per_channel = false;
          scale_value.content.assign((const char*)&scale, sizeof(scale));
          zero_point_value.content.assign((const char*)&zero_point, sizeof(zero_point));
          tensor_max_value.content.assign((const char*)&tensor_max, sizeof(tensor_max));
          tensor_min_value.content.assign((const char*)&tensor_min, sizeof(tensor_min));
          per_channel_value.content.assign((const char*)&per_channel, sizeof(per_channel));
        } else {
          scale_value.content.clear();
          zero_point_value.content.clear();
          tensor_max_value.content.clear();
          tensor_min_value.content.clear();
          for (int64_t i = 0; i < element_count; i++) {
            double scale = (double)scale_data[i];
            double zero_point = (double)zp_data[i];
            double tensor_max = 0;
            double tensor_min = 0;
            scale_value.content.append((const char*)&scale, sizeof(scale));
            zero_point_value.content.append((const char*)&zero_point, sizeof(zero_point));
            tensor_max_value.content.append((const char*)&tensor_max, sizeof(tensor_max));
            tensor_min_value.content.append((const char*)&tensor_min, sizeof(tensor_min));
          }
          bool per_channel = true;
          per_channel_value.content.assign((const char*)&per_channel, sizeof(per_channel));
        }
        int64_t bit_width = 8;
        bit_width_value.content.assign((const char*)&bit_width, sizeof(bit_width));

        param.fields.insert(std::make_pair("scale", scale_value));
        param.fields.insert(std::make_pair("zero_point", zero_point_value));
        param.fields.insert(std::make_pair("tensor_max", tensor_max_value));
        param.fields.insert(std::make_pair("tensor_min", tensor_min_value));
        param.fields.insert(std::make_pair("per_channel", per_channel_value));
        param.fields.insert(std::make_pair("bit_width", bit_width_value));

        auto pair_input = quant_info.find(node_inputs[0]->Name());
        if (pair_input == quant_info.end()) {
          quant_info.insert(std::make_pair(node_inputs[0]->Name().c_str(), param));
        }
        auto pair_output = quant_info.find(node_outputs[0]->Name());
        if (pair_output == quant_info.end()) {
          quant_info.insert(std::make_pair(node_outputs[0]->Name().c_str(), param));
        }
      }
    }

    // Re-construct PPL Model from QDQ
    if (quantizer_) {
      onnxruntime::Graph& ppl_graph = const_cast<onnxruntime::Graph&>(graph_body);
      for (auto& node_idx : graph_body_viewer.GetNodesInTopologicalOrder()) {
        const auto* viewer_node_ptr = graph_body_viewer.GetNode(node_idx);
        auto* graph_node_ptr = ppl_graph.GetNode(node_idx);

        if (viewer_node_ptr == nullptr || graph_node_ptr == nullptr) {
          continue;
        }

        auto& node = *graph_node_ptr;
        if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "DequantizeLinear", {10, 13, 19}) ||
            graph_utils::IsSupportedOptypeVersionAndDomain(node, "QuantizeLinear", {10, 13, 19})) {
          converter_.RemoveNode(ppl_graph, node);
        }
      }

      const auto& metadef = graph_body_viewer.GetFilterInfo()->GetMetaDef();
      IndexedSubGraph ppl_filter_graph;
      for (const auto& node : ppl_graph.Nodes()) {
        if (node.OpType() == metadef->name) {
          // filter fused op type
          continue;
        }
        ppl_filter_graph.nodes.push_back(node.Index());
      }
      if (metadef) {
        auto meta_def_ppl = std::make_unique<IndexedSubGraph::MetaDef>();
        meta_def_ppl->name = metadef->name;
        meta_def_ppl->domain = metadef->domain;
        meta_def_ppl->since_version = metadef->since_version;
        meta_def_ppl->status = metadef->status;
        meta_def_ppl->inputs = metadef->inputs;
        meta_def_ppl->outputs = metadef->outputs;

        ppl_filter_graph.SetMetaDef(std::move(meta_def_ppl));
      }
      GraphViewer ppl_graph_view(ppl_graph, ppl_filter_graph);

      // print graph/node info
      // for (auto& node_idx : ppl_graph_view.GetNodesInTopologicalOrder()) {
      //   auto* graph_node_ptr = ppl_graph.GetNode(node_idx);
      //   auto& node = *graph_node_ptr;
      //   std::cout << node << std::endl;
      // }
      // std::cout << ppl_graph << std::endl;

      onnxruntime::Model model(ppl_graph.Name(), true, ModelMetaData(),
                            PathString(),
                            IOnnxRuntimeOpSchemaRegistryList(),
                            ppl_graph.DomainToVersionMap(),
                            std::vector<ONNX_NAMESPACE::FunctionProto>(),
                            *GetLogger());
      ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
      model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
      GraphViewerToProto(ppl_graph_view, *model_proto.mutable_graph(), true, true);
      model_proto_[fused_node.Name()] = model_proto;

      if (false) {
        // Save ppl Graph to onnx model file
        std::string onnx_string_buffer;
        model_proto.SerializeToString(&onnx_string_buffer);
        std::string model_name = "ppl.onnx";
        std::ofstream ofs(model_name);
        ofs.write(onnx_string_buffer.c_str(), onnx_string_buffer.size());
        ofs.close();

        // Save quantization param to json
        // TODO
      }
    } else {
      onnxruntime::Model model(graph_body_viewer.Name(), true, ModelMetaData(),
                            PathString(),
                            IOnnxRuntimeOpSchemaRegistryList(),
                            graph_body_viewer.DomainToVersionMap(),
                            std::vector<ONNX_NAMESPACE::FunctionProto>(),
                            *GetLogger());

      ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
      GraphViewerToProto(graph_body_viewer, *model_proto.mutable_graph(), true, true);
      model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
      model_proto_[fused_node.Name()] = model_proto;
    }

    input_info_[fused_node.Name()].push_back(input_map);
    output_info_[fused_node.Name()].push_back(output_map);
    // output_info_[fused_node.Name()].push_back(output_types);

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [this](ComputeContext* context, FunctionState* state) {
      ::ppl::common::RetCode ppl_status;
      ::ppl::nn::cuda::EngineOptions options;
      options.device_id = info_.device_id;
      // Default use cudaMaloc replace cuMemCreate (workaround VGG16 hang in C500)
      options.mm_policy = ::ppl::nn::cuda::MM_BEST_FIT;
      std::unordered_map<std::string, ppl::common::dataformat_t> io_data_format;
      for(auto p : *(this->tensor_data_layout_)){
          io_data_format[p.first] = LAYOUT_ORT_MAP_PPL[p.second];
      }
      options.io_data_format = static_cast<void*>(&io_data_format);
      std::queue<cudaStream_t> stream_queue = this->stream_queue_;
      if(MACART_EXECUTE_MODE!=1){
        options.external_stream_queue = static_cast<void*>(&stream_queue);
      }
      auto cuda_engine = std::unique_ptr<::ppl::nn::Engine>(::ppl::nn::cuda::EngineFactory::Create(options));
      // auto cuda_engine = ::ppl::nn::cuda::EngineFactory::Create(options);
      // If set to ture, would have low accuracy in C500 (SSD FP16/INT8)
      cuda_engine->Configure(::ppl::nn::cuda::ENGINE_CONF_USE_DEFAULT_ALGORITHMS, false);
      if(quantizer_) {
        char* qp_info = (char*)(&quant_param_info_);
        cuda_engine->Configure(::ppl::nn::cuda::ENGINE_CONF_SET_QUANT_PARAM_INFO, qp_info);
        // ::ppl::common::datatype_t kernel_type = ::ppl::common::DATATYPE_INT8;
        // cuda_engine->Configure(::ppl::nn::cuda::ENGINE_CONF_SET_KERNEL_TYPE, kernel_type);
      }

      std::unique_ptr<::ppl::nn::onnx::RuntimeBuilder::Resources> resources = std::make_unique<::ppl::nn::onnx::RuntimeBuilder::Resources>();
      resources->engines = new ::ppl::nn::Engine*[1];
      resources->engines[0] = cuda_engine.get();
      resources->engine_num = 1;

      auto builder = std::shared_ptr<::ppl::nn::onnx::RuntimeBuilder>(::ppl::nn::onnx::RuntimeBuilderFactory::Create());

      std::string string_buf;
      if (model_proto_[context->node_name].SerializeToString(&string_buf)) {
        ppl_status = builder->LoadModel(string_buf.data(), string_buf.size(), model_file_dir_);
      }
      ppl_status = builder->SetResources(*resources);
      ppl_status = builder->Preprocess();
      size_t pplruntime_pool_init_size = std::max(MACART_POOL_SIZE, 1);
      if(MACART_ENABLE_PROFILING != 0 ){
        LOGS_DEFAULT(WARNING) << "For enable profiling. Modify MACART_POOL_SIZE to 1";
        pplruntime_pool_init_size = 1;
      }
      std::shared_ptr<PPLRuntimePool> runtime_pool = nullptr;
      if(MACART_EXECUTE_MODE == 1){
        runtime_pool = std::make_shared<PPLStreamPPLRuntimePool>(builder,pplruntime_pool_init_size,1000);
      }else{
        runtime_pool = std::make_shared<ORTStreamPPLRuntimePool>(builder,stream_queue);
      }

      std::shared_ptr<PPLRuntimeWrapper> ppl_runtime_wrapper = runtime_pool->allocatePPLRuntimeWrapper(this->stream_queue_.front());
      auto runtime = ppl_runtime_wrapper->getPPLRuntime();
      int num_inputs = runtime->GetInputCount();
      int num_outputs = runtime->GetOutputCount();
      const std::unordered_map<std::string, size_t>& input_map = input_info_[context->node_name][0];
      const std::unordered_map<std::string, size_t>& output_map = output_info_[context->node_name][0];
      std::unordered_map<std::string, size_t> input_indexes(num_inputs);
      std::unordered_map<std::string, size_t> output_indexes(num_outputs);
      std::unordered_map<std::string, size_t> output_types(num_outputs);

      // Create input to index map
      std::unordered_map<std::string, std::vector<bool>> inputs_axis_dy;
      for (int i = 0; i < num_inputs; ++i) {
        std::vector<bool> input_axis_dy;
        auto input = runtime->GetInputTensor(i);
        const std::string& input_name = input->GetName();
        const auto& iter = input_map.find(input_name);
        if (iter != input_map.end()) {
          input_indexes[input_name] = iter->second;
        }

        auto& shape = *input->GetShape();
        for (size_t j = 0; j < shape.GetDimCount(); j++) {
            if (shape.GetDim(j) == ::ppl::common::INVALID_DIM_VALUE) { // -1
              input_axis_dy.push_back(true);
            }else{
              input_axis_dy.push_back(false);
            }
        }
        inputs_axis_dy[input_name] = input_axis_dy;
      }
      // Create output to index and type maps
      const auto& graph_output = model_proto_[context->node_name].graph().output();
      for (int i = 0; i < num_outputs; ++i) {
        const std::string& output_name = runtime->GetOutputTensor(i)->GetName();
        const auto& iter = output_map.find(output_name);
        if (iter != output_map.end()) {
          output_indexes[output_name] = iter->second;
        }
        const auto& tensor_type = graph_output[i].type().tensor_type();
        output_types[output_name] = tensor_type.elem_type();
      }
      ORT_UNUSED_PARAMETER(ppl_status);
      runtime_pool->recoverPPLRuntimeWrapper(this->stream_queue_.front());
      std::unique_ptr<MACAFuncState> p = std::make_unique<MACAFuncState>();
      *p = {
        std::move(cuda_engine),
        std::move(resources),
        std::move(builder),
        std::move(runtime_pool),
        input_indexes,
        output_indexes,
        output_types,
        inputs_axis_dy
      };
      *state = p.release();
      return 0;
    };

    // Release function state
    compute_info.release_state_func = [](FunctionState state) {
      if (state) {
        MACAFuncState* ppl_state = reinterpret_cast<MACAFuncState*>(state);
        if (ppl_state->resources) {
          if (ppl_state->resources->engines) {
            delete [] ppl_state->resources->engines;
          }
        }
        // delete ppl_state, will get error (TODO)
        // [MCR][E]mc_memory.cpp            :547 : bad ptr 0x7f631fe00000
        // [MCR][E]cuda_runtime_wrapper.cpp :1510: 29225: [7f63c5d00740] wcudaFree: Returned mcErrorInvalidValue
        delete static_cast<MACAFuncState*>(state);
      }
    };

    // Create compute function
    compute_info.compute_func = [this](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      CUDA_CALL_THROW(cudaSetDevice(this->info_.device_id));
      cudaStream_t compute_stream = static_cast<cudaStream_t>(this->GetComputeStream());
      Ort::CustomOpApi ort{*api};
      MACAFuncState* ppl_state = reinterpret_cast<MACAFuncState*>(state);
      const std::unordered_map<std::string, size_t>& input_indexes = ppl_state->input_info;
      const std::unordered_map<std::string, size_t>& output_indexes = ppl_state->output_info;
      ::ppl::common::RetCode ppl_status;
      std::shared_ptr<PPLRuntimeWrapper> ppl_runtime_wrapper = ppl_state->runtime_pool->allocatePPLRuntimeWrapper(static_cast<cudaStream_t>(this->GetComputeStream()));

      while(!ppl_runtime_wrapper){
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        ppl_runtime_wrapper = ppl_state->runtime_pool->allocatePPLRuntimeWrapper(static_cast<cudaStream_t>(this->GetComputeStream()));
      }

      auto runtime = ppl_runtime_wrapper->getPPLRuntime();
      ppl_runtime_wrapper->insertBegineEvent(compute_stream);
      for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);
        auto& shape = *t->GetShape();
        // shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);

        size_t input_index = 0;
        const auto& iter = input_indexes.find(t->GetName());
        if (iter != input_indexes.end()) {
          input_index = iter->second;
        }

        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        const auto& tensor_type = ort.GetTensorElementType(tensor_info);
        auto tensor_shape = ort.GetTensorShape(tensor_info);

        // set shape for dynamic
        if (shape.GetDimCount() == tensor_shape.size()) {
          for (size_t i = 0; i < shape.GetDimCount(); i++) {
            if (ppl_state->input_dy_axis_info[t->GetName()][i]) { // -1
              shape.SetDim(i, tensor_shape[i]);
            } else if (shape.GetDim(i) != tensor_shape[i]) {
              ORT_THROW("error, model input shape != input tensor shape");
            }
          }
        }
        // shape.Reshape(tensor_shape);

        // const void* input_buf = const_cast<const void*>(ort.GetTensorData<void>(input_tensor));
        // input tensors may require different data format
        // auto ppl_status = t->ConvertFromHost(input_buf, src_desc);
        void* input_buf = const_cast<void*>(ort.GetTensorData<void>(input_tensor));
        t->SetBufferPtr(input_buf);
      }


      for (uint32_t i = 0; i < runtime->GetOutputCount(); ++i) {
        auto output_ppl_tensor = runtime->GetOutputTensor(i);
        auto& shape = *output_ppl_tensor->GetShape();
        size_t output_index = 0;
        const auto& index_iter = output_indexes.find(output_ppl_tensor->GetName());
        if (index_iter != output_indexes.end()) {
          output_index = index_iter->second;
        }
        auto ouput_ort_value = ort.KernelContext_GetOriOutput(context, output_index);
        if(ouput_ort_value->IsAllocated()){
          auto tensor_info = ort.GetTensorTypeAndShape(ouput_ort_value);
          const auto& tensor_type = ort.GetTensorElementType(tensor_info);
          auto tensor_shape = ort.GetTensorShape(tensor_info);
          void* output_buf = const_cast<void*>(ort.GetTensorData<void>(ouput_ort_value));

          auto& shape = *output_ppl_tensor->GetShape();
          ORT_ENFORCE(shape.GetDimCount() == tensor_shape.size());
          for (size_t i = 0; i < shape.GetDimCount(); i++) {
            shape.SetDim(i, tensor_shape[i]);
          }
          output_ppl_tensor->SetBufferPtr(output_buf);
        }
      }

      if(MACART_ENABLE_PROFILING == 0){
        ppl_status = runtime->RunAsync();
        if (ppl_status !=  ppl::common::RC_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "runtime run failed: " << ppl::common::GetRetCodeStr(ppl_status);
        }
      }else{
        macaProfiling(runtime,&run_dur_,&run_count_);
      }


      for (uint32_t i = 0; i < runtime->GetOutputCount(); ++i) {
        auto output_ppl_tensor = runtime->GetOutputTensor(i);
        auto& shape = *output_ppl_tensor->GetShape();
        size_t output_index = 0;
        const auto& index_iter = output_indexes.find(output_ppl_tensor->GetName());
        if (index_iter != output_indexes.end()) {
          output_index = index_iter->second;
        }
        auto ouput_ort_value = ort.KernelContext_GetOriOutput(context, output_index);
        if(ouput_ort_value->IsAllocated()){
          void* ort_output_ptr = const_cast<void*>(ort.GetTensorData<void>(ouput_ort_value));
          void* ppl_output_ptr = output_ppl_tensor->GetBufferPtr();
          if(ort_output_ptr != ppl_output_ptr){
            auto tensor_info = ort.GetTensorTypeAndShape(ouput_ort_value);
            const auto& tensor_type = ort.GetTensorElementType(tensor_info);
            auto tensor_shape = ort.GetTensorShape(tensor_info);
            size_t bytes = std::accumulate(tensor_shape.begin(),
                                            tensor_shape.end(), 1,
                                            std::multiplies<int64_t>()) * OrtTypeMapLen[tensor_type];
            CUDA_CALL_THROW(cudaMemcpyAsync(ort_output_ptr, ppl_output_ptr, bytes, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(this->GetComputeStream())));
            // CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
          }
          static_cast<ppl::nn::TensorImpl*>(output_ppl_tensor)->FreeBuffer();
        }else{
          auto& shape = *output_ppl_tensor->GetShape();
          std::vector<int64_t> input64_output_shape;
          for (uint32_t j = 0; j < shape.GetDimCount(); ++j) {
              input64_output_shape.push_back(shape.GetDim(j));
          }
          ppl::nn::Device* ppl_device = static_cast<ppl::nn::TensorImpl*>(output_ppl_tensor)->GetDevice();
          if(static_cast<ppl::nn::TensorImpl*>(output_ppl_tensor)->IsBufferOwner() == false){
            ORT_THROW("error! Detached buffer not belong to ppl_allocator!");
          }
          ppl::nn::BufferDesc ppl_buffer_desc = static_cast<ppl::nn::TensorImpl*>(output_ppl_tensor)->DetachBuffer();
          // int test_buf[15];
          // cudaMemcpy((void*)test_buf, ppl_buffer_desc.addr, 60, cudaMemcpyDeviceToHost);
          // for(int tt =0;tt<15;tt++){
          //   std::cout << ((float*)test_buf)[tt] <<",";
          // }
          std::shared_ptr<IAllocator> ppl_allocator = std::make_shared<PPLOnceAllocator>(this->info_.device_id,ppl_device,ppl_buffer_desc,ppl_runtime_wrapper);
          ppl::common::datatype_t ppl_data_type = shape.GetDataType();
          auto ml_type = DataTypeImpl::TensorTypeFromONNXEnum(DATA_TYPE_PPL_MAP_ORT[ppl_data_type])->GetElementType();
          onnxruntime::TensorShape ort_shape(input64_output_shape.data(), input64_output_shape.size());
          Tensor::InitOrtValue(ml_type, ort_shape, ppl_allocator, *ouput_ort_value);
        }
      }
      ORT_UNUSED_PARAMETER(ppl_status);
      ppl_runtime_wrapper->insertEndEvent(compute_stream);
      if(ppl_runtime_wrapper->isInternalStream(compute_stream) == false){
        CUDA_CALL_THROW(cudaStreamAddCallback(static_cast<cudaStream_t>(this->GetComputeStream()),recoverPPLRuntimeCallback, static_cast<void*>(ppl_state->runtime_pool.get()), 0));
      }

      // ppl_state->runtime_pool->recoverPPLRuntimeWrapper(ppl_runtime_wrapper);
      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}
}  // namespace onnxruntime
