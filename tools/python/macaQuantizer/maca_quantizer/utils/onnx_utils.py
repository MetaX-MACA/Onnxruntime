
import os
import sys
import onnx
from typing import List, Optional, Any, Set
from onnx  import ValueInfoProto
from .utils import maca_error, maca_info, maca_warning
from .update_model_dim import update_inputs_outputs_dims



def get_prev_node_by_input(model, input_):
    n = model.graph.node[0]
    for node in model.graph.node:
        if input_ in node.output:
            return node, 0

    return n, -1

def get_next_node_by_output(model, output):
    n = model.graph.node[0]
    for node in model.graph.node:
        if output in node.input:
            return node, 0

    return n, -1

def get_all_next_node_by_output(model, output):
    node_list = []
    ok = -1

    for node in model.graph.node:
        if output in node.input:
            node_list.append(node)
            ok = 0

    return node_list, ok


def get_value_info_all(m: onnx.ModelProto, name: str) -> Optional[onnx.ValueInfoProto]:
        for v in m.graph.value_info:
            if v.name == name:
                return v

        for v in m.graph.input:
            if v.name == name:
                return v

        for v in m.graph.output:
            if v.name == name:
                return v

        return None


def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> List[int]:
    return [dim.dim_value for dim in v.type.tensor_type.shape.dim]


def init_dim_param_set(
        dim_param_set: Set[str], value_infos: List[onnx.ValueInfoProto]
    ) -> None:
        for info in value_infos:
            shape = info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.HasField("dim_param"):
                    dim_param_set.add(dim.dim_param)  # type: ignore


def update_dim(tensor: onnx.ValueInfoProto, dim: Any, j: int, name: str) -> None:
        dim_proto = tensor.type.tensor_type.shape.dim[j]
        if isinstance(dim, int):
            if dim >= 0:
                if dim_proto.HasField("dim_value") and dim_proto.dim_value != dim:
                    raise ValueError(
                        f"Unable to set dimension value to {dim} for axis {j} of {name}. Contradicts existing dimension value {dim_proto.dim_value}."
                    )
                dim_proto.dim_value = dim
            else:
                generated_dim_param = name + "_" + str(j)
                if generated_dim_param in dim_param_set:
                    raise ValueError(
                        f"Unable to generate unique dim_param for axis {j} of {name}. Please manually provide a dim_param value."
                    )
                dim_proto.dim_param = generated_dim_param
        elif isinstance(dim, str):
            dim_proto.dim_param = dim
        else:
            raise ValueError(
                f"Only int or str is accepted as dimension value, incorrect type: {type(dim)}"
            )


def modify_onnx2dynamic_(onnx_model):
    for idx in range(len(onnx_model.graph.input)):
        if len(onnx_model.graph.input[idx].type.tensor_type.shape.dim) > 0:
            dim_proto_input = onnx_model.graph.input[idx].type.tensor_type.shape.dim[0]
            # dim_proto_input.dim_param = 'bs'
            dim_proto_input.dim_value = -1

    for idx in range(len(onnx_model.graph.value_info)):
        tensor_proto = onnx_model.graph.value_info[idx]

        prev_node, flag = get_prev_node_by_input(onnx_model, tensor_proto.name)
        if prev_node.op_type == "DequantizeLinear" and flag != -1:
            _, flag_ = get_prev_node_by_input(onnx_model, prev_node.input[0])
            if flag_ == -1: continue

        if len(onnx_model.graph.value_info[idx].type.tensor_type.shape.dim) > 0:
            #  maca_info('value info name: {}'.format(onnx_model.graph.value_info[idx].name))
            dim_proto_input = tensor_proto.type.tensor_type.shape.dim[0]
            # dim_proto_input.dim_param = 'bs'
            dim_proto_input.dim_value = -1

    for idx in range(len(onnx_model.graph.output)):
        if len(onnx_model.graph.output[idx].type.tensor_type.shape.dim):
         dim_proto_output = onnx_model.graph.output[idx].type.tensor_type.shape.dim[0]
         # dim_proto_output.dim_param = 'bs'
         dim_proto_output.dim_value = -1

    #onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        maca_warning(f"Dynamic batch model check failed: {e}")
    else:
        maca_info('The model is modified!')

    return onnx_model

def modify_onnx2dynamic(model):
    dim_param_set: Set[str] = set()
    init_dim_param_set(dim_param_set, model.graph.input)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.output)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.value_info)  # type: ignore

    def update_dim(tensor: ValueInfoProto, dim: Any, j: int, name: str) -> None:
        if len(tensor.type.tensor_type.shape.dim) < 1:
            return
        dim_proto = tensor.type.tensor_type.shape.dim[j]
        if isinstance(dim, int):
            if dim >= 0:
                if dim_proto.HasField("dim_value") and dim_proto.dim_value != dim:
                    raise ValueError(
                        f"Unable to set dimension value to {dim} for axis {j} of {name}. Contradicts existing dimension value {dim_proto.dim_value}."
                    )
                dim_proto.dim_value = dim
            else:
                generated_dim_param = name + "_" + str(j)
                if generated_dim_param in dim_param_set:
                    raise ValueError(
                        f"Unable to generate unique dim_param for axis {j} of {name}. Please manually provide a dim_param value."
                    )
                dim_proto.dim_param = generated_dim_param
        elif isinstance(dim, str):
            dim_proto.dim_param = dim
        else:
            raise ValueError(
                f"Only int or str is accepted as dimension value, incorrect type: {type(dim)}"
            )

    for input_ in model.graph.input:
        input_name = input_.name
        update_dim(input_, "-1", 0, input_name)

    for output in model.graph.output:
        output_name = output.name
        update_dim(output, "-1", 0, output_name)

    # ipt_dict = {}
    # for ipt in onnx_model.graph.input:
    #     shape = get_shape_from_value_info_proto(ipt)
    #     shape[0] = -1
    #     ipt_dict[ipt.name] = shape

    # opt_dict = {}
    # for opt in onnx_model.graph.output:
    #     shape = get_shape_from_value_info_proto(opt)
    #     shape[0] = -1
    #     opt_dict[opt.name] = shape

    # update_inputs_outputs_dims(onnx_model, ipt_dict, opt_dict)


    for var in model.graph.value_info:
        var_name = var.name
        prev_node, flag = get_prev_node_by_input(model, var_name)
        if prev_node.op_type == "DequantizeLinear" and flag != -1:
            _, flag_ = get_prev_node_by_input(model, prev_node.input[0])
            if flag_ == -1: continue

        update_dim(var, "-1", 0, var_name)


    return model




