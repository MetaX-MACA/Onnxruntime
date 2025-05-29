

import onnx
import numpy as np
from onnx import helper, numpy_helper
from typing import Any, Dict, Iterable, List, Union

from ppq.IR import BaseGraph, GraphBuilder, Operation, Opset, Variable
from ppq.parser.onnx_parser import OnnxParser
from ppq.core import (DEFAULT_OPSET_DOMAIN, DEFAULT_OPSET_VERSION,
                      GRAPH_OPSET_ATTRIB, NetworkFramework, is_file_exist)
from ...utils.utils import maca_info, maca_warning

class MACAOnnxParser(OnnxParser):

    def mapping_upsample(self, graph, node, op_name, opset_version):
        attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        input_names = [var_name for var_name in node.input]
        output_names = [var_name for var_name in node.output]
        extra_initializer = {}

        extra_initializer[f'{op_name}_roi'] = np.array([], dtype=np.float32)
        input_names.insert(1, f'{op_name}_roi')

        if opset_version == 7:
            extra_initializer[f'{op_name}_scales'] = numpy_helper.to_array(attrs['scales'])
            input_names.append(f'{op_name}_scales')

        map_op = Operation(
            name=op_name, op_type="Resize",
            attributes={'mode': attrs['mode']},
            opset=Opset(domain=DEFAULT_OPSET_DOMAIN, version=11)
        )
        return map_op, input_names, output_names, extra_initializer

    def initialize_params(self, graph: BaseGraph, initializer: Dict[str, Any]) -> BaseGraph:
        for var in graph.variables.values():
            if var.name in initializer:
                for dest_op in var.dest_ops:
                    assert isinstance(dest_op, Operation)
                    dest_op.parameters.append(var)
                var.value = initializer[var.name]
                var.is_parameter = True
        return graph

    def build(self, file_path: str) -> BaseGraph:
        _rand_seed = 0 # used for name generation.
        if not is_file_exist(file_path):
            raise FileNotFoundError(f'file {file_path} does not exist, or it is a directory.')
        model_pb = onnx.load(file_path)
        model_pb = self.simplify_model(model_pb)
        # model_pb = self.maca_simplify_model(model_pb, tofile=False)

        opsets = model_pb.opset_import

        assert isinstance(model_pb, onnx.ModelProto), \
            f'onnx load failed, only ProtoBuffer object is expected here, while {type(model_pb)} is loaded.'
        graph_pb = model_pb.graph
        graph = BaseGraph(name=graph_pb.name, built_from=NetworkFramework.ONNX)
        graph._detail[GRAPH_OPSET_ATTRIB] = self.convert_opsets_to_str(opsets)
        graph._detail['ir_version'] = model_pb.ir_version

        all_import_opset = {}
        onnx_import_opset = DEFAULT_OPSET_VERSION
        for opset in graph._detail[GRAPH_OPSET_ATTRIB]:
            if opset['domain'] == DEFAULT_OPSET_DOMAIN or opset['domain'] == '':
                onnx_import_opset = opset['version']
                all_import_opset[DEFAULT_OPSET_DOMAIN] = opset['version']
                # break
            else:
                all_import_opset[opset['domain']] = opset['version']

        # a temporary storage for operation's inputs and outputs
        op_inputs_dict, op_outputs_dict = {}, {}
        extra_initializer = {}
        for node in graph_pb.node:
            op_name = node.name
            if len(op_name) == 0: # some operation do not have a name, we just generate one.
                # op_name = 'generated_name_' + str(_rand_seed)
                op_name = f'mx.{node.op_type}_'.lower() + str(_rand_seed)
                _rand_seed += 1

            if op_name in graph.operations:
                raise KeyError(f'Duplicated operation {op_name} was found.')

            if node.op_type=="Upsample":
                create_op, input_names, output_names, extra_params = \
                        self.mapping_upsample(graph, node, op_name, onnx_import_opset)
                extra_initializer.update(extra_params)
                graph.operations[op_name] = create_op
                op_inputs_dict[op_name] = input_names
                op_outputs_dict[op_name] = output_names

            else:
                if node.domain in all_import_opset:
                    node_opset = Opset(domain=node.domain, version=all_import_opset[node.domain])
                else:
                    node_opset=Opset(domain=DEFAULT_OPSET_DOMAIN, version=onnx_import_opset)

                graph.operations[op_name] = Operation(
                    name=op_name, op_type=node.op_type,
                    attributes={item.name: helper.get_attribute_value(item) for item in node.attribute},
                    opset=node_opset
                )
                # op_inputs_dict[op_name] = [var_name for var_name in node.input]
                op_inputs_dict[op_name] = [op_name + f'_input_{list(node.input).index(var_name)}' if var_name=="" else var_name 
                                            for var_name in node.input]
                op_outputs_dict[op_name] = [var_name for var_name in node.output]

        initializer = {}
        for item in graph_pb.initializer:
            init_name = item.name
            value = numpy_helper.to_array(item)
            initializer[init_name] = value
        initializer.update(extra_initializer)

        inputs  = [item.name for item in graph_pb.input]
        outputs = [item.name for item in graph_pb.output]
        graph = self.build_variables(
            graph, graph_inputs=inputs, graph_outputs=outputs,
            op_inputs=op_inputs_dict, op_outputs=op_outputs_dict)
        graph = self.initialize_params(graph, initializer)
        self.de_inplace(graph)
        return self.refine_graph(graph)
    
    @staticmethod
    def simplify_model(model_proto: onnx.ModelProto, tofile=False) -> onnx.ModelProto:
        from onnxsim.onnx_simplifier import simplify
        model_fn =  "per-quantize.onnx"

        try:
            model_opt, check_ok = simplify(model_proto)
            if check_ok:
                maca_info(f"Pre-Simplify Sucess")
                if tofile:
                    onnx.save_model(model_proto, model_fn)
                    maca_info(f"Pre-Simplify model file: {model_fn}")
                return model_opt
        except Exception as e:
            maca_warning(f"Pre-Simplify Fail: \n {e}")

        return model_proto


    @staticmethod
    def maca_simplify_model(model_proto: onnx.ModelProto, tofile=False) -> onnx.ModelProto:
        from maca_converter.fuse_mha import mha_fuse
        from maca_converter.gelu_fuse import merge_gelu
        from maca_converter.fuse import fuse_transpose_relu_transpose
        model_fn =  "per-quantize_maca.onnx"

        # model_proto = mha_fuse(model_proto)
        # maca_info(f"Fusion maca MultiHeadAttentionV1 operator")

        model_proto = merge_gelu(model_proto) 
        maca_info(f"Fusion maca Gelu operator")

        model_proto = fuse_transpose_relu_transpose(model_proto) 
        maca_info(f"Fusion maca Gelu operator")

        if tofile:
            onnx.save_model(model_proto, model_fn)
            maca_info(f"Pre-Simplify model file: {model_fn}")
        return model_proto

