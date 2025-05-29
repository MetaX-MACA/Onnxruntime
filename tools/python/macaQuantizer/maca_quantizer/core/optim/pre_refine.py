

import torch
from typing import Iterable, List, Set

from maca_quantizer.core.config.config import ACTIVATIONS_TYPE, PPLMETAX_ACTIVATION, QUANT_COMPUTING_OP
from ppq.IR.search import SearchableGraph, TraversalCommand
from ppq import BaseGraphExecutor, BaseGraph,Operation, QuantableOperation, ALIGNMENT_MANUL_OVERRIDE
from ppq.api.interface import format_graph
from ppq.core import TargetPlatform, QuantizationStates
from ppq.executor.op.torch.base import  GET_ATTRIBUTE_FROM_OPERATION
from ppq.quantization.optim.base import QuantizationOptimizationPass
from maca_quantizer.utils.utils import maca_info, maca_warning


def is_post_op(graph, op_name):
    """
    与最后输出层没有计算节点的path, 就认为时post_op
    """
    flag = True
    search_engine = SearchableGraph(graph)
    op_set = search_engine.opset_matching(
                            sp_expr=lambda x: x.name==op_name, 
                            rp_expr=lambda x, y: True, 
                            ep_expr=lambda x: x.is_boundary, 
                            direction='down')

    op_type_list = [x.type for x in op_set]

    for op_type in op_type_list:
        if op_type in QUANT_COMPUTING_OP:
            flag = False
            break
    return flag


class MetaxDispatchPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Metax Dispatch Post Operation Optimztion')

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        self.dispatch_alignment(graph)
        self.dispatch_postpro(graph)

    def dispatch_alignment(self, graph: BaseGraph) -> None:
        for operation in graph.operations.values():
            if operation.type in {'AveragePool', 'MaxPool'}:
                pooling_1x1 = True
                kernel_size = GET_ATTRIBUTE_FROM_OPERATION(op=operation, attribute='kernel_shape', compulsive=True)
                stride      = GET_ATTRIBUTE_FROM_OPERATION(op=operation, attribute='strides', default=None) 

                stride_ = stride if stride is None else [1] * len(kernel_size)
                for i in range(len(kernel_size)):
                    if kernel_size[i] != 1 or stride_[i] != 1:
                        pooling_1x1 = False
                if pooling_1x1:
                    operation.set_extension_attrib(ALIGNMENT_MANUL_OVERRIDE, 'Align to Output')
                    print()
                    maca_info(f"{operation.name} customize Align to Output.")


    def dispatch_postpro(self, graph: BaseGraph) -> None:
        search_engine = SearchableGraph(graph)

        optimize_opset = set()
        optimize_computer_opset = set()
        boundary_opset = set()

        traversal_pairs = []
        for _, var in graph.outputs.items():
            if var.source_op is None:
                continue
            if var.source_op.is_computing_op:
                optimize_computer_opset.add(var.source_op.name)
                boundary_opset.add(var.source_op.name)
                continue
            paths = search_engine.path_matching(
                                sp_expr=lambda x: x.type in QUANT_COMPUTING_OP, 
                                rp_expr=lambda x, y: (y.type not in QUANT_COMPUTING_OP), 
                                ep_expr=lambda x: x.name==var.source_op.name, 
                                direction='down')
            for path in paths:
                path = path.tolist()
                boundary_opname = (path[0].name, path[-1].name)
                if boundary_opname not in traversal_pairs:
                    traversal_pairs.append(boundary_opname)
                # maca_info("\n PostProcess operations: {} ---> {}".format(path[0].name, path[-1].name))
                optimize_computer_opset.add(path[0].name)
                for idx, op in enumerate(path[1:]): 
                    if is_post_op(graph, op.name):
                        optimize_opset.add(op.name)

        for op_name in optimize_computer_opset:
            # TODO: process fusion activation operation
            down_ops = graph.get_downstream_operations(graph.operations[op_name])
            # if len(down_ops) == 1 and down_ops[0].type in ACTIVATIONS_TYPE:
            if len(down_ops) == 1 and down_ops[0].type in {'Relu', 'Clip', 'PRelu', 'LeakyRelu'}:
                down_ops[0].set_extension_attrib("PostPro_start", True)
            else:
                graph.operations[op_name].set_extension_attrib("PostPro_start", True)

        for op_name in optimize_opset:
            if graph.operations[op_name].extension_attrib.get("PostPro_start", False):
                continue
            graph.operations[op_name].set_extension_attrib("PostPro", True)


        print()
        for op_names in  traversal_pairs:
            maca_info("PostProcess operations: {} ---> {}".format(op_names[0], op_names[1]))



# TODO: to be tested
class FuseConvMulPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Fuse ConvConstMul Optimztion')

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        fuse_conv_type = {'Conv', 'Gemm', 'ConvTranspose'}
        interested_pairs = []
        for operation in graph.operations.values():
            if operation.type in fuse_conv_type:
                down_ops = graph.get_downstream_operations(operation)
                if len(down_ops)==1 and down_ops[0].type=='Mul':
                    interested_pairs.append((operation, down_ops[0]))

        for pair in interested_pairs:
            conv, mul = pair[0], pair[1]
            mul_const_var = []
            c_index = 1 if operation.type in {'ConvTranspose'} else 0
            for var in mul.inputs:
                if var.is_parameter:
                    mul_const_var.append(var)
            if len(mul_const_var) != 1: continue
            const_value = mul_const_var[0].value

            # check mul input 
            if not conv.inputs[1].is_parameter: continue
            if len(conv.inputs) > 2 and not conv.inputs[2].is_parameter: continue 
            if len(conv.inputs[2].value.shape) != 1: continue

            if len(const_value.shape)==0: #  constant single date 
                pass
            elif len(const_value.shape)==1: # 1d constant mul can not merge
                continue
            elif len(const_value.shape)>1:  # broadcast value
                if const_value.shape[1] != conv.inputs[1].value.shape[c_index]: continue
                if const_value.flatten().shape[0] != conv.inputs[1].value.shape[c_index]: continue
                broadcast_shape= [conv.inputs[1].value.shape[i] if i==c_index else 1 for i in range(len(conv.inputs[1].value.shape))]
                const_value = const_value.reshape(broadcast_shape)

            conv.inputs[1].value = conv.inputs[1].value * const_value
            if len(conv.inputs) > 2:
                conv.inputs[2].value = conv.inputs[2].value * const_value.flatten()

            input_var = conv.outputs[0]
            output_var = mul.outputs[0]

            for var in mul.inputs + mul.outputs:
                if var != input_var and var != output_var:
                    graph.remove_variable(var)

            graph.remove_operation(mul)
            graph.create_link_with_var(input_var, output_var)



class MetaxFormatGemmPass(QuantizationOptimizationPass):
    def __init__(self, name: str = 'MacaRT Format Gemm Pass') -> None:
        super().__init__(name)

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:

        for op in graph.operations.values():
            if op.type == 'Gemm':
                try:
                    transB = GET_ATTRIBUTE_FROM_OPERATION(op, 'transB', default=0)
                    if transB == 0:
                        op.attributes['transB'] = 1
                        weight = op.parameters[0].value
                        assert isinstance(weight, torch.Tensor)
                        op.parameters[0].value = weight.transpose(1, 0).contiguous()
                except Exception as e:
                    maca_warning("MetaxFormatGemmPass Failed, msg: {e}")



# ================================== QuantizationOptimizationPass for build_prequant_pipeline  ==================================


class MetaxConvElementwiseActivation(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Dispatch ConvElementwise Activation Operation Optimztion')
        self._fuse_activation_type = PPLMETAX_ACTIVATION.copy()

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        visited_elementwise = []
        # TODO: Clip not support fuse 
        if 'Clip' in self._fuse_activation_type:
            self._fuse_activation_type.remove('Clip')

        self._fuse_v1(graph=graph, visited_ops=visited_elementwise)
        self._fuse_v2(graph=graph, visited_ops=visited_elementwise)
        # print(visited_elementwise)
        # print(self._fuse_activation_type)


    def _fuse_v1(self, graph: BaseGraph, visited_ops=[]) -> None:
        search_engine = SearchableGraph(graph)
        match_nodes_list = []
        for operation in graph.operations.values():
            if operation.type not in {'Add', 'Mul'}: continue
            patterns = search_engine.pattern_matching(patterns=['Conv', 
                                                                lambda x: x.type in self._fuse_activation_type, 
                                                                lambda x: x.name == operation.name ],
                                                        edges=[[0, 1], [1,2]], exclusive=False)

            # TODO: merge code to "select_fuse_branch" interface
            match_nodes = []
            for pattern in patterns:
                # op_name_list = [op.name for op in pattern]
                conv, act, elementwise = pattern
                down_ops = graph.get_downstream_operations(elementwise)
                if elementwise.name in visited_ops: continue  # avoid duplicate 
                if len(graph.get_downstream_operations(act)) != 1:
                    continue
                if all([op in pattern for op in graph.get_upstream_operations(elementwise)]):
                    continue
                # Dnn interface not support elementwise broadcast
                in_shape_list = [var.shape for var in elementwise.inputs]
                broadcast_enable = all([shape == in_shape_list[0] for shape in in_shape_list])
                if not broadcast_enable: 
                    continue

                match_nodes = [conv, act, elementwise] 
                if len(down_ops) == 1 and  down_ops[0].type in self._fuse_activation_type:  # add act2
                    match_nodes.append(down_ops[0])
                visited_ops.append(elementwise.name)
                # print([op.name for op in match_nodes])

            if len(match_nodes) > 0:
                match_nodes_list.append(match_nodes)

        # print(match_nodes_list)
        for fuse_nodes in match_nodes_list:
            for idx, operation in enumerate(fuse_nodes):
                if operation.name not in graph.operations: continue
                if not isinstance(operation, QuantableOperation): continue

                if idx == 0: 
                    for config in operation.output_quant_config:
                        config.state = QuantizationStates.FP32
                elif idx == len(fuse_nodes)-1 : 
                    # get other input index for elementwise
                    for var in operation.inputs:
                        if var.source_op is None or var.is_parameter:
                            continue
                        if var.source_op == fuse_nodes[-2]:
                            operation.input_quant_config[operation.inputs.index(var)].state = QuantizationStates.FP32
                else:
                    operation.platform = TargetPlatform.FP32

                # Add extension attribute avoid operation convert fp16
                operation.set_extension_attrib("ConvEleAct", True)


    @staticmethod
    def select_fuse_branch_v2(graph: BaseGraph, pattern_list: List):
        if len(pattern_list) < 1:
            return []
        search_engine = SearchableGraph(graph)
        selected = pattern_list[0]  
        for pattern in pattern_list:
            conv = pattern[0]
            elementwise = pattern[-1] 
            conv_output_var = conv.outputs[0]
            op_set = search_engine.opset_matching(
                                    sp_expr=lambda x: x.name == conv.name, 
                                    rp_expr=lambda x, y: True, 
                                    ep_expr=lambda x: x.name == elementwise.name, 
                                    direction='down')

            if all([op in op_set for op in graph.get_upstream_operations(elementwise)]):
                selected = []
                continue
            # TODO: conv_output_var.dest_op > 1 ?????
            if len(conv_output_var.dest_ops) == 1:
                selected = pattern
                break 
        return selected



    def _fuse_v2(self, graph: BaseGraph, visited_ops=[]) -> None:
        search_engine = SearchableGraph(graph)
        match_nodes_list = []

        for operation in graph.operations.values():
            if operation.type not in {'Add', 'Mul'}: continue
            patterns = search_engine.pattern_matching(patterns=['Conv', 
                                                                lambda x: x.name == operation.name ],
                                                        edges=[[0, 1]], exclusive=False)
            # TODO: choice branch to merge
            pattern = self.select_fuse_branch_v2(graph, patterns)

            if len(pattern) < 1: 
                continue

            # op_name_list = [op.name for op in pattern]
            # print(op_name_list)
            conv, elementwise = pattern
            down_ops = graph.get_downstream_operations(elementwise)

            match_nodes = [conv, elementwise] 
            if elementwise.name in visited_ops: continue  # avoid duplicate 
            if len(down_ops) == 1 and  down_ops[0].type in self._fuse_activation_type: # add act2
                match_nodes.append(down_ops[0])
            if all([op in pattern for op in graph.get_upstream_operations(elementwise)]):
                    continue
            # Dnn interface not support elementwise broadcast
            in_shape_list = [var.shape for var in elementwise.inputs]
            broadcast_enable = all([shape == in_shape_list[0] for shape in in_shape_list])
            if not broadcast_enable: 
                continue

            # print([op.name for op in match_nodes])
            visited_ops.append(elementwise.name)
            match_nodes_list.append(match_nodes)


        for fuse_nodes in match_nodes_list:
            assert len(fuse_nodes) in {2, 3}
            for idx, operation in enumerate(fuse_nodes):
                if operation.name not in graph.operations: continue
                if not isinstance(operation, QuantableOperation): continue

                if idx == 0:
                    for config in operation.output_quant_config:
                        config.state = QuantizationStates.FP32
                elif idx == 1: 
                    # get other input index for elementwise

                    for var in operation.inputs:
                        if var.source_op is None or var.is_parameter:
                            continue
                        if var.source_op == fuse_nodes[0]:
                            operation.input_quant_config[operation.inputs.index(var)].state = QuantizationStates.FP32

                    if len(fuse_nodes) == 3:
                        for config in operation.output_quant_config:
                            config.state = QuantizationStates.FP32
                else:
                    for config in operation.input_quant_config:
                        config.state = QuantizationStates.FP32



class MetaxRemoveUselessPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Remove Useless Operation Optimztion')
        self._fusion_type = {'Slice', 'Resize', 'Pad', 'AveragePool',}

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        self.remove_useless_node(graph)
        format_graph(graph)

    def canFuse(self, graph: BaseGraph, operation: Operation):
        if operation.type not in self._fusion_type:
            return False
        if operation.inputs[0].shape != operation.outputs[0].shape:
            return False
        if not all([var.is_parameter  for var in operation.inputs[1:]]): # Other parameters must be constants
            return False
        if len(graph.get_upstream_operations(operation)) > 1:
            return False
        if operation.type == 'AveragePool':
            stride       = GET_ATTRIBUTE_FROM_OPERATION(op=operation, attribute='strides', default=[1])
            kernel_size = GET_ATTRIBUTE_FROM_OPERATION(op=operation, attribute='kernel_shape', compulsive=True)
            if any(list(map(lambda x:x!=1, stride))) or any(list(map(lambda x:x!=1, kernel_size))):
                return False

        return True


    def remove_useless_node(self, graph: BaseGraph, **kwargs) -> None:

        selected_operartion = []
        for operation in graph.operations.values():
            if self.canFuse(graph, operation):
                selected_operartion.append(operation.name)

        for op_name in selected_operartion:
            operation = graph.operations[op_name]
            # up_op -- slice -- down_op
            # Delete operator 
            link_var_1 = operation.inputs[0]
            link_var_2 = operation.outputs[0]
            link_var_1.dest_ops.pop(link_var_1.dest_ops.index(operation))
            link_var_2.source_op = None

            operation.outputs.pop(operation.outputs.index(link_var_2))
            # graph.remove_variable(operation.inputs[0])
            graph.remove_operation(operation)

            graph.create_link_with_var(link_var_1, link_var_2)



class MetaxTransposeBetweenPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Remove Useless Transpose Operation Optimztion')
        self._fusion_type = {'LeakyRelu', 'Relu', 'PRelu', 'Swish', 'HardSwish', 'Sigmoid', 'HardSigmoid', 'Clip'}

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        """
                    * - Transpose - [fusion_operator] - Transpose - *  
            
            ===>    * - [fusion_operator]  - *

        """
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(patterns=['Transpose', lambda x: x.type in  self._fusion_type,'Transpose'],
                                                        edges=[[0, 1], [1,2]], exclusive=True)
        
        for pattern in patterns:
            attr_1 = GET_ATTRIBUTE_FROM_OPERATION(op=pattern[0], attribute='perm') 
            attr_2 = GET_ATTRIBUTE_FROM_OPERATION(op=pattern[2], attribute='perm') 
            input_shape = pattern[0].inputs[0].shape
            output_shape = pattern[-1].outputs[0].shape
            if output_shape is None or input_shape is None or input_shape !=output_shape: continue            
            if attr_1 != attr_2: continue
            self.fuse_transpose_between(graph, pattern)

    def fuse_transpose_between(self, graph: BaseGraph, pattern: List[Operation]):
        transpose_1, remain_op, transpose_2 = pattern

        input_var = transpose_1.inputs[0]
        output_var = transpose_2.outputs[0]

        input_var.dest_ops[input_var.dest_ops.index(transpose_1)] = remain_op
        remain_op.inputs[0] = input_var
        transpose_1.inputs.clear()

        output_var.source_op = remain_op
        remain_op.outputs[0] = output_var
        transpose_2.outputs.clear()

        graph.remove_operation(transpose_1)
        graph.remove_operation(transpose_2)

