
import os
import sys
import torch
from typing import Any, Iterable, List, Optional, Set, Dict, Optional, Union
from ...ppq_.ppq.executor.op.torch.base import GET_ATTRIBUTE_FROM_OPERATION
from ...ppq_.ppq.api import format_graph

from ppq.IR.search import SearchableGraph, Operation

from ppq.quantization.optim.base import QuantizationOptimizationPass
from ppq import (BaseGraphExecutor, BaseGraph, COMPUTING_OP, convert_any_to_python_primary_type, 
                 convert_any_to_numpy)
from maca_quantizer.utils.utils import maca_info, maca_warning
from maca_quantizer.core.config.config import ACTIVATIONS_TYPE, PPLMETAX_ACTIVATION, DISPATCH_FP32_ATTRIB_KEY
from ..parser.macaruntime_exporter import MACARUNTIMExporter



class ComposeSwishPass(QuantizationOptimizationPass):
    """
    合成 Swish

            |--------  Sigmoid  ---------|
        ----|                            Mul ---
            |----------------------------|
            and
            beta---|
                   Mul--- Sigmoid ---|
            |------|                 Mul ----
        ----|------------------------|
    """

    def __init__(self) -> None:
        super().__init__('Compose Swish Operation Optimztion')

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        self.compose_swish_v1(graph)
        self.compose_swish_v2(graph)

    @staticmethod
    def compose_swish_v1(graph: BaseGraph):
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(patterns=[lambda x: True, 'Sigmoid', 'Mul'],
                                                  edges=[[0, 1], [1, 2], [0, 2]],
                                                  exclusive=True)

        for pattern in patterns:
            if not int(os.environ.get("MXQ_USE_MACART_BACKEND", 1)):
                for op in pattern:
                    op.set_extension_attrib(DISPATCH_FP32_ATTRIB_KEY, 1)
                continue

            computing, sigmoid, mul = pattern
            assert isinstance(computing, Operation)
            assert isinstance(sigmoid, Operation)
            assert isinstance(mul, Operation)
            swish_attributes = {}
            #swish_attributes['beta'] = 1.0
            swish_op = graph.create_operation(op_type='Swish',
                                              name=f'{computing.name}_Swish',
                                              attributes=swish_attributes,
                                              platform=computing.platform)
            swish_in_var = None
            for index in range(len(computing.outputs)):
                if sigmoid in computing.outputs[index].dest_ops:
                    swish_in_var = computing.outputs[index]
                    break
            assert swish_in_var is not None
            swish_in_var.dest_ops.append(swish_op)
            swish_op.inputs.append(swish_in_var)
            assert len(sigmoid.outputs) == 1
            graph.remove_variable(sigmoid.outputs[0])
            graph.remove_operation(sigmoid)
            assert len(mul.outputs) == 1
            swish_out_var = mul.outputs[0]
            graph.remove_operation(mul)
            swish_out_var.source_op = swish_op
            swish_op.outputs.append(swish_out_var)

    @staticmethod
    def compose_swish_v2(graph: BaseGraph):
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(patterns=[lambda x: True, 'Mul', 'Sigmoid', 'Mul'],
                                                  edges=[[0, 1], [1, 2], [2,3], [0, 3]],
                                                  exclusive=True)
        for pattern in patterns:
            if not int(os.environ.get("MXQ_USE_MACART_BACKEND", 1)):
                for op in pattern:
                    op.set_extension_attrib(DISPATCH_FP32_ATTRIB_KEY, 1)
                continue

            computing, mul_1, sigmoid, mul_2 = pattern
            value_check = convert_any_to_python_primary_type(mul_1.inputs[1].value) == 1
            if not value_check: continue

            swish_attributes = {}
            #swish_attributes['beta'] = 1.0
            swish_op = graph.create_operation(op_type='Swish',
                                              name=f'{computing.name}_Swish',
                                              attributes=swish_attributes,
                                              platform=computing.platform)
            swish_in_var = None
            for index in range(len(computing.outputs)):
                if mul_1 in computing.outputs[index].dest_ops:
                    swish_in_var = computing.outputs[index]
                    break
            assert swish_in_var is not None
            swish_in_var.dest_ops.append(swish_op)
            swish_op.inputs.append(swish_in_var)
            assert len(mul_1.outputs) == 1
            assert len(mul_2.outputs) == 1
            swish_out_var = mul_2.outputs[0]
            graph.remove_operation(mul_2)

                        # delete merged ops
            for op in {mul_1, sigmoid}:
                assert isinstance(op, Operation)
                for var in op.inputs + op.outputs:
                    if var != swish_in_var and var != swish_out_var:
                        graph.remove_variable(var)
                graph.remove_operation(op)

            swish_out_var.source_op = swish_op
            swish_op.outputs.append(swish_out_var)



class ComposeMishPass(QuantizationOptimizationPass):
    """
    合成 Mish

        |--- SoftPlus --- Tanh ---|
    ----|                        Mul ---
        |-------------------------|

    """

    def __init__(self) -> None:
        super().__init__('Compose Mish Operation Optimztion')

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(patterns=[lambda x: x.is_computing_op, 'Softplus', 'Tanh', 'Mul'],
                                                  edges=[[0, 1], [1, 2], [2, 3], [0, 3]],
                                                  exclusive=True)
        for pattern in patterns:
            computing, softplus, tanh, mul = pattern
            assert isinstance(computing, Operation)
            assert isinstance(softplus, Operation)
            assert isinstance(tanh, Operation)
            assert isinstance(mul, Operation)

            mish_op = graph.create_operation(op_type='Mish',
                                             name=f'{computing.name}_Mish',
                                             platform=computing.platform)
            mish_in_var = None
            for index in range(len(computing.outputs)):
                if softplus in computing.outputs[index].dest_ops:
                    mish_in_var = computing.outputs[index]
                    break
            assert mish_in_var is not None
            mish_in_var.dest_ops.append(mish_op)
            mish_op.inputs.append(mish_in_var)
            assert len(softplus.outputs) == 1
            graph.remove_variable(softplus.outputs[0])
            graph.remove_operation(softplus)
            assert len(tanh.outputs) == 1
            graph.remove_variable(tanh.outputs[0])
            graph.remove_operation(tanh)
            assert len(mul.outputs) == 1
            mish_out_var = mul.outputs[0]
            graph.remove_operation(mul)
            mish_out_var.source_op = mish_op
            mish_op.outputs.append(mish_out_var)


class ComposeHardSwishPass(QuantizationOptimizationPass):
    """
    合成 HardSwish

            |------  HardSigmoid  -------|
        ----|                            Mul ---
            |----------------------------|

    &&
            |--- Add --- Clip --- Mul --- Div ---
        ----|                      |
            |----------------------|

    """

    def __init__(self) -> None:
        super().__init__('Compose HardSwish Operation Optimztion')

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        self.compose_hardswish_v1(graph)
        self.compose_hardswish_v2(graph)

    @staticmethod
    def compose_hardswish_v1(graph):
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(patterns=[lambda x: True, 'HardSigmoid', 'Mul'],
                                                  edges=[[0, 1], [1, 2], [0, 2]],
                                                  exclusive=True)

        for pattern in patterns:
            if not int(os.environ.get("MXQ_USE_MACART_BACKEND", 1)) and MACARUNTIMExporter().required_opsets['ai.onnx'] < 14:
                for op in pattern:
                    op.set_extension_attrib(DISPATCH_FP32_ATTRIB_KEY, 1)
                continue
            computing, hardsigmoid, mul = pattern
            assert isinstance(computing, Operation)
            assert isinstance(hardsigmoid, Operation)
            assert isinstance(mul, Operation)
            hardswish_op = graph.create_operation(op_type='HardSwish',
                                              name=f'{computing.name}_HardSwish',
                                              platform=computing.platform)
            hardswish_in_var = None
            for index in range(len(computing.outputs)):
                if hardsigmoid in computing.outputs[index].dest_ops:
                    hardswish_in_var = computing.outputs[index]
                    break
            assert hardswish_in_var is not None
            hardswish_in_var.dest_ops.append(hardswish_op)
            hardswish_op.inputs.append(hardswish_in_var)
            assert len(hardsigmoid.outputs) == 1
            graph.remove_variable(hardsigmoid.outputs[0])
            graph.remove_operation(hardsigmoid)
            assert len(mul.outputs) == 1
            hardswish_out_var = mul.outputs[0]
            graph.remove_operation(mul)
            hardswish_out_var.source_op = hardswish_op
            hardswish_op.outputs.append(hardswish_out_var)


    @staticmethod
    def compose_hardswish_v2(graph: BaseGraph):
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(patterns=[lambda x: True, 'Add', 'Clip', 'Mul', 'Div'],
                                                  edges=[[0, 1], [1, 2], [2,3], [3, 4],[0, 3]],
                                                  exclusive=True)
        for pattern in patterns:
            if not int(os.environ.get("MXQ_USE_MACART_BACKEND", 1)) and MACARUNTIMExporter().required_opsets['ai.onnx'] < 14:
                for op in pattern:
                    op.set_extension_attrib(DISPATCH_FP32_ATTRIB_KEY, 1)
                continue

            start_op, add, clip, mul, div = pattern
            # check operator value
            check_value_1 = convert_any_to_python_primary_type(add.inputs[1].value) == 3
            check_value_2 = convert_any_to_python_primary_type(div.inputs[1].value) == 6
            type_check = add.inputs[1].is_parameter and div.inputs[1].is_parameter
            value_check = check_value_1 and check_value_2
            if not type_check or not value_check:
                continue

            middle_ops = [add, clip, mul]
            hardswish_op = graph.create_operation(op_type='HardSwish',
                                              name=f'{start_op.name}_HardSwish',
                                              platform=start_op.platform)

            # process input_var and output_var
            hardswish_in_var = None
            for index in range(len(start_op.outputs)):
                if add in start_op.outputs[index].dest_ops:
                    hardswish_in_var = start_op.outputs[index]
                    break
            assert hardswish_in_var is not None
            hardswish_in_var.dest_ops.append(hardswish_op)
            hardswish_op.inputs.append(hardswish_in_var)

            hardswish_out_var = div.outputs[0]
            graph.remove_operation(div)
            hardswish_out_var.source_op = hardswish_op
            hardswish_op.outputs.append(hardswish_out_var)

            # delete merged ops
            for op in middle_ops:
                assert isinstance(op, Operation)
                for var in op.inputs + op.outputs:
                    if var != hardswish_in_var and var != hardswish_out_var:
                        graph.remove_variable(var)
                graph.remove_operation(op)



class ComposeHardSigmoidPass(QuantizationOptimizationPass):
    """
    合成 HardSigmoid

            |--- Add --- Clip --- |
        ----|                     Div ---
            |---------------------|

    """

    def __init__(self) -> None:
        super().__init__('Compose HardSigmoid Operation Optimztion')

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(patterns=[lambda x: True, 'Add', 'Clip', 'Div'],
                                                  edges=[[0, 1], [1, 2], [2, 3]],
                                                  exclusive=True)
        for pattern in patterns:
            start_op, add, clip, div = pattern
            # check operator value
            check_value_1 = convert_any_to_python_primary_type(add.inputs[1].value) == 3
            check_value_2 = convert_any_to_python_primary_type(div.inputs[1].value) == 6
            type_check = add.inputs[1].is_parameter and div.inputs[1].is_parameter
            value_check = check_value_1 and check_value_2
            if not type_check or not value_check:
                continue

            middle_ops = [add, clip]
            hardsigmoid_op = graph.create_operation(op_type='HardSigmoid',
                                              name=f'{start_op.name}_HardSigmoid',
                                              attributes={"alpha": float(1/6), "beta":0.5},
                                              platform=start_op.platform)

            # process input_var and output_var
            hardsigmoid_in_var = None
            for index in range(len(start_op.outputs)):
                if add in start_op.outputs[index].dest_ops:
                    hardsigmoid_in_var = start_op.outputs[index]
                    break
            assert hardsigmoid_in_var is not None
            hardsigmoid_in_var.dest_ops.append(hardsigmoid_op)
            hardsigmoid_op.inputs.append(hardsigmoid_in_var)

            hardsigmoid_out_var = div.outputs[0]
            graph.remove_operation(div)
            hardsigmoid_out_var.source_op = hardsigmoid_op
            hardsigmoid_op.outputs.append(hardsigmoid_out_var)

            # delete merged ops
            for op in middle_ops:
                assert isinstance(op, Operation)
                for var in op.inputs + op.outputs:
                    if var != hardsigmoid_in_var and var != hardsigmoid_out_var:
                        graph.remove_variable(var)
                graph.remove_operation(op)



class ComposeReduceL2Pass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Compose ReduceL2 Operation Optimztion')

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        self.compose_reducel2_v1(graph)
        self.compose_reducel2_v2(graph)


    @staticmethod
    def compose_reducel2_v1(graph: BaseGraph):
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(patterns=['Mul', 'ReduceSum', 'Sqrt'], edges=[[0, 1], [1,2]], exclusive=True)

        for pattern in patterns:
            op_name_list = [op.name for op in pattern]
            # print(op_name_list)
            mul, reducesum, sqrt = pattern
            if mul.inputs[0].name != mul.inputs[1].name: continue
            input_var, output_var = mul.inputs[0], sqrt.outputs[0]

            reducesum.type = 'ReduceL2'
            graph.remove_variable(reducesum.inputs[0])
            graph.remove_variable(reducesum.outputs[0])

            input_var.dest_ops.pop(input_var.dest_ops.index(mul))
            input_var.dest_ops.append(reducesum)
            mul.inputs.pop(mul.inputs.index(input_var))
            reducesum.inputs.insert(0, input_var)
            graph.remove_operation(mul)

            output_var.source_op = reducesum
            sqrt.outputs.pop(sqrt.outputs.index(output_var))
            reducesum.outputs.insert(0, output_var)
            graph.remove_operation(sqrt)

            if reducesum.opset.onnx_opset_version()>=13:
                if 'noop_with_empty_axes' in reducesum.attributes:
                    reducesum.attributes.pop('noop_with_empty_axes')
                if len(reducesum.inputs)>1 and reducesum.inputs[1].is_parameter:
                    reducesum.attributes['axes'] = convert_any_to_numpy(reducesum.inputs[1].value).tolist()
                    graph.remove_variable(reducesum.inputs[1])


    @staticmethod
    def compose_reducel2_v2(graph: BaseGraph):
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(patterns=['Pow', 'ReduceSum', 'Sqrt'], edges=[[0, 1], [1,2]], exclusive=True)

        for pattern in patterns:
            op_name_list = [op.name for op in pattern]
            # print(op_name_list)
            pow, reducesum, sqrt = pattern
            if not pow.inputs[1].is_parameter: continue
            if convert_any_to_python_primary_type(pow.inputs[1].value)!=2: continue

            input_var, output_var = pow.inputs[0], sqrt.outputs[0]

            reducesum.type = 'ReduceL2'
            graph.remove_variable(reducesum.inputs[0])
            graph.remove_variable(reducesum.outputs[0])

            input_var.dest_ops.pop(input_var.dest_ops.index(pow))
            input_var.dest_ops.append(reducesum)
            pow.inputs.pop(pow.inputs.index(input_var))
            reducesum.inputs.insert(0, input_var)
            graph.remove_operation(pow)

            output_var.source_op = reducesum
            sqrt.outputs.pop(sqrt.outputs.index(output_var))
            reducesum.outputs.insert(0, output_var)
            graph.remove_operation(sqrt)

            if reducesum.opset.onnx_opset_version()>=13:
                if 'noop_with_empty_axes' in reducesum.attributes:
                    reducesum.attributes.pop('noop_with_empty_axes')
                if len(reducesum.inputs)>1 and reducesum.inputs[1].is_parameter:
                    reducesum.attributes['axes'] = convert_any_to_numpy(reducesum.inputs[1].value).tolist()
                    graph.remove_variable(reducesum.inputs[1])


    @staticmethod
    def compose_reducel2_common(graph: BaseGraph, node_list: List[str]):
        search_engine = SearchableGraph(graph)
        edge_index = [(i, i + 1 )for i in range(len(node_list)-1)]
        patterns = search_engine.pattern_matching(patterns=node_list, edges=edge_index, exclusive=True)
        for pattern in patterns:
            op_name_list = [op.name for op in pattern]
            if len(pattern)<3: continue
            # print(op_name_list)
            start_op = pattern[0]
            end_op = pattern[-1]
            middle_ops = pattern[1:-2]
            reduce_op =None
            for op in pattern:
                if op.name == 'ReduceSum':
                    reduce_op = op
                    break
            assert reduce_op is not None

            input_var, output_var = start_op.inputs[0], end_op.outputs[0]

            input_var.dest_ops.pop(input_var.dest_ops.index(start_op))
            output_var.source_op = None

            graph.create_operation(
                op_type= 'ReduceL2',
                name=f'fuse_reducel2.{start_op.name}_{end_op.name}',
                attributes=reduce_op.attributes.copy(),
                inputs=[input_var],
                outputs=[output_var]
                )

            start_op.inputs.pop(pow.inputs.index(input_var))
            end_op.outputs.pop(end_op.outputs.index(output_var))

            for op in pattern:
                assert isinstance(op, Operation)
                for var in op.inputs + op.outputs:
                    if var != input_var and var != output_var:
                        graph.remove_variable(var)
                graph.remove_operation(op)
            # TODO ..........


class ComposeConvElementwiseActivation(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Compose ConvElementwise Activation Operation Optimztion')

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        search_engine = SearchableGraph(graph)
        # patterns = search_engine.pattern_matching(patterns=['Conv',
        #                                                     lambda x: x.type in ACTIVATIONS_TYPE,
        #                                                     lambda y: True,
        #                                                     lambda x: x.type in {'Add', 'Sub', 'Mul', 'Div'},
        #                                                     lambda x: x.type in ACTIVATIONS_TYPE
        #                                                     ],
        #                                         edges=[[0, 1], [1,3], [2,3], [3,4]], exclusive=False)
        #                                         # edges=[[0, 1], [1,2], [2,3]], exclusive=False)
        # for pattern in patterns:
        #     op_name_list = [op.name for op in pattern]
        #     print(op_name_list)

        match_nodes_list = []
        for operation in graph.operations.values():
            if operation.type not in {'Add', 'Mul'}: continue
            patterns = search_engine.pattern_matching(patterns=['Conv', 
                                                                lambda x: x.type in PPLMETAX_ACTIVATION, 
                                                                lambda x: x.name == operation.name ],
                                                        edges=[[0, 1], [1,2]], exclusive=False)
            for pattern in patterns:
                # op_name_list = [op.name for op in pattern]
                # print(op_name_list)
                conv, act, elementwise = pattern
                down_ops = graph.get_downstream_operations(elementwise)
                match_nodes = [conv, act, elementwise] 
                if len(graph.get_downstream_operations(act)) != 1:
                    continue

                if len(down_ops) == 1 and  down_ops[0].type in PPLMETAX_ACTIVATION:
                    match_nodes.append(down_ops[0])

                # print([op.name for op in match_nodes])
                match_nodes_list.append(match_nodes)

        # print(match_nodes_list)
        for fuse_nodes in match_nodes_list:
            if len(fuse_nodes) == 3:
                conv, act1, elementwise = fuse_nodes
                self._fuse(graph, conv, act1, elementwise)
            elif len(fuse_nodes) == 4:
                conv, act1, elementwise, act2 = fuse_nodes
                self._fuse(graph, conv, act1, elementwise, act2)
            else:
                raise ValueError("ConvElementwise Activation fuse operation number <= 4" )


    def _fuse(self, graph: BaseGraph, conv: Operation, act1: Operation, elementwise: Operation, act2=None):
        assert len(graph.get_downstream_operations(conv)) == 1
        if len(graph.get_downstream_operations(act1)) != 1:
            return False

        input_var_1 = conv.inputs[0]
        input_var_2 = conv.inputs[1]
        input_var_3 = conv.inputs[2] if len(conv.inputs)>2 else None

        # get other input for elementwise
        input_var_4 = elementwise.inputs[1]
        for var in elementwise.inputs:
            if var.source_op==act1:continue
            input_var_4 = var

        output_var  = act2.outputs[0] if act2 is not None else elementwise.outputs[0]

        # TODO: modify by schema
        attribute_dict = {
            "has_bias": 1 if len(conv.inputs)>2 else 0,
            "ele_type": elementwise.type,  
            }
    
        self.set_act_params(attribute_dict, act1, act2)
        attribute_dict.update(conv.attributes)

        opname = conv.name + "_" + elementwise.name + "_fuse"
        conv_ele_act = graph.create_operation(op_type="ConvActEleActFused", name=opname, attributes=attribute_dict, 
                            #    inputs= [input_var_1, input_var_2], 
                               outputs= [output_var]
                               )
        
        input_var_1.dest_ops[input_var_1.dest_ops.index(conv)] = conv_ele_act
        conv_ele_act.inputs.append(input_var_1)
        input_var_2.dest_ops[input_var_2.dest_ops.index(conv)] = conv_ele_act
        conv_ele_act.inputs.append(input_var_2)
        if input_var_3 is not None:
            input_var_3.dest_ops[input_var_3.dest_ops.index(conv)] = conv_ele_act
            conv_ele_act.inputs.append(input_var_3)
        input_var_4.dest_ops[input_var_4.dest_ops.index(elementwise)] = conv_ele_act
        conv_ele_act.inputs.append(input_var_4)

        conv.inputs.clear()

        rm_ops = [conv, act1, elementwise] if act2 is not None else [conv, act1]
        for op in rm_ops:
            assert isinstance(op, Operation)
            for var in op.inputs + op.outputs:
                if var != input_var_1 and var != input_var_2 and var != input_var_3 and var != input_var_4:
                    graph.remove_variable(var)
            graph.remove_operation(op)

        if act2 is not None:
            act2.inputs.clear()
            act2.outputs.clear()
            graph.remove_operation(act2)
        else:
            elementwise.inputs.clear()
            elementwise.outputs.clear()
            graph.remove_operation(elementwise)


    @staticmethod
    def set_act_params(attributes: Dict, act: Operation, act_: Union[Operation, None]):
        act_type = act.type
        attributes['act1_type'] = act_type

        # TODO: modify initialize to attribute
        if act_type in {'HardSigmoid', 'LeakyRelu'}:
            attributes['act1_param_data'] = act.attributes.values()
        elif act_type == 'PRelu':
            if act.inputs[1].is_parameter:
                attributes['act1_param_data'] = act.inputs[1].value
            else:
                return False
        elif act_type == 'Clip':
            # TODO: min, max is optional
            clip_attr = []
            if len(act.inputs)>1 and  act.inputs[1].is_parameter:
                clip_attr.append(act.inputs[1].value)
            else:
                return False
            if len(act.inputs)>2 and act.inputs[2].is_parameter:
                clip_attr.append(act.inputs[2].value)
            else:
                return False

            attributes['act1_param_data'] = clip_attr

        if act_ is not None: 
            attributes['act2_type'] = act_.type
            if act_type in {'HardSigmoid', 'LeakyRelu'}:
                attributes['act2_param_data'] = act.attributes.values()
            elif act_type == 'PRelu':
                if act.inputs[1].is_parameter:
                    attributes['act2_param_data'] = act.inputs[1].value
                else:
                    return False
            elif act_type == 'Clip':
                # TODO: min, max is optional
                clip_attr = []
                if len(act.inputs)>1 and  act.inputs[1].is_parameter:
                    clip_attr.append(act.inputs[1].value)
                else:
                    return False
                if len(act.inputs)>2 and act.inputs[2].is_parameter:
                    clip_attr.append(act.inputs[2].value)
                else:
                    return False

                attributes['act2_param_data'] = clip_attr

        return True


class ComposeGroupNormPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Compose GroupNormlization Operation Optimztion')

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(patterns=['Reshape', 'InstanceNormalization', 'Reshape'],
                                                    edges=[[0, 1], [1,2]], exclusive=True)

        if len(patterns) == 0:
            return
        try:
            dummy_input = next(iter(dataloader))
        except:
            shape = list(dataloader.dataset[0].shape)
            input_shape = dataloader.batch_size + shape
            input_device = dataloader.dataset[0].device
            input_dtype = dataloader.dataset[0].dtype
            dummy_input = torch.ones(size=input_shape, device=input_device, dtype=input_dtype)
        executor.tracing_operation_meta(inputs=dummy_input)

        for pattern in patterns:
            self.fuseGN_v1(graph, pattern)

        
        self.extra_optimize(graph)
        format_graph(graph)


    def fuseGN_v1(self, graph: BaseGraph, pattern: List[Operation], **kwargs):
        reshape1, instanceNorma, reshape2 = pattern

        epsilon = GET_ATTRIBUTE_FROM_OPERATION(instanceNorma, 'epsilon', default=1e-5)
        reshape1_shape, reshape2_shape = reshape1.inputs[1], reshape2.inputs[1]
        input_var  = reshape1.inputs[0]
        scale = instanceNorma.inputs[1]
        bias = instanceNorma.inputs[2]

        output_var = reshape2.outputs[0]

        num_channel = input_var.shape[1]
        per_channel = scale.shape[0]
        num_groups  = num_channel // per_channel

        if not scale.is_parameter: return
        if not bias.is_parameter: return
        if not reshape1_shape.is_parameter or  reshape1_shape.value[1] != per_channel:  return
        if not reshape2_shape.is_parameter:  return

        attribute_dict={"epsilon": epsilon, "num_groups": num_groups}

        input_var.dest_ops.clear()
        scale.dest_ops.clear()
        bias.dest_ops.clear()
        instanceNorma.inputs.pop(instanceNorma.inputs.index(scale))
        instanceNorma.inputs.pop(instanceNorma.inputs.index(bias))

        scale.value = scale.value.repeat(num_groups)
        bias.value = bias.value.repeat(num_groups)

        opname = "+".join([op.name for op in pattern])
        group_norm = graph.create_operation(op_type="GroupNormalization", name=opname,
                                            attributes=attribute_dict,
                                            inputs= [input_var, scale, bias],
                                            # outputs= [output_var]
                                            )
        output_var.source_op = group_norm
        group_norm.outputs.append(output_var)

        reshape1.inputs.pop(reshape1.inputs.index(input_var))
        reshape2.outputs.pop(reshape2.outputs.index(output_var))
        graph.remove_operation(reshape1)
        graph.remove_operation(instanceNorma)

        for var in reshape2.inputs + reshape2.outputs:
            if var != output_var:
                graph.remove_variable(var)

        graph.remove_operation(reshape2)

        return 


    def extra_optimize(self, graph: BaseGraph):
        # graph_ = graph.copy()
        for operation in graph.operations.values():
            scale_mul, bias_add = None, None
            if operation.type not in {"GroupNormalization"}: continue
            scale_mul, bias_add = self.get_scale_bias(graph, operation)

            scale = operation.inputs[1]
            bias  = operation.inputs[2]
            channels = scale.shape[0]
            assert scale.shape[0] == bias.shape[0]
            
            if scale_mul is not None and bias_add is not None: 
                scale.value = scale.value * scale_mul.value.squeeze()
                bias.value = bias.value * scale_mul.value.squeeze() + bias_add.value.squeeze()
                down_ops_1 = graph.get_downstream_operations(operation)[0]
                down_ops_2 = graph.get_downstream_operations(down_ops_1)[0]

                input_var  = down_ops_1.inputs[0]
                output_var = down_ops_2.outputs[0]

                input_var.dest_ops.clear()
                output_var.source_op = None
                
                down_ops_1.inputs.pop(down_ops_1.inputs.index(input_var))
                down_ops_2.outputs.pop(down_ops_2.outputs.index(output_var))

                graph.create_link_with_var(input_var, output_var)

                if down_ops_1 in graph.operations:
                    graph.remove_operation(down_ops_1)
                if down_ops_2 in graph.operations:
                    graph.remove_operation(down_ops_2)


            # TODO
            if scale_mul is not None and scale_mul.shape[0] != channels: continue
            if bias_add  is not None and bias_add.shape[0] != channels: continue



    def get_scale_bias(self, graph: BaseGraph , operation: Operation):
        last_op = operation
        down_ops_1 = graph.get_downstream_operations(last_op)
        scale_mul = None
        bias_add = None
        if len(down_ops_1) != 1:
            return scale_mul, bias_add

        if down_ops_1[0].type in {'Mul'}:
            if down_ops_1[0].inputs[1].is_parameter:
                scale_mul = down_ops_1[0].inputs[1]

        elif down_ops_1[0].type in {'Add'}:
            if down_ops_1[0].inputs[1].is_parameter:
                bias_add = down_ops_1[0].inputs[1]

        down_ops_2 = graph.get_downstream_operations(down_ops_1[0])
        if len(down_ops_2) != 1:
            return scale_mul, bias_add

        if down_ops_2[0].type in {'Add'} and bias_add is None:
            if down_ops_2[0].inputs[1].is_parameter:
                bias_add = down_ops_2[0].inputs[1]

        return scale_mul, bias_add


class ComposeGeluPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Compose Gelu Operation Optimztion')
        
    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        self.fuse_gelu_v1(graph)


    @staticmethod
    def fuse_gelu_v1(graph: BaseGraph):
            """ Fuse Gelu
            
            Pattern: * - Div - Erf - Add - Mul - Mul
                    |                 |
                    -------------------
            """
            fused         = False
            search_engine = SearchableGraph(graph=graph)
            
            matches = search_engine.pattern_matching(
                patterns=[lambda x: True, 'Div', 'Erf', 'Add', 'Mul', 'Mul'], 
                edges=[[0, 1], [1, 2], [2, 3], [3, 4], [0, 4], [4, 5]], exclusive=True)
            
            for _, div, erf, add, mul1, mul2 in matches:
                fuse_opname = f"mx.gelu-{div.name}_{mul2.name}"
                removing_var = []
                removing_var.extend(div.outputs)
                removing_var.extend(erf.outputs)
                removing_var.extend(add.outputs)
                removing_var.extend(mul1.outputs)

                graph.remove_operation(div)
                graph.remove_operation(erf)
                graph.remove_operation(add)
                graph.remove_operation(mul1)
                for var in removing_var:
                    graph.remove_variable(var)

                input_vars  = _.outputs.copy()
                output_vars = mul2.outputs.copy()

                graph.remove_operation(mul2)


                graph.create_operation(op_type='Gelu', name=fuse_opname , inputs=input_vars, outputs=output_vars)
                assert len(input_vars) == 1, 'Fusion failed, Pattern unrecognized.'
                fused = True

            # # final check, if no valid pattern was found, we give a warning.
            # if not fused:
            #     print('No valid Gelu pattern was found, check your graph again.')



class ComposeSDPAttentionPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Compose Scale Dot-Product Attention Operation Optimztion')

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        self.fuse_SDPA_v1(graph)
        self.fuse_SDPA_v2(graph)
    

    def fuse_SDPA_v1(self, graph: BaseGraph, **kwargs):
        search_engine = SearchableGraph(graph)
        # patterns = search_engine.pattern_matching(patterns=['Transpose', 'Transpose', 'Transpose', 'MatMul', 'Div', 'Add', 'Softmax', 'MatMul', 'Transpose'],
        #                                             edges=[[0, 3], [1,3], [2,7], [3,4], [4,5], [5,6], [6,7], [7,8]], exclusive=False)
        
        # Pattern: Transpose ----|  
        #                      MatMul  - Div/Mul - Add - Softmax - MatMul - Transpose 
        #          Transpose ----|  

        patterns_1 = search_engine.pattern_matching(patterns=['MatMul', 'Div', 'Add', 'Softmax', 'MatMul', 'Transpose'],
                                            edges=[[0,1], [1,2], [2,3], [3,4], [4,5]], exclusive=False)
        
        patterns_2 = search_engine.pattern_matching(patterns=['MatMul', 'Mul', 'Add', 'Softmax', 'MatMul', 'Transpose'],
                                            edges=[[0,1], [1,2], [2,3], [3,4], [4,5]], exclusive=False)

        matched_patterns = []
        for pattern in patterns_2 + patterns_1:
            matmul_1 = pattern[0]
            matmul_2 = pattern[4]
            
            up_ops = graph.get_upstream_operations(matmul_1)
            if len(up_ops) != 2: continue
            if not all([op.type =='Transpose' for op in up_ops]): continue
            transpose_1, transpose_2 = up_ops

            up_ops = graph.get_upstream_operations(matmul_2)
            if pattern[3] not in up_ops: continue
            up_ops.pop(up_ops.index(pattern[3]))
            if len(up_ops) !=1 or up_ops[0].type !='Transpose': continue
            transpose_3 = up_ops[0]
            

            attr_1 = GET_ATTRIBUTE_FROM_OPERATION(op=transpose_1, attribute='perm') 
            attr_2 = GET_ATTRIBUTE_FROM_OPERATION(op=transpose_2, attribute='perm') 
            attr_3 = GET_ATTRIBUTE_FROM_OPERATION(op=transpose_3, attribute='perm') 

            per_ops = []
            if attr_1 == attr_3:
                per_ops = [transpose_1, transpose_2, transpose_3]
            elif attr_2 == attr_3:
                per_ops = [transpose_2, transpose_1, transpose_3]
            else:
                maca_warning("Transpose perm attribute not match: {attr_1}, {attr_2}, {attr_3}")
                continue

            matched_patterns.append(per_ops + pattern)

        # maca_info(matched_patterns)

        for pattern in matched_patterns:
            if not int(os.environ.get("MXQ_USE_MACART_BACKEND", 1)):
                for op in pattern:
                    op.set_extension_attrib(DISPATCH_FP32_ATTRIB_KEY, 1)

            else:
                q = pattern[0].inputs[0]
                k = pattern[1].inputs[0]
                v = pattern[2].inputs[0]
                output = pattern[-1].outputs[0]


                attr_dict = {
                    "head_dim": -1,
                    "is_causal": 0,
                    "num_heads": -1,
                    "num_kv_heads": -1
                }
                if q.shape is not None and len(q.shape)>=2:
                    attr_dict["head_dim"] = convert_any_to_python_primary_type(q.shape[0])
                    attr_dict["num_heads"] = convert_any_to_python_primary_type(q.shape[1])
                
                if k.shape is not None and len(k.shape) >= 2:
                    num_kv_heads_ = k.shape[-2]

                q.dest_ops.pop(q.dest_ops.index(pattern[0]))
                k.dest_ops.pop(k.dest_ops.index(pattern[1]))
                v.dest_ops.pop(v.dest_ops.index(pattern[2]))
                pattern[0].inputs.clear()
                pattern[1].inputs.clear()
                pattern[2].inputs.clear()

                mha_input = [q, k, v]

                # Add atten_mask
                for inp in pattern[5].inputs:
                    if inp.source_op is not None and inp.source_op in pattern: 
                        continue
                    if inp.source_op is None and inp.is_parameter: 
                        mha_input.append(inp)
                        inp.dest_ops.pop(pattern[5])
                        pattern[5].input.pop(inp)

                    elif not inp.is_parameter and inp.source_op not in pattern:
                        mha_input.append(inp)
                        inp.dest_ops.pop(inp.dest_ops.index(pattern[5]))
                        pattern[5].inputs.pop(pattern[5].inputs.index(inp))

                assert len(mha_input) <= 4

                op_name = f"fusedMHA_{pattern[-1].name}"
                mha_op = graph.create_operation(op_type="MultiHeadAttentionV1", 
                                                name = op_name,
                                                attributes=attr_dict,
                                                inputs= mha_input,
                                                outputs= None)

                output.source_op = mha_op
                mha_op.outputs.append(output)
                pattern[-1].outputs.clear()

                for op in pattern:
                    assert isinstance(op, Operation)
                    for var in op.inputs + op.outputs:
                        if var not in mha_input:
                            graph.remove_variable(var)
                    graph.remove_operation(op)


    def fuse_SDPA_v2(self, graph: BaseGraph, **kwargs):
        """
        Pattern: 
                Transpose -- Mul/Div --|  
                                    MatMul - Add - Softmax - MatMul - Transpose 
                            Transpose --|  
        """
        search_engine = SearchableGraph(graph)

        patterns_1 = search_engine.pattern_matching(patterns=['Div', 'MatMul', 'Softmax', 'MatMul', 'Transpose'],
                                            edges=[[0,1], [1,2], [2,3], [3,4]], exclusive=False)
        
        patterns_2 = search_engine.pattern_matching(patterns=['Mul', 'MatMul', 'Softmax', 'MatMul', 'Transpose'],
                                            edges=[[0,1], [1,2], [2,3], [3,4]], exclusive=False)

        matched_patterns = []
        for pattern in patterns_2 + patterns_1:
            ele_1 = pattern[0]
            matmul_1 = pattern[1]
            matmul_2 = pattern[3]

            per_ops = []
            up_ops = graph.get_upstream_operations(ele_1)
            if len(up_ops) !=1 or up_ops[0].type !='Transpose': continue
            transpose_1 = up_ops[0]

            up_ops = graph.get_upstream_operations(matmul_1)
            if pattern[0] not in up_ops: continue
            up_ops.pop(up_ops.index(pattern[0]))
            if len(up_ops) !=1 or up_ops[0].type !='Transpose': continue
            transpose_2 = up_ops[0]


            up_ops = graph.get_upstream_operations(matmul_2)
            if pattern[2] not in up_ops: continue
            up_ops.pop(up_ops.index(pattern[2]))
            if len(up_ops) !=1 or up_ops[0].type !='Transpose': continue
            transpose_3 = up_ops[0]

            attr_1 = GET_ATTRIBUTE_FROM_OPERATION(op=transpose_1, attribute='perm') 
            attr_2 = GET_ATTRIBUTE_FROM_OPERATION(op=transpose_2, attribute='perm') 
            attr_3 = GET_ATTRIBUTE_FROM_OPERATION(op=transpose_3, attribute='perm') 


            if attr_1 == attr_3:
                per_ops = [transpose_1, transpose_2, transpose_3]
            elif attr_2 == attr_3:
                per_ops = [transpose_2, transpose_1, transpose_3]
            else:
                maca_warning("Transpose perm attribute not match: {attr_1}, {attr_2}, {attr_3}")
                continue

            matched_patterns.append(per_ops + pattern)

        # maca_info(matched_patterns)

        for pattern in matched_patterns:
            if not int(os.environ.get("MXQ_USE_MACART_BACKEND", 1)):
                for op in pattern:
                    op.set_extension_attrib(DISPATCH_FP32_ATTRIB_KEY, 1)

            else:
                q = pattern[0].inputs[0]
                k = pattern[1].inputs[0]
                v = pattern[2].inputs[0]
                output = pattern[-1].outputs[0]


                attr_dict = {
                    "head_dim": -1,
                    "is_causal": 0,
                    "num_heads": -1,
                    "num_kv_heads": -1
                }
                if q.shape is not None and len(q.shape)>=2:
                    attr_dict["head_dim"] = convert_any_to_python_primary_type(q.shape[0])
                    attr_dict["num_heads"] = convert_any_to_python_primary_type(q.shape[1])
                
                if k.shape is not None and len(k.shape) >= 2:
                    num_kv_heads_ = k.shape[-2]

                q.dest_ops.pop(q.dest_ops.index(pattern[0]))
                k.dest_ops.pop(k.dest_ops.index(pattern[1]))
                v.dest_ops.pop(v.dest_ops.index(pattern[2]))
                pattern[0].inputs.clear()
                pattern[1].inputs.clear()
                pattern[2].inputs.clear()

                mha_input = [q, k, v]

                # Without atten_mask
                assert len(mha_input) <= 4

                op_name = f"fusedMHA_{pattern[-1].name}"
                mha_op = graph.create_operation(op_type="MultiHeadAttentionV1", 
                                                name = op_name,
                                                attributes=attr_dict,
                                                inputs= mha_input,
                                                outputs= None)

                output.source_op = mha_op
                mha_op.outputs.append(output)
                pattern[-1].outputs.clear()

                for op in pattern:
                    assert isinstance(op, Operation)
                    for var in op.inputs + op.outputs:
                        if var not in mha_input:
                            graph.remove_variable(var)
                    graph.remove_operation(op)

