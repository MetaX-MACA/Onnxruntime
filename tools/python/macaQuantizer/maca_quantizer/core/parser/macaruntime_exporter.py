# coding=utf-8
import torch
import onnx
import numpy as np
from onnx import helper, numpy_helper

from ppq import *
from ppq.api import *
from ppq.IR  import Opset
from ppq.parser import ONNXRUNTIMExporter
from ppq.parser.onnxruntime_exporter import OperationExporter,  OP_CONVERTERS
from ppq.core.common import PASSIVE_OPERATIONS
from ppq.executor.op.torch.base import GET_ATTRIBUTE_FROM_OPERATION
from maca_quantizer.utils.utils import maca_info, maca_warning
from .mxq_parser import MACAOnnxParser
from maca_quantizer.version import MXQ_CONFIG


class QDQHelper():
    """Helper class for processing onnx qdq format"""
    @ staticmethod
    def TQC_Exportable_Check(
        TQC: TensorQuantizationConfig, operation: Operation, bounded_var: Variable) -> bool:
        # if not TQC.can_export() and operation.type not in PASSIVE_OPERATIONS: return False
        if not TQC.can_export(): return False

        if TQC.visibility == QuantizationVisibility.INTERNAL: return False
        if TQC.num_of_bits == 8 and TQC.policy.has_property(QuantizationProperty.LINEAR):
            if TQC.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                range_check = (TQC.quant_max <= 255 and TQC.quant_min >= 0)
            else: range_check = (TQC.quant_max <= 127 and TQC.quant_min >= -128)
        else: range_check = True

        if not range_check:
            ppq_warning(f'Is it not safe to export TQC({bounded_var.name}) to Onnx, '
                        f'INT8 value range must be [-128, 127] or [0, 255], '
                        f'however [{TQC.quant_min, TQC.quant_max}] was given.')
            return False
        return True

class BaseOperationExporter(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        """
        if operation domain not in graph opset, add to graph opset 
        Only check domain , not check version
        """
        if GRAPH_OPSET_ATTRIB not in graph._detail:
            maca_warning(f"graph \' {GRAPH_OPSET_ATTRIB}\' arttribute not exist !")
            return op
        if op.opset.domain in MACARUNTIMExporter().required_opsets or op.opset.domain == '':
            return op

        flag = True
        for opset in graph._detail[GRAPH_OPSET_ATTRIB]:
            if opset['domain'] == op.opset.domain:
                flag = False
                break
        if flag:
            graph._detail[GRAPH_OPSET_ATTRIB].append({'domain':op.opset.domain, 'version':op.opset.version})      
        return op



class ConstantExporter(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # PATCH for Constant operation, Constant value Op causes an export error.
        # operation.attributes['value'] = numpy_helper.from_array(operation.attributes['value'])
        op.attributes['value'] = MACARUNTIMExporter.value_to_proto(value=op.attributes['value'],  
                                                                   name=f'{op.name}_const_tensor')
        super().export(op, graph)
        return op


class MMCVExporter(BaseOperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # MMCV operation must have a mmcv domain.
        op.opset = Opset(domain='mmcv', version=1)
        super().export(op, graph)
        return op


class OOSExporter(BaseOperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # onnxruntime operation must have 'com.microsoft' domain.
        op.opset = Opset(domain='com.microsoft', version=1)
        super().export(op, graph)
        return op


class MXExporter(BaseOperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # onnxruntime operation must have 'com.metax-tech' domain.
        op.opset = Opset(domain='com.metax-tech', version=1)
        super().export(op, graph)
        return op
    

class HardSwishExporter(BaseOperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # onnxruntime operation must have 'com.metax-tech' domain.
        if MACARUNTIMExporter().required_opsets['ai.onnx'] < 14:
            op.opset = Opset(domain='com.metax-tech', version=1)
        super().export(op, graph)
        return op
    
class LayerNormalizationExporter(BaseOperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # onnxruntime operation must have 'com.metax-tech' domain.
        if MACARUNTIMExporter().required_opsets['ai.onnx'] < 17:
            op.opset = Opset(domain='com.microsoft', version=1)
        super().export(op, graph)
        return op

class GroupNormalizationExporter(BaseOperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # onnxruntime operation must have 'com.metax-tech' domain.
        if MACARUNTIMExporter().required_opsets['ai.onnx'] < 18:
            op.opset = Opset(domain='com.metax-tech', version=1)
        super().export(op, graph)
        return op


class LSTMExporter(BaseOperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # onnxruntime operation must have 'com.metax-tech' domain.
        if op.inputs[4].value is None:
            op.inputs[4].value = convert_any_to_torch_tensor([op.inputs[0].shape[0]], dtype=torch.int32)
            op.inputs[4].shape = op.inputs[4].value.shape
            op.inputs[4].dtype = DataType.INT32
        super().export(op, graph)
        return op


MACA_OP_CONVERTERS = {
    'Constant'    : ConstantExporter,
    'MMCVRoiAlign': MMCVExporter,
    'grid_sampler': MMCVExporter,
    'Attention'   : OOSExporter,
    'Gelu'        : OOSExporter,
    'Swish'       : MXExporter,
    'Mish'        : MXExporter,
    'LayerNorm'   : MXExporter,
    'HardSwish'   : HardSwishExporter,
    'GroupNormalization': GroupNormalizationExporter,
    # 'LayerNormalization': LayerNormalizationExporter,
    'LSTM'        : LSTMExporter,

    'ConvActEleActFused':   MXExporter,
    'MultiHeadAttentionV1': MXExporter
    
}

OP_CONVERTERS.update(MACA_OP_CONVERTERS)



class MACARUNTIMExporter(ONNXRUNTIMExporter):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def value_to_proto(value: Union[torch.Tensor, np.ndarray, int, float, list, tuple], 
                       name: str ) -> onnx.OperatorProto:
        value = convert_any_to_numpy(value)
        dtype = DataType.convert_from_numpy(value.dtype)
        shape = value.shape
        tensor_proto = helper.make_tensor(name=name, data_type=dtype.value, 
                                          dims=shape, vals=value.flatten())
        return tensor_proto
    

    def remove_duplicated_cast_op(self, graph: BaseGraph) -> BaseGraph:
        interested_pairs = []
        for cast_op in graph.operations.values():
            if cast_op.type != 'Cast': continue
            # if len(graph.get_downstream_operations(cast_op)) != 1: continue
            # if graph.get_downstream_operations(cast_op)[0].type != 'Cast': continue

            up_ops = graph.get_upstream_operations(cast_op)
            if len(up_ops) != 1 or up_ops[0].platform not in {TargetPlatform.FP16}:
                continue 
            down_ops = graph.get_downstream_operations(cast_op)
            if len(down_ops) == 0 : continue

            # TODO: check arrribute 
            check_type = [op.type=='Cast' for op in down_ops]
            check_attr1 = GET_ATTRIBUTE_FROM_OPERATION(op=cast_op, attribute='to') == DataType.FP32
            check_attr2 = [GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='to')==DataType.FP16 for op in down_ops ]
            if all(check_type) and all(check_attr2) and check_attr1:
                interested_pairs.append((cast_op, down_ops))

        for op1, op2_list in interested_pairs:
            # TODO: code optimize
            if len(op2_list)==1:
                op2 = op2_list[0]
                up_op = graph.get_upstream_operations(op1)[0]
                down_ops = graph.get_downstream_operations(op2)
                if len(down_ops)==1:
                    down_op = down_ops[0]
                    assert isinstance(op1, Operation) and isinstance(op2, Operation)
                    input_var, output_var = op1.inputs[0], op2.outputs[0]
                    link_var = op1.outputs[0]
                    up_op.outputs[up_op.outputs.index(input_var)] = link_var
                    link_var.source_op = up_op
                    link_var.dest_ops[link_var.dest_ops.index(op2)] = down_op
                    down_op.inputs[down_op.inputs.index(output_var)] = link_var

                    graph.remove_operation(op1)
                    graph.remove_operation(op2)

                    link_var.dtype = DataType.FP16
                elif len(down_ops) > 1:
                    link_var = op1.outputs[0]
                    link_var.dest_ops.clear()
                    input_var, output_var = op1.inputs[0], op2.outputs[0]
                    up_op.outputs[up_op.outputs.index(input_var)] = link_var
                    link_var.source_op = up_op
                    for _, down_op in enumerate(down_ops): 
                        link_var.dest_ops.append(down_op)
                        down_op.inputs[down_op.inputs.index(output_var)] = link_var

                        link_var.dtype = DataType.FP16
                    graph.remove_operation(op1)
                    graph.remove_operation(op2)
 


            else:
                """
                                      |--- cast_op2 --- down_op1
                 up_op ---cast_op1 ---|
                                      |--- cast_op3 --- down_op2


                """
                up_op = graph.get_upstream_operations(op1)[0]
                input_var = op1.inputs[0]

                for op2 in op2_list:
                    output_var = op2.outputs[0]
                    output_var.source_op = None
                    graph.create_link_with_var(input_var, output_var)
                    graph.remove_operation(op2)
                graph.remove_operation(op1)



    # not use
    def remove_passive_quant_and_dequant(self, graph: BaseGraph) -> BaseGraph:
        """
        Remove passive_op quant & dequant
        Mixed qunatization is currently not considered !!!
        such as Reshape:
        before:
        Q-->DQ--> Reshape -->Q-->DQ
        after: 
        DQ--> Reshape -->Q

        """
        interested_pairs = []
        for op in graph.operations.values():
            if op.type in PASSIVE_OPERATIONS:
                up_ops   = graph.get_upstream_operations(op)
                down_ops = graph.get_downstream_operations(op)
                if len(up_ops)!=1 or len(down_ops)!=1: continue
                if up_ops[0].type not in {'DequantizeLinear'}: continue
                if down_ops[0].type not in {'QuantizeLinear'}: continue
                interested_pairs.append((up_ops[0], down_ops[0]))

        mark_to_remove=set()
        for dq_op, qt_op in interested_pairs:
            scale_1, offset_1 = dq_op.inputs[1].value, dq_op.inputs[2].value
            scale_2, offset_2 = qt_op.inputs[1].value, qt_op.inputs[2].value
            
            scale_diff     = torch.max(torch.abs(scale_1 - scale_2)).item()
            zeropoint_diff = torch.max(torch.abs(offset_1 - offset_2)).item()
            if scale_diff > 1e-5 and zeropoint_diff > 0.5:
                maca_warning(f"{dq_op.name} and {qt_op.name} quantization parameters are not same !")
                continue 
            mark_to_remove.add(dq_op)
            mark_to_remove.add(qt_op)
        
        for op in mark_to_remove:
            assert isinstance(op, Operation)
            input_var, output_var = op.inputs[0], op.outputs[0]
            graph.remove_operation(op)
            # graph.create_link_with_var(input_var, output_var)

            # DequantizeLinear link with input_var, QuantizeLinear link with output_var
            if op.type in {'DequantizeLinear','DequantizeFloating'}:
                dest_ops = output_var.dest_ops
                for dest_op in dest_ops:
                    dest_op.inputs[dest_op.inputs.index(output_var)] = input_var
                    input_var.dest_ops.append(dest_op)
                output_var.dest_ops.clear()
                graph.remove_variable(output_var)
            else: 
                start_op = input_var.source_op
                start_op.outputs[start_op.outputs.index(input_var)] = output_var
                output_var.source_op = start_op
                graph.remove_variable(input_var)

    @ property
    def required_opsets(self) -> Dict[str, int]:
        extra_domain_versions = [('ai.onnx', 13)]
        return dict(extra_domain_versions)

        
    def convert_operation(self, graph: BaseGraph, op: QuantableOperation,
                          quantized_param: bool):
        """Convert an operation to onnx quant & dequant format by inserting
        necessary quant & dequant op around it. There are 2 ways to represent
        quantized ONNX models:

        Operator Oriented. All the quantized operators have their own ONNX definitions,
            like QLinearConv, MatMulInteger and etc.

        Tensor Oriented, aka Quantize and DeQuantize (QDQ).
            This format uses DQ(Q(tensor)) to simulate the quantize and dequantize process,
            and QuantizeLinear and DeQuantizeLinear operators also carry the quantization parameters.

        Quantization-Aware training (QAT) models converted from Tensorflow or exported from PyTorch.

        Quantized models converted from tflite and other framework.

        Args:
            graph (BaseGraph): PPQ IR
            op (Operation): Converting op
            process_activation (bool): Converting op's activation
            process_parameter (bool): Converting op's parameter
            quantized_param (bool): Export parameter in quantized format.
        """
        # collect quantable vars, where we need to insert quant and dequant op
        for config, var in [_ for _ in op.config_with_variable]:
            inserting, inserting_var = op, var
            if not QDQHelper.TQC_Exportable_Check(TQC=config, operation=op, bounded_var=var): continue
            if config.state in {QuantizationStates.FP32}: continue

            if var.is_parameter:
                assert len(var.dest_ops) == 1, (
                f'Can not export variable {var.name}, cause it has more than 1 destination operations. '
                'PPQ require all parameters to have only 1 destination operation.')

                # override quantization state, so that we can export parameter correctly.
                if config.state == QuantizationStates.BAKED:
                    config.state = QuantizationStates.ACTIVATED
                if config.state == QuantizationStates.PASSIVE_BAKED:
                    config.state = QuantizationStates.PASSIVE

                # if not quant parameter to int, all parameter should export as fp32.
                # needs insert both quant and dequant op for them
                if not quantized_param:
                    created = self.insert_quantize_node(
                        graph=graph, var=inserting_var, config=config, op=inserting)
                    inserting_var = created.outputs[0]
                    inserting     = created

                self.insert_dequantize_node(
                    graph=graph, var=inserting_var, config=config, op=inserting)

                if quantized_param and config.policy.has_property(QuantizationProperty.LINEAR):
                    var.value = PPQLinearQuant_toInt(tensor=var.value, config=config)

            elif (not var.is_parameter):
                
                # Patch 20230103:
                # If var.source_op is DequantizeLinear, then we do not need to quantize it twice.
                if var.source_op is not None and var.source_op.type in {'DequantizeLinear', 'DequantizeFloating'}:
                    assert var.source_op.num_of_input == 3, 'Quantize Node Format Error, need as least 3 inputs.'
                    assert isinstance(var.source_op, Operation)
                    scale, offset = var.source_op.inputs[1].value, var.source_op.inputs[2].value
                    
                    scale_diff     = torch.max(torch.abs(scale - config.scale)).item()
                    zeropoint_diff = torch.max(torch.abs(offset - config.offset)).item()
                    if scale_diff < 1e-4 and zeropoint_diff < 1e-1:
                        continue

                if len(var.dest_ops) == 1 and var.dest_ops[0].type in {'QuantizeLinear', 'QuantizeFloating'}:
                    assert var.dest_ops[0].num_of_input == 3, 'Quantize Node Format Error, need as least 3 inputs.'
                    assert isinstance(var.dest_ops[0], Operation)
                    scale, offset = var.dest_ops[0].inputs[1].value, var.dest_ops[0].inputs[2].value
                    
                    scale_diff     = torch.max(torch.abs(scale - config.scale)).item()
                    zeropoint_diff = torch.max(torch.abs(offset - config.offset)).item()
                    if scale_diff < 1e-4 and zeropoint_diff < 1e-1:
                        continue

                created = self.insert_quantize_node(
                    graph=graph, var=inserting_var, config=config, op=inserting)
                inserting_var = created.outputs[0]
                inserting     = created

                self.insert_dequantize_node(
                    graph=graph, var=inserting_var, 
                    config=config, op=inserting)
                

    def build_variable_proto(self, variable: Variable, value_shape=True) -> onnx.TensorProto:
        """
        Convert PPQ Variable to Onnx TensorProto, There are 2 different types of Tensor in Onnx:
            Variable: Represents a Tensor whose value is not known until inference-time.
            Constant: Represents a Tensor whose value is known.
        """
        # Parameter Varaible in PPQ, Constant Variable in Onnx
        if variable.is_parameter:
            if variable.value is not None:
                var_shape     = variable.value.shape
                pytorch_dtype = variable.value.dtype
                onnx_dtype    = DataType.convert_from_torch(pytorch_dtype).value
            else:
                tensor_dtype   = DataType.to_torch(variable.dtype) if variable.dtype is not None else torch.float32
                variable.value = torch.tensor([], dtype=tensor_dtype)
                var_shape      = variable.value.shape
                # pytorch_dtype  = variable.value.dtype
                onnx_dtype     = DataType.convert_from_torch(tensor_dtype).value
 
        # Non Parameter
        else:
            var_shape  = variable.shape
            onnx_dtype = variable.dtype.value

        if not variable.is_parameter:
            if not value_shape:
                var_shape = None
            tensor_proto = helper.make_tensor_value_info(
                name=variable.name, elem_type=onnx_dtype, shape=var_shape)
        else:
            value = variable.value
            is_raw_format = False
            if isinstance(value, torch.Tensor):
                if value.numel() == 0: value = []
                elif value.ndim >= 1:
                    value = convert_any_to_numpy(variable.value).flatten()
                    value = value.tobytes()
                    is_raw_format = True
                elif value.ndim == 0: # Pytorch Scalar Type
                    value = [value.item(), ] # it is fine for onnx, shape for this value will be []
            else: value = value # value is python primary type.
            tensor_proto = helper.make_tensor(
                name=variable.name, data_type=onnx_dtype,
                dims=var_shape, vals=value, raw=is_raw_format)
        return tensor_proto

                
    def build_operator_proto(self, operation: Operation) -> onnx.OperatorProto:
        """
        Convert PPQ Op to Onnx Operation
        An Op consumes zero or more Tensors, and produces zero or more Tensors.
        """
        attributes = operation.attributes
        for key in attributes:
            value = attributes[key]
            if isinstance(value, DataType):
                attributes[key] = value.value
            if isinstance(value, torch.Tensor):
                if value.numel() == 0: attributes[key] = None
                elif value.numel() == 1: attributes[key] = convert_any_to_numpy([value.item()]) # convert to 1d array
                else: attributes[key] = convert_any_to_numpy(value)

        if PPQ_CONFIG.EXPORT_PPQ_INTERNAL_INFO:
            attributes['platform'] = operation.platform.name

        op_domain = None
        if operation._opset.domain != DEFAULT_OPSET_DOMAIN:
            op_domain = operation._opset.domain
            # attributes.update({'domain': op_domain})

        op_proto = helper.make_node(
            op_type=operation.type,
            inputs=[_.name for _ in operation.inputs],
            outputs=[_.name for _ in operation.outputs],
            name=operation.name,
            domain=op_domain,
            **attributes
            )

        return op_proto


    def prepare_graph(
        self, graph: BaseGraph,
        remove_activation_fn: bool = True,
        quant_parameter_to_int: bool = True) -> BaseGraph:
        """Prepare your graph for exporting.

        There are many works to do with your graph:

            1. Insert Quant and Dequant operation within your graph.

            2. Remove all unnecessary activations.

            3. Quantize all parameters of your graph, convert them to int8.

        Args:
            graph (BaseGraph): Processing Graph

        Returns:
            BaseGraph: Processed Graph
        """
        self.convert_operation_from_opset11_to_opset13(graph)

        # mark quantable variables
        for op in graph.topological_sort():
            if not isinstance(op, QuantableOperation): continue
            if op.platform in {TargetPlatform.FP16, TargetPlatform.FP32}: continue
            if op.type in {'QuantizeLinear', 
                           'DequantizeLinear', 
                           'QuantizeFloating', 
                           'DequantizeFloating'}: continue

            self.convert_operation(graph=graph, op=op, quantized_param=quant_parameter_to_int)

        # remove activations
        if remove_activation_fn:
            # remove useless activation.
            self.remove_activation_ops(graph)

        return self.remove_duplicated_quant_op(graph)


    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, 
               quantized_param: bool = True, remove_activation: bool = True, 
               save_as_external_data: bool = False, **kwargs) -> None:
        """
        Export PPQ Graph to Onnx QDQ format.
            This function requires a set of parameters to configure onnx format.
        
        Args:
            file_path (str): Onnx file name.
            
            graph (BaseGraph): Exporting ppq graph.
            
            config_path (str, optional): config file is a json file that contains quant-related
                information, this file is require by TensorRT for initialize its quantization
                pipeline. If config_path = None, no json file will be created.

            export_QDQ_op (bool, optional): whether to export QDQ node in onnx model.

            quantized_param (bool, optional): export quantized parameter, if quantized_param = False,
                PPQ will export parameter in FP32 format.
            
            remove_activation (bool, optional): this option will remove activation op(Relu, Clip),
                requires ASYMMTRICAL quantizaiton.
            
            save_as_external_data (bool, optional): for model larger than 2GB, 
                this option will split model into external param files.
        """
        # Add for maca-c500 fp16 
        self.remove_duplicated_cast_op(graph=graph)

        # In prepare stage, quant & dequant node are inserted into graph.
        graph = self.prepare_graph(
            graph, remove_activation_fn=remove_activation, 
            quant_parameter_to_int=quantized_param)

        # if a valid config path is given, export quantization config to there.
        if config_path is not None:
            super().export_quantization_config(config_path, graph)

        # before we can export them, we firstly convert all ops to proper format.
        for op in [_ for _ in graph.topological_sort()]:
            if op.type in OP_CONVERTERS:
                exporter = OP_CONVERTERS[op.type]()
                assert isinstance(exporter, OperationExporter), (
                    f'Expected an OpExporter here, however {type(exporter)} was given.')
                op = exporter.export(op=op, graph=graph)
        
        # Add for maca-c500 cuda ep
        # self.remove_passive_quant_and_dequant(graph=graph)

        # Add for maca-c500 fp16 
        # self.remove_duplicated_cast_op(graph=graph)

        name = graph._name
        if not name: name = 'MACA Quantization Tool - Onnx Export'

        # Ready to export onnx graph definition.
        _inputs, _outputs, _initilizers, _nodes, _value_info = [], [], [], [], []
        for operation in graph.topological_sort():
            _nodes.append(self.build_operator_proto(operation))

        for variable in graph.variables.values():
            if kwargs.get('dynamic_shape', False) and variable.name not in graph.inputs and variable.name  not in graph.outputs:
                tensor_proto = self.build_variable_proto(variable, value_shape=False)
            else:
                tensor_proto = self.build_variable_proto(variable)

            if variable.name in graph.inputs: _inputs.append(tensor_proto)
            if variable.name in graph.outputs: _outputs.append(tensor_proto)
            if variable.is_parameter: _initilizers.append(tensor_proto)
            else: _value_info.append(tensor_proto)
        
        if "model_file" in kwargs:
            # graph_ = onnx.load_model(kwargs["model_file"]).graph
            graph_ = MACAOnnxParser().build(kwargs["model_file"])
            _inputs  = self._resorted_variable(quantized_graph=graph, org_graph=graph_, variables=_inputs, is_output=False)
            _outputs = self._resorted_variable(quantized_graph=graph, org_graph=graph_, variables=_outputs, is_output=True)

        graph_def = helper.make_graph(
            name=name, nodes=_nodes, inputs=_inputs,
            outputs=_outputs, initializer=_initilizers, 
            value_info=_value_info)
        extra_opsets = self.required_opsets

        opsets = []
        if GRAPH_OPSET_ATTRIB in graph._detail:
            for opset in graph._detail[GRAPH_OPSET_ATTRIB]:
                if opset['domain'] in extra_opsets or opset['domain'] == '':
                    continue
                op = onnx.OperatorSetIdProto()
                op.domain = opset['domain']
                op.version = opset['version']
                opsets.append(op)

        for key, value in extra_opsets.items():
            op = onnx.OperatorSetIdProto()
            op.domain = key
            op.version = value
            opsets.append(op)

        producer_info_str = MXQ_CONFIG.NAME + " " + MXQ_CONFIG.VERSION
        onnx_model = helper.make_model(
            graph_def, producer_name=producer_info_str, 
            opset_imports=opsets)
        onnx_model.ir_version = 7
        # onnx.checker.check_model(onnx_model)
        size_threshold = 0 if save_as_external_data else 1024
        onnx.save(onnx_model, file_path, size_threshold=size_threshold,
                  save_as_external_data=save_as_external_data,
                  all_tensors_to_one_file=(not save_as_external_data))

        # Check Graph
        unsupportable_quant_op = set()
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation):
                for cfg, var in op.config_with_variable:
                    if op.type in {'Conv', 'Gemm', 'ConvTranspose'} and var in op.inputs and op.inputs.index(var)==2: continue
                    if not QDQHelper.TQC_Exportable_Check(TQC=cfg, operation=op, bounded_var=var): continue
                    if cfg.num_of_bits != 8 or cfg.policy.has_property(QuantizationProperty.FLOATING):
                        unsupportable_quant_op.add(op)

        if len(unsupportable_quant_op) != 0:
            ppq_warning('Exported Onnx Model is not executable, following Op has onnxruntime-unsupported quant policy:')
            for op in unsupportable_quant_op:
                ppq_warning(f'{op.name} (bitwidth != 8)')
            ppq_warning('For Generating onnxruntime-executable Model, use TargetPlatform = Onnxruntime or OnnxruntimeQuantizer instead.')


 
    def _resorted_variable(self, quantized_graph: BaseGraph, org_graph: BaseGraph, variables: List, is_output: bool=True):
        """
        keep index for export model
        """
        def get_up_ops(graph, op, up_ops=[]):
            # TODO: 不考虑多分支
            up_op = graph.get_upstream_operations(op)[0]
            if up_op.type not in {'DequantizeLinear', 'QuantizeLinear', 'Cast'}:
                up_ops.append(up_op)
                return up_ops
            else:
                up_ops.append(up_op)
                get_up_ops(graph, up_ops[-1], up_ops)
            return up_ops

        if is_output:
            var_names = [x for x in org_graph.outputs]
        else:
            var_names = [x for x in org_graph.inputs]

        sorted_index = []
        unsorted_var = []
        for var in variables:
            if  var.name in var_names:
                sorted_index.append(var_names.index(var.name))
            else:  
                up_op = get_up_ops(quantized_graph, quantized_graph.variables[var.name].source_op)[-1]
                if up_op.outputs[0].name in var_names:
                    sorted_index.append(var_names.index(up_op.outputs[0].name))
                else:
                    sorted_index.append(None)
                    unsorted_var.append(var)
                    print(f'{var.name} is unsorted')

        if None not in sorted_index:
            dict_outputs = list(zip(sorted_index, variables))
            sorted_zip = sorted(dict_outputs, key = lambda a: a[0])
            sorted_vars = [x[1] for x in sorted_zip]
        else:
            sorted_index_ = []
            unsorted_index = [i for i in range(len(sorted_index))]
            for idx, var in zip(sorted_index, variables):
                if idx is not None: 
                    unsorted_index.pop(idx)
                else:
                    up_op = get_up_ops(quantized_graph, quantized_graph.variables[var.name].source_op)[-1]
                    if up_op.name in org_graph.operations:
                        down_op = org_graph.get_downstream_operations(org_graph.operations[up_op.name])
                        if len(down_op)==1 and down_op[0].outputs[0].name in var_names:
                            idx = var_names.index(down_op[0].outputs[0].name)
                    # print(idx)
                sorted_index_.append(idx)

            dict_outputs = list(zip(sorted_index_, variables))
            # avoid not found index
            unsorted_index = iter(unsorted_index)
            sorted_zip = sorted(dict_outputs, 
                                key = lambda a: next(unsorted_index) if a[0] is None else a[0])
            sorted_vars = [x[1] for x in sorted_zip]

        return sorted_vars