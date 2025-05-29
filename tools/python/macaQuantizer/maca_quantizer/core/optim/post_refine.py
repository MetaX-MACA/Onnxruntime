
from typing import Iterable, List, Set, Union
from ppq import (BaseGraph, BaseGraphExecutor,TargetPlatform, SearchableGraph, 
                 QuantableOperation, QuantizationStates,Operation, convert_any_to_python_primary_type,
                 DataType)
from ppq.quantization.optim.base import QuantizationOptimizationPass
from maca_quantizer.utils.utils import maca_info, maca_warning




class MetaxMixturePass(QuantizationOptimizationPass):
    def __init__(self, ops=[]) -> None:
        super().__init__('Mixture Quantization Optimztion')
        self._op_list = ops

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        for operation in graph.topological_sort():
            # TODO: filter soi and postprocess operation 
            if operation.name not in self._op_list: continue
            if operation.extension_attrib.get("PostPro", False): continue  
            if operation.platform != TargetPlatform.FP32: continue
            if operation.type in {'Cast'}: continue
            pass



class MetaxFP16Pass(QuantizationOptimizationPass):
    """
    insert cast to convert operation to fp16

    cast-fp16 --> op1 --> cast-fp32 --> cast-fp16 --> op2 --> cast-fp32

    ==> remove duplicated pair cast op in exporter  

    cast-fp16 --> op1 --> op2 --> cast-fp32

    """

    def __init__(self, ops) -> None:
        super().__init__('Metax FP16 Optimztion')
        self._op_list = ops

    def optimize(self, graph: BaseGraph,
                 dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:

        # TODO: postprocess hybrid operation to fp16
        self.convert_post_operation_fp16(graph)

        # backbone hybrid operation to fp16
        for operation in graph.topological_sort():
            if operation.platform != TargetPlatform.FP32: continue
            if operation.extension_attrib.get("PostPro", False): continue  
            if operation.type in {'Cast'}: continue
            if operation.extension_attrib.get("ConvEleAct", False): continue

            self.convert_operation_fp16(graph, operation)
            # TODO: platform assign to TargetPlatform.FP16
            operation.platform = TargetPlatform.FP16


    def convert_post_operation_fp16(self, graph: BaseGraph):
        """
        op1 --> var --> op2 
          ===>  
        op1 --> var --> cast_fp16 --> var_out_fp16 --> op2
        """
        for operation in graph.topological_sort():
            if operation.extension_attrib.get("PostPro_start", False):
                # if operation.name in self._op_list: continue
                # if operation.platform==TargetPlatform.FP32: continue
                # if operation.platform==TargetPlatform.FP16: continue
                if operation.is_boundary: continue
                for idx, var in enumerate(operation.outputs):
                    if var.is_parameter: continue
                    # TODO: Maybe cause schema error
                    if any([op.platform.name not in {'FP32', 'FP16'} for  op in var.dest_ops]): continue

                    out_cast_attr = {'to': DataType.FP16}
                    out_cast_name = 'Cast_post_start_{}_{}'.format(operation.name, idx)
                    out_var_name  = 'post_start_out_var_{}_{}'.format(operation.name, idx)

                    var_out_fp16 = graph.create_variable(name=out_var_name, is_parameter=False)
                    cast_fp16 = graph.create_operation(op_type='Cast',
                                                        name=out_cast_name, 
                                                        attributes=out_cast_attr, 
                                                        platform=TargetPlatform.FP32)
                    down_ops = var.dest_ops.copy()

                    # add to graph.
                    if cast_fp16.name not in graph.operations.keys():
                        graph.append_operation(cast_fp16)

                    var.dest_ops.clear()
                    var.dest_ops.append(cast_fp16)

                    cast_fp16.inputs.append(var)
                    cast_fp16.outputs.append(var_out_fp16)

                    for op in down_ops:
                        op.inputs[op.inputs.index(var)] = var_out_fp16

                    if var in graph.outputs:
                        graph.outputs.pop(var)
                        graph.outputs[var_out_fp16.name] = var_out_fp16

                    var_out_fp16.source_op = cast_fp16
                    var_out_fp16.dest_ops.extend(down_ops)

                    var_out_fp16.dtype = DataType.FP16
                    var_out_fp16.shape = var.shape


            elif operation.extension_attrib.get("PostPro", False):
                operation.platform = TargetPlatform.FP16
                # if operation.name in self._op_list: continue
                # TODO: op1 --> var   ==>  op1 --> var_in_fp16 --->  cast-fp16 --> var
                if operation.is_boundary and operation.platform == TargetPlatform.FP16:
                    var_output = operation.outputs[0]
                    if var_output.dtype.name not in {'FP32', 'FP64'}:
                        continue
                    out_cast_attr = {'to': DataType.FP32}
                    out_cast_name = 'Cast_out_op_{}_0'.format(operation.name) 
                    out_var_name  = 'Cast_out_var_{}_0'.format(operation.name)

                    var_out_fp16 = graph.create_variable(name=out_var_name, is_parameter=False)
                    cast_fp32 = graph.create_operation(op_type='Cast',
                                                        name=out_cast_name, 
                                                        attributes=out_cast_attr, 
                                                        platform=TargetPlatform.FP32)

                    operation.outputs[operation.outputs.index(var_output)] = var_out_fp16

                    var_out_fp16.source_op = operation
                    var_out_fp16.dest_ops.append(cast_fp32)

                    cast_fp32.inputs.append(var_out_fp16)
                    cast_fp32.outputs.append(var_output)

                    var_output.source_op = cast_fp32

                    var_out_fp16.dtype = DataType.FP16
                    var_out_fp16.shape = var_output.shape

                # TODO: check multiply input
                num_of_input = len(operation.inputs) - operation.num_of_parameter
                if num_of_input > 1:
                    check_dtype = [var.dtype == DataType.FP16  for var in  operation.inputs]
                    op_check = operation.type in {'Concat', 'Add', 'Sub', 'Mul', 'Div'}
                    # op_check = True
                    for idx, var in enumerate(operation.outputs):
                        if var.dtype == DataType.FP32:
                            var.dtype = DataType.FP16

                    if not all(check_dtype) and op_check:
                        """
                            var1 --> op2 
                                      |
                            var2 -----|
                        ===>  
                            var1 --> cast_fp16 --> var_out_fp16 --> op2
                                                                     |
                            var2 --> cast_fp16 --> var_in_fp16  ---- |
                        """
                        # maca_warning(f"{operation.name} multiply input dtype not same")
                        for idx, var_ in enumerate(operation.inputs):
                            if var_.is_parameter:
                                if var_.dtype == DataType.FP32:
                                    var_.dtype = DataType.FP16
                                    if var_.value is not None:
                                        var_.value = var_.value.half()
                            else:
                                if var_.dtype == DataType.FP16: 
                                    continue
                                if var_.dtype.name not in {'FP32', 'FP64'}:
                                    continue
                                in_cast_attr = {'to': DataType.FP16}
                                in_cast_name = 'Cast_external_op_{}_{}'.format(operation.name, idx) 
                                in_var_name  = 'Cast_external_var_{}_{}'.format(operation.name, idx)

                                var_in_fp16 = graph.create_variable(name=in_var_name, is_parameter=False)
                                cast_fp16 = graph.create_operation(op_type='Cast',
                                                                    name=in_cast_name, 
                                                                    attributes=in_cast_attr, 
                                                                    platform=TargetPlatform.FP32)

                                operation.inputs[operation.inputs.index(var_)] = var_in_fp16
                                var_.dest_ops[var_.dest_ops.index(operation)] = cast_fp16

                                cast_fp16.inputs.append(var_)
                                cast_fp16.outputs.append(var_in_fp16)
                                var_in_fp16.source_op = cast_fp16
                                var_in_fp16.dest_ops.append(operation)

                                var_in_fp16.dtype = DataType.FP16
                                var_in_fp16.shape = var_.shape



                else:
                    for idx, var in enumerate(operation.inputs):
                        # Resize scales/size not support fp16
                        if operation.type == 'Resize' and idx > 1: 
                            continue
                        if var.is_parameter:
                            if var.dtype == DataType.FP32:
                                var.dtype = DataType.FP16
                                if var.value is not None:
                                    var.value = var.value.half()
                        else:
                            if var.dtype == DataType.FP32:
                                var.dtype = DataType.FP16

                    for idx, var in enumerate(operation.outputs):
                        if var.dtype == DataType.FP32:
                            var.dtype = DataType.FP16


    def convert_operation_fp16(self, graph: BaseGraph, op: Operation):
        # convert Initializer to fp16
        for idx, var in enumerate(op.inputs):
            # Resize scales/size not support fp16
            if op.type == 'Resize' and idx > 1:
                continue
            if not var.is_parameter: continue
            if var.dtype == DataType.FP32:
                var.dtype = DataType.FP16
                if var.value is not None:
                    var.value = var.value.half()

        self.insert_cast_pair(graph, op)


    @staticmethod
    def insert_cast_pair(graph: BaseGraph, op: Operation):

        """
        var1 --> op --> var2    

        ==>  

        var1 --> cast_fp16 --> var_in_fp16 --> op --> var_out_fp16 --> cast_fp32 --> var2

        """
        # var1 = op.inputs[0]
        var2 = op.outputs[0]

        # insert cast_fp16 before operation 
        for idx, var1 in enumerate(op.inputs):
            if var1.is_parameter: 
                continue
            if var1.source_op is not None and var1.source_op.type in {'Shape'}:
                continue
            in_cast_attr = {'to': DataType.FP16}
            in_cast_name = 'Cast_in_op_{}_{}'.format(op.name, idx)   
            in_var_name  = 'Cast_in_var_{}_{}'.format(op.name, idx)

            cast_fp16 = graph.create_operation(op_type='Cast',
                                                name=in_cast_name, 
                                                attributes=in_cast_attr, 
                                                platform=TargetPlatform.FP32)
            var_in_fp16 = graph.create_variable(name=in_var_name, is_parameter=False)
            var1.dest_ops[var1.dest_ops.index(op)]= cast_fp16

            cast_fp16.inputs.append(var1)
            cast_fp16.outputs.append(var_in_fp16)

            var_in_fp16.source_op = cast_fp16
            var_in_fp16.dest_ops.append(op)

            op.inputs[op.inputs.index(var1)]  = var_in_fp16

            var_in_fp16.dtype = DataType.FP16
            var_in_fp16.shape = var1.shape

        # insert cast_fp32 behind operation 
        if var2.source_op.type not in {'Shape'}:
            out_cast_attr = {'to': DataType.FP32}
            out_cast_name = 'Cast_out_op_{}_0'.format(op.name) 
            out_var_name  = 'Cast_out_var_{}_0'.format(op.name)

            var_out_fp16 = graph.create_variable(name=out_var_name, is_parameter=False)
            cast_fp32 = graph.create_operation(op_type='Cast',
                                                name=out_cast_name, 
                                                attributes=out_cast_attr, 
                                                platform=TargetPlatform.FP32)

            op.outputs[op.outputs.index(var2)] = var_out_fp16

            var_out_fp16.source_op = op
            var_out_fp16.dest_ops.append(cast_fp32)

            cast_fp32.inputs.append(var_out_fp16)
            cast_fp32.outputs.append(var2)

            var2.source_op = cast_fp32

            var_out_fp16.dtype = DataType.FP16
            var_out_fp16.shape = var2.shape
        
        if op.type in {'LSTM'}:
            for i in range(1, len(op.outputs)):
                var_out = op.outputs[i]

                var_out.dtype = DataType.FP16
                var_out.shape = var_out.shape

                if len(var_out.dest_ops) != 0:
                    out_cast_attr = {'to': DataType.FP32}
                    out_cast_name = 'Cast_out_op_{}_{}'.format(op.name, i) 
                    out_var_name  = 'Cast_out_var_{}_{}'.format(op.name, i)

                    var_out_fp16 = graph.create_variable(name=out_var_name, is_parameter=False)
                    cast_fp32 = graph.create_operation(op_type='Cast',
                                                        name=out_cast_name, 
                                                        attributes=out_cast_attr, 
                                                        platform=TargetPlatform.FP32)

                    op.outputs[op.outputs.index(var_out)] = var_out_fp16

                    var_out_fp16.source_op = op
                    var_out_fp16.dest_ops.append(cast_fp32)

                    cast_fp32.inputs.append(var_out_fp16)
                    cast_fp32.outputs.append(var_out)

                    var_out.source_op = cast_fp32


class MetaxAutoCastPass(QuantizationOptimizationPass):
    def __init__(self, scale_threshold=128) -> None:
        super().__init__('Metax AutoCast Quantization Optimztion')
        self._scale_threshold = scale_threshold

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:

        interested_ops = []
        for operation in graph.topological_sort():
            # TODO: filter soi and postprocess operation 
            if operation.extension_attrib.get("PostPro", False): continue  
            if operation.platform == TargetPlatform.FP32: continue
            if not isinstance(operation, QuantableOperation): continue
            if operation.type in {'Cast'}: continue
            for qconfig in operation.input_quant_config:
                if qconfig.state == QuantizationStates.FP32: continue
                scale = convert_any_to_python_primary_type(qconfig.scale)
                if isinstance(scale, list) and len(scale) > 1: continue
                if scale > self._scale_threshold:
                    interested_ops.append(operation.name)
                    qconfig.state = QuantizationStates.FP32
                    # break
            for qconfig in operation.output_quant_config:
                if qconfig.state == QuantizationStates.FP32: continue
                scale = convert_any_to_python_primary_type(qconfig.scale)
                if isinstance(scale, list) and len(scale) > 1: continue
                if scale > self._scale_threshold:
                    interested_ops.append(operation.name)
                    qconfig.state = QuantizationStates.FP32
                    # break

        interested_ops = set(interested_ops)
        maca_info(f"============== {[op for op in interested_ops]}")
        for op_name in interested_ops:
            graph.operations[op_name].platform = TargetPlatform.FP32


