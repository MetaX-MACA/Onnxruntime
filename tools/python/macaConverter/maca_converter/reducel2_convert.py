import argparse
import sys

import log
import numpy as np
import onnx
import operation
import values
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto

logger = log.getLogger(__name__, log.INFO)


def is_unused_init(model, init):
    for node in model.graph.node:
        if init.name in node.input:
            return False

    return True

def remove_unused_initializer(model, unused_init_list):
    for init in unused_init_list:
        if is_unused_init(model, init):
            logger.debug('remove unused init: {}'.format(init.name))
            model.graph.initializer.remove(init)

def remove_invalid_sub_node(model):
    invalid_sub_node_list = []
    for node in model.graph.node:
        if node.op_type == 'Sub':
            used = False
            sub_output = node.output[0]
            for n in model.graph.node:
                if sub_output in n.input:
                    used = True
                    break

            if used == False:
                invalid_sub_node_list.append(node)

    for node in invalid_sub_node_list:
        model.graph.node.remove(node)

# pattern 1:
# Pow-->ReduceSum-->Add->Sqrt
class MergeReducel2Pattern1():
    def __init__(self, model):
        logger.debug('MergeReducel2Pattern1 Init')

        self.model = model
        self.got_reducel2 = False
        self.search = True
        self.unused_init_list = []
        self.loop = 0

        self.dict_pow = {}
        self.dict_rs = {}
        self.dict_add = {}
        self.dict_sqrt = {}

        self.keepdims = 0
        self.axes = [0]

    def clear(self):
        self.dict_pow = {}
        self.dict_rs = {}
        self.dict_add = {}
        self.dict_sqrt = {}
        self.search = False
        self.keepdims = 1
        self.axes = [0]

    def merge(self):
        while self.search == True:
            self.clear()

            self.loop = self.loop + 1

            for node_id, node in enumerate(self.model.graph.node):
                self.loop = self.loop + 1
                #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
                #        ", op:", node.op_type, ', len(input):', len(node.input))

                if node.op_type == 'Pow':
                    self.dict_pow['input'] = node.input
                    self.dict_pow['output'] = node.output
                    self.dict_pow['id'] = node_id
                    self.dict_pow['name'] = node.name

                if node.op_type == 'ReduceSum':
                    if self.dict_pow and node.input[0] == self.dict_pow['output'][0]:
                        self.dict_rs['input'] = node.input
                        self.dict_rs['output'] = node.output
                        self.dict_rs['id'] = node_id
                        self.dict_rs['name'] = node.name

                        if len(node.input) == 2:
                            self.axes = values.get_init_value(self.model, node.input[1])
                            if len(self.axes) == 0:
                                self.axes = values.get_constant_value(self.model, node.input[1])
                            if isinstance(self.axes, np.ndarray) == True:
                                self.axes = self.axes.tolist()

                        attributes = node.attribute
                        for attr in attributes:
                            if attr.name == 'axes':
                                self.axes = attr.ints
                            if attr.name == 'keepdims':
                                self.keepdims = attr.i

                        logger.debug('got first pair: {} {}'.format(self.dict_rs['input'], self.dict_rs['output']))
                    else:
                        logger.debug('self.clear ReduceMean, self.dict_rs: {}'.format(self.dict_rs))
                        self.clear()

                if node.op_type == 'Add':
                    if self.dict_pow and self.dict_rs and node.input[0] == self.dict_rs['output'][0]:
                        v = values.get_init_value(self.model, node.input[1])
                        if len(v) == 0:
                            v = values.get_constant_value(model, node.input[1])

                        logger.debug('add B: {} {}'.format(v, type(v)))

                        if v < 0.000001:
                            self.dict_add['input'] = node.input
                            self.dict_add['output'] = node.output
                            self.dict_add['id'] = node_id
                            self.dict_add['name'] = node.name

                            logger.debug('got second pair: {} {}'.format(self.dict_add['input'], self.dict_add['output']))
                        else:
                            logger.debug('--self.clear Pow and ReduceSum')
                            self.clear()
                    else:
                        logger.debug('---self.clear Pow and ReduceSum')
                        self.clear()

                if node.op_type == 'Sqrt':
                    if self.dict_pow and self.dict_rs and self.dict_add and node.input[0] == self.dict_add['output'][0]:
                        self.dict_sqrt['input'] = node.input
                        self.dict_sqrt['output'] = node.output
                        self.dict_sqrt['id'] = node_id
                        self.dict_sqrt['name'] = node.name

                        self.search = True
                        self.got_reducel2 = True
                        logger.debug('Got a ReduceL2 op')

                        div_node, _ = values.get_next_node_by_output(self.model, node.output[0])
                        ###
                        pow_node = self.model.graph.node[self.dict_pow['id']]
                        rs_node = self.model.graph.node[self.dict_rs['id']]
                        add_node = self.model.graph.node[self.dict_add['id']]
                        sqrt_node = self.model.graph.node[self.dict_sqrt['id']]

                        self.model.graph.node.remove(pow_node)

                        extra_args = {}
                        if len(self.axes) != 0:
                            axes = []
                            for a in self.axes:
                                axes.append(a)
                            extra_args['axes'] = axes

                        reducesuml2_node = onnx.helper.make_node(
                                                name = self.dict_pow['name'] + '+' + self.dict_rs['name'] + '+' + self.dict_add['name'] + '+' + self.dict_sqrt['name'],
                                                #name = node.name + '_to_reducel2_' + str(self.loop),
                                                op_type='ReduceL2',
                                                inputs=[self.dict_pow['input'][0]],
                                                outputs=self.dict_sqrt['output'],
                                                keepdims=self.keepdims,
                                                **extra_args
                                                )

                        self.model.graph.node.insert(self.dict_pow['id'], reducesuml2_node)

                        add_node.input[0] = self.dict_sqrt['output'][0]
                        div_node.input[1] = add_node.output[0]

                        self.model.graph.node.remove(rs_node)
                        self.model.graph.node.remove(sqrt_node)

                        break
                    else:
                        logger.debug('self.clear Pow Reducesum Add')
                        logger.debug('self.dict_pow: {}'.format(self.dict_pow))
                        logger.debug('self.dict_: {}'.format(self.dict_rs))
                        logger.debug('self.dict_pow: {}'.format(self.dict_add))
                        self.clear()

        '''
        if self.got_reducel2 == True:
            op_set = self.model.opset_import.add()
            op_set.domain = 'com.metax-tech'
            op_set.version = 1
        '''
            #onnx.save(model, export_onnx)

        remove_unused_initializer(self.model, self.unused_init_list)
        remove_invalid_sub_node(self.model)

        return self.model

# pattern 2:
# Pow-->ReduceMean-->Sqrt


class MergeReducel2Pattern2():
    def __init__(self, model):
        logger.debug('MergeReducel2Pattern2 Init')

        self.model = model
        self.got_reducel2 = False
        self.search = True
        self.unused_init_list = []
        self.loop = 0

        self.dict_pow = {}
        self.dict_rs = {}
        self.dict_sqrt = {}

        self.keepdims = 0
        self.axes = [0]

    def clear(self):
        self.dict_pow = {}
        self.dict_rs = {}
        self.dict_sqrt = {}
        self.search = False
        self.keepdims = 1
        self.axes = [0]

    def merge(self):
        while self.search == True:
            self.clear()

            self.loop = self.loop + 1

            for node_id, node in enumerate(self.model.graph.node):
                self.loop = self.loop + 1
                #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
                #        ", op:", node.op_type, ', len(input):', len(node.input))

                if node.op_type == 'Pow' or node.op_type == 'Mul':
                    self.dict_pow['input'] = node.input
                    self.dict_pow['output'] = node.output
                    self.dict_pow['id'] = node_id
                    self.dict_pow['name'] = node.name

                if node.op_type == 'ReduceSum':
                    if self.dict_pow and node.input[0] == self.dict_pow['output'][0]:
                        self.dict_rs['input'] = node.input
                        self.dict_rs['output'] = node.output
                        self.dict_rs['id'] = node_id
                        self.dict_rs['name'] = node.name

                        if len(node.input) == 2:
                            self.axes = values.get_init_value(self.model, node.input[1])
                            if len(self.axes) == 0:
                                self.axes = values.get_constant_value(self.model, node.input[1])
                            if isinstance(self.axes, np.ndarray) == True:
                                self.axes = self.axes.tolist()

                            #logger.debug('axes: {} {}'.format(axes, type(axes)))

                        attributes = node.attribute
                        for attr in attributes:
                            if attr.name == 'axes':
                                self.axes = attr.ints
                            if attr.name == 'keepdims':
                                self.keepdims = attr.i

                        logger.debug('got first pair: {} {}'.format(self.dict_rs['input'], self.dict_rs['output']))
                    else:
                        logger.debug('self.clear ReduceMean, self.dict_rs: {}'.format(self.dict_rs))
                        self.clear()

                if node.op_type == 'Sqrt':
                    if self.dict_pow and self.dict_rs and node.input[0] == self.dict_rs['output'][0]:
                        self.dict_sqrt['input'] = node.input
                        self.dict_sqrt['output'] = node.output
                        self.dict_sqrt['id'] = node_id
                        self.dict_sqrt['name'] = node.name

                        self.search = True
                        self.got_reducel2 = True
                        logger.debug('Got a ReduceL2 op')

                        ###
                        pow_node = self.model.graph.node[self.dict_pow['id']]
                        rs_node = self.model.graph.node[self.dict_rs['id']]
                        sqrt_node = self.model.graph.node[self.dict_sqrt['id']]

                        self.model.graph.node.remove(pow_node)

                        extra_args = {}
                        if len(self.axes) != 0:
                            axes = []
                            for a in self.axes:
                                axes.append(a)
                            extra_args['axes'] = axes

                        reducesuml2_node = onnx.helper.make_node(
                                                name = self.dict_pow['name'] + '+' + self.dict_rs['name'] + '+' + self.dict_sqrt['name'],
                                                #name = node.name + '_to_reducel2_' + str(self.loop),
                                                op_type='ReduceL2',
                                                inputs=[self.dict_pow['input'][0]],
                                                outputs=self.dict_sqrt['output'],
                                                keepdims=self.keepdims,
                                                **extra_args
                                                )

                        self.model.graph.node.insert(self.dict_pow['id'], reducesuml2_node)

                        self.model.graph.node.remove(rs_node)
                        self.model.graph.node.remove(sqrt_node)

                        break
                    else:
                        logger.debug('self.clear Pow Reducesum')
                        logger.debug('self.dict_pow: {}'.format(self.dict_pow))
                        logger.debug('self.dict_: {}'.format(self.dict_rs))
                        self.clear()

        '''
        if self.got_reducel2 == True:
            op_set = self.model.opset_import.add()
            op_set.domain = 'com.metax-tech'
            op_set.version = 1
        '''
            #onnx.save(model, export_onnx)

        remove_unused_initializer(self.model, self.unused_init_list)
        remove_invalid_sub_node(self.model)

        return self.model

# pattern 3:
# Mul-->ReduceSum-->Max->Sqrt
class MergeReducel2Pattern3():
    def __init__(self, model):
        logger.debug('MergeReducel2Pattern3 Init')

        self.model = model
        self.got_reducel2 = False
        self.search = True
        self.unused_init_list = []
        self.loop = 0

        self.dict_mul = {}
        self.dict_rs = {}
        self.dict_max = {}
        self.dict_sqrt = {}

        self.keepdims = 0
        self.axes = [0]
        self.max_input1 = None
        self.max_input1_name = None

    def clear(self):
        self.dict_mul = {}
        self.dict_rs = {}
        self.dict_max = {}
        self.dict_sqrt = {}
        self.search = False
        self.keepdims = 1
        self.axes = [0]
        self.max_input1 = None
        self.max_input1_name = None

    def merge(self):
        onnx_opset_version = get_onnx_opset_version(self.model)
        while self.search == True:
            self.clear()

            self.loop = self.loop + 1

            for node_id, node in enumerate(self.model.graph.node):
                self.loop = self.loop + 1
                #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
                #        ", op:", node.op_type, ', len(input):', len(node.input))

                if node.op_type == 'Mul':
                    self.dict_mul['input'] = node.input
                    self.dict_mul['output'] = node.output
                    self.dict_mul['id'] = node_id
                    self.dict_mul['name'] = node.name

                if node.op_type == 'ReduceSum':
                    if self.dict_mul and node.input[0] == self.dict_mul['output'][0]:
                        self.dict_rs['input'] = node.input[:1]
                        self.dict_rs['output'] = node.output
                        self.dict_rs['id'] = node_id
                        self.dict_rs['name'] = node.name

                        if len(node.input) == 2:
                            self.axes = values.get_init_value(self.model, node.input[1])
                            if len(self.axes) == 0:
                                self.axes = values.get_constant_value(self.model, node.input[1])
                            if isinstance(self.axes, np.ndarray) == True:
                                self.axes = self.axes.tolist()

                        attributes = node.attribute

                        for attr in attributes:
                            if attr.name == 'keepdims':
                                self.keepdims = attr.i
                            if attr.name == 'axes' and 0 < onnx_opset_version < 13:
                                self.axes = attr.ints

                        logger.debug('got first pair: {} {}'.format(self.dict_rs['input'], self.dict_rs['output']))
                    else:
                        logger.debug('self.clear ReduceMean, self.dict_rs: {}'.format(self.dict_rs))
                        self.clear()

                if node.op_type == 'Max':
                    if self.dict_mul and self.dict_rs and node.input[0] == self.dict_rs['output'][0]:
                        self.max_input1_name = node.input[1] if len(node.input) == 2 else None
                        max_input1_input = next((initializer for initializer in self.model.graph.initializer if initializer.name == self.max_input1_name), None) if self.max_input1_name is not None else None
                        self.max_input1 = numpy_helper.to_array(max_input1_input) if max_input1_input is not None else None

                        logger.debug('Max B: {} {}'.format(self.max_input1, len(self.max_input1.shape)))

                        if len(self.max_input1.shape) == 0 and abs(self.max_input1- 1e-10) < 0.00001:
                            self.dict_max['input'] = node.input
                            self.dict_max['output'] = node.output
                            self.dict_max['id'] = node_id
                            self.dict_max['name'] = node.name

                            logger.debug('got second pair: {} {}'.format(self.dict_max['input'], self.dict_max['output']))
                        else:
                            logger.debug('--self.clear Pow and ReduceSum')
                            self.clear()
                    else:
                        logger.debug('---self.clear Pow and ReduceSum')
                        self.clear()

                if node.op_type == 'Sqrt':
                    if self.dict_mul and self.dict_rs and self.dict_max and node.input[0] == self.dict_max['output'][0]:
                        self.dict_sqrt['input'] = node.input
                        self.dict_sqrt['output'] = node.output
                        self.dict_sqrt['id'] = node_id
                        self.dict_sqrt['name'] = node.name

                        self.search = True
                        self.got_reducel2 = True
                        logger.debug('Got a ReduceL2 op')

                        #all_next_nodes, _ = operation.get_all_next_node_by_output(self.model, node.output[0])
                        ###
                        pow_node = self.model.graph.node[self.dict_mul['id']]
                        rs_node = self.model.graph.node[self.dict_rs['id']]
                        max_node = self.model.graph.node[self.dict_max['id']]
                        sqrt_node = self.model.graph.node[self.dict_sqrt['id']]

                        self.model.graph.node.remove(pow_node)


                        extra_args = {}
                        if len(self.axes) != 0:
                            axes = []
                            for a in self.axes:
                                axes.append(a)
                            extra_args['axes'] = axes


                        reducesuml2_node = onnx.helper.make_node(
                                                name = self.dict_mul['name'] + '+' + self.dict_rs['name'] + '+' + self.dict_max['name'] + '+' + self.dict_sqrt['name'],
                                                #name = node.name + '_to_reducel2_' + str(self.loop),
                                                op_type='ReduceL2',
                                                inputs=[self.dict_mul['input'][0]],
                                                outputs=self.dict_sqrt['output'],
                                                keepdims=self.keepdims,
                                                **extra_args
                                                )

                        self.model.graph.node.insert(self.dict_mul['id'], reducesuml2_node)

                        ###
                        all_next_node, _ = operation.get_all_next_node_by_output(self.model, node.output[0])
                        for next_node in all_next_node:
                            for i, input_ in enumerate(next_node.input):
                                if node.output[0] == input_:
                                    next_node.input[i] = max_node.output[0]

                        fused_type = onnx.TensorProto.FLOAT
                        if self.max_input1.dtype == np.float16:
                            fused_type = onnx.TensorProto.FLOAT16

                        fused_b = np.sqrt(self.max_input1)

                        max_b_name_new = self.max_input1_name + '_new_'
                        max_b_new = helper.make_tensor(max_b_name_new, fused_type, fused_b.shape, fused_b.flatten())
                        self.model.graph.initializer.extend([max_b_new])

                        max_node.input[0] = self.dict_sqrt['output'][0]
                        max_node.input[1] = max_b_name_new

                        self.model.graph.node.remove(rs_node)
                        self.model.graph.node.remove(sqrt_node)

                        operation.remove_initializer_if_necessary_by_name(self.model, self.max_input1_name, max_node)

                        break
                    else:
                        logger.debug('self.clear Pow Reducesum Add')
                        logger.debug('self.dict_mul: {}'.format(self.dict_mul))
                        logger.debug('self.dict_: {}'.format(self.dict_rs))
                        logger.debug('self.dict_max: {}'.format(self.dict_max))
                        self.clear()

        '''
        if self.got_reducel2 == True:
            op_set = self.model.opset_import.add()
            op_set.domain = 'com.metax-tech'
            op_set.version = 1
        '''
            #onnx.save(model, export_onnx)

        remove_unused_initializer(self.model, self.unused_init_list)
        remove_invalid_sub_node(self.model)

        return self.model


def get_onnx_opset_version(model_proto):
    onnx_import_opset = -1
    for opset in model_proto.opset_import:
        if opset.domain == 'ai.onnx' or opset.domain == '':
            onnx_import_opset = opset.version
            break
    return onnx_import_opset

def merge_reducel2(model):
    mlp1 = MergeReducel2Pattern1(model)
    model = mlp1.merge()

    mlp2 = MergeReducel2Pattern2(model)
    model = mlp2.merge()

    mlp3 = MergeReducel2Pattern3(model)
    model = mlp3.merge()

    return model

'''
model = onnx.load('/home/zqiu/models/GFPGANv1.3.onnx')
model = onnx.load('/home/zqiu/models/facenet_sim.onnx')
m = merge_reducel2(model)
onnx.save(m, 'reducel2.onnx')
'''
