import onnx
import sys
import argparse
import values, sys
import numpy as np       
from onnx import onnx_pb as onnx_proto
import log

logger = log.getLogger(__name__, log.INFO)

def make_fp32_tensor_from_fp16(fp16_tensor, fp32_tensor_name):
    float_list = []
    if fp16_tensor.int32_data:
        logger.debug('make_fp32_tensor_from_fp16, int32_data')
        num = np.array(fp16_tensor.int32_data)
        float_list = num.astype(np.float_).tolist()

    if fp16_tensor.raw_data:
        logger.debug('make_fp32_tensor_from_fp16, raw_data')
        float_list = np.fromstring(fp16_tensor.raw_data, dtype='float16')
        num = np.array(float_list)
        float_list = num.astype(np.float_).tolist()

    t = onnx.helper.make_tensor(name=fp32_tensor_name,
                                    data_type=onnx_proto.TensorProto.FLOAT,
                                    dims=fp16_tensor.dims,
                                    vals=float_list)

    return t                                

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

def remove_invalid_tensor(model):
    invalid_value_info_list = []
    for vi in model.graph.value_info:
        valid = False
        for node in model.graph.node:
            if vi.name in node.input or vi.name in node.output:
                valid = True
                break

        if valid == False:
           invalid_value_info_list.append(vi)

    for invalid_vi in invalid_value_info_list:
        model.graph.value_info.remove(invalid_vi)
        #print('delete invalid value_info:', invalid_vi.name)


# pattern 1:
#                                 ---     ---     ---      ---        ---       ---    ---    --
#                               |                                                              |
# ***(0) --- ReduceMean(1) --- Sub(2) --- Pow(3) --- ReduceMean(4) --- Add(5) --- Sqrt(6) --- Div(7) --- Mul(8) --- (Add)(9)
#      |                     |
#       ---   ---   ---   ---    

class MergeLNPattern1():
    def __init__(self, model):
        logger.debug('MergeLNPattern1 Init')
        self.model = model
        self.got_ln = False
        self.search = True
        self.unused_init_list = []
        self.loop = 0

        self.dict_rm = {}
        self.dict_sub = {}
        self.dict_pow = {}
        self.dict_rm2 = {}
        self.dict_add = {}
        self.dict_sqrt = {}
        self.dict_div = {}
        self.dict_mul = {}
        self.dict_add2 = {}

        self.rm1_axes = [-1]
        self.rm2_axes = [-1]

        self.ready = False
        self.ready2 =  False
        
    def clear(self):
        self.dict_rm = {}
        self.dict_sub = {}
        self.dict_pow = {}
        self.dict_rm2 = {}
        self.dict_add = {}
        self.dict_sqrt = {}
        self.dict_div = {}
        self.dict_mul = {}
        self.dict_add2 = {}

        self.rm1_axes = [-1]
        self.rm2_axes = [-1]

        self.ready = False
        self.ready2 =  False
        self.search = False

    def merge(self):
        while self.search == True:
            self.clear()

            self.loop = self.loop + 1

            for node_id, node in enumerate(self.model.graph.node):
                self.loop = self.loop + 1
                #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
                #        ", op:", node.op_type, ', len(input):', len(node.input))

                if node.op_type == 'ReduceMean':
                    if self.ready == True:
                        if node.input == self.dict_pow['output']:
                            self.dict_rm2['input'] = node.input
                            self.dict_rm2['output'] = node.output
                            self.dict_rm2['id'] = node_id
                            self.dict_rm2['name'] = node.name

                            attributes = node.attribute
                            for attr in attributes:
                                if attr.name == 'axes':
                                    self.rm2_axes = attr.ints
                                    logger.debug('self.rm2_axes: {}'.format(self.rm2_axes))
                                    if len(self.rm2_axes) != 1 or self.rm2_axes != self.rm1_axes:
                                        logger.debug('--This ReduceMean IsNot we are looking for...')
                                        self.clear()
                        else: 
                            logger.debug('self.clear ReduceMean Sub Pow')
                            logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                            logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                            logger.debug('self.dict_pow: {}'.format(self.dict_pow))
                            self.clear()
    
                    else: 
                        self.dict_rm['input'] = node.input
                        self.dict_rm['output'] = node.output
                        self.dict_rm['id'] = node_id
                        self.dict_rm['name'] = node.name

                        attributes = node.attribute
                        for attr in attributes:
                            if attr.name == 'axes':
                                self.rm1_axes = attr.ints
                                logger.debug('self.rm1_axes: {}'.format(self.rm1_axes))
                                if len(self.rm1_axes) != 1:
                                    logger.debug('This ReduceMean IsNot we are looking for...')
                                    self.clear()

                                break

                if node.op_type == 'Sub':
                    if self.dict_rm and node.input[0] == self.dict_rm['input'][0] and node.input[1] == self.dict_rm['output'][0]:
                        self.dict_sub['input'] = node.input
                        self.dict_sub['output'] = node.output
                        self.dict_sub['id'] = node_id
                        self.dict_sub['name'] = node.name
                        logger.debug('got first pair: {} {}'.format(self.dict_sub['input'], self.dict_sub['output']))
                    else:
                        logger.debug('self.clear ReduceMean, self.dict_rm: {}'.format(self.dict_rm))
                        self.clear()

                if node.op_type == 'Pow':
                    if self.dict_rm and self.dict_sub and node.input[0] == self.dict_sub['output'][0]:
                        v = values.get_init_value(self.model, node.input[1])
                        logger.debug('pow exp: {} {}'.format(v, type(v)))
                        if v == 2:
                            self.ready = True
                            self.dict_pow['input'] = node.input
                            self.dict_pow['output'] = node.output
                            self.dict_pow['id'] = node_id
                            self.dict_pow['name'] = node.name

                            logger.debug('got second pair: {} {}'.format(self.dict_pow['input'], self.dict_pow['output']))
                        else:
                            logger.debug('--self.clear ReduceMean and Sub')
                            logger.debug('--self.dict_rm: {}'.format(self.dict_rm))
                            logger.debug('--self.dict_sub: {}'.format(self.dict_sub))
                            self.clear()     
                    else:
                        logger.debug('self.clear ReduceMean and Sub')
                        logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                        logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                        self.clear()

                if node.op_type == 'Add':
                    if self.ready == True and self.ready2 == True:
                        if node.input[0] == self.dict_mul['output'][0]:
                            self.dict_add2['input'] = node.input
                            self.dict_add2['output'] = node.output
                            self.dict_add2['id'] = node_id
                            self.dict_add2['name'] = node.name

                            self.search = True
                            self.got_ln = True
                            logger.debug('Got a LayerNorm op')
                            ###
                            rm_node = self.model.graph.node[self.dict_rm['id']]
                            sub_node = self.model.graph.node[self.dict_sub['id']]
                            pow_node = self.model.graph.node[self.dict_pow['id']]
                            rm2_node = self.model.graph.node[self.dict_rm2['id']]
                            add_node = self.model.graph.node[self.dict_add['id']]
                            sqrt_node = self.model.graph.node[self.dict_sqrt['id']]
                            div_node = self.model.graph.node[self.dict_div['id']]
                            mul_node = self.model.graph.node[self.dict_mul['id']]
                            add2_node = self.model.graph.node[self.dict_add2['id']]

                            self.model.graph.node.remove(rm_node)


                            scale_name = self.dict_mul['input'][1]
                            beta_name = self.dict_add2['input'][1]

                            found_init_scale = False
                            found_init_beta = False
                            for init in self.model.graph.initializer:
                                if scale_name == init.name:
                                    found_init_scale = True
                                if beta_name == init.name:
                                    found_init_beta = True

                            if found_init_scale == False:
                                scale_name = self.dict_mul['input'][0]    

                            if found_init_beta == False:
                                beta_name = self.dict_add2['input'][0]    

                            '''
                            for init in self.model.graph.initializer:
                                if init.name == self.dict_mul['input'][1]:
                                    if init.data_type == onnx_proto.TensorProto.FLOAT16:
                                        scale_name = scale_name + '_to_fp32__'
                                        t = make_fp32_tensor_from_fp16(init, scale_name)
                                        self.model.graph.initializer.append(t)

                                        if init not in self.unused_init_list:
                                            self.unused_init_list.append(init)

                                elif init.name == self.dict_add2['input'][1]:
                                    if init.data_type == onnx_proto.TensorProto.FLOAT16:
                                        beta_name = beta_name + '_to_fp32__'
                                        t = make_fp32_tensor_from_fp16(init, beta_name)
                                        self.model.graph.initializer.append(t)

                                        if init not in self.unused_init_list:
                                            self.unused_init_list.append(init)            
                            '''

                            ln_node_name = self.dict_rm['name'] + '+' + self.dict_sub['name'] + '+' + self.dict_pow['name'] + '+' + self.dict_rm2['name'] \
                                            + self.dict_add['name'] + '+' + self.dict_sqrt['name'] + '+' + self.dict_div['name'] + '+' + self.dict_mul['name'] + '+' + self.dict_add2['name']
                                             

                            ln_node = onnx.helper.make_node(
                                                    name = ln_node_name,
                                                    #name = node.name + '_to_layernorm_' + str(self.loop),
                                                    op_type='LayerNorm',
                                                    inputs=[self.dict_rm['input'][0], scale_name, beta_name],
                                                    outputs=self.dict_add2['output'],
                                                    axis=self.rm1_axes[0],
                                                    eps=1e-5,
                                                    elementwise_affine=1,
                                                    domain='com.metax-tech'
                                                    )

                            self.model.graph.node.insert(self.dict_rm['id'], ln_node)

                            self.model.graph.node.remove(sub_node)
                            self.model.graph.node.remove(pow_node)
                            self.model.graph.node.remove(rm2_node)
                            self.model.graph.node.remove(add_node)
                            self.model.graph.node.remove(sqrt_node)
                            self.model.graph.node.remove(div_node)
                            self.model.graph.node.remove(mul_node)
                            self.model.graph.node.remove(add2_node)

                            break
                        else:
                            logger.debug('--self.clear ReduceMean Sub Pow ReduceMean2')
                            logger.debug('--self.dict_rm: {}'.format(self.dict_rm))
                            logger.debug('--self.dict_sub: {}'.format(self.dict_sub))
                            logger.debug('--self.dict_pow: {}'.format(self.dict_pow))
                            logger.debug('--self.dict_rm2: {}'.format(self.dict_rm2))
                            self.clear()
                    else:
                        if self.dict_rm and self.dict_sub and self.dict_pow and self.dict_rm2 and node.input[0] == self.dict_rm2['output'][0]:
                            self.dict_add['input'] = node.input
                            self.dict_add['output'] = node.output
                            self.dict_add['name'] = node.name
                            self.dict_add['id'] = node_id
                            logger.debug('got third pair: {} {}'.format(self.dict_add['input'], self.dict_add['output']))
                        else:
                            logger.debug('self.clear ReduceMean Sub Pow ReduceMean2')
                            logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                            logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                            logger.debug('self.dict_pow: {}'.format(self.dict_pow))
                            logger.debug('self.dict_rm2: {}'.format(self.dict_rm2))
                            self.clear()

                if node.op_type == 'Sqrt':
                    if self.dict_rm and self.dict_sub and self.dict_pow and self.dict_rm2 and  \
                            self.dict_add and node.input == self.dict_add['output']:
                        self.dict_sqrt['input'] = node.input
                        self.dict_sqrt['output'] = node.output
                        self.dict_sqrt['name'] = node.name
                        self.dict_sqrt['id'] = node_id
                        logger.debug('got forth pair: {} {}'.format(self.dict_sqrt['input'], self.dict_sqrt['output']))
                    else:
                        logger.debug('self.clear ReduceMean Sub Pow ReduceMean2 Add')
                        logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                        logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                        logger.debug('self.dict_pow: {}'.format(self.dict_pow))
                        logger.debug('self.dict_rm2: {}'.format(self.dict_rm2))
                        logger.debug('self.dict_add: {}'.format(self.dict_add))
                        self.clear()

                if node.op_type == 'Div':
                    if self.dict_rm and self.dict_sub and self.dict_pow and self.dict_rm2 and  \
                            self.dict_add and self.dict_sqrt and node.input[0] == self.dict_sub['output'][0] \
                            and node.input[1] == self.dict_sqrt['output'][0]:
                        self.dict_div['input'] = node.input
                        self.dict_div['output'] = node.output
                        self.dict_div['id'] = node_id
                        self.dict_div['name'] = node.name

                        logger.debug('got fifth pair: {} {}'.format(self.dict_div['input'], self.dict_div['output']))
                    else:
                        logger.debug('-self.clear ReduceMean Sub Pow ReduceMean2 Add Sqrt')
                        logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                        logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                        logger.debug('self.dict_pow: {}'.format(self.dict_pow))
                        logger.debug('self.dict_rm2: {}'.format(self.dict_rm2))
                        logger.debug('self.dict_add: {}'.format(self.dict_add))
                        logger.debug('self.dict_sqrt: {}'.format(self.dict_sqrt))
                        self.clear()

                if node.op_type == 'Mul':
                    if self.dict_rm and self.dict_sub and self.dict_pow and self.dict_rm2 and  \
                            self.dict_add and self.dict_sqrt and self.dict_div and (node.input[0] == self.dict_div['output'][0] or node.input[1] == self.dict_div['output'][0]):
                        self.dict_mul['input'] = node.input
                        self.dict_mul['output'] = node.output
                        self.dict_mul['id'] = node_id
                        self.dict_mul['name'] = node.name
                        self.ready2 = True

                        logger.debug('got sixth pair: {} {}'.format(self.dict_mul['input'], self.dict_mul['output']))
                        #print('got scale:', scale)
                    else:
                        logger.debug('--self.clear ReduceMean Sub Pow ReduceMean2 Add Sqrt Div')
                        logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                        logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                        logger.debug('self.dict_pow: {}'.format(self.dict_pow))
                        logger.debug('self.dict_rm2: {}'.format(self.dict_rm2))
                        logger.debug('self.dict_add: {}'.format(self.dict_add))
                        logger.debug('self.dict_sqrt: {}'.format(self.dict_sqrt))
                        logger.debug('self.dict_mul: {}'.format(self.dict_mul))
                        self.clear()                           

        if self.got_ln == True:
            op_set = self.model.opset_import.add()
            op_set.domain = 'com.metax-tech'
            op_set.version = 1
            
            #onnx.save(model, export_onnx)

        remove_unused_initializer(self.model, self.unused_init_list)
        remove_invalid_sub_node(self.model)
        
        return self.model

# pattern 2:
#  ---   ---  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---  -- Mul(11)   ---   ---   Add(12) ---
#  ^                             |                                                                                ^                     ^
#  |                             v                                                                                |                     |
# ***(0) --- ReduceMean(1) --- Sub(2) --- Mul(3) --- ReduceMean(4) --- Add(5) --- Sqrt(6) --- Reciprocal(7) --- Mul(8) --- Mul(9) --- Sub(10)
#             |                                                                                                             |
#             v                                                                                                             ^
#             --   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---  ---  --- --

class MergeLNPattern2():
    def __init__(self, model):
        logger.debug('MergeLNPattern2 Init--------------------------')
        self.model = model
        self.got_ln = False
        self.search = True
        self.loop = 0
        self.unused_init_list = []

        self.dict_rm = {}
        self.dict_sub = {}
        self.dict_sub2 = {}
        self.dict_mul = {}
        self.dict_rm2 = {}
        self.dict_add = {}
        self.dict_sqrt = {}
        self.dict_div = {}
        self.dict_mul2 = {}
        self.dict_mul3 = {}
        self.dict_mul4 = {}
        self.dict_add2 = {}

        self.rm1_axes = [-1]
        self.rm2_axes = [-1]

        self.ready_for_reducemean = False
        self.ready_for_mul_first = False
        self.ready_for_mul_second = False
        self.ready_for_mul_third = False
        self.ready_for_sub = False
        self.got_rm_axes = False
        self.ready2 =  False

    def clear(self):
        logger.debug('MergeLNPattern2 clear--------------------------')
        self.unused_init_list = []

        self.dict_rm = {}
        self.dict_sub = {}
        self.dict_sub2 = {}
        self.dict_mul = {}
        self.dict_rm2 = {}
        self.dict_add = {}
        self.dict_sqrt = {}
        self.dict_div = {}
        self.dict_mul2 = {}
        self.dict_mul3 = {}
        self.dict_mul4 = {}
        self.dict_add2 = {}

        self.rm1_axes = [-1]
        self.rm2_axes = [-1]

        self.ready_for_reducemean = False
        self.ready_for_mul_first = False
        self.ready_for_mul_second = False
        self.ready_for_mul_third = False
        self.ready_for_sub = False
        self.got_rm_axes = False
        self.ready2 =  False
        self.search  = False

    def merge(self):
        while self.search == True:
            self.loop = self.loop + 1

            self.clear()

            for node_id, node in enumerate(self.model.graph.node):
                self.loop = self.loop + 1
                #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
                #        ", op:", node.op_type, ', len(input):', len(node.input))

                if node.op_type == 'ReduceMean' or node.op_type == 'GlobalAveragePool':
                    if self.ready_for_reducemean == True:
                        if node.input == self.dict_mul['output']:
                            self.dict_rm2['input'] = node.input
                            self.dict_rm2['output'] = node.output
                            self.dict_rm2['id'] = node_id
                            self.dict_rm2['name'] = node.name

                            if node.op_type == 'ReduceMean':
                                attributes = node.attribute
                                for attr in attributes:
                                    if attr.name == 'axes':
                                        self.rm2_axes = attr.ints
                                        logger.debug('self.rm2_axes: {}'.format(self.rm2_axes))
                                        if len(self.rm2_axes) != 1 or self.rm2_axes != self.rm1_axes:
                                            logger.debug('--This ReduceMean IsNot we are looking for...')
                                            self.clear()
                                        else:
                                            self.got_rm_axes = True    
                        else: 
                            logger.debug('self.clear ReduceMean Sub Pow')
                            logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                            logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                            logger.debug('self.dict_mul: {}'.format(self.dict_mul))
                            self.clear()    
                    else: 
                        self.dict_rm['input'] = node.input
                        self.dict_rm['output'] = node.output
                        self.dict_rm['id'] = node_id
                        self.dict_rm['name'] = node.name

                        if node.op_type == 'ReduceMean':
                            attributes = node.attribute
                            for attr in attributes:
                                if attr.name == 'axes':
                                    self.rm1_axes = attr.ints
                                    logger.debug('self.rm1_axes: {}'.format(self.rm1_axes))
                                    if len(self.rm1_axes) != 1:
                                        logger.debug('This ReduceMean IsNot we are looking for...')
                                        self.clear()

                                    break

                if node.op_type == 'Sub':
                    if self.ready_for_sub == True:
                        if node.input[1] == self.dict_mul3['output'][0]:
                            self.dict_sub2['input'] = node.input
                            self.dict_sub2['output'] = node.output
                            self.dict_sub2['id'] = node_id
                            self.dict_sub2['name'] = node.name
                            self.ready2 = True
                            logger.debug('got eighth pair: {} {}'.format(self.dict_sub2['input'], self.dict_sub2['output']))
                        else:
                            logger.debug('---self.clear ReduceMean Sub Pow ReduceMean2 Add Sqrt')
                            logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                            logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                            logger.debug('self.dict_mul: {}'.format(self.dict_mul))
                            logger.debug('self.dict_rm2: {}'.format(self.dict_rm2))
                            logger.debug('self.dict_add: {}'.format(self.dict_add))
                            logger.debug('self.dict_sqrt: {}'.format(self.dict_sqrt))
                            self.clear()
                    else:    
                        if self.dict_rm and node.input[0] == self.dict_rm['input'][0] and node.input[1] == self.dict_rm['output'][0]:
                            self.dict_sub['input'] = node.input
                            self.dict_sub['output'] = node.output
                            self.dict_sub['id'] = node_id
                            self.dict_sub['name'] = node.name
                            logger.debug('got first pair: {} {}'.format(self.dict_sub['input'], self.dict_sub['output']))
                        else:
                            logger.debug('---self.clear ReduceMean, self.dict_rm: {} {}'.format(self.dict_rm, node.name))
                            self.clear()  

                if node.op_type == 'Mul':
                    if self.ready_for_mul_first == True and self.ready_for_mul_second == True:
                        if  node.input[0] == self.dict_rm['output'][0] and node.input[1] == self.dict_mul2['output'][0]:
                            self.dict_mul3['input'] = node.input
                            self.dict_mul3['output'] = node.output
                            self.dict_mul3['id'] = node_id
                            self.dict_mul3['name'] = node.name
                            self.ready_for_sub = True
                            logger.debug('got seventh pair: {} {}'.format(self.dict_mul3['input'], self.dict_mul3['output']))
                        elif  node.input[0] == self.dict_rm['input'][0] and node.input[1] == self.dict_mul2['output'][0]:
                            self.dict_mul4['input'] = node.input
                            self.dict_mul4['output'] = node.output
                            self.dict_mul4['id'] = node_id
                            self.dict_mul4['name'] = node.name
                            self.ready_for_mul_third = True
                            logger.debug('got nineth pair: {} {}'.format(self.dict_mul4['input'], self.dict_mul4['output']))    
                        else:
                            logger.debug('---self.clear ReduceMean and Sub')
                            logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                            logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                            self.clear()  
                    elif self.ready_for_mul_first == True and self.ready_for_mul_second == False and self.ready_for_mul_third == False:
                        if  node.input[0] == self.dict_div['output'][0]:
                            self.dict_mul2['input'] = node.input
                            self.dict_mul2['output'] = node.output
                            self.dict_mul2['id'] = node_id
                            self.dict_mul2['name'] = node.name
                            self.ready_for_mul_second = True
                            logger.debug('got sixth pair: {} {}'.format(self.dict_mul2['input'], self.dict_mul2['output']))
                        else:
                            logger.debug('---self.clear ReduceMean and Sub')
                            logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                            logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                            self.clear() 
                    else:    
                        if self.dict_rm and self.dict_sub and node.input[0] == self.dict_sub['output'][0]  and node.input[1] == self.dict_sub['output'][0]:
                            self.ready_for_reducemean = True
                            self.dict_mul['input'] = node.input
                            self.dict_mul['output'] = node.output
                            self.dict_mul['name'] = node.name
                            self.dict_mul['id'] = node_id

                            logger.debug('got second pair: {} {}'.format(self.dict_mul['input'], self.dict_mul['output']))   
                        else:
                            logger.debug('self.clear ReduceMean and Sub')
                            logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                            logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                            self.clear()

                if node.op_type == 'Add':
                    if self.ready2 == True:
                        if node.input[0] == self.dict_mul4['output'][0] and node.input[1] == self.dict_sub2['output'][0]:
                            self.dict_add2['input'] = node.input
                            self.dict_add2['output'] = node.output
                            self.dict_add2['name'] = node.name
                            self.dict_add2['id'] = node_id

                            self.search = True
                            self.got_ln = True
                            logger.debug('Got a LayerNorm op')
                            ###
                            rm_node = self.model.graph.node[self.dict_rm['id']]
                            sub_node = self.model.graph.node[self.dict_sub['id']]
                            sub2_node = self.model.graph.node[self.dict_sub2['id']]
                            mul_node = self.model.graph.node[self.dict_mul['id']]
                            mul2_node = self.model.graph.node[self.dict_mul2['id']]
                            mul3_node = self.model.graph.node[self.dict_mul3['id']]
                            mul4_node = self.model.graph.node[self.dict_mul4['id']]
                            rm2_node = self.model.graph.node[self.dict_rm2['id']]
                            add_node = self.model.graph.node[self.dict_add['id']]
                            sqrt_node = self.model.graph.node[self.dict_sqrt['id']]
                            div_node = self.model.graph.node[self.dict_div['id']]
                            add2_node = self.model.graph.node[self.dict_add2['id']]

                            self.model.graph.node.remove(rm_node)

                            if self.got_rm_axes == True:
                                axis_ = self.rm1_axes[0]
                            else:
                                axis_ = -1

                            logger.debug('get axis_ = {}'.format(axis_))

                            scale_name = self.dict_mul2['input'][1]
                            beta_name = self.dict_sub2['input'][0]

                            found_init_scale = False
                            found_init_beta = False
                            for init in self.model.graph.initializer:
                                if scale_name == init.name:
                                    found_init_scale = True
                                if beta_name == init.name:
                                    found_init_beta = True

                            if found_init_scale == False:
                                scale_name = self.dict_mul['input'][0]    

                            if found_init_beta == False:
                                beta_name = self.dict_add2['input'][1]

                            '''
                            for init in self.model.graph.initializer:
                                if init.name == self.dict_mul2['input'][1]:
                                    if init.data_type == onnx_proto.TensorProto.FLOAT16:
                                        scale_name = scale_name + '_to_fp32__'
                                        t = make_fp32_tensor_from_fp16(init, scale_name)
                                        self.model.graph.initializer.append(t)
                                        if init not in self.unused_init_list:
                                            self.unused_init_list.append(init) 

                                elif init.name == self.dict_sub2['input'][0]:
                                    if init.data_type == onnx_proto.TensorProto.FLOAT16:
                                        beta_name = beta_name + '_to_fp32__'
                                        t = make_fp32_tensor_from_fp16(init, beta_name)
                                        self.model.graph.initializer.append(t)

                                        if init not in self.unused_init_list:
                                            self.unused_init_list.append(init)               
                            '''
                            
                            ln_node_name = self.dict_rm['name'] + '+' + self.dict_sub['name'] + '+' + self.dict_sub2['name'] + '+' + self.dict_mul['name'] \
                                            + self.dict_mul2['name'] + '+' + self.dict_mul3['name'] + '+' + self.dict_mul4['name'] + '+' + self.dict_rm2['name'] + '+' + self.dict_add['name'] \
                                            + self.dict_sqrt['name'] + '+' + self.dict_div['name'] + '+' + self.dict_add2['name'] 
                                             
                            ln_node = onnx.helper.make_node(
                                                    name = ln_node_name,
                                                    #name = node.name + '_to_layernorm_' + str(self.loop),
                                                    op_type='LayerNorm',
                                                    inputs=[self.dict_rm['input'][0], scale_name, beta_name],
                                                    outputs=self.dict_add2['output'],
                                                    axis=axis_,
                                                    eps=1e-5,
                                                    elementwise_affine=1,
                                                    domain='com.metax-tech'
                                                    )

                            self.model.graph.node.insert(self.dict_rm['id'], ln_node)

                            self.model.graph.node.remove(sub_node)
                            self.model.graph.node.remove(sub2_node)
                            self.model.graph.node.remove(rm2_node)
                            self.model.graph.node.remove(add_node)
                            self.model.graph.node.remove(sqrt_node)
                            self.model.graph.node.remove(div_node)
                            self.model.graph.node.remove(mul_node)
                            self.model.graph.node.remove(mul2_node)
                            self.model.graph.node.remove(mul3_node)
                            self.model.graph.node.remove(mul4_node)
                            self.model.graph.node.remove(add2_node)

                            break
                        else:
                            logger.debug('--self.clear ReduceMean Sub Pow ReduceMean2')
                            logger.debug('--self.dict_rm: {}'.format(self.dict_rm))
                            logger.debug('--self.dict_sub: {}'.format(self.dict_sub))
                            logger.debug('--self.dict_pow: {}'.format(self.dict_pow))
                            logger.debug('--self.dict_rm2: {}'.format(self.dict_rm2))
                            self.clear()
                    else:
                        if self.dict_rm and self.dict_sub and self.dict_mul and self.dict_rm2 and node.input[0] == self.dict_rm2['output'][0]:
                            self.dict_add['input'] = node.input
                            self.dict_add['output'] = node.output
                            self.dict_add['id'] = node_id
                            self.dict_add['name'] = node.name
                            logger.debug('got third pair: {} {}'.format(self.dict_add['input'], self.dict_add['output']))
                        else:
                            logger.debug('self.clear ReduceMean Sub Mul ReduceMean2')
                            logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                            logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                            logger.debug('self.dict_mul: {}'.format(self.dict_mul))
                            logger.debug('self.dict_rm2: {}'.format(self.dict_rm2))
                            self.clear()

                if node.op_type == 'Sqrt':
                    if self.dict_rm and self.dict_sub and self.dict_mul and self.dict_rm2 and  \
                            self.dict_add and node.input == self.dict_add['output']:
                        self.dict_sqrt['input'] = node.input
                        self.dict_sqrt['output'] = node.output
                        self.dict_sqrt['id'] = node_id
                        self.dict_sqrt['name'] = node.name
                        logger.debug('got forth pair: {} {}'.format(self.dict_sqrt['input'], self.dict_sqrt['output']))
                    else:
                        logger.debug('self.clear ReduceMean Sub Pow ReduceMean2 Add')
                        logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                        logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                        logger.debug('self.dict_mul: {}'.format(self.dict_mul))
                        logger.debug('self.dict_rm2: {}'.format(self.dict_rm2))
                        logger.debug('self.dict_add: {}'.format(self.dict_add))
                        self.clear()

                if node.op_type == 'Reciprocal':
                    if self.dict_rm and self.dict_sub and self.dict_mul and self.dict_rm2 and  \
                            self.dict_add and self.dict_sqrt and node.input[0] == self.dict_sqrt['output'][0]:
                        self.dict_div['input'] = node.input
                        self.dict_div['output'] = node.output
                        self.dict_div['id'] = node_id
                        self.dict_div['name'] = node.name
                        self.ready_for_mul_first = True
                        logger.debug('got fifth pair: {} {}'.format(self.dict_div['input'], self.dict_div['output']))
                    else:
                        logger.debug('----self.clear ReduceMean Sub Pow ReduceMean2 Add Sqrt')
                        logger.debug('self.dict_rm: {}'.format(self.dict_rm))
                        logger.debug('self.dict_sub: {}'.format(self.dict_sub))
                        logger.debug('self.dict_mul: {}'.format(self.dict_mul))
                        logger.debug('self.dict_rm2: {}'.format(self.dict_rm2))
                        logger.debug('self.dict_add: {}'.format(self.dict_add))
                        logger.debug('self.dict_sqrt: {}'.format(self.dict_sqrt))
                        self.clear()
                    
        if self.got_ln == True:
            op_set = self.model.opset_import.add()
            op_set.domain = 'com.metax-tech'
            op_set.version = 1
            
            #onnx.save(model, export_onnx)

        remove_unused_initializer(self.model, self.unused_init_list)

        return self.model

# rms layernorm

class MergeRMSLn():
    def __init__(self, model):
        logger.debug('MergeRMSLn Init--------------------------')
        self.model = model
        self.got_ln = False
        self.search = True
        self.unused_init_list = []
        self.loop = 0
        self.ready = False

        self.dict_rm = {}
        self.dict_pow = {}
        self.dict_add = {}
        self.dict_sqrt = {}
        self.dict_div = {}
        self.dict_mul = {}
        self.dict_mul2 = {}

        self.input_type = onnx.TensorProto.FLOAT
        self.input_shape = [1]
        self.scale = [1]

        self.rm1_axes = [-1]

    def clear(self):
        self.dict_rm = {}
        self.dict_pow = {}
        self.dict_add = {}
        self.dict_sqrt = {}
        self.dict_div = {}
        self.dict_mul = {}
        self.dict_mul2 = {}

        self.rm1_axes = [-1]
        self.search = False
        self.ready = False

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

                    for vi in self.model.graph.value_info:
                        if vi.name == node.input[0]:
                            self.input_type = vi.type.tensor_type.elem_type
                            self.input_shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                            print('got input shape: ', self.input_shape)
                            print('got input type: ', self.input_type)

                if node.op_type == 'ReduceMean':
                    if self.dict_pow and node.input == self.dict_pow['output']:
                        self.dict_rm['input'] = node.input
                        self.dict_rm['output'] = node.output
                        self.dict_rm['id'] = node_id 
                        print('got first pair:', self.dict_rm['input'], self.dict_rm['output'])

                        attributes = node.attribute
                        for attr in attributes:
                            if attr.name == 'axes':
                                self.rm1_axes = attr.ints
                                print('self.rm1_axes: ', self.rm1_axes)
                                if len(self.rm1_axes) != 1:
                                    print('This ReduceMean IsNot we are looking for...')
                                    self.clear()

                                break
                    else:
                        self.clear()       

                if node.op_type == 'Add':
                    if self.dict_pow and self.dict_rm and node.input[0] == self.dict_rm['output'][0]:
                        self.dict_add['input'] = node.input
                        self.dict_add['output'] = node.output
                        self.dict_add['id'] = node_id
                        print('got second pair:', self.dict_rm['input'], self.dict_rm['output'])
                    else:
                        print('self.clear ReduceMean Sub Pow ReduceMean2')
                        print('self.dict_rm:', self.dict_rm)
                        print('self.dict_pow:', self.dict_pow)
                        self.clear()

                if node.op_type == 'Sqrt':
                    if self.dict_pow and self.dict_rm and  \
                            self.dict_add and node.input == self.dict_add['output']:
                        self.dict_sqrt['input'] = node.input
                        self.dict_sqrt['output'] = node.output
                        self.dict_sqrt['id'] = node_id
                        print('got third pair:', self.dict_sqrt['input'], self.dict_sqrt['output'])
                    else:
                        print('self.clear ReduceMean Sub Pow ReduceMean2 Add')
                        print('self.dict_rm:', self.dict_rm)
                        print('self.dict_pow:', self.dict_pow)
                        print('self.dict_add:', self.dict_add)
                        self.clear()

                if node.op_type == 'Div':
                    self.scale = values.get_init_value(self.model, node.input[0])
                    if self.scale and self.dict_pow and self.dict_rm and  \
                            self.dict_add and self.dict_sqrt and node.input[1] == self.dict_sqrt['output'][0] :
                        self.dict_div['input'] = node.input
                        self.dict_div['output'] = node.output
                        self.dict_div['id'] = node_id
                        if isinstance(self.scale, np.ndarray):
                            self.scale = self.scale.tolist()
                        print('got fourth pair, scale', self.scale, type(self.scale))
                    else:
                        print('-self.clear ReduceMean Sub Pow ReduceMean2 Add Sqrt')
                        print('self.dict_rm:', self.dict_rm)
                        print('self.dict_pow:', self.dict_pow)
                        print('self.dict_add:', self.dict_add)
                        print('self.dict_sqrt:', self.dict_sqrt)
                        self.clear()

                if node.op_type == 'Mul':
                    if self.ready == True and node.input[0] == self.dict_mul['output'][0]:
                        self.dict_mul2['input'] = node.input
                        self.dict_mul2['output'] = node.output
                        self.dict_mul2['id'] = node_id

                        self.search = True
                        self.got_ln = True
                        print('Got a LayerNorm op')
                        ###
                        pow_node = self.model.graph.node[self.dict_pow['id']]
                        rm_node = self.model.graph.node[self.dict_rm['id']]
                        add_node = self.model.graph.node[self.dict_add['id']]
                        sqrt_node = self.model.graph.node[self.dict_sqrt['id']]
                        div_node = self.model.graph.node[self.dict_div['id']]
                        mul_node = self.model.graph.node[self.dict_mul['id']]
                        mul2_node = self.model.graph.node[self.dict_mul2['id']]

                        self.model.graph.node.remove(pow_node)

                        scale_name = self.dict_mul2['input'][1] #node.name + '_scale_' + str(self.loop)
                        beta_name = node.name + '_beta_' + str(self.loop)

                        '''
                        scale_tensor = onnx.helper.make_tensor(name=scale_name,
                                                        data_type=self.input_type,
                                                        dims=[self.input_shape[-1]],
                                                        vals=self.scale*self.input_shape[-1])   

                        self.model.graph.initializer.append(scale_tensor)
                        '''   

                        beta_tensor = onnx.helper.make_tensor(name=beta_name,
                                data_type=self.input_type,
                                dims=[self.input_shape[-1]],
                                vals=[0]*self.input_shape[-1])   

                        self.model.graph.initializer.append(beta_tensor)                

                        ln_node = onnx.helper.make_node(
                                                name = node.name + '_to_layernorm_' + str(self.loop),
                                                op_type='LayerNorm',
                                                inputs=[self.dict_pow['input'][0], scale_name, beta_name],
                                                outputs=self.dict_mul2['output'],
                                                axis=self.rm1_axes[0],
                                                eps=1e-5,
                                                elementwise_affine=1,
                                                domain='com.metax-tech'
                                                )

                        self.model.graph.node.insert(self.dict_pow['id'], ln_node)

                        self.model.graph.node.remove(rm_node)
                        self.model.graph.node.remove(add_node)
                        self.model.graph.node.remove(sqrt_node)
                        self.model.graph.node.remove(div_node)
                        self.model.graph.node.remove(mul_node)
                        self.model.graph.node.remove(mul2_node)

                        break
                    elif self.dict_pow and self.dict_rm and  \
                            self.dict_add and self.dict_sqrt and self.dict_div and node.input[1] == self.dict_div['output'][0] \
                            and node.input[0] == self.dict_pow['input'][0] :
                        self.dict_mul['input'] = node.input
                        self.dict_mul['output'] = node.output
                        self.dict_mul['id'] = node_id
                        self.ready = True

                        print('got fifth pair:', self.dict_mul['input'], self.dict_mul['output'])
                    else:
                        print('--self.clear ReduceMean Sub Pow ReduceMean2 Add Sqrt Div')
                        print('self.dict_rm:', self.dict_rm)
                        print('self.dict_pow:', self.dict_pow)
                        print('self.dict_add:', self.dict_add)
                        print('self.dict_sqrt:', self.dict_sqrt)
                        print('self.dict_mul:', self.dict_mul)
                        self.clear()                           

        if self.got_ln == True:
            op_set = self.model.opset_import.add()
            op_set.domain = 'com.metax-tech'
            op_set.version = 1
            
            #onnx.save(model, export_onnx)

        remove_unused_initializer(self.model, self.unused_init_list)
        
        return self.model

def merge_layernorm(model):
    mlp1 = MergeLNPattern1(model)
    model = mlp1.merge()

    mlp2 = MergeLNPattern2(model)
    model = mlp2.merge()

    #mlp3 = MergeRMSLn(model)
    #model = mlp3.merge()

    remove_invalid_tensor(model)

    return model

'''
model = onnx.load('./bert_fp16_test.onnx')
#model = onnx.load('./bert_sub.onnx')
m = merge_layernorm(model)
onnx.save(m, 'ln.onnx') 
'''  
  
