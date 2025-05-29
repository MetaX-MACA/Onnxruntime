import onnx
import correct_batch
import numpy as np
import values
import log
import copy
from operation import get_all_next_node_by_output, del_node_by_nodeName

logger = log.getLogger(__name__, log.INFO)

def get_constant_value(model, name):
    shape = []
    for n in model.graph.node:
        if name == n.output[0]:
            attributes = n.attribute
            for attr in attributes:
                if attr.name == 'value':
                    v = values.get_tensor_value(attr.t)
                    dims = len(v)
                    logger.debug('get_constant_value: {} {}'.format(v, dims))
                    shape = v
                    break
            break

    return shape                     

def fuse_pad_to_pool(model):
    dict_pad = {}
    dict_pool = {}
    dict_mul = {}

    got_pad_pool = False

    search = True

    pads = []

    while search == True:
        search = False

        for node_id, node in enumerate(model.graph.node):
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #         ", op:", node.op_type, ', len(input):', len(node.input))
            if node.op_type == 'Pad':
                dict_pad['input'] = node.input
                dict_pad['output'] = node.output
                dict_pad['id'] = node_id

                if len(node.input) == 1:
                    attributes = node.attribute
                    for attr in attributes:
                        if attr.name == 'pads':
                            pads = attr.ints
                            #print('fuse pads:', pads)
                            break

            #print('got pads:', pads, node.op_type)

            if node.op_type == 'MaxPool' or node.op_type == 'AveragePool':
                if len(dict_pad) > 0 and node.input == dict_pad['output']:
                    dict_pool['input'] = node.input
                    dict_pool['output'] = node.output
                    dict_pool['id'] = node_id
                    logger.debug('got pad+pool pair, pad: {} {}'.format(dict_pad['input'], dict_pad['output']))
                    logger.debug('got pad+pool pair, pool: {} {}'.format(dict_pool['input'], dict_pool['output']))
                    #pads = []

                    got_pad_pool = True

                    if len(pads) == 0:
                        for init in model.graph.initializer:
                            if init.name == dict_pad['input'][1]:
                                logger.debug('got init(pads): {}'.format(init.name))
                                dtype = init.data_type
                                np_dtype = correct_batch.convert_ort_type_2_np(dtype)
                                if init.raw_data:
                                    params_list = np.fromstring(init.raw_data, dtype=np_dtype)
                                    for p in params_list:
                                        pads.append(p)
                                else:
                                    data_list = correct_batch.get_data_list(dtype, init)
                                    for p in data_list:
                                        pads.append(p)

                                break        
                            #elif init.name == dict_pad['input'][2]:
                            #    print('got init(constane_value):', init.name)  

                        if pads == []:
                            pads = get_constant_value(model, dict_pad['input'][1])

                    logger.debug('got pads: {}'.format(pads))

                    if len(pads) != 8:
                        logger.debug('skip pad+pool~~~~')
                        dict_pad = {}
                        dict_pool = {}
                        continue

                    pads_real = [pads[2], pads[3], pads[6], pads[7]]
                
                    found_pads_attr = False
                    for attr in node.attribute:
                        if attr.name == 'pads':
                            del attr.ints[:]
                            attr.ints.extend(pads_real)
                            found_pads_attr = True
                            #print('pads:---', attr.ints)
                            break

                    if found_pads_attr == False:
                        attr = onnx.helper.make_attribute('pads', pads_real)
                        node.attribute.append(attr) 

                    if node.op_type == 'AveragePool':
                        found_cip_attr = False
                        for attr in node.attribute:
                            if attr.name == 'count_include_pad':
                                found_cip_attr = True
                                attr.i = 1
                                break
                        
                        if found_cip_attr == False:
                            attr = onnx.helper.make_attribute('count_include_pad', 1)
                            node.attribute.append(attr)           
        
                    node.input[0] = dict_pad['input'][0]

                    old_node = model.graph.node[dict_pad['id']] 
                    model.graph.node.remove(old_node)

                    dict_pad = {}
                    dict_pool = {}
                    pads = []

                    search = True
                    break
                else:
                    #print('clear pad dict')
                    dict_pad = {}
                    pads = []    

    if got_pad_pool == True:
        logger.debug('got pad+pool node------------')

    return model    


def fuse_transpose_relu_transpose(model):
    activate_func = ['LeakyRelu', 'Relu', 'PRelu', 'Swish', 'HardSwish', 'Sigmoid', 'HardSigmoid', 'Clip']
    init_dim_list = [int(i) for i in range(10)]
    for node in model.graph.node:
        if node.op_type == 'Transpose':
            first_transpose_perm = list(node.attribute[0].ints)
            init_dim_list_first = copy.deepcopy(init_dim_list)
            for idx in first_transpose_perm:
                init_dim_list_first[idx] = init_dim_list[idx]
            next_nodes, _ = get_all_next_node_by_output(model, node.output[0])
            if len(next_nodes) > 1 or len(next_nodes) == 0:
                continue
            next_next_nodes = None
            if next_nodes[0].op_type in activate_func:
                next_next_nodes, _ = get_all_next_node_by_output(model, next_nodes[0].output[0])
                if len(next_next_nodes) > 1 or len(next_next_nodes) == 0:
                    next_next_nodes = None
                    continue
            elif next_nodes[0].op_type == 'Transpose':
                next_next_nodes = next_nodes

            if next_next_nodes != None and next_next_nodes[0].op_type == 'Transpose':
                second_transpose_perm = list(next_next_nodes[0].attribute[0].ints)
                init_dim_list_second = copy.deepcopy(init_dim_list)
                for idx in second_transpose_perm:
                    init_dim_list_second[idx] = init_dim_list_first[idx]
                if init_dim_list_second == init_dim_list:
                    del_node_by_nodeName(model, node.name)
                    del_node_by_nodeName(model, next_next_nodes[0].name)
                    continue
    return model
