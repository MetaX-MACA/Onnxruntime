import sys

import log
import numpy as np
import onnx
import values
from fuse_mha import get_prev_node_by_input
from utils import find_initializers_input, convert_any_to_python_primary_type

import operation

logger = log.getLogger(__name__, log.INFO)

def merge_swish_old(model):
    dict_sm = {}
    dict_mul = {}

    got_swish = False
    search = True

    index = 0

    while search == True:
        search = False

        for node_id, node in enumerate(model.graph.node):
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #         ", op:", node.op_type, ', len(input):', len(node.input))

            if node.op_type == 'Sigmoid':
                dict_sm['input'] = node.input
                dict_sm['output'] = node.output
                dict_sm['id'] = node_id
                dict_sm['name'] = node.name

            '''
            if node.name == 'Sigmoid_7':
                next_node, _ = operation.get_next_node_by_output(model, node.output[0])
                print('node.name:{}, next_node.name:{}'.format(node.name, next_node.name))
            '''

            if node.op_type == 'Mul':
                if node.name == 'Mul_8':
                    print('dict_sm:', dict_sm)

                if len(dict_sm) > 0 and node.input[0] == dict_sm['input'][0] and node.input[1] == dict_sm['output'][0]:
                    dict_mul['input'] = node.input
                    dict_mul['output'] = node.output
                    dict_mul['id'] = node_id
                    dict_mul['name'] = node.name

                    logger.debug('got swish pair: {} {}'.format(dict_sm['input'], dict_sm['output']))
                    logger.debug('got swish pair: {} {}'.format(dict_mul['input'], dict_mul['output']))

                    got_swish = True

                    old_node = model.graph.node[dict_sm['id']]
                    model.graph.node.remove(old_node)

                    prev_node, flag = get_prev_node_by_input(model, dict_sm['input'][0])
                    if flag == -1:
                        #print('skip:', node.name)
                        continue

                    #swish_node_name = prev_node.name + '_' + 'Swish_' + str(index)
                    swish_node_name = dict_sm['name'] + '+' + dict_mul['name']
                    # swish_node_name = 'Swish_' + str(index)
                    index = index + 1

                    swish_node = onnx.helper.make_node(
                                            name = swish_node_name,
                                            op_type='Swish',
                                            inputs=dict_sm['input'],
                                            outputs=dict_mul['output'],
                                            domain='com.metax-tech',
                                            )

                    model.graph.node.insert(dict_sm['id'], swish_node)

                    old_node = model.graph.node[dict_mul['id']]
                    model.graph.node.remove(old_node)

                    dict_sm = {}
                    dict_mul = {}
                    search = True
                    break
                else:
                    logger.debug('clear Sigmoid and Tanh')
                    dict_sm = {}

    if got_swish == True:
        op_set = model.opset_import.add()
        op_set.domain = 'com.metax-tech'
        op_set.version = 1

        #onnx.save(model, output)

def get_sm_list(model):
    sm_list = []
    for node in model.graph.node:
        if node.op_type == 'Sigmoid':
            mul_node, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == 0 and mul_node.op_type == 'Mul':
                if mul_node.input[0] == node.input[0] and mul_node.input[1] == node.output[0]:
                    sm = {}
                    sm['Sigmoid'] = node
                    sm['Mul'] = mul_node
                    sm_list.append(sm)

    #for sm in sm_list:
    #    print('Sigmoid name:{}'.format(sm['Sigmoid'].name))

    return sm_list

def get_sm_list_v2(model):
    """
            beta---|
                   Mul--- Sigmoid ---|
            |------|                 Mul ----
        ----|------------------------|
    """

    sm_list = []
    for node in model.graph.node:
        if node.op_type == 'Sigmoid':
            per_mul_node, per_flag = operation.get_prev_node_by_input(model, node.input[0])
            if per_flag != 0 and per_mul_node.op_type != 'Mul':
                continue
            input_const = find_initializers_input(model, per_mul_node)

            if len(input_const) != 1:
                continue
            if values.get_init_value(model, input_const[0].name).size != 1:
                continue

            # input_const_indx = per_mul_node.index(input_const[0].name)

            mul_node, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == 0 and mul_node.op_type == 'Mul':
                if mul_node.input[0] == per_mul_node.input[0] and mul_node.input[1] == node.output[0]:
                    sm = {}
                    sm['Per_Mul'] = per_mul_node
                    sm['Sigmoid'] = node
                    sm['Mul'] = mul_node
                    sm_list.append(sm)

    #for sm in sm_list:
    #    print('Sigmoid name:{}'.format(sm['Sigmoid'].name))

    return sm_list


def handle_sm(model, sm):
    sigmiod_node = sm['Sigmoid']
    mul_node = sm['Mul']

    mul_node.op_type = 'Swish'
    del mul_node.input[1]

    mul_node.domain='com.metax-tech'

    operation.remove_onnx_node(model, sigmiod_node)

    if len(sm)==3:
        per_mul_node = sm['Per_Mul']
        beta_value = np.array(values.get_init_value(model, per_mul_node.input[1])).astype(np.float32)
        beta_value = convert_any_to_python_primary_type(beta_value)
        beta_attr = onnx.helper.make_attribute("beta", beta_value)
        mul_node.attribute.append(beta_attr)
        operation.remove_onnx_node(model, per_mul_node)


def merge_swish(model):
    found = False

    sm_list = get_sm_list(model)
    # sm_list_v2 = get_sm_list_v2(model)

    for sm in sm_list:
        found = True
        handle_sm(model, sm)

    if found == True:
        op_set = model.opset_import.add()
        op_set.domain = 'com.metax-tech'
        op_set.version = 1

def merge_hard_swish(model):
    dict_sm = {}
    dict_mul = {}

    got_hard_swish = False
    index = 0

    for node_id, node in enumerate(model.graph.node):
        #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
        #         ", op:", node.op_type, ', len(input):', len(node.input))

        if node.op_type == 'HardSigmoid':
            dict_sm['input'] = node.input
            dict_sm['output'] = node.output
            dict_sm['id'] = node_id
            dict_sm['name'] = node.name

        if node.op_type == 'Mul':
            if len(dict_sm) > 0 and node.input[0] == dict_sm['input'][0] and node.input[1] == dict_sm['output'][0]:
                dict_mul['input'] = node.input
                dict_mul['output'] = node.output
                dict_mul['id'] = node_id
                dict_mul['name'] = node.name

                logger.debug('got hard_swish pair: {} {}'.format(dict_sm['input'], dict_sm['output']))
                logger.debug('got hard_swish pair: {} {}'.format(dict_mul['input'], dict_mul['output']))

                got_hard_swish = True

                old_node = model.graph.node[dict_sm['id']]
                model.graph.node.remove(old_node)

                prev_node, flag = get_prev_node_by_input(model, dict_sm['input'][0])
                if flag == -1:
                    continue

                #swish_node_name = prev_node.name + '_' + 'HardSwish_' + str(index)
                swish_node_name = dict_sm['name'] + '+' + dict_mul['name']

                index = index + 1

                swish_node = onnx.helper.make_node(
                                        name = swish_node_name,
                                        op_type='HardSwish',
                                        inputs=dict_sm['input'],
                                        outputs=dict_mul['output'],
                                        domain='com.metax-tech',
                                        )

                model.graph.node.insert(dict_sm['id'], swish_node)

                old_node = model.graph.node[dict_mul['id']]
                model.graph.node.remove(old_node)

                dict_sm = {}
                dict_mul = {}
            else:
                logger.debug('clear HardSigmoid')
                dict_sm = {}

    if got_hard_swish == True:
        op_set = model.opset_import.add()
        op_set.domain = 'com.metax-tech'
        op_set.version = 1

        #onnx.save(model, output)

def merge_hard_swish2(model):
    dict_add = {}
    dict_clip = {}
    dict_mul = {}
    dict_div = {}

    got_swish = False

    search = True

    index = 0

    while search == True:
        search = False
        for node_id, node in enumerate(model.graph.node):
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #        ", op:", node.op_type, ', len(input):', len(node.input))

            found_add = False
            if node.op_type == 'Add':
                addB = values.get_init_value(model, node.input[1])
                logger.debug('addB: {}'.format(addB))

                if isinstance(addB, list) and addB == []:
                    logger.debug('addB is not in initilizer')
                    continue

                if addB[0] != 3:
                    logger.debug('this is not the add-node which we wanted(value B is not 3)...')
                    continue

                if isinstance(addB, np.ndarray) == True:
                    if addB.shape != (1, ):
                        logger.debug('this is not the add-node which we wanted(shape is wrong)...')
                        continue
                else:
                    if len(addB) != 1:
                        logger.debug('this is not the add-node which we wanted(list len is wrong)...')
                        continue

                dict_add['input'] = node.input
                dict_add['output'] = node.output
                dict_add['id'] = node_id
                dict_add['name'] = node.name

                logger.debug('got match add node: {}'.format(node.name))

            if node.op_type == 'Clip':
                if dict_add and node.input[0] == dict_add['output'][0] and len(node.input) >= 3:
                    clip_min = values.get_init_value(model, node.input[1])
                    logger.debug('clip_min: {}'.format(clip_min))

                    clip_max = values.get_init_value(model, node.input[2])
                    logger.debug('clip_max: {}'.format(clip_max))

                    if (isinstance(clip_min, list) and clip_min == []) or (isinstance(clip_max, list) and clip_max == []):
                        logger.debug('clip_min or clip_max is not in initilizer')
                        continue

                    if clip_min[0] != 0:
                        logger.debug('this is not the clip-node which we wanted(min is not 0)...')
                        dict_add = {}
                        continue

                    if isinstance(clip_min, np.ndarray) == True:
                        if clip_min.shape != (1, ):
                            logger.debug('this is not the clip-node which we wanted(shape is wrong)...')
                            dict_add = {}
                            continue
                    else:
                        if len(clip_min) != 1:
                            logger.debug('this is not the clip-node which we wanted(list len is wrong)...')
                            dict_add = {}
                            continue

                    if clip_max[0] != 6:
                        logger.debug('this is not the clip-node which we wanted(max is not 6)...')
                        continue

                    if isinstance(clip_max, np.ndarray) == True:
                        if clip_max.shape != (1, ):
                            logger.debug('this is not the clip-node which we wanted(shape is wrong)...')
                            dict_add = {}
                            continue
                    else:
                        if len(clip_max) != 1:
                            logger.debug('this is not the clip-node which we wanted(list len is wrong)...')
                            dict_add = {}
                            continue

                    dict_clip['input'] = node.input
                    dict_clip['output'] = node.output
                    dict_clip['id'] = node_id
                    dict_clip['name'] = node.name

                    logger.debug('got first pair: {} {}'.format(dict_clip['input'], dict_clip['output']))
                else:
                    logger.debug('clear dict_add: {}'.format(dict_add))
                    dict_add = {}

            if node.op_type == 'Mul':
                if dict_add and dict_clip and node.input[1] == dict_clip['output'][0] and node.input[0] == dict_add['input'][0]:
                    dict_mul['input'] = node.input
                    dict_mul['output'] = node.output
                    dict_mul['id'] = node_id
                    dict_mul['name'] = node.name

                    logger.debug('got second pair: {} {}'.format(dict_mul['input'], dict_mul['output']))

                else:
                    logger.debug('clear dict_add and dict_clip')
                    logger.debug('dict_add: {}'.format(dict_add))
                    logger.debug('dict_clip: {}'.format(dict_clip))
                    dict_add = {}
                    dict_clip = {}

            if node.op_type == 'Div':
                if dict_mul and node.input[0] == dict_mul['output'][0]:
                    dict_div['input'] = node.input
                    dict_div['output'] = node.output
                    dict_div['id'] = node_id
                    dict_div['name'] = node.name

                    divB = values.get_init_value(model, node.input[1])
                    logger.debug('divB: {}'.format(divB))

                    if divB[0] != 6:
                        logger.debug('this is not the div-node which we wanted(value B is not 6)...')
                        continue

                    if isinstance(divB, np.ndarray) == True:
                        if divB.shape != (1, ):
                            logger.debug('this is not the div-node which we wanted(shape is wrong)...')
                            continue
                    else:
                        if len(divB) != 1:
                            logger.debug('this is not the div-node which we wanted(list len is wrong)...')
                            continue

                    ###################################
                    old_node = model.graph.node[dict_add['id']]
                    model.graph.node.remove(old_node)

                    prev_node, flag = get_prev_node_by_input(model, dict_add['input'][0])
                    if flag == -1:
                        continue

                    #swish_node_name = prev_node.name + '_' + 'HardSwish__' + str(index)
                    swish_node_name = dict_add['name'] + '+' + dict_clip['name'] + '+' + dict_mul['name'] + '+' + dict_div['name']
                    index = index + 1

                    swish_node = onnx.helper.make_node(
                                            name = swish_node_name,
                                            op_type='HardSwish',
                                            inputs=[dict_add['input'][0]],
                                            outputs=dict_div['output'],
                                            domain='com.metax-tech'
                                            )

                    model.graph.node.insert(dict_add['id'], swish_node)

                    old_node = model.graph.node[dict_div['id']]
                    model.graph.node.remove(old_node)

                    old_node = model.graph.node[dict_mul['id']]
                    model.graph.node.remove(old_node)

                    old_node = model.graph.node[dict_clip['id']]
                    model.graph.node.remove(old_node)

                    dict_add = {}
                    dict_clip = {}
                    dict_mul = {}
                    ###############################

                    got_swish = True
                    search = True
                    break
                else:
                    logger.debug('clear dict_add and dict_clip')
                    logger.debug('dict_add: {}'.format(dict_add))
                    logger.debug('dict_clip: {}'.format(dict_clip))
                    logger.debug('dict_mul: {}'.format(dict_mul))
                    dict_add = {}
                    dict_clip = {}
                    dict_mul = {}

    if got_swish == True:
        op_set = model.opset_import.add()
        op_set.domain = 'com.metax-tech'
        op_set.version = 1

    return model

def merge_swish_and_hard_swish(model):
    merge_swish(model)

    merge_hard_swish(model)
    merge_hard_swish2(model)

    return model
