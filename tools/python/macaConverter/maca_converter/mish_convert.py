import argparse
import sys

import log
import onnx
from fuse_mha import get_prev_node_by_input

logger = log.getLogger(__name__, log.INFO)

def merge_mish_old(onnxfile, export_onnx):
    model = onnx.load(onnxfile)

    dict_sp = {}
    dict_tanh = {}
    dict_mul = {}
    dict_mish_next = {}
    mish_next_list = []

    got_mish = False

    index = 0

    for node_id, node in enumerate(model.graph.node):
        #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
        #         ", op:", node.op_type, ', len(input):', len(node.input))

        if got_mish == True:
            if dict_mul['output'][0] in node.input:
                logger.debug('got next node----')
                dict_mish_next['input'] = node.input
                dict_mish_next['output'] = node.output
                dict_mish_next['id'] = node_id

                mish_next_list.append(dict_mish_next)

            continue

        if node.op_type == 'Softplus':
            dict_sp['input'] = node.input
            dict_sp['output'] = node.output
            dict_sp['id'] = node_id

        if node.op_type == 'Tanh':
            if dict_sp and node.input == dict_sp['output']:
                dict_tanh['input'] = node.input
                dict_tanh['output'] = node.output
                dict_tanh['id'] = node_id
                logger.debug('got first pair: {} {}'.format(dict_tanh['input'], dict_tanh['output']))
            else:
                logger.debug('clear Softplus')
                dict_sp = {}

        if node.op_type == 'Mul':
            if dict_sp and dict_tanh and node.input[1] == dict_tanh['output'][0] and node.input[0] == dict_sp['input'][0]:
                logger.debug('got second pair: {}'.format(dict_tanh['output']))
                dict_mul['input'] = node.input
                dict_mul['output'] = node.output
                dict_mul['id'] = node_id

                got_mish = True

    if got_mish == True:
        old_node = model.graph.node[dict_sp['id']]
        model.graph.node.remove(old_node)

        prev_node, flag = get_prev_node_by_input(model, dict_sm['input'])
        start_suffix = prev_node.name + "_" if flag==0 else ""
        mish_node_name = start_suffix +'Mish_' + str(index)
        index = index + 1

        mish_node = onnx.helper.make_node(
                                name = mish_node_name,
                                op_type='Mish',
                                inputs=dict_sp['input'],
                                outputs=dict_mul['output'],
                                )

        model.graph.node.insert(dict_sp['id'], mish_node)

        #for node in mish_next_list:
            #next_node = model.graph.node[node['id']]
            #next_node.input[0] = dict_sp['output'][0]

        old_node = model.graph.node[dict_mul['id']]
        model.graph.node.remove(old_node)

        old_node = model.graph.node[dict_tanh['id']]
        model.graph.node.remove(old_node)

        onnx.save(model, export_onnx)

    return got_mish

def merge_mish(model):
    dict_sp = {}
    dict_tanh = {}
    dict_mul = {}

    got_mish = False

    search = True

    while search == True:
        search = False
        for node_id, node in enumerate(model.graph.node):
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #        ", op:", node.op_type, ', len(input):', len(node.input))

            if node.op_type == 'Softplus':
                dict_sp['input'] = node.input
                dict_sp['output'] = node.output
                dict_sp['name'] = node.name
                dict_sp['id'] = node_id

            if node.op_type == 'Tanh':
                if dict_sp and node.input == dict_sp['output']:
                    dict_tanh['input'] = node.input
                    dict_tanh['output'] = node.output
                    dict_tanh['name'] = node.name
                    dict_tanh['id'] = node_id
                    logger.debug('got first pair: {} {}'.format(dict_tanh['input'], dict_tanh['output']))
                else:
                    logger.debug('clear Softplus, dict_sp: {}'.format(dict_sp))
                    dict_sp = {}

            if node.op_type == 'Mul':
                if dict_sp and dict_tanh and node.input[1] == dict_tanh['output'][0] and node.input[0] == dict_sp['input'][0]:
                    dict_mul['input'] = node.input
                    dict_mul['output'] = node.output
                    dict_mul['name'] = node.name
                    dict_mul['id'] = node_id

                    logger.debug('got second pair: {} {}'.format(dict_mul['input'], dict_mul['output']))

                    got_mish = True

                    old_node = model.graph.node[dict_sp['id']]
                    model.graph.node.remove(old_node)

                    mish_node = onnx.helper.make_node(
                                            name = dict_sp['name'] + '+' + dict_tanh['name'] + '+' + dict_mul['name'],
                                            op_type='Mish',
                                            inputs=dict_sp['input'],
                                            outputs=dict_mul['output'],
                                            domain='com.metax-tech'
                                            )

                    model.graph.node.insert(dict_sp['id'], mish_node)

                    #for node in mish_next_list:
                        #next_node = model.graph.node[node['id']]
                        #next_node.input[0] = dict_sp['output'][0]

                    old_node = model.graph.node[dict_mul['id']]
                    model.graph.node.remove(old_node)

                    old_node = model.graph.node[dict_tanh['id']]
                    model.graph.node.remove(old_node)

                    dict_sp = {}
                    dict_tanh = {}
                    dict_mul = {}

                    search = True
                    break
                else:
                    logger.debug('clear Softplus and Tanh')
                    logger.debug('dict_sp: {}'.format(dict_sp))
                    logger.debug('dict_tanh: {}'.format(dict_tanh))
                    dict_sp = {}
                    dict_tanh = {}

    if got_mish == True:
        op_set = model.opset_import.add()
        op_set.domain = 'com.metax-tech'
        op_set.version = 1

        #onnx.save(model, export_onnx)

    return model

def Test():
    export_onnx = './tmp.onnx'

    merge_mish('./v4_no_mish.onnx', export_onnx)
'''
    loop = True

    loop = merge_mish_old('./v4_no_mish.onnx', export_onnx)

    while loop:
        loop = merge_mish_old(export_onnx, export_onnx)
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Softplus+Tanh+Mul to Mish')
    parser.add_argument('--onnx_file', type=str, default='', help='source onnx model')
    parser.add_argument('--output_file', type=str, default='mish.onnx', help='dest onnx file')
    args = parser.parse_args()
    merge_mish(onnxfile=args.onnx_file, export_onnx=args.output_file)
