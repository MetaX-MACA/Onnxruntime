import onnx
import sys
import values, operation
import numpy as np
import log

from onnx import numpy_helper, helper

logger = log.getLogger(__name__, log.INFO)

#Shape-->Gather-->Add-->Div-->Mul-->None
def get_sgadm(model):
    node_list = []

    for node in model.graph.node:
        if node.op_type == 'Shape':
            gather_node, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == 0 and gather_node.op_type == 'Gather':
                #print('got gather node:', gather_node.name)
                add_node, ok = operation.get_next_node_by_output(model, gather_node.output[0])
                if ok == 0 and add_node.op_type == 'Add':
                    #print('got add node:', add_node.name)
                    div_node, ok = operation.get_next_node_by_output(model, add_node.output[0])
                    if ok == 0 and div_node.op_type == 'Div':
                        #print('got div node:', div_node.name)
                        mul_node, ok = operation.get_next_node_by_output(model, div_node.output[0])
                        if ok == 0 and mul_node.op_type == 'Mul':
                            #print('got mul node:', mul_node.name)
                            _, ok = operation.get_next_node_by_output(model, mul_node.output[0])
                            if ok == -1:
                                #print('got shape node:', node.name)
                                d = {}
                                d['Shape'] = node
                                d['Gather'] = gather_node
                                d['Add'] = add_node
                                d['Div'] = div_node
                                d['Mul'] = mul_node

                                node_list.append(d)

    return node_list

def handle_sgadm(model):
    node_list = get_sgadm(model)

    for d in node_list:
        shape_node = d['Shape']
        gather_node = d['Gather']
        add_node = d['Add']
        div_node = d['Div']
        mul_node = d['Mul']

        operation.remove_onnx_node(model, shape_node)
        operation.remove_onnx_node(model, gather_node)
        operation.remove_onnx_node(model, add_node)
        operation.remove_onnx_node(model, div_node)
        operation.remove_onnx_node(model, mul_node)

#Slice-->Gelu---->|Mul
#    Slice------->|
def get_sgms(model):
    target_list= []

    for node in model.graph.node:
        if node.op_type == 'Gelu':
            slice_node, ok = operation.get_prev_node_by_input(model, node.input[0])
            if ok == 0 and slice_node.op_type == 'Slice':
                logger.debug('got slice: {}'.format(slice_node.name))
                mul_node, ok = operation.get_next_node_by_output(model, node.output[0])
                if ok == 0 and mul_node.op_type == 'Mul':
                    logger.debug('----got match Mul node: {}'.format(mul_node.name))
                    mul_input_another = mul_node.input[0]
                    if mul_node.input[0] == node.output[0]:
                        mul_input_another = mul_node.input[1]

                    slice_node2, ok = operation.get_prev_node_by_input(model, mul_input_another)
                    if ok == 0 and slice_node2.op_type == 'Slice':
                        logger.debug('----got match second Slicenode: {}'.format(node.name))
                        if slice_node.input[0] == slice_node2.input[0]:
                            logger.debug('----got match Slice+Gelu+Mul node: {}'.format(node.name))

                            const_add = ''

                            slice_prev_node, ok = operation.get_prev_node_by_input(model, slice_node.input[0])
                            if ok == 0 and slice_prev_node.op_type == 'Add':
                                const_addA = next((initializer for initializer in model.graph.initializer if initializer.name == slice_prev_node.input[0]), None)
                                if const_addA == None:
                                    const_addB = next((initializer for initializer in model.graph.initializer if initializer.name == slice_prev_node.input[1]), None)
                                    if const_addB != None:
                                        const_add = slice_prev_node.input[1]
                                else:
                                    const_add = slice_prev_node.input[0]

                            node_dict = {}
                            node_dict['gelu'] = node
                            node_dict['slice1'] = slice_node
                            node_dict['slice2'] = slice_node2
                            node_dict['mul'] = mul_node
                            node_dict['add'] = slice_prev_node
                            node_dict['add_const_input'] = const_add

                            target_list.append(node_dict)

                            #for k, v in node_dict.items():
                            #    print('----name: {}, node: {}'.format(k, v.name))

    return target_list      

def handle_sgms(model, target_list):
    gelu_node = target_list['gelu'] 
    slice_node1 = target_list['slice1']
    slice_node2 = target_list['slice2']
    mul_node = target_list['mul']

    add_node = None

    const_add = target_list['add_const_input']

    if const_add != '':
         add_node = target_list['add']

    mul_node.op_type = 'SplitActivation'
    mul_node.name = gelu_node.name + '+' + slice_node1.name + '+' + slice_node2.name + '+' + mul_node.name

    if const_add != '':
        sa_input = add_node.input[1]
        if sa_input == const_add:
            sa_input = add_node.input[0]

        mul_node.input[0] = sa_input
        mul_node.input[1] = const_add
    else:
        mul_node.input[0] = slice_node1.input[0]
        del mul_node.input[1:]
 
    attr = onnx.helper.make_attribute('activation', 'Gelu')
    mul_node.attribute.append(attr)

    mul_node.domain='com.metax-tech'

    op_set = model.opset_import.add()
    op_set.domain = 'com.metax-tech'
    op_set.version = 1

    operation.remove_onnx_node(model, gelu_node)
    operation.remove_onnx_node(model, slice_node1)
    operation.remove_onnx_node(model, slice_node2)

    if add_node != None:
        operation.remove_onnx_node(model, add_node)

    handle_sgadm(model)    

def sa_fuse(model):
    target_list = get_sgms(model)

    for item in target_list:
        handle_sgms(model, item)  

    return model

'''
if __name__ == "__main__":
    #model = onnx.load('/home/zqiu/models/det_mv3_db.onnx')
    model = onnx.load('/home/zqiu/models/sub.onnx')
    sa_fuse(model)
    onnx.save(model, './fs.onnx')
'''
    