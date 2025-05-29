import onnx
import values
import sys
import numpy as np
import log
import operation

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

def merge_resize(model):
    dict_reshape = {}
    dict_expand = {}
    dict_reshape2 = {}
    unused_init_list = []

    search = True

    while search == True:
        search = False
        ready = False

        for node_id, node in enumerate(model.graph.node):
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #        ", op:", node.op_type, ', len(input):', len(node.input))

            if node.op_type == 'Reshape':
                if ready == True:
                    if dict_reshape and dict_expand and node.input[0] == dict_expand['output'][0]:
                        dict_reshape2['input'] = node.input
                        dict_reshape2['output'] = node.output
                        dict_reshape2['id'] = node_id
                        dict_reshape2['name'] = node.name

                        shape = values.get_init_value(model, node.input[1])
                        logger.debug('got Second Reshape shape: {}'.format(shape))
                        shape_list = shape
                        shape_dims = []

                        for init in model.graph.initializer:
                            if init.name == node.input[1]:
                                shape_dims = init.dims
                                if init not in unused_init_list:
                                    unused_init_list.append(init)

                        if isinstance(shape, np.ndarray):
                            shape_list = shape.tolist()

                        ###################################
                        old_node = model.graph.node[dict_reshape['id']] 
                        old_node_reshape2 = model.graph.node[dict_reshape2['id']]
                        old_node_expand = model.graph.node[dict_expand['id']] 

                        model.graph.node.remove(old_node)

                        empty0 = onnx.helper.make_tensor('empty0', onnx.TensorProto.FLOAT, [0], [])
                        empty1 = onnx.helper.make_tensor('empty1', onnx.TensorProto.FLOAT, [0], [])
                        const_sizes = onnx.helper.make_tensor(name='const_sizes',
                                                                data_type=onnx.TensorProto.INT64,
                                                                dims=shape_dims,
                                                                vals=shape_list)

                        model.graph.initializer.append(empty0)
                        model.graph.initializer.append(empty1)
                        model.graph.initializer.append(const_sizes)    

                        resize_node = onnx.helper.make_node(
                                                name = dict_reshape['name'] + '+' + dict_expand['name'] + '+' + dict_reshape2['name'],
                                                op_type='Resize',
                                                inputs=[dict_reshape['input'][0], 'empty0', 'empty1', 'const_sizes'],
                                                outputs=dict_reshape2['output']
                                                )

                        model.graph.node.insert(dict_reshape['id'], resize_node)

                        model.graph.node.remove(old_node_expand)
                        model.graph.node.remove(old_node_reshape2)

                        dict_reshape = {}
                        dict_expand = {}
                        dict_reshape2 = {} 
                        ###############################

                        search = True
                        break       
                    else:
                        logger.debug('clear dict_reshape and dict_expand, dict_reshapes')
                        logger.debug('dict_reshape: {}'.format(dict_reshape))
                        logger.debug('dict_expand: {}'.format(dict_expand))
                        logger.debug('dict_reshape2: {}'.format(dict_reshape2))
                        dict_reshape = {}
                        dict_expand = {}
                        dict_reshape2 = {} 
                        unused_init_list = []
                else:
                    dict_reshape['input'] = node.input
                    dict_reshape['output'] = node.output
                    dict_reshape['id'] = node_id
                    dict_reshape['name'] = node.name

                    for init in model.graph.initializer:
                        if init.name == node.input[1]:
                            if init not in unused_init_list:
                                unused_init_list.append(init)

                    logger.debug('got match Reshape node: {}'.format(node.name))
                
            if node.op_type == 'Expand':
                if dict_reshape and node.input[0] == dict_reshape['output'][0]:
                    dict_expand['input'] = node.input
                    dict_expand['output'] = node.output
                    dict_expand['id'] = node_id
                    dict_expand['name'] = node.name

                    for init in model.graph.initializer:
                        if init.name == node.input[1]:
                            if init not in unused_init_list:
                                unused_init_list.append(init)

                    ready = True

                    logger.debug('got match Expand node: {}'.format(node.name))
                else:
                    logger.debug('clear dict_reshape: {}'.format(dict_reshape))
                    dict_reshape = {}
                    unused_init_list = []           

    remove_unused_initializer(model, unused_init_list)

    operation.adjust_resize2dynamic(model)

    return model

'''
model = onnx.load('./expand.onnx') 
merge_resize(model)
onnx.save(model, './rs.onnx')
'''