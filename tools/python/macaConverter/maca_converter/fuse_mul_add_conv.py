import onnx
import sys
import values, operation
import numpy as np
import log

from onnx import numpy_helper, helper

logger = log.getLogger(__name__, log.DEBUG)


asmd_op_list = ['Add', 'Sub', 'Mul', 'Div']

#################################Pattern ONE
#Mul-->Add-->Conv
def get_mul_add_conv(model):
    add_mul_conv_list = []

    for node in model.graph.node:
        if node.op_type == 'Mul':
            mul_input_second = True
            mul_input_constant = next((initializer for initializer in model.graph.initializer if initializer.name == node.input[1]), None)
            if mul_input_constant is None:
                mul_input_constant = next((initializer for initializer in model.graph.initializer if initializer.name == node.input[0]), None)
                if mul_input_constant is None:
                    continue
                mul_input_second = False    

            if mul_input_constant != None:
                mul_constant = numpy_helper.to_array(mul_input_constant)
                if len(mul_constant.shape) == 0 or len(mul_constant.shape) == 1:
                    #logger.debug('----maybe got match mul node: {}, mul_constant.shape:{}'.format(node.name, mul_constant.shape))
                    add_node, ok = operation.get_next_node_by_output(model, node.output[0])
                    if ok == 0 and add_node.op_type == 'Add':
                        add_input_second = True
                        if add_node.input[1] == node.output[0]:
                            add_input_second = False

                        if add_input_second == True:
                            add_input_constant = next((initializer for initializer in model.graph.initializer if initializer.name == add_node.input[1]), None)
                        elif add_input_second == False:
                            add_input_constant = next((initializer for initializer in model.graph.initializer if initializer.name == add_node.input[0]), None)

                        if add_input_constant != None:
                            add_constant = numpy_helper.to_array(add_input_constant)
                            #logger.debug('----maybe got match add node: {}, add_constant.shape:{}'.format(node.name, add_constant.shape))
                            if len(add_constant.shape) == 0 or len(add_constant.shape) == 1:
                                conv_node, ok = operation.get_next_node_by_output(model, add_node.output[0])
                                if ok == 0 and conv_node.op_type == 'Conv':
                                    logger.debug('----got match conv node: {}'.format(conv_node.name))

                                    node_dict = {}
                                    node_dict['mul'] = node
                                    node_dict['add'] = add_node
                                    node_dict['conv'] = conv_node
                                    node_dict['mul_constant'] = mul_constant
                                    node_dict['add_constant'] = add_constant
                                    node_dict['mul_input_second'] = mul_input_second
                                    

                                    add_mul_conv_list.append(node_dict)

    return add_mul_conv_list      

#'''
def handle_amc(model, amc_dict):
    mul_node = amc_dict['mul']
    add_node = amc_dict['add']
    conv_node = amc_dict['conv'] 
    input_second = amc_dict['mul_input_second']
    add_constant = amc_dict['add_constant']
    mul_constant = amc_dict['mul_constant']
    
    conv_weights_name = conv_node.input[1]
    conv_biases_name = conv_node.input[2] if len(conv_node.input) > 2 else None
    conv_weights_input = next(initializer for initializer in model.graph.initializer if initializer.name == conv_weights_name)
    conv_biases_input = next((initializer for initializer in model.graph.initializer if initializer.name == conv_biases_name), None) if conv_biases_name is not None else None
    conv_weights = numpy_helper.to_array(conv_weights_input)
    conv_biases = numpy_helper.to_array(conv_biases_input) if conv_biases_input is not None else None

    if conv_weights.shape[-1] != 1 or conv_weights.shape[-2] != 1:
        return

    fused_type = onnx.TensorProto.FLOAT
    if conv_weights.dtype == np.float16:
        fused_type = onnx.TensorProto.FLOAT16

    fused_weight = conv_weights * mul_constant

    fused_bias = conv_weights * add_constant

    fused_bias = np.sum(fused_bias, axis=(1, 2, 3))
    #fused_bias = fused_bias.reshape(fused_bias.shape[0], -1)
    #fused_bias = np.mean(fused_bias, axis=(1))

    print('fused_bias.shape:', fused_bias.shape, conv_biases_name)

    if conv_biases is not None:
        fused_bias = fused_bias + conv_biases

    conv_weights_name_new = conv_weights_name + '_new_'
    conv_weights_new = helper.make_tensor(conv_weights_name_new, fused_type, fused_weight.shape, fused_weight.flatten())
    model.graph.initializer.extend([conv_weights_new])

    conv_node.input[1] = conv_weights_name_new

    conv_bias_name_new = conv_node.input[0] + '_bias_new_'
    conv_bias_new = helper.make_tensor(conv_bias_name_new, fused_type, fused_bias.shape, fused_bias.flatten())
    model.graph.initializer.extend([conv_bias_new])

    operation.remove_initializer_if_necessary_by_name(model, conv_weights_name, conv_node)

    if conv_biases is not None:
        operation.remove_initializer_if_necessary_by_name(model, conv_biases_name, conv_node)

    if conv_biases is not None:
        conv_node.input[2] = conv_bias_name_new
    else:
        conv_node.input.append(conv_bias_name_new)   

    conv_node.input[0] = mul_node.input[0]
    if input_second == False:
        conv_node.input[0] = mul_node.input[1]

    conv_node.name = add_node.name + '+' + mul_node.name + '+' + conv_node.name

    operation.remove_onnx_node(model, add_node)
    operation.remove_onnx_node(model, mul_node)
#'''

def handle_amc_test(model, amc_dict):
    mul_node = amc_dict['mul']
    add_node = amc_dict['add']
    conv_node = amc_dict['conv'] 
    input_second = amc_dict['mul_input_second']
    add_constant = amc_dict['add_constant']
    mul_constant = amc_dict['mul_constant']
    
    conv_weights_name = conv_node.input[1]
    conv_weights_input = next(initializer for initializer in model.graph.initializer if initializer.name == conv_weights_name)
    conv_weights = numpy_helper.to_array(conv_weights_input)

    if conv_weights.shape[-1] != 1 or conv_weights.shape[-2] != 1:
        return

    fused_type = onnx.TensorProto.FLOAT
    if conv_weights.dtype == np.float16:
        fused_type = onnx.TensorProto.FLOAT16

    fused_weight = conv_weights * mul_constant
    print('mul_constant:', mul_constant, mul_node.name)

    conv_weights_name_new = conv_weights_name + '_new_'
    conv_weights_new = helper.make_tensor(conv_weights_name_new, fused_type, fused_weight.shape, fused_weight.flatten())
    model.graph.initializer.extend([conv_weights_new])

    conv_node.input[1] = conv_weights_name_new

    operation.remove_initializer_if_necessary_by_name(model, conv_weights_name, conv_node)

    add_node.input[0] = mul_node.input[0]
    if input_second == False:
        add_node.input[0] = mul_node.input[1]

    operation.remove_onnx_node(model, mul_node)

def fuse_mac(model):
    add_mul_conv_list = get_mul_add_conv(model)

    for amc_dict in add_mul_conv_list:
        handle_amc(model, amc_dict)  

    return model

'''
if __name__ == "__main__":
    #model = onnx.load('/home/zqiu/models/det_mv3_db.onnx')
    model = onnx.load('/home/zqiu/models/rec_v4_sim.onnx')
    fuse_mac(model)
    onnx.save(model, './rec.onnx')
'''
    