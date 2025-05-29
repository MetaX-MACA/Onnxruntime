import onnx
import sys
import values, operation
import numpy as np
import log

from onnx import numpy_helper, helper

from onnx import numpy_helper, helper

logger = log.getLogger(__name__, log.INFO)


#################################Pattern ONE
#ConvTranspose->Add->BN
def get_ct_add_bn(model):
    ct_add_bn_list= []
    ct_add_list= []
    ct_bn_list = []

    for node in model.graph.node:
        if node.op_type == 'ConvTranspose':
            add_or_bn_node, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == 0 and add_or_bn_node.op_type == 'Add':
                logger.debug('got add_or_bn_node: {}'.format(add_or_bn_node.name))
                v, shape = values.get_init_value_and_shape(model, add_or_bn_node.input[1])
                #print('add input1 shape:', shape, v)
                if len(shape) > 0 or len(v) > 0:
                    bn_node, ok = operation.get_next_node_by_output(model, add_or_bn_node.output[0])
                    if ok == 0:
                        if bn_node.op_type == 'BatchNormalization':
                            logger.debug('----got match ConvTranspose+Add+BN node: {}'.format(node.name))

                            node_dict = {}
                            node_dict['ConvTranspose'] = node
                            node_dict['Add'] = add_or_bn_node
                            node_dict['BN'] = bn_node
                            ct_add_bn_list.append(node_dict)

                            #for k, v in node_dict.items():
                            #    print('name: {}, node: {}'.format(k, v.name))
                        else:
                            logger.debug('----got match ConvTranspose+Add node: {}'.format(node.name))

                            node_dict = {}
                            node_dict['ConvTranspose'] = node
                            node_dict['Add'] = add_or_bn_node
                            ct_add_list.append(node_dict)
            elif ok == 0 and add_or_bn_node.op_type == 'BatchNormalization':
                node_dict = {}
                node_dict['ConvTranspose'] = node
                node_dict['BN'] = add_or_bn_node
                ct_bn_list.append(node_dict)

    return ct_add_bn_list, ct_add_list, ct_bn_list

def handle_ct_add(model, ca_dict):
    ct_node = ca_dict['ConvTranspose']
    add_node = ca_dict['Add']

    ###############
    conv_weights_name = ct_node.input[1]
    conv_biases_name = ct_node.input[2] if len(ct_node.input) > 2 else None
    conv_weights_input = next(initializer for initializer in model.graph.initializer if initializer.name == conv_weights_name)
    conv_biases_input = next((initializer for initializer in model.graph.initializer if initializer.name == conv_biases_name), None) if conv_biases_name is not None else None
    conv_weights = numpy_helper.to_array(conv_weights_input)
    conv_biases = numpy_helper.to_array(conv_biases_input) if conv_biases_input is not None else None
    
    add_value_name = add_node.input[1]
    add_value_input = next((initializer for initializer in model.graph.initializer if initializer.name == add_value_name), None)
    if add_value_input is None:
        add_value_name = add_node.input[0]
        add_value_input = next((initializer for initializer in model.graph.initializer if initializer.name == add_value_name), None)

    if add_value_input is None:
        logger.warning('add_value is not conatant')
        return False

    add_value = numpy_helper.to_array(add_value_input)

    #print('XXXXX conv_weights.shape:{}, add_values.shape:{}'.format(conv_weights.shape, add_value.shape))
    tmp_value = add_value.squeeze()
    if len(tmp_value.shape) == 0:
        add_value = add_value.reshape(1)
    else:
        add_value = add_value.squeeze()   

    #print('YYYYY conv_weights.shape[1]:{}, add_values.new_shape:{}, len:{}'.format(conv_weights.shape[1], add_value.shape, len(add_value.shape)))

    if add_value.shape[0] != conv_weights.shape[1] or len(add_value.shape) != 1:
        logger.warning('add_value.shape[0]({}) != conv_weights.shape[1]({}), or len(add_value.shape)({}) != 1'.format(add_value.shape[0], conv_weights.shape[1], len(add_value.shape)))
        return False
    
    ###############

    add_next_node, _ = operation.get_next_node_by_output(model, add_node.output[0])
    for i, input_ in enumerate(add_next_node.input):
        if add_node.output[0] == input_:
            add_next_node.input[i] = ct_node.output[0]

    fused_type = onnx.TensorProto.FLOAT
    if add_value.dtype == np.float16:
        fused_type = onnx.TensorProto.FLOAT16

    if len(ct_node.input) == 2:
        #ct_node.input.append(add_node.input[1])
        conv_bias_name_new = ct_node.input[0] + '_bias_'
        conv_bias_new = helper.make_tensor(conv_bias_name_new, fused_type, add_value.shape, add_value.flatten())
        model.graph.initializer.extend([conv_bias_new])
        ct_node.input.append(conv_bias_name_new)
    elif len(ct_node.input) == 3:
        conv_biases = conv_biases + add_value
        conv_biases = conv_biases.flatten()
        conv_biases = conv_biases.tolist()

        for init in model.graph.initializer:
            if ct_node.input[2] == init.name:
                values.set_tensor_value(init, conv_biases)
                break

    ct_node.name = add_node.name + '+' + ct_node.name

    operation.remove_onnx_node(model, add_node)

    return True

def fuse_bn_into_conv(model, conv_node, bn_node):
    conv_weights_name = conv_node.input[1]
    conv_biases_name = conv_node.input[2] if len(conv_node.input) > 2 else None
    conv_weights_input = next(initializer for initializer in model.graph.initializer if initializer.name == conv_weights_name)
    conv_biases_input = next((initializer for initializer in model.graph.initializer if initializer.name == conv_biases_name), None) if conv_biases_name is not None else None
    conv_weights = numpy_helper.to_array(conv_weights_input)
    conv_biases = numpy_helper.to_array(conv_biases_input) if conv_biases_input is not None else None

    #data_type_str = onnx.TensorProto.DataType.Name(conv_weights_input.data_type)
    #print('data_type_str:', data_type_str)

    bn_epsilon = 0.00001
    for attr in bn_node.attribute:
        if attr.name == 'epsilon':
            bn_epsilon = attr.f
            break

    bn_scale_name = bn_node.input[1]
    bn_bias_name = bn_node.input[2]
    bn_mean_name = bn_node.input[3]
    bn_var_name = bn_node.input[4]
    bn_scale_input = next(initializer for initializer in model.graph.initializer if initializer.name == bn_scale_name)
    bn_bias_input = next(initializer for initializer in model.graph.initializer if initializer.name == bn_bias_name)
    bn_mean_input = next(initializer for initializer in model.graph.initializer if initializer.name == bn_mean_name)
    bn_var_input = next(initializer for initializer in model.graph.initializer if initializer.name == bn_var_name)
    bn_scale = numpy_helper.to_array(bn_scale_input)
    bn_bias = numpy_helper.to_array(bn_bias_input)
    bn_mean = numpy_helper.to_array(bn_mean_input)
    bn_var = numpy_helper.to_array(bn_var_input)

    fused_biases = bn_bias + (conv_biases - bn_mean) * bn_scale / np.sqrt(bn_var + bn_epsilon) if conv_biases is not None else bn_bias - bn_mean * bn_scale / np.sqrt(bn_var + bn_epsilon)
 
    bn_scale = bn_scale.reshape(1, bn_scale.shape[0], 1, 1)
    bn_var = bn_var.reshape(1, bn_var.shape[0], 1, 1)
    bn_mean = bn_mean.reshape(1, bn_mean.shape[0], 1, 1)
    bn_bias = bn_bias.reshape(1, bn_bias.shape[0], 1, 1)
 

    #print('+++ bn_scale:', bn_scale.shape)
    fused_weights = conv_weights * bn_scale / np.sqrt(bn_var + bn_epsilon)

    #print('fused_weights.shape:', fused_weights.shape, fused_weights.dtype)

    #print('fused_biases.shape:', fused_biases.shape)

    fused_type = onnx.TensorProto.FLOAT
    if fused_weights.dtype == np.float16:
        fused_type = onnx.TensorProto.FLOAT16

    new_conv_weights_input = helper.make_tensor(conv_weights_input.name, fused_type, conv_weights.shape, fused_weights.flatten())
    model.graph.initializer.extend([new_conv_weights_input])
    if conv_biases is not None:
        new_conv_biases_input = helper.make_tensor(conv_biases_input.name, fused_type, conv_biases.shape, fused_biases.flatten())
        model.graph.initializer.extend([new_conv_biases_input])

    model.graph.initializer.remove(conv_weights_input)
    if conv_biases_input is not None:
        model.graph.initializer.remove(conv_biases_input)

    conv_node.input[1] = new_conv_weights_input.name
    if conv_biases_input is not None:
        conv_node.input[2] = new_conv_biases_input.name

    bn_next_node, _ = operation.get_next_node_by_output(model, bn_node.output[0])
    for i, input_ in enumerate(bn_next_node.input):
        if bn_node.output[0] == input_:
            bn_next_node.input[i] = conv_node.output[0]

    operation.remove_onnx_node(model, bn_node)

def handle_ct_add_bn(model, cab_dict):
    ret = handle_ct_add(model, cab_dict)

    if ret == True:
        ct_node = cab_dict['ConvTranspose']
        bn_node = cab_dict['BN']

        fuse_bn_into_conv(model, ct_node, bn_node)

def ct_add_bn_fuse(model):
    ct_add_bn_list, ct_add_list, ct_bn_list = get_ct_add_bn(model)

    for ca_dict in ct_add_list:
        handle_ct_add(model, ca_dict)   
   
    for cab_dict in ct_add_bn_list:
        handle_ct_add_bn(model, cab_dict)  

    for cb_dict in ct_bn_list:
        fuse_bn_into_conv(model, cb_dict['ConvTranspose'], cb_dict['BN'])

    return model

'''
if __name__ == "__main__":
    #model = onnx.load('/home/zqiu/models/det_mv3_db.onnx')
    model = onnx.load('/home/zqiu/models/sub.onnx')
    ct_add_bn_fuse(model)
    onnx.save(model, './det.onnx')
'''
    