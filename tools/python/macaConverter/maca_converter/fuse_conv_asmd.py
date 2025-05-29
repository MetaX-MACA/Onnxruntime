import onnx
import sys
import values, operation
import numpy as np
import log

from onnx import numpy_helper, helper

logger = log.getLogger(__name__, log.INFO)


asmd_op_list = ['Add', 'Sub', 'Mul', 'Div']

#################################Pattern ONE
#Conv->Add/Sub/Mul/Div(Constant)
def get_conv_asmd(model):
    conv_asmd_list= []

    for node in model.graph.node:
        if node.op_type == 'Conv':
            #next_node, ok = operation.get_next_node_by_output(model, node.output[0])
            #if ok == 0 and next_node.op_type in asmd_op_list:
            all_next_node, ok = operation.get_all_next_node_by_output(model, node.output[0])
            if ok == 0 and len(all_next_node) == 1:
                next_node = all_next_node[0]
                if next_node.op_type in asmd_op_list:    
                    input_second = True
                    if next_node.input[1] == node.output[0]:
                        input_second = False

                    if input_second == True:
                        asmd_input_constant = next((initializer for initializer in model.graph.initializer if initializer.name == next_node.input[1]), None)
                    elif input_second == False:
                        asmd_input_constant = next((initializer for initializer in model.graph.initializer if initializer.name == next_node.input[0]), None)

                    if asmd_input_constant != None:
                        asmd_constant = numpy_helper.to_array(asmd_input_constant)
                        logger.debug('----got match conv+asmd node: {}, asmd_constant.shape:{}'.format(node.name, asmd_constant.shape))
                        if len(asmd_constant.shape) == 0 or len(asmd_constant.shape) == 1:
                            node_dict = {}
                            node_dict['Conv'] = node
                            node_dict['asmd'] = next_node
                            node_dict['asmd_constant'] = asmd_constant
                            node_dict['asmd_input_second'] = input_second
                            node_dict['operator'] = next_node.op_type

                            conv_asmd_list.append(node_dict)

    return conv_asmd_list      

def handle_conv_asmd(model, conv_asmd_dict):
    conv_node = conv_asmd_dict['Conv'] 
    asmd_node = conv_asmd_dict['asmd']
    input_second = conv_asmd_dict['asmd_input_second']
    asmd_constant = conv_asmd_dict['asmd_constant']
    asmd_op = conv_asmd_dict['operator']

    conv_weights_name = conv_node.input[1]
    conv_biases_name = conv_node.input[2] if len(conv_node.input) > 2 else None
    conv_weights_input = next(initializer for initializer in model.graph.initializer if initializer.name == conv_weights_name)
    conv_biases_input = next((initializer for initializer in model.graph.initializer if initializer.name == conv_biases_name), None) if conv_biases_name is not None else None
    conv_weights = numpy_helper.to_array(conv_weights_input)
    conv_biases = numpy_helper.to_array(conv_biases_input) if conv_biases_input is not None else None

    #data_type_str = onnx.TensorProto.DataType.Name(conv_weights_input.data_type)
    #print('data_type_str:', data_type_str)

    if asmd_op == 'Mul' or asmd_op == 'Div':
        fused_weight = conv_weights * asmd_constant

        if asmd_op == 'Div':
            fused_weight = conv_weights / asmd_constant

        #print('fused_weight.shape:', fused_weight.shape)

        fused_type = onnx.TensorProto.FLOAT
        if fused_weight.dtype == np.float16:
            fused_type = onnx.TensorProto.FLOAT16

        conv_weights_name_new = conv_weights_name + '_' + conv_node.output[0] + '_new'
        conv_weights_new = helper.make_tensor(conv_weights_name_new, fused_type, fused_weight.shape, fused_weight.flatten())
        model.graph.initializer.extend([conv_weights_new])

        conv_node.input[1] = conv_weights_name_new

        print('conv_weights_name_new:', conv_weights_name_new)

        #############
        if conv_biases is not None:
            fused_bias = conv_biases * asmd_constant

            if asmd_op == 'Div':
                fused_bias = conv_biases / asmd_constant

            #print('fused_bias.shape:', fused_bias.shape)

            conv_bias_name_new = conv_biases_name + '_' + conv_node.output[0] + '_new_'
            conv_bias_new = helper.make_tensor(conv_bias_name_new, fused_type, fused_bias.shape, fused_bias.flatten())
            model.graph.initializer.extend([conv_bias_new])

            conv_node.input[2] = conv_bias_name_new

        operation.remove_initializer_if_necessary_by_name(model, conv_weights_name, conv_node)
        operation.remove_initializer_if_necessary_by_name(model, conv_biases_name, conv_node)     
    elif asmd_op == 'Add' or asmd_op == 'Sub':
        if conv_biases is not None:
            fused_bias = conv_biases + asmd_constant

            fused_type = onnx.TensorProto.FLOAT
            if fused_bias.dtype == np.float16:
                fused_type = onnx.TensorProto.FLOAT16

            if asmd_op == 'Sub':
                fused_bias = conv_biases - asmd_constant

            #print('fused_bias.shape:', fused_bias.shape)

            conv_bias_name_new = conv_biases_name + '_' + conv_node.output[0] + '_new_'
            conv_bias_new = helper.make_tensor(conv_bias_name_new, fused_type, fused_bias.shape, fused_bias.flatten())
            model.graph.initializer.extend([conv_bias_new])

            conv_node.input[2] = conv_bias_name_new

            operation.remove_initializer_if_necessary_by_name(model, conv_biases_name, conv_node)
        else:
            fused_bias = asmd_constant

            if asmd_op == 'Sub':
                fused_bias = -1.0 * asmd_constant

            #print('fused_bias.shape:', fused_bias.shape)

            conv_bias_name_new = conv_node.input[0] + '_' + conv_node.output[0] + '_bias_'
            conv_bias_new = helper.make_tensor(conv_bias_name_new, fused_type, fused_bias.shape, fused_bias.flatten())
            model.graph.initializer.extend([conv_bias_new])

            conv_node.input.append(conv_bias_name_new)

    conv_node.name = asmd_node.name + '+' + conv_node.name

    all_next_node, _ = operation.get_all_next_node_by_output(model, asmd_node.output[0])
    for next_node in all_next_node:
        for i, input_ in enumerate(next_node.input):
            if asmd_node.output[0] == input_:
                next_node.input[i] = conv_node.output[0]

    operation.remove_onnx_node(model, asmd_node)

def conv_asmd_to_conv(model):
    conv_asmd_list = get_conv_asmd(model)

    for conv_asmd_dict in conv_asmd_list:
        handle_conv_asmd(model, conv_asmd_dict)  

    return model

'''
if __name__ == "__main__":
    #model = onnx.load('/home/zqiu/models/det_mv3_db.onnx')
    model = onnx.load('/home/zqiu/models/facenet_sim.onnx')
    conv_asmd_to_conv(model)
    onnx.save(model, './fs.onnx')
'''
    