import onnx
import sys
import values, operation
import numpy as np
import log

from onnx import numpy_helper, helper

logger = log.getLogger(__name__, log.INFO)

#################################Pattern ONE
#Reshape->InstanceNormalization->Reshape
def get_rir(model):
    rir_list= []

    for node in model.graph.node:
        if node.op_type == 'Reshape':
            in_node, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == 0 and in_node.op_type == 'InstanceNormalization':
                logger.debug('got InstanceNormalization: {}'.format(in_node.name))
                rs_node2, ok = operation.get_next_node_by_output(model, in_node.output[0])
                if ok == 0:
                    if rs_node2.op_type == 'Reshape':

                        reshape_name1 = node.input[1]
                        reshape_input1 = next(initializer for initializer in model.graph.initializer if initializer.name == reshape_name1)
                        reshape_value1 = numpy_helper.to_array(reshape_input1)

                        has_shape_node = False

                        reshape_name2 = rs_node2.input[1]
                        reshape_input2 = next((initializer for initializer in model.graph.initializer if initializer.name == reshape_name2), None)
                        if reshape_input2 != None:
                            reshape_value2 = numpy_helper.to_array(reshape_input2)
                        else:
                            reshape2_shape = values.get_tensor_shape_by_name(model, reshape_name2)
                            #print('--input {}, shape:{}'.format(reshape_name2, reshape2_shape))
                            reshape_value2 = [1] * reshape2_shape[0]
                            has_shape_node = True

                        if len(reshape_value1) == 3 and len(reshape_value2) == 4:
                            #logger.debug('----got match Reshape+IN+Reshape node: {}'.format(node.name))

                            node_dict = {}
                            node_dict['reshape1'] = node
                            node_dict['IN'] = in_node
                            node_dict['reshape2'] = rs_node2
                            node_dict['has_shape_node'] = has_shape_node

                            rir_list.append(node_dict)

                            #for k, v in node_dict.items():
                            #    print('----name: {}, node: {}'.format(k, v.name))
                        else:
                            logger.debug('xxxx got mismatch Reshape+IN+Reshape node: {}'.format(node.name))


    return rir_list

def handle_rir(model, mrbr_dict):
    rs_node = mrbr_dict['reshape1']
    in_node = mrbr_dict['IN']
    rs_node2 = mrbr_dict['reshape2']
    has_shape_node = mrbr_dict['has_shape_node']

    in_node.op_type = 'GroupNormalization'
    in_node.input[0] = rs_node.input[0]
    in_node.output[0] = rs_node2.output[0]
    in_node.name = rs_node.name + '+' + in_node.name + '+' + rs_node2.name

    groups = -1

    input_shape = values.get_tensor_shape_by_name(model, rs_node.input[0])
    output_shape = values.get_tensor_shape_by_name(model, rs_node.output[0])

    if len(input_shape) >= 2 and len(output_shape) >= 2:
        groups = input_shape[1] // output_shape[1]

    attr = onnx.helper.make_attribute('num_groups', groups)
    in_node.attribute.append(attr)

    in_node.domain='com.metax-tech'

    if has_shape_node == True:
        shape_node, ok = operation.get_prev_node_by_input(model, rs_node2.input[1])
        if ok == 0:
            all_next_node, _ = operation.get_all_next_node_by_output(model, shape_node.output[0])
            if len(all_next_node) == 1:
                operation.remove_onnx_node(model, shape_node)

    operation.remove_onnx_node(model, rs_node)
    operation.remove_onnx_node(model, rs_node2)

    op_set = model.opset_import.add()
    op_set.domain = 'com.metax-tech'
    op_set.version = 1

##############################
def get_rir0(model):
    rir_list= []

    for node in model.graph.node:
        if node.op_type == 'Reshape':
            in_node, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == 0 and in_node.op_type == 'InstanceNormalization':
                logger.debug('got InstanceNormalization: {}'.format(in_node.name))
                in_name_scale = in_node.input[1]
                in_input_scale = next(initializer for initializer in model.graph.initializer if initializer.name == in_name_scale)
                in_scale = numpy_helper.to_array(in_input_scale)

                in_name_bias = in_node.input[2]
                in_input_bias = next(initializer for initializer in model.graph.initializer if initializer.name == in_name_bias)
                in_bias = numpy_helper.to_array(in_input_bias)

                rs_node2, ok = operation.get_next_node_by_output(model, in_node.output[0])
                if ok == 0:
                    if rs_node2.op_type == 'Reshape':

                        reshape_name1 = node.input[1]
                        #print('---reshape_name1:', reshape_name1)
                        reshape_input1 = next(initializer for initializer in model.graph.initializer if initializer.name == reshape_name1)
                        reshape_value1 = numpy_helper.to_array(reshape_input1)

                        reshape_name2 = rs_node2.input[1]

                        #print('---reshape_name2:', reshape_name2)

                        has_shape_node = False

                        reshape_input2 = next((initializer for initializer in model.graph.initializer if initializer.name == reshape_name2), None)
                        if reshape_input2 != None:
                            reshape_value2 = numpy_helper.to_array(reshape_input2)
                        else:
                            reshape2_shape = values.get_tensor_shape_by_name(model, reshape_name2)
                            #print('input {}, shape:{}'.format(reshape_name2, reshape2_shape))
                            reshape_value2 = [1] * reshape2_shape[0]
                            has_shape_node = True

                        if len(reshape_value1) == 3 and len(reshape_value2) == 4:
                            #logger.debug('----got match Reshape+IN+Reshape node: {}'.format(node.name))
                            mul_node, ok = operation.get_next_node_by_output(model, rs_node2.output[0])
                            if ok == 0 and mul_node.op_type == 'Mul':
                                mul_name_b = mul_node.input[1]
                                mul_input_b = next((initializer for initializer in model.graph.initializer if initializer.name == mul_name_b), None)
                                if mul_input_b != None:
                                    mul_value_b = numpy_helper.to_array(mul_input_b)
                                    if len(mul_value_b.shape) == 3 and mul_value_b.shape[-1] == 1 and mul_value_b.shape[-2] == 1:
                                        add_node, ok = operation.get_next_node_by_output(model,mul_node.output[0])
                                        if ok == 0 and add_node.op_type == 'Add':
                                            add_name_b = add_node.input[1]
                                            add_input_b = next((initializer for initializer in model.graph.initializer if initializer.name == add_name_b), None)
                                            if add_input_b != None:
                                                add_value_b = numpy_helper.to_array(add_input_b)
                                                if len(add_value_b.shape) == 3 and add_value_b.shape[-1] == 1 and add_value_b.shape[-2] == 1:
                                                    node_dict = {}
                                                    node_dict['reshape1'] = node
                                                    node_dict['IN'] = in_node
                                                    node_dict['reshape2'] = rs_node2
                                                    node_dict['mul'] = mul_node
                                                    node_dict['mulB'] = mul_value_b
                                                    node_dict['add'] = add_node
                                                    node_dict['addB'] = add_value_b

                                                    node_dict['scale'] = in_scale
                                                    node_dict['bias'] = in_bias
                                                    node_dict['has_shape_node'] = has_shape_node

                                                    rir_list.append(node_dict)

                                                    #print('got match node:', add_node.name)

                                                    #for k, v in node_dict.items():
                                                    #    print('----name: {}, node: {}'.format(k, v.name))
                        else:
                            logger.debug('zzzz got mismatch Reshape+IN+Reshape node: {}'.format(node.name))

    return rir_list

def handle_rir0(model, mrbr_dict):
    rs_node = mrbr_dict['reshape1']
    in_node = mrbr_dict['IN']
    rs_node2 = mrbr_dict['reshape2']
    mul_node = mrbr_dict['mul']
    add_node = mrbr_dict['add']

    scale = mrbr_dict['scale']
    bias = mrbr_dict['bias']

    mulB = mrbr_dict['mulB']
    addB = mrbr_dict['addB']

    has_shape_node = mrbr_dict['has_shape_node']

    #print('scale:', scale)
    #print('bias:', bias)
    #print('in_node:', in_node.name)
    #print('mulB:', mulB)
    #print('addB:', addB)


    # fused_type = onnx.TensorProto.FLOAT16

    if scale.dtype == np.float32:
        fused_type = onnx.TensorProto.FLOAT
    elif scale.dtype == np.float16:
        fused_type = onnx.TensorProto.FLOAT16
    else:
        logger.warning(f"GroupNormalization not support date type : {scale.dtype}")
        return


    in_node.op_type = 'GroupNormalization'
    in_node.input[0] = rs_node.input[0]
    in_node.output[0] = add_node.output[0]
    in_node.name = rs_node.name + '+' + in_node.name + '+' + rs_node2.name + '+' + mul_node.name + '+' + add_node.name

    scale_new = np.zeros(mulB.shape[0])
    bias_new = np.zeros(addB.shape[0])
    count = mulB.shape[0]//scale.shape[0]

    for i in range(scale.shape[0]):
        scale_new[i*count:(i+1)*count] = scale[i]

    for i in range(scale.shape[0]):
        bias_new[i*count:(i+1)*count] = bias[i]

    #print('scale_new:', scale_new)
    #print('bias_new:', bias_new)

    mulB = np.squeeze(mulB)
    addB = np.squeeze(addB)

    scale_new = scale_new*mulB

    #print('---scale_new:', scale_new)

    bias_new = bias_new + addB

    scale_name_new = in_node.input[1] + in_node.output[0] + '_scale_'
    sacle_new_tensor = helper.make_tensor(scale_name_new, fused_type, scale_new.shape, scale_new.flatten())
    model.graph.initializer.extend([sacle_new_tensor])

    operation.remove_initializer_if_necessary_by_name(model, in_node.input[1], in_node)

    bias_name_new = in_node.input[2] + in_node.output[0] + '_bias_'
    bias_new_tensor = helper.make_tensor(bias_name_new, fused_type, bias_new.shape, bias_new.flatten())
    model.graph.initializer.extend([bias_new_tensor])

    operation.remove_initializer_if_necessary_by_name(model, in_node.input[2], in_node)

    in_node.input[1] = scale_name_new
    in_node.input[2] = bias_name_new

    ######
    groups = -1

    input_shape = values.get_tensor_shape_by_name(model, rs_node.input[0])
    output_shape = values.get_tensor_shape_by_name(model, rs_node.output[0])

    if len(input_shape) >= 2 and len(output_shape) >= 2:
        groups = input_shape[1] // output_shape[1]

    attr = onnx.helper.make_attribute('num_groups', groups)
    in_node.attribute.append(attr)

    in_node.domain='com.metax-tech'

    if has_shape_node == True:
        shape_node, ok = operation.get_prev_node_by_input(model, rs_node2.input[1])
        if ok == 0:
            all_next_node, _ = operation.get_all_next_node_by_output(model, shape_node.output[0])
            if len(all_next_node) == 1:
                operation.remove_onnx_node(model, shape_node)

    operation.remove_onnx_node(model, rs_node)
    operation.remove_onnx_node(model, rs_node2)
    operation.remove_onnx_node(model, mul_node)
    operation.remove_onnx_node(model, add_node)

    op_set = model.opset_import.add()
    op_set.domain = 'com.metax-tech'
    op_set.version = 1

    return 0


def gn_fuse(model):
    rir_list0 = get_rir0(model)
    for rir in rir_list0:
        handle_rir0(model, rir)

    rir_list = get_rir(model)
    for rir in rir_list:
        handle_rir(model, rir)

    return model

'''
if __name__ == "__main__":
    #model = onnx.load('/home/zqiu/models/det_mv3_db.onnx')
    model = onnx.load('/home/zqiu/models/sub.onnx')
    gn_fuse(model)
    onnx.save(model, './fs.onnx')
'''
