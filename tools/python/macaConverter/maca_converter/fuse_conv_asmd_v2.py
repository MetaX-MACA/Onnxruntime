import sys
import onnx


import onnx
import sys
import values, operation
import numpy as np
import log

from onnx import numpy_helper, helper

logger = log.getLogger(__name__, log.INFO)


asmd_op_list = ['Add', 'Sub', 'Mul', 'Div']


# Conv+ (Add/Sub/Mul/Div(Constant))*n
def get_conv_asmd(model):
    conv_asmd_list = []
    used_asmd_list = []

    for node in model.graph.node:
        if node.op_type == 'Conv':
            node_name_list = []
            all_next_node, ok = operation.get_all_next_node_by_output(model, node.output[0])
            while ok == 0 and len(all_next_node) == 1 and all_next_node[0].op_type in asmd_op_list and all_next_node[0].name not in used_asmd_list:
                asmd_constant = next((initializer for initializer in model.graph.initializer if initializer.name == all_next_node[0].input[1]), None)
                if asmd_constant is None:
                    break
                used_asmd_list.append(all_next_node[0].name)
                node_name_list.append(all_next_node[0].name)
                all_next_node, ok = operation.get_all_next_node_by_output(model, all_next_node[0].output[0])
            if len(node_name_list) > 0:
                node_name_list.insert(0, node.name)
                conv_asmd_list.append(node_name_list)

    return conv_asmd_list


def handle_conv_asmd(model, conv_asmd_list):
    asmd_nodes = []
    for conv_asmd in conv_asmd_list:
        cur_node = operation.find_node_by_name(model, conv_asmd)
        if cur_node.op_type == 'Conv':
            conv_node = cur_node
        else:
            asmd_nodes.append(cur_node)
    # 目前只验证了 conv+mul+add conv+mul+sub
    conv_weights_name = conv_node.input[1]
    conv_biases_name = conv_node.input[2] if len(conv_node.input) > 2 else None
    conv_weights_input = next(initializer for initializer in model.graph.initializer if initializer.name == conv_weights_name)
    conv_biases_input = next((initializer for initializer in model.graph.initializer if initializer.name == conv_biases_name), None) if conv_biases_name is not None else None
    conv_weights = numpy_helper.to_array(conv_weights_input)
    conv_biases = numpy_helper.to_array(conv_biases_input) if conv_biases_input is not None else None

    for asmd_node in asmd_nodes:
        asmd_op = asmd_node.op_type
        asmd_constant = next((initializer for initializer in model.graph.initializer if initializer.name == asmd_node.input[1]), None)
        if asmd_constant is None:
            break
        asmd_constant = numpy_helper.to_array(asmd_constant).flatten()
        if asmd_op == 'Mul' or asmd_op == 'Div':
            if asmd_constant.shape != conv_weights.shape:
                asmd_constant = asmd_constant.reshape(asmd_constant.shape[0], 1, 1, 1)
            fused_weight = conv_weights * asmd_constant

            if asmd_op == 'Div':
                fused_weight = conv_weights / asmd_constant

            fused_type = onnx.TensorProto.FLOAT
            if fused_weight.dtype == np.float16:
                fused_type = onnx.TensorProto.FLOAT16

            conv_weights_name_new = conv_weights_name + '_' + asmd_node.name + '_fuse_'
            conv_weights_new = helper.make_tensor(conv_weights_name_new, fused_type, fused_weight.shape, fused_weight.flatten())
            model.graph.initializer.extend([conv_weights_new])

            conv_node.input[1] = conv_weights_name_new

            if conv_biases is not None:
                fused_bias = conv_biases * asmd_constant

                if asmd_op == 'Div':
                    fused_bias = conv_biases / asmd_constant

                fused_bias = fused_bias.flatten()
                conv_bias_name_new = conv_biases_name + '_' + asmd_node.name + '_fuse_'
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
                conv_bias_name_new = conv_biases_name + '_' + asmd_node.name + '_fuse_'
                conv_bias_new = helper.make_tensor(conv_bias_name_new, fused_type, fused_bias.shape, fused_bias.flatten())
                model.graph.initializer.extend([conv_bias_new])

                conv_node.input[2] = conv_bias_name_new

                operation.remove_initializer_if_necessary_by_name(model, conv_biases_name, conv_node)
            else:
                fused_bias = asmd_constant

                if asmd_op == 'Sub':
                    fused_bias = -1.0 * asmd_constant
                conv_bias_name_new = conv_node.input[0] + '_' + asmd_node.name + '_fuse_'
                conv_bias_new = helper.make_tensor(conv_bias_name_new, fused_type, fused_bias.shape, fused_bias.flatten())
                model.graph.initializer.extend([conv_bias_new])

                conv_node.input.append(conv_bias_name_new)

        conv_node.name = conv_node.name + '+' + asmd_node.name

        all_next_node, _ = operation.get_all_next_node_by_output(model, asmd_node.output[0])
        for next_node in all_next_node:
            for i, input_ in enumerate(next_node.input):
                if asmd_node.output[0] == input_:
                    next_node.input[i] = conv_node.output[0]

        operation.remove_onnx_node(model, asmd_node)


def conv_asmd_to_conv_v2(model):
    conv_asmd_list = get_conv_asmd(model)
    for conv_asmd in conv_asmd_list:
        handle_conv_asmd(model, conv_asmd)

    return model


if __name__ == "__main__":
    model_file = sys.argv[1]
    save_model = sys.argv[2]
    model = onnx.load(model_file)
    conv_asmd_to_conv_v2(model)
    onnx.save(model, save_model)
