import onnx
import operation

def get_all_node_by_input(model, input_):
    node_list = []
    for node in model.graph.node:
        if input_ in node.input:
            node_list.append(node)

    return node_list         

def fp32_to_uint8(model):
    index = 0
    init_name_list = []
    for init in model.graph.initializer:
        init_name_list.append(init.name)

    for input_ in model.graph.input:
        if input_.name in init_name_list:
            continue

        if input_.type.tensor_type.elem_type == 1: #float
            input_name = input_.name
            output_name = 'mx_cast_output_' + str(index)

            input_.type.tensor_type.elem_type = 2

            cast_node = onnx.helper.make_node(
                            'Cast',
                            name='MX_Cast_' + str(index),
                            inputs=[input_name],
                            outputs=[output_name],
                            to=1)

            index = index + 1

            next_node_list = get_all_node_by_input(model, input_name)
            if len(next_node_list) > 0:
                for n in next_node_list:
                    for idx, in_ in enumerate(n.input):
                        if in_ == input_name:
                            n.input[idx] = output_name
                            break

                operation.insert_onnx_node(model, cast_node, next_node_list[0])            

    return model